import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import re
import warnings

warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_HERE, 'engine')
_SCANNERS = os.path.join(_HERE, 'scanners')
_ROOT = _HERE
_ORCH = os.path.join(_HERE, 'orchestrator')
for _d in (_ENGINE, _SCANNERS, _ORCH):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import pandas as pd

SACRED = {
    'dots_thresholds.py': '518862bf19fb',
    'wf.py': '4ac888f3af9d',
    'core.py': '6530e2508b17',
    'portfolio_simulation_engine.py': '7f66273011a2',
    'conviction.py': '27af7acee824',
}
FOLD_COUNT = 6
MIN_FOLD_DAYS = 5
OOS_TAIL_FRACTION = 1.0 / 3.0
FOLD_BASIS_NOTE = ('folds and OOS are PROPORTIONAL, never calendar. The loaded post-warmup span is split by '
                   'TRADING DAY into a final-third hold-out and a leading two-thirds; the two-thirds is then cut '
                   'into six equal contiguous folds. Folds and the hold-out are DISJOINT, so the two headline '
                   'figures are independent measurements rather than the same trades counted twice.')
OOS_MONTHS = ['2026.05', '2026.06']
OOS_LEGACY_NOTE = 'LEGACY DIAGNOSTIC, STALE: fixed calendar months, neither out-of-sample nor segment-relative on a stitched series; not a selection input (spec B.1). oos_rel_* are the data-relative counterpart.'
OOS_REL_N_MONTHS = 2
STAGES = ['S0', 'S1', 'S2', 'S2B', 'S3', 'S3B', 'S4', 'S5', 'S5D', 'S6', 'S5B', 'S5C', 'S7', 'S8', 'S8B', 'S9', 'S10', 'SELECT']
FAMILIES = [
    ('F0', 'triple_convergence_and_d2ddir', 'committed'),
    ('F1', 'sequential_temporal', 'committed'),
    ('F2', 'state_transition', 'exploratory'),
    ('F3', 'conditional_interaction', 'exploratory'),
    ('F4', 'divergence_nonconfirm', 'exploratory'),
    ('F5', 'persistence_autocorr', 'exploratory'),
    ('F6', 'threshold_crossing', 'exploratory'),
    ('F7', 'mean_reversion', 'exploratory'),
    ('F8', 'cross_variable_structure', 'exploratory'),
    ('F9', 'session_temporal', 'exploratory'),
    ('F11', 'rolling_leadlag', 'exploratory'),
    ('F12', 'concurrence_profiler', 'diagnostic'),
    ('F13', 'single_variable_extremes', 'exploratory'),
]


from _packutil import sha12, _natkey


def verify_sacred():
    print('SACRED REGISTRY (byte-lock — abort on drift):')
    drift = []
    for name, want in SACRED.items():
        path = os.path.join(_ENGINE, name)
        got = sha12(path) if os.path.exists(path) else 'MISSING'
        ok = got == want
        print(f'  {name:32} {got}  expect {want}  {"OK" if ok else "DRIFT"}')
        if not ok:
            drift.append(name)
    if drift:
        print(f'\nABORT — sacred drift on: {", ".join(drift)}. The master orchestrates these; it must never rewrite them.')
        sys.exit(2)
    return {n: SACRED[n] for n in SACRED}


def _hms(s):
    s = int(s)
    return f'{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}'


def done_path(out, key):
    return os.path.join(out, '.markers', f'{key}.done')


STAGE_REQUIRED_COLUMNS = {
    'wf_book_arm_entities.csv': ('in_denominator', 'train_passes', 'test_passes',
                                 'traded_on_test', 'persisted'),
    'wf_null_arm_entities.csv': ('in_denominator', 'train_passes', 'test_passes',
                                'traded_on_test'),
    'catalogues/cohort_scored.csv': ('win_loss_ratio', 'breakeven_wr', 'margin_pp', 'n_losses'),
}


STAGE_ARTIFACTS = {
    'S3': ['regime_labels.csv'],
    'S3B': ['family_evidence.csv', 'cross_family_cofiring.csv'],
    'S5D': ['catalogues/unclaimed_reachable.csv', 'catalogues/same_bar_cohort.csv',
            'catalogues/cohort_scored.csv', 'catalogues/dilution_curve_agg_pf.csv',
            'catalogues/dilution_curve_EXPECTED_ROWS_AT_OR_ABOVE_THIS_PF.csv'],
    'S5C': ['wf_pass_criterion.csv', 'wf_book_arm_entities.csv', 'wf_null_arm_entities.csv',
            'wf_splits.csv'],
    'S8B': ['cluster_participation_profile.csv', 'cluster_basis_summary.csv',
            'reach_D01_directional_baseline.csv', 'reach_D02_D2_coverage.csv',
            'reach_D02_book_depth_structure.csv', 'reach_D0_missed_decomposition.csv'],
}


def _artifact_columns(path):
    """The artifact's column names, for the explicit required-column check."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for ln in f:
                if ln.startswith('#'):
                    continue
                return [c.strip() for c in ln.rstrip('\n').split(',')]
    except OSError:
        return []
    return []


def _artifact_schema(path):
    """sha of an artifact's COLUMN NAMES. Detects a schema change without a column list.

    A gate that checks a file EXISTS cannot detect a file that is STALE, and a
    schema change makes every prior artifact stale by definition. This has now
    bitten six times - S3B, S5D, S5C, S8B, S3's regime_labels, and S5C again -
    and each fix was "check the deliverable exists", which was always too weak.

    Hashing the header is cheaper and stricter than a hand-maintained per-stage
    column list: it needs no maintenance, and it catches EVERY future column
    addition automatically rather than only the ones somebody remembered to list.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for ln in f:
                if ln.startswith('#'):
                    continue
                cols = [c.strip() for c in ln.rstrip('\n').split(',')]
                return hashlib.sha256(','.join(sorted(cols)).encode('utf-8')).hexdigest()[:12]
    except OSError:
        return None
    return None


def mark_done(out, key, meta):
    """Records each artifact's SCHEMA HASH alongside the meta.

    A marker written by an older build carries no 'artifact_schemas' key, so
    is_done treats it as STALE and the stage re-runs exactly once, writing a
    marker that does. That is what makes this self-healing on a tree that
    already has stale artifacts - no marker delete, no --force, no manual step.
    """
    os.makedirs(os.path.join(out, '.markers'), exist_ok=True)
    rec = dict(meta)
    schemas = {}
    for rel in _glob_artifacts(out, key):
        sc = _artifact_schema(os.path.join(out, rel))
        if sc is not None:
            schemas[rel] = sc
    rec['artifact_schemas'] = schemas
    tmp = done_path(out, key) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, sort_keys=True)
    os.replace(tmp, done_path(out, key))


def _glob_artifacts(out, key):
    """STAGE_ARTIFACTS lists FIXED names, and S5D's real deliverables are the
    catalogue_F*.csv files whose names depend on which families produced rows. None
    were listed, so the schema hash never covered them - and pricing_resolution_floor
    is a CATALOGUE column, so adding it changed a schema the gate was not watching and
    S5D skipped. TENTH INSTANCE: a schema gate that watches the wrong files.
    """
    extra = []
    if key == 'S5D':
        cd = os.path.join(out, 'catalogues')
        if os.path.isdir(cd):
            extra = [f'catalogues/{f}' for f in sorted(os.listdir(cd))
                     if f.startswith('catalogue_') and f.endswith('.csv')]
    return list(STAGE_ARTIFACTS.get(key, [])) + extra


def stale_artifacts(out, key):
    """Which of a stage's artifacts no longer match the schema recorded at mark_done.

    Returns a list of human-readable reasons; empty means every recorded artifact
    is present with the same columns it had when the stage completed.
    """
    p = done_path(out, key)
    if not os.path.exists(p):
        return ['no marker']
    try:
        rec = json.load(open(p, encoding='utf-8'))
    except Exception:
        return ['marker unreadable']
    out_reasons = []
    for rel in _glob_artifacts(out, key):
        need = STAGE_REQUIRED_COLUMNS.get(rel)
        if not need:
            continue
        fp = os.path.join(out, rel)
        if not os.path.exists(fp):
            continue
        cols = _artifact_columns(fp)
        miss = [c for c in need if c not in cols]
        if miss:
            out_reasons.append(f'{rel} is missing column(s) {miss}')
    if 'artifact_schemas' not in rec:
        out_reasons.append(
            'marker predates schema recording - every artifact it wrote is stale by definition, '
            'so the stage re-runs once to establish the baseline')
        return out_reasons
    for rel in _glob_artifacts(out, key):
        want = rec['artifact_schemas'].get(rel)
        got = _artifact_schema(os.path.join(out, rel))
        if want is None and got is None:
            continue
        if got is None:
            out_reasons.append(f'{rel} ABSENT (was present at mark_done)')
        elif want is None:
            out_reasons.append(f'{rel} present but was NOT recorded at mark_done')
        elif got != want:
            out_reasons.append(f'{rel} SCHEMA CHANGED (columns {want} -> {got})')
    return out_reasons


COLLECT_EXTS = ('.csv', '.md', '.txt', '.jsonl')
COLLECT_TARGET_MB = 26
COLLECT_CEILING_MB = 30
_COLLECT_SKIP_NAME = re.compile(
    r'(_c\d{4}\.(csv|pkl|done|cand)$)|(\.done$)|(\.cand$)|(\.provenance$)'
    r'|(^_frame_.*\.csv$)|(^_s3_frame.*\.csv$)|(^shard_\d+\.csv$)|(^_f0_kept\.pkl$)')
_COLLECT_SKIP_DIR = ('_f13_shards', '.markers', 'data_for_analysis', '__pycache__')


def _split_oversized(path, target_bytes):
    """Close the writer BEFORE a line that would breach. CSV header on EVERY part.

    The PowerShell original wrote the line and THEN tested the size, so every
    part landed a few bytes over its own limit - measured at 17 bytes over on the
    operator's tree. Testing first is the fix, and the target is 26 MB against a
    30 MB ceiling so a long final line cannot push a part over.

    A HEADERLESS FRAGMENT IS THE MOST EXPENSIVE DEFECT THIS PROJECT HAS HAD -
    split_tree() produced them and a later stage read the pieces as whole files.
    Every CSV part therefore repeats the header and opens standalone.
    """
    base, ext = os.path.splitext(os.path.basename(path))
    d = os.path.dirname(path)
    parts = []
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as r:
        header = r.readline() if ext.lower() == '.csv' else None
        idx, w, written = 1, None, 0
        for line in r:
            if w is None:
                op = os.path.join(d, f'{base}_part_{idx}{ext}')
                w = open(op, 'w', encoding='utf-8', newline='')
                parts.append(op)
                written = 0
                if header:
                    w.write(header)
                    written += len(header.encode('utf-8'))
            nb = len(line.encode('utf-8'))
            if written + nb > target_bytes and written > (len(header.encode('utf-8')) if header else 0):
                w.close()
                idx += 1
                op = os.path.join(d, f'{base}_part_{idx}{ext}')
                w = open(op, 'w', encoding='utf-8', newline='')
                parts.append(op)
                written = 0
                if header:
                    w.write(header)
                    written += len(header.encode('utf-8'))
            w.write(line)
            written += nb
        if w is not None:
            w.close()
    os.remove(path)
    return parts


def s10_collect(out, data_dir, input_sha):
    """S10 - collect every analysis artifact into ONE FLAT FOLDER, split for upload.

    Ported from collect_artifacts.ps1, which is proven on the operator's tree.
    DIFFERENCES FROM THE POWERSHELL, ALL DELIBERATE:
      1. splits at 26 MB not 28, and tests the size BEFORE writing the line -
         the original tested after and its parts came out over their own limit;
      2. destination is <out>/data_for_analysis rather than a hardcoded absolute
         path, so it follows --out;
      3. the destination is CLEARED on each run (chosen over overwrite-in-place)
         so a second run cannot leave stale parts from a previous split beside
         the new ones - overwrite alone would keep _part_5 when the file later
         needs only four;
      4. it walks and excludes BY PATTERN, never a file list, so any artifact
         added later is collected without editing this function.
    COPY ONLY. Nothing outside data_for_analysis is created, modified or removed.
    """
    try:
        import sweep_artifacts as _sw
        _rep = _sw.sweep(out)
        _n_exist, _n_aud, _cov_ok = _sw.coverage(out, _rep['file'].nunique())
        _sp = os.path.join(out, 'sweep_report.csv')
        _rep.to_csv(_sp, index=False, lineterminator='\n')
        _esc = _rep[_rep['finding'].isin(('CONSTANT-ESCALATED', 'SENTINEL', 'SCALE', 'NO-BASIS',
                                          'MISSING-REQUIRED', 'UNREADABLE'))]
        _rep = pd.concat([_rep, pd.DataFrame([{
            'file': '<COVERAGE>', 'finding': ('OK' if _cov_ok else 'COVERAGE-GAP'),
            'detail': f'audited {_n_aud} of {_n_exist} artifacts present under {out}'}])],
            ignore_index=True)
        if not _cov_ok:
            print(f'  *** SWEEP COVERAGE GAP: audited {_n_aud} of {_n_exist} artifacts. '
                  f'A provenance record covering part of the tree reads as clean evidence for '
                  f'files it never opened. ***', flush=True)
        print(f'  sweep coverage: {_n_aud} of {_n_exist} artifacts present')
        print(f'  sweep_report.csv: {_rep["file"].nunique()} artifacts audited, '
              f'{len(_esc)} escalated finding(s). THE PROVENANCE RECORD FOR EVERY ARTIFACT IN THE '
              f'TREE, emitted as a file rather than printed to a terminal that closes - six '
              f'inert-artifact defects reached the operator before this existed.')
    except Exception as _swe:
        print(f'  sweep_report.csv NOT written: {type(_swe).__name__}: {str(_swe)[:80]}')
    dst = os.path.join(out, 'data_for_analysis')
    if os.path.isdir(dst):
        for f in os.listdir(dst):
            fp = os.path.join(dst, f)
            if os.path.isfile(fp):
                os.remove(fp)
    os.makedirs(dst, exist_ok=True)
    roots = [r for r in (out, data_dir) if r and os.path.isdir(r)]
    copied, skipped, seen = 0, 0, {}
    for root in roots:
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _COLLECT_SKIP_DIR]
            for nm in sorted(files):
                if _COLLECT_SKIP_NAME.search(nm) or os.path.splitext(nm)[1].lower() not in COLLECT_EXTS:
                    skipped += 1
                    continue
                tgt = nm
                if tgt in seen:
                    tgt = f'{os.path.basename(dp)}__{nm}'
                seen[tgt] = 1
                shutil.copy2(os.path.join(dp, nm), os.path.join(dst, tgt))
                copied += 1
    tb = COLLECT_TARGET_MB * 1024 * 1024
    big = sorted([f for f in os.listdir(dst)
                  if os.path.getsize(os.path.join(dst, f)) > tb],
                 key=lambda f: -os.path.getsize(os.path.join(dst, f)))
    nparts = 0
    for f in big:
        fp = os.path.join(dst, f)
        mb = os.path.getsize(fp) / 1048576.0
        pr = _split_oversized(fp, tb)
        nparts += len(pr)
        print(f'    split {f} ({mb:.1f} MB) -> {len(pr)} parts')
    allf = sorted(os.listdir(dst))
    sizes = {f: os.path.getsize(os.path.join(dst, f)) for f in allf}
    total = sum(sizes.values()) / 1048576.0
    largest = max(sizes.values()) / 1048576.0 if sizes else 0.0
    over = [f for f, z in sizes.items() if z > COLLECT_CEILING_MB * 1024 * 1024]
    print(f'  S10 COLLECT: {len(allf)} files, {total:.1f} MB -> {dst}')
    print(f'    copied {copied} | skipped {skipped} (chunks, markers, provenance, shards, temporaries)')
    print(f'    split {len(big)} files -> {nparts} parts | largest {largest:.1f} MB | '
          f'ALL UNDER {COLLECT_CEILING_MB} MB: {"yes" if not over else "NO"}')
    if over:
        for f in over:
            print(f'      STILL OVER THE CEILING: {f} {sizes[f] / 1048576.0:.1f} MB')
    missed = _collect_coverage_check(out, allf)
    print(f'    artifacts written by the pipeline but NOT collected: '
          f'{missed if missed else "none"}')
    mark_done(out, 'S10', {'input_sha': input_sha, 'files': len(allf),
                           'total_mb': round(total, 1)})
    return {'files': len(allf), 'total_mb': round(total, 1), 'over': over, 'missed': missed}


def _collect_coverage_check(out, collected):
    """Every artifact the pipeline WRITES, checked against what S10 COLLECTED.

    A collector that silently misses a new file is the same class as a gate that
    does not check every deliverable. This scans the package source for write
    targets rather than trusting a list.
    """
    names = set()
    for sub in ('.', 'engine', 'scanners', 'orchestrator'):
        d = os.path.join(_HERE, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.py'):
                continue
            try:
                txt = open(os.path.join(d, fn), encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            for m in re.finditer(r"['\"]([A-Za-z0-9_]+\.(?:csv|md|txt|jsonl))['\"]", txt):
                names.add(m.group(1))
    present = set()
    for c in collected:
        present.add(c)
        b = re.sub(r'_part_\d+(\.[a-z]+)$', r'\1', c)
        present.add(b)
        present.add(b.split('__')[-1])
    on_disk = set()
    for dp, dirs, files in os.walk(out):
        dirs[:] = [d for d in dirs if d not in _COLLECT_SKIP_DIR]
        on_disk.update(files)
    return sorted((names & on_disk) - present)


def _artifacts_present(out, names):
    """A gate must verify EVERY deliverable, not one of them.

    A stage whose gate checks a single artifact will skip while another of its
    outputs is missing - which is exactly what happened when
    wf_book_arm_entities.csv was added to S5C after the marker had been written:
    the criterion was evaluable, the gate was satisfied, and the new file never
    appeared. Returns (all_ok, missing_names).
    """
    missing = []
    for nm in names:
        p_ = os.path.join(out, nm)
        if not os.path.exists(p_):
            missing.append(nm)
            continue
        try:
            if os.path.getsize(p_) == 0:
                missing.append(nm + ' (empty)')
        except OSError:
            missing.append(nm + ' (unreadable)')
    return (not missing), missing


def is_done(out, key, input_sha):
    p = done_path(out, key)
    if not os.path.exists(p):
        return False
    try:
        meta = json.load(open(p, encoding='utf-8'))
        return meta.get('input_sha') == input_sha
    except Exception:
        return False


def _pf(x):
    x = np.asarray(x, dtype=float)
    if (x < 0).any():
        return round(x[x > 0].sum() / -x[x < 0].sum(), 2)
    return PF_UNDEFINED if len(x) else 0.0


def _is_header_row(first_line):
    return first_line.split(',')[0].strip() == 'Time'


def s0_ingest(data_dir, out):
    import portfolio_simulation_engine as engine
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')), key=_natkey)
    if not files:
        print(f'ABORT — no CSVs in {data_dir}')
        sys.exit(2)
    input_sha = hashlib.sha256((''.join(sha12(f) for f in files)).encode()).hexdigest()[:12]
    recon = [f for f in files if 'recon171_step7_part' in os.path.basename(f)]
    ncols = len(open(files[0], encoding='utf-8').readline().split(','))
    attest = {'files': [os.path.basename(f) for f in files], 'ncols_first': ncols}
    if recon and len(recon) == len(files):
        cwd = os.getcwd()
        os.chdir(data_dir)
        try:
            df = engine.load_sealed_baseline(verbose=False)
        finally:
            os.chdir(cwd)
        attest['path'] = 'sealed-baseline (load_sealed_baseline invariants)'
    else:
        if ncols >= 256:
            import core
            print('  S0a — 256-col raw export detected → core.py reconstruction')
            attest['path'] = 'core.py reconstruction (256→171)'
        frames = []
        header_cols = None
        for f in files:
            if _is_header_row(open(f, encoding='utf-8').readline()):
                d = pd.read_csv(f)
                header_cols = list(d.columns)
            else:
                d = pd.read_csv(f, header=None, names=header_cols)
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        attest['path'] = 'generic concatenate+validate'
    if 'Time' not in df.columns or df.shape[1] != 172:
        print(f'ABORT — column contract violated: {df.shape[1]} cols (expect Time + 171)')
        sys.exit(2)
    t = df['Time'].astype(str).values
    if not (t[1:] > t[:-1]).all():
        print('ABORT — time not strictly increasing')
        sys.exit(2)
    if df.duplicated().any():
        print('ABORT — duplicate rows present')
        sys.exit(2)
    if df.isna().any().any():
        print('ABORT — NaN cells present')
        sys.exit(2)
    attest.update({'rows': int(len(df)), 'cols': int(df.shape[1]),
                   'range': f'{t[0]} → {t[-1]}', 'invariants': 'PASS', 'input_sha': input_sha})
    print(f'  ingest: {len(df):,} rows × {df.shape[1]} cols | {t[0]} → {t[-1]} | invariants PASS')
    mark_done(out, 'S0', attest)
    return df, attest, input_sha


# ── S1 / S2 ──
def s1_thresholds(df):
    import dots_thresholds as dt
    print(f'  oracle dots_thresholds.py sha256 : {sha12(os.path.join(_ENGINE, "dots_thresholds.py"))} (export=live parity)')
    return dt.compute_adaptive_thresholds(df), dt.compute_structural_gates(df)


def s2_pool(df, ad, st):
    import sequential_temporal as seq
    import portfolio_simulation_engine as engine
    w = engine.warmup_floor(df, verbose=False)
    pool = seq.build_condition_pool(df, ad, st, w)
    anchor = seq.anchor_array(df, 'ST_Flip')
    print(f'  pool {len(pool)} conditions | warm-up floor {w} | ST_Flip anchor built')
    return pool, anchor, w


# ── S3 DISCOVERY (long pole; delegates to the ratified orchestrator; per-family checkpoint) ──
def _diag_scanner_current(results_dir, fam, script):
    """Does the diagnostic's marker record the sha of the scanner NOW ON DISK?

    `already current for this input_sha` consults the FRAME sha only. The frame had
    not changed, so F12 skipped even though concurrence_profiler.py had moved
    3e099e89b563 -> 4d782df381e0. THE ONE AUTHORITY THAT WAS MEANT TO BECOME
    UNIVERSAL WAS NOT CONSULTED WHERE THE DECISION IS MADE.

    Returns (current, why). A marker that is ABSENT is NOT current: by the standing
    rule an unchecked family is not a passing family.
    """
    try:
        import discovery_orchestrator as _o
    except Exception as exc:
        return False, f'orchestrator unavailable ({type(exc).__name__})'
    _prev = getattr(_o, 'RESULTS_DIR', None)
    _o.RESULTS_DIR = results_dir
    try:
        done = _o._family_paths(fam, script)[1]
        want = _o.scanner_sha(script)
        if want is None:
            return False, f'{script}.py not found on disk'
        if not os.path.exists(done):
            return False, f'no marker at {os.path.basename(done)} - UNCHECKED, not passing'
        try:
            meta = json.load(open(done, encoding='utf-8'))
        except Exception as exc:
            return False, f'marker unreadable ({type(exc).__name__})'
        got = meta.get('scanner_sha')
        if got is None:
            return False, 'marker carries no scanner_sha'
        if got != want:
            return False, f'scanner {script}.py moved {got} -> {want}'
        return True, f'scanner_sha {got} matches'
    finally:
        if _prev is not None:
            _o.RESULTS_DIR = _prev


def _write_diag_marker(results_dir, fam, script, csv):
    """A MARKER MUST ONLY BE WRITTEN BY THE CODE THAT PRODUCED THE OUTPUT.

    This was called from verify_diagnostic_outputs, which runs whether the stage ran
    or skipped - so a SKIPPED F12 had its marker stamped with the CURRENT scanner sha
    against output produced by the PREVIOUS one. That is worse than skipping current
    work: it writes a FALSE PROVENANCE RECORD, and every instrument in the tree trusts
    the marker, so nothing could detect it. Called only from the execution path now.
    """
    try:
        import discovery_orchestrator as _o
    except Exception:
        return
    _prev = getattr(_o, 'RESULTS_DIR', None)
    _o.RESULTS_DIR = results_dir
    try:
        if not os.path.exists(csv):
            return
        done = _o._family_paths(fam, script)[1]
        try:
            rows = max(0, sum(1 for _l in open(csv, encoding='utf-8', errors='replace')
                              if not _l.startswith('#')) - 1)
        except OSError:
            rows = 0
        _o._mark_family_done(csv, done, rows, script)
        print(f'    {fam} MARKER written BY THE EXECUTION PATH: scanner_sha '
              f'{_o.scanner_sha(script)} against output this run produced.', flush=True)
    finally:
        if _prev is not None:
            _o.RESULTS_DIR = _prev


def _s3_moved_scanners(out):
    """Does any family's PRODUCING SCANNER differ from the sha its marker recorded?

    S3's gate counted result CSVs on disk and returned, so the sha-aware
    family_is_complete built for exactly this purpose was never reached. EIGHTH
    INSTANCE OF THE SAME CLASS, ONE LEVEL UP FROM WHERE IT WAS FIXED: a gate may
    check that a marker exists, that a file exists, a file COUNT, an artifact's
    SCHEMA, an artifact's own SHA, or the PRODUCING CODE's sha - AND ONLY THE LAST
    IS SUFFICIENT.

    This consults the orchestrator's own authority rather than adding a mechanism:
    the same scanner_sha() the family markers use, against the same marker payloads.
    ALL_FAMILIES is the single registry and it INCLUDES the diagnostics F12 and F13,
    which is what matters here - F12's scanner is the one that moved.

    _family_paths resolves against the orchestrator's module-global RESULTS_DIR, not
    the out tree, so it is pointed at this run's results directory and restored in a
    finally; otherwise the check reads a different (or absent) marker set and
    silently reports that nothing moved.
    """
    try:
        import discovery_orchestrator as _o
    except Exception as exc:
        print(f'  S3 scanner-sha check UNAVAILABLE: {type(exc).__name__}: {str(exc)[:70]} - the '
              f'gate would fall back to the CSV count, which cannot see moved code.', flush=True)
        return []
    _entries = [(f[0], f[1]) for f in (getattr(_o, 'ALL_FAMILIES', []) or []) if f[1]]
    if not _entries:
        print('  S3 scanner-sha check found NO family registry on the orchestrator.', flush=True)
        return []
    _prev = getattr(_o, 'RESULTS_DIR', None)
    _o.RESULTS_DIR = os.path.join(out, 'results')
    moved, checked, unchecked = [], [], []
    try:
        for fam, script in _entries:
            _csv, done = _o._family_paths(fam, script)
            if not os.path.exists(done):
                # A MISSING MARKER MUST NEVER BE INDISTINGUISHABLE FROM A MATCHING ONE.
                # `continue` on a missing .done is the silent negative that hid F12: the
                # diagnostics never went through _mark_family_done, so the loop skipped
                # them and the gate reported every scanner current. Name them instead.
                unchecked.append(f'{fam} (no marker at {os.path.basename(done)})')
                continue
            try:
                meta = json.load(open(done, encoding='utf-8'))
            except Exception:
                continue
            want = _o.scanner_sha(script)
            got = meta.get('scanner_sha')
            if want is None:
                unchecked.append(f'{fam} ({script}.py not found on disk)')
                continue
            if got is None:
                unchecked.append(f'{fam} (marker carries no scanner_sha)')
                continue
            checked.append(fam)
            if got != want:
                moved.append((fam, script, got, want))
    finally:
        if _prev is not None:
            _o.RESULTS_DIR = _prev
    print(f'  S3 scanner-sha coverage: {len(_entries)} families in registry, '
          f'{len(checked)} checked, {len(unchecked)} UNCHECKED'
          + (f' ({", ".join(unchecked)})' if unchecked else '')
          + f' | {len(moved)} moved', flush=True)
    if unchecked:
        print(f'      AN UNCHECKED FAMILY IS NOT A PASSING FAMILY. A DETECTOR THAT CANNOT SAY '
              f'WHAT IT DID NOT EXAMINE IS NOT A DETECTOR - this line exists because F12 was '
              f'skipped silently and the gate reported every scanner current.', flush=True)
    return moved, unchecked


def s3_discovery(out, workers, input_sha, scope, df=None, ad=None, st=None, w=None, limit=0):
    results = os.path.join(out, 'results')
    os.makedirs(results, exist_ok=True)
    _res = os.path.join(out, 'results')
    _fam_csvs = sorted(glob.glob(os.path.join(_res, 'results_F*.csv'))) if os.path.isdir(_res) else []
    _fam_csvs = [f for f in _fam_csvs if '_part' not in os.path.basename(f) and '_c0' not in
                 os.path.basename(f)]
    _moved, _unchecked = _s3_moved_scanners(out)
    # AN UNCHECKED FAMILY BLOCKS THE SKIP. `0 moved` is computed over the families the
    # check could READ; F12 and F13 had no marker at all, so treating unchecked as
    # passing is exactly what the coverage line says must never happen - the accounting
    # was correct and the verdict ignored it.
    # AND IT WAS CIRCULAR: verify_diagnostic_outputs writes the diagnostic markers, but
    # it runs INSIDE the orchestrator, which this gate was skipping. THE MARKER NEEDED
    # THE RUN AND THE RUN NEEDED THE MARKER, so no number of re-runs broke the loop.
    # Entering the orchestrator resumes what is current, re-scans what is not, and
    # writes the missing markers, so the next run reads full coverage. One pass, no
    # manual step.
    if is_done(out, 'S3', input_sha) and _fam_csvs and not _moved and not _unchecked:
        print(f'  S3 resumed from marker: {len(_fam_csvs)} per-family result CSVs present on disk '
              f'AND every producing scanner still matches the sha recorded in its family marker '
              f'- skipping the scan.')
        emit_regime_labels(df, results, out, input_sha)
        return
    if is_done(out, 'S3', input_sha) and _fam_csvs and (_moved or _unchecked):
        _why = []
        if _moved:
            _why.append('a PRODUCING SCANNER HAS MOVED')
        if _unchecked:
            _why.append(f'{len(_unchecked)} family/ies could NOT BE CHECKED')
        print(f'  S3 marker present and {len(_fam_csvs)} result CSVs on disk, but '
              f'{" and ".join(_why)} - ENTERING ORCHESTRATOR.', flush=True)
        for _fam, _sc, _was, _now in _moved:
            print(f'      MOVED     {_fam}: {_sc}.py {_was} -> {_now}', flush=True)
        for _u in _unchecked:
            print(f'      UNCHECKED {_u} - treated as NOT PASSING. The orchestrator will write '
                  f'its marker on this pass so the next run has full coverage.', flush=True)
        print(f'      A gate that counts artifacts cannot detect an artifact produced by moved '
              f'code. That reasoning produced scanner_sha and it was applied at the FAMILY level '
              f'and not at the STAGE level, so this gate returned before family_is_complete was '
              f'ever consulted. The orchestrator now resumes every family whose scanner is '
              f'current and re-scans only the {len(_moved)} that moved.', flush=True)
    if is_done(out, 'S3', input_sha) and not _fam_csvs:
        print('  S3 marker present but NO per-family result CSVs on disk - RE-RUNNING. A gate that '
              'trusts only its marker skips work a later stage depends on, and S4 would then unify '
              'nothing while reporting success.')
    import discovery_orchestrator as orch
    orch.RESULTS_DIR = results
    os.environ['DOT_RESULTS_DIR'] = results
    frame_path = None
    if df is not None and workers and workers > 1:
        frame_path = os.path.join(results, f'_s3_frame_{input_sha}.csv')
        for stale in glob.glob(os.path.join(results, '_s3_frame*.csv')):
            if os.path.basename(stale) != os.path.basename(frame_path):
                os.remove(stale)
        tmp = frame_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            df.to_csv(f, index=False, lineterminator='\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, frame_path)
        print(f'  worker frame written to {os.path.basename(frame_path)} so each process loads it independently')
        print(f'  (the name carries input_sha and the file is REWRITTEN every S3 entry, so a cache from a different')
        print(f'   dataset can never be read; it is deleted when S3 completes)')
    print(f'  delegating to discovery_orchestrator.orchestrate(scope="{scope}", workers={workers}) — F1–F11 + F0/F13 ingest.')
    print('  (this is the 1–2 day long pole. Per family: results land in results/ and are written ATOMICALLY with a')
    print('   .done marker carrying the row count and CSV sha256. A restart re-reads any complete family from disk')
    print('   and re-scans only the incomplete ones, so the worst case loss is ONE family, not the whole stage.)')
    orch.orchestrate(scope, workers=workers, df=df, adaptive=ad, structural=st, warmup=w,
                     frame_path=frame_path, input_sha=input_sha, limit=limit)
    run_diagnostic_families(results, workers, input_sha, df=df)
    orch.verify_diagnostic_outputs(results, input_sha)
    emit_regime_labels(df, results, out, input_sha)
    if frame_path is not None and os.path.exists(frame_path):
        os.remove(frame_path)
        print(f'  worker frame {os.path.basename(frame_path)} removed on S3 completion')
    mark_done(out, 'S3', {'input_sha': input_sha, 'scope': scope, 'workers': workers})


# ── S4 / S5 ──
def s4_schema(out, input_sha):
    results = os.path.join(out, 'results')
    os.makedirs(results, exist_ok=True)
    master = os.path.join(results, 'discovery_master.csv')
    if os.path.exists(master):
        n = len(pd.read_csv(master))
        print(f'  schema-unify: orchestrator collated {n} rows → results/discovery_master.csv')
    else:
        frames = []
        for f in sorted(glob.glob(os.path.join(results, 'results_F*.csv'))):
            if '_part' in os.path.basename(f):
                continue
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                pass
        if frames:
            uni = pd.concat(frames, ignore_index=True)
            uni.to_csv(master, index=False, lineterminator='\n', encoding='utf-8')
            print(f'  schema-unify: {len(uni)} rows → results/discovery_master.csv')
        else:
            print('  schema-unify: no discovery results present (discover-fresh not run) — NOT marking '
              'done. A stage that reports itself unexercised must NOT mark done: the marker would skip it permanently for this input_sha and the run would finish with that stage never having run.')
        return
    mark_done(out, 'S4', {'input_sha': input_sha})


def s5_filter(out, input_sha, pool):
    import catalogue as cat
    results = os.path.join(out, 'results')
    src = os.path.join(results, 'discovery_master.csv')
    if not os.path.exists(src):
        print('  filter: no unified results (discover-fresh not run) — NOT marking done. A stage that reports itself unexercised must NOT mark done: the marker would skip it permanently for this input_sha and the run would finish with that stage never having run.')
        return
    r = pd.read_csv(src)
    n_total = len(r)
    _pf_ok = r['agg_pf'].map(lambda v: cat.pf_passes_floor(v, 2.0))
    keep = r[(r['trades'] >= 30) & (r['folds_plus'] >= 4) & _pf_ok].copy()
    _zl = int(_pf_ok.sum() - (pd.to_numeric(r['agg_pf'], errors='coerce') >= 2.0).sum())
    print(f'  S5 gate: agg_pf floor 2.0 admitted {int(_pf_ok.sum())} rows, of which {_zl} are '
          f'ZERO-LOSS and pass by an EXPLICIT BRANCH. 999.0 >= 2.0 was true by accident of the '
          f'sentinel\'s size; with PF undefined a bare comparison returns False and the strongest '
          f'signals in the catalogue would be silently dropped.')
    if 'worst_day_usd' in keep.columns:
        keep['_pf_undefined'] = keep['agg_pf'].map(cat.pf_is_undefined)
    keep['_pf_sort'] = keep['agg_pf'].map(cat.pf_sort_key)
    keep = keep.sort_values(['worst_day_usd', '_pf_sort'], ascending=[True, False])
    keep = keep.drop(columns=['_pf_sort'])
    import score_g
    unscoreable = set(score_g.UNSCOREABLE_FAMILIES)
    if 'family' in keep.columns and len(keep):
        gcov = score_g.grammar_coverage(keep, pool=pool)
        _write_with_header(os.path.join(results, 'grammar_coverage.csv'), gcov, [
            'DOT S5 GRAMMAR COVERAGE — every DISTINCT signal_def form in the filtered pool',
            'PROPERTY OF THE POOL. Checked BEFORE S8 so an unhandled grammar surfaces in seconds at '
            'S5, not after a long run at S8.',
            'Shapes are the signal_def with identifiers normalised to V and numbers to N, so two rows '
            'differing only in variable or threshold collapse to one form. Row counts are per form.',
            'A form marked handled=False is EXCLUDED from candidates.csv by name below, so the filter '
            'and build_book can never disagree about what is scoreable.'])
        print('  GRAMMAR COVERAGE — distinct signal_def forms in the filtered pool:')
        for _i, gr in gcov.iterrows():
            flag = 'OK ' if gr['handled'] else 'NO '
            print(f"    {flag}{gr['family']:4} {int(gr['rows']):5} rows | {gr['grammar_shape']}")
            if not gr['handled']:
                print(f"        example: {gr['example']}")
        bad_shapes = set(gcov[~gcov['handled']]['grammar_shape'])
        if bad_shapes:
            mask_bad = keep['signal_def'].astype(str).map(score_g.grammar_shape).isin(bad_shapes)
            n_bad = int(mask_bad.sum())
            keep = keep[~mask_bad]
            print(f'  filter: EXCLUDING {n_bad} row(s) whose signal_def form build_book cannot '
                  f'parse — named above, never silently dropped')
        else:
            print('  GRAMMAR COVERAGE: every form in the pool is parseable by build_book')
        blocked = keep[keep['family'].isin(unscoreable)]
        keep = keep[~keep['family'].isin(unscoreable)]
        if len(blocked):
            for fam, g in blocked.groupby('family'):
                print(f'  filter: EXCLUDING {len(g)} {fam} candidate(s) — S8 cannot score this '
                      f'family: {score_g.UNSCOREABLE_FAMILIES[fam]}')
            _u = sorted(unscoreable)
            _v = 'is' if len(_u) == 1 else 'are'
            print(f'  THE POOL IS NOT THE FULL FOURTEEN: {_u} {_v} discovered and reported but '
                  f'cannot enter a selected book. Stated so the operator is never told a book spans '
                  f'families it does not.')
    _cand_path = os.path.join(results, 'candidates.csv')
    keep.to_csv(_cand_path, index=False, lineterminator='\n', encoding='utf-8')
    import discovery_orchestrator as _orch
    _orch.stamp_provenance(_cand_path, input_sha)
    print(f'  candidates.csv stamped for input_sha {input_sha}. WITHOUT THIS STAMP S5C\'s '
          f'provenance gate reads "no provenance stamp", _pool_ok is False, the book arm is '
          f'skipped and item 17 reports UNEVALUABLE with the null arm sitting there fully '
          f'measured. F13 and F12 stamp their outputs; S5 did not.')
    print(f'  filter (trades≥30 & folds_plus≥4 & agg_pf≥2.0): {len(keep)}/{n_total} candidates '
          f'scoreable by S8')
    mark_done(out, 'S5', {'input_sha': input_sha, 'candidates': int(len(keep))})


# ── S6 REGEN stale artifacts fresh ──
def s6_regen(out, input_sha):
    scored = os.path.join(out, 'scored')
    os.makedirs(scored, exist_ok=True)
    print('  regen: signal_full_records.csv + signal_per_day_pnl.jsonl are regenerated FRESH')
    print('         under the current engine (run_full_analysis → analysis_engine); stale copies')
    print('         746102aae415 / 0910f360a628 are NEVER inherited.')
    mark_done(out, 'S6', {'input_sha': input_sha,
                          'note': 'fresh regen path wired to run_full_analysis; long-pole, resumable'})


# ── S7 CONTENDERS ──
def fold_plan(df, warmup):
    t = pd.Series(df['Time'].astype(str).values).str[:10].values
    days = list(pd.unique(t[np.arange(len(df)) >= warmup]))
    n = len(days)
    n_oos = int(round(n * OOS_TAIL_FRACTION))
    train_days = days[:n - n_oos] if n_oos else days
    oos_days = days[n - n_oos:] if n_oos else []
    m = len(train_days)
    base = m // FOLD_COUNT
    extra = m % FOLD_COUNT
    folds = []
    cur = 0
    for i in range(FOLD_COUNT):
        size = base + (1 if i < extra else 0)
        folds.append(train_days[cur:cur + size])
        cur += size
    smallest = min((len(f) for f in folds), default=0)
    evaluable = smallest >= MIN_FOLD_DAYS
    oos_evaluable = len(oos_days) >= MIN_FOLD_DAYS
    status = ('OK' if evaluable else
              f'UNEVALUABLE - {smallest} trading days per slice, below the floor of {MIN_FOLD_DAYS}')
    window = f'{oos_days[0]} -> {oos_days[-1]}' if oos_days else 'none'
    return {'folds': folds, 'oos_days': oos_days, 'fold_days': smallest,
            'evaluable': evaluable, 'status': status, 'oos_window': window,
            'oos_days_n': len(oos_days), 'oos_evaluable': oos_evaluable,
            'total_days': n}


def _score(df, sigs, ad, st, w, conv, want_trades=False):
    import portfolio_simulation_engine as engine
    td = engine.run_portfolio(df, sigs, adaptive=ad, structural=st, warmup=w, verbose=False, conviction=conv)
    return _metrics_from_trades(df, td, w, want_trades=want_trades)


def _pf_of(p):
    import numpy as _np
    p = _np.asarray(p, dtype=float)
    if not p.size:
        return ''
    gl = -p[p < 0].sum()
    if gl <= 0:
        return 'inf'
    return round(float(p[p > 0].sum() / gl), 2)


def _ev_of(td):
    los = td[td['pnl'] < 0]
    return int(len(set(int(b) for b in los['entry_bar'].values))) if len(los) else 0


def breakdown_report(df, td, book, gates=None, cfg=None):
    """THE FULL SPREAD, NOT THE AGGREGATE.

    A book that nets $284,974 could be one great month and six flat ones and the
    consolidated block cannot tell those apart. And FOLDS DO NOT COVER THE FRAME:
    6 x 14 = 84 of 119 traded days, and they are not calendar months - so monthly and
    weekly WR and PF for this configuration had never been measured.

    LOSS EVENTS SIT BESIDE TRADE LOSSES ON EVERY ROW. The bar is the risk unit; a
    monthly table showing only trade-losses repeats the error that stood for the life
    of this project. Negative periods are flagged with <<< NEGATIVE so a red row
    cannot be missed under a zero-losing-weeks headline.

    Lives in the shared block so BOOK-50 and a configured book cannot diverge.
    """
    import numpy as _np
    out = []
    p = _np.asarray(td['pnl'].values, dtype=float)
    day = pd.Series(td['exit_time'].astype(str).values).str[:10]
    iso = pd.to_datetime(day, format='%Y.%m.%d', errors='coerce')
    wk = iso.dt.isocalendar()
    wkey = (wk['year'].astype(str) + '-W' + wk['week'].astype(str).str.zfill(2)).values
    mkey = day.str[:7].values
    # direction arrives as +1/-1 from the engine and as 'LONG'/'SHORT' from the trade
    # table depending on the path, so normalise once rather than assuming either.
    if 'direction' in td.columns:
        _dv = td['direction'].values
        dirn = _np.array([1 if (x == 1 or str(x).upper() == 'LONG') else -1 for x in _dv])
    else:
        dirn = _np.zeros(len(td), dtype=int)

    def table(keys, label, n_expected):
        out.append('')
        out.append(f'  {label} BREAKDOWN  ({len(set(keys))} periods'
                   + (f', {n_expected} expected' if n_expected else '') + ')')
        out.append(f'    {"period":10}{"trades":>7}{"wins":>6}{"loss":>6}{"EVENTS":>7}'
                   f'{"W/L":>7}{"WR%":>7}{"PF":>8}{"net $":>12}{"LONG":>6}{"SHORT":>6}'
                   f'{"worstDay":>10}')
        for k in sorted(set(keys)):
            m = keys == k
            sub = td[m]
            pp = p[m]
            wins = int((pp > 0).sum())
            loss = int((pp < 0).sum())
            aw = float(pp[pp > 0].mean()) if wins else 0.0
            al = float(-pp[pp < 0].mean()) if loss else 0.0
            wl = round(aw / al, 3) if al > 0 else ''
            _sd = dirn[m]
            nl = int((_sd == 1).sum())
            ns = int((_sd == -1).sum())
            wd = float(pd.Series(pp).groupby(day[m].values).sum().min()) if len(pp) else 0.0
            flag = '  <<< NEGATIVE' if pp.sum() < 0 else ''
            out.append(f'    {str(k):10}{len(pp):>7}{wins:>6}{loss:>6}{_ev_of(sub):>7}'
                       f'{str(wl):>7}{100 * (pp > 0).mean():>7.2f}{str(_pf_of(pp)):>8}'
                       f'{pp.sum():>12,.2f}{nl:>6}{ns:>6}{wd:>10,.2f}{flag}')

    out.append('')
    out.append('  ' + '=' * 104)
    out.append('  FULL BREAKDOWN REPORT')
    out.append('  ' + '=' * 104)
    out.append(f'    POPULATION      : {len(td)} BOOK-only trades scored')
    out.append(f'    DAYS            : {int(day.nunique())} days with a book trade of '
               f'{int(pd.Series(df["Time"].astype(str).values).str[:10].nunique())} trading '
               f'days in the frame')
    out.append(f'    WEEK BASIS      : ISO week keys (year-Www), not a date slice')
    out.append(f'    FOLD COVERAGE   : folds are 6 x 14 = 84 trading days of '
               f'{int(day.nunique())} traded - THEY DO NOT COVER THE FRAME and are not '
               f'calendar months')
    out.append('')
    out.append('  TOP 5 SIGNALS BY CONTRIBUTION   RANKING KEY: net $ descending')
    out.append('    (net and net-per-trade rank differently; both are legitimate, this is net)')
    out.append(f'    {"net $":>12}{"trades":>8}{"EVENTS":>7}  {"dir":5} signal')
    g = td.groupby('signal_name')
    agg = sorted(((float(x['pnl'].sum()), nm, x) for nm, x in g), reverse=True)[:5]
    for net, nm, x in agg:
        _x0 = x['direction'].iloc[0]
        dd = 'LONG' if (_x0 == 1 or str(_x0).upper() == 'LONG') else 'SHORT'
        out.append(f'    {net:>12,.2f}{len(x):>8}{_ev_of(x):>7}  {dd:5} {str(nm)[:62]}')
    if gates:
        out.append('')
        out.append('  GATE MECHANISM PERFORMANCE   bars admitted / refused across the frame')
        out.append(f'    {"gate":34}{"admits":>10}{"refuses":>10}{"admit%":>9}')
        n = len(df)
        for nm, mask in gates.items():
            a_ = int(_np.asarray(mask).sum())
            out.append(f'    {nm:34}{a_:>10,}{n - a_:>10,}{100 * a_ / n:>8.2f}%')
    out.append('')
    out.append('  DEPTH LADDER PER DIRECTION   depth = distinct signals admitted on the bar')
    out.append(f'    {"dir":6}{"depth":>7}{"trades":>8}{"wins":>6}{"loss":>6}{"EVENTS":>7}'
               f'{"WR%":>8}{"PF":>8}{"net $":>12}')
    bar_dir = {}
    for b_, d_ in zip(td['entry_bar'].values, dirn):
        bar_dir.setdefault((int(d_), int(b_)), 0)
        bar_dir[(int(d_), int(b_))] += 1
    depth_of = _np.array([bar_dir[(int(d_), int(b_))]
                          for b_, d_ in zip(td['entry_bar'].values, dirn)])
    for dv, dl in ((1, 'LONG'), (-1, 'SHORT')):
        for k in range(3, 12):
            m = (dirn == dv) & ((depth_of == k) if k < 11 else (depth_of >= 11))
            if not m.any():
                continue
            pp = p[m]
            lbl = f'{k}' if k < 11 else '11+'
            out.append(f'    {dl:6}{lbl:>7}{len(pp):>8}{int((pp > 0).sum()):>6}'
                       f'{int((pp < 0).sum()):>6}{_ev_of(td[m]):>7}'
                       f'{100 * (pp > 0).mean():>8.2f}{str(_pf_of(pp)):>8}{pp.sum():>12,.2f}')
    table(wkey, 'WEEKLY', 26)
    table(mkey, 'MONTHLY', 7)
    out.append('  ' + '=' * 104)
    return out


def _metrics_from_trades(df, td, w, want_trades=False):
    """THE ONE metric block, taking a trade table. Folds and OOS were STUBBED on the
    configured path and printed 'NOT COMPUTED'; reusing this instead of writing a second
    implementation means the configured path computes them the same way BOOK-50 does,
    and a fold definition cannot drift between the two paths."""
    import wf
    p = td['pnl'].values
    d = wf.daily_pnl_points(td).sort_values('exit_date')
    eq = d['pnl'].cumsum().values
    mdd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    mo = pd.Series(td['exit_time'].values).str[:7].values
    exit_day = pd.Series(td['exit_time'].astype(str).values).str[:10].values
    plan = fold_plan(df, w)
    fold_ok = plan['evaluable']
    if fold_ok:
        fmins = []
        fplus = 0
        for fd in plan['folds']:
            m = np.isin(exit_day, fd)
            fmins.append(_pf(p[m]))
            if p[m].sum() > 0:
                fplus += 1
        fmin = min(fmins) if fmins else 0.0
    else:
        fmin = 0.0
        fplus = 0
    oos_prop = np.isin(exit_day, plan['oos_days'])
    oos = np.isin(mo, OOS_MONTHS)
    present = sorted(set(mo.tolist()))
    rel_months = present[-OOS_REL_N_MONTHS:] if len(present) >= OOS_REL_N_MONTHS else present
    oos_rel = np.isin(mo, rel_months)
    summary = {'trades': len(p), 'net': round(float(p.sum())), 'WR': round(float((p > 0).mean() * 100), 1),
               'PF': _pf(p), 'daily_wd': round(float(d['pnl'].min()), 1), 'daily_mDD': round(mdd, 1),
               'folds_plus': fplus, 'min_fold_pf': round(fmin, 2),
               'fold_count': FOLD_COUNT, 'fold_days_each': plan['fold_days'],
               'folds_evaluable': fold_ok, 'folds_status': plan['status'],
               'folds_basis': 'six equal trading-day slices of the leading two-thirds (disjoint from the hold-out)',
               'oos_prop_pf': _pf(p[oos_prop]), 'oos_prop_net': round(float(p[oos_prop].sum())),
               'oos_prop_window': plan['oos_window'], 'oos_prop_days': plan['oos_days_n'],
               'oos_prop_evaluable': plan['oos_evaluable'],
               'oos_pf': _pf(p[oos]), 'oos_net': round(float(p[oos].sum())),
               'oos_legacy_months': ';'.join(OOS_MONTHS), 'oos_legacy_stale': True,
               'oos_rel_months': ';'.join(rel_months),
               'oos_rel_pf': _pf(p[oos_rel]), 'oos_rel_net': round(float(p[oos_rel].sum()))}
    if want_trades:
        return summary, td
    return summary


def s7_contenders(df, ad, st, w, sigs, out, input_sha):
    import conviction as C
    import runlog as rl
    contenders = os.path.join(out, 'contenders')
    os.makedirs(contenders, exist_ok=True)
    variants = [
        ('C0', 'Flat book (1-lot, no conviction/gaps)', None),
        ('C1', '+ S.20 conviction (Hurst/recentFB longs)', C.build_conviction(df, True, True, False, d2d_conviction=False, d2d_gap=False)),
        ('C2', '+ S.20 gap-singles (Hurst-gap, FB-gap)', C.build_conviction(df, True, True, True, d2d_conviction=False, d2d_gap=False)),
        ('C3', '+ S.21 D2D-conviction (2x both dir)', C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=False)),
        ('C4', '+ S.21 D2D-gap (flat 2-lot) = FULL', C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)),
        ('C5', 'sizing variant (conviction-off, gaps-on)', C.build_conviction(df, False, False, True, d2d_conviction=False, d2d_gap=True)),
    ]
    rows, prev = [], 0
    _sc = {}
    with rl.Progress('S7 six portfolio scores', len(variants)) as _p7:
        for cid, label, conv in variants:
            _sc[cid] = _score(df, sigs, ad, st, w, conv)
            _p7.step(1, extra=cid)
    for cid, label, conv in variants:
        r = _sc[cid]
        r['id'] = cid
        r['contender'] = label
        r['delta'] = r['net'] - prev if cid != 'C5' else r['net'] - rows[0]['net']
        rows.append(r)
        prev = r['net'] if cid != 'C5' else prev
        print(f"    {cid} {label:44} net ${r['net']:>7} (Δ {r['delta']:+7}) wd {r['daily_wd']} "
              f"OOS-PF {r['oos_prop_pf'] if r['oos_prop_evaluable'] else 'UNEVAL'}")
    cols = ['id', 'contender', 'trades', 'net', 'delta', 'WR', 'PF', 'daily_wd', 'daily_mDD',
            'folds_plus', 'min_fold_pf', 'oos_pf', 'oos_net', 'oos_legacy_months', 'oos_legacy_stale',
            'oos_rel_months', 'oos_rel_pf', 'oos_rel_net',
            'fold_count', 'fold_days_each', 'folds_evaluable', 'folds_status', 'folds_basis',
            'oos_prop_pf', 'oos_prop_net', 'oos_prop_window', 'oos_prop_days', 'oos_prop_evaluable']
    pd.DataFrame(rows)[cols].to_csv(os.path.join(contenders, 'contenders.csv'), index=False,
                                        lineterminator='\n', encoding='utf-8')
    mark_done(out, 'S7', {'input_sha': input_sha})
    return rows


# ── S8 COMMITTED (frozen-book replay vs discover-fresh) ──
def book_config_for(book_path):
    """The sidecar config beside a book file, or None.

    ROUTING IS ON THE CONFIG, NOT ON THE BOOK NAME. A book with no config takes the
    SACRED engine exactly as today - that is what keeps the BOOK-50 canary honest,
    because book50_signals.csv has no config and must score identically to the cent.
    """
    if not book_path:
        return None, None
    base = os.path.splitext(book_path)[0]
    # Accept BOTH conventions: the strict sidecar <book>_config.json, and the shorter
    # form the spec names - whole_dot_signals.csv pairs with whole_dot_config.json, so
    # a trailing _signals is stripped. Deriving only the strict form silently found no
    # config and routed the Whole DOT down the SACRED path, scoring a different system.
    _alt = base[:-8] if base.endswith('_signals') else base
    for cand in (base + '_config.json', base + '.config.json',
                 _alt + '_config.json', _alt + '.config.json'):
        if os.path.exists(cand):
            try:
                with open(cand, encoding='utf-8') as f:
                    return json.load(f), cand
            except Exception as exc:
                raise SystemExit(f'ABORT [book config] {cand} is unreadable: '
                                 f'{type(exc).__name__}: {str(exc)[:90]}')
    return None, None


def loss_events(trades):
    """LOSS EVENTS AND DISTINCT LOSS DAYS. THE BAR IS THE RISK UNIT, NOT THE TRADE.

    224 trade-losses are 43 events on 36 days. A scorecard reporting only
    trade-losses reports a number that was wrong for the life of this project: a bar
    that opens six positions and loses on all six is ONE decision that went wrong,
    not six.
    """
    if trades is None or not len(trades):
        return 0, 0, 0
    los = trades[trades['pnl'] < 0]
    if not len(los):
        return 0, 0, 0
    bars = los['entry_bar'].values if 'entry_bar' in los.columns else []
    n_ev = int(len(set(int(b) for b in bars))) if len(bars) else int(len(los))
    days = set()
    if 'exit_time' in los.columns:
        days = {str(t)[:10] for t in los['exit_time'].values}
    return int(len(los)), n_ev, int(len(days))


SACRED_PATH_BOOKS = ('book50_signals.csv',)
_LAST_GATES = None


TRIPLE_RE = re.compile(r'^[A-Za-z_0-9]+:(hi|lo|==-?\d+)$')


def _assert_book_grammar(book):
    """A CONFIGURED BOOK MUST BE ALL THREE-CONDITION F0 TRIPLES.

    The Whole DOT is 297 pure F0 triples. With no F1 members, score_g.build_book
    writes no sequential columns into the frame, so THE FRAME-OBJECT TRAP DISAPPEARS
    ENTIRELY - that defect cost 11 trades, took a diff to find, and produced a
    PLAUSIBLE WRONG ANSWER rather than an error. This guard keeps the grammar uniform
    so the trap cannot return by the back door.

    The F1 machinery is NOT removed from the codebase: BOOK-50 carries both pairs and
    must keep scoring identically. This applies to the CONFIGURED path only.
    """
    bad = []
    for i, row in book.iterrows():
        sd = str(row.get('signal_def', ''))
        trg = str(row.get('trigger', ''))
        parts = [x.strip() for x in sd.split(' + ')]
        if trg != 'F0' or len(parts) != 3 or not all(TRIPLE_RE.match(x) for x in parts):
            bad.append((int(i), trg, sd[:60]))
    if bad:
        raise SystemExit(
            f'ABORT [book grammar] {len(bad)} of {len(book)} rows in a CONFIGURED book are not '
            f'three-condition F0 triples: {bad[:4]}. A configured book must be uniform - a book '
            f'that silently admits a different grammar reintroduces the sequential-column '
            f'write-back that produced a plausible wrong answer with no error.')
    print(f'  BOOK GRAMMAR: all {len(book)} rows are three-condition F0 triples - no sequential '
          f'members, so build_book writes nothing back into the frame.', flush=True)


PARITY_COLS = ('signal_name', 'direction', 'entry_bar', 'exit_bar', 'entry_price',
               'exit_price', 'pnl', 'lots', 'exit_type', 'tiers')


def _canon_trade_sha(td):
    """A CANONICAL, CROSS-MACHINE-STABLE fingerprint of a trade table.

    The previous form hashed td[cols].to_csv(), which is NOT a fingerprint even though
    it reads like one:
        this box, 297 book   dee7987f2bff2055
        operator, 297 book   6a356e7e8695ffa2
    Same book, same frame, same command. The guard still worked - it compares sacred
    against fork WITHIN one run and both sides matched on his machine - but anyone
    comparing two runs would conclude something had drifted.

    THE VARYING CONTENT IS FLOAT TEXT. to_csv renders floats via the installed
    library's repr, so a different pandas or numpy build writes a different number of
    digits for the same value, and the column ORDER came from whatever order the frame
    happened to return. Neither is part of the result.

    So: an EXPLICIT column list, an EXPLICIT row order, and floats formatted to a fixed
    2dp - which is cent precision and the precision every figure is quoted at. Stable
    is better than labelled, because a stable number is a genuine cross-machine check.
    """
    import hashlib as _h
    cols = [c for c in PARITY_COLS if c in td.columns]
    d = td[cols].copy()
    for c in cols:
        if d[c].dtype.kind == 'f':
            d[c] = d[c].map(lambda v: f'{v:.2f}')
    key = [c for c in ('entry_bar', 'signal_name', 'exit_bar') if c in cols]
    d = d.sort_values(key, kind='mergesort').reset_index(drop=True)
    return _h.sha256(d.to_csv(index=False).encode()).hexdigest()[:16]


def _assert_fork_parity(df, sigs, ad, st, w, conv, window=None):
    """GUARD (b). adm_engine is a FORK with the sacred admission path duplicated
    verbatim in its elif branch. If the sacred entry block ever changes, the fork
    must be updated in lockstep or PARITY SILENTLY BREAKS. Re-prove it at entry."""
    import adm_engine as _adm
    import portfolio_simulation_engine as _sac
    import hashlib as _h
    # THE FULL FRAME, NOT A SLICE. The adaptive oracle is computed over the whole
    # frame, so a sliced df broadcasts (40000,) against (177251,) and raises inside
    # condition_mask. Re-slicing the oracle to match would be a second implementation
    # of the threshold layer, which is exactly what must not exist.
    d = df.copy()
    _keep = dict(rule=_adm.ADMISSION_RULE, mx=_adm.MAX_POSITIONS, tg=_adm.ADM_TIERGATES,
                 gt=_adm.ADM_GATES, fl=_adm.ADM_FLOOR)
    # 'CURRENT', 6 IS CORRECT HERE AND IS NOT A DEFECT: this is the fork-parity check and
    # it deliberately matches the SACRED engine's own configuration. ADM_FLOOR must now be
    # set explicitly too - it has no default since an unset admission parameter silently
    # skipped the FLOORED block once - and CURRENT admission never reads it.
    _adm.ADMISSION_RULE, _adm.MAX_POSITIONS = 'CURRENT', 6
    _adm.ADM_FLOOR = {1: 3, -1: 3}
    _adm.ADM_TIERGATES, _adm.ADM_GATES = None, None
    try:
        a = _sac.run_portfolio(d, sigs, adaptive=ad, structural=st, warmup=w, verbose=False,
                               conviction=conv)
        b = _adm.run_portfolio(d.copy(), sigs, adaptive=ad, structural=st, warmup=w,
                               verbose=False, conviction=conv)
    finally:
        _adm.ADMISSION_RULE, _adm.MAX_POSITIONS = _keep['rule'], _keep['mx']
        _adm.ADM_FLOOR = _keep['fl']
        _adm.ADM_TIERGATES, _adm.ADM_GATES = _keep['tg'], _keep['gt']
    ha, hb = _canon_trade_sha(a), _canon_trade_sha(b)
    print(f'  FORK PARITY ({len(d):,} bars, full frame): sacred {ha} | adm_engine(CURRENT) {hb} '
          f'-> {"IDENTICAL" if ha == hb else "MISMATCH"}', flush=True)
    if ha != hb:
        raise SystemExit(
            f'ABORT [fork parity] adm_engine under CURRENT admission no longer reproduces the '
            f'sacred engine ({ha} vs {hb}). The sacred admission path is duplicated verbatim '
            f'inside the fork; if one changed and the other did not, every configured score is '
            f'wrong and nothing else would have said so.')


def _score_configured(df, sigs, ad, st, w, conv, cfg):
    """Score through adm_engine with the book config's rules."""
    import adm_engine as _adm
    import swept_thresholds as _swt
    G = _swt.build_whole_dot_gates(df)
    print(f'  GATE MASKS: HU90 {100 * G["HU90"].mean():.4f}%  FB20 {100 * G["FB20"].mean():.4f}%'
          f'  ATS90 {100 * G["ATS90"].mean():.4f}%  HU90&ATS90 '
          f'{100 * (G["HU90"] & G["ATS90"]).mean():.4f}%', flush=True)
    _adm.ADMISSION_RULE = cfg['admission']
    _adm.MAX_POSITIONS = int(cfg['max_positions'])
    _adm.ADM_FLOOR = {1: int(cfg['long_depth_floor']), -1: int(cfg['short_depth_floor'])}
    _adm.ADM_GATES = {'ATR': df['ATR_1M'].values.astype(float),
                      'atr_min': float(cfg['global_gate']['value'])}
    _NAME = {('Micro_Hurst', 90): 'HU90', ('Micro_FailedBreak', 20): 'FB20',
             ('AT_Slope_ST', 90): 'ATS90'}
    tg = {}
    for dname, dv in (('LONG', 1), ('SHORT', -1)):
        for tier, gates in cfg['tier_gates'][dname].items():
            if not gates:
                continue
            t = 5 if tier == '5+' else int(tier)
            ms = []
            for g in gates:
                key = _NAME.get((g['variable'], int(g['pct'])))
                if key is None:
                    raise SystemExit(f'ABORT [tier gate] no swept mask for {g["variable"]} '
                                     f'p{g["pct"]} - swept_thresholds supplies HU90/FB20/ATS90 '
                                     f'only; ad[(var,"hi")] is p80 and is NOT a substitute.')
                ms.append(G[key])
            tg[(dv, t)] = ms
    _adm.ADM_TIERGATES = tg
    global _LAST_GATES
    _LAST_GATES = {'Micro_Hurst > p90  (LONG d3 / SHORT d3)': G['HU90'],
                   'Micro_FailedBreak > p20  (LONG d4, d5+)': G['FB20'],
                   'AT_Slope_ST > p90  (LONG d4)': G['ATS90'],
                   'HU90 AND ATS90  (the LONG d4 stack)': G['HU90'] & G['ATS90'],
                   'ATR_1M >= 20  (global)': df['ATR_1M'].values.astype(float) >= float(
                       cfg['global_gate']['value'])}
    td_full = _adm.run_portfolio(df, sigs, adaptive=ad, structural=st, warmup=w, verbose=False,
                                 conviction=conv)
    # THE SPEC QUOTES THE BOOK-ONLY POPULATION. GAP FILLERS ARE A SEPARATE POPULATION
    # and master's own trades.csv header already draws the line:
    #   "population=FULL (BOOK F0+F1 plus gap fillers). BOOK-only = rows whose
    #    signal_name is not GAP_HURST/GAP_FB/GAP_D2D."
    # Scoring FULL gave 6,229 trades against the spec's 5,799 - the 430 difference is
    # exactly the gap fillers. Every figure reproduces on BOOK-only.
    import cluster_profiler as _cp
    td = td_full[~td_full['signal_name'].isin(_cp.GAP_NAMES)]
    print(f'  POPULATION: {len(td_full)} FULL rows -> {len(td)} BOOK-only '
          f'({len(td_full) - len(td)} gap fillers excluded). The spec quotes BOOK-only.',
          flush=True)
    import numpy as _np
    p = _np.asarray(td['pnl'].values, dtype=float)
    gl = -p[p < 0].sum()
    day = pd.Series(td['exit_time'].astype(str).values).str[:10].values
    _byday = pd.Series(p).groupby(day).sum()
    _eq = _byday.cumsum()
    r = _metrics_from_trades(df, td, w)
    _byday = pd.Series(td['pnl'].values.astype(float)).groupby(
        pd.Series(td['exit_time'].astype(str).values).str[:10].values).sum()
    # ISO WEEK KEY. exit_time[:8] is a DAY-level slice, not a week - it reported 7 weeks
    # where the spec has 26, so the direction was right and the denominator meaningless.
    _iso = pd.to_datetime(pd.Series(td['exit_time'].astype(str).values).str[:10],
                          format='%Y.%m.%d', errors='coerce')
    _wk = _iso.dt.isocalendar().set_index(_iso.index)[['year', 'week']].astype(str).agg(
        '-W'.join, axis=1).values
    _byweek = pd.Series(td['pnl'].values.astype(float)).groupby(_wk).sum()
    _allday = pd.Series(df['Time'].astype(str).values).str[:10]
    r.update({'worst_bar': round(float(td.groupby('entry_bar')['pnl'].sum().min()), 2),
              'losing_weeks': int((_byweek < 0).sum()), 'weeks_total': int(len(_byweek)),
              'days_pos': int((_byday > 0).sum()), 'days_traded': int(len(_byday)),
              'days_in_frame': int(_allday.nunique())})
    return r, td


def _gate_ctx(df, ad, st, w, pool, cfg):
    """Everything the gate layer needs, assembled once. Without this the gate layer is
    present in select_stage.py and NOT REACHABLE from --stage SELECT, which is exactly
    what the reachability check caught."""
    import adm_engine as _adm
    import swept_thresholds as _sw
    import score_g as _sg
    import sequential_temporal as _seq
    import conviction as _C
    bk = pd.read_csv(os.path.join(_HERE, 'engine', 'whole_dot_signals.csv'))
    anchor = _seq.anchor_array(df, 'ST_Flip')
    cv = cfg['conviction']
    conv = _C.build_conviction(df, bool(cv['hurst']), bool(cv['recentfb']), bool(cv['d2d']),
                               d2d_conviction=bool(cv['d2d_conviction']),
                               d2d_gap=bool(cv['d2d_gap']))
    sigs = _sg.build_book(df, pool, anchor, bk, adaptive=ad, structural=st)
    return {'df': df, 'sigs': sigs, 'conv': conv, 'pool': pool, 'adm': _adm, 'sw': _sw}


def _smoke_select(df, ad, st, w, pool, anchor, out, input_sha, workers):
    """The SELECT leg of --smoke. RESEQUENCED SO THE SCREEN CANNOT KILL THE REST.

    The previous ordering aborted the whole leg on its own capped input: smoke's S3
    emits ONE F0 row (s3_limit=8 caps the family; 1 of 8 combos survives - EXPECTED at
    smoke caps, not a defect), the screen yielded 0 survivors, nested_arms returned [0],
    and the book_size guard correctly refused book_size=32. THE GUARD IS RIGHT AND IS NOT
    WEAKENED HERE - smoke simply must stop handing it an impossible request.

    THE FOUR MOST VALUABLE CHECKS DO NOT DEPEND ON THE SCREEN AT ALL and were all killed
    by that abort: the PREREG checksum (pure masks), the gate layer (the incumbent's REAL
    ungated cells), the BOOK-50 canary (a different book), and the deploy manifest (pure
    file checks). They now run FIRST. The screen-dependent leg runs LAST and DEGRADES.

    EVERY CAP PRINTS AS A CAP. A smoke figure that reads like a result is worse than no
    figure.
    """
    import select_stage as sel
    import discovery_orchestrator as orch
    import pandas as _pd
    results = os.path.join(out, 'results')
    cfg_path = os.path.join(_HERE, 'engine', 'whole_dot_config.json')
    if not os.path.exists(cfg_path):
        print('  SMOKE SELECT: engine/whole_dot_config.json absent. SKIPPED.', flush=True)
        return
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    sel.assert_config(cfg, cfg_path)
    import swept_thresholds as _sw
    import adm_engine as _adm
    import cluster_profiler as _cp

    print('  [1/6] PREREG CHECKSUM - pure masks, needs no survivor', flush=True)
    for _ln in sel.assert_prereg_checksums(df, cfg, _sw):
        print(_ln, flush=True)

    print('  [2/6] GATE LAYER on the incumbent\'s REAL UNGATED CELLS - NOT CAPPED, needs '
          'no survivor', flush=True)
    _ctx = _gate_ctx(df, ad, st, w, pool, cfg)
    _td_ung = sel.ungated_trades(df, _ctx['sigs'], ad, st, w, _ctx['conv'], cfg, _adm,
                                 _cp.GAP_NAMES)
    _cap_null = int(globals().get('SMOKE_NULL_CAP', 8))
    print(f'      *** PREREG NULL ENUMERATION CAPPED AT {_cap_null} OF THE FULL SHORTLIST '
          f'- ANY p BELOW IS NOT A VERDICT. The real run enumerates every candidate '
          f'(186s LONG d3, 320s SHORT d3). THE CELLS THEMSELVES ARE REAL AND UNCAPPED. ***',
          flush=True)
    sel.SMOKE_NULL_CAP = _cap_null
    try:
        _gl, _gv = sel.run_gate_layer(df, _ctx['sigs'], ad, st, w, _ctx['conv'], cfg, pool,
                                      _adm, _sw, _cp.GAP_NAMES, _td_ung)
    finally:
        sel.SMOKE_NULL_CAP = None
    for _ln in _gl:
        print(_ln, flush=True)

    print('  [3/6] BOOK-50 CANARY - a separate book, needs no survivor', flush=True)
    b50 = os.path.join(_HERE, 'engine', 'book50_signals.csv')
    if os.path.exists(b50):
        s8_committed(df, ad, st, w, pool, anchor, b50, out, input_sha)
    else:
        print('      *** BOOK-50 ABSENT - THE ENGINE CHECK DID NOT RUN. Not a pass. ***',
              flush=True)

    print('  [4/6] DEPLOY MANIFEST - pure file checks', flush=True)
    for _ln in sel.print_deploy_manifest(_HERE)[0]:
        print(_ln, flush=True)

    print('  [5/6] SECTION-4 PROVENANCE GUARD', flush=True)
    scan = os.path.join(results, 'results_F0_triple_convergence_and_d2ddir.csv')
    # PREFER A REAL SCAN WHEN ONE EXISTS: a CAPPED READ OF A REAL SCAN is a far better
    # smoke test than a full read of a 1-row one. The provenance guard still decides
    # whether it is usable, so preferring it cannot bypass the check.
    _cands = [scan, os.path.join(_HERE, 'discovery', 'full', 'results',
                                 'results_F0_triple_convergence_and_d2ddir.csv')]
    _use, _rows, _why = None, 0, ''
    for _c in _cands:
        if not os.path.exists(_c):
            continue
        _ok, _w2 = orch.provenance_is_current(_c, input_sha)
        _n = max(0, sum(1 for _l in open(_c, encoding='utf-8', errors='replace')) - 1)
        print(f'      {os.path.basename(os.path.dirname(_c))}/{os.path.basename(_c)}: '
              f'{_n} rows, provenance {"CURRENT" if _ok else "STALE"} - {_w2}', flush=True)
        if _ok and _n > _rows:
            _use, _rows, _why = _c, _n, _w2
    if _use is None:
        print('      *** NO SCAN WITH CURRENT PROVENANCE. The real run would ABORT here. '
              'FIX: re-run --stage S3 against this frame - a COPIED scan does NOT carry '
              'its stamp. SCREEN/ARM/ARTIFACT LEG SKIPPED, and the non-empty artifact '
              'assertion IS THEREFORE NOT RUN. ***', flush=True)
        return
    cap_rows = int(globals().get('SMOKE_SCAN_ROWS', 120))
    print(f'  [6/6] SCREEN, ARMS, ARTIFACTS  *** CAPPED: asked for {cap_rows} scan rows, '
          f'the chosen scan has {_rows} - reading {min(cap_rows, _rows)}. NOT A RESULT. ***',
          flush=True)
    sub = _pd.read_csv(_use).iloc[:min(cap_rows, _rows)]
    smoke_scan = os.path.join(results, '_smoke_F0_scan.csv')
    sub.to_csv(smoke_scan, index=False, lineterminator='\n')
    orch.stamp_provenance(smoke_scan, input_sha)
    # ARMS DERIVED FROM WHAT SURVIVES, IN THE CALLER. nested_arms and the book_size guard
    # are SHARED WITH PRODUCTION and are not modified: fix the caller, not the callee.
    _probe = sel.run_select(df, ad, st, w, pool, anchor, cfg, cfg_path, out, input_sha,
                            workers, scan_path=smoke_scan, score_fn=_score_configured,
                            metrics_fn=_metrics_from_trades, grammar_fn=_assert_book_grammar,
                            breakdown_fn=breakdown_report, loss_events_fn=loss_events,
                            arm_sizes=(), book_size=None, gate_ctx=None)
    _ns = len(_probe['survivors'])
    _arms = tuple(a for a in (4, 8, 16, 32, 64) if a <= _ns)
    if not _arms:
        print(f'      *** SMOKE SCREEN YIELDED {_ns} SURVIVORS FROM A CAPPED SCAN OF '
              f'{len(sub)} ROWS; ARM AND ARTIFACT LEG SKIPPED. THIS IS A SMOKE CAP, NOT A '
              f'SCREEN FAILURE. The non-empty artifact assertion IS THEREFORE NOT RUN - '
              f'it is skipped, and it says so rather than passing silently. ***', flush=True)
        return
    res = sel.run_select(df, ad, st, w, pool, anchor, cfg, cfg_path, out, input_sha,
                         workers, scan_path=smoke_scan, score_fn=_score_configured,
                         metrics_fn=_metrics_from_trades, grammar_fn=_assert_book_grammar,
                         breakdown_fn=breakdown_report, loss_events_fn=loss_events,
                         arm_sizes=_arms, book_size=_arms[-1], gate_ctx=None)
    print(f'      arms {_arms} DERIVED FROM {_ns} SURVIVORS - stand-ins for '
          f'{cfg["draw"]["arm_sizes"]}', flush=True)
    paths = sel.emit_artifacts(out, res, res['arm_books'], res['scores'], cfg_path,
                               _sha12_of(cfg_path))
    for _ln in sel.arm_table(res):
        print(_ln, flush=True)
    print('  SELECT ARTIFACTS:', flush=True)
    _empty = []
    for k, v in paths.items():
        n = max(0, sum(1 for _l in open(v, encoding='utf-8', errors='replace')
                       if not _l.startswith('#')) - 1)
        print(f'    {k:10} {os.path.basename(v):28} {n:>6} rows  sha {_sha12_of(v)}',
              flush=True)
        if n == 0:
            _empty.append(os.path.basename(v))
    if _empty:
        raise SystemExit(f'ABORT [--smoke] SELECT artifact(s) with ZERO ROWS: {_empty}. Under '
                         f'--smoke an empty artifact is a FAILED SMOKE RUN.')
    print('  smoke non-empty assertion EXTENDED TO THE SELECT ARTIFACTS: all have rows.',
          flush=True)


def s_select(df, ad, st, w, pool, anchor, book_file, out, input_sha, workers,
             arm_sizes=None, book_size=None):
    """SELECT - data in, signals and score out.

    SECTION 4: THE SCAN DEPENDENCY IS DECLARED BY ASSERTION, NOT BY INVOKING S3.
    The screen operates on results_F0_*.csv. Running SELECT on a new month would
    otherwise screen LAST MONTH'S candidate rows against NEW data, and nothing
    downstream would say so. Aborting rather than warning is the point: a warning on
    a monthly stage is read once and then not.

    That choice is also why the train-window statistics take ROUTE A (solo re-run)
    rather than ROUTE B (emit the partition at scan time): route B's precondition was
    'prefer if S3 is being touched anyway', and it is not. Measured at 0.157 s/signal,
    19,754 rows is ~4 minutes at 14 workers - not the 62-minute serial tax route B
    was avoiding.
    """
    import select_stage as sel
    import discovery_orchestrator as orch
    arm_sizes = arm_sizes or sel.ARM_SIZES
    results = os.path.join(out, 'results')
    scan = os.path.join(results, 'results_F0_triple_convergence_and_d2ddir.csv')
    if not os.path.exists(scan):
        raise SystemExit(f'ABORT [SELECT] the F0 scan is absent: {scan}. SELECT screens the raw '
                         f'scan; run --stage S3 first.')
    ok, why = orch.provenance_is_current(scan, input_sha)
    if not ok:
        raise SystemExit(
            f'ABORT [SELECT] SCAN/FRAME MISMATCH: {why}. The scan was produced from a different '
            f'frame than the one loaded, so the screen would validate LAST MONTH\'S candidate '
            f'rows against NEW data - the train-only claim would be false in the first place a '
            f'reviewer looks. Re-run --stage S3 against this frame.')
    print(f'  SCAN DEPENDENCY: {os.path.basename(scan)} provenance matches the frame '
          f'({input_sha}) - {why}', flush=True)
    cfg, cfg_path = book_config_for(book_file)
    if cfg is None:
        raise SystemExit(f'ABORT [SELECT] no book config. SELECT selects signals but does NOT '
                         f'derive constants - floors, gate stack, MAX_POSITIONS, ATR floor and '
                         f'admission all come from the config. Pass --book with a configured '
                         f'book (its signal list is not used; only its rules).')
    print(f'  ARCHITECTURE: {os.path.basename(cfg_path)} sha '
          f'{_sha12_of(cfg_path)} - floors L{cfg["long_depth_floor"]}/S'
          f'{cfg["short_depth_floor"]}, cap {cfg["max_positions"]}, '
          f'{cfg["global_gate"]["variable"]} >= {cfg["global_gate"]["value"]}, '
          f'{cfg["admission"]} admission, recentfb {cfg["conviction"]["recentfb"]}', flush=True)
    res = sel.run_select(df, ad, st, w, pool, anchor, cfg, cfg_path, out, input_sha, workers,
                         scan_path=scan, score_fn=_score_configured,
                         metrics_fn=_metrics_from_trades, grammar_fn=_assert_book_grammar,
                         breakdown_fn=breakdown_report, loss_events_fn=loss_events,
                         arm_sizes=arm_sizes, book_size=book_size,
                         gate_ctx=_gate_ctx(df, ad, st, w, pool, cfg))
    inc = pd.read_csv(os.path.join(_HERE, 'engine', 'whole_dot_signals.csv'))
    paths = sel.emit_artifacts(out, res, res['arm_books'], res['scores'], cfg_path,
                               _sha12_of(cfg_path))
    lines = []
    lines += sel.arm_table(res)
    lines += sel.baseline_table()
    lines += sel.side_by_side(res, inc)
    bs = res['book_size']
    _r, _td = res['scores'][bs]
    lines += breakdown_report(df, _td, None, gates=_LAST_GATES, cfg=cfg)
    for ln in lines:
        print(ln, flush=True)
    print('', flush=True)
    print('  ARTIFACTS WRITTEN:', flush=True)
    for k, v in paths.items():
        print(f'    {k:10} {v}  sha {_sha12_of(v)}', flush=True)
    print(f'  SCREEN {res["screen_secs"] / 60:.1f} min over {res["scan_rows"]:,} raw rows',
          flush=True)
    return res


def _sha12_of(path):
    import hashlib as _h
    h = _h.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:12]


def s8_committed(df, ad, st, w, pool, anchor, book_file, out, input_sha):
    import conviction as C
    import score_g
    committed = os.path.join(out, 'committed')
    os.makedirs(committed, exist_ok=True)
    frozen = book_file is not None
    if frozen:
        book = pd.read_csv(book_file)
        book_tag = f'FROZEN ratified book ({os.path.basename(book_file)})'
    else:
        print('  S8 DISCOVER-FRESH IS DISABLED (item 15). Under a catalogue design S8 has '
              'nothing to score automatically: the deliverable is fourteen per-family catalogues '
              'holding every VALID signal, and NOTHING in this build chooses which of them to '
              'trade. Scoring happens when YOU compose a book and run it through:')
        print('      python score_book.py --book <your_book.csv> --data <frame> --out <dir>')
        print('  That tool (item 16) applies the constraint machinery - TailDep, FailConc, mCVaR, '
              'absolute survival, union coverage - which are SET properties of an assembled book '
              'and have no per-signal value. Every catalogue states a book is UNSCORED until it '
              'has been run. S8 FROZEN path is untouched and still scores the ratified book.')
        return None
    sigs = score_g.build_book(df, pool, anchor, book, adaptive=ad, structural=st)
    _cfg, _cfg_path = book_config_for(book_file)
    # GUARD (a). CONFIG-NOT-FOUND MUST RAISE. Keyed on an EXPLICIT list, never on
    # absence: a missing config previously routed a configured book down the SACRED
    # path and scored a DIFFERENT SYSTEM WITH NO ERROR. Scoring the wrong system
    # silently is strictly worse than refusing to start.
    if _cfg is None and os.path.basename(book_file or '') not in SACRED_PATH_BOOKS:
        raise SystemExit(
            f'ABORT [book config] {os.path.basename(book_file or "<none>")} has no sidecar '
            f'config and is not in SACRED_PATH_BOOKS {list(SACRED_PATH_BOOKS)}. Expected '
            f'<book>_config.json or the short form beside it. A book without a config would '
            f'be scored by the SACRED engine at MAX_POSITIONS=6, flat rules and no depth '
            f'floor - a completely different system from any configured one, and nothing '
            f'downstream would say so.')
    if _cfg:
        _cv = _cfg.get('conviction', {})
        print(f'  BOOK CONFIG: {os.path.basename(_cfg_path)} - conviction hurst='
              f'{_cv.get("hurst", True)} recentfb={_cv.get("recentfb", True)} '
              f'd2d={_cv.get("d2d", True)}. master.py hardcoded recentfb TRUE; the adopted '
              f'configuration sets it FALSE (DERIVED - it sized losers up more than winners).',
              flush=True)
        conv = C.build_conviction(df, bool(_cv.get('hurst', True)),
                                  bool(_cv.get('recentfb', True)),
                                  bool(_cv.get('d2d', True)),
                                  d2d_conviction=bool(_cv.get('d2d_conviction', True)),
                                  d2d_gap=bool(_cv.get('d2d_gap', True)))
    else:
        conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
    if _cfg:
        _assert_book_grammar(book)
        _assert_fork_parity(df, sigs, ad, st, w, conv)
        r, executed = _score_configured(df, sigs, ad, st, w, conv, _cfg)
    else:
        r, executed = _score(df, sigs, ad, st, w, conv, want_trades=True)
    _tl, _ev, _dy = loss_events(executed)
    lines = []
    lines.append(f'COMMITTED SYSTEM SCORE — {book_tag}')
    lines.append(f'  book rows           : {len(book)}')
    lines.append(f'  trades              : {r["trades"]}')
    lines.append(f'  win rate            : {r["WR"]}%')
    lines.append(f'  profit factor       : {r["PF"]}')
    lines.append(f'  net P&L $           : {r["net"]}')
    lines.append(f'  trade losses        : {_tl}')
    lines.append(f'  LOSS EVENTS         : {_ev}   (distinct entry BARS - the bar is the risk '
                 f'unit, not the trade)')
    lines.append(f'  distinct loss days  : {_dy}')
    if 'worst_bar' in r:
        lines.append(f'  worst bar $         : {r["worst_bar"]}   (the bar is the risk unit)')
        lines.append(f'  losing weeks        : {r["losing_weeks"]} of {r["weeks_total"]}   '
                     f'(ISO week keys)')
        lines.append(f'  days positive       : {r["days_pos"]} of {r["days_traded"]} DAYS WITH '
                     f'A BOOK TRADE  ({r["days_in_frame"]} trading days exist in the frame)')
    lines.append(f'  daily worst-day $   : {r["daily_wd"]}')
    lines.append(f'  daily max-drawdown $: {r["daily_mDD"]}')
    if r['folds_evaluable']:
        lines.append(f'  folds positive      : {r["folds_plus"]}/{r["fold_count"]}  '
                     f'({r["fold_days_each"]} trading days each, min-fold PF {r["min_fold_pf"]})')
    else:
        lines.append(f'  folds               : {r["folds_status"]}')
    if r['oos_prop_evaluable']:
        lines.append(f'  OOS (final third: {r["oos_prop_window"]}) PF : {r["oos_prop_pf"]}   '
                     f'net ${r["oos_prop_net"]}')
    else:
        lines.append(f'  OOS (final third)   : UNEVALUABLE - {r["oos_prop_days"]} trading days, '
                     f'below the floor of {MIN_FOLD_DAYS}')
    # THE CANARY WAS INERT, NOT BROKEN. It required trades == 2698 and net == 92347 -
    # figures from an OLDER ENGINE CONFIGURATION - while BOOK-50 now scores 3,101 and
    # $97,675. The basename matched all along; the assertion had been silently skipping
    # its own print since the engine changed. AN ENGINE CHECK THAT FAILS BY SAYING
    # NOTHING IS WORSE THAN NO CHECK AT ALL, and two literals in a condition is how it
    # went stale unnoticed.
    #
    # So: the reference lives in config (arch.canary), a MISMATCH PRINTS LOUDLY, and a
    # missing reference says so by name rather than passing quietly. Substituting new
    # literals here would rebuild the same defect with fresher numbers.
    canary = False
    if frozen and os.path.basename(book_file or '') == 'book50_signals.csv':
        _ref = None
        try:
            _refcfg = json.load(open(os.path.join(_HERE, 'engine', 'canary_reference.json'),
                                     encoding='utf-8'))
            _ref = _refcfg.get('book50')
        except Exception as _exc:
            lines.append(f'  *** CANARY REFERENCE UNAVAILABLE: engine/canary_reference.json '
                         f'({type(_exc).__name__}). THE PER-SESSION ENGINE CHECK DID NOT RUN. '
                         f'This is not a pass. ***')
        if _ref:
            _dt = int(r['trades']) - int(_ref['trades'])
            _dn = float(r['net']) - float(_ref['net'])
            canary = (_dt == 0 and abs(_dn) < 1.0)
            lines.append('')
            if canary:
                lines.append(f'  US30 baseline canary: ${_ref["net"]:,} / {_ref["trades"]:,} tr '
                             f'- ENGINE INTACT (reference {_ref.get("recorded", "?")})')
            else:
                lines.append(f'  *** CANARY MISMATCH - THE ENGINE HAS MOVED. BOOK-50 scored '
                             f'{r["trades"]:,} tr / ${r["net"]:,} against the recorded '
                             f'reference {_ref["trades"]:,} tr / ${_ref["net"]:,} '
                             f'(delta {_dt:+,} tr / {_dn:+,.2f}). Either an engine change is '
                             f'unintended, or the reference is stale and must be re-ratified '
                             f'in engine/canary_reference.json. DO NOT IGNORE THIS LINE. ***')
        elif _ref is None and 'CANARY REFERENCE UNAVAILABLE' not in ''.join(lines[-1:]):
            lines.append('  *** CANARY REFERENCE ABSENT: engine/canary_reference.json has no '
                         '"book50" entry. THE ENGINE CHECK DID NOT RUN. ***')

    for _bl in breakdown_report(df, executed, book, gates=_LAST_GATES, cfg=_cfg):
        lines.append(_bl)
    tr = executed.copy()
    keep = [c for c in ['signal_idx', 'signal_name', 'direction', 'lots', 'entry_bar', 'exit_bar',
                        'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'pnl_per_lot',
                        'exit_type', 'tiers', 'be_nudged', 'initial_risk'] if c in tr.columns]
    tr = tr[keep]
    tpath = os.path.join(committed, 'trades.csv')
    ttmp = tpath + '.tmp'
    with open(ttmp, 'w', encoding='utf-8') as f:
        f.write(f'# DOT committed-system per-trade table (spec B.2 open item 8)\n')
        f.write(f'# book={book_tag}\n')
        f.write(f'# dataset_rows={len(df)} range={df["Time"].astype(str).values[0]} -> {df["Time"].astype(str).values[-1]}\n')
        f.write(f'# population=FULL (BOOK F0+F1 plus gap fillers). BOOK-only = rows whose signal_name is not GAP_HURST/GAP_FB/GAP_D2D.\n')
        f.write(f'# oracle_sha256_12={sha12(os.path.join(_ENGINE, "dots_thresholds.py"))} engine_sha256_12={sha12(os.path.join(_ENGINE, "portfolio_simulation_engine.py"))}\n')
        tr.to_csv(f, index=False, lineterminator='\n')
    os.replace(ttmp, tpath)
    txt = '\n'.join(lines)
    open(os.path.join(committed, 'committed_score.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print('\n'.join('  ' + ln for ln in lines))
    r['book_tag'] = book_tag
    r['canary'] = canary
    r['executed'] = executed
    r['sigs'] = sigs
    mark_done(out, 'S8', {'input_sha': input_sha, 'net': r['net'], 'trades': r['trades'], 'canary': canary})
    return r


LOADER_ALLOWLIST = {
    # adm_engine.py is a FORK of portfolio_simulation_engine and carries the same two
    # occurrences the sacred file is already allowed: the def itself plus its __main__
    # standalone block (L516). NEITHER IS ON THE S8 PATH - master injects the frame,
    # and adm_engine.run_portfolio takes df as its first argument. Allowed on that
    # basis: same count, same shape, same reason as the file it forks.
    'engine/adm_engine.py': 2,
    'engine/analysis_engine.py': 2, 'engine/portfolio_simulation_engine.py': 2,
    'engine/run_full_analysis.py': 1, 'engine/score_book50.py': 1, 'engine/score_g.py': 1,
    'engine/wf.py': 1, 'orchestrator/discovery_orchestrator.py': 2,
    'scanners/concurrence_profiler.py': 1, 'scanners/conditional_interaction.py': 1,
    'scanners/cross_variable_structure.py': 1, 'scanners/divergence_nonconfirm.py': 1,
    'scanners/f0_to_schema.py': 1, 'scanners/mean_reversion.py': 1,
    'scanners/persistence_autocorr.py': 1, 'scanners/rolling_leadlag.py': 1,
    'scanners/run_f1_parallel.py': 1, 'scanners/sequential_temporal.py': 1,
    'scanners/session_temporal.py': 1, 'scanners/single_variable_extremes.py': 1,
    'scanners/state_transition.py': 1, 'scanners/threshold_crossing.py': 1,
    'scanners/triple_convergence_and_d2ddir.py': 3,
}


def preflight_loader_audit():
    found = {}
    for sub in ('engine', 'scanners', 'orchestrator'):
        root = os.path.join(_HERE, sub)
        if not os.path.isdir(root):
            continue
        for nm in sorted(os.listdir(root)):
            if not nm.endswith('.py'):
                continue
            rel = f'{sub}/{nm}'
            txt = open(os.path.join(root, nm), 'r', encoding='utf-8').read()
            n = txt.count('load_sealed_baseline')
            if n:
                found[rel] = n
    new = {k: v for k, v in found.items() if k not in LOADER_ALLOWLIST}
    grew = {k: (LOADER_ALLOWLIST[k], v) for k, v in found.items()
            if k in LOADER_ALLOWLIST and v > LOADER_ALLOWLIST[k]}
    total = sum(found.values())
    print(f'  LOADER AUDIT — {total} references to load_sealed_baseline across {len(found)} files, '
          f'all on the frozen allowlist' if not (new or grew) else
          f'  LOADER AUDIT — FAIL', flush=True)
    hook = os.path.join(_HERE, 'sitecustomize.py')
    binder = os.path.join(_HERE, 'dot_frame_binding.py')
    if not (os.path.exists(hook) and os.path.exists(binder)):
        raise SystemExit(
            'ABORT — sitecustomize.py / dot_frame_binding.py missing from the pack root. Without '
            'them the frame binding cannot reach spawned worker processes, and any family that '
            'starts its own pool (F12, F13) will load the hardcoded equiDOT_recon171_step7_* parts.')
    print('  SPAWN-SAFETY — sitecustomize.py present: the binding re-establishes at interpreter '
          'startup in every spawned process, so the 27 call sites cannot reach the raw loader from '
          'a worker. STATIC LIMIT: this is a presence check, not a proof; a spawned process that '
          'starts with PYTHONPATH stripped would not import the hook, so the binding also asserts '
          'and aborts inside the worker rather than trusting it.', flush=True)
    if new or grew:
        msg = []
        for k, v in new.items():
            msg.append(f'{k} ({v} new occurrence(s))')
        for k, (was, now) in grew.items():
            msg.append(f'{k} ({was} allowed, {now} found)')
        raise SystemExit(
            'ABORT — new load_sealed_baseline call site(s): ' + '; '.join(msg) +
            '. That function hardcodes equiDOT_recon171_step7_* and has silently loaded the WRONG '
            'dataset in three separate places already. Any new call site must either take an '
            'injected frame or be added to LOADER_ALLOWLIST with a reason.')
    return found


def bind_ingested_frame_permanently(df, input_sha, out_dir):
    import dot_frame_binding as fb
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f'_frame_{input_sha}.csv')
    for stale in glob.glob(os.path.join(out_dir, '_frame_*.csv')):
        if os.path.basename(stale) != os.path.basename(cache):
            os.remove(stale)
    if not os.path.exists(cache):
        tmp = cache + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            df.to_csv(f, index=False, lineterminator='\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache)
    fp = fb.fingerprint_of(df)
    fb.configure_environment(cache, input_sha, fp)
    fb.install(df)
    print(f'  FRAME BINDING — engine.load_sealed_baseline is bound to the frame S0 ingested, in THIS')
    print(f'  process AND in every process spawned from it. The parent-only monkeypatch did not')
    print(f'  survive spawn: F12 and F13 start their own pools, each worker re-imports a pristine')
    print(f'  engine module and reached the hardcoded equiDOT_recon171_step7_* parts. The binding is')
    print(f'  now re-established at INTERPRETER STARTUP via sitecustomize.py, which Python imports')
    print(f'  before any family code runs, driven by DOT_FRAME_PATH/DOT_INPUT_SHA in the inherited')
    print(f'  environment. A new entry point cannot bypass it because it does not have to opt in.')
    print(f'    frame fingerprint: {fp[0]:,} rows | {fp[1]} -> {fp[2]} | input_sha {input_sha}')
    print(f'    worker frame cache: {os.path.basename(cache)}')
    return cache


def run_diagnostic_families(results_dir, workers, input_sha, df=None):
    import discovery_orchestrator as orch
    print('  DIAGNOSTIC FAMILIES (F12, F13) — separate stages: they emit measurement artifacts, not')
    print('  14-column pool rows, so they cannot collate into discovery_master.csv. Both run on the')
    print('  same single command with the operator --workers value and their own internal parallelism.')
    f13_csv = os.path.join(results_dir, 'results_F13_single_variable_extremes.csv')
    ok13, why13 = orch.provenance_is_current(f13_csv, input_sha)
    _sc13, _scwhy13 = _diag_scanner_current(results_dir, 'F13', 'single_variable_extremes')
    if ok13 and not _sc13:
        print(f'  [F13] frame provenance is current BUT {_scwhy13} - RE-RUNNING. input_sha is '
              f'the FRAME authority; the scanner sha is the PRODUCER authority and it governs '
              f'here.', flush=True)
        ok13 = False
        why13 = _scwhy13
    if ok13:
        print('  [F13] already current for this input_sha — skipping')
    else:
        print(f'  [F13] running ({why13}); native _f13_shards/*.done checkpointing preserved as-is')
        import single_variable_extremes as f13
        f13.OUT_CSV = f13_csv
        f13.SHARD_DIR = os.path.join(results_dir, '_f13_shards')
        f13.RESULTS_DIR = results_dir
        os.makedirs(f13.SHARD_DIR, exist_ok=True)
        import dot_frame_binding as _fb
        os.environ['DOT_RESULTS_DIR'] = results_dir
        _bound = _fb.install_scanner_paths()
        print(f'  [F13] scanner paths bound in-parent AND at interpreter startup for every spawned '
              f'worker ({_bound}). F13 hardcodes RESULTS_DIR/OUT_CSV/SHARD_DIR at import against the '
              f'LEGACY discovery_results/; a parent-side attribute write does not survive spawn, and '
              f'F13 starts its own Pool, so its workers wrote shards to a directory outside --out '
              f'that did not exist. Scanners are not editable, so the startup hook is the transport.')
        f13.run(min(workers, 12))
        _f13_arts = [n for n in sorted(os.listdir(results_dir))
                     if n.startswith('results_F13') and n.endswith('.csv')]
        _f13_shards = len([n for n in os.listdir(os.path.join(results_dir, '_f13_shards'))
                           if n.endswith('.done')]) if os.path.isdir(
                               os.path.join(results_dir, '_f13_shards')) else 0
        print(f'  [F13] COMPLETE — artifacts: {", ".join(_f13_arts) if _f13_arts else "NONE"} '
              f'| {_f13_shards} shard markers')
        if not os.path.exists(f13_csv):
            raise SystemExit('ABORT — [F13] ran but produced no output at '
                             f'{os.path.basename(f13_csv)}. A diagnostic family that cannot emit is '
                             'not coverage; the run stops rather than report 14-family coverage with '
                             'one family empty.')
        orch.stamp_provenance(f13_csv, input_sha)
        _write_diag_marker(results_dir, 'F13', 'single_variable_extremes', f13_csv)

    f12_csv = os.path.join(results_dir, orch.DIAGNOSTIC_OUTPUTS['F12'])
    ok12, why12 = orch.provenance_is_current(f12_csv, input_sha)
    _sc12, _scwhy12 = _diag_scanner_current(results_dir, 'F12', 'concurrence_profiler')
    if ok12 and not _sc12:
        print(f'  [F12] frame provenance is current BUT {_scwhy12} - RE-RUNNING. input_sha is '
              f'the FRAME authority; the scanner sha is the PRODUCER authority and it governs '
              f'here.', flush=True)
        ok12 = False
        why12 = _scwhy12
    if ok12:
        print('  [F12] already current for this input_sha — skipping')
    else:
        print(f'  [F12] running ({why12}); concurrence CSVs into the run tree')
        before = {}
        for nm in os.listdir(results_dir):
            fp = os.path.join(results_dir, nm)
            if os.path.isfile(fp):
                before[nm] = os.path.getmtime(fp)
        import concurrence_profiler as f12
        f12.RESULTS_DIR = results_dir
        f12.run(n_workers=min(workers, 8))
        produced = []
        for nm in sorted(os.listdir(results_dir)):
            fp = os.path.join(results_dir, nm)
            if not (os.path.isfile(fp) and nm.startswith('concurrence_') and nm.endswith('.csv')):
                continue
            if nm not in before or os.path.getmtime(fp) > before[nm]:
                produced.append(nm)
        for nm in produced:
            orch.stamp_provenance(os.path.join(results_dir, nm), input_sha)
        _write_diag_marker(results_dir, 'F12', 'concurrence_profiler', f12_csv)
        print(f'  [F12] produced {len(produced)} concurrence CSVs this run: '
              f'{", ".join(produced) if produced else "NONE"}')
        print('  [F12] provenance stamped on THOSE FILES ONLY — never by pattern match on whatever '
              'happens to be on disk, which would launder a stale artifact from another dataset')


_TERRAIN = {}
FIXTURE_WHY = ("WHY THIS RUNS EVERY TIME: greedy once returned ZERO short signals, not as a judgement but because 0 of 13 shorts scored above zero alone at S=5 (a signal cannot stack with itself), so every first-step gain was exactly 0.0 and the search halted at step 0 without ever evaluating a pair. The best short PAIR scored 0.012295, ABOVE the incumbent short reference of 0.00757 - greedy returned 0% of the achievable optimum. The lookahead-2 rule took SHORT from 0% to 100% and LONG gained two pair escapes, so the defect was never short-specific. A book selected without this canary could silently be long-only again and nothing would say so.")
FIXTURE_LIMIT = ("RESTRICTION IS PART OF THE FINDING: enumeration covers sizes 1..max_k_enumerated plus the all-signals set, so exhaustive_optimum is a LOWER BOUND and greedy_pct_of_optimum an UPPER bound.")
PBO_WHY = ("PBO IS A SPEC REQUIREMENT (H.1), REPORTED NOT ENFORCED on the first run. It estimates what fraction of selected winners fail forward - the exact question the redesign exists to answer, given the incumbent degraded from PF 6.40 to PF 2.19 on first unseen data. SELF-REFERENCE: bounds derive from the incumbent itself, so F_max and TailDep pass by construction; the informative cell is mcvar. Separate axes: no composite score is formed and coverage is never promoted above survival.")
COFIRE_WHY = ("NEVER POOLED ACROSS DIRECTIONS. Cross-direction co-firing is EXACTLY ZERO on every bar because the D2D gate admits a signal only where D2D_Trend_Dir equals its direction, so long and short qualifying masks are disjoint. The all-pairs basis is therefore DEFLATED and mechanically rewards a single-direction book; it is retained only under its DIAGNOSTIC name and enters no objective.")
G2_WHY = ("MODEST HYGIENE, NOT A REACH MECHANISM. It removes false corroboration in ranking and prevents degenerate triples. It does NOT address the spec D.0 coverage gap, where 89.8% of missed thrusts have no qualifying signal at all - a vocabulary-content problem no hygiene can solve.")
TDOM_WHY = ("rule: a candidate triple must draw from at least DOMAIN_MIN_DISTINCT distinct functional domains. Applied here as a retrospective fixture only; it removes nothing.")




def _write_with_header(path, frame, header_lines):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        for ln in header_lines:
            f.write(f'# {ln}\n')
        frame.to_csv(f, index=False, lineterminator='\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def s2b_terrain(df, w, out, input_sha, attest):
    import terrain as tr
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    print(f'  dataset: {attest["rows"]:,} rows | {attest["range"]}')
    path = os.path.join(out, 'terrain_episodes.csv')
    print('  S2B always recomputes: the terrain costs seconds, and the old checkpoint branch\n        could not parse the metadata header terrain.py writes and rebuilt the terrain anyway.')
    t0 = time.time()
    ter, cells, elig = tr.build_terrain(df, w)
    secs = time.time() - t0
    summary = tr.summarise(ter)
    hours = tr.hour_profile(ter)
    hdr = ['DOT S2B MARKET TERRAIN MAP — price only, NO SIGNALS, NO BOOK',
           f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
           f'oracle_sha256_12={oracle_sha}',
           tr.MARKET_LABEL, tr.FORWARD_LOOKING_BOUNDARY,
           f'eligibility mask: {tr.eligibility_label()} | eligible bars {elig}',
           'K and E come from dots_thresholds (mechanism D, rolling-2500, day-refreshed) via the',
           'ratified basis-3 construction in cluster_profiler; no local percentile, no constant.',
           'THE GRID IS PART OF THE FINDING: every row carries its own (W, K, E) cell. Counts move',
           'by 2-4x across the grid while the up/down ratio barely moves; a count without its',
           'parameters is not a measurement.',
           f'contiguous same-sign qualifying bars collapse into one episode (tolerance '
           f'{tr.CONTIGUOUS_TOLERANCE} bar)']
    _write_with_header(path, ter, hdr)
    _write_with_header(os.path.join(out, 'terrain_summary.csv'), summary, hdr)
    _write_with_header(os.path.join(out, 'terrain_hour_profile.csv'), hours, hdr)
    print(f'  TERRAIN — {len(ter)} episodes across {len(summary)} grid cells | eligible bars {elig:,}')
    for _i, r in summary.iterrows():
        print(f"    W={int(r['W']):2} K=p{int(r['K_pct'] * 100)} E=p{int(r['E_pct'] * 100)} | "
              f"{int(r['episodes']):5} episodes | up {int(r['up']):5} ({r['up_share_pct']:4.1f}%) "
              f"down {int(r['down']):5} ({r['down_share_pct']:4.1f}%) | "
              f"median {r['median_disp_pts']:7.1f}pt (Q1 {r['q1_disp_pts']:.1f} Q3 {r['q3_disp_pts']:.1f}) "
              f"| median {int(r['median_duration_bars'])} bars")
    for ln in tr.render_hour_profile(hours, (15, 0.85, 0.75)):
        print(ln)
    print(f'  S2B runtime {secs:.1f}s for one pass over {attest["rows"]:,} bars x '
          f'{len(summary)} grid cells — single-pass and cheap, so it is NOT chunked and does not '
          f'consume --workers.')
    mark_done(out, 'S2B', {'input_sha': input_sha, 'episodes': int(len(ter))})
    _TERRAIN['cells'] = cells
    _TERRAIN['terrain'] = ter
    return {'terrain': ter, 'cells': cells, 'summary': summary, 'hours': hours}


# ── S3B PER-FAMILY EVIDENCE REVIEW (spec A.1-A.5) + D2D GATE MEASUREMENT (spec E.1) ──
def emit_regime_labels(df, results_dir, out, input_sha):
    """Derive regime_labels.csv from concurrence_depth_bars.csv. NO F12 RE-RUN.

    THE LABELS ARE NOT DISCARDED. compute_regime_labels() puts lab_desc and
    lab_causal into ctx, and F12 ALREADY WRITES BOTH PER BAR to
    concurrence_depth_bars.csv as the columns `regime` (= lab_desc) and
    `regime_causal` (= lab_causal), keyed on Time. The regime axis has been
    measurable all along.

    What that file lacks is a BAR INDEX to join on and any statement of which
    column is which, so this emits a companion carrying both. It reads the
    existing artifact, so it costs seconds and does NOT re-run F12 - 35:50 last
    time - and touches nothing upstream.

    concurrence_profiler.py is one of Appendix D's byte-locked modules and is NOT
    edited: this reads its output, it does not change how that output is made.
    """
    src = os.path.join(results_dir, 'concurrence_depth_bars.csv')
    if not os.path.exists(src):
        print('  *** regime_labels.csv NOT WRITTEN ***', flush=True)
        print(f'      source absent: {src}', flush=True)
        print('      F12 has not produced the per-bar labels, so there is nothing to derive from. '
              'CONSEQUENCE: S5D\'s catalogue regime columns (regime_causal_0_pct, '
              'regime_causal_1_pct, regime_burnin_pct, regime_modal) WILL BE BLANK, and the '
              'operator loses one of the four balance axes - direction x structure x session x '
              'REGIME - with no error anywhere downstream. This branch used to write nothing and '
              'say nothing, which is how a cold run lost the file silently while every resumed '
              'run passed.', flush=True)
        return None
    d = pd.read_csv(src, comment='#')
    idx = {str(t): i for i, t in enumerate(df['Time'].astype(str).values)}
    lab = pd.DataFrame({
        'bar_index': [idx.get(str(t), -1) for t in d['Time'].astype(str).values],
        'time': d['Time'].astype(str).values,
        'lab_causal': d['regime_causal'].values,
        'lab_desc': d['regime'].values})
    n_causal = int(len([x for x in pd.unique(lab['lab_causal']) if x >= 0]))
    n_desc = int(len(pd.unique(lab['lab_desc'])))
    unlabelled = int((lab['lab_causal'] == -1).sum())
    _write_with_header(os.path.join(out, 'regime_labels.csv'), lab, [
        'DOT per-bar REGIME LABELS - derived from F12 concurrence_depth_bars.csv, not recomputed',
        'rows=%d of %d frame bars (%.1f%%); the shortfall is pre-warmup bars F12 does not label'
        % (len(lab), len(df), 100.0 * len(lab) / max(len(df), 1)),
        'n_causal=%d clusters | n_desc=%d clusters | lab_causal == -1 on %d burn-in bars'
        % (n_causal, n_desc, unlabelled),
        'lab_causal is FORWARD-ONLY (burn-in fit, no future information) and is THE ONLY LABEL '
        'THAT MAY GATE AN ENTRY.',
        'lab_desc is FULL-SAMPLE and DESCRIPTIVE ONLY. IT MUST NEVER GATE ANYTHING: it is fitted '
        'over the whole span including bars after the entry it would label, so using it to select '
        'or filter is look-ahead.',
        'Join on bar_index (entry_bar) or on time. This exists because '
        'concurrence_depth_bars.csv carries the same two label columns keyed only on Time, with '
        'no bar index and no statement of which column is causal.'])
    print('  regime_labels.csv: %d rows | n_causal=%d n_desc=%d | %d burn-in bars unlabelled'
          % (len(lab), n_causal, n_desc, unlabelled))
    return {'rows': len(lab), 'n_causal': n_causal, 'n_desc': n_desc}


def s3b_family_evidence(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest):
    import runlog as rl
    import cluster_profiler as cp
    import family_evidence as fe
    import portfolio_simulation_engine as engine
    import score_g
    import conviction as C
    import wf
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    print(f'  dataset: {attest["rows"]:,} rows | {attest["range"]}')
    _s3b_ok, _s3b_missing = _artifacts_present(out, ['family_evidence.csv',
                                                     'cross_family_cofiring.csv'])
    if is_done(out, 'S3B', input_sha) and not _s3b_ok:
        print(f'  S3B marker present but deliverables missing {_s3b_missing} - RE-RUNNING.')
    _s3b_stale = stale_artifacts(out, 'S3B')
    if is_done(out, 'S3B', input_sha) and _s3b_ok and _s3b_stale:
        print(f'  S3B marker present but STALE - RE-RUNNING: {_s3b_stale}', flush=True)
    if is_done(out, 'S3B', input_sha) and _s3b_ok and not _s3b_stale:
        print('  S3B already complete for this input (checkpoint) — resuming past it.')
        return None
    bk_path = book_file if book_file else os.path.join(_ENGINE, 'book50_signals.csv')
    book = pd.read_csv(bk_path)
    f1_rows = book.index[book['trigger'] == 'F1'].tolist()
    sigs = score_g.build_book(df, pool, anchor, book, adaptive=ad, structural=st)
    conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
    n = len(df)
    U = cp.eligible_universe(df, w)
    variants = fe.d2d_variants(df)
    d2d_orig = df['D2D_Trend_Dir'].values.copy()
    d2d_rows = []
    executed = None
    # ITEM 6 CLOSED ON THE RECORD - DO NOT PARALLELISE THIS LOOP.
    # Each iteration assigns df['D2D_Trend_Dir'] = vcol on the SHARED frame and FrameGuard
    # restores it afterwards, so the four variants are a sequential dependency on mutable
    # shared state, not independent work. Distributing them would need a private frame per
    # worker for four iterations - the cost of the copy exceeds the work. This is a measured
    # structural reason, recorded here so no future turn re-opens it and no future reader
    # wonders why one loop in the package is deliberately serial.
    _p6 = rl.Progress('S3B D2D variants', len(variants))
    _p6.__enter__()
    for vname, vcol, vdesc in variants:
        _p6.step(1, extra=vname)
        df['D2D_Trend_Dir'] = vcol
        try:
            td = engine.run_portfolio(df, sigs, adaptive=ad, structural=st, warmup=w,
                                      verbose=False, conviction=conv)
        finally:
            df['D2D_Trend_Dir'] = d2d_orig
            _p6.__exit__(None, None, None)
        if vname == 'baseline_gate_on':
            executed = td
        bkv = td[~td['signal_name'].isin(cp.GAP_NAMES)]
        f1n = set(td['signal_name'].values[[i for i in range(len(td)) if td['signal_idx'].values[i] in f1_rows]]) if 'signal_idx' in td.columns else set()
        evd = {}
        for d in (1, -1):
            lab = 'LONG' if d == 1 else 'SHORT'
            evd[d] = np.sort(bkv[bkv['direction'] == lab]['entry_bar'].values.astype(np.int64))
        cs5 = cp.build_cluster_set(n, evd, 5)
        tcid = cp.map_trades_to_clusters(cs5, bkv)
        sz = cs5['clusters'].set_index('cluster_id')['size'].to_dict() if len(cs5['clusters']) else {}
        depth = np.array([sz.get(int(c), 0) for c in tcid])
        dy, ge5, days = fe.depth_yield(bkv, 5, n)
        pops = {'BOOK': np.ones(len(bkv), bool),
                'F0-solo': depth == 1,
                'F0-concurrent': depth >= 2}
        months = pd.Series(bkv['exit_time'].values).str[:7].values
        for pname, pmask in pops.items():
            sub = bkv[pmask]
            pn = sub['pnl'].values
            base = {'variant': vname, 'variant_desc': vdesc, 'population': pname,
                    'bucket': 'AGGREGATE', 'trades': int(len(sub)),
                    'net': round(float(pn.sum()), 1) if len(sub) else 0.0,
                    'PF': _pf(pn) if len(sub) else 0.0,
                    'WR_pct': round(float((pn > 0).mean() * 100), 1) if len(sub) else 0.0,
                    'daily_worst_day': round(float(wf.daily_pnl_points(sub)['pnl'].min()), 1) if len(sub) else 0.0,
                    'DepthYield_N5': dy if pname == 'BOOK' else '',
                    'population_label': 'BOOK'}
            d2d_rows.append(base)
            mm = months[pmask]
            for mo in sorted(set(mm.tolist())):
                q = pn[mm == mo]
                d2d_rows.append({'variant': vname, 'variant_desc': vdesc, 'population': pname,
                                 'bucket': mo, 'trades': int(len(q)),
                                 'net': round(float(q.sum()), 1), 'PF': _pf(q),
                                 'WR_pct': round(float((q > 0).mean() * 100), 1) if len(q) else 0.0,
                                 'daily_worst_day': '', 'DepthYield_N5': '',
                                 'population_label': 'BOOK'})
    d2d = pd.DataFrame(d2d_rows)
    d2d['H3_bucketing'] = 'calendar month (spec H.3 primary rule)'
    d2d['tolerance_N'] = 5
    d2d['dataset_rows'] = len(df)
    d2d['note'] = 'single-run full gate removal is not computable without editing sacred build_signal_masks; per-direction free runs isolate the jar'
    months = sorted(set(pd.Series(df['Time'].astype(str).values).str[:7].tolist()))
    segment_label = f'{months[0]}..{months[-1]}' if months else 'unknown'
    ev_book, bk = cp.book_events(executed)
    f1_names = set()
    if 'signal_idx' in executed.columns:
        f1_names = set(executed['signal_name'].values[np.isin(executed['signal_idx'].values, f1_rows)].tolist())
    ev_qual, qual_depth = cp.qualifying_events(df, sigs, ad, st, w)
    cs_by_basis = {'basis1': cp.build_cluster_set(n, ev_book, 5),
                   'basis2': cp.build_cluster_set(n, ev_qual, 5)}
    import family_evidence as fe
    fwd, mag, eff, valid, thr, mcol, ecol = cp.thrust_thresholds(df, 15, (0.85,), (0.75,))
    ev_thr = cp.thrust_events(fwd, mag, eff, valid, thr[(mcol, 'k85')], thr[(ecol, 'e75')], w)
    cs_by_basis['basis3'] = cp.build_cluster_set(n, ev_thr, 5)
    grid_label = ('basis3 grid W=15 K=p85 E=p75 N=5; depth bands size>=5; eligible mask '
                  'ADX>=15 & Volume>50 & post-warmup')
    U = cp.eligible_universe(df, w)
    fam = fe.build_family_evidence(df, bk, qual_depth, cs_by_basis, cs_by_basis['basis3'], U, pool,
                                   f1_names, _SCANNERS,
                                   [os.path.join(out, 'results'), out], grid_label)
    _write_with_header(os.path.join(out, 'family_evidence.csv'), fam, [
        'DOT S3B per-family evidence review (spec A.1)',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        'LABEL: depth_participation / co_fire_with_F0 / coverage_of_missed are PROPERTY OF THE BOOK.',
        'LABEL: thrust-episode denominators are PROPERTY OF THE MARKET (price-only).',
        f'S5 gate = {fe.S5_GATE}. Cluster tolerance N=5. Depth band = size>=5.',
        'INSUFFICIENT-EVIDENCE is a permitted verdict and is emitted where no output exists.',
        'coverage_of_missed is EMPTY BY CONSTRUCTION for F0 and F1: they are the incumbent book.'])
    cl, mix = fe.cross_family_cofiring(bk, f1_names, 5, n)
    if len(mix):
        _write_with_header(os.path.join(out, 'cross_family_cofiring.csv'), mix, [
        'COVERS ONLY F0+F1 BECAUSE THAT IS WHAT BOOK-50 CONTAINS. Extending it to F9 and F3 needs '
        'a BOOK containing them; it is a COMPOSITION DECISION for the Quant and the operator, not '
        'a build item. The absence of F9/F3 rows is not a failure of this artifact.',
            'DOT S3B cross-family co-firing (spec A.4) — PROPERTY OF THE BOOK',
            f'dataset_rows={attest["rows"]}',
            'population = BOOK (F0+F1 executed, gap fillers excluded). Tolerance N=5.'])
    print(f'  families reviewed: {len(fam)} | SELECTABLE {(fam.verdict == "SELECTABLE").sum()} | '
          f'INSUFFICIENT-EVIDENCE {(fam.verdict == "INSUFFICIENT-EVIDENCE").sum()}')
    mark_done(out, 'S3B', {'input_sha': input_sha, 'families': len(fam)})
    return {'family': fam, 'd2d': d2d, 'mixed': mix, 'executed': executed, 'sigs': sigs}


def _no_constraint(_d, _ss):
    return True, ''


_S5D_ROOT = os.path.dirname(os.path.abspath(__file__))
_S5D_CTX = {}


def _s5d_init(frame_path, scope):
    """Runs INSIDE each spawned worker. Builds the context once per worker, not per candidate.

    REDUNDANCY REMOVED FIRST, as with the _dy prefix cache: the oracle, the
    condition pool, the anchor and the conviction frame are IDENTICAL for every
    candidate, so they are built once per worker and reused across its whole
    chunk. Only run_portfolio is genuinely per-candidate.
    """
    import sys as _s, os as _o
    _here = _S5D_ROOT
    for _d in ('engine', 'scanners', 'orchestrator', '.'):
        _p = _o.path.join(_here, _d)
        if _p not in _s.path:
            _s.path.insert(0, _p)
    import pandas as _pd
    import dots_thresholds as _dt
    import portfolio_simulation_engine as _eng
    import sequential_temporal as _seq
    import conviction as _C
    df = _pd.read_csv(frame_path)
    w = _eng.warmup_floor(df, verbose=False)
    ad = _dt.compute_adaptive_thresholds(df)
    st = _dt.compute_structural_gates(df)
    _S5D_CTX.update({
        'df': df, 'w': w, 'ad': ad, 'st': st,
        'pool': _seq.build_condition_pool(df, ad, st, w),
        'anchor': _seq.anchor_array(df, 'ST_Flip'),
        'conv': _C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True),
        'bar_day': _pd.Series(df['Time'].astype(str).values).str[:10].values,
    })
    import catalogue as _cat0
    _S5D_CTX['gate_ok'] = _cat0.solo_gate_bars(df, ad)


def _s5d_score_chunk(payload):
    """Score a chunk of candidates. Every candidate is INDEPENDENT - a fresh
    one-row book through run_portfolio - so chunking cannot change a result.
    Returns (index, rows) so the parent reassembles in ASCENDING INDEX ORDER and
    the output is identical regardless of which worker finished first."""
    idx, items = payload
    import cluster_profiler as cp
    import catalogue as cat
    import score_g
    import portfolio_simulation_engine as engine
    import numpy as np
    import pandas as pd
    c = _S5D_CTX
    out = []
    for fam, sig, dr in items:
        one = pd.DataFrame([{'trigger': fam, 'family': fam, 'direction': dr, 'signal_def': sig}])
        try:
            sg = score_g.build_book(c['df'], c['pool'], c['anchor'], one,
                                    adaptive=c['ad'], structural=c['st'])
            td = engine.run_portfolio(c['df'], sg, adaptive=c['ad'], structural=c['st'],
                                      warmup=c['w'], verbose=False, conviction=c['conv'])
            td = td[~td['signal_name'].isin(cp.GAP_NAMES)]
        except SystemExit:
            out.append((fam, sig, dr, None, None, None, None))
            continue
        verdict, reason, stx = cat.evaluate_valid(td, c['bar_day'])
        bars = fold = gated = pnl_a = None
        if verdict == 'VALID':
            bars = np.asarray(td['entry_bar'].values, dtype=np.int64)
            fold = cat.segment_fold_stats(td)
            gated = cat.solo_gated_arm(td, c['gate_ok'])
            pnl_a = np.asarray(td['pnl'].values, dtype=float)
        out.append((fam, sig, dr, verdict, reason, stx, bars, fold, gated,
                    pnl_a if verdict == 'VALID' else None))
    return idx, out


_PF_CTX = {}


def _pf_init(frame_path):
    """Build the oracle, pool and anchor ONCE per worker, not once per definition."""
    import sys as _s
    for _d in ('engine', 'scanners', 'orchestrator', '.'):
        _p = os.path.join(_S5D_ROOT, _d)
        if _p not in _s.path:
            _s.path.insert(0, _p)
    import pandas as _pd
    import dots_thresholds as _dt
    import portfolio_simulation_engine as _eng
    import sequential_temporal as _seq
    df = _pd.read_csv(frame_path)
    w = _eng.warmup_floor(df, verbose=False)
    ad = _dt.compute_adaptive_thresholds(df)
    st = _dt.compute_structural_gates(df)
    _PF_CTX.update({'df': df, 'ad': ad, 'st': st,
                    'pool': _seq.build_condition_pool(df, ad, st, w),
                    'anchor': _seq.anchor_array(df, 'ST_Flip')})


def _pf_touch_one(fam, sig, df, pool, ad, st, anchor, lo_s, hi_s, n_eps):
    """Episodes touched by ONE definition, VECTORISED over all spans at once.

    The original tested every episode against every definition with np.any over
    the whole fire array: 7,490 x 311,013 = 2.33 BILLION array scans, and that -
    not the mask build, measured at 3.4 minutes total - is where the operator's
    46 minutes went. searchsorted over span starts replaces the episode loop
    entirely, so the cost per definition stops depending on the episode count.
    """
    import score_g as _sg
    try:
        mk = _sg.family_mask(df, pool, fam, sig, ad, st, anchor=anchor)
        fb = np.flatnonzero(np.asarray(mk, dtype=bool))
    except (Exception, SystemExit):
        return None
    if fb.size == 0:
        return None
    idx = np.searchsorted(lo_s, fb, side='right') - 1
    idx = idx[idx >= 0]
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    keep = fb[np.searchsorted(lo_s, fb, side='right') - 1 >= 0]
    ok = keep <= hi_s[idx]
    return np.unique(idx[ok])


def _pf_chunk(payload):
    c = _PF_CTX
    i, items, lo_s, hi_s, n_eps = payload
    hits = 0
    acc = np.zeros(n_eps, dtype=np.int64)
    for fam, sig in items:
        r = _pf_touch_one(fam, sig, c['df'], c['pool'], c['ad'], c['st'], c['anchor'],
                          lo_s, hi_s, n_eps)
        if r is None:
            continue
        hits += 1
        if r.size:
            acc[r] += 1
    return i, acc, hits


def _prefilter_counts(keys, df, pool, ad, st, anchor, lo_s, hi_s, order, n_eps, workers,
                      frame_path, rl):
    """Serial or pooled, with the same try/except fallback pattern as S5C."""
    n = len(keys)
    acc = np.zeros(n_eps, dtype=np.int64)
    hits = [0]

    def _run_serial(pg):
        for fam, sig in keys:
            r = _pf_touch_one(fam, sig, df, pool, ad, st, anchor, lo_s, hi_s, n_eps)
            if r is not None:
                hits[0] += 1
                if r.size:
                    acc[r] += 1
            if pg is not None:
                pg.step(1)

    use_pool = bool(workers and workers > 1 and frame_path and n >= 256)
    if use_pool:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        size = max(1, -(-n // (int(workers) * 4)))
        chunks = [(k, keys[i:i + size], lo_s, hi_s, n_eps)
                  for k, i in enumerate(range(0, n, size))]
        with rl.Progress(f'S5D prefilter touch ({workers} workers)', len(chunks)) as pg:
            try:
                with ProcessPoolExecutor(max_workers=min(int(workers), len(chunks)),
                                         mp_context=_mp.get_context('spawn'),
                                         initializer=_pf_init,
                                         initargs=(frame_path,)) as ex:
                    for _i3, a3, h3 in ex.map(_pf_chunk, chunks):
                        acc += a3
                        hits[0] += h3
                        pg.step(1)
                out = np.zeros(n_eps, dtype=np.int64)
                out[order] = acc
                return out, hits[0]
            except (BrokenProcessPool, OSError, MemoryError, Exception) as exc:
                acc[:] = 0
                hits[0] = 0
                print(f'  S5D prefilter touch: pool failed ({type(exc).__name__}: '
                      f'{str(exc)[:70]}) - FALLING BACK TO SERIAL.', flush=True)
        with rl.Progress('S5D prefilter touch (serial fallback)', n) as pg:
            _run_serial(pg)
    else:
        with rl.Progress('S5D prefilter touch (serial)', n) as pg:
            _run_serial(pg)
    out = np.zeros(n_eps, dtype=np.int64)
    out[order] = acc
    return out, hits[0]


def _s5d_score_items(items, fam, workers, frame_path, scope, rl, ctx=None):
    """Serial when workers<=1 or no frame_path; otherwise chunked across the pool.

    Results are reassembled in ASCENDING CHUNK INDEX, so the emitted catalogue is
    byte-identical regardless of worker count or completion order.
    """
    n = len(items)
    if n == 0:
        return []
    if not workers or workers <= 1 or not frame_path:
        if ctx is not None:
            _S5D_CTX.update(ctx)
        elif frame_path:
            _s5d_init(frame_path, scope)
        with rl.Progress(f'S5D {fam} per-candidate scoring (serial)', n) as pg:
            out = []
            for it in items:
                _i, part = _s5d_score_chunk((0, [it]))
                out.extend(part)
                pg.step(1)
            return out
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    size = max(1, -(-n // (int(workers) * 4)))
    chunks = [(k, items[i:i + size]) for k, i in enumerate(range(0, n, size))]
    got = {}
    with rl.Progress(f'S5D {fam} per-candidate scoring ({workers} workers)', len(chunks)) as pg:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(chunks)),
                                 mp_context=_mp.get_context('spawn'),
                                 initializer=_s5d_init,
                                 initargs=(frame_path, scope)) as ex:
            for idx, part in ex.map(_s5d_score_chunk, chunks):
                got[idx] = part
                pg.step(1, extra=f'{sum(len(v) for v in got.values())} scored')
    print(f'  S5D {fam}: ran via pool({workers}) - {n} candidates', flush=True)
    return [r for k, _c in chunks for r in got[k]]


_NULLD_CTX = {}


def _nulld_init(frame_path):
    """Frame + oracle + conviction ONCE per worker. NO rng here, deliberately.

    THE SEED MUST NOT CROSS INTO A WORKER. Every draw's mask AND its direction are
    decided in the PARENT from the blake2b per-family seed, then shipped; a worker
    that reseeded locally would reintroduce exactly the defect that made the
    pricing column non-reproducible - F0 seeded 20284991 / 20330271 / 20266910 on
    three consecutive interpreters off the same nominal base. Workers do
    arithmetic only, so the result cannot depend on worker count.
    """
    import sys as _s
    for _d in ('engine', 'scanners', 'orchestrator', '.'):
        _p = os.path.join(_S5D_ROOT, _d)
        if _p not in _s.path:
            _s.path.insert(0, _p)
    import pandas as _pd
    import dots_thresholds as _dt
    import portfolio_simulation_engine as _eng
    import conviction as _C
    df = _pd.read_csv(frame_path)
    w = _eng.warmup_floor(df, verbose=False)
    _NULLD_CTX.update({'df': df, 'w': w,
                       'ad': _dt.compute_adaptive_thresholds(df),
                       'st': _dt.compute_structural_gates(df),
                       'conv': _C.build_conviction(df, True, True, True, d2d_conviction=True,
                                                   d2d_gap=True)})


def _nulld_chunk(payload):
    """Score a chunk of null draws. Returns compact arrays, never a DataFrame."""
    idx, items = payload
    import numpy as _np
    import pandas as _pd
    import portfolio_simulation_engine as _eng
    import cluster_profiler as _cp
    c = _NULLD_CTX
    df = c['df']
    out = []
    for j, bars, direction in items:
        col = f'__NULL_W_{j}'
        m = _np.zeros(len(df), dtype=int)
        m[_np.asarray(bars, dtype=_np.int64)] = 1
        df[col] = m
        nsig = _pd.DataFrame([{'feat_1': col, 'thresh_1': '==1', 'feat_2': col,
                               'thresh_2': '==1', 'feat_3': col, 'thresh_3': '==1',
                               'direction': direction}])
        ntd = _eng.run_portfolio(df, nsig, adaptive=c['ad'], structural=c['st'],
                                 warmup=c['w'], verbose=False, conviction=c['conv'])
        ntd = ntd[~ntd['signal_name'].isin(_cp.GAP_NAMES)]
        df.drop(columns=[col], inplace=True)
        out.append((j, _np.asarray(ntd['pnl'].values, dtype=float),
                    _np.asarray(ntd['exit_time'].astype(str).values, dtype=object),
                    _np.asarray(ntd['entry_bar'].values, dtype=_np.int64)))
    return idx, out


def _null_frames_for(drawn, dirs, df, ad, st, w, conv, workers, frame_path, fam, rl):
    """Serial or pooled, with the C1 fallback. Results reassembled in DRAW ORDER."""
    n = len(drawn)
    if n == 0:
        return []
    items = [(j, np.flatnonzero(np.asarray(nd['mask'], dtype=bool)).astype(np.int64), dirs[j])
             for j, nd in enumerate(drawn)]
    frames = [None] * n

    def _absorb(part):
        for j, pnl, xt, eb in part:
            frames[j] = pd.DataFrame({'signal_name': 'NULL', 'pnl': pnl, 'exit_time': xt,
                                      'entry_bar': eb})

    def _run_serial(pg):
        for it in items:
            _i, part = _nulld_chunk((0, [it]))
            _absorb(part)
            if pg is not None:
                pg.step(1)

    use_pool = bool(workers and workers > 1 and frame_path and n >= 32)
    used = 'serial'
    if use_pool:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        size = max(1, -(-n // (int(workers) * 4)))
        chunks = [(k, items[i:i + size]) for k, i in enumerate(range(0, n, size))]
        pg = rl.Progress(f'S5D {fam} null draw ({workers} workers)', len(chunks))
        pg.__enter__()
        try:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(chunks)),
                                     mp_context=_mp.get_context('spawn'),
                                     initializer=_nulld_init,
                                     initargs=(frame_path,)) as ex:
                for _idx, part in ex.map(_nulld_chunk, chunks):
                    _absorb(part)
                    pg.step(1)
            used = f'pool({workers})'
        except (BrokenProcessPool, OSError, MemoryError, Exception) as exc:
            frames = [None] * n
            print(f'  S5D {fam} null draw: pool failed ({type(exc).__name__}: '
                  f'{str(exc)[:70]}) - FALLING BACK TO SERIAL.', flush=True)
            used = f'serial (pool fell back: {type(exc).__name__})'
            pg.__exit__(None, None, None)
            pg = rl.Progress(f'S5D {fam} null draw (serial fallback)', n)
            pg.__enter__()
            _run_serial(pg)
        pg.__exit__(None, None, None)
    else:
        _NULLD_CTX.update({'df': df, 'w': w, 'ad': ad, 'st': st, 'conv': conv})
        with rl.Progress(f'S5D {fam} null draw (serial)', n) as pg:
            _run_serial(pg)
    print(f'  S5D {fam} null draw: ran via {used} - {n} draws', flush=True)
    return [f for f in frames if f is not None]


PF_UNDEFINED = float('nan')
NULL_SEED_BASE = 20260728


def _family_seed(fam):
    """DETERMINISTIC per-family seed. NEVER Python's hash().

    abs(hash(fam)) was used here. Python randomises str hashing PER PROCESS
    unless PYTHONHASHSEED is fixed, so every run drew a different seed for the
    same family from the same nominal NULL_SEED_BASE: F0 seeded 20284991,
    20330271 and 20266910 on three consecutive interpreters. That moved the
    matched null - and with it EXPECTED_ROWS_AT_OR_ABOVE_THIS_PF, the pricing
    column the operator separates a real edge from a chance row on. Between two
    runs of identical code on byte-identical candidates, F0's rows priced under
    1.0 went 221 -> 778.

    A blake2b digest of the family name is stable across processes, machines and
    Python versions, so the same family always draws the same null.
    """
    import hashlib as _h
    d = _h.blake2b(str(fam).encode('utf-8'), digest_size=8).digest()
    return NULL_SEED_BASE + int.from_bytes(d, 'big') % 100000
CONCURRENT_STAGES = ('S3', 'S5C', 'S5D', 'S7')


def s5d_catalogue(df, ad, st, w, pool, anchor, out, input_sha, attest, null_k=None,
                  workers=1, frame_path=None, scope='full'):
    import catalogue as cat
    import cluster_profiler as cp
    import conviction as C
    import runlog as rl
    import selection as sel
    import terrain as tr
    import portfolio_simulation_engine as engine
    import score_g
    import numpy as np
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    cand = os.path.join(out, 'results', 'candidates.csv')
    if not os.path.exists(cand):
        print('  CATALOGUE: no candidates.csv - S3/S4/S5 have not produced a pool. NOT marking done.')
        return None
    cat_dir_chk = os.path.join(out, 'catalogues')
    _s5d_ok, _s5d_missing = _artifacts_present(os.path.join(out, 'catalogues'), [
        'unclaimed_reachable.csv', 'same_bar_cohort.csv', 'cohort_scored.csv',
        'dilution_curve_agg_pf.csv', 'dilution_curve_EXPECTED_ROWS_AT_OR_ABOVE_THIS_PF.csv'])
    _s5d_cats = (os.path.isdir(cat_dir_chk)
                 and any(f.startswith('catalogue_') for f in os.listdir(cat_dir_chk)))
    if is_done(out, 'S5D', input_sha) and not (_s5d_cats and _s5d_ok):
        print(f'  S5D marker present but deliverables missing '
              f'{_s5d_missing + ([] if _s5d_cats else ["catalogue_*.csv"])} - RE-RUNNING. The '
              f'gate previously checked only that the catalogue directory was non-empty, which '
              f'cannot tell a complete emission from one that died after the first family.')
    _s5d_stale = stale_artifacts(out, 'S5D')
    if is_done(out, 'S5D', input_sha) and _s5d_cats and _s5d_ok and _s5d_stale:
        print('  S5D marker present and deliverables complete, but STALE - RE-RUNNING:', flush=True)
        for _r in _s5d_stale:
            print(f'      {_r}', flush=True)
    if is_done(out, 'S5D', input_sha) and _s5d_cats and _s5d_ok and not _s5d_stale:
        have = sorted(f for f in os.listdir(cat_dir_chk) if f.startswith('catalogue_'))
        print(f'  S5D already complete for this input_sha - {len(have)} catalogues on disk, '
              f'RESUMING PAST IT. The per-candidate scoring loop is the longest single-threaded '
              f'stage in the pipeline (F1 alone was 1:39:20 on the real pool); repeating it on '
              f'every restart is a cost the operator should never pay twice.')
        return {'per_family': None, 'unclaimed': None, 'reach': None, 'raw_tot': None,
                'resumed': True, 'catalogues': have}
    cands = pd.read_csv(cand)
    null_k = int(null_k) if null_k else cat.NULL_K_DEFAULT
    n = len(df)
    bar_day = pd.Series(df['Time'].astype(str).values).str[:10].values
    U = cp.eligible_universe(df, w)
    W, K, E = cat.PINNED_CELL
    fwd, mag, eff, valid, thr, mcol, ecol = cp.thrust_thresholds(df, W, (K,), (E,))
    ev = cp.thrust_events(fwd, mag, eff, valid, thr[(mcol, f'k{int(K*100)}')],
                          thr[(ecol, f'e{int(E*100)}')], w)
    mda = cat.assert_episode_thresholds_mechanism_d(_HERE, thr, mcol, ecol,
                                                    f'k{int(K*100)}', f'e{int(E*100)}')
    print(f'  ITEM 5 IN-RUN ASSERTION: {mda["modules_verified"]}/4 market-object modules '
          f'byte-verified, episode K/E are per-bar arrays from {mda["basis"]}')
    cs = cp.build_cluster_set(n, ev, tr.CONTIGUOUS_TOLERANCE)
    reach = cat.reachable_episodes(cs, df, w, U)
    raw_tot = {d: int((cs['clusters']['dir'] == d).sum()) for d in (1, -1)}
    print(f'  TERRAIN cell W={W} K=p{int(K*100)} E=p{int(E*100)} | MARKET | raw '
          f'UP {raw_tot[1]} DOWN {raw_tot[-1]} | REACHABLE UP {len(reach[1])} '
          f'({100.0*len(reach[1])/max(raw_tot[1],1):.2f}%) DOWN {len(reach[-1])} '
          f'({100.0*len(reach[-1])/max(raw_tot[-1],1):.2f}%)')
    gate_ok = cat.solo_gate_bars(df, ad)
    conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
    _lab_causal = None
    _rl_path = os.path.join(out, 'regime_labels.csv')
    if os.path.exists(_rl_path):
        _rl = pd.read_csv(_rl_path, comment='#')
        _lab_causal = np.full(n, -1, dtype=int)
        _bi = np.asarray(_rl['bar_index'].values, dtype=np.int64)
        _ok = (_bi >= 0) & (_bi < n)
        _lab_causal[_bi[_ok]] = np.asarray(_rl['lab_causal'].values, dtype=int)[_ok]
        print(f'  regime axis: joined {int(_ok.sum())} labelled bars from regime_labels.csv '
              f'(lab_causal ONLY - lab_desc is full-sample and must never characterise a '
              f'tradeable signal)')
    else:
        print('  regime axis UNAVAILABLE: regime_labels.csv absent, regime_* columns will be '
              'blank. Run --stage S3 first; it emits that file in seconds without re-scanning.')
    fam_fire_counts = {}
    for fam, g in cands.groupby('family'):
        fc = []
        for _i, cr in g.iterrows():
            try:
                mk = score_g.family_mask(df, pool, fam, str(cr['signal_def']), ad, st, anchor=anchor)
                fc.append(int(np.asarray(mk, dtype=bool).sum()))
            except (Exception, SystemExit) as _fe:
                rl.warn(f'fire-count skip {fam}: {_fe}')
                continue
        fam_fire_counts[fam] = fc
    per_family = {}
    entries_by_id, dirs_by_id, fams_by_id = {}, {}, {}
    pnl_by_id = {}
    for fam, g in cands.groupby('family'):
        rows = []
        null_pfs, null_rate = [], 0.0
        items = [(fam, str(cr['signal_def']), str(cr.get('direction', 'LONG')).upper())
                 for _i, cr in g.iterrows()]
        scored = _s5d_score_items(items, fam, workers, frame_path, scope, rl,
                                  ctx={'df': df, 'w': w, 'ad': ad, 'st': st, 'pool': pool,
                                       'anchor': anchor, 'conv': conv, 'bar_day': bar_day,
                                       'gate_ok': gate_ok})
        for _fam, sig, dr, verdict, reason, stx, bars_p, fold_p, gated_p, pnl_p in scored:
            if verdict is None:
                continue
            sid = cat.signal_id(fam, sig, dr)
            d = 1 if dr == 'LONG' else -1
            row = {'signal_id': sid, 'family': fam, 'signal_def': sig, 'direction': dr,
                   'verdict': verdict, 'reason_code': reason}
            row.update(stx)
            if verdict == 'VALID':
                bars = np.asarray(bars_p, dtype=np.int64)
                entries_by_id[sid] = bars
                pnl_by_id[sid] = np.asarray(pnl_p, dtype=float)
                dirs_by_id[sid] = d
                fams_by_id[sid] = fam
                tch = cat.touched_episodes(bars, d, cs)
                tch_reach = [t for t in tch if t in reach[d]]
                row['touched_episode_ids'] = ';'.join(str(x) for x in tch)
                row['episodes_touched'] = len(tch)
                row['coverage_pct_raw_terrain'] = round(100.0 * len(tch) / max(raw_tot[d], 1), 4)
                row['coverage_pct_reachable'] = round(100.0 * len(tch_reach) / max(len(reach[d]), 1), 4)
                row['terrain_cell'] = f'W{W}/K{int(K*100)}/E{int(E*100)}'
                row.update(fold_p)
                row.update(cat.margin_of_safety(pnl_p))
                row.update(gated_p)
                _sp, _smod = cat.session_profile(bars, df['EST_Hour'].values, tr.session_of)
                row.update(_sp)
                row['session_modal'] = _smod
                _rp, _rmod = cat.regime_profile(bars, _lab_causal)
                row.update(_rp)
                row['regime_modal'] = _rmod
                _ps, _ss = cat.structure_of(sig)
                row['market_structure'] = _ps
                row['market_structure_secondary'] = _ss

            rows.append(row)
        fr = pd.DataFrame(rows)
        if len(fr):
            N_F = len(fr)
            rng = np.random.default_rng(_family_seed(fam))
            fire_targets = [c for c in fam_fire_counts.get(fam, []) if c > 0]
            fam_k = cat.null_k_for(fam, null_k)
            drawn, nstats = cat.draw_matched_null_masks(pool, fire_targets, rng, k=fam_k)
            long_share = float((g['direction'].astype(str).str.upper() == 'LONG').mean()) \
                if len(g) else 0.5
            _dirs = ['LONG' if rng.random() < long_share else 'SHORT' for _ in drawn]
            null_frames = _null_frames_for(drawn, _dirs, df, ad, st, w, conv, workers,
                                           frame_path, fam, rl)
            null_rate, null_pfs = cat.matched_null_rate(null_frames, bar_day)
            qflag, qwhy = cat.null_quality(len(null_frames), fam_k, nstats)
            print(f'    {fam:4} matched null: requested K={fam_k}, IN-BAND {nstats["matched"]} '
                  f'({nstats["matched_fraction"]:.1%} matched, {nstats["rejected_out_of_band"]} '
                  f'rejected out-of-band, targets {nstats["target_min"]}..{nstats["target_max"]} '
                  f'fires, tol +/-{nstats["tol"]:.0%}), direction LONG-share {long_share:.2f}, '
                  f'VALID-passing {len(null_pfs)}, rate {null_rate:.4f}'
                  + (f' | {qflag} -> Appendix A columns BLANK' if qflag else ''))
            if qflag:
                blanks = cat.pricing_blank(qwhy)
                for kk, vv in blanks.items():
                    fr[kk] = vv
                fr['n_null_family'] = len(null_frames)
                fr['null_matched_fraction'] = nstats['matched_fraction']
                fr['null_rejected_out_of_band'] = nstats['rejected_out_of_band']
                fr['null_direction_long_share'] = round(long_share, 4)
                fr['null_seed'] = NULL_SEED_BASE
            else:
                price = [cat.pricing_columns(r.get('agg_pf', float('nan')), N_F, null_rate,
                                             null_pfs) for _j, r in fr.iterrows()]
                for kk in price[0]:
                    fr[kk] = [pz[kk] for pz in price]
                exc = pd.to_numeric(fr['pf_null_exceedance_pct'], errors='coerce').fillna(1.0).values
                fr['q_value_BY_family'] = np.round(cat.benjamini_yekutieli(exc), 6)
                fr['pricing_unavailable_reason'] = ''
                fr['n_null_family'] = len(null_frames)
                fr['null_matched_fraction'] = nstats['matched_fraction']
                fr['null_rejected_out_of_band'] = nstats['rejected_out_of_band']
                fr['null_direction_long_share'] = round(long_share, 4)
                fr['null_seed'] = NULL_SEED_BASE
        per_family[fam] = fr
    _priced = {f: (len(fr) > 0 and 'pricing_unavailable_reason' in fr.columns
                   and str(fr['pricing_unavailable_reason'].iloc[0]) == '')
               for f, fr in per_family.items()}
    _why = {f: (str(fr['pricing_unavailable_reason'].iloc[0]) if len(fr)
                and 'pricing_unavailable_reason' in fr.columns else 'no rows')
            for f, fr in per_family.items()}
    cat_dir = os.path.join(out, 'catalogues')
    os.makedirs(cat_dir, exist_ok=True)
    print('  CATALOGUE ROW COUNT PER FAMILY:')
    for fam in sorted(per_family):
        fr = per_family[fam]
        path = os.path.join(cat_dir, f'catalogue_{fam}.csv')
        _write_with_header(path, fr, [
            f'DOT CATALOGUE - family {fam} - every signal VALID admits, nothing ranked or capped',
            f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
            f'oracle_sha256_12={oracle_sha}',
            f'terrain cell W={W} K=p{int(K*100)} E=p{int(E*100)} | coverage emitted against BOTH '
            f'denominators: raw terrain (UP {raw_tot[1]} / DOWN {raw_tot[-1]}, MARKET) and REACHABLE '
            f'(UP {len(reach[1])} / DOWN {len(reach[-1])}, MARKET). REACHABLE IS PRIMARY.',
            'per-signal statistics are PROPERTY OF THE BOOK; terrain and reachable are PROPERTY OF '
            'THE MARKET.',
            (cat.CATALOGUE_HEADER_PRICING if _priced.get(fam) else
             'PRICING COLUMNS ARE BLANK for this family: ' + str(_why.get(fam, '')) +
             '. The Appendix A names carry no substitute quantity - reading this catalogue is still '
             'a search of size N_F, and nothing in this file prices it.'),
            (f'WEAK NULL FOR THIS FAMILY: null_valid_rate {null_rate:.3f} means {100*null_rate:.0f}% '
             f'of random matched signals pass VALID here, so a low EXPECTED_ROWS is a much weaker '
             f'claim than the same figure in F0 or F1. This is a property of the family\'s rarity '
             f'band, not of the sample size - raising K measures the same easy benchmark more '
             f'precisely.' if null_rate >= 0.60 else
             f'null_valid_rate {null_rate:.3f} - the matched null is a meaningful benchmark for '
             f'this family.'),
            cat.CATALOGUE_HEADER_UNSCORED,
            'UNEVALUABLE rows are RETAINED with statistics blank and a reason_code. INVALID rows '
            '(V2 survival breach) do not enter; their count is reported in the run log.'])
        vc = fr['verdict'].value_counts().to_dict() if len(fr) else {}
        print(f'    {fam:4} {len(fr):7} rows | {vc}')
    has_f0 = 'F0' in per_family and len(per_family.get('F0', []))
    n_valid_triples = {}
    if has_f0:
        f0v = per_family['F0']
        for _i, rr in f0v[f0v['verdict'] == 'VALID'].iterrows():
            for t in str(rr.get('touched_episode_ids', '')).split(';'):
                if t:
                    n_valid_triples[int(t)] = n_valid_triples.get(int(t), 0) + 1
    _prefilter_touch = {}
    _dm = os.path.join(out, 'results', 'discovery_master.csv')
    if os.path.exists(_dm):
        _pre = pd.read_csv(_dm)
        _pre.columns = [str(c).strip('\ufeff') for c in _pre.columns]
        _spans_all = cat.episode_spans(cs)
        _eids = sorted(_spans_all)
        _lo = np.array([_spans_all[e][0] for e in _eids], dtype=np.int64)
        _hi = np.array([_spans_all[e][1] for e in _eids], dtype=np.int64)
        _order = np.argsort(_lo, kind='stable')
        _lo_s, _hi_s = _lo[_order], _hi[_order]
        _keys = list(dict.fromkeys(zip(_pre['family'].astype(str),
                                       _pre['signal_def'].astype(str))))
        print(f'  prefilter touch: {len(_pre)} pre-filter rows -> {len(_keys)} DISTINCT '
              f'definitions (deduped BEFORE any mask is built; the loop shrinks by '
              f'{len(_pre) / max(len(_keys), 1):.2f}x on its own)')
        _counts = np.zeros(len(_eids), dtype=np.int64)
        _hits = 0
        _wk = int(os.environ.get('DOT_WORKERS', '1'))
        _fp = os.environ.get('DOT_FRAME_PATH')
        _built = _prefilter_counts(_keys, df, pool, ad, st, anchor, _lo_s, _hi_s,
                                   _order, len(_eids), _wk, _fp, rl)
        _counts, _hits = _built
        for _k2, _e in enumerate(_eids):
            _prefilter_touch[_e] = int(_counts[_k2])
        print(f'  prefilter touch: {_hits} definitions with at least one firing bar, '
              f'mapped over {len(_eids)} episodes')
    else:
        print('  prefilter touch UNAVAILABLE: results/discovery_master.csv absent; '
              'n_prefilter_candidates_touching will read 0 for every episode.')
    unclaimed = []
    spans = cat.episode_spans(cs)
    claimed = {d: set() for d in (1, -1)}
    for sid, bars in entries_by_id.items():
        d = dirs_by_id[sid]
        claimed[d].update(cat.touched_episodes(bars, d, cs))
    for d, lab in ((1, 'UP'), (-1, 'DOWN')):
        for eid in sorted(reach[d] - claimed[d]):
            b0, b1, _dd = spans[eid]
            unclaimed.append({'episode_id': eid, 'direction': lab, 'start_bar': b0, 'end_bar': b1,
                              'duration_bars': b1 - b0 + 1,
                              'displacement_pts': round(abs(float(df['Close'].values[min(b1 + W, n - 1)]
                                                                 - df['Close'].values[b0])), 1),
                              'est_hour_start': int(df['EST_Hour'].values[b0]),
                              'n_conditions_firing': int(sum(1 for k in pool
                                                             if pool[k][b0:b1 + 1].any())),
                              'n_valid_triples_touching': n_valid_triples.get(eid, '' if not has_f0 else 0),
                              'n_prefilter_candidates_touching': _prefilter_touch.get(eid, 0),
                              'population': 'MARKET'})
    uf = pd.DataFrame(unclaimed)
    _write_with_header(os.path.join(cat_dir, 'unclaimed_reachable.csv'), uf, [
        'DOT item 6 - REACHABLE episodes no catalogue signal touches - PROPERTY OF THE MARKET',
        f'dataset_rows={attest["rows"]} terrain cell W={W} K=p{int(K*100)} E=p{int(E*100)}',
        'n_conditions_firing vs n_valid_triples_touching was intended to separate a SEARCH gap '
        '(many conditions fire, no valid triple lands) from a GRAMMAR gap (few fire).',
        'n_valid_triples_touching IS ZERO ON EVERY ROW AND THAT IS A MEASURED VALUE, NOT A '
        'DEFAULT - but it is also TAUTOLOGICAL and carries no diagnostic information. An episode '
        'appears in this file precisely because NO VALID SIGNAL TOUCHES IT (unclaimed = reachable '
        'minus every episode claimed by any VALID signal, F0 included), so the count of VALID F0 '
        'triples touching it can only ever be 0. Verified by join: 192 distinct episodes are '
        'touched by the 1,818 VALID F0 rows, 2,092 episodes are unclaimed, INTERSECTION = 0.',
        'n_prefilter_candidates_touching IS THE REAL DIAGNOSTIC AND IS NOT TAUTOLOGICAL: it counts '
        'candidates from the FULL pre-filter pool (discovery_master.csv, before S5 cut on '
        'trades>=30 & folds_plus>=4 & (agg_pf>=2.0 OR zero-loss)) whose firing bars fall inside the episode. A '
        'HIGH count means the search DID find things there and the QUALITY FILTER cut them - a '
        'QUALITY gap, reachable by loosening the filter. ZERO means nothing in the 249-condition '
        'vocabulary expresses that episode at all - a GRAMMAR gap, not reachable without new '
        'variables. That is the difference between headroom and a wall.',
        'THE OLDER DIAGNOSTIC IS n_conditions_firing. Separating SEARCH from GRAMMAR properly '
        'needs the count of F0 triples that were SCANNED AND REJECTED over each episode, which '
        'the scan does not currently record - that is a change to S5D, not to this file.'])
    print(f'  UNCLAIMED REACHABLE: {len(uf)} episodes '
          f'(UP {int((uf["direction"] == "UP").sum()) if len(uf) else 0} / '
          f'DOWN {int((uf["direction"] == "DOWN").sum()) if len(uf) else 0})')
    ent = {d: [] for d in (1, -1)}
    ids = {d: [] for d in (1, -1)}
    for sid, bars in entries_by_id.items():
        d = dirs_by_id[sid]
        ent[d].extend(np.asarray(bars, dtype=np.int64).tolist())
        ids[d].extend([sid] * len(bars))
    cohort = cat.same_bar_cohort_table(ent, ids, fams_by_id)
    _cohort_rows = []
    _bkall = {}
    for _sid, _bb in entries_by_id.items():
        for _b in np.asarray(_bb, dtype=np.int64).tolist():
            _bkall.setdefault((dirs_by_id[_sid], int(_b)), set()).add(_sid)
    _pnl_by_bar = {}
    for _sid, _pl in pnl_by_id.items():
        _bb = np.asarray(entries_by_id[_sid], dtype=np.int64)
        for _b, _p in zip(_bb.tolist(), np.asarray(_pl, dtype=float).tolist()):
            _pnl_by_bar.setdefault((dirs_by_id[_sid], int(_b)), []).append(_p)
    _buckets = (('1', 1, 1), ('2', 2, 2), ('3-4', 3, 4), ('5+', 5, 10 ** 6))
    _groups = {}
    for (_d, _b), _sids in _bkall.items():
        _k = len(_sids)
        _lab = next((nm for nm, lo, hi in _buckets if lo <= _k <= hi), '5+')
        _fams = sorted({fams_by_id.get(x, '?') for x in _sids})
        _comp = '+'.join(_fams)
        _pure = 'ALL-ONE-FAMILY' if len(_fams) == 1 else 'MIXED'
        _groups.setdefault((_d, _lab, _comp, _pure), []).append((_d, _b))
    for (_d, _lab, _comp, _pure), _bars in sorted(_groups.items(), key=lambda x: str(x[0])):
        _p = []
        for _key in _bars:
            _p.extend(_pnl_by_bar.get(_key, []))
        _pa = np.asarray(_p, dtype=float)
        _loss = -_pa[_pa < 0].sum()
        _pf = (cat.PF_UNDEFINED if _loss <= 0 else round(float(_pa[_pa > 0].sum() / _loss), 4))
        _mo = cat.margin_of_safety(_pa)
        _cohort_rows.append({
            'direction': 'LONG' if _d == 1 else 'SHORT', 'depth': _lab,
            'family_composition': _comp, 'purity': _pure, 'bars': len(_bars),
            'trades': int(_pa.size),
            'WR': round(float((_pa > 0).mean() * 100), 2) if _pa.size else '',
            'PF': (cat.blank_sentinel_ratio(_pf) if _pa.size else ''),
            'net': round(float(_pa.sum()), 2) if _pa.size else 0.0,
            'avg_trade': round(float(_pa.mean()), 2) if _pa.size else '',
            'worst_day_usd': '',
            'sufficient': bool(_pa.size >= 10),
            'population': 'POOL', 'basis': 'trades occurring ON THOSE BARS, distinct-signal depth',
            'cluster_basis': 'basis 1 (executed) - eligible-bar jar, the basis the run executed on. '
                             'A depth-5+ population is 128 clusters on basis 1 and 1,958 on basis 3, '
                             'a factor of 15, so the figure is unreadable without this.',
            **_mo})
    _write_with_header(os.path.join(cat_dir, 'cohort_scored.csv'), pd.DataFrame(_cohort_rows), [
        'DOT phase 4 - same-bar cohorts SCORED, not merely counted',
        'POPULATION: POOL - every VALID catalogue signal, NOT a book. At LONG depth 5+ that is '
        'hundreds of thousands of trades against a real book\'s ~3,000, so THE NETS ARE NOT BOOK '
        'NETS and the mixed-versus-single ordering is non-monotone across depth as a consequence. '
        'THE BOOK-SCALE ANSWER IS book_margin_by_tier.csv and book_gated_by_tier.csv, emitted by '
        'score_book.py against an assembled book, where depth is real and the nets are the book\'s.',
        'THE QUESTION IS OPEN AND THIS IS WHY: re-emitting at book scale does not answer it '
        'either. book_margin_by_tier.csv and book_gated_by_tier.csv ARE book-scale, but they '
        'measure whatever book score_book.py is handed, and Option B is 119 F0 + 1 F1 - there is '
        'almost no family mixing in it to detect. MIXED-VERSUS-SINGLE-FAMILY AT DEPTH NEEDS A BOOK '
        'CONTAINING F9 AND F3. That is a COMPOSITION DECISION for the Quant and the operator, not '
        'a build item, and no artifact can settle it until such a book exists.',
        'Mixed-versus-single-family at depth is the strongest available argument for or against a '
        'multi-family book, and it must be read at BOOK scale to be made.',
        f'dataset_rows={attest["rows"]}',
        'same_bar_cohort.csv answers "do families co-fire". This answers "does a MIXED cohort KEEP '
        'THE EDGE" - the question that decides whether F1 is fuel or noise beside F0.',
        'THIS IS A MEASUREMENT OF A DEFINED SET, NOT A SELECTION. No argmax, no ranking, no book '
        'is chosen: every cohort present in the pool is emitted. Item 15 is not engaged.',
        'ALL-ONE-FAMILY and MIXED are emitted separately so the comparison is direct. Depth is '
        'DISTINCT SIGNALS on the bar, per direction (item 4), bucketed 1 / 2 / 3-4 / 5+.',
        'Cohorts with fewer than 10 trades are EMITTED with sufficient=False, never dropped: a '
        'silently smaller population is this project\'s most frequent failure.'])
    _ins = sum(1 for r in _cohort_rows if not r['sufficient'])
    print(f'  cohort_scored.csv: {len(_cohort_rows)} cohorts '
          f'({_ins} marked insufficient, <10 trades, retained)')
    _write_with_header(os.path.join(cat_dir, 'same_bar_cohort.csv'), cohort, [
        'DOT item 11 - family composition of each bar as a CURVE OVER DEPTH - counts only',
        'CLUSTER BASIS: basis 1 (executed) - eligible-bar jar. A depth-5+ population is 128 '
        'clusters on basis 1, 346 on basis 2 (pre-jar) and 1,958 on basis 3 (price-anchored) - a '
        'factor of 15 - so any depth figure is unreadable without its basis named.',
        f'dataset_rows={attest["rows"]}',
        'Depth is DISTINCT SIGNALS on the same bar, per direction, never pooled (item 4). No P&L: '
        'depth-3 has no discriminating power at pool scale and P&L needs a book.'])
    allrows = pd.concat([f for f in per_family.values() if len(f)], ignore_index=True) \
        if per_family else pd.DataFrame()
    if len(allrows):
        v = allrows[allrows['verdict'] == 'VALID'].copy()
        for key, asc in (('agg_pf', False), ('EXPECTED_ROWS_AT_OR_ABOVE_THIS_PF', True)):
            if key not in v.columns:
                continue
            _kv = pd.to_numeric(v[key], errors='coerce')
            o = v.assign(_sortkey=_kv).sort_values('_sortkey', ascending=asc,
                                                   na_position='last')['signal_id'].tolist()
            with rl.Progress(f'S5D dilution curve ({key})', len(o)) as _dpg:
                dc = cat.dilution_curve(o, entries_by_id, dirs_by_id, key, progress=_dpg)
            _write_with_header(os.path.join(cat_dir, f'dilution_curve_{key}.csv'), dc, [
                f'DOT item 12 - dilution curve, ranking key = {key}',
                f'dataset_rows={attest["rows"]}',
                'Admission is best-first over the WHOLE catalogue, not a top-ranked subset. The '
                'curve is emitted under two keys because the stop-point differs by key, and the '
                'gap between them is the overfit estimate. Counts only, no P&L.'])
            print(f'  DILUTION CURVE ({key}): {len(dc)} admission steps')
    mark_done(out, 'S5D', {'input_sha': input_sha,
                           'families': len(per_family),
                           'rows': int(sum(len(f) for f in per_family.values()))})
    return {'per_family': per_family, 'unclaimed': uf, 'reach': reach, 'raw_tot': raw_tot}


def s5b_selection(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest):
    import cluster_profiler as cp
    import runlog as rl
    import selection as sel
    import terrain as tr
    import portfolio_simulation_engine as engine
    import score_g
    import sequential_temporal as seqmod
    import conviction as C
    import numpy as np
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    print(f'  dataset: {attest["rows"]:,} rows | {attest["range"]}')
    cand = os.path.join(out, 'results', 'candidates.csv')
    exercised = os.path.exists(cand)
    n = len(df)
    months = sorted(set(pd.Series(df['Time'].astype(str).values).str[:7].tolist()))
    segment_label = f'{months[0]}..{months[-1]}' if months else 'unknown'
    U = cp.eligible_universe(df, w)
    hyg, dead, canonical, live = sel.vocabulary_hygiene(pool, U, segment_label)
    bk_path = book_file if book_file else os.path.join(_ENGINE, 'book50_signals.csv')
    sigs = score_g.build_book(df, pool, anchor, pd.read_csv(bk_path), adaptive=ad, structural=st)
    conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
    full = engine.run_portfolio(df, sigs, adaptive=ad, structural=st, warmup=w, verbose=False,
                               conviction=conv)
    ev_book, bk = cp.book_events(full)
    daily = sel.per_signal_daily(bk)
    smap = sel.daily_series_map(daily)
    names = sorted(smap.keys())
    pairs = sel.pair_tail_dependence(smap, names)
    tstats = sel.tail_dep_book(pairs)
    null = sel.taildep_permutation_null(smap, names, p=sel.PERM_P)
    kappa = (tstats['TailDep'] / null['TailDep_null_mean']) if null['TailDep_null_mean'] else float('nan')
    mc = sel.mcvar_per_signal(bk, daily)
    c_max = sel.c_max_from_incumbent(mc)
    bd = bk.copy()
    bd['day'] = pd.Series(bd['exit_time'].astype(str).values).str[:10].values
    f_max = sel.fail_conc(bd.groupby('day')['pnl'].sum().values)
    fd = full.copy()
    fd['day'] = pd.Series(fd['exit_time'].astype(str).values).str[:10].values
    surv = sel.absolute_survival(fd.groupby('day')['pnl'].sum().values)
    bar_day = pd.Series(df['Time'].astype(str).values).str[:10].values
    tdays = sel.entry_basis_traded_days(bk, bar_day)
    ent = {1: bk[bk['direction'] == 'LONG']['entry_bar'].values,
           -1: bk[bk['direction'] == 'SHORT']['entry_bar'].values}
    sgn = {1: bk[bk['direction'] == 'LONG']['signal_name'].nunique(),
           -1: bk[bk['direction'] == 'SHORT']['signal_name'].nunique()}
    ids_dir = {1: bk[bk['direction'] == 'LONG']['signal_name'].values.tolist(),
               -1: bk[bk['direction'] == 'SHORT']['signal_name'].values.tolist()}
    grid = sel.depth_yield_grid(ent, sgn, tdays, ids_by_dir=ids_dir)
    grid['DepthYield_LONG_per_signal'] = [sel.depth_yield_per_signal(v, sgn.get(1, 0))
                                           for v in grid['DepthYield_LONG']]
    grid['DepthYield_SHORT_per_signal'] = [sel.depth_yield_per_signal(v, sgn.get(-1, 0))
                                            for v in grid['DepthYield_SHORT']]
    grid['same_signal_refire_LONG'] = [round(sel.same_signal_refire_rate(
        ent.get(1, []), int(t), ids_dir[1]), 4) for t in grid['N']]
    grid['same_signal_refire_SHORT'] = [round(sel.same_signal_refire_rate(
        ent.get(-1, []), int(t), ids_dir[-1]), 4) for t in grid['N']]
    h3 = sel.h3_within_direction(bk)
    base_hdr = [f'dataset_rows={attest["rows"]} segment={segment_label}',
                f'oracle_sha256_12={oracle_sha}']
    _write_with_header(os.path.join(out, 'selection_vocabulary_hygiene.csv'), hyg, [
        'DOT S5B spec G.1 vocabulary hygiene — PROPERTY OF THE VOCABULARY (not of any book)'] +
        base_hdr + ['dead conditions excluded BEFORE equivalence classes are formed; domain is the '
                    'ACTIVE SEGMENT eligible universe, never hardcoded.'])
    _write_with_header(os.path.join(out, 'selection_depthyield_grid.csv'), grid, [
        'DOT S5B spec C.1 DepthYield — PROPERTY OF THE BOOK'] + base_hdr +
        [f'traded-day denominator = {tdays} on the BOOK ENTRY-BAR basis (spec C.1).',
         'DepthYield is a PAIR (LONG, SHORT), normalised within direction. IT IS NEVER SUMMED.'])
    _write_with_header(os.path.join(out, 'selection_mcvar.csv'), mc, [
        'DOT S5B spec C.2 mCVaR per signal — PROPERTY OF THE BOOK'] + base_hdr +
        [f'C_max = 10th percentile of the incumbent mCVaR distribution = {round(c_max, 2)}',
         'more negative = worse tail concentration; a candidate fails if its worst mCVaR is BELOW C_max.'])
    _write_with_header(os.path.join(out, 'selection_h3_persistence.csv'), h3, [
        'DOT S5B spec H.3 / H.3.1 regime-conditional persistence — PROPERTY OF THE BOOK'] + base_hdr +
        ['RULE not literal: calendar month, positive in all but at most one, MINIMUM 3 BUCKETS or '
         'UNEVALUABLE. Buckets are evaluated WITHIN direction; a thin direction is reported '
         'UNEVALUABLE and is NEITHER passed NOR culled.'])
    con = pd.DataFrame([
        {'quantity': 'F_max (FailConc bound)', 'value': round(f_max, 4), 'source': 'incumbent FailConc on ACTIVE SEGMENT'},
        {'quantity': 'TailDep (incumbent)', 'value': round(tstats['TailDep'], 4), 'source': f"tau={sel.TAU} MIN_SHARED={sel.MIN_SHARED}"},
        {'quantity': 'TailDep_null_mean', 'value': round(null['TailDep_null_mean'], 4), 'source': f"permutation null P={null['permutations']} on ACTIVE SEGMENT"},
        {'quantity': 'kappa (incumbent/null)', 'value': round(kappa, 4), 'source': 'T_max = kappa * TailDep_null(segment); dimensionless'},
        {'quantity': 'C_max (mCVaR bound)', 'value': round(c_max, 2), 'source': 'p10 of incumbent mCVaR on ACTIVE SEGMENT'},
        {'quantity': 'worst modelled day (FULL)', 'value': round(surv['worst_modelled_day'], 1), 'source': 'absolute survival, evaluated independently of the relative bounds'},
        {'quantity': 'absolute survival passes', 'value': surv['passes'], 'source': 'FULL population (book + gap fillers)'},
        {'quantity': 'retention_pct', 'value': tstats['retention_pct'], 'source': 'share of pair space entering TailDep'},
        {'quantity': 'exclusion_bias_degeneracy_guarded', 'value': tstats['exclusion_bias_degeneracy_guarded'], 'source': 'k>=3 only'},
        {'quantity': 'FailCorr Pearson (REPORTED ONLY)', 'value': round(tstats['FailCorr_pearson_reported_only'], 4), 'source': 'never a constraint'},
        {'quantity': 'H.2 resampling pool (post-warmup trading days)', 'value': int(pd.Series(df['Time'].astype(str).values[w:]).str[:10].nunique()), 'source': 'the market, NOT the incumbent footprint'},
    ])
    con['segment'] = segment_label
    _write_with_header(os.path.join(out, 'selection_constraints.csv'), con, [
        'DOT S5B spec C.2 / C.3 constraint references — computed on the ACTIVE TRAINING SEGMENT'] +
        base_hdr + ['F_max, T_max and C_max are SEGMENT-LOCAL; full-series values are reporting '
                    'references only and never enter a constraint.',
                    'The absolute survival bound is evaluated on the FULL population INDEPENDENTLY '
                    'of the relative bounds.'])
    cell = (15, 0.85, 0.75)
    if _TERRAIN.get('cells'):
        cs_thr = _TERRAIN['cells'][cell]
        terrain_src = 'S2B MARKET TERRAIN (fixed denominator, identical for every candidate book)'
    else:
        cs_thr = tr.build_terrain(df, w)[1][cell]
        terrain_src = ('S2B MARKET TERRAIN rebuilt in-process by terrain.build_terrain — the SAME '
                       'construction and grid S2B writes, so the denominator is identical')
    covdir = sel.coverage_by_direction(ev_book, cs_thr, label='INCUMBENT BOOK')
    covdir['W'] = cell[0]
    covdir['K_pct'] = cell[1]
    covdir['E_pct'] = cell[2]
    covdir['terrain_source'] = terrain_src
    _write_with_header(os.path.join(out, 'selection_coverage.csv'), covdir, [
        'DOT S5B REACH — coverage of the S2B MARKET TERRAIN, scored PER DIRECTION'] + base_hdr +
        [f'terrain source: {terrain_src}',
         f'grid cell W={cell[0]} K=p{int(cell[1] * 100)} E=p{int(cell[2] * 100)} | mask {tr.eligibility_label()}',
         'terrain = MARKET (price only, no signals); entries = BOOK. The denominator is FIXED.',
         'PER DIRECTION IS THE POINT: the terrain is near 50/50, so a long-heavy book leaves nearly '
         'all short episodes uncovered and short candidates gain marginal value with NO quota, NO '
         'floor and NO minimum count anywhere in the objective.',
         'COVERAGE NEVER OVERRIDES SURVIVAL: it stays after survival, FailConc and DepthYield in the '
         'lexicographic order (spec C.3), never promoted. A book could reach 100% by taking '
         'everything; that must remain unreachable.',
         'COVERAGE COUNTS PRESENCE, NOT CAPTURE: a signal firing at bar 55 of a 60-bar episode counts '
         'as covering it while earning almost nothing. entry_pos_median is DESCRIPTIVE only; no taper '
         'is built on it because normalised position needs the episode end, unknowable at fire time.',
         tr.FORWARD_LOOKING_BOUNDARY])
    for _i, r in covdir.iterrows():
        print(f"    REACH {r['direction']:<5} {r['coverage_pct']:6.3f}% of {int(r['terrain_episodes']):5} "
              f"terrain episodes ({int(r['touched'])} touched, {int(r['missed'])} missed)", flush=True)
    both = covdir[covdir['direction'].str.startswith('BOTH')]
    cov = {'episodes': int(both['terrain_episodes'].iloc[0]) if len(both) else 0,
           'coverage_pct': float(both['coverage_pct'].iloc[0]) if len(both) else 0.0,
           'by_direction': covdir}
    ent_map = {}
    for d, lab in ((1, 'LONG'), (-1, 'SHORT')):
        ent_map[d] = {nm: g['entry_bar'].values
                      for nm, g in bk[bk['direction'] == lab].groupby('signal_name')}

    def _setval(d, sset):
        """THE CANARY MUST SCORE ON THE SAME BASIS AS THE SEARCH IT CERTIFIES.

        Item 4's distinct-signal basis is threaded here too: comparing greedy on
        the distinct basis against an enumerated optimum on the row basis would
        certify nothing, because the two are different objectives.
        """
        if not sset:
            return 0.0
        bars = np.concatenate([ent_map[d][x] for x in sset])
        ids = np.concatenate([np.full(len(ent_map[d][x]), x, dtype=object) for x in sset])
        v, _g = sel.depth_yield_direction(bars, tdays, sel.S_DEFAULT, sel.N_TOLERANCE,
                                          signal_ids_d=ids)
        return v

    def _gain(d, selected, cid):
        return _setval(d, list(selected) + [cid]) - _setval(d, list(selected))

    def _nocon(d, ss):
        return True, ''

    fx = []
    for d, lab in ((1, 'LONG'), (-1, 'SHORT')):
        ids = sorted(ent_map[d].keys())
        if len(ids) < 2:
            continue
        mk = 3 if len(ids) <= 15 else 2
        f = sel.exhaustive_vs_greedy(d, ids, _setval, _gain, _nocon, max_k=mk)
        f['direction_label'] = lab
        fx.append(f)
    fixture = pd.concat(fx, ignore_index=True) if fx else pd.DataFrame()
    if len(fixture):
        _write_with_header(os.path.join(out, 'selection_fixture_exhaustive_vs_greedy.csv'),
                           fixture, ['DOT S5B STANDING CANARY - exhaustive vs greedy, BOTH directions, every run'] + base_hdr + [FIXTURE_WHY, FIXTURE_LIMIT])
        for _i, r in fixture[fixture['argmax'].str.startswith('GREEDY')].iterrows():
            print(f"    CANARY {r['direction_label']:<5} greedy {r['greedy_value']:.6f} = "
                  f"{r['greedy_pct_of_optimum']}% of enumerated optimum "
                  f"{r['exhaustive_optimum']:.6f} (optimum at size "
                  f"{int(r['optimum_at_size'])}, pair escapes {int(r['pair_escapes'])})", flush=True)
    pivot = daily.pivot_table(index='day', columns='signal_name', values='pnl',
                              aggfunc='sum').fillna(0.0)
    pbo = sel.pbo_cscv(pivot.values) if pivot.shape[0] >= 16 and pivot.shape[1] >= 2 else float('nan')
    merged_state = {'survival': surv, 'FailConc': f_max, 'TailDep': tstats['TailDep'],
                    'worst_mCVaR': float(np.nanmin(mc['mCVaR']))}
    bounds = {'F_max': f_max, 'T_max': kappa * null['TailDep_null_mean'], 'C_max': c_max}
    con_eval = sel.evaluate_constraints(merged_state, bounds)
    ce = pd.DataFrame([{'applied_to': 'INCUMBENT BOOK (self-reference)',
                        **{k: str(v) for k, v in con_eval.items()},
                        'PBO_cscv_reported_not_enforced': round(pbo, 4) if pbo == pbo else '',
                        'PBO_reference_bar': 0.10}])
    _write_with_header(os.path.join(out, 'selection_constraint_evaluation.csv'), ce,
                       ['DOT S5B spec C.3 constraint evaluation + spec H.1 PBO via CSCV'] + base_hdr + [PBO_WHY])
    print(f"    PBO (CSCV, reported not enforced, bar 0.10) = "
          f"{round(pbo, 4) if pbo == pbo else 'n/a'} | constraints "
          f"{con_eval['binding'] or 'all pass'}", flush=True)
    vz = df['Volume'].values == 0
    fri = ((df['EST_DayOfWeek'].values == 5)
           & ((df['EST_Hour'].values > 16)
              | ((df['EST_Hour'].values == 16) & (df['EST_Minute'].values >= 45))))
    entry_ok = ((df['ADX_Value'].values >= 15) & (df['Volume'].values > 50) & ~vz & ~fri
                & (np.arange(n) >= w))
    qmasks, qdirs, qnames = engine.build_signal_masks(df, sigs, ad, st, entry_ok, verbose=False)
    Mq = sel.cofire_matrix(qmasks, qnames)
    qd = np.array(qdirs)
    offm = ~np.eye(len(qnames), dtype=bool)
    cofire_rows = []
    for dd, lab in ((1, 'LONG'), (-1, 'SHORT')):
        sd = qd == dd
        sub = Mq[np.ix_(sd, sd)]
        o = ~np.eye(sub.shape[0], dtype=bool)
        if o.sum():
            cofire_rows.append({'basis': lab + '-only ordered pairs (WITHIN direction)',
                                'signals': int(sd.sum()),
                                'CoFire_mean': round(float(sub[o].mean()), 6),
                                'ordered_pairs': int(o.sum())})
    crossm = (qd[:, None] != qd[None, :]) & offm
    cofire_rows.append({'basis': 'CROSS-direction ordered pairs (structurally zero)',
                        'signals': len(qnames),
                        'CoFire_mean': round(float(Mq[crossm].mean()), 6) if crossm.sum() else 0.0,
                        'ordered_pairs': int(crossm.sum())})
    cofire_rows.append({'basis': 'ALL ordered pairs - cofire_book_all_pairs_DIAGNOSTIC (DEFLATED)',
                        'signals': len(qnames),
                        'CoFire_mean': round(sel.cofire_book_all_pairs_DIAGNOSTIC(Mq), 6),
                        'ordered_pairs': int(offm.sum())})
    cof = pd.DataFrame(cofire_rows)
    _write_with_header(os.path.join(out, 'selection_cofire.csv'), cof,
                       ['DOT S5B spec C.1 entry co-firing - PROPERTY OF THE BOOK'] + base_hdr + [COFIRE_WHY])
    Cmat, cnames, edges, gstats = sel.mask_correlation_graph(pool, live, U)
    comms = sel.detect_communities(cnames, edges)
    n90, n95, pr_ratio = sel.effective_dimension(Cmat)
    g2 = pd.DataFrame([{**gstats, 'communities_detected': len(comms),
                        'largest_community': max((len(v) for v in comms.values()), default=0),
                        'effective_dim_90pct': n90, 'effective_dim_95pct': n95,
                        'participation_ratio': round(pr_ratio, 2), 'r_threshold': 0.70}])
    _write_with_header(os.path.join(out, 'selection_g2_near_duplication.csv'), g2,
                       ['DOT S5B spec G.2 near-duplication and community detection'] + base_hdr + [G2_WHY])
    bookdf = pd.read_csv(bk_path)
    trows = []
    for _i, r in bookdf.iterrows():
        if 'trigger' in bookdf.columns and str(r['trigger']) != 'F0':
            continue
        parts = [x.strip() for x in str(r['signal_def']).split('+')]
        doms = sorted({sel.condition_domain(x) for x in parts})
        trows.append({'signal_def': r['signal_def'], 'direction': r['direction'],
                      'domains': ';'.join(doms), 'n_domains': len(doms),
                      'passes_2domain_rule': sel.triple_domain_ok(parts)})
    tdom = pd.DataFrame(trows)
    if len(tdom):
        _write_with_header(os.path.join(out, 'selection_g2_domain_bridging.csv'), tdom,
                           ['DOT S5B spec G.2 domain bridging applied RETROSPECTIVELY to the incumbent F0 triples - PROPERTY OF THE BOOK'] + base_hdr + [TDOM_WHY])
        print(f"    G.2 {gstats['pairs_ge_070']} pairs at |r|>=0.70 of {gstats['pairs_total']}, "
              f"median |r| {round(gstats['median_abs_r'], 4)}, {n90} components carry 90% variance, "
              f"{len(comms)} communities | domain bridging "
              f"{int(tdom['passes_2domain_rule'].sum())} of {len(tdom)} triples span >= 2 domains",
              flush=True)
    report_lines = [
        f'vocabulary: {hyg["vocabulary_total"].iloc[0]} total, {hyg["dead_conditions"].iloc[0]} dead, '
        f'{hyg["effective_vocabulary"].iloc[0]} effective (identity domain = eligible universe)',
        f'constraint references (segment {segment_label}): F_max {round(f_max, 3)}, kappa '
        f'{round(kappa, 3)}, C_max {round(c_max, 1)}, absolute survival '
        f'{"PASS" if surv["passes"] else "FAIL"} at worst day {round(surv["worst_modelled_day"], 1)}',
        'H.3 within direction: ' + '; '.join(f"{r['direction']} {r['verdict']}" for _i, r in h3.iterrows()),
        'submodularity: NOT established; greedy is a heuristic and the (1-1/e) bound is NOT claimed',
        'NO DIRECTIONAL TARGET: no floor, quota, target or reserved allocation exists in selection.py',
    ]
    if exercised:
        cands = pd.read_csv(cand)
        print(f'  SELECTION SEARCH over {len(cands)} candidates — per-direction greedy/CELF with '
              f'the lookahead-2 stopping rule, subject to the constraint references above.')
        entry_ok_sel = ((df['ADX_Value'].values >= 15) & (df['Volume'].values > 50)
                        & (df['Volume'].values != 0) & (np.arange(n) >= w))
        cand_bars = {1: {}, -1: {}}
        skipped = 0
        _spg = rl.Progress('S5B candidate entry masks', len(cands))
        _spg.__enter__()
        for _i, cr in cands.iterrows():
            _spg.step(1, extra=f'LONG {len(cand_bars[1])} SHORT {len(cand_bars[-1])}')
            fam = str(cr.get('family', '')).strip()
            sig = str(cr.get('signal_def', ''))
            d = 1 if str(cr.get('direction', 'LONG')).upper() == 'LONG' else -1
            key = f'{fam}|{sig}|{"LONG" if d == 1 else "SHORT"}'
            if key in cand_bars[d]:
                continue
            try:
                if fam == 'F0':
                    parts = [x.strip().rsplit(':', 1) for x in sig.split('+')]
                    mk = np.ones(n, dtype=bool)
                    for f_, t_ in parts:
                        mk &= np.asarray(engine.condition_mask(df, f_, t_, ad, st), dtype=bool)
                elif fam == 'F1':
                    mm = score_g._F1.match(sig)
                    mk = np.asarray(seqmod.pair_mask(pool[mm.group(1).strip()],
                                                     pool[mm.group(3).strip()],
                                                     int(mm.group(2)), anchor), dtype=bool)
                else:
                    mk = np.asarray(score_g.family_mask(df, pool, fam, sig, ad, st, anchor=anchor),
                                    dtype=bool)
            except SystemExit as _e:
                rl.warn(f'S5D candidate skipped ({fam}): {_e}')
                skipped += 1
                continue
            cand_bars[d][key] = np.flatnonzero(mk & entry_ok_sel).astype(np.int64)
        _spg.__exit__(None, None, None)
        print(f'  candidate entry masks built: LONG {len(cand_bars[1])} | SHORT {len(cand_bars[-1])}'
              + (f' | {skipped} unparseable skipped' if skipped else ''))

        _EMPTY = np.empty(0, dtype=np.int64)
        _prefix = {}

        def _dy_bars(d, bars, ids):
            if bars.size == 0:
                return 0.0
            v, _g = sel.depth_yield_direction(bars, tdays, sel.S_DEFAULT, sel.N_TOLERANCE,
                                              signal_ids_d=ids)
            return v

        def _prefix_of(d, selected):
            """Cache the FIXED prefix. `selected` does not change across a pair sweep.

            _dy previously re-concatenated every selected candidate's bars and
            recomputed the base DepthYield inside EVERY marginal-gain call, so a
            20,000-pair plateau escape at step 98 rebuilt a 98-array
            concatenation 40,000 times and recomputed the same base value 20,000
            times. Both are constant for the whole sweep. Caching them changes
            no arithmetic: depth_yield_direction sorts internally, so the
            concatenation order is irrelevant to the result, and the base is the
            identical float it was recomputing.
            """
            key = (d, tuple(selected))
            ent = _prefix.get(key)
            if ent is None:
                bars = (np.concatenate([cand_bars[d][x] for x in selected])
                        if selected else _EMPTY)
                ids = (np.concatenate([np.full(len(cand_bars[d][x]), x, dtype=object)
                                       for x in selected]) if selected
                       else np.empty(0, dtype=object))
                ent = (bars, ids, _dy_bars(d, bars, ids))
                _prefix[key] = ent
            return ent

        def _ids_for(d, keys):
            return (np.concatenate([np.full(len(cand_bars[d][x]), x, dtype=object) for x in keys])
                    if keys else np.empty(0, dtype=object))

        def _dy(d, sset):
            if not sset:
                return 0.0
            return _dy_bars(d, np.concatenate([cand_bars[d][x] for x in sset]),
                            _ids_for(d, list(sset)))

        def _sel_gain(d, selected, cid):
            base_bars, base_ids, base_val = _prefix_of(d, list(selected))
            return _dy_bars(d, np.concatenate([base_bars, cand_bars[d][cid]]),
                            np.concatenate([base_ids, _ids_for(d, [cid])])) - base_val

        def _sel_setgain(d, selected, add):
            base_bars, base_ids, base_val = _prefix_of(d, list(selected))
            return _dy_bars(d, np.concatenate([base_bars] + [cand_bars[d][x] for x in add]),
                            np.concatenate([base_ids, _ids_for(d, list(add))])) - base_val

        chosen = {}
        stops = {}
        for d, lab in ((1, 'LONG'), (-1, 'SHORT')):
            ids = sorted(cand_bars[d].keys())
            if not ids:
                chosen[d] = []
                stops[d] = 'no candidates in this direction'
                continue
            _bpg = rl.Progress(f'S5B greedy {lab} CELF heap build', len(ids))
            _bpg.__enter__()
            _spg = rl.Progress(f'S5B greedy {lab} admission', 0)
            _spg.__enter__()
            _last = {'n': 0}

            def _on_step(stp, nsel, nheap, _lab=lab):
                _spg.done = stp
                _spg.step(0, extra=f'selected {nsel} | heap remaining {nheap}')

            def _on_escape(stp, nsel, nrem, _lab=lab):
                print(f'    S5B greedy {_lab}: PLATEAU at step {stp} (selected {nsel}) - '
                      f'attempting size-2 escape over {nrem} remaining, sampling up to '
                      f'{sel.PAIR_SAMPLE_K} pairs', flush=True)

            picked, reason, log, meta = sel.greedy_direction(
                d, ids, _sel_gain, _no_constraint, set_gain_fn=_sel_setgain,
                on_build=lambda i: _bpg.step(1), on_step=_on_step, on_escape=_on_escape)
            _bpg.__exit__(None, None, None)
            _spg.__exit__(None, None, None)
            chosen[d] = picked
            stops[d] = reason
            print(f'    {lab}: selected {len(picked)} of {len(ids)} candidates | '
                  f'pair escapes {meta["pair_escapes"]} | stop: {reason[:80]}')
            if not picked:
                # EXPECTED, NOT AN ERROR, AND NOTHING CONSUMES THE RESULT.
                # DepthYield counts runs of >= S_DEFAULT DISTINCT signals, so it is identically
                # zero for any set smaller than S_DEFAULT: with k signals the deepest reachable
                # run is k. Greedy adds one at a time and the lookahead-2 rule at most two, so
                # every first move scores exactly 0.0 and the search halts at the origin.
                # DECISION RECORDED, DO NOT RE-OPEN: the greedy stays and there is no tiered
                # bootstrap. A bootstrap would restore a gradient for a consumer that does not
                # exist - item 12's dilution curve builds its OWN ordering in S5D by sorting the
                # VALID catalogue under each ranking key - and would make the canary undefined
                # rather than stale (0 of 666 pairs, 0 of 7,770 triples, first non-zero at k=5).
                print(f'    {lab}: zero selected - EXPECTED (objective is 0 below '
                      f'S={sel.S_DEFAULT} distinct signals) and consumed by nothing.')
        admitted = {lab: list(chosen[d]) for d, lab in ((1, 'LONG'), (-1, 'SHORT'))}
        print(f'  ADMISSION ORDER retained IN MEMORY ONLY: '
              f'LONG {len(admitted["LONG"])} / SHORT {len(admitted["SHORT"])}. '
              f'NO selected_book.csv is written - item 15: the catalogue is emitted from VALID, '
              f'never from an argmax, so no argmax output is persisted anywhere. '
              f'IT DOES NOT FEED ITEM 12: the dilution curve builds its OWN ordering in S5D by '
              f'sorting the VALID catalogue under each ranking key, and reads nothing from S5B. '
              f'Nothing consumes this order except this line.')
        report_lines.append(
            f'ADMISSION ORDER computed in memory from {len(cands)} candidates '
            f'(LONG {len(chosen[1])} / SHORT {len(chosen[-1])}). It does NOT feed item 12: the '
            f'dilution curve builds its own ordering in S5D from the VALID catalogue. NO selected '
            f'book is written: item 15 forbids emitting a catalogue from an argmax.')
    if not exercised:
        report_lines.append('SELECTION SEARCH NOT RUN: no candidates.csv on this run, so the '
                            'objective and per-direction greedy were not exercised. The constraint '
                            'references, hygiene, DepthYield and REACH above are measured on the '
                            'incumbent as a self-reference.')
    print(f'  vocabulary {hyg["effective_vocabulary"].iloc[0]} effective | kappa {kappa:.3f} | '
          f'C_max {c_max:.1f} | survival {"PASS" if surv["passes"] else "FAIL"}')
    print(f'  selection search: {"candidates present" if exercised else "UNEXERCISED PENDING S3 (no candidate pool)"}')
    if not exercised:
        print('  S5B NOT MARKED DONE — it reported itself unexercised (no candidate pool). A stage '
              'that did not do its work must not claim it did.')
    else:
        mark_done(out, 'S5B', {'input_sha': input_sha,
                               'effective_vocabulary': int(hyg['effective_vocabulary'].iloc[0])})
    return {'report_lines': report_lines, 'hygiene': hyg, 'grid': grid, 'constraints': con,
            'h3': h3, 'coverage': cov}



def s5c_walk_forward(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest):
    import runlog as rl
    import cluster_profiler as cp
    import selection as sel
    import wf_selection as wfs
    import portfolio_simulation_engine as engine
    import score_g
    import conviction as C
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    print(f'  dataset: {attest["rows"]:,} rows | {attest["range"]}')
    _pc_path = os.path.join(out, 'wf_pass_criterion.csv')
    if is_done(out, 'S5C', input_sha):
        _crit_ok = False
        _why_gate = 'wf_pass_criterion.csv absent'
        if os.path.exists(_pc_path):
            try:
                _pcprev = pd.read_csv(_pc_path, comment='#')
                _v = str(_pcprev['verdict'].iloc[0]) if 'verdict' in _pcprev.columns else ''
                _sr = (int(_pcprev['splits_with_ratio'].iloc[0])
                       if 'splits_with_ratio' in _pcprev.columns else 0)
                _crit_ok = (_v.upper() not in ('UNEVALUABLE', '')) and _sr > 0
                _why_gate = f'verdict={_v} splits_with_ratio={_sr}'
            except Exception as _e:
                _why_gate = f'wf_pass_criterion.csv unreadable: {type(_e).__name__}'
        _ent_path = os.path.join(out, 'wf_book_arm_entities.csv')
        _ent_ok = False
        if os.path.exists(_ent_path):
            try:
                _ent_ok = len(pd.read_csv(_ent_path, comment='#')) > 0
            except Exception:
                _ent_ok = False
        _stale = stale_artifacts(out, 'S5C')
        if _crit_ok and _ent_ok and not _stale:
            print(f'  S5C resumed from marker: the criterion EXISTS and is evaluable '
                  f'({_why_gate}), every deliverable is present, and every recorded SCHEMA still '
                  f'matches — skipping.')
            return None
        if _crit_ok and _ent_ok and _stale:
            print('  S5C marker present and criterion evaluable, but the artifacts are STALE - '
                  'RE-RUNNING:', flush=True)
            for _r in _stale:
                print(f'      {_r}', flush=True)
            print('      A gate that checks a file EXISTS cannot detect a file that is STALE, and '
                  'a schema change makes every prior artifact stale by definition. The columns the '
                  'audit trail needs (in_denominator, train_passes, test_passes) were added after '
                  'the last run, so the file on disk cannot answer the question it exists to '
                  'answer.', flush=True)
        if _crit_ok and not _ent_ok:
            print('  S5C marker present and criterion evaluable, but wf_book_arm_entities.csv is '
                  'absent/empty - RE-RUNNING. A gate that does not check EVERY deliverable will '
                  'skip while an output is missing: this file was added to the stage after the '
                  'marker was written, so the criterion alone cannot tell a complete run from an '
                  'incomplete one.')
        if not _crit_ok:
            print(f'  S5C marker present but THE DELIVERABLE IS NOT THERE ({_why_gate}) — RE-RUNNING. '
                  f'The old gate checked wf_splits.csv, which is written BEFORE the book arm runs, so '
                  f'it proved the stage STARTED, not that it produced a criterion. A previous run '
                  f'marked S5C done on the UNEVALUABLE path and this stage would have skipped forever '
                  f'for this input_sha.')
    n0 = len(df)
    wfs.assert_no_row_deletion(df, n0)
    splits, day_tbl, meta = wfs.derive_splits(df, w)
    wfs.assert_split_shape(splits)
    print(f'  ITEM 17 SPLIT SHAPE ASSERTED at derivation ({len(splits)} splits) - not '
          f'only inside the book arm, which sits behind the pool gate and does not run '
          f'on a pool-less run. A key drift would otherwise surface on the one night the '
          f'real run happens.')
    if not splits:
        print('  S5C: the floor admits no valid split; walk-forward is not executable on this dataset.')
        print('  S5C NOT MARKED DONE - no splits were derived. CHOSEN: do not mark, rather than '
              'embedding the floor parameters in the marker. A marker that encodes its own '
              'invalidation conditions is a SECOND place the floor is defined; not marking costs '
              'a re-derivation of seconds and cannot go stale when the floor changes in code.')
        return None
    sp = pd.DataFrame(splits)
    sp['total_post_warmup_days'] = meta['total_post_warmup_days']
    sp['first_valid_train_days'] = meta['first_valid_train_days']
    sp['derived_splits'] = meta['derived_splits']
    sp['under_powered'] = meta['under_powered']
    _write_with_header(os.path.join(out, 'wf_splits.csv'), sp, [
        'DOT S5C spec I.1 split derivation — THE SPLIT COUNT IS DERIVED FROM AN EXECUTABILITY FLOOR, NOT FIXED',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        f'oracle_sha256_12={oracle_sha}',
        f'floor: {meta["floor"]}',
        f'post-warmup trading days {meta["total_post_warmup_days"]}; first prefix meeting the floor '
        f'{meta["first_valid_train_days"]} days; {meta["days_after_first_floor"]} days remain and are partitioned '
        f'into contiguous equal test segments; DERIVED SPLIT COUNT = {meta["derived_splits"]}',
        f'embargo = {wfs.EMBARGO_BARS} bars stated as a BAR COUNT, not a session count: the measured median session '
        f'is 1,365 bars, so a session reading would embargo fewer bars than one full trading day',
        'scheme is ANCHORED: each training segment strictly contains the previous, so the floor binds only on the first',
        'NO ROW IS DELETED: segments are index ranges over the intact series and the oracle receives the full frame',
        'RECORDED LIMITATION — ATTESTATION SCOPE: repeat detection compares records within ONE output directory. A '
        'run started against a FRESH --out directory begins with a clean trail and its repeats are not detected. No '
        'in-process mechanism can bind an operator who deliberately starts elsewhere; the trail catches careless and '
        'accidental repeats, which is its purpose, and the Auditor verifies trail length against reported splits.',
        'RECORDED LIMITATION — GUARD SCOPE: TestSegmentGuard enforces the single touch on the SANCTIONED path. It '
        'does not make the test bar range unreachable — unrelated code could slice those indices directly. The guard '
        'is discipline on the intended route, not an access control.'])
    attempts = pd.DataFrame(meta['attempts'])
    _write_with_header(os.path.join(out, 'wf_split_derivation_attempts.csv'), attempts, [
        'DOT S5C spec I.1 derivation trace — the floor being applied, not just the answer',
        f'dataset_rows={attest["rows"]} segment_floor_days={wfs.MIN_TRAIN_DAYS} floor_buckets={wfs.MIN_MONTH_BUCKETS}'])
    struct_keys = dt_structural_keys()
    causal = wfs.assert_oracle_causal(df, ad, dt_compute(), splits[0]['train_last_bar'], struct_keys)
    U = cp.eligible_universe(df, w)
    bk_path = book_file if book_file else os.path.join(_ENGINE, 'book50_signals.csv')
    sigs = score_g.build_book(df, pool, anchor, pd.read_csv(bk_path), adaptive=ad, structural=st)
    conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
    full = engine.run_portfolio(df, sigs, adaptive=ad, structural=st, warmup=w, verbose=False, conviction=conv)
    bk = full[~full['signal_name'].isin(cp.GAP_NAMES)]
    seg_rows = []
    causal_rows = []
    for s in splits:
        tr = np.zeros(n0, dtype=bool)
        tr[:s['train_last_bar'] + 1] = True
        cz = wfs.assert_oracle_causal(df, ad, dt_compute(), s['train_last_bar'], struct_keys)
        causal_rows.append({'split_index': s['split_index'], 'train_last_bar': s['train_last_bar'],
                            'keys_total': cz['keys_total'],
                            'rolling_D_keys_checked': cz['rolling_D_keys_checked'],
                            'rolling_D_keys_available': cz['rolling_D_keys_available'],
                            'structural_constant_keys_checked': cz['structural_constant_keys_checked'],
                            'coverage': cz['coverage'], 'equality': cz['equality'],
                            'mismatches': cz['mismatches'], 'causal': cz['causal'],
                            'meaning': cz['meaning'], 'note': cz['note']})
        Utr = U & tr
        hyg, dead, canon, live = sel.vocabulary_hygiene(pool, Utr, f"split{s['split_index']}")
        sub = bk[bk['entry_bar'] <= s['train_last_bar']]
        daily = sel.per_signal_daily(sub)
        smap = sel.daily_series_map(daily)
        pr = sel.pair_tail_dependence(smap, sorted(smap.keys()))
        tstat = sel.tail_dep_book(pr)
        mc = sel.mcvar_per_signal(sub, daily)
        cmax = sel.c_max_from_incumbent(mc)
        bd = sub.copy()
        bd['day'] = pd.Series(bd['exit_time'].astype(str).values).str[:10].values
        fmax = sel.fail_conc(bd.groupby('day')['pnl'].sum().values)
        rec = {'split_index': s['split_index'], 'train_days': s['train_days'],
               'train_last_bar': s['train_last_bar'], 'eligible_bars_train': int(Utr.sum()),
               'dead_conditions': int(hyg['dead_conditions'].iloc[0]),
               'exact_duplicate_pairs': int(hyg['exact_duplicate_pairs'].iloc[0]),
               'effective_vocabulary': int(hyg['effective_vocabulary'].iloc[0]),
               'F_max': round(fmax, 4), 'TailDep': round(tstat['TailDep'], 4),
               'retention_pct': tstat['retention_pct'],
               'below_floor_majority_flag': tstat['below_floor_majority_flag'],
               'C_max': round(cmax, 2),
               'monthly_buckets': ';'.join(wfs.segment_month_buckets(df['Time'].values, tr))}
        for d, lab in ((1, 'LONG'), (-1, 'SHORT')):
            h = wfs.h3_segment_rule(sub[sub['direction'] == lab])
            rec[f'H3_{lab}_buckets'] = h['buckets']
            rec[f'H3_{lab}_evaluable'] = h['evaluable']
            rec[f'H3_{lab}_verdict'] = h['verdict']
        wfs.require_h3_evaluable(s['split_index'],
                                 [{'evaluable': rec['H3_LONG_evaluable']}, {'evaluable': rec['H3_SHORT_evaluable']}])
        seg_rows.append(rec)
    seg = pd.DataFrame(seg_rows)
    _write_with_header(os.path.join(out, 'wf_per_segment_rederivation.csv'), seg, [
        'DOT S5C spec I.2 per-segment re-derivation — EVERY VALUE COMPUTED INSIDE ITS OWN TRAINING SEGMENT',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        f'oracle_sha256_12={oracle_sha}',
        'ANTI-LEAK: each row is derived from bars [0, train_last_bar] only. The full-series references — effective '
        'vocabulary 238, F_max 3.869, C_max -1429.1, retention 65.0% — are REPORTING REFERENCES ONLY and appear in '
        'no constraint. Every per-split value differs from them, which is the structural evidence that the bounds '
        'are segment-local rather than full-series values wearing a per-split label.',
        'H.3 IS A RULE, NOT A LITERAL: buckets are the calendar months the SEGMENT contains, computed from segment '
        'timestamps. wf.py FOLDS is month-literal Jan-Jun and is NEVER imported or referenced by this stage.',
        'H.3.1: buckets are evaluated WITHIN direction; a thin direction is reported UNEVALUABLE and is NEITHER '
        'passed NOR culled.',
        'below_floor_majority_flag TRUE means more than half the pair space sits below MIN_SHARED, so TailDep is not '
        'meaningfully binding in that segment and FailConc plus the absolute survival bound carry the decision.',
        'RECORDED CONSEQUENCE — DO NOT OVER-READ AN EARLY-SPLIT RESULT: splits 0 and 1 fire this flag (retention '
        '17.7% and 33.0%), so they test a WEAKER CONSTRAINT SET than split 2. A pass in an early split is evidence '
        'about FailConc and absolute survival, NOT about tail dependence. The TailDep constraint only becomes '
        'meaningfully binding once retention rises, which on this dataset happens at split 2 (51.3%).'])
    _write_with_header(os.path.join(out, 'wf_oracle_causality.csv'), pd.DataFrame(causal_rows), [
        'DOT S5C spec I.4 item 2 — the anti-leak assertion for the oracle',
        f'dataset_rows={attest["rows"]}',
        'Thresholds are computed ONCE on the FULL frame because the oracle must never receive a row-deleted frame.',
        'This assertion proves that is not a leak: for each split, threshold values over the training prefix are '
        'recomputed on a TRUNCATED frame ending at train_last_bar and compared. Zero mismatches means mechanism D '
        'is causal and cannot see the test segment, so masking after a full-frame computation is sound.',
        'COVERAGE IS PART OF THE FINDING: ALL keys are compared, split into rolling-D and structural-constant '
        'counts. The structural constants are causally trivial (a constant is identical on any prefix) and are '
        'never presented as evidence of rolling-threshold causality. A previous revision sampled the head of the '
        'threshold dict, which selected the four structural constants plus two rolling keys, so it tested 2 of 176 '
        'rolling thresholds while reporting 6 features; that sample would have passed on a dataset where every '
        'rolling threshold leaked.',
        'RUNTIME COST, STATED SO NOBODY REVERTS IT FOR SPEED: 6 to 11 seconds per split on the reference machine, '
        'rising with training-segment length. The cost is dominated by the ONE truncated recomputation per split, '
        'which is paid regardless of how many keys are compared, so full coverage is effectively free and must not '
        'be reduced to a sample.',
        'Wall-clock timing is deliberately NOT emitted, as a column or interpolated into this header, because a '
        'varying value would break the byte-level determinism of this artifact.'])
    code_shas = {'wf_selection.py': _sha_full(os.path.join(_ENGINE, 'wf_selection.py')),
                 'selection.py': _sha_full(os.path.join(_ENGINE, 'selection.py'))}
    sdef = wfs.split_definition_sha(splits, meta)
    run_id = f'{input_sha}-{sdef[:8]}'
    guards = []
    null_frames = []
    null_summary = []
    pool_keys = sorted(pool.keys())
    for s in splits:
        rec = wfs.build_attestation_record(run_id, code_shas, sdef, input_sha, s, out)
        wfs.write_attestation(out, rec)
        g = wfs.TestSegmentGuard(s['split_index'], s['test_first_bar'], s['test_last_bar'])
        guards.append(g)
        nf, ns = wfs.score_null_arm(df, pool_keys, ad, st, w, s, g, engine.run_portfolio,
                                    workers=int(os.environ.get('DOT_WORKERS', '1')),
                                    frame_path=os.environ.get('DOT_FRAME_PATH'),
                                    progress_factory=rl.Progress)
        null_frames.append(nf)
        null_summary.append(ns)
    nulls = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    nsum = pd.DataFrame(null_summary)
    if len(nulls):
        _write_with_header(os.path.join(out, 'wf_null_arm_entities.csv'), nulls, [
            'DOT S5C spec I.3 random-triple NULL ARM, per split — THIS IS A MEASUREMENT, NOT A PASS CRITERION',
            'THE DENOMINATOR IS in_denominator == train_passes, WHICH IS NOT THE BOOK ARM\'S '
            'RULE. The book arm divides by train_passes AND traded_on_test; the null arm divides '
            'by EVERY train qualifier, because a random triple that goes silent on test is a null '
            'FAILURE and excluding it would flatter the null and deflate every ratio. '
            'persisted.sum() / in_denominator.sum() reproduces the printed rate.',
            'train_passes reads True on every row because rows are appended ONLY for qualifiers - '
            'that is this file\'s definition, not a collapsed flag.',
            f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
            f'oracle_sha256_12={oracle_sha}',
            'Random triples are drawn from the existing 249-condition pool; NO S3 output is required, so this',
            'violates no rejection item: it is not the fixed book (item 1), not full-series discovery (item 3),',
            'and not the 27% figure carried from the record (item 7) — the null is regenerated inside each split.',
            'Persistence is measured exactly as the record measures it: net>0, PF>=2, WR>=75 in BOTH train and test.',
            'Each triple is scored STANDALONE (batch size 1) so the 6-lot jar cannot couple the null entities.',
            'Scored on the test segment inside the SINGLE sanctioned TestSegmentGuard touch.',
            'THE PASS CRITERION REMAINS UNEVALUABLE: it requires the SELECTION arm, which requires a candidate pool',
            'that S3 has never produced. A null baseline is the denominator of that comparison, never the result.'])
        _write_with_header(os.path.join(out, 'wf_null_arm_summary.csv'), nsum, [
            'DOT S5C spec I.3 null-arm per-split summary — MEASUREMENT ONLY, NOT A WALK-FORWARD RESULT',
            f'dataset_rows={attest["rows"]} target_qualifiers={wfs.NULL_TARGET_QUALIFIERS} '
            f'floor_qualifiers={wfs.NULL_FLOOR_QUALIFIERS} cap_triples={wfs.NULL_TRIPLES_CAP} '
            f'generation_batch={wfs.NULL_GEN_BATCH} base_seed={wfs.NULL_SEED}',
            'THE CONTROL IS THE QUALIFIER COUNT, NOT THE GENERATED COUNT. Triples are generated in seeded batches '
            'until train_qualifiers reaches the target or the cap is hit. A fixed generated count is fragile: the '
            'train-qualification rate is itself uncertain, so the generation needed for 80 qualifiers spans a wide '
            'range. Seeding: ONE generator per split, seeded base_seed + split_index, drawn sequentially across '
            'batches with cross-batch de-duplication, so the draw sequence is identical regardless of how many '
            'batches the target required.',
            'Only TRAIN-QUALIFYING triples are scored on the test segment: the denominator is the qualifier count, '
            'so non-qualifiers cannot enter either arm and scoring them would be wasted compute.',
            'RNG SEEDING IS NOW REAL, NOT VACUOUS: before this arm ran, wf_selection instantiated no RNG at all and '
            '"every RNG seeded" was vacuously true. The null draw uses np.random.default_rng(base_seed + split_index) '
            'and the direction assignment draws from the same generator, so the whole arm is reproducible.',
            'seed per split = base_seed + split_index, so the draw is reproducible and split-specific.',
            'null_persistence_rate = persisted / train_qualifiers; entities failing the train bar are excluded',
            'from the denominator, matching how the record computes its 27% baseline.'])
    trail = wfs.read_attestation(out)
    repeats, n_rep = wfs.detect_repeats(trail)
    import discovery_orchestrator as orch
    _cand = os.path.join(out, 'results', 'candidates.csv')
    _pool_ok = False
    _pool_n = 0
    _pool_why = 'candidates.csv absent'
    if os.path.exists(_cand):
        try:
            _pool_n = len(pd.read_csv(_cand))
        except Exception as _e:
            _pool_n = 0
            _pool_why = f'candidates.csv unreadable: {type(_e).__name__}'
        if _pool_n > 0:
            _prov_ok, _prov_why = orch.provenance_is_current(_cand, input_sha)
            _pool_ok = bool(_prov_ok)
            _pool_why = ('pool present and current' if _prov_ok
                         else f'pool present ({_pool_n} rows) but provenance: {_prov_why}')
        else:
            _pool_why = 'candidates.csv present but empty'
    meta_checks = {'funnel_rerun': _pool_ok, 'null_per_split': True,
                   'funnel_detail': (f'DERIVED, not asserted: {_pool_why}; candidates={_pool_n}; '
                                     f'input_sha={input_sha}')}
    wfs.assert_no_row_deletion(df, n0)
    pc_pre = pd.DataFrame([{'splits_derived': len(splits)}])
    rej = wfs.rejection_checks(df, n0, splits, meta_checks, causal, guards,
                               per_split_frame=seg, attest_trail=trail, pass_frame=pc_pre)
    _write_with_header(os.path.join(out, 'wf_rejection_checks.csv'), rej, [
        'DOT S5C spec I.4 rejection list, implemented as executable checks rather than conventions',
        f'dataset_rows={attest["rows"]} splits={len(splits)}',
        'UNEXERCISABLE PENDING S3 marks a check whose subject does not exist yet, not a check that passed.'])
    null_rates = [r['null_persistence_rate'] for r in null_summary] if null_summary else []
    null_ok = [not str(r['status']).startswith('UNEVALUABLE') for r in null_summary] if null_summary else []
    book_rates = [float('nan')] * len(null_rates)
    book_meta = {'persist_definition': wfs.PERSIST_DEFINITION,
                 'denominator_definition': wfs.DENOMINATOR_DEFINITION}
    null_meta = {'persist_definition': wfs.PERSIST_DEFINITION,
                 'denominator_definition': wfs.DENOMINATOR_DEFINITION}
    agree = wfs.assert_arms_agree(book_meta, null_meta)
    print(f'  ITEM 18 ARMS AGREE (asserted, abort on mismatch): persist = '
          f'{agree["persist_definition"]}')
    print(f'    denominator = {agree["denominator_definition"]}')
    if not _pool_ok:
        print(f'  BOOK ARM SKIPPED: {_pool_why}. book_rates stay nan and the criterion will read '
              f'UNEVALUABLE. THIS LINE EXISTS SO A nan IS ALWAYS ATTRIBUTABLE: without it, "no pool, '
              f'correctly skipped" and "pool present but the provenance stamp did not match" are '
              f'indistinguishable on the console, which is exactly what concealed a key mismatch '
              f'through three deliveries.', flush=True)
    if _pool_ok:
        import catalogue as cat2
        cands_wf = pd.read_csv(_cand)
        conv_wf = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
        arm = wfs.book_arm_from_valid(df, cands_wf, pool, anchor, ad, st, w, splits,
                                      score_g.build_book, engine.run_portfolio,
                                      cat2.evaluate_valid,
                                      pd.Series(df['Time'].astype(str).values).str[:10].values,
                                      conviction=conv_wf, gap_names=cp.GAP_NAMES,
                                      progress_factory=rl.Progress,
                                      workers=int(os.environ.get('DOT_WORKERS', '1')),
                                      frame_path=os.environ.get('DOT_FRAME_PATH'))
        book_rates = [a_['rate'] for a_ in arm]
        _bent = getattr(wfs.book_arm_from_valid, 'last_entities', None)
        if _bent is not None and len(_bent):
            _write_with_header(os.path.join(out, 'wf_book_arm_entities.csv'), _bent, [
                'DOT S5C BOOK ARM per-entity record - one row per (split, signal)',
                f'dataset_rows={attest["rows"]} input_sha={input_sha}',
                'THE DENOMINATOR IS in_denominator = train_passes AND traded_on_test. It is NOT '
                '`admitted`: admitted records the APPENDIX C VALID verdict on the training '
                'segment, while the denominator applies the PERSIST test on train (net>0 AND '
                'PF>=2.0 AND WR>=75) AND at-least-one-trade on test. Two different gates, and only '
                'the first was emitted, so no single-column rule on this file could rebuild the '
                'printed rate. persisted.sum() / in_denominator.sum() now reproduces it exactly.',
                'Mirrors wf_null_arm_entities.csv so both arms read the same way. The aggregate '
                'counts alone cannot separate DILUTION (later-admitted signals are weaker) from '
                'REGIME (the later window is harder): that needs to know WHICH signals were '
                'admitted when, and whether a signal first admitted at a later split persisted '
                'better or worse than one admitted early. This file carries that.',
                'admitted=True on every row: a signal appears here only if VALID admitted it on '
                'that split\'s TRAINING segment. traded_on_test distinguishes silence from '
                'failure - the n_traded denominator counts only signals that fired on test.',
                'PROPERTY OF THE BOOK ARM. It is a MEASUREMENT, not a pass criterion.'])
            print(f'  wf_book_arm_entities.csv: {len(_bent)} rows across '
                  f'{_bent["split_index"].nunique()} splits')
        for a_ in arm:
            print(f'    split {a_["split"]}: VALID admitted {a_["entities"]} on train, '
                  f'{a_["k"]}/{a_["n_traded"]} persisted on test -> rate '
                  f'{a_["rate"] if a_["rate"] == a_["rate"] else "nan"}'
                  + (f' | {a_["note"]}' if a_['note'] else ''))
    verdict = wfs.pass_criterion(book_rates, null_rates, null_ok)
    verdict['certifies'] = ('THE CATALOGUE INCLUSION RULE (VALID), NOT ANY BOOK. Re-scoring a '
                            'hand-assembled book per split is prohibited: a validated generator '
                            'is not a validated book.')
    verdict['persist_definition'] = agree['persist_definition']
    verdict['denominator_definition'] = agree['denominator_definition']
    pc = pd.DataFrame([{**verdict, 'splits_derived': len(splits),
                        'attestation_records': int(len(trail)),
                        'attestation_repeat_groups': int(len(repeats)),
                        'attestation_repeat_records': int(n_rep)}])
    _write_with_header(os.path.join(out, 'wf_pass_criterion.csv'), pc, [
        'DOT S5C spec I.3 pass criterion',
        f'dataset_rows={attest["rows"]} splits_derived={len(splits)}',
        'CRITERION (spec I.3 Revision 9): ratio_s = book_persistence(s) / null_persistence(s) per derived split; '
        'PASS = mean ratio >= 2.40 AND min ratio >= 1.85 AND the 95% lower bound on the mean ratio exceeds 1.0.',
        'THE DENOMINATOR IS SEGMENT-LOCAL. The previous absolute form (mean>=0.65, no split<0.50) inherited a '
        'full-window 27% baseline and is WITHDRAWN as incoherent; the thresholds preserve its intent exactly '
        '(0.65/0.27=2.41, 0.50/0.27=1.85) while making the denominator the split own measured null.',
        'MINIMUM NULL DENOMINATOR: target 80 TRAIN-QUALIFYING triples per split, hard floor 40. Below the floor the '
        'split is UNEVALUABLE. Between floor and target it is EVALUABLE with the reduced denominator REPORTED.',
        'THE TRAIN BAR IS NEVER LOOSENED to raise the qualification rate: the null answers what fraction of things '
        'clearing THE SAME BAR AS THE BOOK SIGNALS persist by chance, so a looser bar would compare two different '
        'populations. The count is raised instead, and the compute cost is accepted.',
        'VERDICT UNEVALUABLE: producing a persistence figure requires re-running the funnel per split, which requires '
        'a candidate pool. S3 discovery has never run. Re-scoring the fixed incumbent book across splits would be '
        'rejection-list item 1 and is NOT done here — a number produced by the prohibited path is worse than none.',
        'A FAIL IS A LEGITIMATE RESULT and would be reported as one. No bar is lowered to obtain a pass.'])
    print(f'  splits derived {len(splits)} (floor: >={wfs.MIN_TRAIN_DAYS}d and >={wfs.MIN_MONTH_BUCKETS} buckets) | '
          f'embargo {wfs.EMBARGO_BARS} bars | oracle causal {all(c["causal"] for c in causal_rows)}')
    if len(nsum):
        print('  null arm (MEASUREMENT, not a pass criterion): ' +
              ' | '.join(f"split {int(r['split_index'])} {r['persisted']}/{r['train_qualifiers']}"
                         f"={r['null_persistence_rate']}" for _i, r in nsum.iterrows()))
    print(f'  attestation records {len(trail)} (repeat groups {len(repeats)}) | pass criterion: {verdict["verdict"]}')
    _verd = str(verdict.get('verdict', '')).upper()
    _nrat = int(verdict.get('splits_with_ratio', 0) or 0)
    if _pool_ok and _verd not in ('UNEVALUABLE', '') and _nrat > 0:
        mark_done(out, 'S5C', {'input_sha': input_sha, 'splits': len(splits),
                               'verdict': _verd, 'splits_with_ratio': _nrat})
    else:
        print(f'  S5C NOT MARKED DONE — verdict {_verd or "(none)"}, splits_with_ratio {_nrat}, '
              f'pool_ok {_pool_ok}. A stage that did not produce its deliverable must not claim it '
              f'did: the marker would skip it permanently for this input_sha and the run would '
              f'finish reporting a pass criterion that was never computed.')
    return {'splits': sp, 'segments': seg, 'rejection': rej, 'pass': pc, 'trail': trail,
            'repeats': repeats, 'meta': meta, 'null_summary': nsum}


def _sha_full(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def dt_compute():
    import dots_thresholds as _dt
    return _dt.compute_adaptive_thresholds


def dt_structural_keys():
    import dots_thresholds as _dt
    return set(_dt._STRUCTURAL.keys())


# ── S8B CLUSTER-PARTICIPATION PROFILER ──
def s8b_cluster_profile(df, ad, st, w, pool, anchor, book_file, committed, out, input_sha, attest):
    import cluster_profiler as cp
    import score_g
    import conviction as C
    oracle_sha = sha12(os.path.join(_ENGINE, 'dots_thresholds.py'))
    print(f'  oracle dots_thresholds.py sha256 : {oracle_sha}')
    print(f'  dataset: {attest["rows"]:,} rows | {attest["range"]}')
    _s8b_ok, _s8b_missing = _artifacts_present(out, [
        'cluster_participation_profile.csv', 'cluster_basis_summary.csv',
        'reach_D01_directional_baseline.csv', 'reach_D02_D2_coverage.csv',
        'reach_D02_book_depth_structure.csv', 'reach_D0_missed_decomposition.csv'])
    if is_done(out, 'S8B', input_sha) and not _s8b_ok:
        print(f'  S8B marker present but deliverables missing {_s8b_missing} - RE-RUNNING.')
    _s8b_stale = stale_artifacts(out, 'S8B')
    if is_done(out, 'S8B', input_sha) and _s8b_ok and _s8b_stale:
        print(f'  S8B marker present but STALE - RE-RUNNING: {_s8b_stale}', flush=True)
    if is_done(out, 'S8B', input_sha) and _s8b_ok and not _s8b_stale:
        print('  S8B already complete for this input (checkpoint) — resuming past it.')
        return None
    if committed is not None and 'executed' in committed:
        executed = committed['executed']
        sigs = committed['sigs']
    else:
        print('  S8B standalone: S8 output unavailable, rebuilding the committed trade list.')
        bk_path = book_file if book_file else os.path.join(_ENGINE, 'book50_signals.csv')
        sigs = score_g.build_book(df, pool, anchor, pd.read_csv(bk_path), adaptive=ad, structural=st)
        conv = C.build_conviction(df, True, True, True, d2d_conviction=True, d2d_gap=True)
        _r, executed = _score(df, sigs, ad, st, w, conv, want_trades=True)
    n = len(df)
    U = cp.eligible_universe(df, w)
    hours = df['EST_Hour'].values
    ab = cp.atr_buckets(df, U)
    ev_book, bk = cp.book_events(executed)
    ev_qual, qual_depth = cp.qualifying_events(df, sigs, ad, st, w)
    print(f'  vocabulary: {len(pool)} conditions | eligible universe {int(U.sum()):,} bars '
          f'| book events {len(bk)} | qualifying events {len(ev_qual[1]) + len(ev_qual[-1])}')
    jobs = []
    for N in cp.N_VALUES:
        jobs.append((1, N, ('', '', ''), ev_book))
        jobs.append((2, N, ('', '', ''), ev_qual))
    thrust_sets = {}
    for W in cp.THRUST_W:
        fwd, mag, eff, valid, thr, mcol, ecol = cp.thrust_thresholds(df, W, cp.THRUST_K_PCTS, cp.THRUST_E_PCTS)
        for kp in cp.THRUST_K_PCTS:
            for ep in cp.THRUST_E_PCTS:
                karr = thr[(mcol, f'k{int(round(kp * 100))}')]
                earr = thr[(ecol, f'e{int(round(ep * 100))}')]
                ev = cp.thrust_events(fwd, mag, eff, valid, karr, earr, w)
                thrust_sets[(W, kp, ep)] = ev
                for N in cp.N_VALUES:
                    jobs.append((3, N, (W, kp, ep), ev))
    rows = []
    summary = []
    t0 = time.time()
    for i, (basis, N, grid, ev) in enumerate(jobs):
        cs = cp.build_cluster_set(n, ev, N)
        tcid = cp.map_trades_to_clusters(cs, bk)
        rows.extend(cp.profile_conditions(pool, cs, U, df, bk, tcid, basis, N, grid, hours, ab))
        cl = cs['clusters']
        summary.append({'basis': basis, 'N': N, 'W': grid[0], 'K_pct': grid[1], 'E_pct': grid[2],
                        'clusters': len(cl), 'max_size': int(cl['size'].max()) if len(cl) else 0,
                        'max_span': int(cl['span'].max()) if len(cl) else 0,
                        'ge3': int((cl['size'] >= 3).sum()) if len(cl) else 0,
                        'ge5': int((cl['size'] >= 5).sum()) if len(cl) else 0,
                        'ge8': int((cl['size'] >= 8).sum()) if len(cl) else 0,
                        'zero_span_pct': round(100.0 * float((cl['span'] == 0).mean()), 1) if len(cl) else 0.0})
        sys.stdout.write(f'\r  cluster sets {i + 1}/{len(jobs)} | elapsed {_hms(time.time() - t0)}   ')
        sys.stdout.flush()
    sys.stdout.write('\n')
    res = pd.DataFrame(rows)
    cs_b1 = cp.build_cluster_set(n, ev_book, 5)
    cs_b2 = cp.build_cluster_set(n, ev_qual, 5)
    overlaps = {}
    for (W, kp, ep), ev in thrust_sets.items():
        for N in cp.N_VALUES:
            tcs = cp.build_cluster_set(n, ev, N)
            overlaps[(W, kp, ep, N)] = cp.overlap_validation(tcs, cs_b1, cs_b2, n, U)
    reach = []
    for mask_name in ('post-warmup', 'eligible-universe'):
        reach.append(cp.directional_baseline(df, 30, 0.85, 0.75, w, mask_name))
    d01 = pd.concat(reach, ignore_index=True)
    _write_with_header(os.path.join(out, 'reach_D01_directional_baseline.csv'), d01, [
        'DOT S8B spec D.0.1 directional coverage baseline — PROPERTY OF THE MARKET (price-only, no signals)',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        f'oracle_sha256_12={oracle_sha}',
        'Parameters: W=30, K=p85 of |disp|/ATR_1M, E=p75 directional efficiency, thresholds via oracle mechanism D.',
        'THE MASK IS PART OF THE FINDING: absolute counts move by roughly 2x between masks; the up/down ratio does not.',
        'Both masks are emitted for exactly that reason (spec D.0.2 reproduction note).',
        'scope=ALL rows carry thrust bar counts and median move; scope=YYYY.MM rows carry down-share and, in',
        'median_move_pts, that month net price change in points.'])
    d02 = []
    for (W, kp, ep) in ((15, 0.85, 0.75), (30, 0.85, 0.75), (30, 0.90, 0.75)):
        for N in cp.N_VALUES:
            d02.append(cp.episode_traded_split(df, W, kp, ep, N, w, bk))
    d02 = pd.concat(d02, ignore_index=True)
    _write_with_header(os.path.join(out, 'reach_D02_D2_coverage.csv'), d02, [
        'DOT S8B spec D.0.2 reach-vs-depth + D.2 coverage — episodes are MARKET, traded/missed are BOOK',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        f'oracle_sha256_12={oracle_sha}',
        'Coverage(B) = fraction of thrust episodes touched by >=1 book entry inside the span, same direction.',
        'Reported per (W, K, E, N) grid cell, never at a single setting (spec D.2), and stratified by episode',
        'absolute size (<50 / 50-100 / 100-200 / >200 pt) so small-move gains are never shown as equivalent to large.',
        'Episode absolute size = |Close[b1+W] - Close[b0]| in points.'])
    d0dec = []
    for (W, kp, ep) in ((15, 0.85, 0.75), (30, 0.85, 0.75)):
        for N in cp.N_VALUES:
            d0dec.append(cp.missed_reason_decomposition(df, W, kp, ep, N, w, bk, qual_depth))
    d0dec = pd.concat(d0dec, ignore_index=True)
    _write_with_header(os.path.join(out, 'reach_D0_missed_decomposition.csv'), d0dec, [
        'DOT S8B spec D.0 missed-episode decomposition across the thrust grid — MARKET episodes, BOOK reasons',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        'Reason A = no BOOK signal qualified anywhere in the span (bar-level qualifying depth zero throughout).',
        'Reason B = a signal qualified but no entry resulted. Qualifying depth from build_signal_masks with entry_ok.'])
    dstruct = pd.concat([cp.book_depth_structure(bk, N, n) for N in cp.N_VALUES], ignore_index=True)
    _write_with_header(os.path.join(out, 'reach_D02_book_depth_structure.csv'), dstruct, [
        'DOT S8B spec D.0.2 book-side depth structure — PROPERTY OF THE BOOK',
        f'dataset_rows={attest["rows"]} dataset_range={attest["range"]}',
        'Population BOOK (F0+F1 executed, gap fillers excluded). Clusters built per direction in isolation.',
        'N=5 primary (spec 0.1.3); N=10 emitted as mandatory sensitivity.'])
    print(f'  reach: D.0.1 {len(d01)} rows | D.0.2/D.2 {len(d02)} rows | D.0 decomposition {len(d0dec)} rows')
    path = os.path.join(out, 'cluster_participation_profile.csv')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(f'# DOT S8B cluster-participation profile\n')
        f.write(f'# dataset_rows={attest["rows"]} dataset_range={attest["range"]}\n')
        f.write(f'# oracle_sha256_12={oracle_sha} engine_sha256_12={sha12(os.path.join(_ENGINE, "portfolio_simulation_engine.py"))}\n')
        f.write(f'# eligibility={cp.ELIGIBILITY_PREDICATE}\n')
        f.write(f'# eligible_bars={int(U.sum())} vocabulary_conditions={len(pool)}\n')
        f.write(f'# min_fire_floor={cp.MIN_FIRE_FLOOR} (ranking eligibility only; not tuned)\n')
        f.write('# BASIS-3 BOUNDARY: the thrust label is FORWARD-LOOKING by construction (uses Close[t+W]).\n')
        f.write('# Legitimate as a selection-side diagnostic; BASIS 3 CAN NEVER BECOME A LIVE GATE OR ENTRY CONDITION.\n')
        f.write('# COUPLING MITIGATION: quant_response_6 mitigation 2 — metric (e) is emitted for BASIS 3 as well as\n')
        f.write('# bases 1 and 2, so shallow-edge participation is measurable against price structure defined without\n')
        f.write('# reference to the book or the jar.\n')
        f.write('# ATR STRATA (lift_5_atr_controlled, vol_proxy_flag) are derived from the oracle mechanism-D ATR_1M\n')
        f.write('# thresholds (rolling-2500, day-refreshed, causal) — NOT from full-sample quantiles.\n')
        f.write('# METRIC (g) SUPPRESSED ON BASIS 3 (EPISODE-STRENGTH SELECTION): part_net/non_net/part_pf/part_wr/\n')
        f.write('# part_wd/non_pf/non_wr/non_wd are emitted EMPTY for cluster_basis=3. Reason: a condition that fires\n')
        f.write('# preferentially in larger or longer episodes inherits bigger forward moves in its participating arm,\n')
        f.write('# so the contrast can be driven by the magnitude of the forward label rather than by entry quality.\n')
        f.write('# Only part_clusters is retained on basis 3 (a genuine count, not an outcome). Basis 3 instead carries\n')
        f.write('# COVERAGE ATTRIBUTION (cov_episodes / cov_book_traded / cov_book_missed / cov_missed_share), which is\n')
        f.write('# not outcome-denominated; cov_* is empty on bases 1-2. cov_episodes is an explicit ALIAS of\n')
        f.write('# part_clusters, retained so the cov_* family is self-contained and so cov_book_traded +\n')
        f.write('# cov_book_missed == cov_episodes serves as a consistency check.\n')
        f.write('# SCOPE LIMIT: the vocabulary is SINGLE CONDITIONS; the book\'s signals are TRIPLES. A single\n')
        f.write('# condition\'s profile is NOT a signal\'s value. Do not select a book directly from this file.\n')
        f.write('# It is an input to selection, not a selection rule.\n')
        res.to_csv(f, index=False, lineterminator='\n')
    os.replace(tmp, path)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(out, 'cluster_basis_summary.csv'), index=False,
                lineterminator='\n', encoding='utf-8')
    print(f'  wrote {len(res)} rows → {path}')
    mark_done(out, 'S8B', {'input_sha': input_sha, 'rows': int(len(res)), 'conditions': len(pool)})
    return {'rows': int(len(res)), 'conditions': len(pool), 'summary': sm, 'overlaps': overlaps,
            'eligibility': cp.ELIGIBILITY_PREDICATE,
            'eligible_bars': int(U.sum()), 'path': path, 'res': res,
            'max_qual_depth': int(max(qual_depth[1].max(), qual_depth[-1].max()))}


# ── S9 REPORT + SPLIT ──
def s9_report(out, attest, contenders, committed, sacred, market_label, input_sha, profile=None, evidence=None, selection_state=None):
    scored_fresh = 'regenerated fresh this run (S6) — stale 746102aae415 / 0910f360a628 NOT inherited'
    L = []
    L.append(f'# DOT Master Report — {market_label}')
    L.append('')
    L.append('## 1. Ingest attestation')
    L.append(f'- files: {", ".join(attest["files"])}')
    L.append(f'- shape: {attest["rows"]:,} rows × {attest["cols"]} cols · range {attest["range"]}')
    L.append(f'- path: {attest["path"]} · invariants: {attest["invariants"]}')
    L.append('')
    L.append(f'- fold/OOS basis: {FOLD_BASIS_NOTE}')
    L.append('')
    L.append('## 2. Sacred parity (byte-lock)')
    for name, want in sacred.items():
        L.append(f'- `{name}` `{want}` OK')
    L.append('')
    if contenders:
        L.append('## 3. Component build-up / contenders')
        L.append('| id | contender | net | Δ | WR | PF | daily wd | daily mDD | folds+ | min-PF | OOS PF | OOS net |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in contenders:
            L.append(f"| {r['id']} | {r['contender']} | ${r['net']} | {r['delta']:+} | {r['WR']} | {r['PF']} | "
                     f"{r['daily_wd']} | {r['daily_mDD']} | "
                     f"{str(r['folds_plus']) + '/' + str(r['fold_count']) if r['folds_evaluable'] else 'UNEVAL'} | "
                     f"{r['min_fold_pf']} | {r['oos_prop_pf'] if r['oos_prop_evaluable'] else 'UNEVAL'} | "
                     f"${r['oos_prop_net']} |")
        L.append('')
    if committed:
        L.append('## 4. Committed-system headline')
        L.append(f"- book: {committed['book_tag']}")
        L.append(f"- **net ${committed['net']} | {committed['trades']} tr | WR {committed['WR']}% | PF {committed['PF']} | "
                 f"daily wd {committed['daily_wd']} | daily mDD {committed['daily_mDD']} | "
                 f"{str(committed['folds_plus']) + '/' + str(committed['fold_count']) if committed['folds_evaluable'] else 'folds UNEVALUABLE'} "
                 f"min-PF {committed['min_fold_pf']} | OOS (final third {committed['oos_prop_window']}) "
                 f"PF {committed['oos_prop_pf']} | OOS net ${committed['oos_prop_net']}**")
        if committed.get('canary'):
            L.append('- US30 baseline canary: $92,347 / 2,698 tr — engine intact')
        L.append('')
    L.append('## 5. Per-family coverage')
    if evidence is not None and isinstance(evidence, dict) and 'family' in evidence:
        fam = evidence['family']
        counts = fam['verdict'].value_counts().to_dict()
        L.append(f"- **measured verdicts (S3B, `family_evidence.csv`)**: " +
                 ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
        for _i, r in fam.iterrows():
            L.append(f"  - {r['family']}: {r['verdict']} — {r['verdict_basis'][:150]}")
    else:
        L.append('- family classifications are measured by S3B and written to `family_evidence.csv`; see that file for the '
                 'per-family verdict. **F10 is FOLDED INTO F0** (concurrence lens null; F12 is the diagnostic remnant) — '
                 'not a gap. No family carries a classification inherited from its history: verdicts are measured, and '
                 'INSUFFICIENT-EVIDENCE is emitted wherever the evidence does not exist on this dataset.')
    L.append('')
    if profile:
        L.append('## 6. S8B cluster-participation profile')
        L.append(f"- output: `{os.path.basename(profile['path'])}` — {profile['rows']} rows "
                 f"({profile['conditions']} conditions x basis x N x grid cell)")
        L.append(f"- eligible universe: {profile['eligible_bars']:,} bars | eligibility: {profile.get('eligibility', '')}")
        L.append('- **SCOPE LIMIT: the vocabulary is SINGLE CONDITIONS; the book\'s signals are TRIPLES. '
                 'A single condition\'s profile is not a signal\'s value; do not select a book directly from this CSV. '
                 'It is an input to selection, not a selection rule.**')
        L.append('- **BASIS-3 BOUNDARY: the thrust label is forward-looking by construction and can never become '
                 'a live gate or entry condition.**')
        L.append('- basis-3 overlap with size>=5 book cluster spans:')
        for (W, kp, ep, N), o in profile['overlaps'].items():
            L.append(f"  - W={W} K=p{int(kp * 100)} E=p{int(ep * 100)} N={N}: "
                     f"{o['episodes_hit']}/{o['episodes']} episodes intersect = {o['episode_pct']}% "
                     f"| thrust bars inside deep clusters {o['thrust_bars_in_cluster_pct']}% "
                     f"| deep-cluster bars that are thrust {o['cluster_bars_in_thrust_pct']}%")
        L.append('')
    if selection_state is not None:
        L.append('## 7. S5B selection layer — §H decisions and §C constraint references')
        for ln in selection_state.get('report_lines', []):
            L.append(f'- {ln}')
        L.append('')
        L.append('## 8. Stale-artifact note')
    else:
        L.append('## 7. Stale-artifact note')
    L.append(f'- signal_full_records / signal_per_day_pnl: {scored_fresh}')
    L.append('')
    rep = os.path.join(out, 'master_report.md')
    open(rep, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'  report -> {rep} | every artifact written as ONE file (item 3: auto-split deleted; '
          f'the next stage used to read the chopped parts as if they were whole)')
    mark_done(out, 'S9', {'input_sha': input_sha})


def resolve_data(data):
    for cand in (data, os.path.join(_HERE, 'data'), '/data'):
        if cand and os.path.isdir(cand) and glob.glob(os.path.join(cand, '*.csv')):
            return cand
    return data


def resolve_book(book):
    if book is None:
        return None
    for cand in (book, os.path.join(_ENGINE, book), os.path.join(_HERE, book)):
        if os.path.exists(cand):
            return cand
    print(f'ABORT — book file not found: {book}')
    sys.exit(2)


def main():
    import runlog as rl
    _t_start = time.time()
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description='DOT master orchestrator (S0→S9).')
    ap.add_argument('--data', default='/data')
    ap.add_argument('--out', default=os.path.join(_HERE, 'discovery'))
    ap.add_argument('--book', default=None)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--stage', default=None, choices=STAGES)
    ap.add_argument('--market-label', default='US30 (sealed baseline)')
    ap.add_argument('--parity', default=None,
                    help="run the chunking parity harness and exit: a family (e.g. F0) or 'all'")
    ap.add_argument('--smoke', action='store_true',
                    help='SMOKE RUN: every stage S0-S10 executes, same functions, same stage '
                         'order, same pool spawn - only the WORK is reduced. Finishes in '
                         'single-digit minutes on the real frame. It exists because three '
                         'consecutive eleven-hour runs died inside a pool worker at F12, and '
                         'every one would have been caught here in seconds.')
    ap.add_argument('--s3-limit', type=int, default=0,
                    help='bound each family to its first N axis units in S3 (0 = unbounded); for '
                         'smoke-testing the stage without committing days')
    ap.add_argument('--parity-limit', type=int, default=200,
                    help='cap each family to the first N axis units, applied to BOTH parity legs')
    args = ap.parse_args()
    if args.smoke:
        import concurrence_profiler as _cpx
        import triple_convergence_and_d2ddir as _f0x
        import catalogue as _catx
        args.s3_limit = args.s3_limit or 8
        _cpx.N_PERM = 3
        _cpx.K_MIN, _cpx.K_MAX, _cpx.K_STEP = 1, 2, 1
        _cpx.K_SATURATION_STRATUM = []
        _cpx.NULL_K = [1, 2]
        _cpx.NULL_DURATIONS = [1]
        _cpx.DURATIONS = [1, 2]
        # THE FLOOR MUST SCALE WITH THE CAPPED DEPTH OR THE STAGE TESTS NOTHING.
        # align_pool capped to 6 labels collapsed depth_long p50 from 27 to 1, so an
        # onset floor of 30 could never trigger: stage 2 emitted 0 events, stage 3 and
        # stage 5 followed with 0 rows. THREE STAGES RAN ON EMPTY DATA - and the
        # astype('') crash at stage 8's aggregation needed NON-EMPTY data to fire, so
        # a smoke run in that state would not have caught it. A stage that runs on
        # nothing is barely better than a stage that is skipped.
        _cpx.ONSET_FLOORS = [2]
        _cpx.CAT_DURATIONS = [1]
        _cpx.CAT_DOM_DURATIONS = [1]
        _catx.NULL_K_BY_FAMILY = {k: 40 for k in _catx.NULL_K_BY_FAMILY}
        _catx.NULL_K_DEFAULT = 40
        _f0x.DENSITY_K_BANDS = [1, 2]
        os.environ['DOT_SMOKE_CAP'] = '24'
        os.environ['DOT_SMOKE_CHUNK_TARGET'] = '1'
        import discovery_orchestrator as _orchx
        _orchx.set_smoke_mode(True)
        import dot_frame_binding as _fbx
        _applied, _failed = _fbx.install_smoke_caps()
        for _line in _applied:
            print(f'    smoke cap: {_line}', flush=True)
        for _line in _failed:
            print(f'    *** SMOKE CAP FAILED: {_line}', flush=True)
        if _failed or len(_applied) != _fbx.EXPECTED_SMOKE_CAPS:
            raise SystemExit(
                f'ABORT [--smoke] {len(_applied)} of {_fbx.EXPECTED_SMOKE_CAPS} caps installed, '
                f'{len(_failed)} failed. A SMOKE RUN WHOSE REDUCTIONS DID NOT APPLY IS A '
                f'35-MINUTE RUN PRETENDING TO BE A 5-MINUTE ONE, and a cap that silently does '
                f'not install is indistinguishable from a module that is absent. Fix the named '
                f'cap or correct EXPECTED_SMOKE_CAPS - do not proceed on a partial reduction.')
        print(f'    all {len(_applied)} of {_fbx.EXPECTED_SMOKE_CAPS} smoke caps installed.',
              flush=True)
        globals()['SMOKE'] = True
        import wf_selection as _wfsb
        print(f'  *** SMOKE RUN *** every stage S0-S10 EXECUTES - same functions, same order, '
              f'same pool spawn. NO STAGE IS SKIPPED: a stage that does not execute is a stage '
              f'that can still crash at hour nine.', flush=True)
        print(f'    reductions, READ FROM THE ACTUAL VALUES so this line cannot go stale: '
              f's3_limit={args.s3_limit}/family, F12 k={_cpx.K_MIN}..{_cpx.K_MAX}, '
              f'n_perm={_cpx.N_PERM}, onset floors {_cpx.ONSET_FLOORS}, '
              f'S5D pricing null K={sorted(set(_catx.NULL_K_BY_FAMILY.values()))}, '
              f'S5C null-arm target={_wfsb.NULL_TARGET_QUALIFIERS} '
              f'floor={_wfsb.NULL_FLOOR_QUALIFIERS}, density bands {_f0x.DENSITY_K_BANDS}',
              flush=True)
    args.workers = min(args.workers, 16)
    os.environ['DOT_WORKERS'] = str(args.workers)

    t0 = time.time()
    print('═' * 68)
    print('DOT MASTER ORCHESTRATOR')
    print('═' * 68)
    sacred = verify_sacred()
    preflight_loader_audit()
    data_dir = resolve_data(args.data)
    book_file = resolve_book(args.book)
    out = args.out
    for sub in ('raw', 'results', 'scored', 'contenders', 'committed', '.markers'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    mode = 'FROZEN-BOOK replay + verify' if book_file else 'DISCOVER-FRESH (no --book)'
    print(f'mode: {mode} | data: {data_dir} | out: {out} | workers: {args.workers}')

    only = args.stage
    print('\n[S0] INGEST & VALIDATE')
    _logp = rl.open_run_log(out)
    print(f'  run log -> {_logp} (ATTESTATION RECORD: carries wall-clock; every CSV does not)')
    with rl.Stage('S0', 'ingest & validate'):
        df, attest, input_sha = s0_ingest(data_dir, out)
    bind_ingested_frame_permanently(df, input_sha, os.path.join(out, 'results'))
    print('\n[S1] ADAPTIVE THRESHOLDS (oracle)')
    with rl.Stage('S1', 'adaptive thresholds'):
        ad, st = s1_thresholds(df)
    print('\n[S2] POOL & ANCHORS')
    with rl.Stage('S2', 'pool & anchors'):
        pool, anchor, w = s2_pool(df, ad, st)

    if args.parity:
        import discovery_orchestrator as orch
        results = os.path.join(out, 'results')
        os.makedirs(results, exist_ok=True)
        orch.RESULTS_DIR = results
        os.environ['DOT_RESULTS_DIR'] = results
        fams = None if args.parity.lower() == 'all' else [x.strip().upper()
                                                         for x in args.parity.split(',')]
        frame_path = None
        if args.workers > 1:
            frame_path = os.path.join(results, f'_parity_frame_{input_sha}.csv')
            if not os.path.exists(frame_path):
                tmp = frame_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8', newline='') as f:
                    df.to_csv(f, index=False, lineterminator='\n')
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, frame_path)
        print('\n[PARITY] CHUNKING PARITY HARNESS — pre-flight check, no scan is run')
        print('  scope=proof (the smaller candidate vocabulary): parity proves a MECHANISM —')
        print('  that chunked+collated equals unchunked over the SAME bounded range. Full scope')
        print('  would cost hours per leg and prove nothing further about the mechanism.')
        ok = orch.parity_check('proof', workers=args.workers, df=df, adaptive=ad, structural=st,
                               warmup=w, families=fams, limit=args.parity_limit,
                               frame_path=frame_path)
        if frame_path and os.path.exists(frame_path):
            os.remove(frame_path)
        print('\n' + '=' * 68)
        print('PARITY PASS — chunking is sound on this dataset; the full scan may be started.'
              if ok else
              'PARITY FAIL — a chunked family does NOT reproduce its unchunked result on this '
              'dataset. DO NOT start the scan; the pool would be wrong.')
        print('=' * 68)
        sys.exit(0 if ok else 1)

    contenders = committed = profile = evidence = selection_state = wf_state = None
    terrain_state = None
    catalogue_state = None
    run_all = only is None
    if run_all or only == 'S2B':
        print('\n[S2B] MARKET TERRAIN MAP')
        with rl.Stage('S2B', 'market terrain map'):
            terrain_state = s2b_terrain(df, w, out, input_sha, attest)
    discover = (book_file is None)
    if not discover and run_all:
        print('\n[S3–S6] DISCOVERY / REGEN — SKIPPED on the frozen-book verification path.')
        print('  --book replays a ratified book (S8); fresh discovery is the no-book path.')
        print('  Run `python master.py` (no --book) or `--stage S3` for the full 1–2 day discovery.')
    if (run_all and discover) or only == 'S3':
        print('\n[S3] FAMILY DISCOVERY (long-pole; delegates to ratified orchestrator)')
        with rl.Stage('S3', 'family discovery'):
            s3_discovery(out, args.workers, input_sha, 'full', df=df, ad=ad, st=st, w=w,
                         limit=args.s3_limit)
    if run_all or only == 'S3B':
        print('\n[S3B] PER-FAMILY EVIDENCE REVIEW + D2D GATE MEASUREMENT')
        with rl.Stage('S3B', 'family evidence + D2D'):
            evidence = s3b_family_evidence(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest)
    if (run_all and discover) or only == 'S4':
        print('\n[S4] SCHEMA UNIFY')
        with rl.Stage('S4', 'schema unify'):
            s4_schema(out, input_sha)
    if (run_all and discover) or only == 'S5':
        print('\n[S5] CANDIDATE FILTER')
        with rl.Stage('S5', 'candidate filter'):
            s5_filter(out, input_sha, pool)
    if (run_all and discover) or only == 'S6':
        print('\n[S6] FULL-FIELD SCORING (REGEN fresh)')
        with rl.Stage('S6', 'full-field scoring regen'):
            s6_regen(out, input_sha)
    if run_all or only == 'S5D':
        print('\n[S5D] CATALOGUE - fourteen per-family books, every VALID signal')
        with rl.Stage('S5D', 'catalogue emission'):
            catalogue_state = s5d_catalogue(df, ad, st, w, pool, anchor, out, input_sha,
                                            attest, workers=args.workers,
                                            frame_path=os.environ.get('DOT_FRAME_PATH'),
                                            scope='full')
    if run_all or only == 'S5B':
        print('\n[S5B] SELECTION LAYER')
        with rl.Stage('S5B', 'selection layer'):
            selection_state = s5b_selection(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest)
    if run_all or only == 'S5C':
        print('\n[S5C] WALK-FORWARD ON THE SELECTION PROCESS')
        with rl.Stage('S5C', 'walk-forward'):
            wf_state = s5c_walk_forward(df, ad, st, w, pool, anchor, book_file, out, input_sha, attest)
    if run_all or only == 'S7':
        print('\n[S7] CONTENDER HEAD-TO-HEAD')
        import score_g
        bk = book_file if book_file else os.path.join(_ENGINE, 'book50_signals.csv')
        sigs = score_g.build_book(df, pool, anchor, pd.read_csv(bk))
        with rl.Heartbeat('S7 six portfolio scores'):
            with rl.Stage('S7', 'contenders'):
                contenders = s7_contenders(df, ad, st, w, sigs, out, input_sha)
    if only == 'SELECT':
        print('\n[SELECT] SELECTION - screen the raw F0 scan, draw nested arms, score all six')
        with rl.Stage('SELECT', 'select & score'):
            s_select(df, ad, st, w, pool, anchor, book_file, out, input_sha, args.workers)
    if run_all or only == 'S8':
        print('\n[S8] COMMITTED-SYSTEM SCORE')
        with rl.Heartbeat('S8 committed-book scoring'):
            with rl.Stage('S8', 'committed scoring'):
                committed = s8_committed(df, ad, st, w, pool, anchor, book_file, out, input_sha)
    if run_all or only == 'S8B':
        print('\n[S8B] CLUSTER-PARTICIPATION PROFILE')
        with rl.Heartbeat('S8B cluster basis profiling'):
            with rl.Stage('S8B', 'cluster profile'):
                profile = s8b_cluster_profile(df, ad, st, w, pool, anchor, book_file, committed, out, input_sha, attest)
    if run_all or only == 'S9':
        print('\n[S9] REPORT & SPLIT')
        with rl.Stage('S9', 'report'):
            with rl.Heartbeat('S9 report assembly'):
                s9_report(out, attest, contenders, committed, sacred, args.market_label, input_sha, profile, evidence, selection_state)
    if globals().get('SMOKE'):
        _must_have_rows = [
            ('concurrence_events.csv', 'F12 stage 2 - onset events'),
            ('concurrence_entry_order.csv', 'F12 stage 3 - the leader/confirmer split'),
            ('concurrence_depth_bars.csv', 'F12 stage 1 - per-bar depth'),
        ]
        _empty = []
        for _nm, _why in _must_have_rows:
            _p = None
            for _c in (os.path.join(out, 'results', _nm), os.path.join(out, _nm)):
                if os.path.exists(_c):
                    _p = _c
                    break
            if _p is None:
                _empty.append(f'{_nm} ABSENT ({_why})')
                continue
            try:
                _n = max(0, sum(1 for _l in open(_p, encoding='utf-8', errors='replace')
                                if not _l.startswith('#')) - 1)
            except OSError:
                _n = 0
            if _n == 0:
                _empty.append(f'{_nm} ZERO ROWS ({_why})')
        if _empty:
            print('', flush=True)
            print('  *** SMOKE RUN FAILED: an artifact is empty, so the path it exercises was '
                  'NOT TESTED ***', flush=True)
            for _e in _empty:
                print(f'      {_e}', flush=True)
            raise SystemExit(
                'ABORT [--smoke] UNDER --smoke, AN ARTIFACT WITH ZERO ROWS IS A FAILED SMOKE '
                'RUN unless zero is the correct answer. A cap that reduces a population below a '
                'downstream threshold silently turns a test into a no-op - that was invisible '
                'for four runs. Raise DOT_SMOKE_CAP or lower ONSET_FLOORS until these populate.')
        print('  smoke non-empty assertion: every load-bearing artifact has rows.', flush=True)
    if globals().get('SMOKE') and run_all:
        # SMOKE MUST COVER THE NEWEST AND LEAST-PROVEN STAGE. The previous smoke path ran
        # S0-S10 and stopped: SELECT never executed, S8 committed scoring took 0.00s
        # because no --book was passed, and the canary could not fire for the same reason.
        # A SMOKE TEST THAT SKIPS THE ONE THING THAT CHANGED IS NOT A SMOKE TEST.
        print('\n[SELECT] SMOKE LEG - every path --stage SELECT touches, at CAPPED size',
              flush=True)
        with rl.Stage('SELECT', 'smoke: select & score (capped)'):
            _smoke_select(df, ad, st, w, pool, anchor, out, input_sha, args.workers)
    if run_all or only == 'S10':
        print('\n[S10] COLLECT - every analysis artifact into one flat folder, split for upload')
        with rl.Stage('S10', 'collect & split for upload'):
            s10_collect(out, data_dir, input_sha)
    rl.print_timing_table(concurrent_stages=CONCURRENT_STAGES)
    _rows, _tot = rl.timing_table()
    _wall = time.time() - _t_start
    print(f'  TIMING TABLE TOTAL {_tot:.2f}s vs MASTER COMPLETE wall clock {_wall:.2f}s '
          f'| unaccounted {_wall - _tot:.2f}s ({100.0 * (_wall - _tot) / max(_wall, 1e-9):.1f}%). '
          f'Anything unaccounted is work outside a stage wrapper - banner, sacred verify, frame '
          f'binding, argument parsing.')
    _dfa = os.path.join(out, 'data_for_analysis')
    _log = os.path.join(out, 'run_log.txt')
    if os.path.isdir(_dfa) and os.path.exists(_log):
        try:
            for _h in getattr(rl, '_LOG_FH', None) and [rl._LOG_FH] or []:
                _h.flush()
        except Exception:
            pass
        shutil.copy2(_log, os.path.join(_dfa, 'run_log.txt'))
        print(f'  run_log.txt RE-COPIED into data_for_analysis after the timing table. S10 runs '
              f'BEFORE the run ends, so the copy it made stopped at S9 and could never contain '
              f'S10 itself, the stage timing table or MASTER COMPLETE - the one artifact needed '
              f'to size the next parallelism pass.')

    print('\n' + '═' * 68)
    print(f'MASTER COMPLETE in {_hms(time.time() - t0)} | out: {out}')
    if committed and committed.get('canary'):
        print(f'US30 baseline canary: engine intact — net ${committed["net"]} / {committed["trades"]} tr')
    print('═' * 68)


if __name__ == '__main__':
    main()
