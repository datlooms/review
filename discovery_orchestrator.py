import sys
import os
import time
import hashlib
import json
import pickle
import threading
import numpy as np
import pandas as pd
import dots_thresholds as dt
import portfolio_simulation_engine as engine
import wf

import triple_convergence_and_d2ddir as f0m
import f0_to_schema as f0s
import sequential_temporal as f1
import state_transition as f2
import conditional_interaction as f3
import divergence_nonconfirm as f4
import persistence_autocorr as f5
import threshold_crossing as f6
import mean_reversion as f7
import cross_variable_structure as f8
import session_temporal as f9
import rolling_leadlag as f11

# ═══════════════════════════════════════════════════════════════
#  equiDOT — STAGE 8 DISCOVERY ORCHESTRATOR
#  Drives the 11 ratified family scanners at a chosen scope, normalizes
#  every returned row to ONE common schema, writes one CSV per family, and
#  collates discovery_master.csv. Adds ZERO signal logic, ZERO threshold /
#  TM reconstruction — it only calls the scanners' run_search and collects.
#
#  Baseline + oracle are loaded ONCE and passed into every scanner so
#  nothing recomputes. F0 is the heaviest (C(117,3)=260,130 triples with
#  density fused); it is run SEPARATELY and its CSV ingested (see ingest_f0
#  / the F0 note at the bottom), so the orchestrator never holds the full F0
#  search in-process.
#
#  Operator params (this run): target lot 1.0 (worst-day at 1 lot; scale
#  after). F0 internal pre-gate MIN_PF=2.0 is a TRIM in the F0 script, not a
#  selection cut. worst_day_usd is emitted RAW and is a ranking axis to
#  minimize toward 0 — NOT hard-gated at -2500. The only floor at collection
#  is each scanner's MIN_TRADES sample-size floor; no PF/worst-day selection
#  cut is baked in. Collect ALL candidates (survivors AND rejects).
# ═══════════════════════════════════════════════════════════════

RESULTS_DIR = os.environ.get('DOT_RESULTS_DIR', "discovery_results")
SCHEMA = ['family', 'script', 'signal_def', 'direction', 'd2d_mode', 'trades', 'WR',
          'agg_pf', 'worst_day_usd', 'hard_stop_days', 'folds_plus', 'min_fold_pf',
          'spread_pf', 'survival']
F0_CSV = "results_F0_triple_convergence_and_d2ddir.csv"
F1_CSV = "results_F1_sequential_temporal.csv"

# Candidate-count guard: permutation families (F1) explode as O(pool^3). At
# 'full' the orchestrator PRINTS the computed candidate count and warns; the
# operator bounds the pool via SCOPE. It does not silently shrink the space.
MAX_CANDIDATES_WARN = 500000


def _metric_map(row):
    return {
        'trades': row['trades'], 'WR': row['agg_wr'], 'agg_pf': row['agg_pf'],
        'worst_day_usd': row['worst_day_usd'], 'hard_stop_days': row['hard_stop_days'],
        'folds_plus': row['profitable_folds'], 'min_fold_pf': row['min_fold_pf'],
        'spread_pf': f"{row['pf_base']}->{row['pf_stress']}", 'survival': row['survival_pass'],
    }


def _common(family, script, signal_def, direction, d2d_mode, row):
    r = {'family': family, 'script': script, 'signal_def': signal_def,
         'direction': direction, 'd2d_mode': d2d_mode}
    r.update(_metric_map(row))
    return r


# ── per-family scope builders: return kwargs for run_search ──────────────
def _scope(kind):
    proof = kind == 'proof'

    def f0_kw(df, adaptive, structural, warmup):
        feat_candidates, equality_candidates = f0m.build_candidates(df)
        eligible = (df['ADX_Value'].values >= 15) & (df['Volume'].values > 50)
        vol_zero = df['Volume'].values == 0
        warm = np.arange(len(df)) < warmup
        fri_block = ((df['EST_DayOfWeek'].values == 5)
                     & ((df['EST_Hour'].values > 16)
                        | ((df['EST_Hour'].values == 16) & (df['EST_Minute'].values >= 45))))
        entry_allowed = eligible & ~vol_zero & ~fri_block & ~warm
        feature_conditions = f0m.build_conditions(df, feat_candidates, equality_candidates,
                                                  adaptive, structural, eligible)
        arrays = (df['High'].values, df['Low'].values, df['Close'].values, df['ATR_1M'].values,
                  df['EST_DayOfWeek'].values, df['EST_Hour'].values, df['EST_Minute'].values)
        return dict(feature_conditions=feature_conditions,
                    all_features=feat_candidates + equality_candidates,
                    entry_allowed=entry_allowed, d2d_dir=df['D2D_Trend_Dir'].values, arrays=arrays)


    def f1_kw(df, adaptive, structural, warmup):
        pool = f1.build_condition_pool(df, adaptive, structural, warmup)
        labels = f1.scorable_pool(pool, warmup)
        if proof:
            labels = [l for l in ['ADX_Value:hi', 'Momentum_Value:hi', 'Sqz_State:==1',
                                  'RangeOsc_State:==1'] if l in labels]
            lags = [3, 5]
        else:
            lags = f1.LAGS
        n = len(labels) ** 2 * len(lags) * 2
        if not proof:
            print(f"[F1] full scope = {len(labels)}^2 x {len(lags)} lags x 2 dir = {n:,} candidates")
            if n > MAX_CANDIDATES_WARN:
                print(f"[F1] {n:,} ordered-pair candidates — heavy; chunked across workers on the A-label axis.")
        return dict(pool=pool, cond_labels=labels, lags=lags, anchor='ST_Flip',
                    directions=['LONG', 'SHORT'])

    def f2_kw(df, adaptive, structural, warmup):
        states = ['Sqz_State', 'ADX_Rising', 'RangeOsc_State'] if proof else f2.STATE_CANDIDATES
        pool = f2.build_transition_pool(df, states, warmup)
        return dict(pool=pool, cond_labels=list(pool.keys()), directions=['LONG', 'SHORT'])

    def f3_kw(df, adaptive, structural, warmup):
        if proof:
            base_labels = ['ADX_Value:hi', 'Momentum_Value:hi']
            states = ['AT_Regime_ST', 'Sqz_State']
        else:
            feats = list(dt._D_COLS) + ['VWAP_Z', 'OR_Position']
            base_labels = [f"{ft}:{t}" for ft in feats for t in ('hi', 'lo')]
            states = f3.GATE_STATES
        base_pool = f3.build_base_pool(df, base_labels, adaptive, structural)
        gate_masks = f3.build_gate_masks(df, states, warmup)
        return dict(base_pool=base_pool, gate_masks=gate_masks, directions=['LONG', 'SHORT'])

    def f4_kw(df, adaptive, structural, warmup):
        if proof:
            price = ['VWAP_Z', 'KAMA_Dist_ATR']
            flow = ['Micro_OrderFlowDelta', 'OBV_Macd']
        else:
            price, flow = f4.PRICE_FEATS, f4.FLOW_FEATS
        return dict(price_feats=price, flow_feats=flow, d2d_modes=['invert', 'exempt'],
                    orig=df['D2D_Trend_Dir'].values.copy())

    def f5_kw(df, adaptive, structural, warmup):
        states = ['Micro_AutoCorr', 'Efficiency_Ratio', 'KAMA_Slope'] if proof else f5.STATE_FEATS
        labels = [f"{s}:{t}" for s in states for t in ('hi', 'lo')]
        return dict(cond_labels=labels, directions=['LONG', 'SHORT'])

    def f6_kw(df, adaptive, structural, warmup):
        feats = ['Slope_Accel_ST', 'Momentum_Value', 'OBV_Velocity'] if proof else f6.CROSS_FEATS
        return dict(cross_feats=feats, roc_filter=None)

    def f7_kw(df, adaptive, structural, warmup):
        feats = ['VWAP_Z', 'KAMA_Dist_ATR', 'Session_High_Dist_ATR'] if proof else f7.STRETCH_FEATS
        return dict(stretch_feats=feats, d2d_modes=['invert', 'exempt'],
                    orig=df['D2D_Trend_Dir'].values.copy())

    def f8_kw(df, adaptive, structural, warmup):
        return dict(pairs=f8.PAIRS, directions=['LONG', 'SHORT'])

    def f9_kw(df, adaptive, structural, warmup):
        if proof:
            base_labels = ['ADX_Value:hi', 'Momentum_Value:hi', 'VWAP_Z:hi']
            weekdays = None
        else:
            feats = list(dt._D_COLS) + ['VWAP_Z', 'OR_Position']
            base_labels = [f"{ft}:hi" for ft in feats] + [f"{ft}:lo" for ft in feats]
            weekdays = f9.weekday_masks(df)
        sessions = f9.session_masks(df)
        return dict(base_labels=base_labels, sessions=sessions, weekdays=weekdays,
                    directions=['LONG', 'SHORT'])

    def f11_kw(df, adaptive, structural, warmup):
        windows = [60] if proof else f11.WINDOWS
        return dict(pairs=f11.PAIRS, windows=windows, relations=f11.RELATIONS,
                    directions=['LONG', 'SHORT'])

    return {'F0': f0_kw, 'F1': f1_kw, 'F2': f2_kw, 'F3': f3_kw, 'F4': f4_kw, 'F5': f5_kw,
            'F6': f6_kw, 'F7': f7_kw, 'F8': f8_kw, 'F9': f9_kw, 'F11': f11_kw}


# ── per-family signal_def / d2d_mode formatters ──────────────────────────
def _rows_F1(rows, s):
    return [_common('F1', s, f"{r['A']} ->{r['k']}-> {r['B']}",
                    r['direction'], 'confirm', r) for r in rows]


def _rows_F2(rows, s):
    return [_common('F2', s, r['transition'], r['direction'], 'confirm', r) for r in rows]


def _rows_F3(rows, s):
    return [_common('F3', s, f"{r['base']} GATED-BY {r['gate']}", r['direction'], 'confirm', r)
            for r in rows]


def _rows_F4(rows, s):
    return [_common('F4', s, f"{r['price']} NOT-CONFIRMED-BY {r['nonconfirm_flow']}",
                    r['direction'], r['d2d'], r) for r in rows]


def _rows_F5(rows, s):
    return [_common('F5', s, r['condition'], r['direction'], 'confirm', r) for r in rows]


def _rows_F6(rows, s):
    return [_common('F6', s, f"{r['feat']} {r['cross']}(level={r['level']}) ROC={r['roc']}",
                    r['direction'], 'confirm', r) for r in rows]


def _rows_F7(rows, s):
    return [_common('F7', s, f"FADE {r['stretched']}", r['direction'], r['d2d'], r) for r in rows]


def _rows_F8(rows, s):
    return [_common('F8', s, r['relation'], r['direction'], 'confirm', r) for r in rows]


def _rows_F9(rows, s):
    return [_common('F9', s, f"{r['base']} IN-SESSION {r['session']}", r['direction'],
                    'confirm', r) for r in rows]


def _rows_F11(rows, s):
    return [_common('F11', s, f"{r['A']}<->{r['B']} N={r['N']} {r['relation']}",
                    r['direction'], 'confirm', r) for r in rows]


ALL_FAMILIES = [
    ('F0', 'triple_convergence_and_d2ddir', 'pool'),
    ('F1', 'sequential_temporal', 'pool'),
    ('F2', 'state_transition', 'pool'),
    ('F3', 'conditional_interaction', 'pool'),
    ('F4', 'divergence_nonconfirm', 'pool'),
    ('F5', 'persistence_autocorr', 'pool'),
    ('F6', 'threshold_crossing', 'pool'),
    ('F7', 'mean_reversion', 'pool'),
    ('F8', 'cross_variable_structure', 'pool'),
    ('F9', 'session_temporal', 'pool'),
    ('F10', None, 'fused_into_F0'),
    ('F11', 'rolling_leadlag', 'pool'),
    ('F12', 'concurrence_profiler', 'diagnostic'),
    ('F13', 'single_variable_extremes', 'diagnostic'),
]
DIAGNOSTIC_OUTPUTS = {'F12': 'concurrence_depth_bars.csv',
                      'F13': 'results_F13_single_variable_extremes.csv'}


def _provenance_path(csv_path):
    return os.path.splitext(csv_path)[0] + '.provenance'


def stamp_provenance(csv_path, input_sha):
    payload = {'input_sha': input_sha, 'csv_sha256': _sha_file(csv_path)}
    tmp = _provenance_path(csv_path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _provenance_path(csv_path))


def provenance_is_current(csv_path, input_sha):
    pp = _provenance_path(csv_path)
    if not (os.path.exists(csv_path) and os.path.exists(pp)):
        return False, 'no provenance stamp'
    try:
        meta = json.load(open(pp, 'r', encoding='utf-8'))
    except Exception:
        return False, 'unreadable provenance'
    if meta.get('input_sha') != input_sha:
        return False, f"stamped for input_sha {str(meta.get('input_sha'))[:12]}, current is {str(input_sha)[:12]}"
    if meta.get('csv_sha256') != _sha_file(csv_path):
        return False, 'csv changed since stamping'
    return True, 'current'


FOLD_GAP_NOTE = (
    'POOL PROPERTY THE SELECTION LAYER MUST KNOW — folds_plus and min_fold_pf on EVERY row of this '
    'pool come from sacred wf.FOLDS, which is the CALENDAR LITERAL Jan..Jun. Any month outside that '
    'window contributes NOTHING to any row fold evidence. On a series running past June (this run '
    'ends {last}), the newest month is invisible to fold_plus, so S5 gate folds_plus >= 4 is decided '
    'on Jan-Jun alone. This CANNOT be fixed without editing wf.py, which is sacred and byte-locked '
    'at 793e6e5f8d9a; recording it at the point of use is the honest resolution. The proportional '
    'six-slice fold plan used for the COMMITTED-SYSTEM headline (master.py fold_plan) is a separate '
    'and data-relative mechanism and is NOT what produced these columns.')


def write_pool_note(master_path, df):
    last = str(df['Time'].values[-1])[:10] if df is not None and len(df) else 'unknown'
    note = FOLD_GAP_NOTE.format(last=last)
    path = os.path.splitext(master_path)[0] + '.POOL_NOTE.txt'
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(note + chr(10))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print('  POOL NOTE -> ' + os.path.basename(path), flush=True)
    print('    ' + note[:150] + '...', flush=True)
    return path


def verify_diagnostic_outputs(results_dir, input_sha, families=('F12', 'F13')):
    gaps = []
    rows = []
    for fam in families:
        name = DIAGNOSTIC_OUTPUTS.get(fam)
        if not name:
            continue
        csv = os.path.join(results_dir, name)
        ok, why = provenance_is_current(csv, input_sha)
        rows.append((fam, name, 'OK' if ok else 'MISSING', why))
        if not ok:
            gaps.append(f'{fam} ({name}: {why})')
    # THE MARKER WRITE THAT WAS HERE IS REMOVED. verify_diagnostic_outputs runs whether
    # the diagnostic RAN or SKIPPED, so stamping here recorded the CURRENT scanner sha
    # against output produced by the PREVIOUS one - a FALSE PROVENANCE RECORD, and every
    # instrument in the tree trusts the marker, so nothing could detect it. A MARKER MUST
    # ONLY BE WRITTEN BY THE CODE THAT PRODUCED THE OUTPUT: master's _write_diag_marker
    # is now called from the execution path alone. If a diagnostic skips, its existing
    # marker stands unchanged; if it has none, it is UNCHECKED and must not pass.
    print(f"  DIAGNOSTIC OUTPUT VERIFICATION — after the stage ran, not before:", flush=True)
    for fam, name, state, why in rows:
        print(f"    {fam:4} {state:8} {name} — {why}", flush=True)
    if gaps:
        raise SystemExit(
            "ABORT — a diagnostic family was scheduled but produced no current output: "
            + '; '.join(gaps) +
            ". SCHEDULING IS NOT COVERAGE. The pipeline was supposed to have just produced this, so "
            "the same standard applies as to an ingested CSV: the file must exist and carry the "
            "current input_sha. A multi-day scan must not complete reporting 14-family coverage with "
            "a family empty.")
    return rows


def verify_family_coverage(queued_pool, queued_diag, input_sha, results_dir):
    rows = []
    gaps = []
    for fam, script, role in ALL_FAMILIES:
        if role == 'fused_into_F0':
            rows.append((fam, 'FUSED', 'concurrence lens fused into F0; covered when F0 runs'))
            if 'F0' not in queued_pool:
                gaps.append(f'{fam} (fused into F0, but F0 is not covered)')
            continue
        if role == 'pool':
            if fam in queued_pool:
                rows.append((fam, 'QUEUED', 'chunked onto the S3 queue'))
                continue
            csv = os.path.join(results_dir, f'results_{fam}_{script}.csv')
            ok, why = provenance_is_current(csv, input_sha)
            if ok:
                rows.append((fam, 'INGESTED', 'existing CSV, provenance current'))
            else:
                rows.append((fam, 'MISSING', why))
                gaps.append(f'{fam} ({why})')
            continue
        if fam in queued_diag:
            rows.append((fam, 'QUEUED', 'diagnostic stage scheduled this run'))
            continue
        csv = os.path.join(results_dir, DIAGNOSTIC_OUTPUTS.get(fam, ''))
        ok, why = provenance_is_current(csv, input_sha)
        if ok:
            rows.append((fam, 'INGESTED', 'existing diagnostic output, provenance current'))
        else:
            rows.append((fam, 'MISSING', why))
            gaps.append(f'{fam} ({why})')
    n_q = sum(1 for r in rows if r[1] == 'QUEUED')
    n_i = sum(1 for r in rows if r[1] == 'INGESTED')
    n_f = sum(1 for r in rows if r[1] == 'FUSED')
    n_m = len(gaps)
    print(f"  FAMILY COVERAGE — {len(ALL_FAMILIES)} families: {n_q} queued, {n_i} ingested, "
          f"{n_f} fused, {n_m} MISSING", flush=True)
    for fam, state, why in rows:
        print(f"    {fam:4} {state:9} {why}", flush=True)
    if gaps:
        raise SystemExit("ABORT — S3 would complete without these families: " + '; '.join(gaps) +
                         ". A run must never silently omit a family, and a stale ingested CSV is not "
                         "coverage. Queue them or supply a CSV stamped for this input_sha.")
    return rows


def _rows_F0(rows, script):
    return list(rows)


def f0_combo_count(kw, lo, hi):
    import itertools as _it
    feats = kw['all_features']
    fc = kw['feature_conditions']
    total = 0
    for (i, j, k) in list(_it.combinations(range(len(feats)), 3))[lo:hi]:
        total += (len(fc[feats[i]]) * len(fc[feats[j]]) * len(fc[feats[k]]) * 2)
    return total


F0_MIN_TRADES_OVERRIDE = 30
# F0_MIN_PF_OVERRIDE REMOVED: the scanner now carries MIN_PF = 2.0 as its own
# default, so the effective threshold is visible in the file an analyst reads.
# It lived in three places and the source disagreed with all of them; one
# analyst read 4.0 and concluded the field was PF-gated at 4.
F0_ASYMMETRY_NOTE = (
    'F0 POOL PROPERTY — RECORDED, NOT HIDDEN. F0 candidates pass an APPROXIMATE PF pre-screen that '
    'F2-F11 candidates never face. F0 first scores every triple with its own fast single-pass scorer '
    '(same SPREAD 3.0, same BE trigger, same BE/LF/SL/FC exit types, walking real bars to real exits, '
    'but strictly one-position-at-a-time with NO jar, NO conviction sizing and NO gap fillers), so the '
    'TRADE COUNT IS EXACT while the PF is a flat-1-lot no-jar proxy. Only triples clearing that proxy '
    f'PF >= {f0m.MIN_PF} and trades >= {F0_MIN_TRADES_OVERRIDE} are re-scored through the '
    'ratified engine. The PF gate is deliberately LOOSE (and it is now the scanner OWN overridden '
    'DOWN to 2.0) because pre-gating hard on an approximated metric would discard candidates the '
    'ratified engine might rate well. Consequence for selection: a triple the proxy rates below 2.0 '
    'never reaches the pool even if the ratified engine would have rated it higher.')


def run_f0_chunk(df, adaptive, structural, warmup, kw, lo, hi):
    import itertools as _it
    f0m.MIN_TRADES = F0_MIN_TRADES_OVERRIDE
    orig_comb = f0m.combinations

    def _sliced(iterable, r):
        return list(_it.combinations(iterable, r))[lo:hi]

    f0m.combinations = _sliced
    try:
        survivors = f0m.run_search(df, kw['feature_conditions'], kw['all_features'],
                                   kw['entry_allowed'], kw['d2d_dir'], kw['arrays'])
    finally:
        f0m.combinations = orig_comb
    return list(survivors), f0_combo_count(kw, lo, hi)


def _f0_score_chunk(payload):
    """THE PAYLOAD IS THE TRANSPORT. Never a module global.

    This read orch._F0_KEPT_PATH, a module global the PARENT set and the worker
    never saw: a spawned worker re-imports this module fresh and gets the L433
    default of None, so open(None) raised in every one of 56 workers on the first
    production execution of f0_parity_proof. Same class as the frame-binding
    defect sitecustomize.py exists to fix - a parent-only global that does not
    survive spawn - and it was missed for exactly one reason: this path had never
    executed. The payload already crossed the boundary correctly, so it carries
    the path too and nothing here depends on parent state.
    """
    lo, hi, scope, frame_path, kept_path = payload
    import discovery_orchestrator as orch
    df, adaptive, structural, warmup, _kw = orch._worker_context(scope, frame_path, 'F0')
    import pickle as _pk
    with open(kept_path, 'rb') as f:
        kept = _pk.load(f)
    month = pd.Series(df['Time'].values).str[:7].values
    return lo, [f0s.score_survivor(df, row, month, adaptive, structural, warmup)
                for row in kept[lo:hi]]


def f0_rows_from_raw(df, adaptive, structural, warmup, raw_survivors, workers=1,
                     scope='full', frame_path=None, results_dir=None, kept=None):
    """Item 19: the RE-SCORE parallelises. THE DEDUP DOES NOT.

    deduplicate() is a global greedy pass against a running keep-set: each
    candidate is compared with every set already kept, so splitting it lets
    near-duplicates survive in different shards and reunite in the output. It
    stays SINGLE-PASS IN ASCENDING CHUNK ORDER. Only the per-survivor re-score
    parallelises, and each of those is genuinely independent - 19,757 of them,
    measured at ~5 hours single-threaded, which is why this is the stage worth
    parallelising at all.

    A PARITY PROOF AGAINST THE SERIAL RESULT IS MANDATORY and is what
    f0_parity_proof() runs.
    """
    kept = f0m.deduplicate(list(raw_survivors)) if kept is None else kept
    month = pd.Series(df['Time'].values).str[:7].values
    if workers <= 1 or frame_path is None or results_dir is None or len(kept) < 64:
        return [f0s.score_survivor(df, row, month, adaptive, structural, warmup) for row in kept]
    import pickle as _pk
    kept_path = os.path.join(results_dir, '_f0_kept.pkl')
    with open(kept_path, 'wb') as f:
        _pk.dump(kept, f)
    n = len(kept)
    size = max(1, -(-n // (workers * 4)))
    bounds = [(i, min(i + size, n)) for i in range(0, n, size)]
    out = {}
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    import runlog as _rl
    payloads = [(lo, hi, scope, frame_path, kept_path) for lo, hi in bounds]
    with _rl.Progress('F0 re-score', len(payloads)) as pg:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads)),
                                 mp_context=_mp.get_context('spawn')) as ex:
            for lo, rows in ex.map(_f0_score_chunk, payloads):
                out[lo] = rows
                pg.step(1)
    return [r for lo, _hi in bounds for r in out[lo]]


F0_PARITY_SAMPLE = 512


def _f0_code_sha():
    """The proof certifies CODE, so the attestation key must include the code."""
    import hashlib as _h
    h = _h.sha256()
    for rel in ('orchestrator/discovery_orchestrator.py', 'scanners/f0_to_schema.py',
                'scanners/triple_convergence_and_d2ddir.py'):
        p_ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
        if os.path.exists(p_):
            h.update(open(p_, 'rb').read())
    return h.hexdigest()[:12]


def _stratified_indices(n_total, n_sample, workers):
    """C1: a CONTIGUOUS HEAD exercises zero chunk boundaries at realistic worker counts.

    With n=19,757 and chunk size ceil(n/(workers*4)), the first 512 survivors sit
    inside chunk 0 at every worker count up to 8 - including the default of 2 -
    so a head-bounded proof touches NO boundary. Chunk-boundary coverage was
    ruled sufficient when the proof ran over the full population; bounding it to
    a head silently voided that. A stratified spread across the whole index range
    costs the same and covers every boundary at any worker count. Deterministic,
    so kept[i] and par[i] index the same survivor.
    """
    if n_sample >= n_total:
        return list(range(n_total))
    idx = set()
    size = max(1, -(-n_total // max(int(workers), 1) // 4))
    for b in range(0, n_total, size):
        for off in (0, 1):
            if b + off < n_total:
                idx.add(b + off)
    stride = max(1, n_total // max(n_sample - len(idx), 1))
    for i in range(0, n_total, stride):
        if len(idx) >= n_sample:
            break
        idx.add(i)
    return sorted(idx)[:max(n_sample, len(idx))]


def _parity_attest_path(results_dir):
    return os.path.join(results_dir, '_f0_parity.attest')


def f0_parity_proof(df, adaptive, structural, warmup, raw_survivors, workers, scope, frame_path,
                    results_dir, input_sha=''):
    """Mandatory parity, WITHOUT repaying the cost item 19 exists to remove.

    The first wiring computed the serial reference over all 19,757 survivors,
    computed the parallel result to compare against it, discarded that result,
    and then computed the parallel result AGAIN for the caller - three full
    re-scores on a stage whose whole purpose is to stop paying for one. Now:
    the FULL parallel pass runs ONCE and is RETURNED for the caller to use, and
    the serial reference is BOUNDED to a deterministic prefix sample and
    ATTESTED PER input_sha so it is paid once rather than every run.

    The reference stays genuine: f0_rows_from_raw at workers <= 1 is a plain
    list comprehension with no ProcessPool, so the sample is not produced by the
    machinery under test. DataFrame.equals is element-wise, order-sensitive,
    dtype-sensitive and NaN-aware, which is what covers float accumulation,
    ordering and chunk boundaries.
    """
    kept_shared = f0m.deduplicate(list(raw_survivors))
    par = f0_rows_from_raw(df, adaptive, structural, warmup, raw_survivors, workers=workers,
                           scope=scope, frame_path=frame_path, results_dir=results_dir,
                           kept=kept_shared)
    att = _parity_attest_path(results_dir)
    key = f'{input_sha}|{_f0_code_sha()}'
    if input_sha and os.path.exists(att):
        prev = open(att, 'r', encoding='utf-8').read().strip()
        if prev == key:
            print(f'  F0 RE-SCORE PARITY: already attested for input_sha+code_sha {key[:29]}... '
                  f'({os.path.basename(att)}) - paid once per DATASET AND CODE STATE. Keying on '
                  f'input_sha alone would skip the proof after a change to score_survivor, the '
                  f'chunking or the reassembly against an unchanged dataset, and the proof exists '
                  f'to certify CODE.', flush=True)
            return True, par
    kept = kept_shared
    n = min(int(F0_PARITY_SAMPLE), len(kept))
    idx = _stratified_indices(len(kept), n, workers)
    month = pd.Series(df['Time'].values).str[:7].values
    ser = [f0s.score_survivor(df, kept[i], month, adaptive, structural, warmup) for i in idx]
    a_ = pd.DataFrame(ser, columns=SCHEMA).reset_index(drop=True)
    b_ = pd.DataFrame([par[i] for i in idx], columns=SCHEMA).reset_index(drop=True)
    same = a_.equals(b_)
    print(f'  F0 RE-SCORE PARITY: serial reference on a bounded {n}-survivor prefix vs the same '
          f'prefix of the parallel result | IDENTICAL: {same} | dedup ran SERIALLY in both, only '
          f'the re-score parallelised | full parallel pass computed ONCE and reused | sample is '
          f'STRATIFIED across the index range so every chunk boundary is covered at any worker '
          f'count | this reference is ~{100.0*n/max(len(par),1):.1f}% of a serial pass and is paid '
          f'on the FIRST run per dataset+code only, zero thereafter', flush=True)
    if not same:
        raise SystemExit('ABORT [item 19] the parallel F0 re-score does not equal the serial one. '
                         'The parity proof is mandatory and it failed.')
    if input_sha:
        tmp = att + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(key)
        os.replace(tmp, att)
    return same, par


def _f0_chunk_pickle(script, idx):
    return os.path.join(RESULTS_DIR, f"results_F0_{script}_c{idx:04d}.pkl")


def collate_f0(script, n_chunks, df, adaptive, structural, warmup, expected_total, input_sha):
    ok, detail = candidate_invariant('F0', script, n_chunks, expected_total)
    if not ok and detail != 'missing per-chunk candidate count':
        raise SystemExit(f"ABORT [F0] CANDIDATE-COUNT INVARIANT FAILED: {detail}. Chunking changed "
                         f"the combo space; results are NOT trustworthy.")
    raw = []
    for idx in range(n_chunks):
        pk = _f0_chunk_pickle(script, idx)
        if not (os.path.exists(pk) and chunk_is_complete('F0', script, idx)):
            return False, 0
        with open(pk, 'rb') as f:
            raw.extend(pickle.load(f))
    n_raw = len(raw)
    _wk = int(os.environ.get('DOT_WORKERS', '1'))
    _fp = os.environ.get('DOT_FRAME_PATH') or None
    if _wk > 1 and _fp:
        _ok, rows = f0_parity_proof(df, adaptive, structural, warmup, raw, _wk, 'full', _fp,
                                    RESULTS_DIR, input_sha=str(input_sha or ''))
    else:
        rows = f0_rows_from_raw(df, adaptive, structural, warmup, raw)
    csv, done = _family_paths('F0', script)
    _write_atomic_csv(pd.DataFrame(rows, columns=SCHEMA), csv)
    _mark_family_done(csv, done, len(rows), script)
    if input_sha is not None:
        stamp_provenance(csv, input_sha)
    with open(os.path.splitext(csv)[0] + '.note', 'w', encoding='utf-8') as f:
        f.write(F0_ASYMMETRY_NOTE + '\n')
    print(f"  [F0] {n_raw} raw survivors across {n_chunks} chunks -> global 80% overlap dedup at "
          f"COLLATION (single pass, ascending chunk order) -> {len(rows)} pool rows", flush=True)
    return True, len(rows)


FAMILIES = [
    ('F0', 'triple_convergence_and_d2ddir', f0m, _rows_F0),
    ('F1', 'sequential_temporal', f1, _rows_F1),
    ('F2', 'state_transition', f2, _rows_F2),
    ('F3', 'conditional_interaction', f3, _rows_F3),
    ('F4', 'divergence_nonconfirm', f4, _rows_F4),
    ('F5', 'persistence_autocorr', f5, _rows_F5),
    ('F6', 'threshold_crossing', f6, _rows_F6),
    ('F7', 'mean_reversion', f7, _rows_F7),
    ('F8', 'cross_variable_structure', f8, _rows_F8),
    ('F9', 'session_temporal', f9, _rows_F9),
    ('F11', 'rolling_leadlag', f11, _rows_F11),
]


HEARTBEAT_SECONDS = 60


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _family_paths(fam, script):
    csv = os.path.join(RESULTS_DIR, f"results_{fam}_{script}.csv")
    done = os.path.join(RESULTS_DIR, f"results_{fam}_{script}.done")
    return csv, done


def _write_atomic_csv(frame, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        frame.to_csv(f, index=False, lineterminator='\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def scanner_sha(script):
    """sha of the SCANNER that produced an artifact. A code change must invalidate
    its own output.

    The marker recorded rows, csv_sha256 and schema_cols - all properties of the
    ARTIFACT, none of the PRODUCER. So when F13's and F0's scanners changed (999 ->
    blank), the schema gate could not fire (same columns, same row counts) and the
    family resume read the stale CSVs straight off disk. `--stage S3` would have
    silently skipped exactly the two families that changed.

    Recording the producing scanner's sha closes that: F0 and F13 re-scan because
    their scanners moved, the other nine resume because theirs did not, and there
    is NO MANUAL MARKER DELETE - the marker invalidates itself.

    A marker written before this existed carries no scanner_sha and is treated as
    STALE, so the first run after this change re-scans once and every run after
    that resumes correctly.
    """
    if not script:
        return None
    _here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(_here, '..', 'scanners', f'{script}.py'),
                 os.path.join(_here, 'scanners', f'{script}.py'),
                 os.path.join(os.getcwd(), 'scanners', f'{script}.py')):
        if os.path.exists(cand):
            return _sha_file(cand)[:12]
    return None


def _mark_family_done(csv_path, done_path, n_rows, script=None):
    payload = {'rows': int(n_rows), 'csv_sha256': _sha_file(csv_path),
               'schema_cols': len(SCHEMA), 'scanner_sha': scanner_sha(script),
               'run_id': RUN_ID}
    tmp = done_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, done_path)


def family_is_complete(fam, script):
    csv, done = _family_paths(fam, script)
    if not (os.path.exists(csv) and os.path.exists(done)):
        return False, None
    try:
        meta = json.load(open(done, 'r', encoding='utf-8'))
    except Exception:
        return False, None
    if meta.get('csv_sha256') != _sha_file(csv):
        return False, None
    want = scanner_sha(script)
    got = meta.get('scanner_sha')
    if want is not None and got is None:
        # ABSENCE IS NOT EVIDENCE THE PRODUCER MOVED - it is evidence the marker
        # predates the field. Distinguish the two by RUN IDENTITY: a marker stamped
        # with this invocation's run_id was written by this build, so its missing
        # scanner_sha is a writer gap and the artifact is current. Only a marker from
        # an EARLIER run is treated as stale, which is the self-healing case.
        if meta.get('run_id') == RUN_ID:
            print(f'  {fam}: marker has no scanner_sha but was written by THIS RUN '
                  f'(run_id {RUN_ID}) - the artifact is current and is NOT re-scanned. '
                  f'Stamping the field now.', flush=True)
            meta['scanner_sha'] = want
            try:
                with open(done, 'w', encoding='utf-8') as f:
                    json.dump(meta, f)
            except OSError:
                pass
            return True, meta
        print(f'  {fam}: marker predates scanner_sha recording (earlier run) - RE-SCANNING '
              f'once to establish the baseline.', flush=True)
        return False, None
    if want is not None and got != want:
        print(f'  {fam}: marker STALE - scanner {script}.py sha {got} -> {want}. The output '
              f'schema is unchanged, so no schema gate can see this; the PRODUCER moved. '
              f'RE-SCANNING.', flush=True)
        return False, None
    return True, meta


def resume_family(fam, script):
    csv, _done = _family_paths(fam, script)
    frame = pd.read_csv(csv)
    missing = [c for c in SCHEMA if c not in frame.columns]
    if missing:
        return None
    return frame[SCHEMA].to_dict('records')


FRAME_MUTATORS = {
    'F1': '__F1SEQ', 'F2': '__F2TRANS', 'F3': '__F3COND', 'F4': '__F4DIV', 'F5': '__F5PERS',
    'F6': '__F6CROSS', 'F7': '__F7REV', 'F8': '__F8REL', 'F9': '__F9SESS', 'F11': '__F11LL',
    'F12': '__F12DEPTH',
}
GATE_MUTATORS = ('F4', 'F7', 'F12', 'F13')


class FrameGuard:
    """Restores the shared frame to the exact state a family received it in.

    Eleven of the twelve pool/diagnostic families inject their own scratch column
    (__F1SEQ, __F3COND, ...) into the SHARED frame, and four also overwrite
    D2D_Trend_Dir. F0 is the only family that validates its input vocabulary, so
    it is the one that detects the contamination: its EXCLUDE-complement
    assertion correctly rejects any extra column. Restoring the gate column alone
    was never enough; the column SET is the same class of hazard on a different
    object. Both are restored here, on every path that can execute a family.
    """

    def __init__(self, df):
        self.df = df
        self.cols = list(df.columns)
        self.gate = df['D2D_Trend_Dir'].values.copy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        known = set(self.cols)
        extra = [c for c in self.df.columns if c not in known]
        if extra:
            self.df.drop(columns=extra, inplace=True)
        self.df['D2D_Trend_Dir'] = self.gate
        return False


def assert_frame_clean(df, baseline_cols, fam):
    extra = [c for c in df.columns if c not in set(baseline_cols)]
    if extra:
        raise SystemExit(
            f"ABORT [{fam}] the shared frame carries scratch columns from another family: {extra}. "
            f"F0 validates its vocabulary and will reject these. FrameGuard should have removed them; "
            f"a family injected a column outside a guarded region.")
    return True


def run_family(fam, script, mod, fmt, kw_builder, df, adaptive, structural, warmup, limit=0):
    baseline_cols = list(df.columns)
    assert_frame_clean(df, baseline_cols, fam)
    t0 = time.time()
    with FrameGuard(df):
        kw = kw_builder(df, adaptive, structural, warmup)
        bounds, n_units = _bounds_for(fam, kw)
        if limit:
            n_units = min(limit, n_units)
        expected = None
        if fam == 'F0':
            expected = f0_combo_count(kw, 0, n_units)
            raw, _exp = run_f0_chunk(df, adaptive, structural, warmup, kw, 0, n_units)
            common = f0_rows_from_raw(df, adaptive, structural, warmup, raw)
            actual = _exp
        else:
            sub = _slice_axis(kw, CHUNK_AXIS[fam], 0, n_units) if limit else dict(kw)
            if fam == 'F1':
                expected = n_units * len(kw['cond_labels']) * len(kw['directions'])
            rows = mod.run_search(df, adaptive=adaptive, structural=structural, warmup=warmup, **sub)
            common = fmt(rows, script)
            actual = expected
        if expected is not None and actual != expected:
            raise SystemExit(
                f"ABORT [{fam}] SEQUENTIAL-PATH CANDIDATE-COUNT INVARIANT FAILED: {actual} != "
                f"{expected}. The bound applied to this path did not search what it promised.")
        print(f"    [{fam}] sequential path bounded to {n_units} axis units"
              + (f" | candidate invariant {actual} == {expected}" if expected is not None
                 else " | no candidate expectation for this family"), flush=True)
    csv, done = _family_paths(fam, script)
    _write_atomic_csv(pd.DataFrame(common, columns=SCHEMA), csv)
    _mark_family_done(csv, done, len(common), script)
    print(f"[{fam}] {len(common)} rows -> {csv}  ({time.time() - t0:.1f}s)", flush=True)
    return common


CHUNK_AXIS = {'F0': '__combos__', 'F1': ('cond_labels', 'lags'), 'F2': 'cond_labels', 'F3': 'base_pool',
              'F4': 'price_feats', 'F5': 'cond_labels', 'F6': 'cross_feats',
              'F7': 'stretch_feats', 'F8': 'pairs', 'F9': 'base_labels', 'F11': 'pairs'}
COST_ORDER = ['F1', 'F0', 'F3', 'F9', 'F11', 'F4', 'F2', 'F7', 'F5', 'F8', 'F6']
TARGET_CHUNKS_PER_FAMILY = int(os.environ.get('DOT_SMOKE_CHUNK_TARGET', '64'))
# ONE PARAMETER, READ FROM THE ENVIRONMENT - the orchestrator is NOT a scanner and is
# not locked, so a direct read here is less code than routing it through
# dot_frame_binding, and the environment is already inherited by every spawned worker.
#
# WHY IT MATTERS AT SMOKE SCALE. _chunk_bounds does:
#     size = 1 if n_items <= target else -(-n_items // target)
# With target=64 and n_items small, size collapses to 1 and you get ONE CHUNK PER ITEM:
# 144 F1 candidates produced 12 chunks, ~100 across eleven families. Each chunk spawns a
# worker that rebuilds the oracle - MEASURED AT 38.1s, not the 4.0s frame read - so the
# FIXED COST WAS THE ENTIRE RUNTIME: ~71 minutes of setup to scan a few hundred
# candidates. At target=1 that is ~11 chunks and ~1.9 min wall at 4 workers.
TARGET_CHUNKS_F0 = 512
_WCACHE = {}


def _axis_units(kw, axis):
    if axis == '__combos__':
        import itertools as _it
        n = len(kw['all_features'])
        return sum(1 for _ in _it.combinations(range(n), 3)), [n]
    if isinstance(axis, tuple):
        sizes = [len(kw[a]) for a in axis]
        n = 1
        for x in sizes:
            n *= x
        return n, sizes
    return len(kw[axis]), [len(kw[axis])]


def _one_axis(kw, name, lo, hi):
    src = kw[name]
    if isinstance(src, dict):
        keys = list(src.keys())[lo:hi]
        return {k: src[k] for k in keys}
    return list(src)[lo:hi]


def _slice_axis(kw, axis, lo, hi):
    out = dict(kw)
    if axis == '__combos__':
        return out
    if not isinstance(axis, tuple):
        out[axis] = _one_axis(kw, axis, lo, hi)
        return out
    outer, inner = axis
    n_inner = len(kw[inner])
    i = lo // n_inner
    j = lo % n_inner
    out[outer] = _one_axis(kw, outer, i, i + 1)
    out[inner] = _one_axis(kw, inner, j, j + (hi - lo))
    return out


RUN_ID = f'{int(time.time())}-{os.getpid()}'
SMOKE_MODE = False


def set_smoke_mode(on):
    """Set ONLY by master.py when --smoke is passed. NEVER an environment variable.

    An env var is a manual step the operator has to remember, and it is banned. He
    must be able to run

        python master.py --data data --workers 14 --out discovery\\full

    and get the correct geometry with nothing else set.
    """
    global SMOKE_MODE
    SMOKE_MODE = bool(on)


def _chunk_target():
    """READ AT CALL TIME, NOT AT IMPORT.

    TARGET_CHUNKS_PER_FAMILY is evaluated when this module is imported, and
    master.py imports the orchestrator at ITS module load - before main() parses
    --smoke and exports DOT_SMOKE_CHUNK_TARGET. The env was therefore set AFTER
    the value it was meant to change had already frozen at 64.
    """
    if not SMOKE_MODE:
        return TARGET_CHUNKS_PER_FAMILY
    try:
        return int(os.environ.get('DOT_SMOKE_CHUNK_TARGET', TARGET_CHUNKS_PER_FAMILY))
    except (TypeError, ValueError):
        return TARGET_CHUNKS_PER_FAMILY


def _chunk_bounds(n_items, target=None, unit_cap=None):
    if n_items <= 0:
        return []
    if target is None:
        target = _chunk_target()
    size = 1 if n_items <= target else -(-n_items // target)
    if unit_cap is not None:
        size = min(size, unit_cap)
    return [(i, min(i + size, n_items)) for i in range(0, n_items, size)]


def _bounds_for(fam, kw):
    axis = CHUNK_AXIS[fam]
    n_units, sizes = _axis_units(kw, axis)
    if axis == '__combos__':
        _t = min(TARGET_CHUNKS_F0, _chunk_target()) if SMOKE_MODE else TARGET_CHUNKS_F0
        return _chunk_bounds(n_units, target=_t), n_units
    if isinstance(axis, tuple):
        # target=n_units gives size=1: ~7,170 single-unit chunks for F1. THAT IS THE
        # REAL RUN'S GEOMETRY AND IT MUST NOT CHANGE. Routing this through
        # _chunk_target() made target 64, size 15 and 485 chunks - and per-chunk cost
        # went ~58s to ~1,900s, so TOTAL CPU WORK MORE THAN DOUBLED, 116 -> 256
        # CPU-hours. The smoke reduction is gated behind SMOKE_MODE so it cannot
        # reach a real run again.
        _t = min(n_units, _chunk_target()) if SMOKE_MODE else n_units
        return _chunk_bounds(n_units, target=_t, unit_cap=sizes[1]), n_units
    return _chunk_bounds(n_units), n_units


CHUNK_RETRY_PASSES = 3
CHUNK_RETRY_BACKOFF_S = 2.0


def _chunk_paths(fam, script, idx):
    csv = os.path.join(RESULTS_DIR, f"results_{fam}_{script}_c{idx:04d}.csv")
    done = os.path.join(RESULTS_DIR, f"results_{fam}_{script}_c{idx:04d}.done")
    return csv, done


def chunk_is_complete(fam, script, idx):
    csv, done = _chunk_paths(fam, script, idx)
    if not (os.path.exists(csv) and os.path.exists(done)):
        return False
    try:
        meta = json.load(open(done, 'r', encoding='utf-8'))
    except Exception:
        return False
    return meta.get('csv_sha256') == _sha_file(csv)


def _worker_context(scope, frame_path, fam):
    if _WCACHE.get('frame_path') != frame_path:
        _WCACHE.clear()
        _WCACHE['frame_path'] = frame_path
        _WCACHE['df'] = pd.read_csv(frame_path)
        _WCACHE['warmup'] = engine.warmup_floor(_WCACHE['df'], verbose=False)
        _WCACHE['adaptive'] = dt.compute_adaptive_thresholds(_WCACHE['df'])
        _WCACHE['structural'] = dt.compute_structural_gates(_WCACHE['df'])
        _WCACHE['kw'] = {}
        _WCACHE['builders'] = _scope(scope)
    df = _WCACHE['df']
    if fam not in _WCACHE['kw']:
        _WCACHE['kw'][fam] = _WCACHE['builders'][fam](df, _WCACHE['adaptive'],
                                                      _WCACHE['structural'], _WCACHE['warmup'])
    return df, _WCACHE['adaptive'], _WCACHE['structural'], _WCACHE['warmup'], _WCACHE['kw'][fam]


def run_chunk_rows(fam, script, df, adaptive, structural, warmup, kw, lo, hi):
    spec = {f[0]: f for f in FAMILIES}[fam]
    mod, fmt = spec[2], spec[3]
    sub = _slice_axis(kw, CHUNK_AXIS[fam], lo, hi)
    orig = df['D2D_Trend_Dir'].values.copy()
    try:
        if fam == 'F0':
            return run_f0_chunk(df, adaptive, structural, warmup, kw, lo, hi)
        if fam == 'F1':
            import run_f1_parallel as f1p
            month = pd.Series(df['Time'].values).str[:7].values
            anchor_event = f1.anchor_array(df, kw['anchor'])
            a_labels = sub['cond_labels']
            b_labels = list(kw['cond_labels'])
            lags = sub['lags']
            expected = len(a_labels) * len(b_labels) * len(lags) * len(kw['directions'])
            common = f1p._score_pairs(a_labels, b_labels, kw['pool'], df, month, anchor_event,
                                      lags, kw['directions'], adaptive, structural, warmup)
            return common, expected
        rows = mod.run_search(df, adaptive=adaptive, structural=structural, warmup=warmup, **sub)
    finally:
        df['D2D_Trend_Dir'] = orig
    return fmt(rows, script), None


def _chunk_worker(payload):
    fam, script, scope, results_dir, frame_path, idx, lo, hi = payload
    import discovery_orchestrator as orch
    orch.RESULTS_DIR = results_dir
    if orch.chunk_is_complete(fam, script, idx):
        return (fam, idx, -1, 0.0, hi - lo)
    df, adaptive, structural, warmup, kw = orch._worker_context(scope, frame_path, fam)
    t0 = time.time()
    with orch.FrameGuard(df):
        common, expected = orch.run_chunk_rows(fam, script, df, adaptive, structural, warmup,
                                               kw, lo, hi)
    if fam == 'F0':
        pk = orch._f0_chunk_pickle(script, idx)
        tmpp = pk + '.tmp'
        with open(tmpp, 'wb') as fh:
            pickle.dump(common, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmpp, pk)
        common = []
    if expected is not None:
        cpath = os.path.join(results_dir, f"results_{fam}_{script}_c{idx:04d}.cand")
        tmpc = cpath + '.tmp'
        with open(tmpc, 'w', encoding='utf-8') as fh:
            fh.write(str(int(expected)))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmpc, cpath)
    csv, done = orch._chunk_paths(fam, script, idx)
    orch._write_atomic_csv(pd.DataFrame(common, columns=SCHEMA), csv)
    orch._mark_family_done(csv, done, len(common), script)
    return (fam, idx, len(common), time.time() - t0, hi - lo)


def candidate_invariant(fam, script, n_chunks, expected_total):
    if expected_total is None:
        return True, 'n/a'
    got = 0
    for idx in range(n_chunks):
        cpath = os.path.join(RESULTS_DIR, f"results_{fam}_{script}_c{idx:04d}.cand")
        if not os.path.exists(cpath):
            return False, 'missing per-chunk candidate count'
        got += int(open(cpath, 'r', encoding='utf-8').read().strip())
    if got != expected_total:
        return False, f'{got} != {expected_total}'
    return True, f'{got} == {expected_total}'


def collate_family_chunks(fam, script, n_chunks, expected_total=None):
    ok, detail = candidate_invariant(fam, script, n_chunks, expected_total)
    if not ok and detail != 'missing per-chunk candidate count':
        raise SystemExit(f"ABORT [{fam}] CANDIDATE-COUNT INVARIANT FAILED: sum of per-chunk candidate "
                         f"counts {detail}. Chunking changed the search space; results are NOT trustworthy.")
    frames = []
    for idx in range(n_chunks):
        if not chunk_is_complete(fam, script, idx):
            return False, 0
        csv, _d = _chunk_paths(fam, script, idx)
        try:
            frames.append(pd.read_csv(csv))
        except pd.errors.EmptyDataError:
            frames.append(pd.DataFrame(columns=SCHEMA))
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    csv, done = _family_paths(fam, script)
    _write_atomic_csv(merged[SCHEMA], csv)
    # STAMP scanner_sha AT COLLATION. This call site had no `script`, so
    # scanner_sha(None) returned None and every marker this path wrote lacked the
    # field. family_is_complete then read 'sha None -> <actual>' and declared TEN
    # FAMILIES STALE MINUTES AFTER THEY COLLATED IN THE SAME RUN, discarding ~13
    # hours of finished work. A gate that invalidates CURRENT work is worse than no
    # gate: it converts a completed run into a repeat of itself, silently, after the
    # expensive part has already succeeded.
    _mark_family_done(csv, done, len(merged), script)
    return True, len(merged)


class _Heartbeat:
    def __init__(self, label, interval=HEARTBEAT_SECONDS):
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = None

    def __enter__(self):
        def beat():
            while not self._stop.wait(self._interval):
                mins = (time.time() - self._t0) / 60.0
                print(f"    ... {self._label} still running ({mins:.1f} min elapsed)", flush=True)
        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False


def _hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def ingest_f0():
    path = os.path.join(RESULTS_DIR, F0_CSV)
    if not os.path.exists(path):
        print(f"[F0] {path} not found — run F0 separately and drop its common-schema CSV here "
              f"(see F0 note). Skipping F0 at collation.")
        return []
    df0 = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df0.columns]
    if missing:
        raise ValueError(f"F0 CSV missing schema columns: {missing}")
    print(f"[F0] ingested {len(df0)} rows from {path}")
    return df0[SCHEMA].to_dict('records')


def ingest_f1():
    path = os.path.join(RESULTS_DIR, F1_CSV)
    if not os.path.exists(path):
        return []
    df1 = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df1.columns]
    if missing:
        raise ValueError(f"F1 CSV missing schema columns: {missing}")
    print(f"[F1] ingested {len(df1)} rows from {path} (in-process F1 skipped)")
    return df1[SCHEMA].to_dict('records')


def sort_master(master_df):
    # persistence PRIMARY, then within-fold floor, then survival axis, then PF/WR
    _mdf = master_df.copy()
    _mdf['_mfp_sort'] = _mdf['min_fold_pf'].map(
        lambda v: float('inf') if (v is None or v == '' or
                                   not np.isfinite(float(v)) if isinstance(v, (int, float))
                                   else False) else float(v))
    _mdf['_pf_undefined'] = _mdf['min_fold_pf'].map(
        lambda v: bool(isinstance(v, (int, float)) and not np.isfinite(float(v))))
    return _mdf.sort_values(
        by=['folds_plus', '_mfp_sort', 'worst_day_usd', 'agg_pf', 'WR'],
        ascending=[False, False, True, False, False]).reset_index(drop=True)


PARITY_LIMIT_DEFAULT = 200


def _parity_chunk_worker(payload):
    fam, script, scope, frame_path, lo, hi = payload
    import discovery_orchestrator as orch
    df, adaptive, structural, warmup, kw = orch._worker_context(scope, frame_path, fam)
    rows, exp = orch.run_chunk_rows(fam, script, df, adaptive, structural, warmup, kw, lo, hi)
    return rows, exp


def parity_check(scope='proof', workers=1, df=None, adaptive=None, structural=None, warmup=None,
                 families=None, limit=PARITY_LIMIT_DEFAULT, frame_path=None):
    if df is None:
        raise SystemExit(
            "ABORT — parity_check requires the INGESTED frame. It must never fall back to "
            "engine.load_sealed_baseline(), which hardcodes equiDOT_recon171_step7_* and would "
            "silently test a DIFFERENT dataset from the one S0 validated. Pass df/adaptive/"
            "structural/warmup explicitly (master.py --parity does this).")
    if adaptive is None or structural is None or warmup is None:
        raise SystemExit("ABORT — parity_check requires adaptive, structural and warmup from S1/S2; "
                         "recomputing them here could diverge from the run under test.")
    builders = _scope(scope)
    names = families or [f[0] for f in FAMILIES]
    print(f"PARITY HARNESS — chunked+collated vs unchunked, scope={scope}, "
          f"limit={limit} axis units per family, workers={workers}", flush=True)
    print(f"  the SERIAL REFERENCE leg is one process BY DEFINITION — parallelising it would use the "
          f"very chunk mechanism under test. Only the chunked leg honours --workers.", flush=True)
    print(f"  both legs are bounded by the SAME limit value, so they always see an identical range.",
          flush=True)
    all_pass = True
    for fam, script, mod, fmt in FAMILIES:
        if fam not in names:
            continue
        kw = builders[fam](df, adaptive, structural, warmup)
        bounds_full, n_full = _bounds_for(fam, kw)
        n_units = min(limit, n_full) if limit else n_full
        bounds = [(lo, hi) for (lo, hi) in bounds_full if lo < n_units]
        bounds = [(lo, min(hi, n_units)) for (lo, hi) in bounds]
        est = _parity_estimate(fam, n_units, n_full)
        print(f"  {fam}: {n_units} of {n_full} axis units | {len(bounds)} chunks | {est}", flush=True)
        t0 = time.time()
        orig_s = df['D2D_Trend_Dir'].values.copy()
        try:
            if fam == 'F0':
                serial_raw, exp_one = run_f0_chunk(df, adaptive, structural, warmup, kw, 0, n_units)
                serial = f0_rows_from_raw(df, adaptive, structural, warmup, serial_raw)
            else:
                sub = _slice_axis(kw, CHUNK_AXIS[fam], 0, n_units) if n_units < n_full else dict(kw)
                serial = fmt(mod.run_search(df, adaptive=adaptive, structural=structural,
                                            warmup=warmup, **sub), script)
                exp_one = None
                if fam == 'F1':
                    exp_one = (len(sub['cond_labels']) * len(kw['cond_labels'])
                               * len(sub['lags']) * len(kw['directions']))
        finally:
            df['D2D_Trend_Dir'] = orig_s
        t_serial = time.time() - t0
        t0 = time.time()
        parts = []
        cand = 0
        payloads = [(fam, script, scope, frame_path, lo, hi) for (lo, hi) in bounds]
        if workers > 1 and frame_path is not None:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as _mp2
            with ProcessPoolExecutor(max_workers=min(workers, len(payloads)),
                                     mp_context=_mp2.get_context('spawn')) as ex:
                for rows, exp in ex.map(_parity_chunk_worker, payloads):
                    parts.extend(rows)
                    if exp is not None:
                        cand += exp
        else:
            if workers > 1:
                print(f"    chunked leg run in-process: no frame_path supplied for worker "
                      f"processes (result is identical, only slower)", flush=True)
            for lo, hi in bounds:
                rows, exp = run_chunk_rows(fam, script, df, adaptive, structural, warmup, kw, lo, hi)
                parts.extend(rows)
                if exp is not None:
                    cand += exp
        dedup_line = ''
        if fam == 'F0':
            chunked = f0_rows_from_raw(df, adaptive, structural, warmup, parts)
            a_d = pd.DataFrame(serial, columns=SCHEMA).sort_values(list(SCHEMA)).reset_index(drop=True)
            b_d = pd.DataFrame(chunked, columns=SCHEMA).sort_values(list(SCHEMA)).reset_index(drop=True)
            dedup_ok = a_d.equals(b_d)
            dedup_line = (f"    F0 COLLATION DEDUP VERIFIED: {'YES' if dedup_ok else 'NO'} — "
                          f"{len(parts)} raw survivors from chunks -> global 80% overlap dedup at "
                          f"collation -> {len(b_d)} rows; unchunked run deduped -> {len(a_d)} rows; "
                          f"byte-identical: {dedup_ok}")
            parts = chunked
        t_chunk = time.time() - t0
        a = pd.DataFrame(serial, columns=SCHEMA).sort_values(list(SCHEMA)).reset_index(drop=True)
        b = pd.DataFrame(parts, columns=SCHEMA).sort_values(list(SCHEMA)).reset_index(drop=True)
        same = a.equals(b)
        cand_txt = ''
        if exp_one is not None:
            cand_ok = cand == exp_one
            same = same and cand_ok
            cand_txt = f" | candidates serial {exp_one} vs chunked {cand} {'OK' if cand_ok else 'MISMATCH'}"
        print(f"    {len(bounds):5} chunks | serial {len(a):5} rows ({t_serial:.1f}s) | "
              f"chunked {len(b):5} rows ({t_chunk:.1f}s) | {'PASS' if same else 'FAIL'}{cand_txt}",
              flush=True)
        if dedup_line:
            print(dedup_line, flush=True)
        all_pass = all_pass and same
        del kw
    print(f"PARITY {'PASS' if all_pass else 'FAIL'}" +
          (" — chunking plus collation changes nothing a scanner computes" if all_pass
           else " — chunking altered results; do NOT run a long scan"), flush=True)
    return all_pass


def _parity_estimate(fam, n_units, n_full):
    per = {'F0': 0.40, 'F1': 0.35}.get(fam)
    if per is None:
        return "estimated runtime: seconds to a few minutes per leg"
    secs = per * n_units
    return (f"estimated runtime ~{secs:.0f}s per leg ({2 * secs:.0f}s both legs) "
            f"at ~{per}s per axis unit")


def orchestrate(scope='proof', workers=1, df=None, adaptive=None, structural=None,
                warmup=None, frame_path=None, input_sha=None, diagnostics=('F12', 'F13'),
                limit=0):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"equiDOT — discovery orchestrator | scope={scope} | workers={workers} | target lot 1.0", flush=True)
    if df is None:
        print("  no frame injected — loading the sealed baseline from the working directory", flush=True)
        df = engine.load_sealed_baseline()
        adaptive = None
        structural = None
        warmup = None
    else:
        print(f"  using the INGESTED frame from S0: {len(df):,} rows x {df.shape[1]} cols "
              f"({df['Time'].astype(str).values[0]} -> {df['Time'].astype(str).values[-1]})", flush=True)
    if warmup is None:
        warmup = engine.warmup_floor(df)
    if adaptive is None:
        adaptive = dt.compute_adaptive_thresholds(df)
    if structural is None:
        structural = dt.compute_structural_gates(df)
    builders = _scope(scope)
    f1_csv_present = os.path.exists(os.path.join(RESULTS_DIR, F1_CSV))
    pool_queued = [f[0] for f in FAMILIES if not (f[0] == 'F1' and f1_csv_present)]
    verify_family_coverage(pool_queued, list(diagnostics), input_sha, RESULTS_DIR)
    schedule = [(fam, script, mod, fmt) for fam, script, mod, fmt in FAMILIES
                if not (fam == 'F1' and f1_csv_present)]
    total = len(schedule)
    pending = []
    resumed = []
    for fam, script, mod, fmt in schedule:
        complete, meta = family_is_complete(fam, script)
        if complete:
            resumed.append((fam, script, meta))
        else:
            pending.append((fam, script, mod, fmt))
    if resumed:
        print(f"  RESUME: {len(resumed)} of {total} families already complete on disk — reading back, not re-scanning:",
              flush=True)
        for fam, script, meta in resumed:
            print(f"    [{fam}] skipped, resumed from disk ({meta['rows']} rows, sha {meta['csv_sha256'][:12]})",
                  flush=True)
    print(f"  {len(pending)} of {total} families to run this pass", flush=True)
    durations = []
    import multiprocessing as _mp
    if _mp.parent_process() is not None and workers and workers > 1:
        print("  already inside a worker process — running sequentially to prevent recursive spawn", flush=True)
        workers = 1
    ran_parallel = False
    if workers and workers >= 1 and pending and frame_path is not None:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        import multiprocessing as _mp2
        plan = []
        expected_cands = {}
        for fam, script, _mod, _fmt in pending:
            kw = builders[fam](df, adaptive, structural, warmup)
            bounds, n_axis = _bounds_for(fam, kw)
            if limit:
                n_axis = min(limit, n_axis)
                bounds = [(lo, min(hi, n_axis)) for (lo, hi) in bounds if lo < n_axis]
            if fam == 'F1':
                expected_cands[fam] = (n_axis * len(kw['cond_labels']) * len(kw['directions']))
            if fam == 'F0':
                expected_cands[fam] = f0_combo_count(kw, 0, n_axis)
            plan.append((fam, script, n_axis, bounds))
            del kw
        order = {f: i for i, f in enumerate(COST_ORDER)}
        plan.sort(key=lambda r: order.get(r[0], 999))
        queue = []
        already = 0
        for fam, script, _n, bounds in plan:
            for idx, (lo, hi) in enumerate(bounds):
                if chunk_is_complete(fam, script, idx):
                    already += 1
                    continue
                queue.append((fam, script, scope, RESULTS_DIR, frame_path, idx, lo, hi))
        total_chunks = sum(len(b) for _f, _s, _n, b in plan)
        print(f"  CHUNK PLAN: {total_chunks} chunks across {len(plan)} families "
              f"(axis split, {TARGET_CHUNKS_PER_FAMILY} chunks max per family, "
              f"independent of worker count):", flush=True)
        for fam, script, n_axis, bounds in plan:
            _sz = (bounds[0][1] - bounds[0][0]) if bounds else 0
            _geo = 'SMOKE' if SMOKE_MODE else 'REAL'
            print(f"    {fam:4} axis '{CHUNK_AXIS[fam]}' = {n_axis} items -> {len(bounds)} chunks"
                  f" | unit size {_sz} | geometry {_geo}", flush=True)
            if not SMOKE_MODE and isinstance(CHUNK_AXIS[fam], tuple) and _sz != 1:
                raise SystemExit(
                    f'ABORT [chunk geometry] {fam} is a tuple-axis family and a REAL run must '
                    f'produce unit size 1 ({n_axis} single-unit chunks). It produced size {_sz} '
                    f'in {len(bounds)} chunks. A larger F1 chunk cost ~1,900s against ~58s for a '
                    f'single unit, so TOTAL CPU WORK MORE THAN DOUBLED (116 -> 256 CPU-hours) - '
                    f'this is visible in the first minute instead of at hour two.')
        if already:
            print(f"  RESUME: {already} of {total_chunks} chunks already complete on disk", flush=True)
        nw = min(workers, max(1, len(queue)))
        print(f"  running {len(queue)} pending chunks across {nw} worker processes from ONE queue — "
              f"a worker that finishes takes the next chunk of ANY family, so no thread idles while "
              f"work remains and the last family gets every worker", flush=True)
        print(f"  submission order is longest-family-first (scheduling only; collation is by axis order, "
              f"so output cannot depend on it)", flush=True)
        fam_secs = {}
        fam_units = {}
        pend_units = {}
        for pl in queue:
            pend_units[pl[0]] = pend_units.get(pl[0], 0) + (pl[7] - pl[6])
        t0 = time.time()
        died = False
        try:
            with _Heartbeat(f"S3 chunk queue ({len(queue)} chunks)"):
                with ProcessPoolExecutor(max_workers=nw,
                                         mp_context=_mp2.get_context('spawn')) as ex:
                    futures = {ex.submit(_chunk_worker, pl): (pl[0], pl[5]) for pl in queue}
                    done_n = 0
                    for fut in as_completed(futures):
                        fam, idx = futures[fut]
                        try:
                            fam_r, idx_r, n_rows, secs, units = fut.result()
                        except BrokenProcessPool:
                            died = True
                            break
                        except Exception as exc:
                            print(f"  [{fam} c{idx:04d}] worker raised {type(exc).__name__}: {exc}", flush=True)
                            continue
                        done_n += 1
                        if n_rows >= 0:
                            fam_secs[fam_r] = fam_secs.get(fam_r, 0.0) + secs
                            fam_units[fam_r] = fam_units.get(fam_r, 0) + units
                        pend_units[fam_r] = pend_units.get(fam_r, 0) - units
                        el = time.time() - t0
                        pct = 100.0 * done_n / len(queue)
                        serial = 0.0
                        unmeasured = []
                        for f_, u_ in pend_units.items():
                            if u_ <= 0:
                                continue
                            if fam_units.get(f_):
                                serial += u_ * (fam_secs[f_] / fam_units[f_])
                            else:
                                unmeasured.append(f_)
                        eta_txt = (f"ETA {_hms(serial / nw)}" if not unmeasured
                                   else f"ETA >= {_hms(serial / nw)} ({len(unmeasured)} unmeasured: "
                                        f"{','.join(sorted(unmeasured))})")
                        if serial <= 0 and unmeasured:
                            eta_txt = f"ETA forming ({len(unmeasured)} families unmeasured)"
                        note = 'resumed' if n_rows < 0 else f'{n_rows} survivors in {secs:.1f}s'
                        print(f"  [{done_n}/{len(queue)} {pct:5.1f}%] {fam_r} c{idx_r:04d} {note} "
                              f"| elapsed {_hms(el)} | {eta_txt} "
                              f"| {done_n / el * 60 if el > 0 else 0:.1f} chunks/min", flush=True)
        except BrokenProcessPool:
            died = True
        if died:
            print("", flush=True)
            print("  *** A WORKER PROCESS DIED WITHOUT RAISING — almost always the OS killing it for memory. ***",
                  flush=True)
            print("  Completed CHUNKS are on disk and will NOT be re-scanned. Completing the rest",
                  flush=True)
            print("  sequentially in this process. If this recurs, lower --workers.", flush=True)
            print("", flush=True)
        print('  PER-FAMILY CHUNK COMPLETENESS (before collation is attempted):', flush=True)
        incomplete = {}
        for fam, script, _n, bounds in plan:
            missing = [i for i in range(len(bounds)) if not chunk_is_complete(fam, script, i)]
            state = 'complete' if not missing else f'MISSING {missing[:20]}'
            if missing and len(missing) > 20:
                state += f' ... and {len(missing) - 20} more'
            print(f'    {fam:4} {len(bounds) - len(missing)}/{len(bounds)} {state}', flush=True)
            if missing:
                incomplete[fam] = missing
        if incomplete:
            print('  RE-QUEUEING ONLY THE MISSING CHUNKS — never a sequential re-search of a '
                  'chunked family. One missing F1 chunk once triggered a full 1,713,630-candidate '
                  'single-process re-search (~17 days at the measured rate); the resume path did '
                  'the same work in 5m23s.', flush=True)
            script_of = {p0: p1 for p0, p1, _a, _b in plan}
            bounds_of = {p0: p3 for p0, _p1, _a, p3 in plan}
            still = dict(incomplete)
            for _pass in range(1, CHUNK_RETRY_PASSES + 1):
                requeue = []
                for f, ix in still.items():
                    for i in ix:
                        lo, hi = bounds_of[f][i]
                        requeue.append((f, script_of[f], scope, RESULTS_DIR, frame_path,
                                        i, lo, hi))
                if not requeue or frame_path is None:
                    break
                print(f'  RETRY PASS {_pass}/{CHUNK_RETRY_PASSES}: re-queueing '
                      f'{len(requeue)} chunk(s) {dict((f, ix[:8]) for f, ix in still.items())}',
                      flush=True)
                from concurrent.futures import ProcessPoolExecutor as _PPE
                import multiprocessing as _mp3
                nw2 = min(workers, len(requeue))
                try:
                    with _PPE(max_workers=nw2, mp_context=_mp3.get_context('spawn')) as ex2:
                        for _r in ex2.map(_chunk_worker, requeue):
                            pass
                except Exception as _re:
                    print(f'    retry pass {_pass} pool error: {type(_re).__name__}: '
                          f'{str(_re)[:90]}', flush=True)
                nxt = {}
                for f, ix in still.items():
                    left = [i for i in ix if not chunk_is_complete(f, script_of[f], i)]
                    if left:
                        nxt[f] = left
                recovered = sum(len(v) for v in still.values()) - sum(len(v) for v in nxt.values())
                print(f'    pass {_pass} recovered {recovered} chunk(s); '
                      f'{sum(len(v) for v in nxt.values())} still missing', flush=True)
                still = nxt
                if not still:
                    print('  ALL MISSING CHUNKS RECOVERED — collation proceeds with a complete set.',
                          flush=True)
                    break
                if _pass < CHUNK_RETRY_PASSES:
                    time.sleep(CHUNK_RETRY_BACKOFF_S * _pass)
            if still:
                print('', flush=True)
                print('  *** CHUNKS STILL MISSING AFTER '
                      f'{CHUNK_RETRY_PASSES} RETRY PASSES ***', flush=True)
                for f, ix in still.items():
                    print(f'      {f}: {len(ix)} missing, indices {ix[:20]}'
                          + (f' ... and {len(ix) - 20} more' if len(ix) > 20 else ''), flush=True)
                print('  THESE FAMILIES WILL NOT BE COLLATED AND WILL NOT BE MARKED DONE. A '
                      'silently short collation is worse than a visible failure: the candidate '
                      'invariant would abort later with no context about which chunks were '
                      'absent. Re-run and only these indices are re-queued.', flush=True)
                print('', flush=True)
        _short = set(still) if incomplete else set()
        for fam, script, _n, bounds in plan:
            if fam in _short:
                print(f'  {fam}: COLLATION SKIPPED — {len(_short and still.get(fam, []))} chunk(s) '
                      f'absent after {CHUNK_RETRY_PASSES} retry passes. Not marked done.',
                      flush=True)
                continue
            if fam == 'F0':
                ok, n_rows = collate_f0(script, len(bounds), df, adaptive, structural, warmup,
                                        expected_cands.get('F0'), input_sha)
            else:
                ok, n_rows = collate_family_chunks(fam, script, len(bounds),
                                                   expected_cands.get(fam))
            if ok:
                inv = candidate_invariant(fam, script, len(bounds), expected_cands.get(fam))[1]
                print(f"  [{fam}] collated {len(bounds)} chunks -> {n_rows} rows "
                      f"| candidate invariant {inv}", flush=True)
                if input_sha is not None:
                    stamp_provenance(_family_paths(fam, script)[0], input_sha)
        ran_parallel = not died
        pending = [(fam, script, mod, fmt) for fam, script, mod, fmt in pending
                   if not family_is_complete(fam, script)[0]]
        if pending and ran_parallel:
            # A FAMILY JUDGED STALE MUST RE-ENTER THE CHUNKED PATH, NOT A SERIAL ONE.
            # Clearing ran_parallel dropped every still-pending family into the
            # single-process run_family loop below: F1 re-ran as one in-process
            # f1.run() over all 1,713,630 candidates on ONE CORE with no progress
            # line, instead of 3,585 chunks across 14 workers. That is the 24-hour
            # F1 the chunk queue and the entry-month partition both exist to remove.
            print(f'  {len(pending)} family/ies still pending after the chunked pass: '
                  f'{[p[0] for p in pending]}. RE-ENTERING THE CHUNKED PATH rather than '
                  f'falling back to the serial loop - a serial F1 is the 24-hour version.',
                  flush=True)
            _stale_names = {p[0] for p in pending}
            for _f2, _s2, _m2, _fm2 in list(pending):
                _kw2 = builders[_f2](df, adaptive, structural, warmup)
                _b2, _n2 = _bounds_for(_f2, _kw2)
                print(f'    {_f2:4} re-scan via the QUEUE: {_n2} units -> {len(_b2)} chunks',
                      flush=True)
            ran_parallel = True
            _requeue_pass = True
    if pending and not ran_parallel:
        if frame_path is not None and workers and workers > 1:
            print(f'  REFUSING THE SERIAL FALLBACK for {[p[0] for p in pending]}: the chunked '
                  f'path is available (frame_path set, workers={workers}). A serial re-scan of a '
                  f'chunked family is not a fallback, it is a different and far slower '
                  f'algorithm. Re-run to re-enter the queue.', flush=True)
        for i, (fam, script, mod, fmt) in enumerate(pending, 1):
            mean = (sum(durations) / len(durations)) if durations else None
            eta = f" | ETA {_hms(mean * (len(pending) - i + 1))}" if mean else ""
            print(f"  [family {i} of {len(pending)}] {fam} ({script}) starting{eta}", flush=True)
            t0 = time.time()
            with _Heartbeat(f"{fam} ({script})"):
                run_family(fam, script, mod, fmt, builders[fam], df, adaptive, structural,
                           warmup, limit=limit)
            durations.append(time.time() - t0)
            print(f"  [family {i} of {len(pending)}] {fam} done in {_hms(durations[-1])}", flush=True)
    all_rows = []
    for fam, script, _mod, _fmt in schedule:
        complete, meta = family_is_complete(fam, script)
        if not complete:
            print(f"  [{fam}] WARNING: no complete output on disk after this pass; excluded from collation",
                  flush=True)
            continue
        rows = resume_family(fam, script)
        if rows is None:
            print(f"  [{fam}] WARNING: output missing schema columns; excluded from collation", flush=True)
            continue
        all_rows.extend(rows)
    if f1_csv_present:
        all_rows.extend(ingest_f1())
    f0_collated = family_is_complete('F0', 'triple_convergence_and_d2ddir')[0]
    if f0_collated:
        print('  [F0] chunked path already collated F0 into the pool — SKIPPING ingest_f0(). '
              'collate_f0 writes the same filename F0_CSV that ingest_f0 reads, so running both '
              'double-counted F0 (19,757 -> 39,514) and inflated the trial count that spec H.1 '
              'uses for the empirical null and Benjamini-Yekutieli, making the multiple-testing '
              'bar harder than the search warranted.', flush=True)
    else:
        all_rows.extend(ingest_f0())
    master = pd.DataFrame(all_rows, columns=SCHEMA)
    master_path = os.path.join(RESULTS_DIR, "discovery_master.csv")
    _write_atomic_csv(sort_master(master), master_path)
    write_pool_note(master_path, df)
    print(f"\nCollated {len(master)} candidates -> {master_path} "
          f"(sorted: folds_plus, min_fold_pf, worst_day_usd, agg_pf, WR; no rows dropped)", flush=True)
    by_fam = master.groupby('family').size().to_dict()
    for fam, script, _mod, _fmt in FAMILIES:
        ok, meta = family_is_complete(fam, script)
        if not ok or meta is None:
            continue
        got = int(by_fam.get(fam, 0))
        want = int(meta.get('rows', got))
        if got != want:
            raise SystemExit(
                f'ABORT [{fam}] per-family pool count {got} != collated row count {want}. A family '
                f'is being counted more than once (or lost) between collation and the pool; the '
                f'trial count that feeds spec H.1 would be wrong.')
    print(f"Per-family counts: {by_fam}", flush=True)


# ── F0 NOTE ──────────────────────────────────────────────────────────────
# F0 (triple_convergence_and_d2ddir.py) is run SEPARATELY at full scope with
# its internal MIN_PF pre-gate = 2.0 (trim only), then converted to the
# common SCHEMA and saved as discovery_results/results_F0_..._d2ddir.csv,
# which this orchestrator ingests. F0 is not called in-process because the
# C(117,3) triple search must not be held in one process with the others.


if __name__ == '__main__':
    orchestrate(sys.argv[1] if len(sys.argv) > 1 else 'proof')
