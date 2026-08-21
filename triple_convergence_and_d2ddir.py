import pandas as pd
import numpy as np

PF_UNDEFINED = float('nan')
from itertools import combinations
import time
import os
import sys
import dots_thresholds as dt
import portfolio_simulation_engine as engine
import wf

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

PARTS = [f"equiDOT_recon171_step7_part{i}.csv" for i in range(1, 9)]
OUTPUT_DIR = "dots_results"
DOTS_INITBARS = 6900
SPREAD = 3.0
RISK_MULT = 2.0
MAX_RISK = 150.0
STEP_PCT = 0.30
BE_TRIG_FRAC = 1.0
LOCK_FRAC = 1.0
MIN_TRADES = 10
EMIT_ALL = False
# --emit-all. DEFAULTS OFF AND THE DEFAULT PATH MUST STAY BYTE-REPRODUCIBLE: six
# specifications and every figure in the R&D plan rest on the 19,754-row output, so if it
# moves they all become unverifiable.
#
# WHY IT EXISTS: 14 of the adopted 297 signals appear in NO scanner output at all - no
# agg_pf, no folds_plus, no trade count, not a row in any file - because they fire fewer
# than MIN_TRADES times. Every analysis seat that looked for them burned turns concluding
# they "do not exist". AND SIX OF THE FOURTEEN ARE LOAD-BEARING: spec v3 §4.2 measures
# removing all fourteen at $3,923 and three trading days, more than their own trades earn,
# because they raise depth and let other signals clear the floor. A SELECTION PROCEDURE
# THAT CANNOT SEE THEM IS STRUCTURALLY BLIND TO A REAL PART OF THE BOOK.


MIN_PF = 2.0
# 2.0, NOT 4.0. THE EFFECTIVE VALUE LIVED IN THREE PLACES AND THIS FILE DISAGREED WITH ALL
# OF THEM: discovery_orchestrator set F0_MIN_PF_OVERRIDE = 2.0, run_f0_full.py set
# f0.MIN_PF = 2.0, the non-negotiables and master_stage_spec both record 2.0 - and an
# analyst reading THIS file saw 4.0 and concluded the field was PF-gated at 4. One already
# did. THE 19,754 WERE PRODUCED UNDER 2.0, so setting it here and removing the override
# machinery changes no output; it makes the effective threshold visible where it is read.
OVERLAP_THRESHOLD = 0.80


def _min_trades():
    """0 under --emit-all, so every signal is emitted regardless of trade count."""
    return 0 if EMIT_ALL else MIN_TRADES


def _min_pf():
    """0.0 under --emit-all. LIFTING MIN_TRADES ALONE DEFEATS THE FLAG: a sub-10-trade
    signal whose PF sits below the threshold stays invisible, which is exactly the
    population the 14 orphans live in."""
    return 0.0 if EMIT_ALL else MIN_PF


def _overlap_threshold():
    """1.01 under --emit-all - above any achievable overlap ratio, so nothing is dropped.

    DEDUP DOES DROP SIGNALS: deduplicate() writes deduped_survivors.csv and discards any
    signal sharing more than OVERLAP_THRESHOLD of its entry bars with a higher-PF one. An
    orphan that survives the trade and PF gates could still vanish here, so --emit-all
    lifts it. The DEFAULT path is untouched and still dedups at 0.80, and run_search's own
    output is unaffected either way - dedup writes a SEPARATE artifact.
    """
    # 1.01, NOT 1.00: the test is `> threshold`, and a signal whose entry set is a strict
    # subset of a kept one gives a ratio of exactly 1.0. At 1.00 that comparison is False
    # and the signal survives by luck of the operator; at 1.01 nothing can exceed it and
    # the lift is total. ABOVE ANY ACHIEVABLE RATIO, NOT EQUAL TO IT.
    return 1.01 if EMIT_ALL else OVERLAP_THRESHOLD
CHUNK_SIZE = 10000

EQUALITY_CANDIDATES = [
    'D2D_Signal', 'D2D_DirStep', 'OBVf_Signal', 'OBVf_Trend_Dir',
    'Harmonic_OBVf_Concordance', 'Harmonic_D2D_Concordance', 'ADX_Rising',
    'Sqz_State', 'RangeOsc_State', 'PoC_Side', 'ST_Flip_Event', 'Trend_Concordance',
    'Trend_Conflict', 'AT_Regime_ST', 'AT_Regime_LT', 'VWAP_Side', 'VAH_Side',
    'VAL_Side', 'PrevDay_High_Side', 'PrevDay_Low_Side', 'PrevDay_Close_Side',
    'DailyOpen_Side', 'OR_High_Side', 'OR_Low_Side', 'Session_High_Side',
    'Session_Low_Side', 'WeeklyOpen_Side',
]
EXCLUDE_REFERENCE = {
    'Open', 'High', 'Low', 'Close', 'EST_Hour', 'EST_Minute', 'EST_DayOfWeek',
    'D2D_Trend_Dir', 'D2D_Trend', 'D2D_Upper_Band', 'D2D_Lower_Band', 'D2D_Basis',
    'D2D_UpTrend_Trail', 'D2D_DownTrend_Trail', 'OBVf_Trend', 'OBVf_Upper_Band',
    'OBVf_Lower_Band', 'OBVf_Basis', 'OBVf_ATR', 'OBVf_ATR_MA', 'OBVf_DirStep',
    'OBVf_Persist', 'OBVf_UpTrend_Trail', 'OBVf_DownTrend_Trail', 'OBV_Line',
    'OBV_Line_Prev', 'OBV_Accum', 'OBV_Fast', 'OBV_Slow', 'OBV_Zero_Value', 'TChan_B5',
    'KAMA_Value', 'KAMA_Side', 'HarmVol_EMA8', 'HarmVol_EMA21', 'Harmonic_Sign',
    'ATR_Assigned', 'Hist_Volume', 'PoC_Price', 'DecayState_ST', 'DecayState_LT',
    'Lock_Time', 'VWAP_Price', 'VAH_Price', 'VAL_Price', 'PrevDay_High', 'PrevDay_Low',
    'PrevDay_Close', 'DailyOpen_Price', 'OR_High', 'OR_Low', 'Session_High',
    'Session_Low', 'WeeklyOpen_Price',
}

# ═══════════════════════════════════════════════════════════════
#  8-PART SEALED-BASELINE LOADER (shared, identical across tools)
# ═══════════════════════════════════════════════════════════════

def load_sealed_baseline(verbose=True):
    hdr = list(pd.read_csv(PARTS[0], nrows=0).columns)
    frames = [pd.read_csv(PARTS[0])]
    for p in PARTS[1:]:
        frames.append(pd.read_csv(p, header=None, names=hdr))
    df = pd.concat(frames, ignore_index=True)
    assert df.shape == (152983, 171), f"baseline shape {df.shape} != (152983, 171)"
    times = df['Time'].values
    assert (times[1:] > times[:-1]).all(), "Time not strictly increasing"
    assert int(df.duplicated().sum()) == 0, "duplicate rows present"
    assert int(df.isna().sum().sum()) == 0, "NaN present in baseline"
    vsa = pd.Series(dt._vwap_sigma_atr(df, df['ATR_1M'].values.astype(float)),
                    index=df.index, name='VWAP_Sigma_ATR')
    df = pd.concat([df, vsa], axis=1)
    if verbose:
        print(f"Baseline loaded: {df.shape[0]} rows x 171 cols (+VWAP_Sigma_ATR derived)")
        print(f"  Time strictly increasing: OK | 0 duplicate rows | 0 NaN")
        print(f"  Range: {df['Time'].iloc[0]} -> {df['Time'].iloc[-1]}")
    return df

def warmup_floor(df, verbose=True):
    eligible = (df['ADX_Value'].values >= 15) & (df['Volume'].values > 50)
    cum = np.cumsum(eligible)
    sat = int(np.argmax(cum >= dt._ROLL_CAP))
    chosen = max(DOTS_INITBARS, sat)
    if verbose:
        print(f"Warmup-trim: Dots_InitBars floor={DOTS_INITBARS}, ring-saturation bar={sat} "
              f"({df['Time'].iloc[sat]}); chosen={chosen}")
        print(f"  First scannable Time: {df['Time'].iloc[chosen]}")
    return chosen

# ═══════════════════════════════════════════════════════════════
#  CANDIDATE VOCABULARY + PARTITION GUARD (A2.7 completeness)
# ═══════════════════════════════════════════════════════════════

def build_candidates(df):
    feat_candidates = list(dt._D_COLS) + ['VWAP_Z', 'OR_Position']
    assert len(feat_candidates) == 90, f"FEAT candidates {len(feat_candidates)} != 90"
    assert len(EQUALITY_CANDIDATES) == 27, f"EQUALITY candidates {len(EQUALITY_CANDIDATES)} != 27"
    baseline_cols = [c for c in df.columns if c != 'VWAP_Sigma_ATR']
    feat_in_base = [c for c in feat_candidates if c in baseline_cols]
    feat_not_base = [c for c in feat_candidates if c not in baseline_cols]
    assert feat_not_base == ['VWAP_Sigma_ATR'], f"unexpected non-baseline FEAT: {feat_not_base}"
    exclude = set(baseline_cols) - set(feat_in_base) - set(EQUALITY_CANDIDATES) - {'Time'}
    assert exclude == EXCLUDE_REFERENCE, (
        f"EXCLUDE complement mismatch: extra={exclude - EXCLUDE_REFERENCE}, "
        f"missing={EXCLUDE_REFERENCE - exclude}")
    partition = set(feat_in_base) | set(EQUALITY_CANDIDATES) | exclude | {'Time'}
    assert partition == set(baseline_cols), "partition does not cover all 171 columns"
    assert len(feat_in_base) + len(EQUALITY_CANDIDATES) + len(exclude) + 1 == 171
    overlap = (set(feat_in_base) & set(EQUALITY_CANDIDATES)) | (set(feat_in_base) & exclude) | (set(EQUALITY_CANDIDATES) & exclude)
    assert not overlap, f"candidate overlap: {overlap}"
    print(f"Candidate partition OK: 90 FEAT_ (89 baseline + VWAP_Sigma_ATR derived) + "
          f"27 equality + {len(exclude)} excluded + Time = 171 (no overlap, no gap)")
    return feat_candidates, list(EQUALITY_CANDIDATES)

# ═══════════════════════════════════════════════════════════════
#  UNIFIED CONDITION LIST (oracle-only FEAT_ thresholds + equality)
# ═══════════════════════════════════════════════════════════════

def build_conditions(df, feat_candidates, equality_candidates, adaptive, structural, eligible):
    feature_conditions = {}
    n_hilo = 0
    for feat in feat_candidates:
        hi = df[feat].values > adaptive[(feat, 'hi')]
        lo = df[feat].values < adaptive[(feat, 'lo')]
        if feat == 'OR_Position':
            hi = hi & structural['OR_Position']
            lo = lo & structural['OR_Position']
        feature_conditions[feat] = [('hi', hi), ('lo', lo)]
        n_hilo += 2
    print(f"\nFEAT_ conditions: {len(feat_candidates)} features x (hi,lo) = {n_hilo}")
    print("Equality values enumerated DATA-DRIVEN from the post-warmup scannable "
          "eligible baseline (neutral 0 retained where live):")
    n_eq = 0
    n_warm_only = 0
    nonwarm = np.arange(len(df)) >= warmup_floor(df, verbose=False)
    scannable = eligible & nonwarm
    for feat in equality_candidates:
        vals_all = sorted(set(int(v) for v in df[feat].values[eligible]))
        vals_scan = sorted(set(int(v) for v in df[feat].values[scannable]))
        warm_only = [v for v in vals_all if v not in vals_scan]
        n_warm_only += len(warm_only)
        conds = []
        for v in vals_scan:
            conds.append((f'=={v}', df[feat].values == v))
        feature_conditions[feat] = conds
        n_eq += len(conds)
        flag = f"  [warmup-only, excluded: {warm_only}]" if warm_only else ""
        print(f"  {feat}: {vals_scan}{flag}")
    total = n_hilo + n_eq
    status = "MATCHES S.10 (90x2 + 69 = 249)" if total == 249 else "DIFFERS from S.10 249 — investigate"
    print(f"\nResolved conditions: {n_hilo} FEAT_ (hi/lo) + {n_eq} equality = {total} scan conditions")
    print(f"  ({n_warm_only} warmup-only equality value(s) excluded) -> {status}")
    return feature_conditions

# ═══════════════════════════════════════════════════════════════
#  FAST SCAN SCORER (per-trade TM behavior-identical to the engine)
# ═══════════════════════════════════════════════════════════════

def simulate_signal(entry_indices, direction, highs, lows, closes, atrs, day_of_week, est_hour, est_minute):
    trades = []
    n_bars = len(highs)
    last_exit = -1
    for entry_idx in entry_indices:
        if entry_idx <= last_exit:
            continue
        entry_price = closes[entry_idx]
        raw_risk = atrs[entry_idx] * RISK_MULT
        initial_risk = min(raw_risk, MAX_RISK)
        if initial_risk <= 0:
            continue
        step_size = STEP_PCT * initial_risk
        be_trigger = BE_TRIG_FRAC * step_size
        be_lock_dist = LOCK_FRAC * be_trigger
        if direction == 1:
            sl_price = entry_price - initial_risk
        else:
            sl_price = entry_price + initial_risk
        tiers = 0
        be_nudged = False
        current_sl = sl_price
        exit_bar = -1
        exit_price = 0.0
        exit_type = ''
        for j in range(entry_idx + 1, n_bars):
            h = highs[j]; l = lows[j]; c = closes[j]
            dow = day_of_week[j]; hr = est_hour[j]; mn = est_minute[j]
            if direction == 1:
                if l <= current_sl:
                    exit_bar = j; exit_price = current_sl
                    exit_type = ('BE' if tiers < 3 else 'LF') if be_nudged else 'SL'
                    break
            else:
                if h >= current_sl:
                    exit_bar = j; exit_price = current_sl
                    exit_type = ('BE' if tiers < 3 else 'LF') if be_nudged else 'SL'
                    break
            if dow == 5 and (hr > 16 or (hr == 16 and mn >= 45)):
                exit_bar = j; exit_price = c; exit_type = 'FC'; break
            if direction == 1:
                target = entry_price + (tiers + 1) * step_size
                while h >= target:
                    tiers += 1; target = entry_price + (tiers + 1) * step_size
            else:
                target = entry_price - (tiers + 1) * step_size
                while l <= target:
                    tiers += 1; target = entry_price - (tiers + 1) * step_size
            if not be_nudged:
                unrealised = (h - entry_price) if direction == 1 else (entry_price - l)
                if unrealised >= be_trigger:
                    if direction == 1:
                        new_sl = entry_price + be_lock_dist
                        if new_sl > current_sl: current_sl = new_sl
                    else:
                        new_sl = entry_price - be_lock_dist
                        if new_sl < current_sl: current_sl = new_sl
                    be_nudged = True
            if tiers >= 3:
                if direction == 1:
                    trail = entry_price + (tiers - 2) * step_size
                    if trail > current_sl: current_sl = trail
                else:
                    trail = entry_price - (tiers - 2) * step_size
                    if trail < current_sl: current_sl = trail
        if exit_bar == -1:
            exit_bar = n_bars - 1; exit_price = closes[exit_bar]; exit_type = 'EOD'
        pnl = (exit_price - entry_price - SPREAD) if direction == 1 else (entry_price - exit_price - SPREAD)
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': exit_bar,
            'entry_price': entry_price, 'exit_price': exit_price,
            'direction': direction, 'pnl': pnl, 'exit_type': exit_type,
            'tiers': tiers, 'be_nudged': be_nudged,
            'initial_risk': initial_risk, 'be_lock_dist': be_lock_dist,
        })
        last_exit = exit_bar
    return trades

def compute_metrics(trades):
    if len(trades) < _min_trades():
        return None
    pnls = np.array([t['pnl'] for t in trades])
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls <= 0].sum())
    pf = PF_UNDEFINED if gross_losses == 0 and gross_wins > 0 else (0.0 if gross_losses == 0 else gross_wins / gross_losses)
    n = len(trades)
    wins = int(np.sum(pnls > 0))
    wr = wins / n if n > 0 else 0
    sl_count = sum(1 for t in trades if t['exit_type'] == 'SL')
    be_count = sum(1 for t in trades if t['exit_type'] == 'BE')
    lf_count = sum(1 for t in trades if t['exit_type'] == 'LF')
    fc_count = sum(1 for t in trades if t['exit_type'] == 'FC')
    be_losses = sum(1 for t in trades if t['exit_type'] == 'BE' and t['pnl'] <= 0)
    lf_losses = sum(1 for t in trades if t['exit_type'] == 'LF' and t['pnl'] <= 0)
    sl_wins = sum(1 for t in trades if t['exit_type'] == 'SL' and t['pnl'] > 0)
    t1_plus = sum(1 for t in trades if t['tiers'] >= 1)
    t1_pct = t1_plus / n if n > 0 else 0
    return {
        'trades': n, 'pf': round(pf, 2),
        'wr': round(wr * 100, 1), 't1_pct': round(t1_pct * 100, 1),
        'total_pnl': round(float(pnls.sum()), 1), 'avg_pnl': round(float(pnls.mean()), 1),
        'sl': sl_count, 'be': be_count, 'lf': lf_count, 'fc': fc_count,
        'be_losses': be_losses, 'lf_losses': lf_losses, 'sl_wins': sl_wins,
    }

# ═══════════════════════════════════════════════════════════════
#  PARITY CHECK (scanner per-trade TM == engine per-trade TM)
# ═══════════════════════════════════════════════════════════════

def _engine_single_trade(entry_idx, direction, highs, lows, closes, atrs, dow, hr, mn):
    n_bars = len(highs)
    raw_risk = atrs[entry_idx] * RISK_MULT
    initial_risk = min(raw_risk, MAX_RISK)
    if initial_risk <= 0:
        return None
    tr = engine.Trade(0, entry_idx, closes[entry_idx], direction, initial_risk)
    for bar in range(entry_idx + 1, n_bars):
        h = highs[bar]; l = lows[bar]; c = closes[bar]; ep = tr.entry_price; d = tr.direction
        if d == 1 and l <= tr.current_sl:
            ex = tr.current_sl; et = ('BE' if tr.tiers < 3 else 'LF') if tr.be_nudged else 'SL'
            return (bar, round((ex - ep - SPREAD), 6), et)
        if d == -1 and h >= tr.current_sl:
            ex = tr.current_sl; et = ('BE' if tr.tiers < 3 else 'LF') if tr.be_nudged else 'SL'
            return (bar, round((ep - ex - SPREAD), 6), et)
        if dow[bar] == 5 and (hr[bar] > 16 or (hr[bar] == 16 and mn[bar] >= 45)):
            return (bar, round(((c - ep - SPREAD) if d == 1 else (ep - c - SPREAD)), 6), 'FC')
        if d == 1:
            target = ep + (tr.tiers + 1) * tr.step_size
            while h >= target:
                tr.tiers += 1; target = ep + (tr.tiers + 1) * tr.step_size
        else:
            target = ep - (tr.tiers + 1) * tr.step_size
            while l <= target:
                tr.tiers += 1; target = ep - (tr.tiers + 1) * tr.step_size
        if not tr.be_nudged:
            unr = (h - ep) if d == 1 else (ep - l)
            if unr >= tr.be_trigger:
                if d == 1:
                    ns = ep + tr.be_lock_dist
                    if ns > tr.current_sl: tr.current_sl = ns
                else:
                    ns = ep - tr.be_lock_dist
                    if ns < tr.current_sl: tr.current_sl = ns
                tr.be_nudged = True
        if tr.tiers >= 3:
            if d == 1:
                tl = ep + (tr.tiers - 2) * tr.step_size
                if tl > tr.current_sl: tr.current_sl = tl
            else:
                tl = ep - (tr.tiers - 2) * tr.step_size
                if tl < tr.current_sl: tr.current_sl = tl
    return (n_bars - 1, round(((closes[-1] - tr.entry_price - SPREAD) if direction == 1 else (tr.entry_price - closes[-1] - SPREAD)), 6), 'EOD')

def parity_check(df, feature_conditions, entry_allowed, d2d_dir, arrays, n_signals=5):
    print("\nPARITY CHECK (scanner vs engine per-trade TM on identical entries):")
    highs, lows, closes, atrs, dow, hr, mn = arrays
    feats = list(feature_conditions.keys())
    checked = 0
    mism = 0
    rng = np.random.RandomState(0)
    while checked < n_signals:
        combo = rng.choice(len(feats), 3, replace=False)
        f1, f2, f3 = feats[combo[0]], feats[combo[1]], feats[combo[2]]
        l1, m1 = feature_conditions[f1][rng.randint(len(feature_conditions[f1]))]
        l2, m2 = feature_conditions[f2][rng.randint(len(feature_conditions[f2]))]
        l3, m3 = feature_conditions[f3][rng.randint(len(feature_conditions[f3]))]
        direction = 1 if rng.randint(2) == 0 else -1
        mask = m1 & m2 & m3 & entry_allowed & (d2d_dir == direction)
        ei = np.where(mask)[0]
        if len(ei) < _min_trades():
            continue
        inline = simulate_signal(ei, direction, highs, lows, closes, atrs, dow, hr, mn)
        ok = True
        for t in inline:
            ref = _engine_single_trade(t['entry_idx'], direction, highs, lows, closes, atrs, dow, hr, mn)
            if ref is None:
                continue
            if ref[0] != t['exit_idx'] or ref[2] != t['exit_type'] or abs(ref[1] - round(t['pnl'], 6)) > 1e-6:
                ok = False; break
        checked += 1
        if not ok:
            mism += 1
            print(f"  MISMATCH on {f1}({l1})+{f2}({l2})+{f3}({l3}) {('LONG' if direction==1 else 'SHORT')}")
        else:
            print(f"  OK {f1}({l1})+{f2}({l2})+{f3}({l3}) {('LONG' if direction==1 else 'SHORT')} | {len(inline)} trades match")
    if mism:
        print(f"  PARITY DIVERGENCE: {mism}/{checked} signals differ — FLAG to Supervisor.")
    else:
        print(f"  PARITY OK: {checked}/{checked} signals match (scanner TM == engine TM).")
    return mism == 0

# ═══════════════════════════════════════════════════════════════
#  COMBINATORIAL SEARCH over the unified condition list
# ═══════════════════════════════════════════════════════════════

def run_search(df, feature_conditions, all_features, entry_allowed, d2d_dir, arrays):
    print("\n" + "=" * 60)
    print("COMBINATORIAL SEARCH (triple convergence + D2D gate)")
    print("=" * 60)
    highs, lows, closes, atrs, dow, hr, mn = arrays
    n_feats = len(all_features)
    combos = list(combinations(range(n_feats), 3))
    n_combos = len(combos)
    total_variants = sum(
        len(feature_conditions[all_features[i]]) * len(feature_conditions[all_features[j]]) *
        len(feature_conditions[all_features[k]]) * 2
        for (i, j, k) in combos)
    print(f"Features: {n_feats} | C({n_feats},3) = {n_combos} feature-triples")
    print(f"Total directional variants (condition products x 2 dirs): {total_variants:,}")
    survivors = []
    tested = 0; simulated = 0; passed = 0
    t_start = time.time(); last_report = t_start
    for ci, (i, j, k) in enumerate(combos):
        fi, fj, fk = all_features[i], all_features[j], all_features[k]
        for (li, mi) in feature_conditions[fi]:
            for (lj, mj) in feature_conditions[fj]:
                base_ijk = mi & mj & entry_allowed
                for (lk, mk) in feature_conditions[fk]:
                    base = base_ijk & mk
                    for direction in (1, -1):
                        tested += 1
                        signal_mask = base & (d2d_dir == direction)
                        entry_indices = np.where(signal_mask)[0]
                        if len(entry_indices) < _min_trades():
                            continue
                        simulated += 1
                        trades = simulate_signal(entry_indices, direction, highs, lows, closes, atrs, dow, hr, mn)
                        metrics = compute_metrics(trades)
                        if metrics is None:
                            continue
                        if metrics['sl_wins'] > 0:
                            print(f"\nFATAL BUG: {metrics['sl_wins']} SL wins on {fi}+{fj}+{fk}")
                            sys.exit(1)
                        if metrics['pf'] >= _min_pf():
                            survivors.append({
                                'feat_1': fi, 'thresh_1': li,
                                'feat_2': fj, 'thresh_2': lj,
                                'feat_3': fk, 'thresh_3': lk,
                                'direction': 'LONG' if direction == 1 else 'SHORT',
                                'entry_indices': entry_indices.tolist(),
                                **metrics,
                            })
                            passed += 1
        now = time.time()
        if now - last_report > 30:
            elapsed = now - t_start
            pct = (ci + 1) / n_combos * 100
            print(f"  {pct:.1f}% combos | {tested:,} tested | {simulated:,} simmed | "
                  f"{passed} passed | {elapsed:.0f}s")
            last_report = now
    elapsed = time.time() - t_start
    print(f"\nSearch complete in {elapsed:.1f}s")
    print(f"Tested: {tested:,} | Simulated: {simulated:,} | Passed PF>={_min_pf()}: {passed}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, "raw_survivors.csv")
    if survivors:
        pd.DataFrame([{k: v for k, v in s.items() if k != 'entry_indices'} for s in survivors]).to_csv(raw_path, index=False)
        print(f"Raw survivors saved: {raw_path}")
    else:
        print("No survivors found.")
        pd.DataFrame().to_csv(raw_path, index=False)
    return survivors

def deduplicate(survivors):
    print("\n" + "=" * 60)
    print("DEDUPLICATION")
    print("=" * 60)
    if not survivors:
        print("No survivors to dedup."); return []
    survivors.sort(key=lambda x: -x['pf'])
    entry_sets = [set(s['entry_indices']) for s in survivors]
    keep = []; keep_sets = []
    for idx, s in enumerate(survivors):
        es = entry_sets[idx]
        is_dup = False
        for ks in keep_sets:
            if len(es) == 0 or len(es & ks) / len(es) > _overlap_threshold():
                is_dup = True; break
        if not is_dup:
            keep.append(s); keep_sets.append(es)
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx+1}/{len(survivors)}, kept {len(keep)}")
    print(f"Before dedup: {len(survivors)} | After dedup: {len(keep)}")
    dedup_path = os.path.join(OUTPUT_DIR, "deduped_survivors.csv")
    pd.DataFrame([{k: v for k, v in s.items() if k != 'entry_indices'} for s in keep]).to_csv(dedup_path, index=False)
    print(f"Saved: {dedup_path}")
    return keep

def final_report(survivors):
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    if not survivors:
        print(f"No signals found meeting PF >= {_min_pf()}."); return
    longs = sum(1 for s in survivors if s['direction'] == 'LONG')
    shorts = sum(1 for s in survivors if s['direction'] == 'SHORT')
    print(f"Total unique signals: {len(survivors)} (LONG: {longs}, SHORT: {shorts})")
    for rank, s in enumerate(survivors[:50], 1):
        print(f"{rank:>4} | {s['feat_1']:<22}{s['thresh_1']:>4} | {s['feat_2']:<22}{s['thresh_2']:>4} | "
              f"{s['feat_3']:<22}{s['thresh_3']:>4} | {s['direction']:<5} | "
              f"Tr {s['trades']:>4} | PF {s['pf']:>7.2f} | WR {s['wr']:>5.1f}")
    pfs = [s['pf'] for s in survivors]
    import catalogue as _cat9
    _mn, _n1, _x1 = _cat9.pf_aggregate(pfs, 'min')
    _mx, _n2, _x2 = _cat9.pf_aggregate(pfs, 'max')
    _md, _n3, _x3 = _cat9.pf_aggregate(pfs, 'median')
    print(f"\nPF range: {_mn} - {_mx} | Median: {_md} "
          f"| {_x3} undefined EXCLUDED of {len(pfs)} (a single NaN would otherwise return NaN "
          f"for the whole statistic; a 999 previously skewed it)")

# ═══════════════════════════════════════════════════════════════
#  CONVERGENCE DENSITY — fused F10 dimension of the F0 run
#  Co-firing count >= k over the CANDIDATE SIGNAL SET under evaluation
#  (NOT the raw 249 pool — that saturates). Direction-aligned per the F10
#  rule (long = hi + eq>0, short = lo + eq<0, ==0 neutral excluded), tallied
#  only over the conditions used by the selected set. Applied as a mask_window
#  density gate through the ratified engine — F0 and TM untouched, oracle-only.
# ═══════════════════════════════════════════════════════════════

DENSITY_K_BANDS = [1, 2, 3, 4, 5, 6, 8, 10]


def build_set_density(df, signals_df, adaptive, structural):
    long_masks, short_masks = [], []
    seen = set()
    for _, row in signals_df.iterrows():
        for i in (1, 2, 3):
            feat = row[f'feat_{i}']
            thr = str(row[f'thresh_{i}'])
            key = (feat, thr)
            if key in seen:
                continue
            seen.add(key)
            if thr == 'hi':
                long_masks.append(engine.condition_mask(df, feat, thr, adaptive, structural))
            elif thr == 'lo':
                short_masks.append(engine.condition_mask(df, feat, thr, adaptive, structural))
            elif thr.startswith('=='):
                v = int(thr[2:])
                m = engine.condition_mask(df, feat, thr, adaptive, structural)
                if v > 0:
                    long_masks.append(m)
                elif v < 0:
                    short_masks.append(m)
            else:
                raise ValueError(f"unknown thresh '{thr}'")
    count_long = np.sum(np.vstack(long_masks), axis=0).astype(int) if long_masks else np.zeros(len(df), int)
    count_short = np.sum(np.vstack(short_masks), axis=0).astype(int) if short_masks else np.zeros(len(df), int)
    return count_long, count_short


def _score_set(df, signals_df, mask_window, month, adaptive, structural, warmup):
    fold_trades = []
    for _, mkey in wf.FOLDS:
        td = engine.run_portfolio(df, signals_df, mask_window=(mask_window & (month == mkey)),
                                  adaptive=adaptive, structural=structural, warmup=warmup, verbose=False)
        if len(td):
            fold_trades.append(td)
    all_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame(columns=['pnl', 'exit_time'])
    n = len(all_trades)
    pnls = all_trades['pnl'].values if n else np.array([])
    agg_pf = round(wf.pf_from_pnls(pnls), 2)
    wr = round((pnls > 0).sum() / n * 100.0, 1) if n else 0.0
    daily = wf.daily_pnl_points(all_trades)
    daily_usd = wf.points_to_usd(daily['pnl'].values) if len(daily) else np.array([])
    worst = float(daily_usd.min()) if len(daily_usd) else 0.0
    hard = int((daily_usd <= -wf.DAILY_LOSS_CEILING_USD).sum())
    survive = worst > -wf.DAILY_LOSS_CEILING_USD
    pf_base, pf_stress = wf.spread_stress(all_trades)
    return {'trades': n, 'agg_pf': agg_pf, 'wr': wr, 'worst': round(worst, 1),
            'hard': hard, 'survive': survive, 'pf_base': pf_base, 'pf_stress': pf_stress}


def density_sweep(df, signals_df, k_bands, adaptive, structural, warmup):
    month = pd.Series(df['Time'].values).str[:7].values
    count_long, count_short = build_set_density(df, signals_df, adaptive, structural)
    print(f"\nCandidate set: {len(signals_df)} signals | set-conditions co-fire count "
          f"long {count_long.min()}..{count_long.max()} | short {count_short.min()}..{count_short.max()}")
    print("Density dimension: entry restricted to bars with >=k aligned set-conditions firing")
    dir_map = {'LONG': count_long, 'SHORT': count_short}
    for direction in ['LONG', 'SHORT']:
        sub = signals_df[signals_df['direction'] == direction].reset_index(drop=True)
        if len(sub) == 0:
            continue
        cnt = dir_map[direction]
        print(f"\n  {direction} subset ({len(sub)} signals, D2D=confirm):")
        for k in k_bands:
            mw = cnt >= k
            if mw.sum() < _min_trades():
                continue
            sc = _score_set(df, sub, mw, month, adaptive, structural, warmup)
            if sc['trades'] < _min_trades():
                continue
            surv = 'PASS' if sc['survive'] else 'REJECT'
            print(f"    count>={k:<2} [{surv:6}] bars {int(mw.sum()):>6} | trades {sc['trades']:>4} | "
                  f"aggPF {sc['agg_pf']:>5.2f} | WR {sc['wr']:>4.1f}% | worst-day ${sc['worst']:>9,.0f} | "
                  f"hard-stop {sc['hard']:>2} | spread {sc['pf_base']:.2f}->{sc['pf_stress']:.2f}")


def run_density(signals_path):
    df = load_sealed_baseline()
    warmup = warmup_floor(df)
    adaptive = dt.compute_adaptive_thresholds(df)
    structural = dt.compute_structural_gates(df)
    signals_df = pd.read_csv(signals_path)
    print(f"equiDOT — F0 convergence-density dimension (fused F10) on: {signals_path}")
    density_sweep(df, signals_df, DENSITY_K_BANDS, adaptive, structural, warmup)
    print("\nDone.")


def main():
    global EMIT_ALL
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument('--emit-all', action='store_true',
                    help='Emit EVERY signal regardless of trade count. DEFAULTS OFF: the '
                         'default path must reproduce the 19,754-row output byte-for-byte.')
    _a, _ = _p.parse_known_args()
    if _a.emit_all:
        EMIT_ALL = True
        print('  *** --emit-all: MIN_TRADES gating DISABLED at all five sites, MIN_PF lifted to 0.0, and dedup OVERLAP_THRESHOLD lifted above any achievable ratio. Every signal '
              'is emitted regardless of trade count, and the `trades` column is populated on '
              'every row so a downstream screen applies its OWN threshold rather than '
              'inheriting the scanner\'s. THIS OUTPUT IS NOT THE 19,754-ROW BASELINE. ***',
              flush=True)
    print("equiDOT — Triple-Convergence Discovery (117 candidates, mechanism-D oracle)")
    print(f"Config: risk={RISK_MULT}, step={STEP_PCT}, min_trades={MIN_TRADES}, "
          f"min_PF={_min_pf()}, dedup={_overlap_threshold()}")
    df = load_sealed_baseline()
    warmup = warmup_floor(df)
    feat_candidates, equality_candidates = build_candidates(df)
    adaptive = dt.compute_adaptive_thresholds(df)
    structural = dt.compute_structural_gates(df)
    eligible = (df['ADX_Value'].values >= 15) & (df['Volume'].values > 50)
    vol_zero = df['Volume'].values == 0
    warm = np.arange(len(df)) < warmup
    fri_block = (df['EST_DayOfWeek'].values == 5) & ((df['EST_Hour'].values > 16) | ((df['EST_Hour'].values == 16) & (df['EST_Minute'].values >= 45)))
    entry_allowed = eligible & ~vol_zero & ~fri_block & ~warm
    d2d_dir = df['D2D_Trend_Dir'].values
    print(f"Eligible bars: {int(eligible.sum())} | Entry-allowed bars: {int(entry_allowed.sum())} | "
          f"Friday blocks: {int(fri_block.sum())}")
    feature_conditions = build_conditions(df, feat_candidates, equality_candidates, adaptive, structural, eligible)
    all_features = feat_candidates + equality_candidates
    arrays = (df['High'].values, df['Low'].values, df['Close'].values, df['ATR_1M'].values,
              df['EST_DayOfWeek'].values, df['EST_Hour'].values, df['EST_Minute'].values)
    parity_check(df, feature_conditions, entry_allowed, d2d_dir, arrays, n_signals=5)
    survivors = run_search(df, feature_conditions, all_features, entry_allowed, d2d_dir, arrays)
    deduped = deduplicate(survivors)
    final_report(deduped)
    print(f"\nResults in: {OUTPUT_DIR}/")
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'density':
        run_density(sys.argv[2] if len(sys.argv) > 2 else 'recommended_set_76.csv')
    else:
        main()
