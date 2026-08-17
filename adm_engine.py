import pandas as pd
import numpy as np

PF_UNDEFINED = float('nan')
import os
import sys
import time
import dots_thresholds as dt

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

PARTS = [f"equiDOT_recon171_step7_part{i}.csv" for i in range(1, 9)]
DEFAULT_SIGNALS_PATH = "recommended_set_76.csv"
OUTPUT_DIR = "dots_portfolio_results"
DOTS_INITBARS = 6900
SPREAD = 3.0
RISK_MULT = 2.0
MAX_RISK = 150.0
STEP_PCT = 0.30
BE_TRIG_FRAC = 1.0
LOCK_FRAC = 1.0
MOMENTUM_THRESHOLD = 0.00012
MOMENTUM_SL_MULT = 4.0
LAG_BASE = 2
LAG_MOMENTUM = 3
HURST_SIZE_PCT = 0.90
HURST_GAP_PCT = 0.97
FB_PCT = 0.90
CONV_HURST_MULT = 2.0
CONV_RECENTFB_MULT = 1.25
RECENTFB_WINDOW = 5
GAP_LOCK = 3.0
D2D_HURST_PCT = 0.30
D2D_ADX_MIN = 30.0
D2D_GAP_LOTS = 2.0
MAX_POSITIONS = 6
ELIG_ADX = 15.0
ELIG_VOL = 50.0
FRIDAY_CUT = True
SOLO_TICK_GATE = True
ADMISSION_RULE = 'CURRENT'
ADM_FLOOR = {1: 3, -1: 3}
ADM_TIERGATES = None
ADM_SUBFLOOR = ()
ADM_GATES = None
ADM_LOG = None
ADM_ORDER = (1, -1)

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

def _adm_pass(bar, d, depth):
    g = ADM_GATES
    if g is not None and g['ATR'][bar] < g['atr_min']:
        return False
    if ADM_TIERGATES is None:
        return True
    masks = ADM_TIERGATES.get((d, min(depth, 5)))
    if not masks:
        return True
    for m in masks:
        if not m[bar]:
            return False
    return True


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
#  CONDITION / SIGNAL MASKS (oracle-only thresholds + equality)
# ═══════════════════════════════════════════════════════════════

def condition_mask(df, feat, thresh, adaptive, structural):
    if thresh == 'hi':
        m = df[feat].values > adaptive[(feat, 'hi')]
        if feat == 'OR_Position':
            m = m & structural['OR_Position']
        return m
    if thresh == 'lo':
        m = df[feat].values < adaptive[(feat, 'lo')]
        if feat == 'OR_Position':
            m = m & structural['OR_Position']
        return m
    if thresh.startswith('=='):
        return df[feat].values == int(thresh[2:])
    raise ValueError(f"unknown thresh '{thresh}' for feature '{feat}'")

def build_signal_masks(df, signals_df, adaptive, structural, entry_ok, verbose=True):
    d2d_dir = df['D2D_Trend_Dir'].values
    signal_masks = []
    signal_dirs = []
    signal_names = []
    for idx, sig in signals_df.iterrows():
        direction = 1 if sig.direction == 'LONG' else -1
        feats = [(sig.feat_1, sig.thresh_1), (sig.feat_2, sig.thresh_2), (sig.feat_3, sig.thresh_3)]
        mask = entry_ok.copy()
        for feat, thresh in feats:
            mask = mask & condition_mask(df, feat, str(thresh), adaptive, structural)
        mask = mask & (d2d_dir == direction)
        signal_masks.append(mask)
        signal_dirs.append(direction)
        name = f"{'L' if direction == 1 else 'S'}_{idx+1}_{str(sig.feat_1)[:8]}+{str(sig.feat_2)[:8]}+{str(sig.feat_3)[:8]}"
        signal_names.append(name)
        if verbose:
            print(f"  Signal {idx+1:>2}: {int(mask.sum()):>4} qualifying bars | {'LONG' if direction==1 else 'SHORT'} | "
                  f"{sig.feat_1}({sig.thresh_1}) + {sig.feat_2}({sig.thresh_2}) + {sig.feat_3}({sig.thresh_3})")
    return signal_masks, signal_dirs, signal_names

# ═══════════════════════════════════════════════════════════════
#  TRADE MANAGEMENT (S.7 + S.12 runner + S.15 jar + S.16 per-bar + S.17 + S.19 SL)
# ═══════════════════════════════════════════════════════════════

_UID = [0]


class Trade:
    def __init__(self, signal_idx, entry_bar, entry_price, direction, initial_risk, lag, base_risk, lock_frac, lots):
        self.signal_idx = signal_idx
        self.entry_bar = entry_bar
        self.entry_price = entry_price
        self.direction = direction
        self.initial_risk = initial_risk
        self.step_size = STEP_PCT * base_risk
        self.be_trigger = BE_TRIG_FRAC * self.step_size
        self.be_lock_dist = lock_frac * self.be_trigger
        if direction == 1:
            self.current_sl = entry_price - initial_risk
        else:
            self.current_sl = entry_price + initial_risk
        self.tiers = 0
        self.be_nudged = False
        self.plan_depth = 0
        self.exec_depth = 0
        _UID[0] += 1
        self.uid = _UID[0]
        self.lag = lag
        self.lots = lots

def run_portfolio(df, signals_df, mask_window=None, adaptive=None, structural=None, conviction=None,
                  warmup=None, verbose=True):
    if adaptive is None:
        adaptive = dt.compute_adaptive_thresholds(df)
    if structural is None:
        structural = dt.compute_structural_gates(df)
    if warmup is None:
        warmup = warmup_floor(df, verbose=verbose)
    n_bars = len(df)
    warm = np.arange(n_bars) < warmup
    eligible = (df['ADX_Value'].values >= ELIG_ADX) & (df['Volume'].values > ELIG_VOL)
    vol_zero = df['Volume'].values == 0
    fri_block = (df['EST_DayOfWeek'].values == 5) & ((df['EST_Hour'].values > 16) | ((df['EST_Hour'].values == 16) & (df['EST_Minute'].values >= 45)))
    if not FRIDAY_CUT:
        fri_block = np.zeros(len(df), dtype=bool)
    entry_ok = eligible & ~vol_zero & ~fri_block & ~warm
    if mask_window is not None:
        entry_ok = entry_ok & mask_window
    if verbose:
        print(f"Eligible bars: {int(eligible.sum())}, Entry-allowed bars: {int(entry_ok.sum())}")

    signal_masks, signal_dirs, signal_names = build_signal_masks(
        df, signals_df, adaptive, structural, entry_ok, verbose=verbose)
    n_signals = len(signal_masks)
    signal_masks_arr = np.array(signal_masks) if n_signals else np.zeros((0, n_bars), dtype=bool)

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    atrs = df['ATR_1M'].values
    dow = df['EST_DayOfWeek'].values
    hr = df['EST_Hour'].values
    mn = df['EST_Minute'].values
    times = df['Time'].values
    volume = df['Volume'].values
    long_mult = conviction.get('long_mult') if conviction is not None else None
    short_mult = conviction.get('short_mult') if conviction is not None else None
    gap_hurst_mask = conviction.get('gap_hurst') if conviction is not None else None
    gap_fb_mask = conviction.get('gap_fb') if conviction is not None else None
    gap_d2d_dir = conviction.get('gap_d2d_dir') if conviction is not None else None
    GAP_H_IDX = n_signals
    GAP_F_IDX = n_signals + 1
    GAP_D2D_IDX = n_signals + 2
    signal_names = signal_names + ['GAP_HURST', 'GAP_FB', 'GAP_D2D']

    _UID[0] = 0
    TRACE = []
    all_trades = []
    micro_logret = df['Micro_LogReturn'].values
    active_trades = []
    signal_in_trade = [False] * n_signals
    gate_blocks = 0
    cap_blocks = 0
    t_start = time.time()

    for bar in range(n_bars):
        closed_this_bar = []
        for trade in active_trades:
            TRACE.append((bar, trade.uid, trade.current_sl, trade.be_nudged, trade.lots, trade.direction, trade.entry_price))
            h = highs[bar]; l = lows[bar]; c = closes[bar]
            d = trade.direction; ep = trade.entry_price
            exited = False
            exit_price = 0.0; exit_type = ''

            if d == 1 and l <= trade.current_sl:
                exit_price = trade.current_sl
                exit_type = ('BE' if trade.tiers < 3 else 'LF') if trade.be_nudged else 'SL'
                exited = True
            elif d == -1 and h >= trade.current_sl:
                exit_price = trade.current_sl
                exit_type = ('BE' if trade.tiers < 3 else 'LF') if trade.be_nudged else 'SL'
                exited = True

            if not exited and dow[bar] == 5 and (hr[bar] > 16 or (hr[bar] == 16 and mn[bar] >= 45)):
                exit_price = c; exit_type = 'FC'; exited = True

            if exited:
                pnl = (exit_price - ep - SPREAD) if d == 1 else (ep - exit_price - SPREAD)
                all_trades.append({
                    'signal_idx': trade.signal_idx,
                    'signal_name': signal_names[trade.signal_idx],
                    'direction': 'LONG' if d == 1 else 'SHORT',
                    'entry_bar': trade.entry_bar,
                    'exit_bar': bar,
                    'entry_time': times[trade.entry_bar],
                    'exit_time': times[bar],
                    'entry_price': ep,
                    'exit_price': exit_price,
                    'pnl': round(pnl * trade.lots, 1),
                    'pnl_per_lot': round(pnl, 1),
                    'lots': trade.lots,
                    'exit_type': exit_type,
                    'tiers': trade.tiers,
                    'be_nudged': trade.be_nudged,
                    'initial_risk': trade.initial_risk,
                    'tid': trade.uid,
                    'plan_depth': trade.plan_depth,
                    'exec_depth': trade.exec_depth,
                })
                if trade.signal_idx < n_signals:
                    signal_in_trade[trade.signal_idx] = False
                closed_this_bar.append(trade)
                continue

            if d == 1:
                target = ep + (trade.tiers + 1) * trade.step_size
                while h >= target:
                    trade.tiers += 1
                    target = ep + (trade.tiers + 1) * trade.step_size
            else:
                target = ep - (trade.tiers + 1) * trade.step_size
                while l <= target:
                    trade.tiers += 1
                    target = ep - (trade.tiers + 1) * trade.step_size

            if not trade.be_nudged:
                unrealised = (h - ep) if d == 1 else (ep - l)
                if unrealised >= trade.be_trigger:
                    if d == 1:
                        new_sl = ep + trade.be_lock_dist
                        if new_sl > trade.current_sl: trade.current_sl = new_sl
                    else:
                        new_sl = ep - trade.be_lock_dist
                        if new_sl < trade.current_sl: trade.current_sl = new_sl
                    trade.be_nudged = True

            if trade.tiers >= trade.lag + 1:
                if d == 1:
                    trail = ep + (trade.tiers - trade.lag) * trade.step_size
                    if trail > trade.current_sl: trade.current_sl = trail
                else:
                    trail = ep - (trade.tiers - trade.lag) * trade.step_size
                    if trail < trade.current_sl: trade.current_sl = trail

        for t in closed_this_bar:
            active_trades.remove(t)

        if n_signals == 0:
            continue
        qual_indices = np.where(signal_masks_arr[:, bar])[0]
        n_qual = len(qual_indices)
        if ADMISSION_RULE != 'CURRENT':
            live_lots = sum(1 for t in active_trades if not t.be_nudged)
            free = MAX_POSITIONS - live_lots
            _order = ADM_ORDER
            if _order == 'depth':
                _na = {}
                for _dd in (1, -1):
                    _na[_dd] = sum(1 for i in qual_indices if signal_dirs[i] == _dd and not signal_in_trade[i])
                _order = (1, -1) if _na[1] >= _na[-1] else (-1, 1)
            for _d in _order:
                idxs = [int(i) for i in qual_indices if signal_dirs[i] == _d]
                Q = len(idxs)
                if Q == 0:
                    continue
                A = [i for i in idxs if not signal_in_trade[i]]
                nA = len(A)
                if nA == 0:
                    continue
                if ADMISSION_RULE in ('ATOMIC', 'ATOMIC_TIER'):
                    if nA > free:
                        if ADM_LOG is not None: ADM_LOG['refused'].append((bar, _d, Q, nA, free))
                        continue
                    n_plan = nA
                else:
                    if free < (1 if ADM_SUBFLOOR else ADM_FLOOR[_d]):
                        if ADM_LOG is not None: ADM_LOG['refused'].append((bar, _d, Q, nA, free))
                        continue
                    n_plan = min(nA, free)
                    if n_plan < nA and ADM_LOG is not None: ADM_LOG['clipped'].append((bar, _d, Q, nA, n_plan))
                gate_depth = Q if ADMISSION_RULE == 'ATOMIC_TIER' else n_plan
                if gate_depth < ADM_FLOOR[_d]:
                    if not (ADM_SUBFLOOR and nA <= 2 and n_plan == nA and nA in ADM_SUBFLOOR):
                        continue
                    if not (ADM_TIERGATES or {}).get((_d, gate_depth)):
                        continue
                if not _adm_pass(bar, _d, gate_depth):
                    continue
                for sig_idx in A[:n_plan]:
                    entry_price = closes[bar]
                    v = micro_logret[bar] * signal_dirs[sig_idx]
                    momentum = v >= MOMENTUM_THRESHOLD
                    raw_risk = atrs[bar] * (MOMENTUM_SL_MULT if momentum else RISK_MULT)
                    initial_risk = min(raw_risk, MAX_RISK)
                    if initial_risk <= 0:
                        continue
                    lag = LAG_MOMENTUM if momentum else LAG_BASE
                    base_risk = min(atrs[bar] * RISK_MULT, MAX_RISK)
                    lots = (long_mult[bar] if long_mult is not None else 1.0) if signal_dirs[sig_idx] == 1 else (short_mult[bar] if short_mult is not None else 1.0)
                    trade = Trade(sig_idx, bar, entry_price, signal_dirs[sig_idx], initial_risk, lag, base_risk, LOCK_FRAC, lots)
                    trade.plan_depth = gate_depth
                    trade.exec_depth = n_plan
                    active_trades.append(trade)
                    signal_in_trade[sig_idx] = True
                    live_lots += 1
                    free -= 1
        elif n_qual >= 1 and not (SOLO_TICK_GATE and n_qual == 1 and volume[bar] < 300):
            live_lots = sum(1 for t in active_trades if not t.be_nudged)
            for sig_idx in qual_indices:
                if signal_in_trade[sig_idx]:
                    continue
                if live_lots >= MAX_POSITIONS:
                    cap_blocks += 1
                    continue
                entry_price = closes[bar]
                v = micro_logret[bar] * signal_dirs[sig_idx]
                momentum = v >= MOMENTUM_THRESHOLD
                raw_risk = atrs[bar] * (MOMENTUM_SL_MULT if momentum else RISK_MULT)
                initial_risk = min(raw_risk, MAX_RISK)
                if initial_risk <= 0:
                    continue
                lag = LAG_MOMENTUM if momentum else LAG_BASE
                base_risk = min(atrs[bar] * RISK_MULT, MAX_RISK)
                if signal_dirs[sig_idx] == 1:
                    lots = long_mult[bar] if long_mult is not None else 1.0
                else:
                    lots = short_mult[bar] if short_mult is not None else 1.0
                trade = Trade(sig_idx, bar, entry_price, signal_dirs[sig_idx], initial_risk, lag, base_risk, LOCK_FRAC, lots)
                active_trades.append(trade)
                signal_in_trade[sig_idx] = True
                live_lots += 1
        elif SOLO_TICK_GATE and n_qual == 1 and volume[bar] < 300:
            gate_blocks += 1

        if conviction is not None and len(active_trades) == 0 and bar >= warmup:
            gfire = None
            gdir = 1
            glots = 1.0
            if gap_d2d_dir is not None and gap_d2d_dir[bar] != 0:
                gfire = GAP_D2D_IDX
                gdir = int(gap_d2d_dir[bar])
                glots = D2D_GAP_LOTS
            elif gap_hurst_mask is not None and gap_hurst_mask[bar]:
                gfire = GAP_H_IDX
            elif gap_fb_mask is not None and gap_fb_mask[bar]:
                gfire = GAP_F_IDX
            if gfire is not None and glots <= MAX_POSITIONS:
                entry_price = closes[bar]
                v = micro_logret[bar] * gdir
                momentum = v >= MOMENTUM_THRESHOLD
                raw_risk = atrs[bar] * (MOMENTUM_SL_MULT if momentum else RISK_MULT)
                initial_risk = min(raw_risk, MAX_RISK)
                if initial_risk > 0:
                    lag = LAG_MOMENTUM if momentum else LAG_BASE
                    base_risk = min(atrs[bar] * RISK_MULT, MAX_RISK)
                    _lock = LOCK_FRAC if gfire == GAP_D2D_IDX else GAP_LOCK
                    trade = Trade(gfire, bar, entry_price, gdir, initial_risk, lag, base_risk, _lock, glots)
                    active_trades.append(trade)

        if verbose and (bar + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            print(f"  Bar {bar+1}/{n_bars} ({100*(bar+1)/n_bars:.1f}%) | "
                  f"{len(all_trades)} trades closed | {len(active_trades)} open | {elapsed:.0f}s")

    for trade in active_trades:
        ep = trade.entry_price
        exit_price = closes[-1]
        d = trade.direction
        pnl = (exit_price - ep - SPREAD) if d == 1 else (ep - exit_price - SPREAD)
        all_trades.append({
            'signal_idx': trade.signal_idx,
            'signal_name': signal_names[trade.signal_idx],
            'direction': 'LONG' if d == 1 else 'SHORT',
            'entry_bar': trade.entry_bar,
            'exit_bar': n_bars - 1,
            'entry_time': times[trade.entry_bar],
            'exit_time': times[-1],
            'entry_price': ep,
            'exit_price': exit_price,
            'pnl': round(pnl * trade.lots, 1),
            'pnl_per_lot': round(pnl, 1),
            'lots': trade.lots,
            'exit_type': 'EOD',
            'tiers': trade.tiers,
            'be_nudged': trade.be_nudged,
            'initial_risk': trade.initial_risk,
            'tid': trade.uid,
            'plan_depth': trade.plan_depth,
            'exec_depth': trade.exec_depth,
        })

    trades_df = pd.DataFrame(all_trades)
    trades_df.attrs['gate_blocks'] = gate_blocks
    trades_df.attrs['cap_blocks'] = cap_blocks
    trades_df.attrs['n_signals'] = n_signals
    trades_df.attrs['TRACE'] = TRACE
    return trades_df

# ═══════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════

def report(trades_df, signals_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUTPUT_DIR, "all_trades.csv"), index=False)
    n_trades = len(trades_df)
    if n_trades == 0:
        print("No trades produced.")
        return
    pnls = trades_df['pnl'].values
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls <= 0].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else PF_UNDEFINED
    wr = (pnls > 0).sum() / n_trades * 100
    total_pnl = pnls.sum()

    sl_trades = trades_df[trades_df.exit_type == 'SL']
    be_trades = trades_df[trades_df.exit_type == 'BE']
    lf_trades = trades_df[trades_df.exit_type == 'LF']
    fc_trades = trades_df[trades_df.exit_type == 'FC']
    sl_wins = (sl_trades.pnl > 0).sum()
    be_losses = (be_trades.pnl <= 0).sum()
    lf_losses = (lf_trades.pnl <= 0).sum()

    td = trades_df.copy()
    td['exit_date'] = td['exit_time'].str[:10]
    daily = td.groupby('exit_date').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
    ).reset_index()
    daily['cum_pnl'] = daily['pnl'].cumsum()
    daily['losing_day'] = daily['pnl'] < 0
    daily.to_csv(os.path.join(OUTPUT_DIR, "daily_pnl.csv"), index=False)

    print()
    print("=" * 60)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Signals:           {len(signals_df)} ({(signals_df.direction=='LONG').sum()}L / {(signals_df.direction=='SHORT').sum()}S)")
    print(f"  Total trades:      {n_trades}")
    print(f"  Trading days:      {daily.shape[0]}")
    print(f"  Win Rate:          {wr:.1f}%")
    print(f"  Profit Factor:     {pf:.2f}")
    print(f"  Total P&L:         {total_pnl:+.1f} pts")
    print(f"  Avg P&L/trade:     {total_pnl/n_trades:+.1f} pts")
    print(f"  SL/BE/LF/FC:       {len(sl_trades)}/{len(be_trades)}/{len(lf_trades)}/{len(fc_trades)}")
    print(f"  SL wins (BUG):     {sl_wins}")
    print(f"  BE/LF losses:      {be_losses}/{lf_losses} (spread-floor)")
    print(f"  Gate/Cap blocks:   {trades_df.attrs.get('gate_blocks',0)}/{trades_df.attrs.get('cap_blocks',0)}")
    print(f"  Best/Worst day:    {daily.pnl.max():+.1f} / {daily.pnl.min():+.1f} pts")
    print(f"  Winning/Losing days: {(~daily.losing_day).sum()}/{daily.losing_day.sum()}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")

def main(signals_path=DEFAULT_SIGNALS_PATH):
    print("equiDOT — Portfolio Simulation Engine")
    df = load_sealed_baseline()
    adaptive = dt.compute_adaptive_thresholds(df)
    structural = dt.compute_structural_gates(df)
    warmup = warmup_floor(df)
    signals_df = pd.read_csv(signals_path)
    print(f"\nLoading {len(signals_df)} signals from {signals_path} ...")
    trades_df = run_portfolio(df, signals_df, mask_window=None,
                              adaptive=adaptive, structural=structural, warmup=warmup, verbose=True)
    report(trades_df, signals_df)
    print("Done.")

if __name__ == '__main__':
    sp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIGNALS_PATH
    main(sp)
