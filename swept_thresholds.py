"""swept_thresholds.py — mechanism-D percentiles at percentiles other than p80/p20.

WHY THIS EXISTS. dots_thresholds.compute_adaptive_thresholds only ever emits
two percentiles per variable: 'hi' = p80 and 'lo' = p20, fixed in _D_SPEC at
import time. The Whole DOT gate stack needs p90 thresholds (Micro_Hurst,
AT_Slope_ST) and a p20 threshold used as a > comparison (Micro_FailedBreak).
There is no public accessor for an arbitrary percentile, and _floor_pct is
internal.

DO NOT REIMPLEMENT THE ROLLING WALK. This module does not reimplement it. It
temporarily substitutes _D_SPEC and calls the sacred function, so the ring, the
eligibility mask, the day-refresh and the floor-index are BIT-IDENTICAL to
production by construction rather than by inspection. _D_SPEC is always
restored, including on exception.

SEMANTICS, INHERITED FROM dots_thresholds.compute_adaptive_thresholds:
  ring            deque(maxlen=_ROLL_CAP), _ROLL_CAP = 2500
  ring admission  a bar enters the ring iff ADX_Value >= 15.0 AND Volume > 50.0.
                  THE RING IS APPENDED BEFORE THE DAY CHECK, so a refresh on
                  bar i sees bar i's own value if bar i is eligible.
  refresh         DAY-REFRESHED, not per-bar. The key is
                  int(str(Time)[8:10]) — the DAY-OF-MONTH FIELD ONLY. The
                  threshold changes only on a bar whose day-of-month differs
                  from the previous bar's, and is held flat across every bar in
                  between.
  percentile      FLOOR-INDEX, NOT INTERPOLATED:
                      idx = int(floor(count * pct)), clamped to [0, count-1]
                      value = sorted(ring)[idx]
                  Returns 0.0 when count < 2.
  warm-up         No special handling. Before the ring fills, the percentile is
                  taken over however many eligible bars have accumulated. The
                  caller is responsible for the warmup floor (bar >= 6900).
  comparison      STRICT. The gate masks below use  >  for a high-side
                  threshold and  <  for a low-side one, matching
                  dots_thresholds/score_g condition_mask exactly.

USAGE
    import swept_thresholds as sw
    G = sw.build_whole_dot_gates(df)
    # G['HU90'], G['FB20'], G['ATS90'] -> bool arrays of len(df)
"""

import numpy as np
import dots_thresholds as dt


def swept(df, specs):
    """Return mechanism-D thresholds for arbitrary (name -> (column, pct)) specs.

    specs: dict mapping an arbitrary key to (column_name, percentile_float).
           e.g. {('Micro_Hurst', 'p90'): ('Micro_Hurst', 0.90)}
    Returns: dict with the same keys, each a float array of len(df).
    """
    saved = dict(dt._D_SPEC)
    try:
        dt._D_SPEC.clear()
        dt._D_SPEC.update(specs)
        out = dt.compute_adaptive_thresholds(df)
    finally:
        dt._D_SPEC.clear()
        dt._D_SPEC.update(saved)
    return {k: out[k] for k in specs}


WHOLE_DOT_SPECS = {
    ('Micro_Hurst', 'p90'): ('Micro_Hurst', 0.90),
    ('Micro_FailedBreak', 'p20'): ('Micro_FailedBreak', 0.20),
    ('AT_Slope_ST', 'p90'): ('AT_Slope_ST', 0.90),
}


def build_whole_dot_gates(df):
    """The three gate masks the Whole DOT spec section 5.1 requires."""
    t = swept(df, WHOLE_DOT_SPECS)
    return {
        'HU90': df['Micro_Hurst'].values > t[('Micro_Hurst', 'p90')],
        'FB20': df['Micro_FailedBreak'].values > t[('Micro_FailedBreak', 'p20')],
        'ATS90': df['AT_Slope_ST'].values > t[('AT_Slope_ST', 'p90')],
    }


def pass_rates(gates):
    """Checksum helper: fraction of ALL bars each mask admits."""
    return {k: 100.0 * float(np.asarray(v, dtype=bool).mean()) for k, v in gates.items()}
