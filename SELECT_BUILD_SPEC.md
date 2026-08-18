# SELECT — BUILD SPECIFICATION

**This is the Developer's only input. Every component below is stated as a criterion,
a threshold, the data it reads, the symbol that computes it, and what it does on
failure. Nothing here requires interpretation.**

Target file: `dot_master_discovery/select_stage.py` (already at package root, correct —
`_SEL_ROOT` resolves `engine`, `scanners`, `orchestrator` in spawned workers).

---

## 0. WHAT EXISTS AND WHAT IS MISSING

| component | state |
|---|---|
| train-window screen, 4 criteria | BUILT |
| nested seeded draw, 6 sizes | BUILT |
| scoring via `_score_configured` | BUILT |
| four outputs, side-by-side, overlap | BUILT |
| random bands per size | SPEC BELOW — §3 |
| gate derivation over 117 variables | SPEC BELOW — §5 |
| direction balance | **NEGATIVE — DO NOT BUILD (§6.1)** |
| terrain coverage as constraint | **NEGATIVE — DO NOT BUILD (§6.2)** |
| participation constraint | **NEGATIVE — DO NOT BUILD (§6.3)** |
| short-quality screen criterion | **SPEC BELOW — §4.5 (POSITIVE)** |
| regime as selection input | NOT TESTED — §7 |

---

## 1. PORTABILITY — BINDING ON EVERY COMPONENT

**NOTHING MAY BE HARDCODED TO US30 OR TO THIS FRAME.** Every threshold, window rule,
band expectation and size comes from `whole_dot_config.json`. The stage must run on a
new instrument with a new config and no code change.

Specifically forbidden as literals in `select_stage.py`: `12`, `2.0`, `2/3`, `3`, `21`,
`20`, `297`, `0.90`, `0.20`, any date, any month count, any band figure.

**REQUIRED CONFIG KEYS (abort with the missing key name if any is absent):**

```
screen.min_trades              int     (this frame: 12)
screen.min_train_pf            float   (this frame: 2.0)
screen.min_buckets_present     int     (this frame: 3)
screen.min_bucket_profit_frac  float   (this frame: 0.6667)
screen.holdout_months_N        int     ** SEE §2 — THIS DEFECT COST AN EXCHANGE **
draw.seed                      int
draw.book_size                 int
draw.arm_sizes                 list[int]
arch.floor_long                int
arch.floor_short               int
arch.max_positions             int
arch.atr_min                   float
arch.recentfb_sizing           bool
bands.<N>.<metric>             see §3 — bands are DATA, not code
```

---

## 2. `N` — THE DEFECT THAT MUST NOT RECUR

The screen window is "all but the final N months". **`N` WAS NEVER PINNED.** Quant read
N=2, the Developer implemented N=1. Both legitimate. The consequence, measured:

| | N=2 (train Jan–May) | N=1 (train Jan–Jun) |
|---|---|---|
| survivors | 6,267 | **8,016** |
| survivors from the old 6,034 field | 4,575 | 5,885 |
| survivors from the 13,720 excluded | ~1,692 | 2,131 |

**Agreement on the same 6,034 field: 4,505 signals. Developer-only: 1,380 — signals at
`train_PF` 1.87–1.99 on five months that clear 2.0 on six, and `bucket_profit_frac`
3/5 = 0.600 that reach 4/6 = 0.667. Quant-only: 70.**

**ONE EXTRA TRAINING MONTH MOVES THE SURVIVOR COUNT 28%.**

**REQUIREMENTS:**
1. `screen.holdout_months_N` is a required config key. Abort if absent.
2. The stage prints `N`, the resolved train window (first and last bar timestamp), and
   the bucket count, in the banner, on every run.
3. **EVERY BAND, EXPECTATION OR CALIBRATION FIGURE THE STAGE PRINTS CARRIES THE `N` AND
   THE WINDOW IT WAS CONDITIONED ON, IN THE SAME LINE.** A band without its window is a
   defect.

---

## 3. THE RANDOM BANDS — DATA, NOT CODE

Bands are **calibration constants that live in config**, keyed by `N`, pool size and
arm size. The stage reads them and prints them beside each arm. It does not compute them
at runtime and does not hardcode them.

**MEASURED: pool = 8,016 survivors · train = 2026.01–06 · N=1 · full-frame scoring ·
adopted architecture (L3/S3, cap 21, `recentfb_sizing=False`) · 1.0 lot.**

| signals | seeds | loss events | PF | worst day | losing wks | days | margin | net |
|---|---|---|---|---|---|---|---|---|
| 150 | 3 | 13 – 22 | 7.95 – 12.17 | −880 to −2,087 | 1 – 3 | 76 – 85 | 31.2 – 33.7 | 68.4k – 85.8k |
| 200 | 3 | 22 – 41 | 5.90 – 9.69 | −880 to −1,115 | 2 – 4 | 89 – 107 | 26.9 – 29.0 | 104k – 121k |
| 297 | 5 | 51 – 61 | 5.87 – 7.75 | −1,110 to −2,703 | 1 – 2 | 109 – 118 | 24.4 – 30.2 | 164k – 184k |
| 500 | 3 | 144 – 168 | 4.16 – 4.34 | −2,133 to −2,795 | 1 – 2 | 120 – 122 | 20.3 – 20.8 | 314k – 339k |
| **incumbent 297** | — | **42** | **14.53** | **−346.60** | **0** | **119** | **33.07** | **284,974** |

Seeds 1000–1004 (n=297), 1000–1002 (others). **Trial count 14 draws.**

**SUPERSEDED — DO NOT USE:** the 4,575-pool / N=2 bands (150: 21–27, 200: 35–49,
297: 84–95, 500: 184–185). They were conditioned on a different window and a smaller pool.

**WITHDRAWN READING:** "four of five arms sit below their band" was an artefact of
comparing an N=1 draw against N=2 bands. Re-measured: 150 (16) inside 13–22, 200 (28)
inside 22–41, 500 (151) inside 144–168, and **297 (68) sits ABOVE 51–61.** Nothing is
below. **The screen is not doing work beyond size control.**

**WHAT THE STAGE PRINTS PER ARM:** the arm's result, its band, and `INSIDE` / `ABOVE` /
`BELOW`. **An arm inside its band has reproduced the random baseline, not the book.**

---

## 4. THE SCREEN

Runs on the **raw F0 scan**, not the 6,488/6,034 pre-filter. The pre-filter uses
full-sample `agg_pf`, `folds_plus` and `trades` computed over the months the screen is
meant to validate against.

**§4.1–4.4 are BUILT. Retain unchanged:**

```
train_trades          >= screen.min_trades
train_PF              >= screen.min_train_pf
buckets_present       >= screen.min_buckets_present
bucket_profit_frac    >= screen.min_bucket_profit_frac      (PROPORTION, never a count)
```

`bucket_profit_frac` is `profitable_buckets / buckets_present`. **NEVER a count** —
`folds_plus >= 4` as a count is unsatisfiable for signals lacking four buckets.

**TRAIN-WINDOW STATISTICS MUST BE COMPUTED, NOT DERIVED.** `results_F0_*.csv` carries
only full-sample `trades`, `agg_pf`, `folds_plus`, `min_fold_pf`, `spread_pf`. No
per-fold net, no per-month partition, no trade list. Measured cost 0.157 s/signal,
~4 min at 14 workers.

### 4.5 NEW CRITERION — SHORT-SIDE QUALITY IS REAL AND IT IS NOT SHORT-SPECIFIC

**MEASURED.** Incumbent members vs the 8,016 pool, same direction, Mann-Whitney
two-sided, full-sample scan statistics (Jan–Jun folds):

| statistic | SHORT: incumbent (98) vs pool (2,386) | p | LONG control | p |
|---|---|---|---|---|
| `trades` | 31.0 vs 19.0 | **0.0000** | 32.0 vs 22.0 | **0.0000** |
| `agg_pf` | 4.49 vs 3.48 | **0.0000** | 5.69 vs 3.65 | **0.0000** |
| `folds_plus` | 5.0 vs 4.0 | **0.0000** | 5.0 vs 5.0 | **0.0000** |
| `worst_day_usd` | −108.75 vs −78.90 | **0.0007** | — | — |
| `WR` | 92.3 vs 91.7 | 0.105 | — | — |
| `min_fold_pf` | 0.815 vs 0.625 | 0.315 | — | — |

**THIS IS THE FIRST PER-MEMBER SIGNAL FOUND IN THIS PROJECT.** The incumbent's members
are drawn from the upper tail of `trades`, `agg_pf` and `folds_plus` on both sides.

**AND IT IS NOT A SHORT-SIDE EFFECT — THE LONG CONTROL SEPARATES JUST AS STRONGLY.**
It is a whole-book property: **the operator selected high-trade-count, high-PF,
high-fold-consistency members, and the screen's floors are far below where he actually
selected.** Pool medians sit at `trades` 19–22 and `agg_pf` 3.5–3.7; incumbent medians
at 31–32 and 4.5–5.7.

**BUILD THIS AS A CONFIGURABLE PERCENTILE FLOOR, NOT A FIXED THRESHOLD:**

```
screen.min_train_pf_pctile        float or null   (percentile of the SURVIVING pool)
screen.min_train_trades_pctile    float or null
```

Applied **after** the four absolute criteria, **computed per direction on the surviving
pool**, and **null disables it**. Rationale for percentile rather than level: a fixed
level is fitted to this frame and fails §1 portability; a percentile of the pool
transfers to a new instrument.

**DO NOT SET THESE FROM THE INCUMBENT'S MEDIANS.** That is fitting to the answer.
**Emit arms at `pctile` in {null, 0.25, 0.50} and let the operator rule from the score.**

**CAUTION, MUST BE PRINTED:** `trades` and `agg_pf` are correlated with each other and
with survivorship; a high floor on both may simply reproduce a smaller book. **The arm
table must show book size after the percentile filter so a size effect is not read as a
quality effect.**

### 4.6 TWO SENSITIVITIES TO EMIT, NOT TO ACT ON

**`trades >= 12` cuts 9,069 of 11,738 rejections (77.3%). Of those, 3,942 (43.5%) clear
every other criterion** — median 8 train trades, median train PF 3.73, median train net
$254, with 2,093 sitting at 8–11 trades. **Same shape as `folds_plus >= 4`: an absolute
count on a population where the count is partly firing rate, not quality.**

**Counter-evidence that must be printed alongside:** the trade floor protects against
small-sample PF artefacts, measured at ~85% of PF=999 cases in this project.

**EMIT `screen.min_trades` AT 8, 10 AND 12 AS THREE ARMS.** Two extra screens on cached
solo data. **Do not move the default on a fitted basis.**

**`train_PF >= 2.0` rejected 3 of the 297.** All LONG, all profitable in train, all with
bucket consistency 0.80–0.83 against a 0.667 bar:

| signal | train tr | train PF | buckets | frac | train net |
|---|---|---|---|---|---|
| `Micro_RangeVelocity:hi + Micro_TickIntensity:hi + Lower_Wick:hi` | 21 | 1.8664 | 6 | 0.833 | +334.60 |
| `RangeOsc_Val:hi + OR_Position:lo + PrevDay_Low_Side:==1` | 39 | 1.5553 | 5 | 0.800 | +381.80 |
| `OBV_Macd:lo + MultiDay_Slope:lo + Lower_Wick:hi` | 37 | 1.9950 | 5 | 0.800 | +903.10 |

**One fails by 0.005.** Reconciliation exact: **280 survived + 3 rejected + 14 absent
from the scan = 297.** The 14 are BOOK-50 members never emitted by the F0 scanner at
`MIN_TRADES = 10`; no screen can admit them.

**EMIT `screen.min_train_pf` AT 1.5 AND 2.0. Report the cost of the level. Do not tune
it to recover those three.**

---

## 5. THE GATE SCAN — THE COMPONENT THAT DOES NOT EXIST

The stage currently reads gates frozen from config. **It must derive them.**

### 5.1 CANDIDATE SPACE

```
90 FEAT_ variables x {hi, lo}           via dots_thresholds, swept percentile
27 binary/state/side variables          via equality ==value
6 gateable cells: (LONG,3) (LONG,4) (LONG,5+) (SHORT,3) (SHORT,4) (SHORT,5+)
```

`tier = min(depth, 5)`. **Tiers 1 and 2 are unreachable under a floor of 3 and MUST NOT
be scanned.** `min(depth,5)` itself is INHERITED with no derivation — record it, do not
re-derive it here.

### 5.2 THE TRIAL-COUNT PROBLEM — THIS IS THE DESIGN CONSTRAINT

`90 x 2 x 6 = 1,080` continuous trials `+ ~486` equality trials `= ~1,566` against
**42 loss events**. Bonferroni requires **p < 0.000032**. A rarity-matched null at 30
draws has a resolution floor of **1/30 = 0.033**. **A NAIVE SWEEP CANNOT CLEAR ITS OWN
CORRECTION BY THREE ORDERS OF MAGNITUDE.**

### 5.3 THE PROCEDURE — HIERARCHICAL, WITH A BOTH-HALVES REQUIREMENT

**STAGE A — CELL ADMISSION (6 tests, not 1,566).**
For each of the 6 cells, test whether the cell is worth gating at all: compare the cell's
losing-bar rate against the book's overall rate, Fisher exact, **Bonferroni at 6 →
p < 0.0083**. Cells failing this are declared FREE and are not scanned further.
**Evidence this works: SHORT d4 and SHORT d5+ survived 20 relocations with nothing
beating free.** Stage A would have closed them for 2 tests instead of 20.

**STAGE B — CANDIDATE SHORTLIST WITHIN AN ADMITTED CELL.**
Rank candidates by **pass rate on that cell's entry bars only**, and retain those in
`[0.5x, 1.5x]` of a target rarity band. **This is not a performance ranking** — it is a
rarity filter and it is what makes the null rarity-matched. Typical shortlist: 40–130.

**STAGE C — SPLIT-HALF, BOTH HALVES REQUIRED.**
Split the train window in half. A candidate must reduce loss events in **BOTH** halves
independently. **This is the correction that replaces Bonferroni**, and it is the one
with direct evidence: the F1 reach ratio was anti-predictive at rho = −0.064 and the
`d_net` ranking at rho = −0.060, both caught by split-half and by nothing else.
Expected survivors per cell: single digits.

**STAGE D — RARITY-MATCHED NULL ON STAGE-C SURVIVORS ONLY.**
Replace-not-stack: the candidate replaces the cell's gate, the rest of the stack holds,
comparison is **loss EVENTS**. Draws: **200 minimum** (resolution 0.005), against the
shortlist from Stage B. Adopt only if `p < 0.05 / (cells admitted in Stage A)`.

**TRIAL ACCOUNTING PRINTED PER CELL:** Stage A tests, Stage B shortlist size, Stage C
survivors, Stage D draws, and the final corrected threshold.

### 5.4 ACCEPTANCE TEST — NON-NEGOTIABLE

**RE-DERIVING FROM SCRATCH MUST RECOVER `Micro_Hurst > p90` AT BOTH d3 CELLS.**

Known: SHORT d3 **p = 0.000** (0 of 30 rarity-matched better, 383 entry bars, 14.9% pass
rate on them); LONG d3 **p = 0.022** (1 of 45, complete rarity-matched set, 553 entry
bars, 11.8% pass). Both found by hand.

**IF THE PROCEDURE CANNOT FIND IT, THE PROCEDURE IS WRONG — NOT THE GATE.** Abort the
gate layer and report which stage dropped it.

**ALSO REPORT WHAT ELSE IT FINDS.** `Micro_FailedBreak > p20` at 80.19% pass rate sits at
LONG d4 and LONG d5+ with **no null behind it** — it is the softest constant in the
adopted stack. A proper scan may beat it.

### 5.5 GATE MASKS — THE SYMBOL, NOT A DESCRIPTION

```python
import swept_thresholds as sw
t = sw.swept(df, {(feat, side): (feat, pct)})[(feat, side)]
mask = df[feat].values > t     # side 'hi', STRICT
mask = df[feat].values < t     # side 'lo', STRICT
```

`sw.swept` substitutes `dt._D_SPEC` and calls the sacred
`dt.compute_adaptive_thresholds`, restoring in a `finally`. **Ring 2500, day-refreshed on
the day-of-month field only, floor-index percentile, no warm-up special case.**
**`dt` emits only p80/p20 — using `ad[(var,'hi')]` for a `> p90` gate is a looser gate and
was a shipped defect.**

**CHECKSUM BEFORE SCORING (this frame, 177,251 bars):**
`HU90 9.7478%` · `FB20 80.1874%` · `ATS90 6.2217%` · `HU90 AND ATS90 0.6838%` ·
`ATR_1M >= 20 20.6072%`. A mask near 20% where HU90 belongs means the p80 series is live.

**`swept_thresholds.py` IS NOT IN THE REPO AT HEAD `69bd7a1`. COMMIT IT.**

---

## 6. DOCUMENTED NEGATIVES — DO NOT BUILD, DO NOT RE-PROPOSE

**A future instance will propose all five of these again. Each is recorded with the
measurement that killed it.**

### 6.1 DIRECTION BALANCE — NEGATIVE
5 balanced draws (191L/106S, stratified) vs 5 unconstrained, n=297, 8,016 pool, seeds
1000–1004. **Loss events: unconstrained median 59 (51–61); balanced median 57 (44–75).**
Worst day **worse** balanced (−1,751 vs −1,478 median). Losing weeks **worse** balanced
(median 2 vs 1). Days and margin identical.
**A two-event median difference inside a 17-event and 31-event spread. The incumbent's
1.80:1 is a property of the 297, not a cause of its performance.**

### 6.2 TERRAIN COVERAGE AS A CONSTRAINT — NEGATIVE
8 draws at n=297, seeds 2000–2007, coverage 27–37 permitted episodes.
**Spearman(coverage, loss events) = −0.184, p = 0.663. Spearman(coverage, PF) = +0.218,
p = 0.604.** Low-coverage half: median 61 events, PF 6.35. High-coverage half: median 63
events, PF 6.55.
**Coverage does not predict outcome among fixed-size books.** It correlates only with
days traded (+0.679, p=0.064) and worst day (+0.764, p=0.027) — i.e. **more coverage
means more participation and a WORSE tail, which is the concentration finding again.**
Coverage-as-objective was already dead (98% of the landscape is reachable by firing
everywhere; the density control killed every non-F0 family). **Coverage-as-constraint is
now dead too.**

### 6.3 PARTICIPATION — NEGATIVE, PRICED
Unconstrained band at n=297 is **109–118 days; the incumbent is 119.** The incumbent sits
at the top of the band, not outside it. **Maximum headroom is one day.** Not worth a
constraint.

### 6.4 PER-SIGNAL RANKING — NEGATIVE
A rebuilt selector ranking members **lost to 12 of 12 random draws** on a holdout. Its
objective — minimising the union of losing bars — selected for signals that lose
together, which is the many-clones-on-one-bar mechanism written into a scoring function.

### 6.5 `d_net` / LEAVE-ONE-OUT AS A QUALIFIER — NEGATIVE
297 leave-one-out runs: `d_net` IQR 3.4x (p25 $531.60, p75 $1,798.90), 12 of 297
negative, **split-half rho = −0.060 at p = 0.305**, `d_events` zero for 212 of 297, four
source objectives differing 1.7x, 50-draw ablation ratio 0.960 ± 0.066 (no superadditive
structure). **And `d_net` measures floor-criticality, not quality: 121 of 973 entry bars
sit at depth exactly 3, so a member's delta is largely how often it is the marginal third
signal.** Two identical signals score differently.

---

## 7. NOT TESTED — REGIME

Regime has never been tested as a selection input. `concurrence_regimes.csv` and
`regime_labels.csv` exist as artifacts. **No claim either way. Do not build it until it
is measured; do not treat its absence as a negative.**

---

## 8. WHAT THE SHELL MUST PRINT — THE OPERATOR'S ACCEPTANCE TEST

**IF HE CANNOT SEE IT IN THE SHELL, IT IS NOT BUILT.** One run, in this order:

```
BANNER      config name + sha, N, resolved train window (first/last bar ts), bucket count,
            scan_input_sha == frame_input_sha assertion result
VARIABLES   117 scanned: 90 FEAT_ x2 + 27 equality; gateable cells 6; tiers 1-2 excluded
SCREEN      raw rows -> survivors, with the 4-criterion rejection breakdown,
            AND whether reject_criterion is first-match or all-match
SENSITIVITY min_trades at 8/10/12 and min_train_pf at 1.5/2.0: survivor counts
GATES       per cell: Stage A verdict, Stage B shortlist, Stage C both-halves survivors,
            Stage D null draws and p, adopted gate or FREE
            AND the acceptance line: Micro_Hurst > p90 recovered at LONG d3 / SHORT d3 Y/N
MASKS       pass-rate checksum for every adopted gate
ARMS        each size: result, its band, INSIDE/ABOVE/BELOW, and the band's N and window
BOOK        the book_size arm, direction split, artifacts written with shas
SCORE       full scorecard, loss EVENTS beside trade losses, weekly, monthly,
            depth ladder per direction, gate admit rates
SIDE-BY-SIDE  vs incumbent on every column, plus overlap: in-297 / new / rejected-with-criterion
EXPECTATION the §3 statement: an arm inside its band has reproduced the random baseline,
            not the book
RUNTIME     per stage and per arm
```

---

## 9. THE EXPECTATION THAT MUST BE PRINTED

**DO NOT STATE A NET BAND AS THE PASS MARK.** A random draw lands at 60–65% of the
incumbent's net, which reads as success, while **loss events run 51–61 against 42, the
worst day is 3–8x deeper, and no draw holds 26 clean weeks.**

**A READER CHECKING NET ALONE WOULD REPORT SUCCESS ON A FAILING BOOK. THAT IS THE MOST
LIKELY WAY THIS GOES WRONG — NOT A CRASH, A PASSING VERDICT.**

**AND PRINT THIS, SO IT IS NOT REDISCOVERED AS A BUG:**

> The incumbent's 297 sits at roughly the 4th percentile of random draws of its own size.
> **280 of its 297 members survive the screen — the book is in the pool at 94.3% — and a
> seeded draw finds 8, against 10.4 expected by chance.** The draw is performing exactly
> as random selection should.
> **Nothing in the members explains the incumbent's edge**: 297 leave-one-out runs, a
> 50-draw ablation, split-half at rho = −0.060, four indistinguishable source objectives.
> **The one per-member signal that does exist (§4.5) is an upper-tail effect on `trades`,
> `agg_pf` and `folds_plus`, present on BOTH directions.**
> **If SELECT reproduces the random baseline after §4.5 and §5 are built, that is the
> finding: the 297's edge lives in the specific combination, and no reproducible procedure
> can currently regenerate it.**

---

## 10. CONSTRAINTS

Byte-identical artifacts across runs and worker counts. No wall-clock in any CSV. Every
draw seeded, seed recorded in every artifact row.

**Survivors sorted by `(signal_def, direction)` — never a float — before drawing, and the
survivor list asserted identical across worker counts before the draw.** A recorded seed
on an unordered population is cosmetic.

**Arms nested from one permutation**, `book_size` naming which arm is emitted as
`<name>_signals.csv`; abort if `book_size` is not among `draw.arm_sizes`.

Sacred five untouched. `adm_engine` with the fork-parity assertion. `_assert_book_grammar`
kept — the book must be 100% three-condition F0 triples.

**BOOK-50 and the 297 must score identically to the cent after wiring:**
`3,101 / 90.6% / 4.81 / $97,675` with the canary — **this reference has never been
independently verified; it originates in the Developer's environment** — and
`5,776 / 96.12% / 14.53 / $284,974 / 42 events on 35 days / 0 losing weeks of 26 /
119 of 132` — **independently reproduced, every figure confirmed.**
