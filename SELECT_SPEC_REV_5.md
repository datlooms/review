# SELECT_BUILD_SPEC.md — REVISION TO §5.3 AND §5.4
# Supersedes §5.3 and §5.4 in full. §1, §2, §3, §4, §6-§10 unchanged.

## §5.0 WHY THIS REVISION EXISTS

**STAGE A AS WRITTEN IS FALSIFIED.** It asked "is this cell worse than the book?" and
admitted only if so. **SHORT d3 is BETTER than the book (5.21% vs 7.29%) and is therefore
closed before any candidate is tested — yet `Micro_Hurst > p90` demonstrably works there
at p = 0.000, 0 of 30 rarity-matched alternatives better.**

**"WORTH GATING" IS NOT "WORSE THAN AVERAGE".** A gate separates winners from losers
*within* a cell; that is orthogonal to whether the cell beats the book average. The
one-sided test was structurally incapable of admitting an outperforming cell, and §5.4's
acceptance test could not have passed as written.

**AND EVERY MEASUREMENT CITED IN §5 CARRIED THE WRONG BASIS.** LONG d3 at 1,269 trades /
8.75% and SHORT d3 at 945 / 5.40% were measured on the ungated field at **FLOOR = 1**;
the stage runs **FLOOR 3/3**, where the jar is not consumed by solos and duals. **The
§2 discipline — every figure carries the configuration it was measured on — extends to
every measurement this document cites, not only to bands.**

---

## §5.1 THE CELL BASIS — MEASURED, AND THIS IS THE BASIS ALL OF §5 USES

**Configuration: 297 book · FLOORED · floor 3/3 · cap 21 · `ATR_1M >= 20` ·
`recentfb_sizing = False` · NO TIER GATES · full frame 177,251 bars · 1.0 lot.**

Book: **9,123 trades · 1,990 entry bars · 145 loss events · 7.29%**

| cell | trades | bars | **loss events** | loss% | vs book |
|---|---|---|---|---|---|
| LONG d3 | 1,614 | 538 | **61** | 11.34% | worse |
| LONG d4 | 1,212 | 303 | **32** | 10.56% | worse |
| LONG d5+ | 3,767 | 478 | **26** | 5.44% | better |
| SHORT d3 | 1,152 | 384 | **20** | 5.21% | better |
| SHORT d4 | 652 | 163 | **4** | 2.45% | better |
| SHORT d5+ | 726 | 124 | **2** | 1.61% | better |

Reproduces the Developer's run exactly. **The candidate space is the S2 pool as received:
249 conditions. 249 x 6 = 1,494 trials — not the 117 x 6 = ~1,566 §5 estimated.**

---

## §5.2 TWO MEASURED CONSTRAINTS THAT DECIDE THE PROCEDURE

**CONSTRAINT 1 — TWO CELLS CANNOT SUPPORT A TEST AT ALL.**
SHORT d4 carries **4** loss events, SHORT d5+ carries **2**. A gate removing one event
there is a one-event observation, and this project's standing rule is to report the count
and decline the rate.

**CONSTRAINT 2 — SPLIT-HALF IS WEAKEST EXACTLY WHERE IT IS MOST NEEDED. THIS IS NEW AND
IT OVERTURNS THE §5.3 DRAFT.**
**SHORT d3's 20 loss events split 7 in half A and 13 in half B.** A both-halves
requirement at SHORT d3 rests on **7 events in one half**. Split-half was proposed as the
primary correction because it caught the F1 reach ratio at rho = −0.064 and `d_net` at
rho = −0.060 — **but both of those were rankings over hundreds of members, not a
7-event cell.** *(Measured on the §5.1 basis.)*

**SPLIT-HALF CANNOT CARRY THE CORRECTION ALONE. IT IS A FILTER, NOT A TEST.**

**CONSTRAINT 3 — THE RARITY SHORTLIST DOES NOT CUT MUCH.**
At SHORT d3, `Micro_Hurst > p90` passes **14.84%** of the cell's 384 entry bars. A
`[0.5x, 1.5x]` rarity band — `[0.074, 0.223]` — retains **66 of 249 candidates (26.5%)**,
not "a stated few". *(Measured on the §5.1 basis.)*

**RESULTING ARITHMETIC.** 4 testable cells x 66 shortlisted = **264 trials**. Per-cell
Bonferroni at 66 requires **p < 0.00076**. A rarity-matched null resolves at `1/draws`,
so 0.00076 needs **>1,300 draws per candidate** at ~9.3 s/draw = **>3 hours per
candidate**. **A DISCOVERY-SCALE SCAN CANNOT CLEAR ITS OWN CORRECTION AT THIS EVENT COUNT.
THAT IS ARITHMETIC, NOT AN IMPLEMENTATION PROBLEM.**

---

## §5.3 THE PROCEDURE — REVISED

### STAGE A — POWER FLOOR (replaces the quality gate)

```
gates.min_cell_events   int    required config key    (this frame: 10)
```

For each of the 6 cells compute loss events on the §5.1 ungated basis. **Admit the cell
iff `cell_events >= gates.min_cell_events`.** Print every cell with its event count and
`TESTABLE` / `BELOW POWER FLOOR`.

**This asks "can this cell support a test?", not "is this cell bad?".** On this frame it
admits LONG d3 (61), LONG d4 (32), LONG d5+ (26), SHORT d3 (20) and closes SHORT d4 (4)
and SHORT d5+ (2) **on a stated basis rather than by accident** — which is the same
verdict 20 hand-relocations reached for those two cells.

**FAILURE MODE:** if fewer than 2 cells admit, abort the gate layer and print the event
counts. A frame that cannot support 2 cells cannot support gate derivation.

### STAGE B — RARITY SHORTLIST, BEFORE ANY OUTCOME IS READ

```
gates.rarity_lo   float   (this frame: 0.5)
gates.rarity_hi   float   (this frame: 1.5)
gates.rarity_ref  float   target pass rate, or null to use the cell's own median
```

Compute each of the 249 pool conditions' pass rate **on that cell's entry bars only**.
Retain candidates whose pass rate is within `[rarity_lo x ref, rarity_hi x ref]`.
**This is a rarity filter, not a performance filter — it is what makes the Stage D null
rarity-matched.** Print the band and the shortlist size. Expect ~25% of the pool.

### STAGE C — SPLIT-HALF AS A FILTER (demoted from primary correction)

Candidate must reduce book loss events in **both** halves of the frame independently.
**Print the per-half event base for the cell** — Constraint 2 — so a both-halves result
on 7 events is not read as a test. **This removes obvious noise. It does not establish
significance and must not be reported as if it does.**

### STAGE D — RARITY-MATCHED NULL, TWO-TIER OUTPUT

Replace-not-stack against the Stage B shortlist. Comparison is **loss EVENTS**.

```
gates.null_draws          int    (this frame: 200; resolution 1/draws = 0.005)
gates.confirm_alpha       float  (this frame: 0.05)
```

**EMIT TWO TIERS AND NEVER CONFLATE THEM:**

**`CONFIRMED`** — cleared `p < confirm_alpha / (shortlist_size x cells_admitted)`.
On this frame that is `0.05 / 264 = 0.00019`, which **200 draws cannot resolve**.
**EXPECT THIS TIER TO BE EMPTY. AN EMPTY CONFIRMED TIER IS THE CORRECT RESULT, NOT A
FAILURE.**

**`CANDIDATE`** — cleared Stage C and `p < confirm_alpha` uncorrected. **Each row prints
its uncorrected p, the trial count it was drawn from, the corrected threshold it did NOT
meet, and the per-half event base.** This is a shortlist for the operator to rule on, and
the document must say so in the header of the artifact.

**THE STAGE ADOPTS NOTHING FROM THE `CANDIDATE` TIER AUTOMATICALLY.** Adopted gates come
from config. The scan proposes; the operator disposes.

---

## §5.4 ACCEPTANCE TEST — REVISED, AND THE REASON IT MUST CHANGE

**ORIGINAL:** re-deriving from scratch must recover `Micro_Hurst > p90` at both d3 cells.

**REVISED:** `Micro_Hurst > p90` must appear in the **`CANDIDATE`** tier at **both**
LONG d3 and SHORT d3. **It is not required to reach `CONFIRMED`, and §5.2 shows why: at
264 trials against 20 events the corrected threshold is unreachable with any feasible
null.**

**AND HERE IS WHAT HAND-DERIVATION HAD THAT THE PROCEDURE DOES NOT — the operator asked
for this to be named if the procedure could not match it.**

`Micro_Hurst > p90` cleared at **p = 0.022 (LONG d3, 1 of 45)** and **p = 0.000
(SHORT d3, 0 of 30)** because it was tested as **ONE pre-registered candidate at ONE cell,
on a stated mechanism — persistence, for an entry that must stand on fewer confirmations.
Trial count 1.** The same p-value drawn as the best of 66 shortlisted candidates means
nothing: at 66 trials, `p = 0.022` is expected roughly 1.5 times by chance.

**THE MISSING INFORMATION IS A PRIOR.** Hand-derivation restricted the candidate space to
one variable *before seeing outcomes*, on a mechanism argument. A scan over 249 conditions
has no prior and must pay for the breadth.

**THEREFORE ADD A THIRD, OPTIONAL PATH:**

```
gates.preregistered   list[{cell, condition}]   or null
```

Candidates named here are tested at **trial count = len(preregistered)**, separately from
the scan, and may reach `CONFIRMED` at a threshold the scan cannot. **A pre-registered
candidate must be declared in config BEFORE the run and printed in the banner.** This is
how a mechanism argument enters the pipeline honestly — **it does not license adding a
candidate after seeing the scan output, and the artifact must record the config sha so
that cannot be done silently.**

---

## §5.5 WHAT THE GATE LAYER PRINTS

```
CELL BASIS    per cell on the §5.1 ungated basis: trades, bars, loss events, loss%,
              and the configuration line (floor, cap, ATR, gates=NONE)
STAGE A       per cell: events, min_cell_events, TESTABLE / BELOW POWER FLOOR
STAGE B       per admitted cell: rarity ref, band, shortlist size of 249
STAGE C       per admitted cell: per-half event base (e.g. SHORT d3: 7 / 13),
              candidates improving both halves
STAGE D       per admitted cell: null draws, resolution, corrected threshold,
              CONFIRMED list (expect empty), CANDIDATE list with uncorrected p,
              trial count, and the threshold not met
PREREG        pre-registered candidates, their trial count, and their verdict
ACCEPTANCE    Micro_Hurst>p90 in CANDIDATE at LONG d3: Y/N | at SHORT d3: Y/N
TRIALS        total trials attempted across all cells
```

---

## §5.6 NOT COMPLETED — NAMED

**The Stage B/C scan at SHORT d3 was executed and did not finish.** 66 shortlisted
candidates at ~9.3 s each; the run reached 20 of 66 before exceeding its wall, with no
checkpoint — an error in the harness, not the procedure.

**What it established before stopping and what this revision rests on:** the cell basis
(§5.1, reproduces the Developer exactly), the 7/13 half split at SHORT d3 (§5.2
Constraint 2), and the 66-of-249 shortlist size (§5.2 Constraint 3).

**What remains: run the 66 shortlisted candidates at SHORT d3 and the equivalent at
LONG d3, and report whether `Micro_Hurst > p90` appears in the CANDIDATE tier at both.**
Roughly 10 minutes per cell with checkpointing every 5 candidates. **Until that runs, the
§5.4 acceptance test is specified but UNVERIFIED — the procedure may still fail it, and
if it does, the pre-registered path in §5.4 is the answer rather than a further revision
of the scan.**
