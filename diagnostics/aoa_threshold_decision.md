# AoA applicability threshold — decision record

**Date:** 2026-08-10 · **Status:** decided, pipeline reconciliation pending (see Consequences) ·
**Supersedes:** the `measured_skill_envelope` rule in `aoa_calibration.py` / `models/aoa_threshold.json`
(threshold 0.506) and FINDINGS.md flag 7's "~15× floor, no decay" wording.

## Decision

The area-of-applicability cutoff is **DI = 0.27**, defined as the **99.9th percentile of the
cross-validated (out-of-fold, spatial-block) training dissimilarity index**. Inside the AoA iff
`DI ≤ 0.27`. This is a **feature-space envelope** rule: a cell is inside iff it is no more
dissimilar from the training data than all but the most isolated 0.1% of training points are from
one another. It is **not** a skill boundary.

## Why we revisited it (the trigger)

The old calibration reported that pooled-OOF AUC-PR stayed **~15× the prevalence floor across the
entire tested DI range with no decay**, and used that flat curve to justify pushing the threshold
out to the maximum OOF DI (0.506), ~2.8× the box-plot fence (0.179). A flat, no-decay skill curve
across the whole applicability range is exactly what an applicability index should *not* produce,
so we audited it.

## What we found (the old result is an artifact)

The "~15× floor, no decay" is a metric artifact, from three compounding causes:

1. **Wrong baseline.** Per-bin AUC-PR was compared to the *global* prevalence floor (0.0574), but
   AUC-PR's chance baseline is the *local* per-bin prevalence, which **drifts ~8× with DI** (bin
   prevalence rises 0.04 → 0.34 from low to high DI as the minority class concentrates at higher
   dissimilarity). Normalised by the correct local baseline, the AUC-PR "lift" *falls* from ~22× to
   ~3×. The flat curve is two effects cancelling, not stable skill.
2. **Equal-count binning erased the tail.** DI is heavily right-skewed (grid median 0.167, training
   self-DI median 0.079). 98% of OOF points sit below the box-plot fence; only 12 of 19,288 exceed
   DI 0.30. The calibration's top equal-count bin lumps everything from 0.135 to 0.506 into one
   average dominated by low-DI points, so decay above ~0.15 is invisible by construction.
3. **Threshold pinned to an extreme order statistic.** 0.506 is essentially the single most-isolated
   OOF point. The interval 0.18–0.51 (which the old rule declared "reliable") is "verified" by ≤12
   points.

AUC-PR is the wrong instrument here: the data is severely imbalanced (~93/7) and per-bin prevalence
drifts with DI, so AUC-PR against any fixed floor cannot read as a degradation curve.

## What we can trust

- **Prevalence-invariant skill does not decay.** Per equal-width DI bin, **OOF AUC-ROC = 0.97–0.99,
  flat**, up to the last bin with enough held-out positives to measure (≈0.25). Skill was never
  observed to degrade anywhere it could be measured.
- **DI is a valid applicability index.** `Spearman(DI, |OOF residual|) = +0.49` — DI positively
  tracks where the model errs. (The residual rise is partly prevalence-driven, so it is corroborating,
  not a cutoff basis — see rejected options.)
- Pooled OOF AUC-PR = 0.84; class-1 (Non-abrupt) floor = 0.0574; CV = albers_grid 10 km, 5 folds,
  buffer 0 km, seed 42; dbar = 0.116.

## Options considered

| principle | DI | % state flagged | verdict |
| --- | --- | --- | --- |
| training-DI p95 | 0.149 | 62% | too strict |
| box-plot fence Q75+1.5·IQR | 0.179 | 44% | literal M&P, arbitrary 1.5·IQR |
| training-DI p99 | 0.194 | 36% | robust but flags salt-and-pepper interior |
| **training-DI p99.9** | **0.267** | **9.7%** | **chosen** |
| grid-DI p90 | 0.263 | ~10% | rejected: circular (a grid quantile, not an envelope) |
| max OOF DI (old rule) | 0.506 | 3% | rejected: single-point extreme, over-permissive |

Two principles were explicitly **rejected**:

- **Residual-onset ("cut where OOF error rises", the Spearman idea).** `mean|residual|` crosses 1.5×
  baseline at DI≈0.11 (86% flagged) and 2× at 0.15 (59%), but that rise tracks prevalence, not skill
  (AUC-ROC stays 0.97–0.99). Same trap as AUC-PR. The Spearman coefficient is a correlation in
  [-1,1], not a DI value, so it cannot *be* a cutoff.
- **Snow-feature removal.** Dropping the two SWE features from the distance shrinks the flagged area
  from 3.0% to 0.4% at the max cutoff, but the domination just relocates to Slope (18% → **71%** of
  the outside-cell distance), and the model still *uses* SWE to predict, so removing it from the
  reliability metric would blind the AoA to genuine snow-extrapolation. Kept as a **diagnostic only**
  (it shows the flagged region is a snow/relief coverage story), not the operative metric.

## Why p99.9 / 0.27

Skill shows no decay anywhere measurable, so the cutoff cannot be defined by "where skill fails" —
there is no such point in-sample. The honest basis is therefore a pure **feature-space envelope**:
a quantile of the training set's own out-of-fold dissimilarity, using the training set as its own
yardstick. p99.9 is robust (not a single-point max), literature-aligned (the M&P AoA idea with an
explicit percentile instead of the arbitrary 1.5·IQR fence), and sits just past the last DI where
skill was directly verifiable (~0.25). It is **not** "the last flat-skill bin" (a coincidentally
near, binning-dependent quantity); flat skill is corroboration, not the definition.

At 0.27, the flagged 9.7% is dominated by **snow and relief coverage gaps** (Trend in SWE ~30%,
Mean Annual SWE ~29%, Slope ~18% of the outside-cell distance) — genuine feature-space extrapolation,
not demonstrated model failure.

## Consequences / follow-ups

- **Fig 3b** rewritten (`output/fig03_thaw_mode_and_aoa.py`): continuous DI graded blue inside the
  AoA, single solid red for `DI > 0.27`, colorbar shows the ramp ending in an "outside AoA" red
  block. The earlier hatch + per-pixel contour rendering visually inflated the flagged area and is
  gone.
- **Reconciliation (step 4, DONE):** `models/aoa_threshold.json` (0.267), `models/aoa.py` (loads the
  threshold from that JSON — no hardcode), `diagnostics/aoa_calibration.py` (`choose_threshold`,
  `ENVELOPE_PCTL = 99.9`), FINDINGS.md flag 7, PIPELINE.md, and memory all now carry 0.27; the
  `aoa.nc` `aoa_threshold` attr and `inside_aoa` layer follow from `aoa.py` on regeneration. The
  continuous DI raster in `aoa.nc` was **unchanged** (DI does not depend on the threshold).
- **Headline recomputed (step 4, DONE):** the in-AoA abrupt-favoring fraction is reported under the
  0.27 mask as 24.7% on an area basis (≈264,000 km² of 1.07 million km²) in the manuscript; the
  ~25.6% under the old 0.506 mask is superseded.

## Reproduce

- Per-bin AUC-ROC / AUC-PR / local-prevalence vs DI, and the %-flagged-vs-threshold sensitivity, are
  computed from the pooled-OOF predictions (`spatial_cv.pooled_oof_predict`) and the operative
  rank-CDF DI helpers in `models/aoa.py`, over the equal-width DI bins (not the equal-count bins
  `aoa_calibration.py` uses). Training self-DI percentiles: median 0.079, p99 0.194, **p99.9 0.267**,
  max 0.506. Driver shares are `models/aoa.py:extrapolation_drivers`.
