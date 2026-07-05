# Blind FMC Imaging of Sub-Wavelength Voids — Insights

Summary of the findings from the blind-imaging study (PR #9, June 2026):
reconstructing sub-wavelength air voids from Full Matrix Capture (FMC)
ultrasonic data at 5 MHz **without Born modelling and without a void-free
reference simulation**, using the pylops `Kirchhoff` operator with external
traveltime/amplitude tables (`mode="byot"`).

## The four studies

| script | scenario | result |
|---|---|---|
| `tutorials/deepwave_fmc_kirchhoff_blind.py` | contact-style array, homogeneous aluminum plate, one 0.6 mm (0.48 λ) void | 0.30 mm (0.24 λ) error, 151× contrast |
| `tutorials/deepwave_fmc_kirchhoff_blind_multi.py` | same, five random voids (0.44–0.80 mm, all < λ) | all detected; mean 0.34 mm (0.27 λ), contrast 32–86× |
| `tutorials/deepwave_fmc_kirchhoff_blind_multi_compare.py` | five-void data, old pipelines vs new (time-zero corrected for all) | localisation is a tie (0.32–0.38 mm for every pipeline); the blind pipeline wins on **contrast** (min/mean 32×/57× vs 25×/35× for the best old conditioning; ungated pipelines ~1–3×, voids invisible) |
| `tutorials/deepwave_fmc_kirchhoff_blind_immersion.py` | water / 20 mm aluminum slab / water, five voids, array in water | all detected; mean 0.37 mm (0.30 λ), contrast 25–73× |

All data come from a **single** deepwave finite-difference simulation of the
true medium per case. The imaging side knows only: the receiver signals, the
array geometry, the layer velocities/thicknesses, and traveltime + amplitude
tables.

Everything the studies share — the seeded void recipe, the simulation
harness, the preprocessing chain, the eikonal solver and the byot table
construction — lives once in `tutorials/fmc_blind_utils.py`; the shoot-out
loads (and verifies) the five-void dataset saved by the blind_multi run
instead of re-deriving it from copied code. Two validation guards run before
any result is trusted: an eikonal self-test against the analytic homogeneous
solution, and a cross-check of the fork's byot + raw-amplitude Kirchhoff path
against an independently assembled VStack of per-shot operators (passes at
machine precision, ~2×10⁻¹⁶).

## The blind preprocessing pipeline

Ordered by impact:

1. **Common-offset median subtraction** — the core trick and the legitimate
   replacement for Born-style reference subtraction. At a fixed element offset
   `k = j − i`, every laterally invariant event (direct wave, front-wall echo,
   back-wall echo, water multiples, slab reverberations) is *identical* for
   all element pairs, while a localized void diffraction only appears in a few
   pairs. The median trace over pairs therefore estimates exactly the clutter
   and none of the signal. In the corrected shoot-out it lifts the *minimum*
   per-void contrast from 25× (best old conditioning) to 32× and the mean
   from 35× to 57×; against ungated data (contrast ~1–3×, voids invisible)
   it is the difference between detecting and not detecting. Offsets with
   fewer than ~4 pairs have no robust median and are zeroed (negligible
   aperture loss).
2. **Time-zero correction** — the excitation pulse peaks `peak_time` after the
   electrical time zero (0.3 µs for a 1.5/f Ricker delay). Uncorrected, this
   maps to a systematic ~0.9 mm depth bias — exactly what the old studies
   exhibited. It is a known instrument property, not a conditioning choice,
   so the redone shoot-out applies it to *every* pipeline; with it in place,
   localisation error is essentially pipeline-independent (0.32–0.38 mm
   mean for all five pipelines) and the pipelines differentiate on contrast
   only. (The earlier version of the shoot-out corrected only the blind
   pipeline, which overstated its localisation advantage.)
3. **Inspection time gates** — top mute after the direct wave / front-wall
   echo, bottom gate just before the back-wall echo. Built only from known
   geometry and velocity. In the layered case the wall reflection time for
   pair (i, j) is obtained *from the traveltime tables themselves*:
   `min over x of [T_i(x, z_wall) + T_j(x, z_wall)]`.
4. **Zero-phase bandpass** (2.5–7.5 MHz) and a **linear t-gain** — minor but
   free.
5. **Envelope AFTER migration, never before.** Taking the Hilbert envelope of
   the data before migration (an old variant) destroys the phase information
   the Kirchhoff correlation needs: enveloping the total field leaves the
   voids as invisible as migrating the raw data (contrast ~1× at the worst
   void), and enveloping the muted data costs contrast relative to keeping
   the phase (21×/28× vs 25×/35× min/mean). The envelope of the migration
   *image* (along z) is purely cosmetic and always helps readability.

## Operator usage (pylops fork)

- `Kirchhoff(mode="byot", trav=(trav_srcs, trav_recs), amp=(amp_srcs,
  amp_recs), dynamic=False)` applies the external amplitudes verbatim. Dense
  FMC (every element records every shot) is the operator's default geometry;
  `shot_recs` is only needed for sparse acquisitions (and would also disable
  `engine="cuda"`).
- Homogeneous medium: both tables come from the operator itself (traveltimes
  *and* Euclidean distances; spreading weights `1/sqrt(dist + eps)` with the
  operator's own `eps = 1e-2 · max dist` regularisation). This recipe — and
  the underlying call to the private `Kirchhoff._traveltime_table` — lives
  only in `fmc_blind_utils.byot_tables`, so a future public table-building
  API in the fork needs a one-file change.
- Layered medium: traveltimes from the shared numba fast-sweeping eikonal
  solver (no scikit-fmm wheel for Python 3.14), self-tested against the
  analytic homogeneous solution before use. For laterally invariant models
  one padded solve is shifted to all elements (~30× less eikonal work than
  one solve per element). The distance/amplitude tables still come from the
  operator's analytic mode because Euclidean distance is
  velocity-independent.

## Physics insights

- **Scenario dominates pipeline.** Rerunning the old pipelines on the new
  contact-scenario data made them look far better than in their original
  immersion setting. Cleaner input physics (no interface losses, no
  reverberation train) and an exact analytic operator improve *every*
  pipeline indiscriminately; only the controlled same-data comparison
  isolates the pipeline's own contribution.
- **The water–aluminum interface costs ~an order of magnitude.** ~30% energy
  transmission per crossing, so a void echo (two crossings each way) returns
  ~1000× weaker than the front-wall echo — visible directly in the A-scan
  amplitude scales (1e-10 raw vs 1e-13 void echoes). The blind pipeline still
  pulls all five voids out of it.
- **Sub-wavelength voids are detectable and localisable to ~λ/4–λ/3**, but
  not *sized*: a point-like focus is all Kirchhoff migration can return below
  the diffraction limit; the focus width reflects the imaging wavelength and
  aperture, not the void diameter (0.44 mm and 0.80 mm voids look alike).
- **Contrast decays with depth** (86× shallow → 25× deep): spreading loss plus
  shadowing by shallower scatterers; the static `1/sqrt(dist)` weights only
  partly compensate.
- **Thicker slab = cleaner immersion imaging.** Going from the original 14 mm
  to a 20 mm slab pushes the first in-slab reverberation past the back-wall
  echo, leaving a wide clean inspection window. With the median subtraction
  also cancelling the reverberations, the 20 mm slab worked on the first
  attempt.

## Limitations / honest caveats

- Scalar acoustics: no shear waves or mode conversion (a real aluminum
  inspection has both), no attenuation, no electronic noise.
- The medium must be laterally invariant for the median subtraction; a curved
  or rough wall would need a model-based or adaptive clutter estimate.
- First-arrival eikonal times include head waves, which can differ slightly
  from the specular reflection path used for gating (fine at these geometries).
- Kirchhoff migration remains a linearised (Born-type) *imaging* operator even
  when the data are full-wave: amplitudes near strong scatterers are
  qualitative, multiples between voids are not modelled (they were gated or
  median-subtracted away here).

## Environment notes

- Run tutorials with `PYTHONPATH=.` from the repo root: site-packages holds
  stock pylops 2.7.0, which lacks `shot_recs` and byot `amp` tuples and will
  otherwise shadow this fork.
- GPU (torch 2.12 + cu126): a 32-shot 800×600 deepwave FMC simulation takes
  ~2 s; the three migrations ~2 s; the 32-element eikonal tables are now
  effectively free (single padded solve + shifts, <0.1 s); the byot operator
  cross-check ~2 s.

## Next steps (from the broader survey)

The natural escalation path toward fully realistic inspection: elastic
simulation (mode conversion), noise robustness, adaptive clutter estimation
for non-flat geometry, and ultimately **FWI with deepwave's differentiable
propagators** — which removes the linearised-imaging limitation as well.
