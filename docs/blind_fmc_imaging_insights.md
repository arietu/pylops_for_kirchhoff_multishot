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
| `tutorials/deepwave_fmc_kirchhoff_blind_multi_compare.py` | five-void data, old pipelines vs new | blind pipeline wins every void on both metrics |
| `tutorials/deepwave_fmc_kirchhoff_blind_immersion.py` | water / 20 mm aluminum slab / water, five voids, array in water | all detected; mean 0.37 mm (0.30 λ), contrast 25–73× |

All data come from a **single** deepwave finite-difference simulation of the
true medium per case. The imaging side knows only: the receiver signals, the
array geometry, the layer velocities/thicknesses, and traveltime + amplitude
tables.

## The blind preprocessing pipeline

Ordered by impact:

1. **Common-offset median subtraction** — the core trick and the legitimate
   replacement for Born-style reference subtraction. At a fixed element offset
   `k = j − i`, every laterally invariant event (direct wave, front-wall echo,
   back-wall echo, water multiples, slab reverberations) is *identical* for
   all element pairs, while a localized void diffraction only appears in a few
   pairs. The median trace over pairs therefore estimates exactly the clutter
   and none of the signal. Worth most of the 2–17× contrast gain over the old
   conditionings. Offsets with fewer than ~4 pairs have no robust median and
   are zeroed (negligible aperture loss).
2. **Time-zero correction** — the excitation pulse peaks `peak_time` after the
   electrical time zero (0.3 µs for a 1.5/f Ricker delay). Uncorrected, this
   maps to a systematic ~0.65–0.9 mm depth bias — exactly what all the old
   pipelines exhibited (0.63–0.68 mm mean error vs 0.34 mm corrected). A
   one-line fix that halves the localisation error on its own.
3. **Inspection time gates** — top mute after the direct wave / front-wall
   echo, bottom gate just before the back-wall echo. Built only from known
   geometry and velocity. In the layered case the wall reflection time for
   pair (i, j) is obtained *from the traveltime tables themselves*:
   `min over x of [T_i(x, z_wall) + T_j(x, z_wall)]`.
4. **Zero-phase bandpass** (2.5–7.5 MHz) and a **linear t-gain** — minor but
   free.
5. **Envelope AFTER migration, never before.** Taking the Hilbert envelope of
   the data before migration (an old variant) destroys the phase information
   the Kirchhoff correlation needs — it was the worst pipeline in the
   shoot-out (contrast down to 1.5×). The envelope of the migration *image*
   (along z) is purely cosmetic and always helps readability.

## Operator usage (pylops fork)

- `Kirchhoff(mode="byot", trav=(trav_srcs, trav_recs), amp=(amp_srcs,
  amp_recs), dynamic=False, shot_recs=[arange(nr)]*ns)` encodes FMC and
  applies the external amplitudes verbatim.
- Homogeneous medium: both tables come from the operator itself —
  `Kirchhoff._traveltime_table(zm, xm, srcs, recs, v, mode="analytic")`
  returns traveltimes *and* Euclidean distances; spreading weights
  `1/sqrt(dist + eps)`.
- Layered medium: traveltimes from the repo's numba fast-sweeping eikonal
  solver (no scikit-fmm wheel for Python 3.14); the distance/amplitude tables
  can still come from the operator's analytic mode because Euclidean distance
  is velocity-independent.

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
  ~2 s; the three migrations ~2 s; the 32-element eikonal tables ~2 s.

## Next steps (from the broader survey)

The natural escalation path toward fully realistic inspection: elastic
simulation (mode conversion), noise robustness, adaptive clutter estimation
for non-flat geometry, and ultimately **FWI with deepwave's differentiable
propagators** — which removes the linearised-imaging limitation as well.
