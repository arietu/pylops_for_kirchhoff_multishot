# deepwave FMC simulation + Kirchhoff migration (multishot vs VStack)

Date: 2026-06-09

## Goal

Simulate a Full Matrix Capture (FMC) ultrasonic acquisition over a three-layer
medium (water / aluminum / water) containing randomly placed air voids using
the `deepwave` FDTD package, then migrate the recorded data with the pylops
`Kirchhoff` operator built two ways — the built-in multi-shot interface and a
`pylops.VStack` of per-shot operators — using a migration velocity equal to the
three-layer model **without** the air voids. Verify the two migrations are
equal, then plot and save the results.

## Physical model (ultrasonic NDT scale, mm / MHz)

- Domain: 40 mm (x) x 30 mm (z), simulation grid `dx = dz = 0.1 mm` (400 x 300).
- Layers: water 0-8 mm (1480 m/s), aluminum 8-22 mm (6300 m/s), water 22-30 mm.
- Air voids: 6 random non-overlapping discs (radius ~1 mm, 343 m/s) placed
  inside the aluminum layer only; fixed RNG seed.
- `vel_true`: model with voids (drives deepwave).
- `vel_mig`: identical three-layer model with voids filled back to aluminum
  (drives Kirchhoff migration). Downsampled to the 0.2 mm migration grid.

## Acquisition (deepwave, acoustic scalar, GPU)

- 32-element array, pitch 1 mm, depth ~2 mm (top water), centered.
- FMC = 32 shots batched on the shot dimension: shot i fires a 1 MHz Ricker at
  element i; all 32 elements record. Recorded data cube: `(32, 32, nt)`.
- `dt = 1e-8 s` (Courant ~0.63 < 0.707 at v_max=6300, dx=0.1 mm); record ~17 us
  (`nt = 1700`) to cover two-way traveltimes into the aluminum.
- PML/absorbing boundaries on all sides.

## Migration (pylops Kirchhoff)

- Migration grid: 0.2 mm (200 x 150), `vel_mig` downsampled by 2 (layer
  interfaces at 8/22 mm fall on grid nodes).
- Traveltimes: `scikit-fmm` has no wheel for Python 3.14 and no compiler is
  available, so traveltime tables are computed with a custom numba
  **fast-sweeping eikonal** solver (Godunov upwind, square cells) and supplied
  via `mode="byot"` as a `(trav_srcs, trav_recs)` tuple of shape
  `(nx*nz, ns)` / `(nx*nz, nr)` — matching pylops' internal convention
  (model raveled as `(nx, nz)`). The solver is validated against the analytic
  `dist / v` solution on a homogeneous model before use.
- `dynamic=False`, `engine="numba"`. Same Ricker passed as the wavelet.
- Two operators on the recorded data:
  - multishot: single `Kirchhoff(..., shot_recs=[arange(nr)]*ns)`.
  - VStack: `pylops.VStack` of 32 per-shot `Kirchhoff` operators (single source,
    all receivers, same byot traveltime columns sliced per shot).
- `madj = Op.H @ d` for each.

## Comparison, plots, saving

- Equality check: `max|madj_ms - madj_vs|` (expect machine precision).
- Saved to `outputs/deepwave_fmc/`:
  - `velocity_true.png` (model + voids + array), `fmc_shot_gather.png`,
    `migration_compare.png` (multishot | VStack | difference).
  - `results.npz`: vel_true, vel_mig, fmc data, both migration images, axes.

## Notes

- Migration images the full scattered field; strong water/aluminum interface
  reflections appear alongside void diffractions (no direct-wave mute).
- Large velocity contrast (1480<->6300) and tiny air wavelength mean voids are
  under-resolved as a propagating medium but act as reflectors, which is the
  point.
