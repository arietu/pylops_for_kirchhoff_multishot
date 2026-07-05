r"""
Blind immersion FMC: five sub-wavelength voids in aluminum under water layers
=============================================================================
The toughest configuration so far: the original immersion geometry -- a water
layer on top (with the 32-element 5 MHz array in it), an aluminum slab with
**five randomly placed sub-wavelength air voids**, and water below -- imaged
with the **blind pipeline** (single full-wave simulation, no Born modelling,
no void-free reference run).

Compared with the original three-layer studies the aluminum slab is made
thicker (20 mm instead of 14 mm, slab from 6 to 26 mm depth): the first
in-slab reverberation then arrives well after the back-wall echo and the
inspection window between front wall and back wall is wide enough for the
void diffractions of the whole slab.

Imaging-side knowledge is unchanged: the receiver signals, the geometry and
layer velocities, and traveltime + amplitude tables. The slab makes the
medium laterally invariant but *vertically* layered, so:

* traveltimes come from the shared numba fast-sweeping eikonal solver
  (:func:`fmc_blind_utils.eikonal_trav_table`; ``scikit-fmm`` has no
  Python 3.14 wheel) on the void-free layered model, validated first against
  the analytic homogeneous solution
  (:func:`fmc_blind_utils.eikonal_self_test`) and passed to
  :py:class:`pylops.waveeqprocessing.Kirchhoff` via ``mode="byot"``. Because
  the model is laterally invariant, one padded eikonal solve is shifted to
  all 32 elements instead of solving 32 times;
* amplitudes still come **from the Kirchhoff operator itself**: its Euclidean
  distance tables (a purely geometric quantity, valid in any medium) feed the
  1/sqrt(dist) spreading weights
  (:func:`fmc_blind_utils.operator_distance_amplitudes`);
* the front-wall / back-wall **inspection gates are derived from the
  traveltime tables**: the wall reflection time for an element pair (i, j) is
  ``min over x of [T_i(x, z_wall) + T_j(x, z_wall)]``
  (:func:`fmc_blind_utils.pair_wall_times`).

The rest of the blind preprocessing is the shared chain from
:mod:`fmc_blind_utils`: time-zero correction, zero-phase bandpass,
common-offset median subtraction (now also cancelling the front-wall echo and
every water/slab reverberation, all laterally invariant), and a linear time
gain.

Outputs go to ``outputs/deepwave_fmc_blind_immersion/``.
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

from fmc_blind_utils import (
    bandpass,
    common_offset_median_subtract,
    cosine_gate_cube,
    eikonal_self_test,
    eikonal_trav_table,
    make_random_voids,
    carve_voids,
    operator_distance_amplitudes,
    pair_wall_times,
    simulate_fmc,
    time_zero_correct,
    T_BW_MARGIN,
)

OUTDIR = os.path.join("outputs", "deepwave_fmc_blind_immersion")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. True medium: water / thick aluminum slab with five voids / water
# ---------------------------------------------------------------------------
dx = 0.5e-4  # 0.05 mm grid spacing (~6 ppw in water at 5 MHz)
nx_s, nz_s = 800, 600  # 40 mm x 30 mm
xs = np.arange(nx_s) * dx
zs = np.arange(nz_s) * dx

V_WATER, V_ALU, V_AIR = 1480.0, 6300.0, 343.0
FREQ = 5.0e6
LAM_ALU = V_ALU / FREQ  # 1.26 mm
Z_TOP, Z_BACK = 0.006, 0.026  # 20 mm slab (thicker than the original 14 mm)

vel_layers = np.full((nx_s, nz_s), V_WATER, dtype=np.float32)
vel_layers[:, (zs >= Z_TOP) & (zs < Z_BACK)] = V_ALU

# five random voids inside the slab, clear of both walls
voids = make_random_voids(
    seed=7, n_voids=5, min_sep=3.5e-3,
    x_range=(0.008, 0.032), z_range=(0.0095, 0.0225),
    d_range=(0.3e-3, 0.8e-3),
)
N_VOIDS = len(voids)

vel_true = vel_layers.copy()
carve_voids(vel_true, xs, zs, voids, V_AIR)
print("true voids (sorted by depth):")
for k, (cx, cz, r) in enumerate(voids):
    print(f"  #{k}: x = {cx * 1e3:6.2f} mm, z = {cz * 1e3:6.2f} mm, "
          f"d = {2 * r * 1e3:.2f} mm = {2 * r / LAM_ALU:.2f} lambda")

# ---------------------------------------------------------------------------
# 2. FMC array in the upper water layer + ONE deepwave simulation
# ---------------------------------------------------------------------------
n_el = 32
elem_ix = 90 + np.arange(n_el) * 20  # 1 mm pitch from x = 4.5 mm
elem_iz = np.full(n_el, 40)  # z = 2 mm, in the water
elem_x = elem_ix * dx
elem_z = elem_iz * dx

dt = 5.0e-9
nt = 2800  # 14 us: back-wall echo at ~11.8 us + margin
peak_time = 1.5 / FREQ

t0 = time.perf_counter()
fmc = simulate_fmc(vel_true, dx, dt, nt, elem_ix, elem_iz, FREQ, peak_time)
print(f"deepwave FMC sim: {time.perf_counter() - t0:.1f} s, data {fmc.shape}")
t = np.arange(nt) * dt

# ---------------------------------------------------------------------------
# 3. Layered-model traveltimes: shared fast-sweeping eikonal (byot tables)
# ---------------------------------------------------------------------------
dxm = 1.0e-4  # 0.1 mm migration grid
vel_mig = vel_layers[::2, ::2].astype(np.float64)  # layered, NO voids
nxm, nzm = vel_mig.shape
xm = np.arange(nxm) * dxm
zm = np.arange(nzm) * dxm

rel = eikonal_self_test()  # halt here if the solver is silently broken
print(f"eikonal self-test vs analytic homogeneous solution: "
      f"max rel err {rel:.4f}")

six = np.round(elem_x / dxm).astype(np.int64)
siz = np.round(elem_z / dxm).astype(np.int64)
t0 = time.perf_counter()
trav = eikonal_trav_table(vel_mig, dxm, six, siz)  # (nxm*nzm, n_el)
print(f"eikonal traveltime tables: {time.perf_counter() - t0:.1f} s")

# ---------------------------------------------------------------------------
# 4. Amplitude tables FROM THE KIRCHHOFF OPERATOR (geometric distances)
# ---------------------------------------------------------------------------
srcs = np.vstack((elem_x, elem_z))
recs = np.vstack((elem_x, elem_z))
ns, nr = srcs.shape[1], recs.shape[1]

amp_srcs, amp_recs = operator_distance_amplitudes(zm, xm, srcs, recs)

# ---------------------------------------------------------------------------
# 5. Blind preprocessing (gates derived from the traveltime tables)
# ---------------------------------------------------------------------------
fmc_t0 = time_zero_correct(fmc, peak_time, dt)
fmc_bp = bandpass(fmc_t0, dt)

# common-offset median subtraction: in this laterally invariant medium it
# cancels the water direct wave, the front-wall echo, the back-wall echo and
# every water/slab reverberation; the localised void diffractions survive
fmc_cos = common_offset_median_subtract(fmc_bp)

# inspection gates from the eikonal tables: wall reflection time for the
# pair (i, j) = min over x of [T_i(x, z_wall) + T_j(x, z_wall)]
iz_fw = int(round(Z_TOP / dxm))
iz_bw = int(round(Z_BACK / dxm)) - 1  # just above the back wall
T3 = trav.reshape(nxm, nzm, n_el)
t_fw_pair = pair_wall_times(T3, iz_fw)
t_bw_pair = pair_wall_times(T3, iz_bw)

T_PAD = 0.7e-6  # the compact water front-wall echo needs less clearance
T_RAMP = 0.3e-6
mute = cosine_gate_cube(t, t_fw_pair + T_PAD, t_bw_pair - T_BW_MARGIN, T_RAMP)
fmc_pre = fmc_cos * mute * (t / t[-1])
fmc_mut = fmc_bp * mute * (t / t[-1])  # gates+gain only, for comparison

# ---------------------------------------------------------------------------
# 6. Kirchhoff operator (byot: eikonal trav + operator-generated amp) + migrate
# ---------------------------------------------------------------------------
wav, _, wavc = ricker(t[:121], f0=FREQ)
Op = Kirchhoff(
    zm, xm, t, srcs, recs, vel_mig, wav, wavc,
    mode="byot",
    trav=(trav, trav),
    amp=(amp_srcs, amp_recs),
    dynamic=False,
    engine="numba",
)


def migrate(data):
    return (Op.H @ data.ravel()).reshape(nxm, nzm)


t0 = time.perf_counter()
m_raw = migrate(fmc_bp)
m_mut = migrate(fmc_mut)
m_pre = migrate(fmc_pre)
print(f"3 migrations: {time.perf_counter() - t0:.1f} s")
m_env = np.abs(hilbert(m_pre, axis=1))

# ---------------------------------------------------------------------------
# 7. Per-void localisation and contrast (inside the slab)
# ---------------------------------------------------------------------------
Xm, Zm = np.meshgrid(xm, zm, indexing="ij")
zone_mask = (Zm >= Z_TOP + 0.0015) & (Zm <= Z_BACK - 0.0015)
far_mask = zone_mask.copy()
for cx, cz, _ in voids:
    far_mask &= (Xm - cx) ** 2 + (Zm - cz) ** 2 > (2 * LAM_ALU) ** 2
bg_rms = np.sqrt((m_env[far_mask] ** 2).mean())

SEARCH_R = 1.5e-3
results = []
print("\nPer-void blind-imaging quality (envelope image, immersion case)")
print(f"  background RMS (slab zone) = {bg_rms:.3e}")
for k, (cx, cz, r) in enumerate(voids):
    near = (Xm - cx) ** 2 + (Zm - cz) ** 2 <= SEARCH_R**2
    vals = np.where(near, m_env, 0.0)
    pk = np.unravel_index(np.argmax(vals), vals.shape)
    pk_x, pk_z, pk_v = xm[pk[0]], zm[pk[1]], vals[pk]
    err = np.hypot(pk_x - cx, pk_z - cz)
    contrast = pk_v / bg_rms
    results.append((pk_x, pk_z, err, contrast))
    print(f"  void #{k} (d={2 * r * 1e3:.2f} mm, {2 * r / LAM_ALU:.2f} lam) "
          f"at ({cx * 1e3:5.2f}, {cz * 1e3:5.2f}) mm -> peak "
          f"({pk_x * 1e3:5.2f}, {pk_z * 1e3:5.2f}) mm, "
          f"err = {err * 1e3:.2f} mm ({err / LAM_ALU:.2f} lam), "
          f"contrast = {contrast:.1f}x")
errs = np.array([r[2] for r in results])
cons = np.array([r[3] for r in results])
print(f"  mean localisation error = {errs.mean() * 1e3:.2f} mm "
      f"({errs.mean() / LAM_ALU:.2f} lambda), max = {errs.max() * 1e3:.2f} mm")

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
ext_mm = (xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3)
ext_mig = (xm[0] * 1e3, xm[-1] * 1e3, zm[-1] * 1e3, zm[0] * 1e3)


def mark(a, label=False):
    a.scatter(elem_x * 1e3, elem_z * 1e3, marker="v", s=12, c="r", edgecolors="k",
              label="array" if label else None)
    for k, (cx, cz, r) in enumerate(voids):
        a.scatter(cx * 1e3, cz * 1e3, marker="o", s=70, facecolors="none",
                  edgecolors="lime", linewidths=1.4,
                  label="true voids" if (label and k == 0) else None)
    for zw in (Z_TOP, Z_BACK):
        a.axhline(zw * 1e3, color="w", lw=0.6, ls=":")
    a.set_xlabel("x [mm]"), a.set_ylabel("z [mm]")


# (a) true model
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(vel_true.T, cmap="viridis", extent=ext_mm, aspect="equal")
plt.colorbar(im, ax=ax, label="velocity [m/s]")
mark(ax, label=True)
ax.set_title("Immersion setup: water / 20 mm aluminum slab with five "
             "sub-$\\lambda$ voids / water")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "velocity_true.png"), dpi=130)

# (b) preprocessing stages on the central shot gather
ishot = ns // 2
stages = [
    (fmc_bp[ishot], "(1) bandpassed total field"),
    (fmc_cos[ishot], "(2) + common-offset median subtraction"),
    (fmc_pre[ishot], "(3) + wall gates + t gain"),
]
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for a, (g, title) in zip(axs, stages):
    gmax = np.percentile(np.abs(g), 99.5) + 1e-30
    a.imshow(g.T, cmap="gray", vmin=-gmax, vmax=gmax,
             extent=(0, nr, t[-1] * 1e6, t[0] * 1e6), aspect="auto")
    a.set_title(title, fontsize=10)
    a.set_xlabel("receiver index")
axs[0].set_ylabel("t [us]")
fig.suptitle(f"Blind preprocessing in immersion, shot #{ishot}")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "preprocessing_stages.png"), dpi=130)

# (b2) representative A-scans: pulse-echo of the element nearest each void,
# raw (bandpassed) vs after the blind preprocessing, with the eikonal-predicted
# event times (front wall, void, back wall) marked
t_us = t * 1e6
fig, axs = plt.subplots(N_VOIDS, 2, figsize=(13, 2.1 * N_VOIDS),
                        sharex=True, sharey=False)
for k, (cx, cz, r) in enumerate(voids):
    ie = int(np.argmin(np.abs(elem_x - cx)))  # element nearest the void
    ixm = int(round(cx / dxm))
    izm = int(round(cz / dxm))
    t_void = 2.0 * T3[ixm, izm, ie] * 1e6
    evs = {"front wall": t_fw_pair[ie, ie] * 1e6,
           f"void #{k}": t_void,
           "back wall": t_bw_pair[ie, ie] * 1e6}
    for col, (tr, title) in enumerate(
        [(fmc_bp[ie, ie], "raw (bandpassed)"),
         (fmc_pre[ie, ie], "after blind preprocessing")]
    ):
        a = axs[k, col]
        a.plot(t_us, tr, "k", lw=0.7)
        for name, te in evs.items():
            c = "C3" if "void" in name else "C0"
            a.axvline(te, color=c, ls="--", lw=0.9)
            if k == 0 or "void" in name:
                a.text(te, 0.95, f" {name}", color=c, fontsize=8, rotation=90,
                       va="top", transform=a.get_xaxis_transform())
        a.set_ylabel(f"el #{ie}", fontsize=8)
        if k == 0:
            a.set_title(f"pulse-echo A-scan, {title}", fontsize=10)
for a in axs[-1]:
    a.set_xlabel("t [us]")
fig.suptitle("Immersion case: pulse-echo signals at the element above each "
             "void (red dashes = predicted void echo)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "ascans_per_void.png"), dpi=130)

# (c) migration images
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
panels = [
    (m_raw, "(1) migrate raw total field", "gray_r"),
    (m_mut, "(2) wall gates + gain only", "gray_r"),
    (m_pre, "(3) full blind pipeline", "gray_r"),
    (m_env, "(3) envelope", "magma"),
]
for a, (img, title, cmap) in zip(axs, panels):
    mmax = np.percentile(np.abs(img), 99.8) + 1e-30
    vmin = 0.0 if cmap == "magma" else -mmax
    a.imshow(img.T, cmap=cmap, extent=ext_mig, aspect="equal",
             vmin=vmin, vmax=mmax)
    a.set_title(title, fontsize=10)
    mark(a)
fig.suptitle("Immersion blind imaging: Kirchhoff migration (byot eikonal trav "
             "+ operator amp) -- no Born, no reference subtraction")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "migration_blind.png"), dpi=130)

# (d) per-void zooms on the envelope image
fig, axs = plt.subplots(1, N_VOIDS, figsize=(4 * N_VOIDS, 4.4))
vmax = np.percentile(m_env, 99.95)
zoom_mm = 2.5
for k, (a, (cx, cz, r)) in enumerate(zip(axs, voids)):
    a.imshow(m_env.T, cmap="magma", extent=ext_mig, aspect="equal",
             vmin=0, vmax=vmax)
    a.scatter(cx * 1e3, cz * 1e3, marker="o", s=80, facecolors="none",
              edgecolors="lime", linewidths=1.5)
    pk_x, pk_z, err, contrast = results[k]
    a.scatter(pk_x * 1e3, pk_z * 1e3, marker="+", s=90, c="cyan", linewidths=1.5)
    a.set_xlim(cx * 1e3 - zoom_mm, cx * 1e3 + zoom_mm)
    a.set_ylim(cz * 1e3 + zoom_mm, cz * 1e3 - zoom_mm)
    a.set_title(f"void #{k}: d = {2 * r * 1e3:.2f} mm\n"
                f"err {err * 1e3:.2f} mm, {contrast:.0f}x", fontsize=10)
    a.set_xlabel("x [mm]")
axs[0].set_ylabel("z [mm]")
fig.suptitle("Immersion case, envelope zooms (green = true void, cyan + = peak)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "void_zooms.png"), dpi=130)

np.savez_compressed(
    os.path.join(OUTDIR, "results.npz"),
    vel_true=vel_true, vel_mig=vel_mig,
    fmc=fmc.astype(np.float32), fmc_pre=fmc_pre.astype(np.float32),
    m_raw=m_raw, m_mut=m_mut, m_pre=m_pre, m_env=m_env,
    xm=xm, zm=zm, elem_x=elem_x, elem_z=elem_z,
    voids=np.array(voids), t=t,
    peaks=np.array(results), bg_rms=bg_rms,
)
print(f"\nSaved figures and results.npz to {OUTDIR}/")
