r"""
Blind FMC imaging of a sub-wavelength void (no Born, no reference run)
======================================================================
This study simulates a *realistic* ultrasonic NDT scenario: a 32-element
5 MHz array sitting on an aluminum plate containing a **single air void
smaller than the wavelength** (0.6 mm diameter vs. the 1.26 mm aluminum
wavelength). The Full Matrix Capture (FMC) data are produced by a **single**
:mod:`deepwave` finite-difference simulation of the true medium -- there is no
Born modelling and *no void-free reference simulation to subtract*. The imaging
side only knows:

1. the recorded receiver signals,
2. the array geometry and the background velocity, and
3. traveltime + amplitude tables obtained from the pylops
   :py:class:`pylops.waveeqprocessing.Kirchhoff` operator itself
   (``Kirchhoff._traveltime_table``), handed back to the operator through
   ``mode="byot"`` as ``trav=(trav_srcs, trav_recs)`` and
   ``amp=(amp_srcs, amp_recs)`` tuples.

Because the raw total field is dominated by the direct wave along the array
and the plate back-wall echo, the focus is on **data-driven preprocessing**:

* zero-phase bandpass around the transducer band;
* **common-offset median subtraction** -- for each source-receiver offset the
  median trace over all element pairs is removed. In a laterally invariant
  plate this cancels the direct wave, the back-wall echo and their multiples,
  while the void diffraction (localised in x) survives. This is the standard
  "no reference" alternative to Born reference subtraction;
* a direct-wave top mute built only from the known velocity and geometry;
* a linear time gain compensating geometrical spreading of the data.

Outputs (PNG figures + ``results.npz``) go to ``outputs/deepwave_fmc_blind/``.
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, hilbert, sosfiltfilt

from deepwave import scalar

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

np.random.seed(0)
OUTDIR = os.path.join("outputs", "deepwave_fmc_blind")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. True medium: aluminum plate, air backing, one sub-wavelength air void
# ---------------------------------------------------------------------------
# Simulation grid in (nx, nz) layout (pylops convention); deepwave gets the
# transpose. At 5 MHz the aluminum wavelength is 1.26 mm, so the 0.05 mm grid
# gives ~25 points per wavelength.
dx = 0.5e-4  # 0.05 mm grid spacing [m]
nx_s, nz_s = 800, 600  # 40 mm x 30 mm
xs = np.arange(nx_s) * dx
zs = np.arange(nz_s) * dx

V_ALU, V_AIR = 6300.0, 343.0
FREQ = 5.0e6
LAM_ALU = V_ALU / FREQ  # 1.26 mm
Z_BACK = 0.024  # plate back wall at 24 mm depth (air below)

vel_true = np.full((nx_s, nz_s), V_ALU, dtype=np.float32)
vel_true[:, zs >= Z_BACK] = V_AIR

# single air void, diameter 0.6 mm = 0.48 aluminum wavelengths (< lambda)
void_r = 0.3e-3
void_c = (0.020, 0.013)  # x = 20 mm (array centre), z = 13 mm
Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
vel_true[(Xg - void_c[0]) ** 2 + (Zg - void_c[1]) ** 2 <= void_r**2] = V_AIR
print(f"void diameter = {2 * void_r * 1e3:.2f} mm "
      f"= {2 * void_r / LAM_ALU:.2f} aluminum wavelengths at {FREQ / 1e6:.0f} MHz")

# ---------------------------------------------------------------------------
# 2. FMC acquisition geometry (32 elements, 1 mm pitch, inside the aluminum)
# ---------------------------------------------------------------------------
n_el = 32
pitch_cells = 20  # 1 mm pitch
x0_cells = 90  # first element at x = 4.5 mm
z_cells = 40  # element depth = 2 mm
elem_ix = x0_cells + np.arange(n_el) * pitch_cells
elem_iz = np.full(n_el, z_cells)
elem_x = elem_ix * dx
elem_z = elem_iz * dx

# ---------------------------------------------------------------------------
# 3. ONE deepwave simulation of the true medium (the only data we get)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"deepwave device: {device}")

dt = 5.0e-9  # 5 ns -> Courant 0.63 at 6300 m/s
nt = 2200  # 11 us record
peak_time = 1.5 / FREQ

import deepwave  # noqa: E402  (wavelet helper)

src_amp = (
    deepwave.wavelets.ricker(FREQ, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_el, 1, 1)
    .to(device)
)
src_loc = torch.tensor(
    np.stack([elem_iz, elem_ix], axis=-1)[:, None, :], dtype=torch.long, device=device
)
rec_loc = torch.tensor(
    np.broadcast_to(np.stack([elem_iz, elem_ix], axis=-1), (n_el, n_el, 2)).copy(),
    dtype=torch.long,
    device=device,
)

t0 = time.perf_counter()
out = scalar(
    torch.tensor(vel_true.T.copy(), device=device),
    dx,
    dt,
    source_amplitudes=src_amp,
    source_locations=src_loc,
    receiver_locations=rec_loc,
    accuracy=4,
    pml_width=20,
    pml_freq=FREQ,
)
fmc_raw = out[-1].cpu().numpy().astype(np.float64)  # (n_src, n_rec, nt)
print(f"deepwave FMC sim: {time.perf_counter() - t0:.1f} s, data {fmc_raw.shape}")

t = np.arange(nt) * dt

# ---------------------------------------------------------------------------
# 4. Blind preprocessing (uses only the data + geometry/velocity knowledge)
# ---------------------------------------------------------------------------
# (a) time-zero correction: the excitation pulse peaks ``peak_time`` after the
# electrical time zero (always known in a real system), so advance the data to
# make the recorded traveltimes match the impulse-response times the Kirchhoff
# operator works with.
n0 = int(round(peak_time / dt))
fmc_t0 = np.concatenate(
    [fmc_raw[..., n0:], np.zeros((n_el, n_el, n0))], axis=-1
)

# (b) zero-phase bandpass around the 5 MHz transducer band
sos = butter(4, [2.5e6, 7.5e6], btype="bandpass", fs=1.0 / dt, output="sos")
fmc_bp = sosfiltfilt(sos, fmc_t0, axis=-1)

# (c) common-offset median subtraction: at fixed element offset k = j - i the
# direct wave, back-wall echo and their multiples are identical for every i
# (laterally invariant plate), so the median trace over i estimates exactly
# those events and none of the localised void diffraction. Offsets with too few
# pairs for a robust median are zeroed (a tiny aperture loss) so their
# unsubtracted back-wall echo cannot leak artefacts into the image.
fmc_cos = fmc_bp.copy()
MIN_PAIRS = 4
for k in range(-(n_el - 1), n_el):
    ii = np.arange(max(0, -k), min(n_el, n_el - k))
    if len(ii) < MIN_PAIRS:
        fmc_cos[ii, ii + k, :] = 0.0
        continue
    med = np.median(fmc_bp[ii, ii + k, :], axis=0)
    fmc_cos[ii, ii + k, :] -= med

# (d) time gates from geometry + velocity only, with cosine ramps: a top mute
# just after the direct wave along the array, and a bottom gate just before the
# back-wall echo (the NDT "inspection gate" -- the plate thickness is known, and
# everything at/after the back-wall arrival is wall echo and multiples).
T_PAD = 0.9e-6
T_RAMP = 0.3e-6
h_bw = 2.0 * (Z_BACK - elem_z[0])  # two-way vertical path to the back wall
mute = np.ones((n_el, n_el, nt))
for i in range(n_el):
    offs = np.abs(elem_x - elem_x[i])
    t_dir = offs / V_ALU + T_PAD
    t_bw = np.sqrt(h_bw**2 + offs**2) / V_ALU - 0.4e-6
    for j in range(n_el):
        up = np.clip((t - t_dir[j]) / T_RAMP, 0.0, 1.0)
        dn = np.clip((t_bw[j] - t) / T_RAMP, 0.0, 1.0)
        mute[i, j] = (0.5 - 0.5 * np.cos(np.pi * up)) * (0.5 - 0.5 * np.cos(np.pi * dn))
tgain = t / t[-1]
fmc_pre = fmc_cos * mute * tgain

# intermediate product for the figure: mute+gain only (no median subtraction)
fmc_mut = fmc_bp * mute * tgain

# ---------------------------------------------------------------------------
# 5. Traveltime and amplitude tables FROM THE KIRCHHOFF OPERATOR (byot)
# ---------------------------------------------------------------------------
dxm = 1.0e-4  # 0.1 mm migration grid
nxm, nzm = nx_s // 2, nz_s // 2
xm = np.arange(nxm) * dxm
zm = np.arange(nzm) * dxm

srcs = np.vstack((elem_x, elem_z))
recs = np.vstack((elem_x, elem_z))
ns, nr = srcs.shape[1], recs.shape[1]

trav_srcs, trav_recs, dist_srcs, dist_recs, _, _ = Kirchhoff._traveltime_table(
    zm, xm, srcs, recs, V_ALU, mode="analytic"
)
eps = 1e-2 * (dist_srcs.max() + dist_recs.max())
amp_srcs = 1.0 / np.sqrt(dist_srcs + eps)
amp_recs = 1.0 / np.sqrt(dist_recs + eps)
print(f"byot tables: trav {trav_srcs.shape}, amp {amp_srcs.shape}")

wav, _, wavc = ricker(t[:121], f0=FREQ)
shot_recs = [np.arange(nr) for _ in range(ns)]
Op = Kirchhoff(
    zm, xm, t, srcs, recs, V_ALU, wav, wavc,
    mode="byot",
    trav=(trav_srcs, trav_recs),
    amp=(amp_srcs, amp_recs),
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)


def migrate(data):
    return (Op.H @ data.ravel()).reshape(nxm, nzm)


t0 = time.perf_counter()
m_raw = migrate(fmc_bp)  # bandpassed total field (baseline)
m_mut = migrate(fmc_mut)  # mute + gain, but no median subtraction
m_pre = migrate(fmc_pre)  # full blind pipeline
print(f"3 migrations: {time.perf_counter() - t0:.1f} s")

# envelope image (analytic signal along z) of the full pipeline result
m_env = np.abs(hilbert(m_pre, axis=1))

# ---------------------------------------------------------------------------
# 6. Quantitative clarity: localisation error and image contrast
# ---------------------------------------------------------------------------
# evaluate inside the plate, away from the array dead zone and the back wall
zone = (slice(None), slice(int(0.005 / dxm), int(0.022 / dxm)))
env_zone = m_env[zone]
pk = np.unravel_index(np.argmax(env_zone), env_zone.shape)
pk_x = xm[pk[0]]
pk_z = zm[pk[1] + zone[1].start]
loc_err = np.hypot(pk_x - void_c[0], pk_z - void_c[1])

# contrast: peak vs RMS outside a 2-wavelength disc around the true void
Xm, Zm = np.meshgrid(xm, zm, indexing="ij")
far = ((Xm - void_c[0]) ** 2 + (Zm - void_c[1]) ** 2 > (2 * LAM_ALU) ** 2)
bg = m_env[zone][far[zone]]
contrast = env_zone.max() / np.sqrt((bg**2).mean())

print("\nBlind-imaging quality (full pipeline, envelope image)")
print(f"  true void centre   : x = {void_c[0] * 1e3:.2f} mm, z = {void_c[1] * 1e3:.2f} mm")
print(f"  image peak         : x = {pk_x * 1e3:.2f} mm, z = {pk_z * 1e3:.2f} mm")
print(f"  localisation error : {loc_err * 1e3:.2f} mm "
      f"({loc_err / LAM_ALU:.2f} wavelengths)")
print(f"  peak/background    : {contrast:.1f}x")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
ext_mm = (xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3)
ext_mig = (xm[0] * 1e3, xm[-1] * 1e3, zm[-1] * 1e3, zm[0] * 1e3)


def mark(a, label=False):
    a.scatter(elem_x * 1e3, elem_z * 1e3, marker="v", s=12, c="r", edgecolors="k",
              label="array" if label else None)
    a.scatter(void_c[0] * 1e3, void_c[1] * 1e3, marker="o", s=70,
              facecolors="none", edgecolors="lime", linewidths=1.4,
              label="true void" if label else None)
    a.set_xlabel("x [mm]"), a.set_ylabel("z [mm]")


# (a) true model
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(vel_true.T, cmap="viridis", extent=ext_mm, aspect="equal")
plt.colorbar(im, ax=ax, label="velocity [m/s]")
mark(ax, label=True)
ax.set_title(f"True model: aluminum plate, air backing, "
             f"{2 * void_r / LAM_ALU:.2f}$\\lambda$ air void")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "velocity_true.png"), dpi=130)

# (b) preprocessing stages on the central shot gather
ishot = ns // 2
stages = [
    (fmc_bp[ishot], "(1) bandpassed total field"),
    (fmc_cos[ishot], "(2) + common-offset median subtraction"),
    (fmc_pre[ishot], "(3) + direct-wave mute + t gain"),
]
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for a, (g, title) in zip(axs, stages):
    gmax = np.percentile(np.abs(g), 99.5) + 1e-30
    a.imshow(g.T, cmap="gray", vmin=-gmax, vmax=gmax,
             extent=(0, nr, t[-1] * 1e6, t[0] * 1e6), aspect="auto")
    a.set_title(title, fontsize=10)
    a.set_xlabel("receiver index")
axs[0].set_ylabel("t [us]")
fig.suptitle(f"Blind preprocessing, shot #{ishot} (element above the void)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "preprocessing_stages.png"), dpi=130)

# (c) pulse-echo A-scan before/after preprocessing with expected event times
t_us = t * 1e6
ev = {
    "void": (2 * (void_c[1] - elem_z[0]) / V_ALU) * 1e6,
    "back wall": (2 * (Z_BACK - elem_z[0]) / V_ALU) * 1e6,
}
fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for a, tr, title in zip(
    axs,
    [fmc_bp[ishot, ishot], fmc_pre[ishot, ishot]],
    ["raw (bandpassed) pulse-echo trace", "after blind preprocessing"],
):
    a.plot(t_us, tr, "k", lw=0.8)
    for name, te in ev.items():
        a.axvline(te, color="C3", ls="--", lw=0.9)
        a.text(te, 0.95, f" {name}", color="C3", fontsize=9, rotation=90,
               va="top", transform=a.get_xaxis_transform())
    a.set_title(title, fontsize=10)
    a.set_ylabel("amplitude")
axs[1].set_xlabel("t [us]")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "ascan_before_after.png"), dpi=130)

# (d) migration images: raw vs mute-only vs full pipeline (+ envelope)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
panels = [
    (m_raw, "(1) migrate raw total field", "gray_r"),
    (m_mut, "(2) mute + gain only", "gray_r"),
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
fig.suptitle("Kirchhoff migration (byot trav+amp from the operator) of "
             "single-simulation FMC data -- no Born, no reference subtraction")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "migration_blind.png"), dpi=130)

# (e) zoom on the void in the envelope image
fig, ax = plt.subplots(figsize=(6.5, 5.5))
zoom_mm = 4.0
im = ax.imshow(m_env.T, cmap="magma", extent=ext_mig, aspect="equal",
               vmin=0, vmax=np.percentile(m_env, 99.95))
mark(ax, label=True)
ax.set_xlim(void_c[0] * 1e3 - zoom_mm, void_c[0] * 1e3 + zoom_mm)
ax.set_ylim(void_c[1] * 1e3 + zoom_mm, void_c[1] * 1e3 - zoom_mm)
plt.colorbar(im, ax=ax, label="envelope")
ax.set_title(f"Envelope zoom: localisation error "
             f"{loc_err * 1e3:.2f} mm ({loc_err / LAM_ALU:.2f}$\\lambda$)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "void_zoom.png"), dpi=130)

np.savez_compressed(
    os.path.join(OUTDIR, "results.npz"),
    vel_true=vel_true,
    fmc_raw=fmc_raw.astype(np.float32),
    fmc_pre=fmc_pre.astype(np.float32),
    m_raw=m_raw, m_mut=m_mut, m_pre=m_pre, m_env=m_env,
    xm=xm, zm=zm, elem_x=elem_x, elem_z=elem_z,
    void_center=np.array(void_c), void_r=void_r, t=t,
    peak_xz=np.array([pk_x, pk_z]), loc_err=loc_err, contrast=contrast,
)
print(f"\nSaved figures and results.npz to {OUTDIR}/")
