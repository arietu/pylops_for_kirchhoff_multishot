r"""
deepwave FMC simulation + Kirchhoff migration (multi-shot vs VStack)
====================================================================
This script simulates a **Full Matrix Capture (FMC)** ultrasonic acquisition
over a three-layer medium (water / aluminum / water) containing randomly placed
**air voids**, using the :mod:`deepwave` finite-difference wave solver. The
recorded FMC data are then migrated with the pylops
:py:class:`pylops.waveeqprocessing.Kirchhoff` operator built two different ways
-- the built-in **multi-shot** interface and a :py:class:`pylops.VStack` of
per-shot operators -- and the two migration images are compared.

The migration velocity is the *same* three-layer model used by deepwave **but
without the air voids** (the voids are the scatterers we want to image; the
background velocity should not know about them).

Because Python 3.14 has no ``scikit-fmm`` wheel and no C++ compiler is available
here, the layered-model traveltime tables are computed with a small **numba
fast-sweeping eikonal solver** and handed to ``Kirchhoff`` through
``mode="byot"`` (bring-your-own-traveltimes).

Outputs (PNG figures + a ``.npz`` of arrays) are written to
``outputs/deepwave_fmc/``.
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import njit, prange

import deepwave
from deepwave import scalar

import pylops
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

np.random.seed(0)
OUTDIR = os.path.join("outputs", "deepwave_fmc")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Build the three-layer model with random air voids (ultrasonic NDT scale)
# ---------------------------------------------------------------------------
# Simulation grid (fine). Arrays are stored in (nx, nz) layout to match the
# pylops Kirchhoff convention; deepwave needs (nz, nx) so we transpose later.
# At 5 MHz the water wavelength is ~0.30 mm, so we use a 0.05 mm grid (~6 points
# per wavelength in water) to keep numerical dispersion under control.
dx = 0.5e-4  # 0.05 mm simulation grid spacing [m]
nx_s, nz_s = 800, 600  # 40 mm x 30 mm
xs = np.arange(nx_s) * dx
zs = np.arange(nz_s) * dx

V_WATER, V_ALU, V_AIR = 1480.0, 6300.0, 343.0
Z_TOP, Z_BOT = 0.008, 0.022  # aluminum slab between 8 mm and 22 mm depth

# layered background (no voids) in (nx, nz)
vel_layers = np.full((nx_s, nz_s), V_WATER, dtype=np.float32)
alu_mask = (zs >= Z_TOP) & (zs < Z_BOT)
vel_layers[:, alu_mask] = V_ALU

# random air voids inside the aluminum layer only.
# Voids are ~1 aluminum wavelength across at 5 MHz (alu lambda ~ 1.26 mm), i.e.
# small (sub-wavelength-ish) point-like scatterers.
n_voids = 6
void_r = 0.63e-3  # ~0.63 mm radius -> ~1.26 mm diameter ~ 1 alu wavelength
vel_true = vel_layers.copy()
Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
centers = []
rng = np.random.default_rng(0)
while len(centers) < n_voids:
    cx = rng.uniform(0.006, 0.034)  # keep voids within the array footprint
    cz = rng.uniform(Z_TOP + void_r, Z_BOT - void_r)
    if all(np.hypot(cx - px, cz - pz) > 3 * void_r for px, pz in centers):
        centers.append((cx, cz))
        vel_true[(Xg - cx) ** 2 + (Zg - cz) ** 2 <= void_r**2] = V_AIR

# ---------------------------------------------------------------------------
# 2. FMC acquisition geometry (32-element array in the top water layer)
# ---------------------------------------------------------------------------
n_el = 32
pitch_cells = 20  # 1 mm pitch
x0_cells = 90  # first element at x = 4.5 mm
z_cells = 40  # array depth = 2 mm (top water)
elem_ix = x0_cells + np.arange(n_el) * pitch_cells  # x grid index (sim)
elem_iz = np.full(n_el, z_cells)
elem_x = elem_ix * dx  # physical coords [m]
elem_z = elem_iz * dx

# ---------------------------------------------------------------------------
# 3. deepwave forward FMC simulation (acoustic scalar, GPU)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"deepwave device: {device}")

dt = 5.0e-9  # 5 ns -> Courant ~0.63 at v_max, dx=0.05 mm
nt = 2800  # ~14 us record
freq = 5.0e6  # 5 MHz Ricker
peak_time = 1.5 / freq

v_dw = torch.tensor(vel_true.T.copy(), device=device)  # (nz, nx) for deepwave

# FMC: shot i fires at element i, all elements record
src_amp = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_el, 1, 1)
    .to(device)
)  # (n_shots, 1, nt)
# locations as [depth_idx, x_idx]
src_loc = torch.tensor(
    np.stack([elem_iz, elem_ix], axis=-1)[:, None, :], dtype=torch.long, device=device
)  # (n_shots, 1, 2)
rec_loc = torch.tensor(
    np.broadcast_to(np.stack([elem_iz, elem_ix], axis=-1), (n_el, n_el, 2)).copy(),
    dtype=torch.long,
    device=device,
)  # (n_shots, n_rec, 2)

t0 = time.perf_counter()
out = scalar(
    v_dw,
    dx,
    dt,
    source_amplitudes=src_amp,
    source_locations=src_loc,
    receiver_locations=rec_loc,
    accuracy=4,
    pml_width=20,
    pml_freq=freq,
)
fmc = out[-1].cpu().numpy().astype(np.float64)  # (n_shots, n_rec, nt)
print(f"deepwave FMC sim: {time.perf_counter() - t0:.1f} s, data {fmc.shape}")

# Reference (void-free) simulation through the *background* model. Subtracting it
# from the with-voids data isolates the void-scattered field -- exactly what the
# Born/Kirchhoff migration operator assumes as its data (see section 6).
v_ref = torch.tensor(vel_layers.T.copy(), device=device)  # (nz, nx), no voids
out_ref = scalar(
    v_ref, dx, dt, source_amplitudes=src_amp, source_locations=src_loc,
    receiver_locations=rec_loc, accuracy=4, pml_width=20, pml_freq=freq,
)
fmc_ref = out_ref[-1].cpu().numpy().astype(np.float64)
fmc_scat = fmc - fmc_ref  # pure void-scattered field
print(f"scattered-field / total-field rms = "
      f"{np.sqrt((fmc_scat**2).mean()) / np.sqrt((fmc**2).mean()):.3e}")

# ---------------------------------------------------------------------------
# 4. Fast-sweeping eikonal solver (numba) for the migration traveltime tables
# ---------------------------------------------------------------------------
@njit(cache=True)
def _eikonal_fsm(slow, h, six, siz, n_sweeps=8):
    """First-arrival traveltimes on a square grid via fast sweeping (Godunov)."""
    nx, nz = slow.shape
    BIG = 1.0e30
    T = np.full((nx, nz), BIG)
    T[six, siz] = 0.0
    for _ in range(n_sweeps):
        for di in range(2):
            for dj in range(2):
                i_range = range(nx) if di == 0 else range(nx - 1, -1, -1)
                for i in i_range:
                    j_range = range(nz) if dj == 0 else range(nz - 1, -1, -1)
                    for j in j_range:
                        if i == six and j == siz:
                            continue
                        ux = T[i - 1, j] if i > 0 else BIG
                        if i < nx - 1 and T[i + 1, j] < ux:
                            ux = T[i + 1, j]
                        uz = T[i, j - 1] if j > 0 else BIG
                        if j < nz - 1 and T[i, j + 1] < uz:
                            uz = T[i, j + 1]
                        f = slow[i, j] * h
                        if abs(ux - uz) >= f:
                            cand = min(ux, uz) + f
                        else:
                            cand = 0.5 * (ux + uz + np.sqrt(2.0 * f * f - (ux - uz) ** 2))
                        if cand < T[i, j]:
                            T[i, j] = cand
    return T


@njit(parallel=True, cache=True)
def _trav_table(slow, h, ix, iz):
    """Traveltime table (nx*nz, n_pts), raveled in (nx, nz) C-order."""
    nx, nz = slow.shape
    npts = ix.shape[0]
    out = np.empty((nx * nz, npts))
    for k in prange(npts):
        T = _eikonal_fsm(slow, h, ix[k], iz[k])
        out[:, k] = T.ravel()
    return out


# self-test against the analytic homogeneous solution (T = dist / v)
_h = 0.0002
_slow = np.full((60, 60), 1.0 / 1500.0)
_T = _eikonal_fsm(_slow, _h, 0, 0)
_ii, _jj = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
_Tana = np.hypot(_ii * _h, _jj * _h) / 1500.0
_rel = np.abs(_T - _Tana)[1:, 1:].max() / _Tana[1:, 1:].max()
print(f"eikonal FSM self-test: max rel error vs analytic = {_rel:.2%}")
assert _rel < 0.05, "fast-sweeping eikonal solver failed homogeneous sanity check"

# ---------------------------------------------------------------------------
# 5. Migration setup: coarse grid + inclusion-free velocity + traveltimes
# ---------------------------------------------------------------------------
dxm = 1.0e-4  # 0.1 mm migration grid (sim grid downsampled by 2)
vel_mig = vel_layers[::2, ::2].astype(np.float64)  # three layers, NO voids
nxm, nzm = vel_mig.shape
xm = np.arange(nxm) * dxm
zm = np.arange(nzm) * dxm

# element grid indices on the migration grid
six = np.round(elem_x / dxm).astype(np.int64)
siz = np.round(elem_z / dxm).astype(np.int64)

t0 = time.perf_counter()
trav = _trav_table(1.0 / vel_mig, dxm, six, siz)  # (nxm*nzm, n_el)
print(f"eikonal traveltime tables: {time.perf_counter() - t0:.1f} s, {trav.shape}")
# sources and receivers are co-located (FMC), so the two tables are identical
trav_srcs = trav
trav_recs = trav

# Kirchhoff geometry and wavelet
t = np.arange(nt) * dt
srcs = np.vstack((elem_x, elem_z))
recs = np.vstack((elem_x, elem_z))
ns, nr = srcs.shape[1], recs.shape[1]
wav, _, wavc = ricker(t[:121], f0=freq)

# ---------------------------------------------------------------------------
# 6. Build both operators (byot traveltimes) and migrate
# ---------------------------------------------------------------------------
shot_recs = [np.arange(nr) for _ in range(ns)]
Op_ms = Kirchhoff(
    zm, xm, t, srcs, recs, vel_mig, wav, wavc,
    mode="byot", trav=(trav_srcs, trav_recs), dynamic=False,
    shot_recs=shot_recs, engine="numba",
)

ops = [
    Kirchhoff(
        zm, xm, t, srcs[:, i : i + 1], recs, vel_mig, wav, wavc,
        mode="byot", trav=(trav_srcs[:, i : i + 1], trav_recs), dynamic=False,
        engine="numba",
    )
    for i in range(ns)
]
Op_vs = pylops.VStack(ops)

# ---------------------------------------------------------------------------
# 6b. Data conditioning -- three ways to feed the migration operator
# ---------------------------------------------------------------------------
# The Kirchhoff operator is a *linear (Born) adjoint*: its data model is the
# field SCATTERED by perturbations relative to the background velocity. Feeding
# it the raw total field (dominated by the direct wave + the strong water/
# aluminum wall reflections + slab reverberations) buries the weak void
# diffractions. We compare three conditionings:
#   (1) total      -- raw recorded field (the problematic baseline)
#   (2) mute+gain  -- direct-wave top-mute + t^2 spreading gain on the total field
#   (3) scattered  -- reference-subtracted field (matches the operator's model)
tgain = (t / t[-1]) ** 2  # depth/spreading compensation

# offset-dependent direct-wave top mute: zero everything earlier than the
# water direct arrival (+pad). Reflections from depth arrive later, so they
# survive while the dominant source/direct energy is removed.
mute = np.ones((ns, nr, nt))
for i in range(ns):
    tdir = np.abs(elem_x - elem_x[i]) / V_WATER + peak_time + 0.5e-6
    for j in range(nr):
        mute[i, j, t < tdir[j]] = 0.0

fmc_tot = fmc
fmc_mut = fmc * mute * tgain
fmc_sct = fmc_scat * tgain


def migrate(data):
    return (Op_ms.H @ data.ravel()).reshape(nxm, nzm)


madj_tot = migrate(fmc_tot)
madj_mut = migrate(fmc_mut)
madj_sct = migrate(fmc_sct)

# ---------------------------------------------------------------------------
# 7. multishot == VStack equality (operator property, checked on scattered data)
# ---------------------------------------------------------------------------
madj_sct_vs = (Op_vs.H @ fmc_sct.ravel()).reshape(nxm, nzm)
absdiff = np.abs(madj_sct - madj_sct_vs).max()
rel = absdiff / np.abs(madj_sct).max()
print("\nMulti-shot vs VStack migration image (scattered field)")
print(f"  max |m_ms - m_vs|     = {absdiff:.3e}")
print(f"  relative to image max = {rel:.3e}")
print(f"  EQUAL (rel < 1e-6)?   = {rel < 1e-6}")

ext_mm = (xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3)
ext_mig = (xm[0] * 1e3, xm[-1] * 1e3, zm[-1] * 1e3, zm[0] * 1e3)


def mark(a):
    a.scatter(elem_x * 1e3, elem_z * 1e3, marker="v", s=12, c="r", edgecolors="k")
    for cx, cz in centers:
        a.scatter(cx * 1e3, cz * 1e3, marker="o", s=55, facecolors="none",
                  edgecolors="lime", linewidths=1.3)
    a.set_xlabel("x [mm]"), a.set_ylabel("z [mm]")


# (a) true velocity model + array
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(vel_true.T, cmap="viridis", extent=ext_mm, aspect="equal")
ax.scatter(elem_x * 1e3, elem_z * 1e3, marker="v", s=30, c="r", edgecolors="k",
           label="FMC elements")
plt.colorbar(im, ax=ax, label="velocity [m/s]")
ax.set_title("True model: water / aluminum / water with air voids")
ax.set_xlabel("x [mm]"), ax.set_ylabel("z [mm]")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "velocity_true.png"), dpi=130)

# (b) FMC shot gather: total vs scattered field
ishot = ns // 2
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for a, g, title in zip(axs, [fmc[ishot], fmc_scat[ishot]],
                       ["total field", "scattered (reference-subtracted)"]):
    gmax = np.abs(g).max()
    a.imshow(g.T, cmap="gray", vmin=-gmax, vmax=gmax,
             extent=(0, nr, t[-1] * 1e6, t[0] * 1e6), aspect="auto")
    a.set_title(f"FMC shot #{ishot}: {title}")
    a.set_xlabel("receiver index")
axs[0].set_ylabel("t [us]")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fmc_shot_gather.png"), dpi=130)

# (c) three-way migration comparison (each on its own amplitude scale)
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for a, img, title in zip(
    axs,
    [madj_tot, madj_mut, madj_sct],
    ["(1) total field (raw)", "(2) direct-mute + t$^2$ gain",
     "(3) reference-subtracted (scattered)"],
):
    mmax = np.percentile(np.abs(img), 99.8)
    a.imshow(img.T, cmap="gray_r", extent=ext_mig, aspect="equal",
             vmin=-mmax, vmax=mmax)
    a.set_title(title)
    mark(a)
fig.suptitle("Kirchhoff migration of deepwave FMC data "
             "(green = true voids) -- effect of data conditioning")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "migration_compare.png"), dpi=130)

np.savez_compressed(
    os.path.join(OUTDIR, "results.npz"),
    vel_true=vel_true, vel_mig=vel_mig,
    fmc=fmc.astype(np.float32), fmc_scat=fmc_scat.astype(np.float32),
    madj_tot=madj_tot, madj_mut=madj_mut, madj_sct=madj_sct,
    xm=xm, zm=zm, elem_x=elem_x, elem_z=elem_z,
    void_centers=np.array(centers), t=t,
)
print(f"\nSaved figures and results.npz to {OUTDIR}/")
