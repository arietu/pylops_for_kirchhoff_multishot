r"""
24. Kirchhoff multi-shot via VStack
===================================
This tutorial reproduces the **Full Matrix Capture (FMC)** acquisition of the
:ref:`Kirchhoff multi-shot tutorial <sphx_glr_tutorials_kirchhoffmultishot.py>`,
but instead of the built-in ``shot_recs`` multi-shot interface of
:py:class:`pylops.waveeqprocessing.Kirchhoff`, it assembles the survey from
**one plain Kirchhoff operator per shot**, stacked together with
:py:class:`pylops.VStack`.

The two approaches describe the *same* linear operator:

- **Multi-shot** -- a single ``Kirchhoff`` operator built with
  ``shot_recs=[arange(nr)] * ns``. Every shot listens to every receiver and the
  forward data are returned as a padded ``(n_shots, n_r, n_t)`` cube.
- **VStack** -- ``n_s`` independent ``Kirchhoff`` operators, the ``i``-th built
  with a single source ``srcs[:, i]`` (and all receivers, no ``shot_recs``).
  Each block models one shot gather ``(1, n_r, n_t)``; stacking them vertically
  with :py:class:`pylops.VStack` concatenates the shots back into the same
  ``(n_shots, n_r, n_t)`` data layout.

Because both express the identical FMC operator, their forward data and
migration images must agree to machine precision. We verify this, time the two
methods, and display the images side by side.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

import pylops
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)

###############################################################################
# We use the exact same homogeneous velocity model, reflectivity inclusions and
# Full Matrix Capture array as the Kirchhoff multi-shot tutorial, so that the
# two acquisitions can be compared apples-to-apples.

# Spatial axes
nx, nz = 101, 101
dx, dz = 4.0, 4.0
x, z = np.arange(nx) * dx, np.arange(nz) * dz
v0 = 1000.0  # constant background velocity [m/s]

# Reflectivity model: a few small reflective inclusions
refl = np.zeros((nx, nz))
inclusions = [(30, 35), (50, 60), (72, 38), (40, 80), (62, 82)]
for ix, iz in inclusions:
    refl[ix - 1 : ix + 1, iz - 1 : iz + 1] = 1.0

# Full Matrix Capture array along the surface (sources == receivers)
narr = 64  # number of array elements
ax_ = np.linspace(10 * dx, (nx - 10) * dx, narr)
az = np.full(narr, dz)  # just below the surface

srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

# Time axis and source wavelet
nt = 500
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)

###############################################################################
# Method 1 -- the built-in multi-shot operator
# --------------------------------------------
# For Full Matrix Capture every shot records at every receiver, which is encoded
# by ``shot_recs[i] = arange(nr)`` for all shots. This is the same operator as in
# the Kirchhoff multi-shot tutorial.

shot_recs = [np.arange(nr) for _ in range(ns)]

t0 = time.perf_counter()
Op_ms = Kirchhoff(
    z,
    x,
    t,
    srcs,
    recs,
    v0,
    wav,
    wavc,
    mode="analytic",
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)
t_build_ms = time.perf_counter() - t0
print(f"multi-shot operator dimsd = {Op_ms.dimsd}  (n_shots, n_r, nt)")

###############################################################################
# Method 2 -- one plain Kirchhoff operator per shot, stacked with VStack
# ---------------------------------------------------------------------
# We now build the *same* FMC survey from ``n_s`` independent operators. The
# ``i``-th operator sees a single source ``srcs[:, i:i+1]`` and all receivers,
# and is a plain ``Kirchhoff`` operator (no ``shot_recs``). Each block models one
# shot gather of shape ``(1, n_r, n_t)``; :py:class:`pylops.VStack` concatenates
# them along the data axis, yielding a combined operator whose output, reshaped
# to ``(n_shots, n_r, n_t)``, matches the multi-shot cube.

t0 = time.perf_counter()
ops = [
    Kirchhoff(
        z,
        x,
        t,
        srcs[:, i : i + 1],  # single source for this shot
        recs,  # FMC: every shot listens to all receivers
        v0,
        wav,
        wavc,
        mode="analytic",
        dynamic=False,
        engine="numba",
    )
    for i in range(ns)
]
Op_vs = pylops.VStack(ops)
t_build_vs = time.perf_counter() - t0
print(f"VStack operator shape = {Op_vs.shape}  (= ({ns}*{nr}*{nt}, {nx}*{nz}))")

###############################################################################
# Forward and adjoint for both methods
# ------------------------------------
# We model the FMC data and migrate it back with the adjoint, timing each pass.

m = refl.ravel()

t0 = time.perf_counter()
d_ms = Op_ms @ m
madj_ms = Op_ms.H @ d_ms
t_apply_ms = time.perf_counter() - t0

t0 = time.perf_counter()
d_vs = Op_vs @ m
madj_vs = Op_vs.H @ d_vs
t_apply_vs = time.perf_counter() - t0

d_ms = d_ms.reshape(ns, nr, nt)
d_vs = d_vs.reshape(ns, nr, nt)
madj_ms = madj_ms.reshape(nx, nz)
madj_vs = madj_vs.reshape(nx, nz)

###############################################################################
# Numerical equivalence
# ---------------------
# Since the two constructions describe the identical FMC operator, their forward
# data and migration images agree to machine precision.

print("\nNumerical equivalence (multi-shot vs VStack)")
print(f"  max |d_ms  - d_vs|    forward data = {np.abs(d_ms - d_vs).max():.3e}")
print(f"  max |m_ms  - m_vs|  migration image = {np.abs(madj_ms - madj_vs).max():.3e}")

###############################################################################
# Runtime comparison
# ------------------
# The two methods are mathematically identical, so any difference is pure
# bookkeeping overhead. The single multi-shot operator fuses all shots into one
# kernel call, whereas the VStack drives ``n_s`` separate operators from Python.

print("\nRuntime [s]            build      apply (fwd+adj)")
print(f"  multi-shot        {t_build_ms:9.3f}   {t_apply_ms:9.3f}")
print(f"  VStack ({ns:3d} ops)  {t_build_vs:9.3f}   {t_apply_vs:9.3f}")

###############################################################################
# Images side by side
# -------------------
# Finally we show the true reflectivity, the migration image from each method,
# and their difference. The two images are visually indistinguishable and the
# difference is at the noise floor.

# sphinx_gallery_thumbnail_number = 1
mmax = np.abs(madj_ms).max()
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
axs[0].imshow(refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[0].set_title(r"True reflectivity $m$")

axs[1].imshow(
    madj_ms.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=mmax
)
axs[1].set_title("Multi-shot migration")

axs[2].imshow(
    madj_vs.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=mmax
)
axs[2].set_title("VStack migration")

diff = madj_ms - madj_vs
dmax = max(np.abs(diff).max(), np.finfo(diff.dtype).tiny)
im = axs[3].imshow(
    diff.T, cmap="seismic", extent=(x[0], x[-1], z[-1], z[0]), vmin=-dmax, vmax=dmax
)
axs[3].set_title("Difference (ms - VStack)")
fig.colorbar(im, ax=axs[3], shrink=0.7)

for ax in axs:
    ax.scatter(srcs[0], srcs[1], marker="*", s=40, c="r", edgecolors="k")
    ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
    ax.axis("tight")
plt.tight_layout()

###############################################################################
# Both routes produce the same FMC operator and therefore the same image. The
# choice between them is one of convenience rather than correctness:
#
# - The **multi-shot** interface fuses every shot into a single operator with a
#   compact padded data layout. It carries the least per-call overhead and is the
#   natural choice when shots record *ragged* receiver subsets (the padded
#   ``shot_recs`` layout handles variable per-shot receiver counts transparently).
# - The **VStack** route keeps each shot as an independent ``Kirchhoff`` operator.
#   This composes cleanly with the rest of the pylops ecosystem -- the stack can
#   be combined with other operators, fed to any pylops solver for least-squares
#   migration, or distributed shot-by-shot across workers -- at the cost of more
#   Python-level overhead from driving ``n_s`` separate operators.

plt.show()
