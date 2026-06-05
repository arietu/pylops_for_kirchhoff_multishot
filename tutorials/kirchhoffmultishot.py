r"""
23. Kirchhoff multi-shot acquisition
====================================
This tutorial illustrates the *multi-shot* interface of the
:py:class:`pylops.waveeqprocessing.Kirchhoff` operator.

In a real seismic (or ultrasonic) survey the data are acquired one **shot** at a
time: a single source is fired and a (possibly shot-dependent) subset of
receivers records the scattered wavefield. Each shot is an independent
acquisition session, yet the subsurface image is reconstructed from the
contribution of *all* shots together.

The ``shot_recs`` parameter encodes this acquisition geometry. It is a list of
length :math:`n_s` (one entry per source/shot) where ``shot_recs[i]`` holds the
indices of the receivers that are active during shot ``i``. The forward data are
returned in a padded ``(n_shots, max_recs, n_t)`` array, so a single shot gather
is simply ``d[i]``.

Here we reproduce a **Full Matrix Capture (FMC)** acquisition, in which every
element of a transducer array transmits in turn while *all* elements record.
This is expressed in the multi-shot API by making every shot listen to every
receiver, i.e. ``shot_recs[i] = arange(n_r)`` for all ``i``.

The reflectivity model consists of a handful of small reflective inclusions
embedded in an otherwise homogeneous medium. We model the data with a forward
pass of the demigration operator and then apply its adjoint (the migration
operator) to obtain an image of the inclusions.
"""
import matplotlib.pyplot as plt
import numpy as np

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)

###############################################################################
# Let us start by defining a homogeneous velocity model and a reflectivity
# model containing several small inclusions (each a little square of
# reflectivity), which act as point-like scatterers.

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

###############################################################################
# We now place a transducer array along the surface (top of the model). For a
# Full Matrix Capture acquisition the transmitting and receiving elements
# coincide, so sources and receivers share the same positions.

narr = 21  # number of array elements
ax_ = np.linspace(10 * dx, (nx - 10) * dx, narr)
az = np.full(narr, dz)  # just below the surface

srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

###############################################################################
# Display the model together with the acquisition geometry.

plt.figure(figsize=(8, 7))
im = plt.imshow(
    refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1
)
plt.scatter(
    recs[0], recs[1], marker="v", s=80, c="b", edgecolors="k", label="receivers"
)
plt.scatter(
    srcs[0], srcs[1], marker="*", s=120, c="r", edgecolors="k", label="sources"
)
plt.colorbar(im, label="reflectivity")
plt.axis("tight")
plt.xlabel("x [m]"), plt.ylabel("z [m]")
plt.title("Reflectivity model and FMC array")
plt.legend(loc="lower right")
plt.tight_layout()

###############################################################################
# Next we define the time axis and the source wavelet, and build the multi-shot
# Kirchhoff operator. The Full Matrix Capture geometry is encoded by letting
# every shot record at every receiver.

nt = 500
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)

# Full Matrix Capture: every shot listens to all receivers
shot_recs = [np.arange(nr) for _ in range(ns)]

Op = Kirchhoff(
    z,
    x,
    t,
    srcs,
    recs,
    v0,
    wav,
    wavc,
    mode="analytic",  # constant-velocity analytic traveltimes (no eikonal solver)
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)

# With FMC every shot records at all receivers, so the padded data axis has
# max_recs == nr and the data shape is (n_shots, n_r, n_t).
print(f"operator dimsd = {Op.dimsd}  (n_shots, max_recs, nt)")

###############################################################################
# **Forward pass** -- model the FMC data from the reflectivity. The result is a
# padded ``(n_shots, max_recs, nt)`` cube; each ``d[i]`` is the shot gather of
# source ``i``.

d = Op @ refl.ravel()
d = d.reshape(Op.dimsd)

###############################################################################
# **Adjoint pass** -- apply the migration operator (the adjoint of demigration)
# to the modelled data to obtain an image of the subsurface reflectivity.

madj = Op.H @ d.ravel()
madj = madj.reshape(nx, nz)

###############################################################################
# Because FMC records every source-receiver pair, the multi-shot operator is
# equivalent to the classic full-Cartesian Kirchhoff operator. We can verify
# this by comparing against the operator built without ``shot_recs``.

Op_dense = Kirchhoff(
    z, x, t, srcs, recs, v0, wav, wavc, mode="analytic", dynamic=False, engine="numba"
)
d_dense = (Op_dense @ refl.ravel()).reshape(ns, nr, nt)
print(
    "max |FMC multishot - dense Cartesian| forward data: "
    f"{np.abs(d - d_dense).max():.3e}"
)

###############################################################################
# Let us inspect a couple of shot gathers from the FMC data set.

itimes = slice(0, 350)  # only show the early part of the records
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, ishot in zip(axs, [0, ns // 2]):
    dmax = np.abs(d[ishot]).max()
    ax.imshow(
        d[ishot, :, itimes].T,
        cmap="gray",
        vmin=-dmax,
        vmax=dmax,
        extent=(0, nr, t[itimes][-1], t[0]),
        aspect="auto",
    )
    ax.set_title(f"Shot gather #{ishot} (source x={srcs[0, ishot]:.0f} m)")
    ax.set_xlabel("receiver index")
    ax.set_ylabel("t [s]")
plt.tight_layout()

###############################################################################
# Finally we compare the true reflectivity model with the migration image
# obtained from the adjoint. The small inclusions are recovered at their correct
# locations; the characteristic migration "smiles" and limited-aperture blur are
# expected for a single adjoint application (a least-squares inversion would
# sharpen these, see the Least-squares migration tutorial).

# sphinx_gallery_thumbnail_number = 3
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
axs[0].imshow(refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[0].scatter(srcs[0], srcs[1], marker="*", s=60, c="r", edgecolors="k")
axs[0].set_title(r"True reflectivity $m$")
axs[0].set_xlabel("x [m]"), axs[0].set_ylabel("z [m]")
axs[0].axis("tight")

mmax = np.abs(madj).max()
im = axs[1].imshow(
    madj.T,
    cmap="gray_r",
    extent=(x[0], x[-1], z[-1], z[0]),
    vmin=0,
    vmax=mmax,
)
axs[1].scatter(srcs[0], srcs[1], marker="*", s=60, c="r", edgecolors="k")
axs[1].set_title(r"Migration image $m_{adj} = \mathbf{Op}^H \mathbf{d}$")
axs[1].set_xlabel("x [m]"), axs[1].set_ylabel("z [m]")
axs[1].axis("tight")
plt.tight_layout()

###############################################################################
# Sparse acquisition: zero-offset (pulse-echo) recording
# ------------------------------------------------------
# The real power of ``shot_recs`` lies in *sparse* acquisitions, where each shot
# records at a different, limited subset of receivers. Here we set up a
# **zero-offset** (monostatic / pulse-echo) survey: since the transmitting and
# receiving elements are co-located, the only receiver recording shot ``i`` is
# the one sitting at the source position, i.e. ``shot_recs[i] = [i]``.
#
# Each shot therefore contributes a single trace, and the padded data cube
# collapses to ``(n_shots, 1, nt)`` -- a fraction of the FMC data volume
# (:math:`n_s` traces instead of :math:`n_s \times n_r`).

shot_recs_zo = [np.array([i]) for i in range(ns)]

Op_zo = Kirchhoff(
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
    shot_recs=shot_recs_zo,
    engine="numba",
)
print(f"zero-offset operator dimsd = {Op_zo.dimsd}  (n_shots, max_recs, nt)")

# forward and adjoint for the zero-offset acquisition
d_zo = (Op_zo @ refl.ravel()).reshape(Op_zo.dimsd)
madj_zo = (Op_zo.H @ d_zo.ravel()).reshape(nx, nz)

###############################################################################
# The single active receiver per shot makes ``d_zo`` a classic zero-offset
# section: collapsing the trivial receiver axis, ``d_zo[:, 0]`` is an
# ``(n_shots, nt)`` image in which each scatterer draws a diffraction hyperbola
# centred on the scatterer's surface position.

zo = d_zo[:, 0]  # (n_shots, nt)
zomax = np.abs(zo).max()
plt.figure(figsize=(7, 5))
plt.imshow(
    zo[:, itimes].T,
    cmap="gray",
    vmin=-zomax,
    vmax=zomax,
    extent=(srcs[0, 0], srcs[0, -1], t[itimes][-1], t[0]),
    aspect="auto",
)
plt.xlabel("source = receiver position x [m]")
plt.ylabel("t [s]")
plt.title("Zero-offset section (one trace per shot)")
plt.tight_layout()

###############################################################################
# Migrating the sparse zero-offset data still recovers the inclusions, though
# with lower fold than the Full Matrix Capture image: fewer recorded traces mean
# fewer contributions stacking at each image point.

# sphinx_gallery_thumbnail_number = 4
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[0].set_title(r"True reflectivity $m$")

mmax = np.abs(madj).max()
axs[1].imshow(
    madj.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=mmax
)
axs[1].set_title(r"FMC migration ($n_s \times n_r$ traces)")

zomax = np.abs(madj_zo).max()
axs[2].imshow(
    madj_zo.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=zomax
)
axs[2].set_title(r"Zero-offset migration ($n_s$ traces)")

for ax in axs:
    ax.scatter(srcs[0], srcs[1], marker="*", s=40, c="r", edgecolors="k")
    ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
    ax.axis("tight")
plt.tight_layout()

###############################################################################
# The two acquisitions are driven by the exact same operator class; only the
# ``shot_recs`` lists differ. The Full Matrix Capture survey returns a dense
# ``(n_shots, n_r, nt)`` cube, while the zero-offset survey returns a compact
# ``(n_shots, 1, nt)`` cube -- both handled transparently by the padded
# multi-shot layout. In a more general survey the per-shot receiver counts can
# vary freely, and the trailing slots of shorter shots are simply left as zeros.

plt.show()
