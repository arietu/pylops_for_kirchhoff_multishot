r"""
25. Kirchhoff multi-shot migration with eikonal traveltimes
===========================================================
The :ref:`multi-shot acquisition tutorial <sphx_glr_tutorials_kirchhoffmultishot.py>`
used ``mode="analytic"``, which assumes a **constant** velocity so that
traveltimes are simply straight-ray distances divided by the velocity. Real
media are heterogeneous, and there the traveltimes must be computed by solving
the eikonal equation

.. math::
        |\nabla t(\mathbf{x})|\, v(\mathbf{x}) = 1 ,

which :py:class:`pylops.waveeqprocessing.Kirchhoff` does internally with
``mode="eikonal"`` (a fast-marching solver, ``scikit-fmm``).

This tutorial mirrors the multi-shot Full Matrix Capture (FMC) example but uses a
**depth-varying velocity model**. With a non-constant velocity ``mode="analytic"``
is not applicable, so eikonal traveltimes are required. The important point is
that the ``shot_recs`` multi-shot interface is completely independent of how the
traveltimes are obtained: the very same acquisition description works for
``analytic`` and ``eikonal`` alike.

.. note::
   ``mode="eikonal"`` requires the ``scikit-fmm`` package to be installed.
"""
import matplotlib.pyplot as plt
import numpy as np

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)

###############################################################################
# We build a smoothly varying velocity model whose velocity increases linearly
# with depth (a common first approximation for the subsurface), and embed a few
# small reflective inclusions that act as point-like scatterers.

# Spatial axes
nx, nz = 81, 61
dx, dz = 4.0, 4.0
x, z = np.arange(nx) * dx, np.arange(nz) * dz

# Heterogeneous velocity: linear vertical gradient v(z) = v0 + k * z
v0, kgrad = 1500.0, 2.0
vel = np.broadcast_to(v0 + kgrad * z, (nx, nz)).copy()  # shape (nx, nz)

# Reflectivity model: a few small reflective inclusions
refl = np.zeros((nx, nz))
inclusions = [(20, 18), (38, 40), (58, 22), (30, 50), (50, 48)]
for ix, iz in inclusions:
    refl[ix - 1 : ix + 1, iz - 1 : iz + 1] = 1.0

###############################################################################
# As before we place a surface transmit/receive array recording in Full Matrix
# Capture mode (sources and receivers co-located, every shot listens to every
# receiver).

narr = 16
ax_ = np.linspace(10 * dx, (nx - 10) * dx, narr)
az = np.full(narr, dz)
srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

###############################################################################
# Display the velocity model with the inclusions and the acquisition geometry.

plt.figure(figsize=(8, 6))
im = plt.imshow(vel.T, cmap="viridis", extent=(x[0], x[-1], z[-1], z[0]))
plt.contour(
    x, z, refl.T, levels=[0.5], colors="w", linewidths=1.5
)  # outline inclusions
plt.scatter(srcs[0], srcs[1], marker="*", s=120, c="r", edgecolors="k")
plt.colorbar(im, label="velocity [m/s]")
plt.axis("tight")
plt.xlabel("x [m]"), plt.ylabel("z [m]")
plt.title("Heterogeneous velocity, inclusions (white) and FMC array")
plt.tight_layout()

###############################################################################
# Build the multi-shot Kirchhoff operator in ``eikonal`` mode. The velocity is
# now passed as a 2-D array; the operator solves the eikonal equation for the
# source- and receiver-side traveltime tables internally.

nt = 450
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)

shot_recs = [np.arange(nr) for _ in range(ns)]  # Full Matrix Capture

Op = Kirchhoff(
    z,
    x,
    t,
    srcs,
    recs,
    vel,  # 2-D heterogeneous velocity model
    wav,
    wavc,
    mode="eikonal",
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)
print(f"operator dimsd = {Op.dimsd}  (n_shots, max_recs, nt)")

###############################################################################
# Forward demigration models the FMC data, and the adjoint migrates it back to
# an image of the inclusions.

d = (Op @ refl.ravel()).reshape(Op.dimsd)
madj = (Op.H @ d.ravel()).reshape(nx, nz)

###############################################################################
# A couple of shot gathers from the eikonal FMC data set. Compared with the
# constant-velocity case the diffraction events are no longer symmetric
# hyperbolas, because the increasing velocity with depth bends the rays.

itimes = slice(0, 320)
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
# Finally, the true reflectivity next to the eikonal migration image. The
# inclusions are imaged at their correct positions; because the migration uses
# the heterogeneous-velocity traveltimes, the deeper events are focused at the
# right depth despite the velocity gradient.

# sphinx_gallery_thumbnail_number = 3
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
axs[0].imshow(refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[0].scatter(srcs[0], srcs[1], marker="*", s=60, c="r", edgecolors="k")
axs[0].set_title(r"True reflectivity $m$")
axs[0].set_xlabel("x [m]"), axs[0].set_ylabel("z [m]")
axs[0].axis("tight")

mmax = np.abs(madj).max()
axs[1].imshow(
    madj.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=mmax
)
axs[1].scatter(srcs[0], srcs[1], marker="*", s=60, c="r", edgecolors="k")
axs[1].set_title(r"Eikonal migration $m_{adj} = \mathbf{Op}^H \mathbf{d}$")
axs[1].set_xlabel("x [m]"), axs[1].set_ylabel("z [m]")
axs[1].axis("tight")
plt.tight_layout()

###############################################################################
# The only change relative to the analytic multi-shot tutorial is
# ``mode="eikonal"`` together with a 2-D velocity model -- the ``shot_recs``
# acquisition description, the padded data layout, and the forward/adjoint usage
# are all identical. The same holds for sparse acquisitions (e.g. zero-offset),
# which can be combined with eikonal traveltimes in exactly the same way.

plt.show()
