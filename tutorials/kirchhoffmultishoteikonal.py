r"""
25. Kirchhoff multi-shot migration with eikonal traveltimes
===========================================================
The :ref:`multi-shot acquisition tutorial <sphx_glr_tutorials_kirchhoffmultishot.py>`
used ``mode="analytic"``, which assumes a **constant** velocity so that
traveltimes are simply straight-ray distances divided by the velocity. Real
media are heterogeneous, and there the traveltimes must be computed by solving
the eikonal equation

.. math::
        |\nabla t(\mathbf{x})|\, v(\mathbf{x}) = 1 .

:py:class:`pylops.waveeqprocessing.Kirchhoff` can do this internally with
``mode="eikonal"`` (which relies on the ``scikit-fmm`` fast-marching package).
To keep this tutorial completely dependency-free, however, we instead compute
the eikonal traveltime tables with a small self-contained fast-marching solver
and hand them to the operator via ``mode="byot"`` ("bring your own tables").
This is exactly the table layout ``mode="eikonal"`` would produce -- a pair of
source- and receiver-side tables -- and it is also the form required by the
multi-shot interface.

The key takeaway is that the ``shot_recs`` multi-shot interface is completely
independent of how the traveltimes are obtained: the same acquisition
description works whether the tables come from the analytic formula, from an
eikonal solver, or are supplied directly.
"""
import heapq

import matplotlib.pyplot as plt
import numpy as np

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)


###############################################################################
# A compact first-order fast-marching eikonal solver
# ---------------------------------------------------
# This solves :math:`|\nabla t|\,v = 1` on a regular 2-D grid using the classic
# Osher--Sethian upwind update with a heap-based fast-marching ordering. It is a
# small stand-in for ``scikit-fmm`` so that the tutorial needs no compiled
# dependency; for production use simply pass ``mode="eikonal"`` to the operator
# instead.


def eikonal_fmm(speed, dx, dz, isrc, jsrc):
    """First-arrival traveltimes from grid point (isrc, jsrc).

    Parameters
    ----------
    speed : :obj:`numpy.ndarray`
        Velocity model of shape ``(nx, nz)``.
    dx, dz : :obj:`float`
        Grid spacing along the two axes.
    isrc, jsrc : :obj:`int`
        Source grid indices.

    Returns
    -------
    T : :obj:`numpy.ndarray`
        Traveltime field of shape ``(nx, nz)``.
    """
    nx, nz = speed.shape
    slow = 1.0 / speed  # slowness s = 1/v
    T = np.full((nx, nz), np.inf)
    frozen = np.zeros((nx, nz), dtype=bool)
    T[isrc, jsrc] = 0.0
    heap = [(0.0, isrc, jsrc)]

    def update(i, j):
        # smallest frozen neighbour traveltime along each axis
        ux = min(
            T[i - 1, j] if i > 0 and frozen[i - 1, j] else np.inf,
            T[i + 1, j] if i < nx - 1 and frozen[i + 1, j] else np.inf,
        )
        uz = min(
            T[i, j - 1] if j > 0 and frozen[i, j - 1] else np.inf,
            T[i, j + 1] if j < nz - 1 and frozen[i, j + 1] else np.inf,
        )
        s = slow[i, j]
        # try the two-sided (Godunov) update, fall back to one-sided
        if np.isinf(ux):
            return uz + dz * s
        if np.isinf(uz):
            return ux + dx * s
        a = 1.0 / dx**2 + 1.0 / dz**2
        b = -2.0 * (ux / dx**2 + uz / dz**2)
        c = ux**2 / dx**2 + uz**2 / dz**2 - s**2
        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return min(ux + dx * s, uz + dz * s)
        t = (-b + np.sqrt(disc)) / (2.0 * a)
        if t < max(ux, uz):  # causality violated -> one-sided
            return min(ux + dx * s, uz + dz * s)
        return t

    while heap:
        t, i, j = heapq.heappop(heap)
        if frozen[i, j]:
            continue
        frozen[i, j] = True
        T[i, j] = t
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < nz and not frozen[ni, nj]:
                cand = update(ni, nj)
                if cand < T[ni, nj]:
                    T[ni, nj] = cand
                    heapq.heappush(heap, (cand, ni, nj))
    return T


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
# Compute the eikonal traveltime tables. For each source and each receiver we
# run the fast-marching solver and store the traveltime to every image point.
# The operator expects each table flattened in C order with shape
# ``(n_x n_z, n_s)`` and ``(n_x n_z, n_r)`` respectively -- a tuple of these two
# is precisely what ``mode="byot"`` (and the multi-shot interface) consumes.


def traveltime_table(positions):
    table = np.zeros((nx * nz, positions.shape[1]))
    for k in range(positions.shape[1]):
        isrc = int(round((positions[0, k] - x[0]) / dx))
        jsrc = int(round((positions[1, k] - z[0]) / dz))
        table[:, k] = eikonal_fmm(vel, dx, dz, isrc, jsrc).ravel()
    return table


trav_srcs = traveltime_table(srcs)
trav_recs = traveltime_table(recs)

###############################################################################
# Build the multi-shot Kirchhoff operator with the eikonal traveltime tables and
# the Full Matrix Capture geometry (every shot records at every receiver).

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
    vel,
    wav,
    wavc,
    mode="byot",
    trav=(trav_srcs, trav_recs),  # eikonal tables computed above
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
# The only thing that changed relative to the analytic multi-shot tutorial is
# the *origin* of the traveltime tables (here an eikonal solver for a
# heterogeneous velocity, supplied via ``mode="byot"``) -- the ``shot_recs``
# acquisition description, the padded data layout, and the forward/adjoint usage
# are all identical. If ``scikit-fmm`` is installed, the same result is obtained
# simply by passing ``mode="eikonal"`` with the 2-D velocity model and dropping
# the ``trav`` argument. The same recipe applies to sparse acquisitions (e.g.
# zero-offset) combined with eikonal traveltimes.

plt.show()
