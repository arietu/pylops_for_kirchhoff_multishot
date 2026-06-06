r"""
26. FMC migration with external traveltime and amplitude tables
===============================================================
The previous multi-shot tutorials computed the Green's-function traveltimes
internally (``mode="analytic"`` or ``mode="eikonal"``). In many workflows,
however, the traveltimes *and* the amplitudes are produced by an external engine
(a ray tracer, an eikonal/amplitude solver, or a full wave-equation modelling
code) and one simply wants to plug them into the Kirchhoff operator.

:py:class:`pylops.waveeqprocessing.Kirchhoff` supports this through
``mode="byot"`` ("bring your own tables"): the traveltimes are passed as a tuple
``trav=(trav_srcs, trav_recs)`` of source- and receiver-side tables. As of the
latest update, when ``dynamic=False`` you may *also* pass an amplitude tuple
``amp=(amp_srcs, amp_recs)``, and these amplitudes are applied **directly** -- the
weight at image point :math:`\mathbf{x}` for a source-receiver pair is the product
``amp_srcs[x, src] * amp_recs[x, rec]`` placed at traveltime
``trav_srcs[x, src] + trav_recs[x, rec]``, with no extra opening-angle, velocity,
or aperture scaling. This is exactly what you want when your tables already encode
the full Green's-function amplitude.

This tutorial demonstrates the feature for a Full Matrix Capture (FMC)
acquisition: every shot records at every receiver (``shot_recs[i] = arange(n_r)``).
We build the external tables ourselves (analytic traveltimes and a
geometrical-spreading amplitude), then compare the amplitude-weighted migration
against the purely kinematic one.
"""
import matplotlib.pyplot as plt
import numpy as np

from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)

###############################################################################
# Homogeneous medium with a few small reflective inclusions and a surface
# transmit/receive array recording in Full Matrix Capture mode.

# Spatial axes
nx, nz = 81, 61
dx, dz = 4.0, 4.0
x, z = np.arange(nx) * dx, np.arange(nz) * dz
v0 = 1000.0  # background velocity [m/s]

# Reflectivity model: small reflective inclusions
refl = np.zeros((nx, nz))
inclusions = [(20, 18), (38, 40), (58, 22), (30, 50), (50, 48)]
for ix, iz in inclusions:
    refl[ix - 1 : ix + 1, iz - 1 : iz + 1] = 1.0

# Surface FMC array (sources and receivers co-located)
narr = 16
ax_ = np.linspace(10 * dx, (nx - 10) * dx, narr)
az = np.full(narr, dz)
srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

# Time axis and wavelet
nt = 450
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)

###############################################################################
# Build the *external* traveltime and amplitude tables
# -----------------------------------------------------
# Here we obtain the source- and receiver-side traveltimes from the analytic
# constant-velocity solver, and define the amplitudes as the 2-D
# geometrical-spreading factor :math:`a(\mathbf{x}) = 1/\sqrt{\text{dist}}`. In
# practice these tables would come from your own propagation engine; the operator
# only cares that they are arrays of shape ``(n_x n_z, n_s)`` and
# ``(n_x n_z, n_r)``.

trav_srcs, trav_recs, dist_srcs, dist_recs, _, _ = Kirchhoff._traveltime_table(
    z, x, srcs, recs, v0, mode="analytic"
)

# geometrical-spreading amplitude tables (a small epsilon avoids division by 0
# right at the source/receiver positions)
eps = 1e-2 * (dist_srcs.max() + dist_recs.max())
amp_srcs = 1.0 / np.sqrt(dist_srcs + eps)
amp_recs = 1.0 / np.sqrt(dist_recs + eps)

print("trav_srcs:", trav_srcs.shape, " amp_srcs:", amp_srcs.shape)

###############################################################################
# Visualise one source-side amplitude table (it decays away from the source, as
# expected for geometrical spreading).

isrc_show = ns // 2
plt.figure(figsize=(8, 6))
im = plt.imshow(
    amp_srcs[:, isrc_show].reshape(nx, nz).T,
    cmap="magma",
    extent=(x[0], x[-1], z[-1], z[0]),
)
plt.contour(x, z, refl.T, levels=[0.5], colors="w", linewidths=1.0)
plt.scatter(srcs[0, isrc_show], srcs[1, isrc_show], marker="*", s=200,
            c="cyan", edgecolors="k")
plt.colorbar(im, label="amplitude")
plt.title(f"External amplitude table, source #{isrc_show}")
plt.xlabel("x [m]"), plt.ylabel("z [m]")
plt.axis("tight")
plt.tight_layout()

###############################################################################
# Build the FMC multi-shot operator with the external tables
# ----------------------------------------------------------
# ``mode="byot"`` with ``dynamic=False`` and an ``amp`` tuple applies the
# amplitudes directly. The FMC geometry is encoded by letting every shot record
# at every receiver.

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
    mode="byot",
    trav=(trav_srcs, trav_recs),
    amp=(amp_srcs, amp_recs),
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)
print(f"operator dimsd = {Op.dimsd}  (n_shots, max_recs, nt)")

# For comparison, the purely kinematic operator (same traveltimes, no amplitude)
Op_kin = Kirchhoff(
    z,
    x,
    t,
    srcs,
    recs,
    v0,
    wav,
    wavc,
    mode="byot",
    trav=(trav_srcs, trav_recs),
    dynamic=False,
    shot_recs=shot_recs,
    engine="numba",
)

###############################################################################
# Forward demigration models the FMC data; the adjoint migrates it back. We do
# this for both the amplitude-weighted operator and the kinematic one.

d = (Op @ refl.ravel()).reshape(Op.dimsd)
madj = (Op.H @ d.ravel()).reshape(nx, nz)

d_kin = (Op_kin @ refl.ravel()).reshape(Op_kin.dimsd)
madj_kin = (Op_kin.H @ d_kin.ravel()).reshape(nx, nz)

###############################################################################
# A shot gather from the amplitude-weighted FMC data set.

itimes = slice(0, 320)
ishot = ns // 2
plt.figure(figsize=(6, 5))
dmax = np.abs(d[ishot]).max()
plt.imshow(
    d[ishot, :, itimes].T,
    cmap="gray",
    vmin=-dmax,
    vmax=dmax,
    extent=(0, nr, t[itimes][-1], t[0]),
    aspect="auto",
)
plt.title(f"Shot gather #{ishot} (amplitude-weighted)")
plt.xlabel("receiver index"), plt.ylabel("t [s]")
plt.tight_layout()

###############################################################################
# Migration images. Both place the inclusions correctly; the amplitude weighting
# changes the relative strength of the events. Here the geometrical-spreading
# factor :math:`1/\sqrt{\text{dist}}` decays with distance, so it down-weights
# the deeper reflectors relative to the shallow ones.

# sphinx_gallery_thumbnail_number = 3
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(refl.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[0].set_title(r"True reflectivity $m$")

mk = madj_kin / np.abs(madj_kin).max()
axs[1].imshow(mk.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[1].set_title("Kinematic migration (no amp)")

ma = madj / np.abs(madj).max()
axs[2].imshow(ma.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
axs[2].set_title("Amplitude-weighted migration")

for ax in axs:
    ax.scatter(srcs[0], srcs[1], marker="*", s=40, c="r", edgecolors="k")
    ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
    ax.axis("tight")
plt.tight_layout()

###############################################################################
# The key point is that both the traveltimes and the amplitudes were supplied
# externally as ``(source_table, receiver_table)`` tuples and applied verbatim by
# the operator -- no internal velocity model, ray angles, or aperture tapering
# are involved. This makes it straightforward to drive the multi-shot/FMC
# Kirchhoff operator with traveltimes and amplitudes from any propagation engine
# of choice.

plt.show()
