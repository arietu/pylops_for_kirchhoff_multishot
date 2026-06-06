r"""
24. Sparse least-squares Kirchhoff migration (multi-shot)
=========================================================
This tutorial builds on the :ref:`multi-shot acquisition example
<sphx_glr_tutorials_kirchhoffmultishot.py>` and shows how to turn the
*migration* (adjoint) of the multi-shot :py:class:`pylops.waveeqprocessing.Kirchhoff`
operator into a proper *inversion* for the reflectivity.

The adjoint of the demigration operator gives a quick image of the subsurface,
but it is blurred by the source/receiver aperture and the band-limited wavelet.
We can do much better by solving the inverse problem

.. math::
        \mathbf{m} = \operatorname*{arg\,min}_\mathbf{m}
        \tfrac{1}{2}\lVert \mathbf{d} - \mathbf{Op}\,\mathbf{m} \rVert_2^2
        + \epsilon \lVert \mathbf{m} \rVert_1 .

Because the reflectivity here consists of a few small, isolated inclusions, the
model is genuinely **sparse**, which makes an :math:`\ell_1` penalty (solved with
FISTA) a natural choice. We compare three reconstructions -- the migration
(adjoint), a least-squares (:math:`\ell_2`) inversion, and the sparse
(:math:`\ell_1`) inversion -- and we tune the FISTA regularization strength
:math:`\epsilon` by sweeping it and keeping the value that minimises the
reconstruction error against the known model.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lsqr

from pylops.optimization.sparsity import fista
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

plt.close("all")
np.random.seed(0)

###############################################################################
# As in the multi-shot tutorial we use a homogeneous medium with a few small
# reflective inclusions and a surface transmit/receive array recording in Full
# Matrix Capture (FMC) mode. We keep the model and the array deliberately small
# because the hyperparameter sweep below solves the inverse problem several
# times.

# Spatial axes
nx, nz = 51, 31
dx, dz = 4.0, 4.0
x, z = np.arange(nx) * dx, np.arange(nz) * dz
v0 = 1000.0  # constant background velocity [m/s]

# Reflectivity model: a few small reflective inclusions
refl = np.zeros((nx, nz))
inclusions = [(13, 10), (24, 20), (37, 12), (18, 25), (32, 24)]
for ix, iz in inclusions:
    refl[ix - 1 : ix + 1, iz - 1 : iz + 1] = 1.0

# Surface FMC array (sources and receivers co-located)
narr = 10
ax_ = np.linspace(8 * dx, (nx - 8) * dx, narr)
az = np.full(narr, dz)
srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

# Time axis and wavelet
nt = 300
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)

###############################################################################
# Build the multi-shot Kirchhoff operator with the FMC geometry (every shot
# records at every receiver) and model the data with a forward pass.

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
    mode="analytic",
    shot_recs=shot_recs,
    engine="numba",
)

d = Op @ refl.ravel()

###############################################################################
# The migration image is simply the adjoint applied to the data. We also run a
# least-squares inversion with :py:func:`scipy.sparse.linalg.lsqr`.

madj = (Op.H @ d).reshape(nx, nz)

minv_ls = lsqr(Op, d, iter_lim=30, atol=0, btol=0)[0].reshape(nx, nz)


###############################################################################
# To quantify reconstruction quality we use the relative :math:`\ell_2` error
# against the known model. (In a real survey the true model is unknown and one
# would instead rely on the data misfit and an L-curve; here we exploit it to
# pick a good regularization strength.)


def rel_error(m):
    return np.linalg.norm(m.ravel() - refl.ravel()) / np.linalg.norm(refl.ravel())


###############################################################################
# **Hyperparameter sweep.** We solve the :math:`\ell_1` problem with FISTA for a
# range of regularization strengths :math:`\epsilon` (fixed iteration budget)
# and keep the value that minimises the reconstruction error.

niter = 40
epss = np.logspace(0, 3.0, 4)
errs = np.zeros_like(epss)
models = []
for i, eps in enumerate(epss):
    minv = fista(Op, d, niter=niter, eps=eps)[0].reshape(nx, nz)
    models.append(minv)
    errs[i] = rel_error(minv)

ibest = int(np.argmin(errs))
eps_best = epss[ibest]
minv_l1 = models[ibest]

print(f"adjoint            rel. error = {rel_error(madj):.3f}")
print(f"least-squares      rel. error = {rel_error(minv_ls):.3f}")
print(f"best L1 (eps={eps_best:.3g}) rel. error = {errs[ibest]:.3f}")

###############################################################################
# The error-vs-:math:`\epsilon` curve shows the classic trade-off: too small an
# :math:`\epsilon` barely regularizes (noisy, adjoint-like), too large an
# :math:`\epsilon` over-sparsifies and starts erasing true inclusions.

plt.figure(figsize=(7, 4))
plt.semilogx(epss, errs, "o-", color="C0")
plt.semilogx(
    eps_best, errs[ibest], "*", ms=18, color="C3",
    label=fr"best $\epsilon$ = {eps_best:.3g}",
)
plt.xlabel(r"FISTA regularization strength $\epsilon$")
plt.ylabel("relative $\\ell_2$ error")
plt.title("L1 hyperparameter sweep")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

###############################################################################
# Finally we compare the true reflectivity with the three reconstructions. The
# :math:`\ell_1` solution recovers the small inclusions as compact, well-located
# events, strongly suppressing the migration smiles and the residual blur left
# by the least-squares solution.

# sphinx_gallery_thumbnail_number = 2
fig, axs = plt.subplots(1, 4, figsize=(16, 4.5))
panels = [
    (refl, "True reflectivity $m$"),
    (madj / np.abs(madj).max(), "Migration (adjoint)"),
    (minv_ls, r"Least-squares ($\ell_2$)"),
    (minv_l1, fr"Sparse ($\ell_1$), $\epsilon$={eps_best:.3g}"),
]
for ax, (img, title) in zip(axs, panels):
    ax.imshow(img.T, cmap="gray_r", extent=(x[0], x[-1], z[-1], z[0]), vmin=0, vmax=1)
    ax.scatter(srcs[0], srcs[1], marker="*", s=40, c="r", edgecolors="k")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("tight")
plt.tight_layout()

###############################################################################
# The migration (adjoint) gives a fast but blurred image, the least-squares
# inversion sharpens it by accounting for the operator, and the sparse
# :math:`\ell_1` inversion -- well matched to the point-like nature of the
# inclusions -- yields the cleanest reconstruction. The same recipe applies to
# any ``shot_recs`` acquisition geometry, including sparse zero-offset surveys.

plt.show()
