r"""
Shared helpers for the blind FMC imaging tutorials
==================================================
Single home for everything the four ``deepwave_fmc_kirchhoff_blind*`` studies
have in common, so that the scientific core of the study -- the dataset
recipe, the deepwave simulation harness, the blind preprocessing chain, the
eikonal solver and the byot table construction -- exists exactly once:

* :func:`make_random_voids` / :func:`carve_voids` -- the seeded void-placement
  recipe. ``deepwave_fmc_kirchhoff_blind_multi.py`` and the pipeline shoot-out
  both draw from :data:`FIVE_VOID_CASE`, which is what guarantees they really
  score the same medium (previously enforced only by copy-pasted code).
* :func:`simulate_fmc` -- the one deepwave finite-difference FMC simulation.
* :func:`time_zero_correct`, :func:`bandpass`,
  :func:`common_offset_median_subtract`, :func:`cosine_gate_cube`,
  :func:`contact_gate_times`, :func:`pair_wall_times` -- the blind
  preprocessing chain (time-zero shift, zero-phase bandpass, common-offset
  median subtraction, cosine-ramp inspection gates).
* :func:`byot_tables` / :func:`operator_distance_amplitudes` -- the only
  places that touch the private ``Kirchhoff._traveltime_table`` API and the
  operator's internal ``eps = 1e-2 * (max dist)`` spreading regularisation
  (mirrors ``kirchhoff.py``'s ``dynamic`` path); if the fork ever changes
  either, only this module needs updating.
* :func:`eikonal_trav_table` -- numba fast-sweeping eikonal traveltime tables
  for layered models (``scikit-fmm`` has no Python 3.14 wheel), with a
  homogeneous-model :func:`eikonal_self_test` and a fast path for laterally
  invariant models (one padded solve + integer x-shifts instead of one solve
  per element).
* :func:`crosscheck_byot_operator` -- validates the fork's byot + raw-amp
  Kirchhoff code path against an independently assembled VStack of per-shot
  operators on a small asymmetric problem (the ancestor tutorials carried an
  equivalent guard that had been dropped).

Gate-margin conventions used by the studies (dominant period at 5 MHz is
0.2 us): the top gate opens ``T_PAD`` after the predicted direct-wave /
front-wall arrival and the bottom gate closes ``T_BW_MARGIN`` before the
predicted back-wall arrival, both with ``T_RAMP`` cosine ramps.  The contact
studies use ``T_PAD = 0.9 us`` (the in-solid direct wave along the array has
a long coda to clear) while the immersion study uses ``0.7 us`` (the water
front-wall echo is compact); ``T_BW_MARGIN = 0.4 us`` is ~2 dominant periods.
"""
from types import SimpleNamespace

import numpy as np
import torch
from numba import njit, prange
from scipy.signal import butter, sosfiltfilt

import deepwave
from deepwave import scalar

from pylops import VStack
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

# back-wall pre-gate margin: close the inspection gate ~2 dominant periods
# before the predicted wall arrival so the gate ramp never touches the echo
T_BW_MARGIN = 0.4e-6

# ---------------------------------------------------------------------------
# Shared case definitions
# ---------------------------------------------------------------------------
# Contact-style acquisition shared by the single-void, five-void and shoot-out
# studies: 40 mm x 30 mm aluminum plate (air backing below z_back), 32-element
# 1 mm-pitch array buried at z = 2 mm, 5 MHz, 11 us records.
CONTACT = SimpleNamespace(
    dx=0.5e-4, nx=800, nz=600,
    v_alu=6300.0, v_air=343.0, freq=5.0e6, z_back=0.024,
    n_el=32, pitch_cells=20, x0_cells=90, z_cells=40,
    dt=5.0e-9, nt=2200,
)

# The five-void placement recipe: the shoot-out reproduces the dataset of
# deepwave_fmc_kirchhoff_blind_multi.py from these parameters, so they must
# have exactly one definition.
FIVE_VOID_CASE = dict(
    seed=7, n_voids=5, min_sep=3.5e-3,
    x_range=(0.008, 0.032), z_range=(0.006, 0.021),
    d_range=(0.3e-3, 0.8e-3),
)


def make_random_voids(seed, n_voids, min_sep, x_range, z_range, d_range):
    """Rejection-sample ``n_voids`` circular voids ``(cx, cz, r)``, sorted by
    depth. The draw order (cx, cz, then diameter only on acceptance) is part
    of the recipe: changing it changes every seeded dataset."""
    rng = np.random.default_rng(seed)
    voids = []
    while len(voids) < n_voids:
        cx = rng.uniform(*x_range)
        cz = rng.uniform(*z_range)
        if all(np.hypot(cx - vx, cz - vz) >= min_sep for vx, vz, _ in voids):
            r = 0.5 * rng.uniform(*d_range)
            voids.append((cx, cz, r))
    voids.sort(key=lambda v: v[1])  # shallow to deep, for reporting
    return voids


def carve_voids(vel, xs, zs, voids, v_fill):
    """Set ``vel`` to ``v_fill`` inside each circular void (in place)."""
    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
    for cx, cz, r in voids:
        vel[(Xg - cx) ** 2 + (Zg - cz) ** 2 <= r**2] = v_fill
    return vel


# ---------------------------------------------------------------------------
# deepwave FMC simulation harness
# ---------------------------------------------------------------------------
def simulate_fmc(vel, dx, dt, nt, elem_ix, elem_iz, freq, peak_time,
                 device=None):
    """One deepwave scalar FMC simulation of ``vel`` (pylops (nx, nz) layout);
    every element fires in turn, all elements record.

    Returns the receiver data as ``(n_src, n_rec, nt)`` float64."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_el = len(elem_ix)
    src_amp = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .reshape(1, 1, -1)
        .repeat(n_el, 1, 1)
        .to(device)
    )
    src_loc = torch.tensor(
        np.stack([elem_iz, elem_ix], axis=-1)[:, None, :],
        dtype=torch.long, device=device,
    )
    rec_loc = torch.tensor(
        np.broadcast_to(
            np.stack([elem_iz, elem_ix], axis=-1), (n_el, n_el, 2)
        ).copy(),
        dtype=torch.long, device=device,
    )
    out = scalar(
        torch.tensor(vel.T.copy(), device=device),
        dx, dt,
        source_amplitudes=src_amp,
        source_locations=src_loc,
        receiver_locations=rec_loc,
        accuracy=4, pml_width=20, pml_freq=freq,
    )
    return out[-1].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Blind preprocessing chain
# ---------------------------------------------------------------------------
def time_zero_correct(fmc, peak_time, dt):
    """Advance the data by the known excitation delay (the pulse peaks
    ``peak_time`` after the electrical time zero), zero-padding the tail."""
    n0 = int(round(peak_time / dt))
    pad = np.zeros(fmc.shape[:-1] + (n0,))
    return np.concatenate([fmc[..., n0:], pad], axis=-1)


def bandpass(fmc, dt, f_lo=2.5e6, f_hi=7.5e6, order=4):
    """Zero-phase Butterworth bandpass around the transducer band."""
    sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=1.0 / dt,
                 output="sos")
    return sosfiltfilt(sos, fmc, axis=-1)


def common_offset_median_subtract(fmc, min_pairs=4):
    """Common-offset median subtraction: at fixed element offset ``k = j - i``
    every laterally invariant event (direct wave, wall echoes, multiples,
    reverberations) is identical for all pairs, so the median trace over pairs
    estimates exactly that clutter and none of the localised diffractions.
    Offsets with fewer than ``min_pairs`` pairs have no robust median and are
    zeroed (a tiny aperture loss)."""
    n_el = fmc.shape[0]
    out = fmc.copy()
    for k in range(-(n_el - 1), n_el):
        ii = np.arange(max(0, -k), min(n_el, n_el - k))
        if len(ii) < min_pairs:
            out[ii, ii + k, :] = 0.0
            continue
        out[ii, ii + k, :] -= np.median(fmc[ii, ii + k, :], axis=0)
    return out


def cosine_gate_cube(t, t_open, t_close, t_ramp):
    """``(n_el, n_el, nt)`` inspection gate from per-pair open/close times
    (``(n_el, n_el)`` arrays), with half-cosine ramps of length ``t_ramp``."""
    up = np.clip((t[None, None, :] - t_open[:, :, None]) / t_ramp, 0.0, 1.0)
    dn = np.clip((t_close[:, :, None] - t[None, None, :]) / t_ramp, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * up)) * (0.5 - 0.5 * np.cos(np.pi * dn))


def contact_gate_times(elem_x, elem_z, v, z_back, t_pad,
                       t_bw_margin=T_BW_MARGIN):
    """Per-pair gate times for the contact case, from geometry and velocity
    only: open after the in-solid direct wave, close before the back-wall
    reflection (two-way specular path)."""
    offs = np.abs(elem_x[:, None] - elem_x[None, :])
    t_open = offs / v + t_pad
    h_bw = 2.0 * (z_back - elem_z[0])  # two-way vertical path to the wall
    t_close = np.sqrt(h_bw**2 + offs**2) / v - t_bw_margin
    return t_open, t_close


def pair_wall_times(T3, iz):
    """Wall reflection time for every element pair from a traveltime cube
    ``T3`` of shape ``(nx, nz, n_el)``: ``min over x of T_i(x, z_wall) +
    T_j(x, z_wall)``."""
    A = T3[:, iz, :]  # (nx, n_el)
    return (A[:, :, None] + A[:, None, :]).min(axis=0)


# ---------------------------------------------------------------------------
# byot tables from the Kirchhoff operator
# ---------------------------------------------------------------------------
# NOTE: these two functions are deliberately the only places in the tutorials
# that call the private ``Kirchhoff._traveltime_table`` staticmethod and that
# replicate the operator's internal spreading regularisation
# (``epsdist = 1e-2`` in kirchhoff.py's dynamic branch). A public
# table-building API in the fork would supersede them.
def spreading_amplitudes(dist_srcs, dist_recs, eps_frac=1e-2):
    """2D geometrical-spreading weights ``1/sqrt(dist + eps)`` with the
    operator's own near-source regularisation recipe."""
    eps = eps_frac * (dist_srcs.max() + dist_recs.max())
    return 1.0 / np.sqrt(dist_srcs + eps), 1.0 / np.sqrt(dist_recs + eps)


def byot_tables(zm, xm, srcs, recs, v):
    """Analytic traveltime + spreading-amplitude tables for a homogeneous
    medium, generated by the Kirchhoff operator itself.

    Returns ``trav_srcs, trav_recs, amp_srcs, amp_recs`` ready for
    ``Kirchhoff(mode="byot", trav=(...), amp=(...), dynamic=False)``."""
    trav_srcs, trav_recs, dist_srcs, dist_recs, _, _ = (
        Kirchhoff._traveltime_table(zm, xm, srcs, recs, v, mode="analytic")
    )
    amp_srcs, amp_recs = spreading_amplitudes(dist_srcs, dist_recs)
    return trav_srcs, trav_recs, amp_srcs, amp_recs


def operator_distance_amplitudes(zm, xm, srcs, recs):
    """Spreading-amplitude tables only, for media where the traveltimes come
    from elsewhere (e.g. the eikonal solver): the operator's Euclidean
    distance tables are velocity-independent, so any velocity works."""
    _, _, dist_srcs, dist_recs, _, _ = Kirchhoff._traveltime_table(
        zm, xm, srcs, recs, 1.0, mode="analytic"
    )
    return spreading_amplitudes(dist_srcs, dist_recs)


# ---------------------------------------------------------------------------
# Numba fast-sweeping eikonal solver (layered / arbitrary 2D models)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _eikonal_fsm(slow, h, six, siz, max_sweeps=8, tol=0.0):
    """First-arrival traveltimes on a square grid via fast sweeping (Godunov
    upwind). Stops early once the largest per-iteration update falls below
    ``tol`` (seconds); ``tol=0`` runs all ``max_sweeps`` iterations."""
    nx, nz = slow.shape
    BIG = 1.0e30
    T = np.full((nx, nz), BIG)
    T[six, siz] = 0.0
    for _ in range(max_sweeps):
        maxdiff = 0.0
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
                            cand = 0.5 * (
                                ux + uz
                                + np.sqrt(2.0 * f * f - (ux - uz) ** 2)
                            )
                        if cand < T[i, j]:
                            if T[i, j] - cand > maxdiff:
                                maxdiff = T[i, j] - cand
                            T[i, j] = cand
        if maxdiff < tol:
            break
    return T


@njit(parallel=True, cache=True)
def _trav_table_persource(slow, h, ix, iz, tol):
    """One eikonal solve per source; table (nx*nz, n_pts), C-order raveled
    in (nx, nz) -- the same x-major layout the Kirchhoff operator uses."""
    nx, nz = slow.shape
    npts = ix.shape[0]
    out = np.empty((nx * nz, npts))
    for k in prange(npts):
        T = _eikonal_fsm(slow, h, ix[k], iz[k], 8, tol)
        out[:, k] = T.ravel()
    return out


def eikonal_trav_table(vel, h, six, siz):
    """Traveltime table ``(nx*nz, n_pts)`` on model ``vel`` (``(nx, nz)``,
    m/s) for sources at grid indices ``(six[k], siz[k])``.

    Fast path: when the model is laterally invariant and all sources share
    one depth, a single solve on an x-padded grid is shifted column-wise to
    every source position (~n_pts times cheaper, and with *less* lateral
    boundary truncation than per-source solves on the unpadded grid)."""
    six = np.asarray(six, dtype=np.int64)
    siz = np.asarray(siz, dtype=np.int64)
    slow = 1.0 / np.ascontiguousarray(vel, dtype=np.float64)
    nx, nz = vel.shape
    tol = 1e-3 * h * slow.min()

    if np.allclose(vel, vel[0:1, :]) and np.all(siz == siz[0]):
        smin, smax = int(six.min()), int(six.max())
        ref_col = smax  # reference source column in the padded grid
        w = nx - smin + smax  # so that i - six[k] + ref_col stays in [0, w-1]
        slow_pad = np.repeat(slow[0:1, :], w, axis=0)
        T_ref = _eikonal_fsm(slow_pad, h, ref_col, int(siz[0]), 8, tol)
        out = np.empty((nx * nz, len(six)))
        rows = np.arange(nx)
        for k in range(len(six)):
            out[:, k] = T_ref[rows - six[k] + ref_col, :].ravel()
        return out
    return _trav_table_persource(slow, h, six, siz, tol)


def eikonal_self_test(v0=6000.0, h=1.0e-4, n=201, tol=0.05):
    """Validate the fast-sweeping solver against the analytic homogeneous
    solution ``|x - x_src| / v0``. The asserted metric is the historic guard
    from the ancestor tutorials: max absolute error normalised by the largest
    traveltime in the domain (< ``tol``). The pointwise relative error is
    intrinsically a few percent along diagonals for a first-order Godunov
    scheme, so it is not asserted. Raises AssertionError on failure; guards
    against silent solver corruption (stale numba cache, kernel edits)."""
    slow = np.full((n, n), 1.0 / v0)
    T = _eikonal_fsm(slow, h, n // 2, 0, 8, 0.0)
    X, Z = np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing="ij")
    T_exact = np.hypot(X - (n // 2) * h, Z) / v0
    rel = np.abs(T - T_exact).max() / T_exact.max()
    assert rel < tol, f"eikonal self-test FAILED: norm max err {rel:.4f} >= {tol}"
    return rel


# ---------------------------------------------------------------------------
# Operator correctness guard
# ---------------------------------------------------------------------------
def crosscheck_byot_operator(rtol=1e-8, seed=0):
    """Cross-validate the fork's byot + raw-amplitude Kirchhoff path: a full
    multi-source FMC operator must equal an independently assembled VStack of
    per-source operators, in forward and adjoint, on a small deliberately
    asymmetric problem (ns != nr, distinct source/receiver depths -- so any
    source/receiver axis swap in the kernels is caught).

    Raises AssertionError on mismatch; returns (rel_forward, rel_adjoint)."""
    dxm, v0, dt, nt = 1.0e-4, 6000.0, 5.0e-9, 640
    nxm, nzm = 60, 50
    xm, zm = np.arange(nxm) * dxm, np.arange(nzm) * dxm
    t = np.arange(nt) * dt
    srcs = np.vstack((np.array([1.0e-3, 2.1e-3, 3.4e-3, 4.8e-3]),
                      np.full(4, 0.2e-3)))
    recs = np.vstack((np.array([0.7e-3, 1.6e-3, 2.5e-3, 3.6e-3, 4.4e-3,
                                5.3e-3]),
                      np.full(6, 0.5e-3)))
    ns, nr = srcs.shape[1], recs.shape[1]

    trav_s, trav_r, amp_s, amp_r = byot_tables(zm, xm, srcs, recs, v0)
    wav, _, wavc = ricker(t[:41], f0=5.0e6)

    def make_op(sl):
        return Kirchhoff(
            zm, xm, t, srcs[:, sl], recs, v0, wav, wavc,
            mode="byot",
            trav=(trav_s[:, sl], trav_r),
            amp=(amp_s[:, sl], amp_r),
            dynamic=False, engine="numba",
        )

    Op = make_op(slice(None))
    Vop = VStack([make_op(slice(i, i + 1)) for i in range(ns)])

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(nxm * nzm)
    d = rng.standard_normal(ns * nr * nt)
    y1, y2 = np.ravel(Op @ x), np.ravel(Vop @ x)
    m1, m2 = np.ravel(Op.H @ d), np.ravel(Vop.H @ d)
    rel_fwd = np.linalg.norm(y1 - y2) / (np.linalg.norm(y2) + 1e-30)
    rel_adj = np.linalg.norm(m1 - m2) / (np.linalg.norm(m2) + 1e-30)
    assert rel_fwd < rtol and rel_adj < rtol, (
        f"byot operator cross-check FAILED: forward rel {rel_fwd:.2e}, "
        f"adjoint rel {rel_adj:.2e} (tolerance {rtol:.0e})"
    )
    return rel_fwd, rel_adj
