import os

import numpy as np

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

PAR = {
    "ny": 3,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nsy": 4,
    "nry": 3,
    "nsx": 6,
    "nrx": 2,
}

# Check if skfmm is available and by-pass tests using it otherwise. This is
# currently used in GPU CI not to run tests that require skfmm to compute the
# traveltime tables - should consider fixing it
try:
    import skfmm  # noqa: F401

    skfmm_enabled = True if backend == "numpy" else False
except ImportError:
    skfmm_enabled = False

v0 = 500
y = npp.arange(PAR["ny"]) * PAR["dy"]
x = npp.arange(PAR["nx"]) * PAR["dx"]
z = npp.arange(PAR["nz"]) * PAR["dz"]
t = npp.arange(PAR["nt"]) * PAR["dt"]

sy = npp.linspace(y.min(), y.max(), PAR["nsy"])
sx = npp.linspace(x.min(), x.max(), PAR["nsx"])
syy, sxx = npp.meshgrid(sy, sx, indexing="ij")
s2d = npp.vstack((sx, 2 * npp.ones(PAR["nsx"])))
s3d = npp.vstack((syy.ravel(), sxx.ravel(), 2 * npp.ones(PAR["nsx"] * PAR["nsy"])))

ry = npp.linspace(y.min(), y.max(), PAR["nry"])
rx = npp.linspace(x.min(), x.max(), PAR["nrx"])
ryy, rxx = npp.meshgrid(ry, rx, indexing="ij")
r2d = npp.vstack((rx, 2 * npp.ones(PAR["nrx"])))
r3d = npp.vstack((ryy.ravel(), rxx.ravel(), 2 * npp.ones(PAR["nrx"] * PAR["nry"])))

wav, _, wavc = ricker(t[:21], f0=40)

par1 = {"mode": "analytic", "dynamic": False}
par2 = {"mode": "eikonal", "dynamic": False}
par3 = {"mode": "byot", "dynamic": False}
par1d = {"mode": "analytic", "dynamic": True}
par2d = {"mode": "eikonal", "dynamic": True}
par3d = {"mode": "byot", "dynamic": True}


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_identify_geometry():
    """Identify geometry, check expected outputs"""
    # 2d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s2d, r2d)
    assert ndims == 2
    assert shiftdim == 0
    assert [1, 2] == [1, 2]
    assert list(dims) == [PAR["nx"], PAR["nz"]]
    assert ny == 1
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsx"]
    assert nr == PAR["nrx"]
    assert list(dsamp) == [dx, dz]
    assert list(origin) == [0, 0]

    # 3d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s3d, r3d, y=y)
    assert ndims == 3
    assert shiftdim == 1
    assert list(dims) == [PAR["ny"], PAR["nx"], PAR["nz"]]
    assert ny == PAR["ny"]
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsy"] * PAR["nsx"]
    assert nr == PAR["nry"] * PAR["nrx"]
    assert list(dsamp) == [dy, dx, dz]
    assert list(origin) == [0, 0, 0]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_traveltime_ana():
    """Check analytical traveltimes in homogenous medium for horizontal and
    vertical paths
    """
    src = np.array([100, 0])[:, np.newaxis]

    (
        trav_srcs_ana,
        trav_recs_ana,
        dist_srcs_ana,
        dist_recs_ana,
        _,
        _,
    ) = Kirchhoff._traveltime_table(
        np.arange(0, 200, 1), np.arange(0, 200, 1), src, src, v0, mode="analytic"
    )
    assert dist_srcs_ana[0, 0] + dist_recs_ana[0, 0] == 200
    assert trav_srcs_ana[0, 0] == 100 / v0
    assert trav_recs_ana[0, 0] == 100 / v0


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_traveltime_table():
    """Compare analytical and eikonal traveltimes in homogenous medium"""
    if skfmm_enabled:
        # 2d
        (
            trav_ana,
            trav_srcs_ana,
            trav_recs_ana,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s2d, r2d, v0, mode="analytic")

        (
            trav_eik,
            trav_srcs_eik,
            trav_recs_eik,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0 * np.ones((PAR["nx"], PAR["nz"])), mode="eikonal"
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_ana, decimal=2)
        assert_array_almost_equal(trav_ana, trav_eik, decimal=2)

        # 3d
        (
            trav_srcs_ana,
            trav_recs_ana,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(z, x, s3d, r3d, v0, y=y, mode="analytic")

        (
            trav_srcs_eik,
            trav_recs_eik,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._traveltime_table(
            z,
            x,
            s3d,
            r3d,
            v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"])),
            y=y,
            mode="eikonal",
        )

        assert_array_almost_equal(trav_srcs_ana, trav_srcs_eik, decimal=2)
        assert_array_almost_equal(trav_recs_ana, trav_recs_eik, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par1d), (par2d), (par3d)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff2d(par, dtype):
    """Dot-test for Kirchhoff operator"""
    if backend == "cupy" and par["mode"] == "byot":
        pytest.skip("cuda engine not available for single trav table")
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]), dtype=dtype)

    if par["mode"] == "byot":
        trav_srcs, trav_recs, _, _, _, _ = Kirchhoff._traveltime_table(
            z, x, s2d, r2d, v0, mode="analytic"
        )
        trav_srcs = trav_srcs.astype(dtype)
        trav_recs = trav_recs.astype(dtype)
        trav = trav_srcs.reshape(
            PAR["nx"] * PAR["nz"], PAR["nsx"], 1
        ) + trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
        trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])
        amp = None
    else:
        trav = None
        amp = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z.astype(dtype),
            x.astype(dtype),
            t.astype(dtype),
            s2d.astype(dtype),
            r2d.astype(dtype),
            vel if par["mode"] == "eikonal" else v0,
            np.asarray(wav.astype(dtype)),
            wavc,
            y=None,
            trav=trav,
            amp=amp,
            mode=par["mode"],
            engine="numpy" if backend == "numpy" else "cuda",
            dtype=dtype,
        )
        if par["mode"] == "byot":
            Dop.trav = np.asarray(Dop.trav)
        else:
            Dop.trav_srcs = np.asarray(Dop.trav_srcs)
            Dop.trav_recs = np.asarray(Dop.trav_recs)
        if par["mode"] == "dynamic":
            Dop.amp_srcs = np.asarray(Dop.amp_srcs)
            Dop.amp_recs = np.asarray(Dop.amp_recs)

        assert dottest(
            Dop,
            PAR["nsx"] * PAR["nrx"] * PAR["nt"],
            PAR["nz"] * PAR["nx"],
            backend=backend,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"]).astype(dtype)
        yy = Dop * xx
        xadj = Dop.H * yy
        assert yy.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kirchhoff3d(par, dtype):
    """Dot-test for Kirchhoff operator"""
    if backend == "cupy" and par["mode"] == "byot":
        pytest.skip("cuda engine not available for single trav table")
    vel = v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"]), dtype=dtype)

    if par["mode"] == "byot":
        trav_srcs, trav_recs, _, _, _, _ = Kirchhoff._traveltime_table(
            z, x, s3d, r3d, v0, y=y, mode="analytic"
        )
        trav_srcs = trav_srcs.astype(dtype)
        trav_recs = trav_recs.astype(dtype)
        trav = trav_srcs.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nsy"] * PAR["nsx"], 1
        ) + trav_recs.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"], 1, PAR["nry"] * PAR["nrx"]
        )
        trav = trav.reshape(
            PAR["ny"] * PAR["nx"] * PAR["nz"],
            PAR["nsy"] * PAR["nry"] * PAR["nsx"] * PAR["nrx"],
        )
    else:
        trav = None

    if skfmm_enabled or par["mode"] != "eikonal":
        Dop = Kirchhoff(
            z.astype(dtype),
            x.astype(dtype),
            t.astype(dtype),
            s3d.astype(dtype),
            r3d.astype(dtype),
            vel if par["mode"] == "eikonal" else v0,
            np.asarray(wav.astype(dtype)),
            wavc,
            y=y.astype(dtype),
            trav=trav,
            mode=par["mode"],
            engine="numpy" if backend == "numpy" else "cuda",
            dtype=dtype,
        )
        if par["mode"] == "byot":
            Dop.trav = np.asarray(Dop.trav)
        else:
            Dop.trav_srcs = np.asarray(Dop.trav_srcs)
            Dop.trav_recs = np.asarray(Dop.trav_recs)

        assert dottest(
            Dop,
            PAR["nsx"] * PAR["nrx"] * PAR["nsy"] * PAR["nry"] * PAR["nt"],
            PAR["nz"] * PAR["nx"] * PAR["ny"],
            backend=backend,
            rtol=1e-4 if dtype == np.float32 else 1e-6,
        )

        xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"]).astype(dtype)
        yy = Dop * xx
        xadj = Dop.H * yy
        assert yy.dtype == dtype
        assert xadj.dtype == dtype


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par1d),
    ],
)
def test_kirchhoff2d_trav_vs_travsrcrec(par):
    """Compare 2D Kirchhoff operator forward and adjoint when using trav (original behavior)
    or trav_src and trav_rec (new recommended behaviour)"""

    # new behaviour
    Dop = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        v0,
        wav,
        wavc,
        y=None,
        mode=par["mode"],
        dynamic=par["dynamic"],
        angleaperture=None,
        engine="numpy" if backend == "numpy" else "cuda",
    )
    Dop.trav_srcs = np.asarray(Dop.trav_srcs)
    Dop.trav_recs = np.asarray(Dop.trav_recs)
    if par["dynamic"]:
        Dop.amp_srcs = np.asarray(Dop.amp_srcs)
        Dop.amp_recs = np.asarray(Dop.amp_recs)

    # old behaviour
    trav = Dop.trav_srcs.reshape(
        PAR["nx"] * PAR["nz"], PAR["nsx"], 1
    ) + Dop.trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
    trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"])
    if par["dynamic"]:
        amp = (Dop.amp_srcs, Dop.amp_recs)

    D1op = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        v0,
        wav,
        wavc,
        y=None,
        trav=trav,
        amp=amp if par["dynamic"] else None,
        mode=par["mode"],
        dynamic=par["dynamic"],
        angleaperture=None,
        engine="numpy" if backend == "numpy" else "cuda",
    )
    D1op.trav_srcs = np.asarray(D1op.trav_srcs)
    D1op.trav_recs = np.asarray(D1op.trav_recs)
    if par["dynamic"]:
        D1op.amp_srcs = np.asarray(D1op.amp_srcs)
        D1op.amp_recs = np.asarray(D1op.amp_recs)

    # forward
    xx = np.random.normal(0, 1, PAR["nx"] * PAR["nz"])
    assert_array_almost_equal(Dop @ xx, D1op @ xx, decimal=2)

    # adjoint
    yy = np.random.normal(0, 1, PAR["nrx"] * PAR["nsx"] * PAR["nt"])
    assert_array_almost_equal(Dop.H @ yy, D1op.H @ yy, decimal=2)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
    ],
)
def test_kirchhoff3d_trav_vs_travsrcrec(par):
    """Compare 3D Kirchhoff operator forward and adjoint when using trav (original behavior)
    or trav_src and trav_rec (new recommended behaviour)"""

    # new behaviour
    Dop = Kirchhoff(
        z,
        x,
        t,
        s3d,
        r3d,
        v0,
        wav,
        wavc,
        y=y,
        mode=par["mode"],
        engine="numpy" if backend == "numpy" else "cuda",
    )
    Dop.trav_srcs = np.asarray(Dop.trav_srcs)
    Dop.trav_recs = np.asarray(Dop.trav_recs)
    if par["dynamic"]:
        Dop.amp_srcs = np.asarray(Dop.amp_srcs)
        Dop.amp_recs = np.asarray(Dop.amp_recs)

    # old behaviour
    trav = Dop.trav_srcs.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nsy"] * PAR["nsx"], 1
    ) + Dop.trav_recs.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"], 1, PAR["nry"] * PAR["nrx"]
    )
    trav = trav.reshape(
        PAR["ny"] * PAR["nx"] * PAR["nz"],
        PAR["nsy"] * PAR["nsx"] * PAR["nry"] * PAR["nrx"],
    )

    D1op = Kirchhoff(
        z,
        x,
        t,
        s3d,
        r3d,
        v0,
        wav,
        wavc,
        y=y,
        trav=trav,
        mode=par["mode"],
        engine="numpy" if backend == "numpy" else "cuda",
    )
    D1op.trav_srcs = np.asarray(D1op.trav_srcs)
    D1op.trav_recs = np.asarray(D1op.trav_recs)
    if par["dynamic"]:
        D1op.amp_srcs = np.asarray(D1op.amp_srcs)
        D1op.amp_recs = np.asarray(D1op.amp_recs)

    # forward
    xx = np.random.normal(0, 1, PAR["ny"] * PAR["nx"] * PAR["nz"])
    assert_array_almost_equal(Dop @ xx, D1op @ xx, decimal=2)

    # adjoint
    yy = np.random.normal(
        0, 1, PAR["nry"] * PAR["nrx"] * PAR["nsy"] * PAR["nsx"] * PAR["nt"]
    )
    assert_array_almost_equal(Dop.H @ yy, D1op.H @ yy, decimal=2)


def _shot_recs_example(nsx, nrx):
    """Per-shot receiver subsets with *uneven* sizes so the padded layout is
    genuinely exercised: even shots record at all receivers, odd shots drop the
    last one. With nrx>=2 this yields max_recs=nrx and shorter shots whose
    trailing padded slots must be zero-filled."""
    return [
        npp.arange(nrx, dtype=npp.int32)
        if i % 2 == 0
        else npp.arange(nrx - 1, dtype=npp.int32)
        for i in range(nsx)
    ]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par1d), (par2d)])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_kirchhoff_multishot_dottest(par, engine):
    """Dot-test for multishot Kirchhoff operator (analytic/eikonal, kinematic/dynamic).

    Parametrized over the numpy and numba engines (numba falls back to numpy
    with a warning if not installed)."""
    if par["mode"] == "eikonal" and not skfmm_enabled:
        pytest.skip("skfmm not available")
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]))
    shot_recs = _shot_recs_example(PAR["nsx"], PAR["nrx"])
    ntrace = sum(len(sr) for sr in shot_recs)
    max_recs = max(len(sr) for sr in shot_recs)

    Dop = Kirchhoff(
        z,
        x,
        t,
        s2d,
        r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav,
        wavc,
        y=None,
        mode=par["mode"],
        dynamic=par["dynamic"],
        shot_recs=shot_recs,
        engine=engine,
    )
    assert Dop.nsnr == ntrace
    assert Dop.dimsd == (PAR["nsx"], max_recs, PAR["nt"])
    assert dottest(
        Dop,
        PAR["nsx"] * max_recs * PAR["nt"],
        PAR["nz"] * PAR["nx"],
        backend=backend,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par1d)])
def test_kirchhoff_multishot_matches_dense(par):
    """Forward of multishot operator equals the full-Cartesian operator
    restricted to the active (shot, receiver) traces; inactive padded slots
    are exactly zero."""
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]))
    shot_recs = _shot_recs_example(PAR["nsx"], PAR["nrx"])
    max_recs = max(len(sr) for sr in shot_recs)

    common = dict(
        mode=par["mode"],
        dynamic=par["dynamic"],
        engine="numpy",
    )
    Dfull = Kirchhoff(
        z, x, t, s2d, r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav, wavc, y=None, **common,
    )
    Dshot = Kirchhoff(
        z, x, t, s2d, r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav, wavc, y=None, shot_recs=shot_recs, **common,
    )

    m = npp.random.normal(0, 1, PAR["nx"] * PAR["nz"])
    yfull = (Dfull * m).reshape(PAR["nsx"], PAR["nrx"], PAR["nt"])
    yshot = (Dshot * m).reshape(PAR["nsx"], max_recs, PAR["nt"])

    for ishot, sr in enumerate(shot_recs):
        for slot, irec in enumerate(sr):
            assert_array_almost_equal(
                yshot[ishot, slot], yfull[ishot, irec], decimal=10
            )
        # padded (inactive) slots must be exactly zero
        for slot in range(len(sr), max_recs):
            assert np.all(yshot[ishot, slot] == 0)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_kirchhoff_multishot_validation():
    """shot_recs validation and unsupported-combination errors."""
    base = dict(mode="analytic", engine="numpy")

    # wrong length
    with pytest.raises(ValueError, match="must equal the number of sources"):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=[npp.array([0])], **base,
        )

    # out-of-range receiver index
    bad = [npp.array([0]) for _ in range(PAR["nsx"])]
    bad[0] = npp.array([PAR["nrx"]])  # == nr, out of range
    with pytest.raises(ValueError, match="outside"):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=bad, **base,
        )

    # non-1-D entry
    bad2 = [npp.array([0]) for _ in range(PAR["nsx"])]
    bad2[0] = npp.zeros((1, 2), dtype=npp.int32)
    with pytest.raises(ValueError, match="must be a 1-D array"):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=bad2, **base,
        )

    # single-table trav + shot_recs not supported
    good = _shot_recs_example(PAR["nsx"], PAR["nrx"])
    trav_srcs, trav_recs, _, _, _, _ = Kirchhoff._traveltime_table(
        z, x, s2d, r2d, v0, mode="analytic"
    )
    trav_single = trav_srcs.reshape(
        PAR["nx"] * PAR["nz"], PAR["nsx"], 1
    ) + trav_recs.reshape(PAR["nx"] * PAR["nz"], 1, PAR["nrx"])
    trav_single = trav_single.reshape(
        PAR["nx"] * PAR["nz"], PAR["nsx"] * PAR["nrx"]
    )
    with pytest.raises(ValueError, match="separate traveltime tables"):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            mode="byot", trav=trav_single, shot_recs=good, engine="numpy",
        )

    # cuda + shot_recs not supported
    with pytest.raises(NotImplementedError, match="cuda"):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=good, mode="analytic", engine="cuda",
        )
