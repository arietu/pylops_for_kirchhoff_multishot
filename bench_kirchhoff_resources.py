"""Benchmark computational resources: multi-shot vs VStack Kirchhoff (FMC)."""
import gc
import time
import tracemalloc

import numpy as np

import pylops
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

np.random.seed(0)

# --- same FMC setup as the tutorial -----------------------------------------
nx, nz = 101, 101
dx, dz = 4.0, 4.0
x, z = np.arange(nx) * dx, np.arange(nz) * dz
v0 = 1000.0

refl = np.zeros((nx, nz))
for ix, iz in [(30, 35), (50, 60), (72, 38), (40, 80), (62, 82)]:
    refl[ix - 1 : ix + 1, iz - 1 : iz + 1] = 1.0
m = refl.ravel()

narr = 64
ax_ = np.linspace(10 * dx, (nx - 10) * dx, narr)
az = np.full(narr, dz)
srcs = np.vstack((ax_, az))
recs = np.vstack((ax_, az))
ns, nr = srcs.shape[1], recs.shape[1]

nt = 500
dt = 0.004
t = np.arange(nt) * dt
wav, _, wavc = ricker(t[:41], f0=20)


def build_ms():
    return Kirchhoff(
        z, x, t, srcs, recs, v0, wav, wavc, mode="analytic", dynamic=False,
        shot_recs=[np.arange(nr) for _ in range(ns)], engine="numba",
    )


def build_vs():
    ops = [
        Kirchhoff(
            z, x, t, srcs[:, i : i + 1], recs, v0, wav, wavc,
            mode="analytic", dynamic=False, engine="numba",
        )
        for i in range(ns)
    ]
    return pylops.VStack(ops)


def measure(label, build):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    Op = build()
    t_build = time.perf_counter() - t0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # warm-up (triggers numba JIT compile; not timed)
    d = Op @ m
    _ = Op.H @ d

    # timed forward
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        d = Op @ m
    t_fwd = (time.perf_counter() - t0) / reps

    # timed adjoint
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = Op.H @ d
    t_adj = (time.perf_counter() - t0) / reps

    # traveltime-table footprint stored on the operator(s)
    def trav_bytes(op):
        return sum(
            getattr(op, a).nbytes
            for a in ("trav_srcs", "trav_recs", "trav",
                      "amp_srcs", "amp_recs", "amp")
            if isinstance(getattr(op, a, None), np.ndarray)
        )

    if isinstance(Op, pylops.VStack):
        tt = sum(trav_bytes(o) for o in Op.ops)
        n_blocks = len(Op.ops)
    else:
        tt = trav_bytes(Op)
        n_blocks = 1

    print(f"\n=== {label} ===")
    print(f"  operators (numba kernels)        : {n_blocks}")
    print(f"  build time                       : {t_build:8.3f} s")
    print(f"  build peak alloc (tracemalloc)   : {peak_build/1e6:8.1f} MB")
    print(f"  traveltime tables held in memory : {tt/1e6:8.1f} MB")
    print(f"  forward  (avg of {reps})              : {t_fwd:8.3f} s")
    print(f"  adjoint  (avg of {reps})              : {t_adj:8.3f} s")
    return Op


print(f"FMC geometry: ns={ns} shots, nr={nr} receivers, nt={nt} samples, "
      f"model {nx}x{nz}")
print(f"data cube: {ns}x{nr}x{nt} = {ns*nr*nt:,} samples "
      f"({ns*nr*nt*8/1e6:.1f} MB float64)")

measure("multi-shot (single fused operator)", build_ms)
measure(f"VStack ({ns} per-shot operators)", build_vs)
