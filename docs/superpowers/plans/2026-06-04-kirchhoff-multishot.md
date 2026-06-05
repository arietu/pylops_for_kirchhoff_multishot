# Kirchhoff Multi-Shot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the half-built `shot_recs` multi-shot support in `Kirchhoff` into the dispatch and matvec/rmatvec paths so per-shot receiver subsets actually work, with a padded `(n_shots, max_recs, nt)` data layout.

**Architecture:** Internally the operator keeps working on a flat `(ntrace_total, nt)` trace buffer using the existing `_*_shots_kirch_*` numba/numpy kernels. A precomputed `_pad_index` scatters the flat buffer into the external padded 3-D layout (forward) and gathers it back (adjoint); scatter-last / gather-first keeps forward and adjoint exact transposes. When `shot_recs is None`, all paths are unchanged.

**Tech Stack:** Python, NumPy, Numba (optional), PyLops `LinearOperator`, pytest.

---

## Context the engineer needs

- File under change: `pylops/waveeqprocessing/kirchhoff.py`.
- Tests: `pytests/test_kirchhoff.py` (run with `python -m pytest pytests/test_kirchhoff.py -v` from repo root; on Windows use the same).
- The `shot_recs` parameter, validation block, the flat mapping arrays
  (`self._multishot`, `self._src_indices`, `self._rec_indices`,
  `self._shot_offsets`, `ntrace_total`), and the four kernels
  (`_travsrcrec_shots_kirch_matvec`, `_travsrcrec_shots_kirch_rmatvec`,
  `_ampsrcrec_shots_kirch_matvec`, `_ampsrcrec_shots_kirch_rmatvec`) **already
  exist** in the file. Do not rewrite them.
- The kernels take, in order, the per-trace mapping `(..., ntrace, src_indices, rec_indices)`
  and write/read a flat `y`/`x` of shape `(ntrace, nt)`.
- `self.nsnr` is already set to `ntrace_total` for the multishot case.
- The `@reshaped` decorator reshapes the input of `_matvec` to `self.dims` and
  the input of `_rmatvec` to `self.dimsd`, and `.ravel()`s the returned array.
- The shot kernels reference `np` internally; under numba they are jitted, under
  numpy they run as plain Python — both already supported by the existing kernel
  bodies. No kernel edits are required.

What is currently **broken / missing** (this plan fixes it):
1. `Convolve1D` (`self.cop`) is hardcoded to `ns * nr` rows — wrong for multishot.
2. `dimsd` for multishot is flat `(ntrace_total, nt)` — must become padded `(n_shots, max_recs, nt)`.
3. No `max_recs` / `_pad_index` / `nshots` / `shot_offsets` attributes.
4. `_register_multiplications` never dispatches to the `_shots_` kernels and does not guard `engine='cuda'` + multishot.
5. `_matvec` / `_rmatvec` have no multishot branch (scatter / gather).

---

## Task 1: Add multishot geometry attributes (max_recs, pad index, nshots)

**Files:**
- Modify: `pylops/waveeqprocessing/kirchhoff.py` (the `if self._multishot:` setup block in `__init__`, just after `ntrace_total = int(self._shot_offsets[-1])`)

- [ ] **Step 1: Add attributes after `ntrace_total` is computed**

In `__init__`, locate this block:

```python
            self._shot_offsets = np.zeros(ns + 1, dtype=np.int32)
            for ishot, sr in enumerate(shot_recs):
                self._shot_offsets[ishot + 1] = self._shot_offsets[ishot] + len(sr)
            ntrace_total = int(self._shot_offsets[-1])
        else:
            ntrace_total = ns * nr
```

Replace it with:

```python
            self._shot_offsets = np.zeros(ns + 1, dtype=np.int32)
            for ishot, sr in enumerate(shot_recs):
                self._shot_offsets[ishot + 1] = self._shot_offsets[ishot] + len(sr)
            ntrace_total = int(self._shot_offsets[-1])
            # padded-layout bookkeeping
            self.nshots = ns
            self.max_recs = int(max(len(sr) for sr in shot_recs))
            self.shot_offsets = self._shot_offsets
            # map each flat trace to its position in the flattened padded
            # (n_shots, max_recs) grid: shot * max_recs + within-shot slot
            slots = np.concatenate(
                [np.arange(len(sr), dtype=np.int64) for sr in shot_recs]
            )
            self._pad_index = (
                self._src_indices.astype(np.int64) * self.max_recs + slots
            )
        else:
            ntrace_total = ns * nr
            self.nshots = ns
            self.max_recs = nr
```

(Setting `self.nshots`/`self.max_recs` in the non-multishot branch keeps the
attributes always defined; they are only *used* in the multishot branch.)

- [ ] **Step 2: Verify the file still imports**

Run: `python -c "import pylops.waveeqprocessing.kirchhoff"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add pylops/waveeqprocessing/kirchhoff.py
git commit -m "feat(kirchhoff): add multishot pad-index and shot bookkeeping"
```

---

## Task 2: Fix Convolve1D sizing and padded dimsd

**Files:**
- Modify: `pylops/waveeqprocessing/kirchhoff.py` (`self.cop = Convolve1D(...)` and the `dimsd` block in `__init__`)

- [ ] **Step 1: Size `cop` to the flat trace count**

Locate:

```python
        self.cop = Convolve1D(
            (ns * nr, self.nt), h=self.wav, offset=wavcenter, axis=1, dtype=dtype
        )
```

Replace `ns * nr` with `ntrace_total` (equals `ns * nr` when not multishot, so
behaviour is identical in the default case):

```python
        self.cop = Convolve1D(
            (ntrace_total, self.nt), h=self.wav, offset=wavcenter, axis=1, dtype=dtype
        )
```

- [ ] **Step 2: Make multishot `dimsd` padded**

Locate:

```python
        if self._multishot:
            dimsd = (ntrace_total, self.nt)
        else:
            dimsd = (ns, nr, self.nt)
```

Replace with:

```python
        if self._multishot:
            dimsd = (self.nshots, self.max_recs, self.nt)
        else:
            dimsd = (ns, nr, self.nt)
```

- [ ] **Step 3: Verify import still works**

Run: `python -c "import pylops.waveeqprocessing.kirchhoff"`
Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add pylops/waveeqprocessing/kirchhoff.py
git commit -m "fix(kirchhoff): size cop to ntrace_total and use padded multishot dimsd"
```

---

## Task 3: Dispatch shot kernels and guard cuda in _register_multiplications

**Files:**
- Modify: `pylops/waveeqprocessing/kirchhoff.py` (`_register_multiplications`)

- [ ] **Step 1: Add cuda guard at the top of the method**

Locate the start of `_register_multiplications`:

```python
    def _register_multiplications(self, engine: Tengine_nnc) -> None:
        if engine not in ["numpy", "numba", "cuda"]:
            msg = f"engine must be numpy or numba or cuda, got {engine}"
            raise ValueError(msg)
```

Insert immediately after the `raise ValueError(msg)` line (still inside the
method, before the `if engine == "numba"` block):

```python
        if self._multishot and engine == "cuda":
            msg = "engine='cuda' is not supported together with shot_recs; use engine='numpy' or 'numba'"
            raise NotImplementedError(msg)
```

- [ ] **Step 2: Dispatch shot kernels in the numba branch**

Locate the numba branch:

```python
        if engine == "numba" and jit_message is None:
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,
            if self.dynamic and self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._ampsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._ampsrcrec_kirch_rmatvec)
            elif self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._travsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._travsrcrec_kirch_rmatvec)
            elif not self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._trav_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._trav_kirch_rmatvec)
```

Replace the inner `if/elif` chain so the multishot cases come first:

```python
        if engine == "numba" and jit_message is None:
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,
            if self._multishot and self.dynamic:
                self._kirch_matvec = jit(**numba_opts)(
                    self._ampsrcrec_shots_kirch_matvec
                )
                self._kirch_rmatvec = jit(**numba_opts)(
                    self._ampsrcrec_shots_kirch_rmatvec
                )
            elif self._multishot:
                self._kirch_matvec = jit(**numba_opts)(
                    self._travsrcrec_shots_kirch_matvec
                )
                self._kirch_rmatvec = jit(**numba_opts)(
                    self._travsrcrec_shots_kirch_rmatvec
                )
            elif self.dynamic and self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._ampsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._ampsrcrec_kirch_rmatvec)
            elif self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._travsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._travsrcrec_kirch_rmatvec)
            elif not self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._trav_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._trav_kirch_rmatvec)
```

- [ ] **Step 3: Dispatch shot kernels in the numpy (else) branch**

Locate the final `else` branch:

```python
        else:
            if engine == "numba" and jit_message is not None:
                logger.warning(jit_message)
            if self.dynamic and self.travsrcrec:
                self._kirch_matvec = self._ampsrcrec_kirch_matvec
                self._kirch_rmatvec = self._ampsrcrec_kirch_rmatvec
            elif self.travsrcrec:
                self._kirch_matvec = self._travsrcrec_kirch_matvec
                self._kirch_rmatvec = self._travsrcrec_kirch_rmatvec
            elif not self.travsrcrec:
                self._kirch_matvec = self._trav_kirch_matvec
                self._kirch_rmatvec = self._trav_kirch_rmatvec
```

Replace with (multishot cases first):

```python
        else:
            if engine == "numba" and jit_message is not None:
                logger.warning(jit_message)
            if self._multishot and self.dynamic:
                self._kirch_matvec = self._ampsrcrec_shots_kirch_matvec
                self._kirch_rmatvec = self._ampsrcrec_shots_kirch_rmatvec
            elif self._multishot:
                self._kirch_matvec = self._travsrcrec_shots_kirch_matvec
                self._kirch_rmatvec = self._travsrcrec_shots_kirch_rmatvec
            elif self.dynamic and self.travsrcrec:
                self._kirch_matvec = self._ampsrcrec_kirch_matvec
                self._kirch_rmatvec = self._ampsrcrec_kirch_rmatvec
            elif self.travsrcrec:
                self._kirch_matvec = self._travsrcrec_kirch_matvec
                self._kirch_rmatvec = self._travsrcrec_kirch_rmatvec
            elif not self.travsrcrec:
                self._kirch_matvec = self._trav_kirch_matvec
                self._kirch_rmatvec = self._trav_kirch_rmatvec
```

(Note: the `elif engine == "cuda"` branch in the middle is left unchanged; the
cuda guard in Step 1 already prevents reaching it when multishot.)

- [ ] **Step 4: Verify import and that a cuda+multishot operator raises**

Run:
```bash
python -c "import pylops.waveeqprocessing.kirchhoff as k; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pylops/waveeqprocessing/kirchhoff.py
git commit -m "feat(kirchhoff): dispatch shot kernels and guard cuda+multishot"
```

---

## Task 4: Multishot branch in _matvec (scatter)

**Files:**
- Modify: `pylops/waveeqprocessing/kirchhoff.py` (`_matvec`)

- [ ] **Step 1: Add the multishot branch**

Locate the current `_matvec`:

```python
    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros((self.nsnr, self.nt), dtype=self.dtype)
        if self.dynamic and self.travsrcrec:
```

Insert a multishot branch right after `y = ncp.zeros(...)` and before the
existing `if self.dynamic and self.travsrcrec:` so the flat buffer is filled by
the shot kernels, then scattered into the padded layout. Change the head of the
method to:

```python
    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros((self.nsnr, self.nt), dtype=self.dtype)
        if self._multishot:
            if self.dynamic:
                inputs = (
                    x.ravel(),
                    y,
                    self.nt,
                    self.ni,
                    self.dt,
                    self.vel,
                    self.trav_srcs,
                    self.trav_recs,
                    self.amp_srcs,
                    self.amp_recs,
                    self.aperture[0],
                    self.aperture[1],
                    self.aperturetap,
                    self.nz,
                    self.six,
                    self.rix,
                    self.angleaperture[0],
                    self.angleaperture[1],
                    self.angle_srcs,
                    self.angle_recs,
                    self.nsnr,
                    self._src_indices,
                    self._rec_indices,
                )
            else:
                inputs = (
                    x.ravel(),
                    y,
                    self.nt,
                    self.ni,
                    self.dt,
                    self.trav_srcs,
                    self.trav_recs,
                    self.nsnr,
                    self._src_indices,
                    self._rec_indices,
                )
            y = self._kirch_matvec(*inputs)
            y = self.cop._matvec(y.ravel()).reshape(self.nsnr, self.nt)
            ypad = ncp.zeros(
                (self.nshots * self.max_recs, self.nt), dtype=self.dtype
            )
            ypad[self._pad_index] = y
            return ypad
        if self.dynamic and self.travsrcrec:
```

The rest of `_matvec` (the existing `if self.dynamic and self.travsrcrec` /
`elif` chain and the final `y = self._kirch_matvec(...)` / `cop` / `return`)
stays exactly as-is.

- [ ] **Step 2: Verify import**

Run: `python -c "import pylops.waveeqprocessing.kirchhoff"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add pylops/waveeqprocessing/kirchhoff.py
git commit -m "feat(kirchhoff): multishot forward with scatter to padded layout"
```

---

## Task 5: Multishot branch in _rmatvec (gather)

**Files:**
- Modify: `pylops/waveeqprocessing/kirchhoff.py` (`_rmatvec`)

- [ ] **Step 1: Add the multishot branch**

Locate the current `_rmatvec`:

```python
    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nsnr, self.nt)
        y = ncp.zeros(self.ni, dtype=self.dtype)
        if self.dynamic and self.travsrcrec:
```

Replace that head with a version that, when multishot, gathers the padded input
into the flat buffer *before* applying `cop`:

```python
    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self._multishot:
            x = x.reshape(self.nshots * self.max_recs, self.nt)[self._pad_index]
            x = self.cop._rmatvec(x.ravel()).reshape(self.nsnr, self.nt)
            y = ncp.zeros(self.ni, dtype=self.dtype)
            if self.dynamic:
                inputs = (
                    x,
                    y,
                    self.nt,
                    self.ni,
                    self.dt,
                    self.vel,
                    self.trav_srcs,
                    self.trav_recs,
                    self.amp_srcs,
                    self.amp_recs,
                    self.aperture[0],
                    self.aperture[1],
                    self.aperturetap,
                    self.nz,
                    self.six,
                    self.rix,
                    self.angleaperture[0],
                    self.angleaperture[1],
                    self.angle_srcs,
                    self.angle_recs,
                    self.nsnr,
                    self._src_indices,
                    self._rec_indices,
                )
            else:
                inputs = (
                    x,
                    y,
                    self.nt,
                    self.ni,
                    self.dt,
                    self.trav_srcs,
                    self.trav_recs,
                    self.nsnr,
                    self._src_indices,
                    self._rec_indices,
                )
            y = self._kirch_rmatvec(*inputs)
            return y
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nsnr, self.nt)
        y = ncp.zeros(self.ni, dtype=self.dtype)
        if self.dynamic and self.travsrcrec:
```

The rest of `_rmatvec` (the existing `if/elif` chain and final
`y = self._kirch_rmatvec(...)` / `return y`) stays exactly as-is.

- [ ] **Step 2: Verify import**

Run: `python -c "import pylops.waveeqprocessing.kirchhoff"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add pylops/waveeqprocessing/kirchhoff.py
git commit -m "feat(kirchhoff): multishot adjoint with gather from padded layout"
```

---

## Task 6: Dot-test for multishot (analytic + eikonal, kinematic + dynamic)

**Files:**
- Test: `pytests/test_kirchhoff.py` (append new test at end of file)

- [ ] **Step 1: Write the failing test**

Append to `pytests/test_kirchhoff.py`:

```python
def _shot_recs_example(nsx, nrx):
    """Per-shot receiver subsets: each shot drops one (rotating) receiver,
    so shot sizes are uneven and exercise the padded layout."""
    return [
        npp.array([j for j in range(nrx) if j != (i % nrx)], dtype=npp.int32)
        for i in range(nsx)
    ]


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2), (par1d), (par2d)])
def test_kirchhoff_multishot_dottest(par):
    """Dot-test for multishot Kirchhoff operator (analytic/eikonal, kinematic/dynamic)."""
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
        engine="numpy",
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
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `python -m pytest pytests/test_kirchhoff.py::test_kirchhoff_multishot_dottest -v`
Expected: PASS for all 4 parametrizations (eikonal ones may be skipped if skfmm
missing). If `numba` is the active engine elsewhere this still uses
`engine="numpy"`.

- [ ] **Step 3: Commit**

```bash
git add pytests/test_kirchhoff.py
git commit -m "test(kirchhoff): dot-test for multishot operator"
```

---

## Task 7: Correctness test vs. masked full-Cartesian operator

**Files:**
- Test: `pytests/test_kirchhoff.py` (append)

- [ ] **Step 1: Write the test**

Append to `pytests/test_kirchhoff.py`:

```python
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

    m = np.random.normal(0, 1, PAR["nx"] * PAR["nz"])
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
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest pytests/test_kirchhoff.py::test_kirchhoff_multishot_matches_dense -v`
Expected: PASS for both `par1` and `par1d`.

- [ ] **Step 3: Commit**

```bash
git add pytests/test_kirchhoff.py
git commit -m "test(kirchhoff): multishot forward matches masked dense operator"
```

---

## Task 8: Validation-error tests

**Files:**
- Test: `pytests/test_kirchhoff.py` (append)

- [ ] **Step 1: Write the tests**

Append to `pytests/test_kirchhoff.py`:

```python
@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
def test_kirchhoff_multishot_validation():
    """shot_recs validation and unsupported-combination errors."""
    base = dict(mode="analytic", engine="numpy")

    # wrong length
    with pytest.raises(ValueError):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=[npp.array([0])], **base,
        )

    # out-of-range receiver index
    bad = [npp.array([0]) for _ in range(PAR["nsx"])]
    bad[0] = npp.array([PAR["nrx"]])  # == nr, out of range
    with pytest.raises(ValueError):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=bad, **base,
        )

    # non-1-D entry
    bad2 = [npp.array([0]) for _ in range(PAR["nsx"])]
    bad2[0] = npp.zeros((1, 2), dtype=npp.int32)
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            mode="byot", trav=trav_single, shot_recs=good, engine="numpy",
        )

    # cuda + shot_recs not supported
    with pytest.raises(NotImplementedError):
        Kirchhoff(
            z, x, t, s2d, r2d, v0, wav, wavc, y=None,
            shot_recs=good, mode="analytic", engine="cuda",
        )
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest pytests/test_kirchhoff.py::test_kirchhoff_multishot_validation -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pytests/test_kirchhoff.py
git commit -m "test(kirchhoff): multishot validation and unsupported-combination errors"
```

---

## Task 9: Full suite + numba smoke check

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full kirchhoff test module**

Run: `python -m pytest pytests/test_kirchhoff.py -v`
Expected: all tests pass (eikonal-dependent ones may skip if skfmm missing).

- [ ] **Step 2: Smoke-check the numba engine path for multishot (if numba installed)**

Run:
```bash
python -c "
import numpy as np
from numpy.testing import assert_array_almost_equal
from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
nx, nz, nt = 12, 20, 50
x = np.arange(nx)*1.0; z = np.arange(nz)*2.0; t = np.arange(nt)*0.004
s = np.vstack((np.linspace(0, nx-1, 6), 2*np.ones(6)))
r = np.vstack((np.linspace(0, nx-1, 4), 2*np.ones(4)))
wav, _, wavc = ricker(t[:21], f0=40)
shot_recs = [np.array([j for j in range(4) if j != i % 4]) for i in range(6)]
Op = Kirchhoff(z, x, t, s, r, 500.0, wav, wavc, mode='analytic', shot_recs=shot_recs, engine='numba')
ntr = sum(len(sr) for sr in shot_recs); mr = max(len(sr) for sr in shot_recs)
print('numba multishot dottest:', dottest(Op, 6*mr*nt, nx*nz, rtol=1e-6))
"
```
Expected: prints `numba multishot dottest: True`. If numba is not installed the
operator falls back to numpy with a warning and the dot-test still prints `True`.

- [ ] **Step 3: Final commit (if any stray changes)**

```bash
git status
# if clean, nothing to do
```

---

## Self-Review notes

- **Spec coverage:** API/validation → Tasks 1 & 8; flat↔padded mapping → Task 1; cop sizing + padded dimsd → Task 2; dispatch + cuda guard → Task 3; forward scatter → Task 4; adjoint gather → Task 5; dottest across modes/dynamic → Task 6; correctness vs dense + zero padding → Task 7; degenerate "all receivers" case is implicitly covered by the dense-match test when a shot keeps all but one receiver (full equivalence per active trace); validation errors → Task 8; numba path → Task 9.
- **Type/name consistency:** attribute names `nshots`, `max_recs`, `shot_offsets`, `_pad_index`, `_src_indices`, `_rec_indices`, `nsnr` used consistently across tasks and match the existing code. Kernel argument order matches the existing kernel signatures (`..., ntrace, src_indices, rec_indices`).
- **No placeholders:** every code step contains full code.
