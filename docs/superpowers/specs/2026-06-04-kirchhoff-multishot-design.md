# Kirchhoff multi-shot acquisition — design

Date: 2026-06-04
Operator: `pylops.waveeqprocessing.kirchhoff.Kirchhoff`

## Problem

The `Kirchhoff` demigration/migration operator currently assumes a full
Cartesian acquisition: every one of the `n_s` sources is recorded by every one
of the `n_r` receivers, producing `n_s * n_r` traces. Real seismic surveys are
organized as **shots** (independent acquisition sessions): each shot fires a
source and records only at a specific, shot-dependent subset of receivers. The
overall image is reconstructed from the contribution of all shots, but each
shot is otherwise independent. Packages such as Deepwave model acquisition this
way.

We want `Kirchhoff` to support per-shot receiver subsets while leaving the
existing full-Cartesian behaviour completely unchanged when the feature is not
used.

## Scope and decisions

- **Shot model:** one source per shot. The number of shots equals the number of
  sources (`n_shots == n_s`); shot `i` fires source `i` and records at an
  arbitrary subset of receivers. (Multiple-sources-per-shot is explicitly out of
  scope.)
- **Engines:** `numpy` and `numba` only. `engine='cuda'` combined with
  `shot_recs` raises a clear `NotImplementedError`. (CUDA shot-aware kernels are
  out of scope.)
- **External data layout:** padded `(n_shots, max_recs, nt)`, where
  `max_recs = max(len(shot_recs[i]))`. Inactive `(shot, slot)` positions are
  zero-filled in the forward output and ignored in the adjoint.
- **Backward compatibility:** when `shot_recs is None`, all code paths are
  identical to the current implementation (`dimsd = (n_s, n_r, nt)`).

## Public API

New optional parameter (already present in signature):

```python
shot_recs: list[NDArray] | None = None
```

- List of length `n_s`. `shot_recs[i]` is a 1-D integer array of receiver
  indices active for shot `i`, with values in `[0, n_r)`.
- `None` (default) → original full-Cartesian behaviour.

Validation (raise `ValueError` / `NotImplementedError` with clear messages):

- `len(shot_recs) != n_s`.
- any entry is not 1-D.
- any receiver index `< 0` or `>= n_r`.
- single-table `trav` (i.e. `mode='byot'` with non-tuple `trav`) combined with
  `shot_recs` → not supported (multishot requires separate src/rec tables).
- `engine='cuda'` combined with `shot_recs` → `NotImplementedError`.

New attributes:

- `nshots` — number of shots (`== n_s` when multishot).
- `max_recs` — largest per-shot receiver count.
- `shot_offsets` — `int32` array length `n_s + 1`; flat-trace offset of each
  shot (exposed so callers can slice per-shot if they reconstruct the flat form).

## Internal representation

Internally the operator works on a **flat** `(ntrace_total, nt)` trace buffer,
reusing the already-written shot-aware kernels
(`_travsrcrec_shots_kirch_matvec/rmatvec`,
`_ampsrcrec_shots_kirch_matvec/rmatvec`). The per-trace mapping arrays are:

- `_src_indices[itrace]` — source (== shot) index of trace `itrace`.
- `_rec_indices[itrace]` — receiver index of trace `itrace`.
- `_shot_offsets` — cumulative trace counts per shot.
- `ntrace_total = shot_offsets[-1]`.

`Convolve1D` (`self.cop`) is sized to `(ntrace_total, nt)` in the multishot case
(currently hardcoded to `ns * nr` — a bug for multishot). Convolution runs only
on the flat buffer, so no compute is spent on padding.

## Flat ↔ padded mapping

`max_recs = max(len(shot_recs[i]))`. For trace `itrace` belonging to shot `s`
at within-shot slot `k = itrace - shot_offsets[s]`, its padded flattened
position is:

```
_pad_index[itrace] = s * max_recs + k
```

`_pad_index` is precomputed once (length `ntrace_total`).

- **Scatter (flat → padded):** `padded_flat[_pad_index] = flat`. Positions not
  referenced by `_pad_index` remain zero (inactive slots).
- **Gather (padded → flat):** `flat = padded_flat[_pad_index]`.

Scatter is a 0/1 selection matrix `S`; gather is `S^T`. Because forward applies
scatter **last** and adjoint applies gather **first**, the two remain exact
transposes and `dottest` passes.

## Data flow

`_matvec(x)` (image → data), multishot:
1. `y_flat = zeros((ntrace_total, nt))`
2. run shot kernel (`_*_shots_kirch_matvec`) filling `y_flat`
3. `y_flat = cop._matvec(y_flat.ravel())`
4. scatter into `y_pad` of shape `(n_shots * max_recs, nt)`
5. return `y_pad.ravel()` (reshaped by harness to `(n_shots, max_recs, nt)`)

`_rmatvec(x)` (data → image), multishot:
1. reshape `x` to `(n_shots * max_recs, nt)`
2. gather → `x_flat = x[_pad_index]` of shape `(ntrace_total, nt)`
3. `x_flat = cop._rmatvec(x_flat.ravel()).reshape(ntrace_total, nt)`
4. `y = zeros(ni)`
5. run shot kernel (`_*_shots_kirch_rmatvec`) filling `y`
6. return `y`

The non-multishot path is unchanged.

## Dispatch

`_register_multiplications` adds a multishot branch chosen before the existing
ones:

- `engine='cuda'` and multishot → `NotImplementedError`.
- multishot + dynamic + travsrcrec → `_ampsrcrec_shots_kirch_*`.
- multishot + travsrcrec (kinematic) → `_travsrcrec_shots_kirch_*`.

numpy and numba both supported (numba via `jit(**numba_opts)` like the existing
kernels). The shot kernels take the per-trace mapping arrays
(`ntrace`, `src_indices`, `rec_indices`) instead of `ns, nr`.

## Testing

Add to `pytests/test_kirchhoff.py`:

- **dottest** for multishot across `mode` ∈ {analytic, eikonal, byot} and
  `dynamic` ∈ {False, True}, engines numpy and numba.
- **Correctness:** build a multishot operator and compare its forward/adjoint
  against the full-Cartesian operator with the inactive traces masked out — the
  active `(shot, rec)` traces must match the corresponding rows of the dense
  result; inactive padded slots must be exactly zero.
- **Edge case:** a degenerate `shot_recs` selecting all receivers for every
  shot must reproduce (up to padded reshape) the full-Cartesian result.
- **Validation errors:** wrong `len(shot_recs)`, out-of-range receiver index,
  non-1-D entry, single-table `trav` + `shot_recs`, `engine='cuda'` +
  `shot_recs`.

## Out of scope

- Multiple sources per shot.
- CUDA shot-aware kernels.
- 3-D aperture filtering (already a pre-existing limitation, unchanged).
