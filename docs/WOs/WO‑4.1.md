## WO-4.1 — `05_laws/atoms.py`: Coordinates & distances (A1–A7)

**Goal:** For each canonical output grid, compute the “A” atoms exactly per spec:

* `H, W`
* `r, c`
* `r + c`, `r - c`
* `d_top, d_bottom, d_left, d_right` (reused from scaffold)
* `midrow`, `midcol`
* Mod classes `(r mod m, c mod m)` for the exact `m` set in spec
* Block coords `(r // b, c // b)` and remainders for exact `b` set in spec

These atoms are **derivations**, not detectors, and are computed in canonical coords.

---

### 0. Anchors to read before coding

Implementer + reviewer must re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * §5.1 **Atom universe (A. Scaffold & coords)**:

     * Exact definitions of `(H,W)`, `(r,c)`, `r±c`, midrow/midcol,
     * Mod classes set:
       `m ∈ {2,…,min(6,max(H,W))} ∪ divisors(H) ∪ divisors(W)`
     * Block sizes set:
       `b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W)`.
   * §4 **Stage F**: confirms `d_top/bottom/left/right`, midrow/midcol are computed in scaffold and aggregated.

2. **` @docs/anchors/01_STAGES.md `**

   * Stage **03_scaffold** (WHERE) and **05_laws** (WHAT):

     * Scaffold produces frame/distances,
     * Laws consume scaffold + atoms, no re-doing BFS.

3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * Mapping for **scaffold & laws**:

     * Atoms are pure “seeing” (free sector), no paid bits,
     * No minted structure: everything is canonical, grid-aware.

We stick to these, no reinterpretation.

---

### 1. Libraries to use (and **only** these)

For A-atoms we don’t need heavy image libs; **NumPy** is enough. No custom algorithms.

Use:

```python
import numpy as np
from typing import Dict, Any
```

**Do not** implement any fancy traversal or graph stuff here. Distances already come from Stage F.

---

### 2. Input / output contracts

#### 2.1 Where this lives

Create / extend:

```bash
05_laws/
  atoms.py   # A, B, C, D, E, F, G atoms live here
  step.py    # calls atoms.* for train_out + test canvas
```

This WO only covers the A-atoms; later WOs will add B–G.

#### 2.2 Input to A-atoms

`05_laws/step.py` will, for each **output grid** (train_out and test canvas), call:

```python
from 05_laws.atoms import compute_A_atoms

A_atoms = compute_A_atoms(
    H=H_out,
    W=W_out,
    scaffold_info=scaffold_for_this_grid,
)
```

Where:

* `H, W`: ints from `grid.shape` in canonical coords.
* `scaffold_info` is a **per-output** dict from Stage F (03_scaffold), e.g.:

  ```python
  scaffold["per_output"][i] = {
      "d_top": np.ndarray[(H,W)],     # int
      "d_bottom": np.ndarray[(H,W)],
      "d_left": np.ndarray[(H,W)],
      "d_right": np.ndarray[(H,W)],
      "has_midrow": bool,
      "has_midcol": bool,
      # (other F atoms, ignore here)
  }
  ```

For the **test canvas**, `scaffold_info` may not exist yet (we only built scaffold on train_out). In that case, `compute_A_atoms` must:

* Either be called with `scaffold_info=None` and:

  * raise `NotImplementedError("A-atoms for test_out require scaffold distances; build them first in 03_scaffold.")`
* Or laws/step must **not** call it for test_out until we have the chosen size + scaffold.

For now, to avoid spec drift, **only support A-atoms for train_out** in this WO. If called with `scaffold_info is None`, fail loudly.

#### 2.3 Output

`compute_A_atoms(...)` returns a dict of NumPy arrays:

```python
{
  "H": int,             # scalar
  "W": int,             # scalar
  "r": np.ndarray[(H,W), int],
  "c": np.ndarray[(H,W), int],
  "r_plus_c": np.ndarray[(H,W), int],
  "r_minus_c": np.ndarray[(H,W), int],
  "d_top": np.ndarray[(H,W), int],
  "d_bottom": np.ndarray[(H,W), int],
  "d_left": np.ndarray[(H,W), int],
  "d_right": np.ndarray[(H,W), int],
  "midrow_flag": np.ndarray[(H,W), bool],
  "midcol_flag": np.ndarray[(H,W), bool],
  "mod_r": Dict[int, np.ndarray[(H,W), int]],
  "mod_c": Dict[int, np.ndarray[(H,W), int]],
  "block_row": Dict[int, np.ndarray[(H,W), int]],
  "block_col": Dict[int, np.ndarray[(H,W), int]],
}
```

Keys:

* `mod_r[m] = r % m`, `mod_c[m] = c % m` for all m in the spec set.
* `block_row[b] = r // b`, `block_col[b] = c // b` for all b in the spec set.

We keep them as dicts keyed by m / b so later stages can iterate over sorted keys easily.

---

### 3. Exact semantics for A-atoms

#### 3.1 Basic coord arrays

Given `H, W`:

```python
r = np.arange(H, dtype=int)[:, None]  # shape (H,1)
c = np.arange(W, dtype=int)[None, :]  # shape (1,W)

r_grid = np.broadcast_to(r, (H, W))
c_grid = np.broadcast_to(c, (H, W))
```

Then:

```python
r_plus_c   = r_grid + c_grid
r_minus_c  = r_grid - c_grid
```

Store these as `r`, `c`, `r_plus_c`, `r_minus_c`.

#### 3.2 Distances

Reuse from scaffold; **do not recompute**:

```python
d_top    = np.asarray(scaffold_info["d_top"], dtype=int)
d_bottom = np.asarray(scaffold_info["d_bottom"], dtype=int)
d_left   = np.asarray(scaffold_info["d_left"], dtype=int)
d_right  = np.asarray(scaffold_info["d_right"], dtype=int)
```

Validate shape `(H,W)` or raise `ValueError` if mismatch.

#### 3.3 Midrow / midcol flags

Spec says:

* `has_midrow` is per-grid; midrow cells are where `d_top == d_bottom`.
* `has_midcol` is per-grid; midcol cells are where `d_left == d_right`.

So:

```python
midrow_flag = (d_top == d_bottom)
midcol_flag = (d_left == d_right)
```

You **do not** need `scaffold_info["has_midrow"]` to compute the per-cell flags. That aggregate lives in scaffold; we recompute the mask from distances to ensure consistency.

---

### 4. Grid-aware mod and block ranges (watchpoints applied)

#### 4.1 Mod classes: exact m set

From updated spec §5.1 A:

> Mod classes: `(r mod m, c mod m)` for
> `m ∈ {2,…,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)`.

Implement:

```python
def _divisors(n: int) -> np.ndarray:
    # simple int loop is fine for H,W <= 30
    divs = []
    for d in range(1, n+1):
        if n % d == 0:
            divs.append(d)
    return np.array(divs, dtype=int)
```

Then inside `compute_A_atoms`:

```python
max_dim = max(H, W)
base_ms = list(range(2, min(6, max_dim) + 1))
div_H = _divisors(H)
div_W = _divisors(W)

m_set = set(base_ms)
m_set.update(div_H.tolist())
m_set.update(div_W.tolist())
m_set = {m for m in m_set if m >= 2}  # avoid mod 1 (useless)

mod_r = {}
mod_c = {}

for m in sorted(m_set):
    mod_r[m] = r_grid % m
    mod_c[m] = c_grid % m
```

This matches spec **exactly** and honors the grid-aware policy and watchpoint.

#### 4.2 Block coords: exact b set

Spec:

> Block sizes: `b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W)`.

Note `divisors(H,W)` means divisors that divide **both** H and W (tiling-valid).

Implement:

```python
min_dim = min(H, W)
base_bs = list(range(2, min(5, min_dim) + 1))

div_H = _divisors(H)
div_W = _divisors(W)
div_HW = np.intersect1d(div_H, div_W)

b_set = set(base_bs)
b_set.update(div_HW.tolist())
b_set = {b for b in b_set if b >= 2}

block_row = {}
block_col = {}

for b in sorted(b_set):
    block_row[b] = r_grid // b
    block_col[b] = c_grid // b
```

No other b’s allowed. This is exactly spec.

---

### 5. Brainstem `run.py` changes

For WO-4.1 **no new wiring** is required in `run.py` beyond what you already have:

```python
present   = load_present(...)
canonical = canonicalize_truth(...)
scaffold  = build_scaffold(...)
size_ch   = choose_size(...)
laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
```

`05_laws/step.py` will internally:

* For each train_out grid, call `compute_A_atoms(...)` with scaffold info.
* For test canvas later (once size + scaffold exist), do the same.

So:

* **No changes needed** to `run.py` for this WO.
* `run.py` remains a thin orchestrator.

---

### 6. Receipts / trace for A-atoms

Add an optional trace helper in `05_laws/atoms.py`:

```python
def trace_A_atoms(A_atoms: Dict[str, Any]) -> None:
    H = A_atoms["H"]; W = A_atoms["W"]
    print(f"[A-atoms] H,W = {H},{W}")
    print(f"[A-atoms] r_plus_c min,max = {A_atoms['r_plus_c'].min()},{A_atoms['r_plus_c'].max()}")
    print(f"[A-atoms] midrow_flag sum = {A_atoms['midrow_flag'].sum()}")
    print(f"[A-atoms] midcol_flag sum = {A_atoms['midcol_flag'].sum()}")
    print(f"[A-atoms] mod m keys = {sorted(A_atoms['mod_r'].keys())}")
    print(f"[A-atoms] block b keys = {sorted(A_atoms['block_row'].keys())}")
```

Then `05_laws/step.py` can call this under `trace=True` for a couple of grids.

These receipts let reviewer check:

* Shapes are right,
* Mod / block key sets match spec,
* midrow/midcol flags behave sensibly.

---

### 7. Reviewer instructions

#### 7.1 Using `run.py` on real tasks (sanity)

Pick 2–3 ARC tasks, e.g.:

* `00576224` (6×6 tiling structure),
* A simple identity task with small grid (e.g. 3×3),
* A rectangular grid (e.g. 5×7) to see asymmetric mod/block sets.

Run:

```bash
python run.py --task tasks/00576224.json --trace
```

Ensure `05_laws/step.py` calls `compute_A_atoms` for each train_out when `trace=True` and prints the A-atoms summary.

Check:

1. `H,W` match `train_out[i].shape` for each grid.
2. `mod` keys:

   * For H=W=6:

     * `max(H,W)=6` → base m ∈ {2,3,4,5,6}; divisors(6) = {1,2,3,6}.
     * Combined m_set (≥2) = {2,3,4,5,6}.
3. `block` keys:

   * For H=W=6:

     * base b ∈ {2,3,4,5} (capped by min(5,min(H,W)) = 5),
     * divisors(H,W) = {1,2,3,6} → intersection filtered to ≥2: {2,3,6}.
     * Combined b_set = {2,3,4,5,6} but 6 > min_dim? Wait, spec allows `b` up to 5 or divisors(H,W); here 6 is a divisor of H,W but > min(5,min(H,W))?
       → spec allows it explicitly; verify `block_row[6]` exists with values in {0}.
4. `midrow_flag.sum()`:

   * For symmetric grids with a clear midrow in scaffold, sum should be a row’s worth of True on that midline.
5. `midcol_flag.sum()` similarly for columns.

If any of these deviate from spec (incorrect m/b sets, wrong shapes), that’s an implementation bug.

#### 7.2 Linking to M5 golden for 00576224

The Milestone 5 golden for `00576224` says:

* On the **test canvas**, after laws + equalities, there are:

  * 4 equivalence classes,
  * each of size 9.

That structure depends heavily on:

* Correct mod classes (parity / `(r mod 2, c mod 2)`),
* Correct distances and r±c.

If A-atoms are wrong (e.g. mod set missing `m=2`, or miscomputed r/c), the M5 golden will fail.

So for 00576224, once WO-5 is implemented:

* If Milestone 5 golden fails (class counts ≠ `{9:4}`), and A-atoms receipts are off (mod keys wrong, midrow flags wrong), the root cause is very likely in WO-4.1.

This ties A-atoms correctness to a downstream structural check.

---