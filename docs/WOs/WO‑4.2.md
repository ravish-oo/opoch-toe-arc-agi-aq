## WO-4.2 — `05_laws/atoms.py`: Local Texture Atoms (B8–B11)

**Goal:** For each canonical output grid, compute the “B” local texture atoms exactly as per spec:

From ` @docs/anchors/00_MATH_SPEC.md ` §5.1 **B. Local texture**:

* N4 / N8 neighbor counts per color
* 3×3 neighborhood hash (base-11, pad sentinel=10)
* 5×5 ring signature (perimeter only, sentinel=10 outside)
* Row/col run-lengths through the cell: `(span_len, span_start, span_end)`

All done in canonical coordinates, grid-aware, deterministic.

---

### 0. Anchors to read before coding

Implementer + reviewer must re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * §5.1 **B. Local texture** (exact list of atoms and semantics).
   * §5.1 A + §4 to understand how these tie into distances and frame (we won’t recompute frame here, just use grid geometry).

2. **` @docs/anchors/01_STAGES.md `**

   * Stage **05_laws**: atoms are derivations over canonical grids; laws are mined from train_out only.

3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * Mapping where these local texture atoms sit in the “seeing” side (free sector). No paid bits, no randomness.

We do not change any meanings, only implement what is written.

---

### 1. Libraries to use (no custom algorithms)

For B-atoms we use:

```python
import numpy as np
from scipy import ndimage
# scikit-image is not strictly needed for B; we keep it for C-atoms later but not used here.
```

**Explicit functions:**

* `scipy.ndimage.convolve`

  * For N4 / N8 neighbor counts via 3×3 kernels.
* `scipy.ndimage.generic_filter` **or** small NumPy slicing loops

  * For 3×3 neighborhood hashes and 5×5 ring signatures.

We **do not**:

* Write our own neighbor count loops,
* Implement manual padding logic in a clever way; we use NumPy padding.

Everything is small grid (≤30), CPU is fine, we prefer clarity over micro-optimizations.

---

### 2. Input / output contracts

#### 2.1 Where this lives

Extend `05_laws/atoms.py`:

```bash
05_laws/
  atoms.py   # A-atoms (done), now add B-atoms
  step.py    # calls atoms.compute_A_atoms, atoms.compute_B_atoms, etc.
```

#### 2.2 Input to B-atoms

`05_laws/step.py` will, for each **output grid** (train_out and later test canvas), call:

```python
from 05_laws.atoms import compute_B_atoms

B_atoms = compute_B_atoms(
    grid: np.ndarray[(H,W)],    # canonical train_out grid (colors 0..9)
)
```

We do **not** need scaffold here; B is purely local on the color grid.

#### 2.3 Output

`compute_B_atoms(grid)` returns a dict:

```python
{
  # N4/N8 neighbor counts per color
  "n4_counts":  Dict[int, np.ndarray[(H,W), int]],  # per color k: grid of N4 neighbors of color k
  "n8_counts":  Dict[int, np.ndarray[(H,W), int]],  # per color k: grid of N8 neighbors of color k

  # 3×3 hash and 5×5 ring
  "hash_3x3":   np.ndarray[(H,W), int],
  "ring_5x5":   np.ndarray[(H,W), int],

  # row run-lengths
  "row_span_len":   np.ndarray[(H,W), int],
  "row_span_start": np.ndarray[(H,W), int],  # column index of span start
  "row_span_end":   np.ndarray[(H,W), int],  # column index of span end (inclusive)

  # col run-lengths
  "col_span_len":   np.ndarray[(H,W), int],
  "col_span_start": np.ndarray[(H,W), int],  # row index of span start
  "col_span_end":   np.ndarray[(H,W), int],  # row index of span end (inclusive)
}
```

All arrays must be exact shape `(H,W)`.

---

### 3. Exact semantics

#### 3.1 N4 / N8 neighbor counts per color

Interpretation (per spec):

* For each color `k` in 0..9:

  * `n4_counts[k][r,c]` = number of 4-neighbors of cell (r,c) that have color k.
  * `n8_counts[k][r,c]` = number of 8-neighbors of cell (r,c) that have color k.

Algorithm:

1. Compute color masks:

   ```python
   H, W = grid.shape
   n4_counts = {}
   n8_counts = {}
   # Option: restrict to palette used in this grid
   colors = np.unique(grid)
   ```

2. Define convolution kernels:

   ```python
   # N4: up, down, left, right
   kernel_n4 = np.array([[0,1,0],
                         [1,0,1],
                         [0,1,0]], dtype=int)

   # N8: all 8 neighbors (no center)
   kernel_n8 = np.array([[1,1,1],
                         [1,0,1],
                         [1,1,1]], dtype=int)
   ```

3. For each `k` in `colors`:

   ```python
   mask_k = (grid == k).astype(int)
   n4_counts[k] = ndimage.convolve(mask_k, kernel_n4, mode="constant", cval=0)
   n8_counts[k] = ndimage.convolve(mask_k, kernel_n8, mode="constant", cval=0)
   ```

* `mode="constant", cval=0` means out-of-bounds neighbors are treated as not matching any color → count 0, which matches spec.

#### 3.2 3×3 full hash (base-11, sentinel=10)

Spec:

* 3×3 neighborhood hash (base-11 packing; palette 0..9; sentinel=10 for out-of-grid padding).

We define:

* For each cell (r,c), consider the 3×3 window centered at (r,c).
* For positions falling outside grid, use sentinel 10.
* Encode the 3×3 window into an integer in base 11:

  Let `vals` be the 3×3 window flattened in a **fixed order**, e.g. row-major:

  `vals = [v00, v01, ..., v22]`

  Hash:

  ```python
  h = 0
  for v in vals:
      h = h * 11 + v   # base 11
  ```

Implementation:

1. Pad grid with sentinel=10:

   ```python
   SENT = 10
   padded = np.pad(grid, pad_width=1, mode="constant", constant_values=SENT)  # shape (H+2,W+2)
   ```

2. For each of the 9 positions, collect shifted views:

   ```python
   # offsets relative to center (0,0)
   offsets = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1), ( 0,0), ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]

   windows = []
   for dr, dc in offsets:
       # note: center corresponds to padded[1:H+1, 1:W+1]
       r0 = 1 + dr; r1 = r0 + H
       c0 = 1 + dc; c1 = c0 + W
       windows.append(padded[r0:r1, c0:c1])  # each (H,W)
   ```

3. Now encode:

   ```python
   hash_3x3 = np.zeros((H, W), dtype=int)
   for w in windows:
       hash_3x3 = hash_3x3 * 11 + w
   ```

This yields a deterministic base-11 hash for each 3×3 neighborhood, including boundaries (sentinel pads).

#### 3.3 5×5 ring signature (perimeter only, sentinel=10)

Spec:

* 5×5 **ring** signature (perimeter only; sentinel=10 when ring steps outside grid).

Interpretation:

* For each cell (r,c), take the 5×5 window centered at (r,c).
* Any position outside grid → sentinel 10.
* Consider only the 16 perimeter cells of that 5×5 window, in a fixed order, and pack in base-11.

Perimeter index order (explicit):

Indices `(i,j)` in a 5×5 window with center at (2,2):

* Top row:    (0,0), (0,1), (0,2), (0,3), (0,4)              5
* Right col:  (1,4), (2,4), (3,4)                            3
* Bottom row: (4,4), (4,3), (4,2), (4,1), (4,0)              5
* Left col:   (3,0), (2,0), (1,0)                            3

Total 5+3+5+3 = 16 positions.

Implementation:

1. Pad grid with sentinel=10, pad 2 in each direction:

   ```python
   SENT = 10
   padded5 = np.pad(grid, pad_width=2, mode="constant", constant_values=SENT)  # (H+4,W+4)
   ```

2. Construct 5×5 windows via slicing:

   For each perimeter offset `(dr,dc)` relative to center (0,0) where the window center corresponds to padded5[2:H+2, 2:W+2]:

   ```python
   offsets_ring = [
       (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
       (-1, 2), ( 0,2), ( 1,2),
       ( 2, 2), ( 2, 1), ( 2,0), ( 2,-1), ( 2,-2),
       ( 1,-2), ( 0,-2), (-1,-2),
   ]
   ```

   Then:

   ```python
   ring_5x5 = np.zeros((H,W), dtype=int)
   for dr, dc in offsets_ring:
       r0 = 2 + dr; r1 = r0 + H
       c0 = 2 + dc; c1 = c0 + W
       patch = padded5[r0:r1, c0:c1]  # (H,W)
       ring_5x5 = ring_5x5 * 11 + patch
   ```

This gives a base-11 signature of just the perimeter, with sentinel padding.

#### 3.4 Row / col run-lengths through each cell

Spec:

* Row/col run-lengths through the cell: `(span_len, span_start, span_end)`.

Interpretation (per color of that cell):

* For each cell (r,c), look along row r:

  * Let color value `k = grid[r,c]`.
  * Treat the row as a string.
  * Find the maximal contiguous segment of that row where color = k that contains column c.
  * `span_start` = index of leftmost column of that segment;
  * `span_end`   = index of rightmost column of that segment;
  * `span_len`   = span_end - span_start + 1.
* Similarly along column c for `col_span_*`.

Implementation (NumPy + simple loops; grids ≤30):

Row run-lengths:

```python
row_span_len   = np.zeros((H,W), dtype=int)
row_span_start = np.zeros((H,W), dtype=int)
row_span_end   = np.zeros((H,W), dtype=int)

for r in range(H):
    row_vals = grid[r, :]                    # shape (W,)
    # Precompute spans for this row
    c = 0
    while c < W:
        k = row_vals[c]
        # find end of this same-color run
        c2 = c + 1
        while c2 < W and row_vals[c2] == k:
            c2 += 1
        # span is [c, c2)
        span_start = c
        span_end = c2 - 1
        span_len = span_end - span_start + 1
        row_span_start[r, span_start:span_end+1] = span_start
        row_span_end[r, span_start:span_end+1] = span_end
        row_span_len[r, span_start:span_end+1] = span_len
        c = c2
```

Column run-lengths: analogous, iterating over columns:

```python
col_span_len   = np.zeros((H,W), dtype=int)
col_span_start = np.zeros((H,W), dtype=int)
col_span_end   = np.zeros((H,W), dtype=int)

for c in range(W):
    col_vals = grid[:, c]          # shape (H,)
    r = 0
    while r < H:
        k = col_vals[r]
        r2 = r + 1
        while r2 < H and col_vals[r2] == k:
            r2 += 1
        span_start = r
        span_end = r2 - 1
        span_len = span_end - span_start + 1
        col_span_start[span_start:span_end+1, c] = span_start
        col_span_end[span_start:span_end+1, c] = span_end
        col_span_len[span_start:span_end+1, c] = span_len
        r = r2
```

This is O(HW) and trivial for ARC sizes.

---

### 4. Brainstem `run.py` changes

For WO-4.2 we **do not** need any new wiring in `run.py`. Existing flow stays:

```python
present   = load_present(...)
canonical = canonicalize_truth(...)
scaffold  = build_scaffold(...)
size_ch   = choose_size(...)
laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
```

`05_laws/step.py` is responsible for:

* Calling `compute_A_atoms` and `compute_B_atoms` internally for each train_out.

`run.py` remains a thin orchestrator.

---

### 5. Receipts / trace for B-atoms

Add helper in `05_laws/atoms.py`:

```python
def trace_B_atoms(B_atoms: Dict[str, Any], grid: np.ndarray) -> None:
    H, W = grid.shape
    print(f"[B-atoms] H,W = {H},{W}")
    print(f"[B-atoms] colors present: {np.unique(grid)}")
    # show one example color for neighbor counts
    example_k = int(np.unique(grid)[0])
    print(f"[B-atoms] n4_counts[{example_k}] sum =", B_atoms["n4_counts"][example_k].sum())
    print(f"[B-atoms] n8_counts[{example_k}] sum =", B_atoms["n8_counts"][example_k].sum())
    print("[B-atoms] hash_3x3 min,max =", B_atoms["hash_3x3"].min(), B_atoms["hash_3x3"].max())
    print("[B-atoms] ring_5x5 min,max =", B_atoms["ring_5x5"].min(), B_atoms["ring_5x5"].max())
    print("[B-atoms] row_span_len min,max =", B_atoms["row_span_len"].min(), B_atoms["row_span_len"].max())
    print("[B-atoms] col_span_len min,max =", B_atoms["col_span_len"].min(), B_atoms["col_span_len"].max())
```

Then `05_laws/step.py` can call this under `trace=True` on one training grid per task.

---

### 6. Reviewer instructions

#### 6.1 Using `run.py` on 2–3 tasks

Pick:

* Task A: small grid, simple pattern (e.g. 3×3 identity).
* Task B: `00576224` (6×6 periodic pattern).
* Task C: non-square grid (e.g. 5×7) to exercise boundaries and ring padding.

Run:

```bash
python run.py --task tasks/00576224.json --trace
```

Ensure `mine_laws` calls `compute_B_atoms` and `trace_B_atoms` for at least the first train_out.

Manual checks on a small 3×3 or 4×4 grid:

1. N4/N8:

   * For a center cell in a uniform grid, N4 count for that color should be 4, N8 count = 8 (except near borders).
   * For a corner cell, N4 count ≤2, N8 count ≤3.

2. 3×3 hash:

   * On a uniform grid with all value k, `hash_3x3` should be the same at all interior cells, and different near borders due to sentinel 10 padding.

3. 5×5 ring:

   * On a small grid (e.g. 3×3), all cells’ rings will include many sentinel=10; verify `ring_5x5` not constant across cells if you change center, and sentinel effect is visible (values differ from uniform interior case).

4. Run-lengths:

   * For a row `[1,1,2,2,2,3]`, row_span_len for positions 0–1 → 2, for 2–4 → 3, for 5 → 1.
   * Similarly for columns.

If any of these basic expectations fail, the implementation is wrong.

#### 6.2 Linking to later golden (00576224)

For Milestone 5 on `00576224` you expect:

* 4 equivalence classes of size 9 each on the test canvas.

This structure is driven partly by local texture (B-atoms), especially:

* run-lengths and periodic relations,
* 3×3 hashes.

If M5 golden fails and A-atoms look correct, B-atoms are the next place to look:

* Check neighbor counts and 3×3 ring values on train_out for obvious mismatches.
* Compare `row_span_len` against visual runs in the PNG for that task.

Legit unsatisfiable behavior for this WO is extremely unlikely; almost any failure here is implementation mismatch, not spec inconsistency.

---