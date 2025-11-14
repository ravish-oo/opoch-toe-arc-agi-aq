## WO-4.4 — `05_laws/atoms.py`: Repetition & palette/global (D16–D18, E19–E23)

**Goal:** Implement D and E atoms exactly as in `00_MATH_SPEC`:

From §5.1:

* **D. Repetition & tiling**

  * Minimal period along row/col (≤ dimension)
  * 2D tiling flags for block sizes dividing (H,W)
* **E. Palette/global**

  * Per-color pixel counts
  * Per-color component counts
  * Palette present/missing
  * Most/least frequent color(s)
  * Input↔output color permutation (bijective) & cyclic class over active palette

Atoms are derivations, not learned, and must be deterministic and grid-aware.

---

### 0. Anchors to read

Implementer + reviewer must re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * §5.1 **D. Repetition & tiling**, **E. Palette/global**
   * §5.1 A/C for how D/E relate to coords and components
2. **` @docs/anchors/01_STAGES.md `**

   * Stage **05_laws**: atoms → miner; inputs mirrored only for evaluation (F-section).
3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * “Seeing” vs “doing”: these atoms are pure seeing; no heuristics, no paid bits.

---

### 1. Libraries to use

We keep the stack:

```python
import numpy as np
from scipy import ndimage
```

* **Repetition / periods / tiling**: pure NumPy, small loops (H,W ≤ 30).
* **Per-color component counts**: `ndimage.label` from SciPy (same as C-atoms).

No custom BFS, no home-grown labeling.

---

### 2. API: where this lives and how it’s called

Extend `05_laws/atoms.py` with:

```python
def compute_D_atoms(grid: np.ndarray) -> dict:
    ...

def compute_E_atoms_for_grid(grid: np.ndarray, C_atoms: dict | None = None) -> dict:
    ...

def compute_global_palette_mapping(train_in: list[np.ndarray],
                                   train_out: list[np.ndarray]) -> dict:
    ...
```

`05_laws/step.py` is responsible for:

* Calling `compute_D_atoms` and `compute_E_atoms_for_grid` per train_out grid.
* Calling `compute_global_palette_mapping` once at task level (over all train pairs).

No `run.py` changes.

---

### 3. D-atoms: Repetition & tiling

#### 3.1 Minimal row / column periods

Spec: minimal period along row/col (≤ dimension).
This is **raw** least period; S0 uses its own “real repetition” filter.

Define a helper:

```python
def _least_period_1d(seq: np.ndarray) -> int:
    """
    Return smallest p (1 <= p <= L) s.t. seq is p-periodic.
    """
    L = len(seq)
    for p in range(1, L+1):
        if np.all(seq[:-p] == seq[p:]):
            return p
    return L  # fallback, but loop always returns by p=L
```

Then in `compute_D_atoms`:

```python
H, W = grid.shape
row_periods = np.zeros(H, dtype=int)
col_periods = np.zeros(W, dtype=int)

for r in range(H):
    row_periods[r] = _least_period_1d(grid[r, :])

for c in range(W):
    col_periods[c] = _least_period_1d(grid[:, c])
```

Store:

```python
D = {
  "row_periods": row_periods,   # shape (H,)
  "col_periods": col_periods,   # shape (W,)
}
```

No 2p≤L here; that’s S0’s job when aggregating to “real periods”.

#### 3.2 2D tiling flags for factor pairs

Spec: “2D tiling flags for block sizes dividing (H,W).”

Interpretation:

* For each block height `b_r` dividing `H` and block width `b_c` dividing `W`, flag whether the grid is exactly tiled by identical `b_r×b_c` blocks.

Implementation:

```python
def compute_D_atoms(grid: np.ndarray) -> dict:
    H, W = grid.shape
    D = { ... row_periods/col_periods as above ... }

    # divisors for tiling
    div_H = [d for d in range(1, H+1) if H % d == 0]
    div_W = [d for d in range(1, W+1) if W % d == 0]

    tiling_flags = {}  # (b_r, b_c) -> bool

    for b_r in div_H:
        for b_c in div_W:
            # Extract canonical tile at (0,0)
            tile = grid[0:b_r, 0:b_c]
            ok = True
            for r0 in range(0, H, b_r):
                if not ok: break
                for c0 in range(0, W, b_c):
                    block = grid[r0:r0+b_r, c0:c0+b_c]
                    if not np.array_equal(block, tile):
                        ok = False
                        break
            tiling_flags[(b_r, b_c)] = ok

    D["tiling_flags"] = tiling_flags
    return D
```

Tiling flags are deterministic and ≤(H*W) loop; fine on CPU.

---

### 4. E-atoms: Palette/global per grid

We separate **per-grid** palette/global from **task-level** input↔output permutation.

#### 4.1 Per-grid palette stats

`compute_E_atoms_for_grid(grid, C_atoms=None)`:

Per-color pixel counts:

```python
flat = grid.ravel()
counts = np.bincount(flat, minlength=10)
```

Palette set and missing:

```python
palette = [k for k in range(10) if counts[k] > 0]
missing = [k for k in range(10) if counts[k] == 0]
```

Most/least frequent:

* **Most**: all colors achieving max count (>0).
* **Least**: among colors with count>0, all achieving min positive count.

```python
max_count = counts.max()
most_freq = [k for k in range(10) if counts[k] == max_count and counts[k] > 0]

positive_counts = [(k, counts[k]) for k in range(10) if counts[k] > 0]
if positive_counts:
    min_pos = min(c for k,c in positive_counts)
    least_freq = [k for k,c in positive_counts if c == min_pos]
else:
    least_freq = []
```

Per-color component counts:

* If `C_atoms` is provided (from `compute_C_atoms`), reuse:

  ```python
  if C_atoms is not None:
      comp_counts = {k: len(C_atoms["components"].get(k, [])) for k in range(10)}
  else:
      # optional: recompute via ndimage.label if C_atoms not passed
      comp_counts = {}
      for k in range(10):
          mask = (grid == k).astype(np.uint8)
          if mask.any():
              _, num = ndimage.label(mask)
              comp_counts[k] = int(num)
          else:
              comp_counts[k] = 0
  ```

Return per-grid E:

```python
E_grid = {
  "pixel_counts": counts,             # np.array length 10
  "component_counts": comp_counts,    # dict k->int
  "palette": palette,                 # list[int]
  "missing": missing,                 # list[int]
  "most_frequent": most_freq,         # list[int]
  "least_frequent": least_freq,       # list[int]
}
```

No “global” info here yet.

---

### 5. E-atoms: Global input↔output permutation & cycles

Spec: “Input↔output color permutation (bijective) & cyclic class over active palette.”

This is **task-level**, across train pairs.

Implement:

```python
def compute_global_palette_mapping(train_in: list[np.ndarray],
                                   train_out: list[np.ndarray]) -> dict:
    """
    Compute a single bijective color perm π: palette_in → palette_out
    if it exists and is consistent across all training pairs.
    Else, no permutation.
    """
```

Algorithm:

1. Initialize mapping dicts:

   ```python
   fwd = {}  # color_in -> color_out
   rev = {}  # color_out -> color_in
   consistent = True
   ```

2. For each training pair i:

   * Flatten both grids:

     ```python
     gi = train_in[i].ravel()
     go = train_out[i].ravel()
     assert gi.shape == go.shape
     ```

   * For all positions j:

     ```python
     for cin, cout in zip(gi, go):
         if cin in fwd and fwd[cin] != cout:
             consistent = False; break
         if cout in rev and rev[cout] != cin:
             consistent = False; break
         fwd[cin] = cout
         rev[cout] = cin
     if not consistent:
         break
     ```

3. If not consistent:

   ```python
   return {
     "has_bijection": False,
     "perm": None,
     "cycles": None,
   }
   ```

4. If consistent, we must ensure bijection on **active palette only**:

   * `palette_in = sorted(set(fwd.keys()))`
   * `palette_out = sorted(set(rev.keys()))`
   * If `len(palette_in) != len(palette_out)` → not bijective.
   * Optionally require `palette_out == sorted(fwd.values())`.

5. Build explicit permutation list or dict:

   ```python
   perm = {cin: fwd[cin] for cin in palette_in}
   ```

6. Compute cyclic class = cycle decomposition of this permutation on `palette_in`:

   ```python
   visited = set()
   cycles = []
   for start in palette_in:
       if start in visited:
           continue
       cycle = []
       x = start
       while x not in visited:
           visited.add(x)
           cycle.append(x)
           x = perm.get(x, x)
       if len(cycle) > 0:
           cycles.append(cycle)
   ```

7. Return:

   ```python
   return {
     "has_bijection": True,
     "perm": perm,    # cin -> cout
     "cycles": cycles # list[list[int]]
   }
   ```

This is a natural, exact interpretation of “permutation” and “cyclic class” and uses no heuristics.

---

### 6. Brainstem `run.py` changes

No new changes.

* `run.py` still calls `mine_laws` once per task.
* `05_laws/step.py` is responsible for:

  * per-grid D/E (`compute_D_atoms`, `compute_E_atoms_for_grid`),
  * task-level mapping (`compute_global_palette_mapping`).

`run.py` remains minimal.

---

### 7. Receipts / trace

Add a small trace helper:

```python
def trace_D_E_atoms(D_atoms: dict, E_grid: dict, global_map: dict | None = None) -> None:
    print("[D-atoms] row_periods:", D_atoms["row_periods"])
    print("[D-atoms] col_periods:", D_atoms["col_periods"])
    print("[D-atoms] tiling_flags (True):",
          [k for k,v in D_atoms["tiling_flags"].items() if v])

    print("[E-atoms:grid] palette:", E_grid["palette"])
    print("[E-atoms:grid] pixel_counts:", E_grid["pixel_counts"])
    print("[E-atoms:grid] comp_counts:",
          {k: v for k, v in E_grid["component_counts"].items() if v > 0})

    if global_map is not None:
        print("[E-atoms:global] has_bijection:", global_map["has_bijection"])
        if global_map["has_bijection"]:
            print("  perm:", global_map["perm"])
            print("  cycles:", global_map["cycles"])
```

`05_laws/step.py` can call this under `trace=True` for the first training grid and once for the global mapping.

---

### 8. Reviewer instructions

#### 8.1 Using `run.py` on 2–3 tasks

Pick:

* Task A: trivial identity (train_in == train_out), multiple colors.
* Task B: a periodic tiling task (e.g., grid is 2×2 tile repeated).
* Task C: a color-perm task (input colors permuted to output).

Run:

```bash
python run.py --task tasks/<task_id>.json --trace
```

Check:

**Task A (identity):**

* `compute_global_palette_mapping`:

  * `has_bijection == True`,
  * `perm` is identity on all used colors,
  * `cycles` are singletons `[k]`.

**Task B (tiling):**

* If task grid is exactly e.g. 2×2 tile repeated on a 6×6:

  * `tiling_flags[(2,2)] == True`,
  * `tiling_flags[(3,3)]` etc. should be False unless by chance.

* Row/col periods should match visible repetitiveness.

**Task C (simple permutation):**

* Example: all color 1 → 3, 3 → 5, etc., consistently across all training pairs.

  * `global_map["has_bijection"] == True`
  * `perm` maps input palette to output palette exactly.
  * `cycles` reflect permutation cycles, e.g. `[1,3,5]`.

If mapping is known to be non-bijective (e.g., two input colors map to same output color in train), `has_bijection` must be False.

#### 8.2 Identifying implementation bug vs spec issue

* If `tiling_flags` shows True for a pair `(b_r,b_c)` when visual inspection clearly shows blocks differ → implementation bug.
* If `global_map["has_bijection"]` is True but some training cells break the mapping (same cin mapped to different cout) → bug.
* If periods or tiling flags don’t exist when they obviously should, check `_least_period_1d` and tiling loops.

Almost any failure here is an implementation mismatch; atoms D/E are fully specified by spec.

---
