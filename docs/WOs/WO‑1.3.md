## WO-1.3 — `02_truth`: canonical labeling (120–160 LOC)

**Goal:**
Take the union graph from WO-1.2 and:

1. Call `igraph.Graph.canonical_permutation` with vertex color codes,
2. Extract per-grid **canonical row/col order** from the permutation (with stable tie-break),
3. Remap each grid array (`train_in`, `train_out`, `test_in`) into canonical coordinates.

**Acceptance:**

* Repeated runs on the same task produce **identical** canonical arrays (deterministic up to hash),
* Graph components remain disjoint (one per grid),
* No grid id leaks into colors.

---

### 0. Anchors to read before coding

Mandatory:

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §2 “Stage A — Canonical labeling (awareness & gauge)”

     * 2.1 Disjoint-union graph
     * 2.2 Canonical labeling
     * 2.3 Canonical local coordinates (R_X, C_X)

2. ` @docs/anchors/01_STAGES.md `

   * “truth”: defines this stage as Π / canonicalization.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * The “truth” section: apply Π, kill minted differences; canonicalization is how we do that.

4. ` @goldens/00576224/M1_M2_checkpoints.md `

   * Section 2 (Milestone 2 – `scaffold`) is gauge-invariant, but it assumes canonicalization is correct. After WO-1.3, we should not break any of those invariants. 

---

### 1. Libraries to use (mature, well-documented)

For this WO we need:

* `python-igraph`

  * We already use it in WO-1.2.
  * APIs needed:

    * `import igraph as ig`
    * `g = canonical["graph"]`
    * `perm = g.canonical_permutation(color=g.vs["color_code"])`
    * `list(perm)` to get the permutation as a Python list.

* `numpy`

  * Index remapping for arrays:

    * `np.argsort` for inverse permutation of rows/cols,
    * `grid_old[inv_rows][:, inv_cols]` to remap.

* `typing`

  * `Dict`, `Any`, `List`, `Tuple`.

* `logging`

  * For trace logs.

* `hashlib` (stdlib)

  * For a simple stable hash of canonical grids to check determinism.

We **do not** implement any graph algorithm ourselves; igraph handles canonical permutation for us.

---

### 2. Input & output contract for `truth.canonicalize` after WO-1.3

We assume WO-1.2 has already updated `canonicalize` to build the union graph and return:

```python
canonical = {
    "task_id": str,
    "train_in":  List[np.ndarray],
    "train_out": List[np.ndarray],
    "test_in":   List[np.ndarray],  # length 1 (thanks to test_index)
    "shapes":    dict,
    "palettes":  dict,
    "graph":     ig.Graph,          # union graph G with v["color_code"]
    "row_orders": None,
    "col_orders": None,
}
```

After WO-1.3, we keep the same keys but:

* `train_in`, `train_out`, `test_in` are **remapped** into canonical gauge,
* `row_orders` and `col_orders` store the **old→canonical** indices for each grid.

Concrete output shape:

```python
canonical["row_orders"] = row_orders  # List[List[int]], one per grid
canonical["col_orders"] = col_orders  # List[List[int]], one per grid
# grids themselves are updated in-place in canonical["train_in"/"train_out"/"test_in"]
```

Order of “grids” is always:

```python
all_grids = train_in + train_out + test_in
# row_orders and col_orders follow this order.
```

---

### 3. Canonical permutation and per-grid row/col ordering

#### 3.1 Canonical permutation with igraph

Spec says:

> “Run a canonical labeling … using vertex colors, then derive canonical row/col order from the permutation; tie-break stable on (canon_index, original_index).”

In igraph:

```python
g_all = canonical["graph"]
perm = g_all.canonical_permutation(color=g_all.vs["color_code"])
perm_list = list(perm)  # perm_list[canon_index] = old_vertex_index
```

That means:

* `perm_list[canon_idx]` = original vertex index at canonical position `canon_idx`.

We need the **inverse mapping** (canonical index of each original vertex):

```python
n = g_all.vcount()
inv = [0] * n
for canon_idx, old_idx in enumerate(perm_list):
    inv[old_idx] = canon_idx  # canonical index of old vertex
```

`inv[v]` is now exactly the “canon_index” used in the spec.

#### 3.2 Reconstruct grid meta (offsets & shapes)

We must know per vertex which grid, row, col, or row_node/col_node it corresponds to. In WO-1.2 we used a deterministic layout; we can recompute that now:

* Build a list `grids = train_in + train_out + test_in`, similar to WO-1.2.
* For each grid index `gidx`:

  ```python
  H, W = grid.shape
  n_cells = H * W
  n_rows  = H
  n_cols  = W
  n_verts = n_cells + n_rows + n_cols
  base    = cumulative sum
  ```

For each grid:

* Cell vertex index: `base + r * W + c`
* Row node index: `base + H*W + r`
* Col node index: `base + H*W + H + c`

This layout matches the one used to build the graph in WO-1.2.

#### 3.3 Computing row_orders / col_orders (old→canonical)

For grid `gidx`:

* For each row `r`:

  * `v_row = base + H*W + r`
  * `canon_idx_row = inv[v_row]`
  * Collect `(canon_idx_row, r)` pairs.
* Sort rows by `(canon_idx_row, r)` (tie-break on original index r).
* After sorting, the order corresponds to canonical row indices `[0..H-1]`. So:

```python
R_X = [0] * H  # old r -> canonical index
for canonical_r, (canon_idx_row, old_r) in enumerate(sorted_rows):
    R_X[old_r] = canonical_r
```

Same for columns:

```python
C_X = [0] * W  # old c -> canonical index
for canonical_c, (canon_idx_col, old_c) in enumerate(sorted_cols):
    C_X[old_c] = canonical_c
```

These `R_X` and `C_X` arrays are exactly the `R_X(r)` and `C_X(c)` from the spec (old→canonical).

#### 3.4 Remapping grids to canonical coordinates

To remap a grid `G` with `R_X` and `C_X` (old→canonical):

* Compute inverse mapping (canonical→old) using `np.argsort`:

```python
import numpy as np

inv_rows = np.argsort(R_X)  # inv_rows[new_r] = old_r
inv_cols = np.argsort(C_X)  # inv_cols[new_c] = old_c

G_canon = G[inv_rows, :][:, inv_cols]
```

Apply this to each:

* `train_in[gidx]`,
* `train_out[gidx]`,
* `test_in[0]` (only one test grid per run).

Store:

```python
canonical["train_in"]  = canonical_train_in
canonical["train_out"] = canonical_train_out
canonical["test_in"]   = canonical_test_in
canonical["row_orders"] = row_orders  # list of R_X for all grids
canonical["col_orders"] = col_orders
```

Now all further stages work on canonical arrays.

---

### 4. Implementation sketch (02_truth/step.py additions)

Assuming you have the union graph logic from WO-1.2 already in `canonicalize`, extend it like this:

```python
# 02_truth/step.py
from typing import Any, Dict, List, Tuple
import logging
import hashlib

import numpy as np
import igraph as ig

def canonicalize(present: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: truth (Π / canonical gauge)
    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2: Stage A — Canonical labeling
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π, kill minted differences

    Input:
      present: from 01_present.load
    Output:
      canonical: present with union graph G and canonicalized grids.
    """

    # 1) Build union graph as in WO-1.2 (or reuse existing code)
    #    g_all has v["color_code"] set, and vertices laid out in deterministic order.
    #    We assume you already have this 'g_all' and 'grids' (train_in+train_out+test_in).
    #
    # Example:
    train_in  = present["train_in"]
    train_out = present["train_out"]
    test_in   = present["test_in"]   # length 1 (due to test_index)
    grids: List[np.ndarray] = train_in + train_out + test_in

    g_all, grid_shapes, vert_offsets = _build_union_graph(grids)  # from WO-1.2 helper

    # 2) Canonical permutation by vertex color
    perm = g_all.canonical_permutation(color=g_all.vs["color_code"])
    perm_list = list(perm)  # perm_list[canon_idx] = old_vertex_index

    n = g_all.vcount()
    inv = [0] * n  # inv[old_idx] = canonical index
    for canon_idx, old_idx in enumerate(perm_list):
        inv[old_idx] = canon_idx

    # 3) Compute row_orders and col_orders (old->canonical)
    row_orders: List[List[int]] = []
    col_orders: List[List[int]] = []

    for grid_idx, (H, W) in enumerate(grid_shapes):
        base = vert_offsets[grid_idx]

        def cell_index(r: int, c: int) -> int:
            return base + r * W + c

        def row_node_index(r: int) -> int:
            return base + H * W + r

        def col_node_index(c: int) -> int:
            return base + H * W + H + c

        # rows
        row_pairs = []
        for r in range(H):
            v = row_node_index(r)
            canon_idx_row = inv[v]
            row_pairs.append((canon_idx_row, r))
        row_pairs.sort(key=lambda x: (x[0], x[1]))
        R_X = [0] * H
        for canon_r, (_, old_r) in enumerate(row_pairs):
            R_X[old_r] = canon_r
        row_orders.append(R_X)

        # cols
        col_pairs = []
        for c in range(W):
            v = col_node_index(c)
            canon_idx_col = inv[v]
            col_pairs.append((canon_idx_col, c))
        col_pairs.sort(key=lambda x: (x[0], x[1]))
        C_X = [0] * W
        for canon_c, (_, old_c) in enumerate(col_pairs):
            C_X[old_c] = canon_c
        col_orders.append(C_X)

    # 4) Remap grids to canonical coordinates
    canonical_grids: List[np.ndarray] = []
    for grid_idx, G in enumerate(grids):
        H, W = grid_shapes[grid_idx]
        R_X = np.array(row_orders[grid_idx], dtype=int)
        C_X = np.array(col_orders[grid_idx], dtype=int)
        inv_rows = np.argsort(R_X)
        inv_cols = np.argsort(C_X)
        G_canon = G[inv_rows, :][:, inv_cols]
        canonical_grids.append(G_canon)

    num_train = len(train_in)
    num_out   = len(train_out)

    canonical_train_in  = canonical_grids[0:num_train]
    canonical_train_out = canonical_grids[num_train:num_train+num_out]
    canonical_test_in   = canonical_grids[num_train+num_out:]

    canonical: Dict[str, Any] = {
        "task_id": present["task_id"],
        "train_in":  canonical_train_in,
        "train_out": canonical_train_out,
        "test_in":   canonical_test_in,
        "shapes":    present["shapes"],   # shapes don’t change under permutation
        "palettes":  present["palettes"], # palettes don’t change either
        "graph":     g_all,
        "row_orders": row_orders,
        "col_orders": col_orders,
    }

    if trace:
        # simple hash over all canonical grids for determinism check
        h = hashlib.sha256()
        for G in canonical_train_in + canonical_train_out + canonical_test_in:
            h.update(G.tobytes())
        digest = h.hexdigest()[:16]
        logging.info(f"[truth] canonicalization hash={digest}")

    return canonical
```

> Note: `_build_union_graph` is the helper from WO-1.2 that returns `(graph, grid_shapes, vert_offsets)`. If you inlined it there, you can either:
>
> * factor it into `utils/graph.py`, or
> * recompute `grid_shapes` and `vert_offsets` here using the same deterministic layout.

---

### 5. `run.py` behavior after WO-1.3

Still minimal, still:

```python
present   = load_present(task_bundle, trace=trace)
canonical = canonicalize(present, trace=trace)
scaffold  = build_scaffold(canonical, trace=trace)
...
```

After WO-1.3:

* `present.load` works (WO-1.1),
* `truth.canonicalize` now returns canonicalized grids and logs a hash under `--trace`,
* The pipeline proceeds to `03_scaffold.build`, which still raises `NotImplementedError`.

No new logic in `run.py`.

---

### 6. Receipts for WO-1.3

Under `--trace`, `truth.canonicalize` will:

* Log the canonicalization hash, e.g.:

  ```text
  [truth] canonicalization hash=abc123deadbeef00
  ```

Reviewer can:

* Run the **same** command twice and ensure the hash is identical,
* Change `--test-index` (on tasks with multiple tests) and check hashes differ in expected ways.

Optionally, you can also log:

* `row_orders` and `col_orders` lengths per grid (not contents, to keep logs small),
* The component count of `graph` (should still equal number of grids).

---

### 7. Reviewer instructions & expectations

**Commands:**

1. Single-test task (like `00576224`):

   ```bash
   python run.py --task-id 00576224 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

2. Multi-test task (like `5d2a5c43`):

   ```bash
   python run.py --task-id 5d2a5c43 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace

   python run.py --task-id 5d2a5c43 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 1 \
                 --trace
   ```

**What to expect:**

* `present` logs shapes/palettes as in Milestone 1 golden. 
* `truth` logs:

  * union graph stats (from WO-1.2)
  * canonical hash (from WO-1.3)
* Pipeline then hits `NotImplementedError` at `03_scaffold.build`.

**Determinism check:**

* Running the same command twice for the same `(task_id, test_index)` must produce the **same canonicalization hash**.
* If the hash changes between runs, canonicalization is non-deterministic → WO-1.3 is wrong.

**Math/spec consistency checks:**

* Palettes and shapes in `canonical` must not change from `present`. Canonicalization must permute rows/cols only, not recolor or resize.
* Number of connected components in `graph` must still equal `len(train_in)+len(train_out)+1`.

**Legit gap vs bug:**

* **Legit WO-1.3 gap:**

  * `present` is correct; `truth` logs union graph & hash; `scaffold` still unimplemented.

* **Bug:**

  * `truth` doesn’t log or returns errors before canonicalization,
  * `row_orders` / `col_orders` mis-sized (not matching H/W),
  * canonical hash is unstable across runs,
  * palettes or shapes changed by canonicalization.

---

### 8. Optimization / CPU considerations

* igraph’s `canonical_permutation` is C-optimized; our graphs are tiny; CPU costs are negligible.
* We do **not** need any simplified or approximate canonicalization,—no “just sort rows lexicographically” hacks. That would violate spec and Π.

---
