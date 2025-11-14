## WO-1.2 — `02_truth`: union graph (120–160 LOC)

**Goal:**
Given the `present` object from WO-1.1, build the **disjoint-union graph** (G) exactly as in the math spec so we can later run canonical labeling on it.

We do **not** do canonical labeling yet (that’s WO-1.3). Here we only:

* Build one igraph `Graph` with:

  * vertices = cells + row_node_r + col_node_c for every grid,
  * vertex colors = `("cell", color)` or `("row_node",100)` or `("col_node",101)` encoded as integers,
  * edges = N4 cell adjacencies + incidence edges (cell↔row_node, cell↔col_node),
* Ensure there is **no grid_id** in any vertex color,
* Attach this graph to the `canonical` object returned by `02_truth.canonicalize`.

**Acceptance:** for real tasks like `00576224`, the graph has reasonable `|V|`, `|E|` and the number of connected components equals the number of grids in `present`. Dumps a checksum under `--trace`.

---

### 0. Anchors to read before coding

Mandatory:

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §2 “Stage A — Canonical labeling (awareness & gauge)”
     Especially 2.1 “Disjoint–union graph (G)” and the definition of vertex colors, edges, and “no grid_id in colors”.

2. ` @docs/anchors/01_STAGES.md `

   * Section “truth”: explains that `truth` is the Π stage that builds a canonicalizable representation.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * The part mapping “truth” to “apply Π / kill minted differences”; union graph is the structure we canonicalize.

4. ` @goldens/00576224/M1_M2_checkpoints.md ` (skim)

   * Understand that Milestone 2 golden is about **scaffold distances**, but it assumes canonicalization + union graph are correct. For this WO we won’t match numeric scaffold yet, but we should keep in mind that each grid will become a connected component in (G).

---

### 1. Libraries to use (mature, well-documented)

We want to reuse existing graph libraries, not reinvent them.

For this WO, use:

* `python-igraph`

  * We’ll use:

    * `import igraph as ig`
    * `ig.Graph(n=num_vertices, edges=edge_list, directed=False)`
    * `g.vs["color_code"] = [...]` for vertex colors
  * This avoids writing our own graph structure and is exactly what we’ll use in WO-1.3 for canonical labeling (`canonical_permutation`).

* `numpy`

  * To get shapes (`H,W`) from `present["train_in"]`, `present["train_out"]`, `present["test_in"]`.
  * Arrays are already created in WO-1.1.

* `typing` (optional)

  * For `Any`, `Dict`.

* `logging`

  * To log vertex/edge counts and component counts when `trace=True`.

We do **not** implement adjacency algorithms ourselves beyond simple deterministic loops to build the edge list.

---

### 2. Input & output contract for `truth.canonicalize`

#### Input

From `run.py` and `present.load`:

```python
present = {
    "task_id": str,
    "train_in":  List[np.ndarray[int8]],
    "train_out": List[np.ndarray[int8]],
    "test_in":   List[np.ndarray[int8]],
    "shapes":    {...},
    "palettes":  {...},
}
```

We assume this schema from WO-1.1.

#### Output: `canonical` object

For this WO, `canonicalize` must return a dict with at least:

```python
canonical = {
    "task_id": present["task_id"],
    "train_in":  present["train_in"],
    "train_out": present["train_out"],
    "test_in":   present["test_in"],
    "shapes":    present["shapes"],
    "palettes":  present["palettes"],
    "graph":     ig.Graph,    # union graph G
    # placeholders for later WOs:
    "row_orders": None,       # to be filled by WO-1.3
    "col_orders": None,
}
```

We **do not** alter the grids yet; canonical row/col orders are added in WO-1.3. For now, `graph` is the main new field.

---

### 3. Graph structure: exact spec

From `00_MATH_SPEC.md` §2.1:

Per grid (X):

* **Vertices:**

  * One cell vertex for each cell (v_{r,c}),
  * One row node (v^{\text{row}}_r) for each row index (r),
  * One col node (v^{\text{col}}_c) for each col index (c).

* **Vertex colors** (vcolor(v)), encoded as single integer `color_code`:

  * For cells: `(“cell”, c_X(r,c))`
    → an integer in `0..9` (the actual color).
  * For row nodes: `(“row_node”, 100)`
    → integer `100`.
  * For col nodes: `(“col_node”, 101)`
    → integer `101`.

  **No grid_id is ever included** in this color code.

* **Edges:**

  * N4 cell adjacency: for each grid, for each cell `(r,c)`, connect to `(r+1,c)` (down) and `(r,c+1)` (right) if in bounds (this yields undirected 4-adjacency without duplicates).
  * Incidence edges:

    * `(v_{r,c}, v^{row}_r)` with some edge color “row_edge” (we don’t need a separate attribute now),
    * `(v_{r,c}, v^{col}_c)` similarly.

* **Disjoint union:**

  * Vertex indices for grids are **offset** by the number of vertices already created.
  * No edges between different grids.

So in code, we will:

* Iterate over all grids in order: `train_in`, `train_out`, `test_in`,
* For each grid, assign a contiguous index range for:

  * cells: `H * W` vertices,
  * row nodes: `H` vertices,
  * col nodes: `W` vertices,
* Append edges with index offsets so grids are disconnected.

---

### 4. Implementation sketch (02_truth/step.py)

```python
# 02_truth/step.py
from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import igraph as ig

def canonicalize(present: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: truth (Π / canonical gauge) — union graph only
    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2.1: Disjoint-union graph G
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π / build canonicalizable structure

    Input:
      present: object from 01_present.load, containing train_in, train_out, test_in.
      trace: if True, log vertex/edge counts and component count.

    Output:
      canonical: present + union graph G. No canonical row/col orders yet.
    """

    grids: List[np.ndarray] = (
        present["train_in"]
        + present["train_out"]
        + present["test_in"]
    )

    # Compute vertex count per grid: cells + rows + cols
    vert_offsets: List[int] = []
    total_vertices = 0
    grid_shapes: List[Tuple[int, int]] = []

    for g in grids:
        H, W = int(g.shape[0]), int(g.shape[1])
        n_cells = H * W
        n_rows  = H
        n_cols  = W
        n_verts = n_cells + n_rows + n_cols
        vert_offsets.append(total_vertices)
        total_vertices += n_verts
        grid_shapes.append((H, W))

    edges: List[Tuple[int, int]] = []
    color_codes: List[int] = [0] * total_vertices

    # Build vertices and edges per grid
    for grid_idx, g in enumerate(grids):
        H, W = grid_shapes[grid_idx]
        base = vert_offsets[grid_idx]

        # Index layout (per grid):
        # cells: base + 0 .. base + H*W - 1
        # row_nodes: base + H*W .. base + H*W + H - 1
        # col_nodes: base + H*W + H .. base + H*W + H + W - 1
        def cell_index(r: int, c: int) -> int:
            return base + r * W + c

        def row_node_index(r: int) -> int:
            return base + H * W + r

        def col_node_index(c: int) -> int:
            return base + H * W + H + c

        # Set color codes for cells, row_nodes, col_nodes
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                color_codes[v] = int(g[r, c])  # 0..9

        for r in range(H):
            v = row_node_index(r)
            color_codes[v] = 100  # row_node

        for c in range(W):
            v = col_node_index(c)
            color_codes[v] = 101  # col_node

        # N4 adjacency edges (right, down) — undirected
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                if r + 1 < H:  # down neighbor
                    edges.append((v, cell_index(r + 1, c)))
                if c + 1 < W:  # right neighbor
                    edges.append((v, cell_index(r, c + 1)))

        # Incidence edges: cell↔row_node, cell↔col_node
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                edges.append((v, row_node_index(r)))
                edges.append((v, col_node_index(c)))

    # Build igraph Graph
    g_all = ig.Graph(n=total_vertices, edges=edges, directed=False)
    g_all.vs["color_code"] = color_codes

    if trace:
        comp = g_all.components()
        n_comp = len(comp)
        logging.info(
            f"[truth] union graph built: |V|={g_all.vcount()}, |E|={g_all.ecount()}, "
            f"components={n_comp}"
        )

    canonical: Dict[str, Any] = {
        "task_id": present["task_id"],
        "train_in": present["train_in"],
        "train_out": present["train_out"],
        "test_in": present["test_in"],
        "shapes": present["shapes"],
        "palettes": present["palettes"],
        "graph": g_all,
        # placeholders for later WO-1.3
        "row_orders": None,
        "col_orders": None,
    }

    return canonical
```

This is fully deterministic and within the LOC budget.

---

### 5. `run.py` impact

`run.py` stays **minimal**. No changes to structure:

```python
canonical = canonicalize(present, trace=trace)
```

Now, instead of raising `NotImplementedError`, it returns a `canonical` dict with `graph`. The pipeline will proceed to `03_scaffold.build`, which is still not implemented and will raise `NotImplementedError`.

**Key point:** run.py remains an orchestrator. All graph logic lives in `02_truth/step.py`.

---

### 6. Receipts for WO-1.2

When `--trace` is passed:

* Log line:
  `"[truth] union graph built: |V|=..., |E|=..., components=..."`

This acts as a **receipt**:

* |V| should equal the sum over all grids of `H*W + H + W`.
* `components` should equal the total number of grids, i.e.,

  ```python
  num_grids = len(present["train_in"]) + len(present["train_out"]) + len(present["test_in"])
  ```

We can optionally later add a small debug helper that asserts:

```python
assert n_comp == num_grids
```

in trace mode, but for WO-1.2 it’s fine to just log and let the reviewer check.

---

### 7. Reviewer instructions (WO-1.2 checkpoint)

**Commands:**

```bash
python run.py --task-id 00576224 --data data/arc-agi_training_challenges.json --trace
python run.py --task-id 1caeab9d --data data/arc-agi_training_challenges.json --trace
python run.py --task-id 4c4377c9 --data data/arc-agi_training_challenges.json --trace
```

**Expected behavior:**

* `present.load` should succeed and log shapes/palettes.

* `truth.canonicalize` should now succeed and log something like:

  ```text
  [truth] union graph built: |V|=..., |E|=..., components=5
  ```

  for `00576224`, where:

  * `components = number of grids = len(train_in) + len(train_out) + len(test_in) = 2 + 2 + 1 = 5`,

  * `|V|` should be:

    [
    \sum_{\text{grids}} (H \cdot W + H + W)
    ]

    For 00576224:

    * two `2×2` grids → each has `4 + 2 + 2 = 8` vertices,
    * two `6×6` grids → each has `36 + 6 + 6 = 48` vertices,
    * one `2×2` test grid → `4 + 2 + 2 = 8` vertices.
      Total: `8 + 8 + 48 + 48 + 8 = 120`.
      So we should see `|V|=120`.

  * `|E|` should be consistent with:

    * Per grid: N4 internal edges + `H*W` row incidence + `H*W` col incidence; we don’t require the exact number in this WO, but it should be positive and scale logically with the grid size.

* The pipeline will then call `03_scaffold.build` and hit `NotImplementedError` there. That’s expected.

**How to identify legit gap vs bug:**

* **Legit WO-1.2 gap:**

  * `present` logs look correct as in WO-1.1.
  * `truth` logs something like `|V|=120, |E|=... , components=5`.
  * Then `03_scaffold.build` raises `NotImplementedError`.

* **Bug (spec/implementation mismatch):**

  * `truth` never logs (canonicalize not called).
  * `graph` has wrong number of components (not equal to total number of grids).
  * `color_code` for vertices includes different values per grid that encode grid_id (e.g., `1000 + color`), which would violate the “no grid_id in colors” rule.
  * `edges` include connections between grids (components < num_grids).

**Math / implementation consistency:**

* Reviewer should confirm that:

  * Vertex color codes belong to `{0..9, 100, 101}` and **do not** depend on grid index, as per `00_MATH_SPEC.md` §2.1.
  * Connected components count equals number of grids, confirming disjoint union.
  * `run.py` still doesn’t do any math; all the graph logic is confined to `02_truth/step.py`.

---

### 8. Optimization / CPU concerns

* Graph sizes are tiny (≤ 30×30 grids, small train/test counts), so building the union graph in Python and feeding it to igraph is cheap.
* No reason to skip edges, reduce adjacency, or otherwise approximate; we should build the full graph as spec’d.
* No optimization tricks are introduced; everything is straightforward loops and a single `igraph.Graph` constructor.

---