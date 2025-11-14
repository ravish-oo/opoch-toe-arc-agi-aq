## WO-2.2 — `03_scaffold`: distance atlas (≈120–180 LOC inside `build` + helpers)

**Goal:**
Compute directional distance fields over the **canonical** test-output canvas:

* If a **frame exists** (from WO-2.1): distance to nearest frame cell,
* Else: distance to the **outer border**,
* Four integer fields: `d_top`, `d_bottom`, `d_left`, `d_right`,
* Inner region in WO-2.3 will use these.

The math spec:

> For each output grid X (training outputs and test canvas): build 4-adjacency graph;
> For each cell u, compute distance to the nearest frame cell, or if no frame, to the grid border:
> (d_{\text{top}}(u)): minimal steps in −row direction to frame/border, etc.

We implement this as **directional distances along straight lines** (vertical and horizontal), which matches that definition and is simpler than a full generic BFS.

**Acceptance:**

* For any task:

  * `d_top, d_bottom, d_left, d_right` have `min == 0`,
  * Border cells (or frame cells when frame exists) have distance 0 in the appropriate direction,
  * Logs show reasonable max values and a checksum.
* For a Milestone 2 “distance golden” task from `M1_M2_checkpoints.md`, sums and extrema align with golden stats.

---

### 0. Anchors to read before coding

Mandatory:

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §4.2 “Distance fields”: definition of `d_top, d_bottom, d_left, d_right` as directional distances to frame or border.

2. ` @docs/anchors/01_STAGES.md `

   * Section “scaffold”: this stage computes WHERE: frame + distances + inner region, from **train_out-only** (for structure) and applies to test canvas.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * “WHERE” mapping: distances are pure geometry, no rules.

4. ` @goldens/00576224/M1_M2_checkpoints.md `

   * Milestone 2 distance examples: which task_ids to use, and what distance stats (min/max/sum) we expect for sanity.

---

### 1. Libraries to use (no reinvention)

Here we only need:

* `numpy`

  * For array slices and broadcasting in per-row/per-col scans.

* `typing`

  * `Dict`, `Any`, `List`, `Tuple`.

* `logging`

  * For receipts of min/max/sum per distance field.

We **do not** need:

* `networkx`, `scipy.ndimage`, or a general graph library here.
  The directional distance definition matches a simple linear scan per row/column; using a full BFS library would be overkill.

The math spec gave “BFS or Dijkstra” as an implementation tip, not a requirement. Straight-line directional distances via simple passes are equivalent for our use cases (midlines, inner region, parity, etc.) and easier to implement robustly.

---

### 2. Input & output contract (extending WO-2.1)

We extend `03_scaffold.build(canonical, trace=False)`.

#### Input

From WO-2.1, `build` already receives:

```python
canonical = {
    "task_id": str,
    "train_in":  List[np.ndarray],  # canonical
    "train_out": List[np.ndarray],  # canonical
    "test_in":   List[np.ndarray],  # len == 1 (canonical)
    "shapes":    dict,
    "palettes":  dict,
    "graph":     ig.Graph,
    "row_orders": List[List[int]],
    "col_orders": List[List[int]],
}
```

And returns:

```python
scaffold = {
    "frame_mask": np.ndarray[bool],     # H×W on test canvas
    "train_out_shapes": List[(H,W)],
    "has_frame": bool,
    # after WO-2.2 we’ll add:
    "d_top":    np.ndarray[int],
    "d_bottom": np.ndarray[int],
    "d_left":   np.ndarray[int],
    "d_right":  np.ndarray[int],
}
```

We assume the **reference shape** for distances is:

* The canonical shape of the **test output canvas**, which for now matches the canonical test input shape, since S0 is not yet applied. Later, S0 will define `(H_out, W_out)`, but the same directional logic applies.

Implementation-wise, at this WO we can:

* Use `H,W = canonical["test_in"][0].shape` as the grid size for distances, and
* Restrict the frame mask to this size (if needed). For typical ARC tasks, `train_out` shapes match the intended test output shape.

---

### 3. Distance fields: precise algorithm

We define directional distances along lines:

* `d_top[r,c]`: number of steps to go **upwards** (decreasing row index) until hitting a frame cell (if `has_frame=True`) or the outer border row (`r == 0`) if no frame.
* `d_bottom[r,c]`: steps **downwards** (increasing row index) until hitting frame or bottom border (`r == H-1` if no frame).
* `d_left[r,c]`: steps **left** until frame or `c == 0` (no frame).
* `d_right[r,c]`: steps **right** until frame or `c == W-1` (no frame).

Concrete rules:

* If `has_frame=True`:

  * A cell `(r,c)` that is itself in `frame_mask` has all distances 0.
  * For cells above the topmost frame in their column, `d_top` will be the number of steps to reach the first frame cell upwards.
  * If there is no frame cell in that direction for that column, we treat the border as “blocking”; distances still measure steps until leaving the grid (but in practice we won’t rely on such cells for midline/inner).

* If `has_frame=False`:

  * Distances are to **borders only**:

    * `d_top[r,c] = r`,
    * `d_bottom[r,c] = H-1 - r`,
    * `d_left[r,c] = c`,
    * `d_right[r,c] = W-1 - c`.

Implementation using simple scans:

#### Case A: no frame

Easy vectorized formulas above.

#### Case B: frame exists

We do four passes:

* Top-down (for `d_top`)
* Bottom-up (for `d_bottom`)
* Left-right (for `d_left`)
* Right-left (for `d_right`)

Example for `d_top`:

```python
d_top = np.zeros((H, W), dtype=int)
for c in range(W):
    dist = 0
    for r in range(H):
        if frame_mask[r, c]:
            dist = 0
        elif r == 0 and not frame_mask[r, c]:
            dist = 0  # at border; spec uses frame or border
        else:
            dist += 1
        d_top[r, c] = dist
```

We can tweak edge behavior slightly, but the idea remains: if frame exists, distances reset to 0 at frame cells; otherwise border cells have 0 (ensuring `min == 0` on all fields).

Same pattern for `d_bottom` (scan from bottom), `d_left` (from left), `d_right` (from right).

---

### 4. Implementation sketch in `03_scaffold/step.py`

Extend the previous `build` to include distances:

```python
# 03_scaffold/step.py
from typing import Any, Dict, List, Tuple
import logging

import numpy as np

def _detect_frame(train_out: List[np.ndarray], trace: bool = False):
    # As in WO-2.1
    shapes = [g.shape for g in train_out]
    unique_shapes = sorted(set(shapes))
    H, W = unique_shapes[0]
    stack = np.stack(train_out, axis=0)
    base = stack[0]
    equal_all = np.all(stack == base, axis=0)
    frame_mask = equal_all.astype(bool)
    return frame_mask, shapes

def _distance_fields(frame_mask: np.ndarray, has_frame: bool) -> Dict[str, np.ndarray]:
    H, W = frame_mask.shape

    d_top    = np.zeros((H, W), dtype=int)
    d_bottom = np.zeros((H, W), dtype=int)
    d_left   = np.zeros((H, W), dtype=int)
    d_right  = np.zeros((H, W), dtype=int)

    if not has_frame:
        # Distances to borders only
        for r in range(H):
            for c in range(W):
                d_top[r, c]    = r
                d_bottom[r, c] = H - 1 - r
                d_left[r, c]   = c
                d_right[r, c]  = W - 1 - c
        return {
            "d_top": d_top,
            "d_bottom": d_bottom,
            "d_left": d_left,
            "d_right": d_right,
        }

    # With frame: reset distances to 0 at frame cells; otherwise count steps to frame/border.

    # Top-down
    for c in range(W):
        dist = 0
        for r in range(H):
            if frame_mask[r, c]:
                dist = 0
            elif r == 0:
                dist = 0  # border
            else:
                dist += 1
            d_top[r, c] = dist

    # Bottom-up
    for c in range(W):
        dist = 0
        for r in reversed(range(H)):
            if frame_mask[r, c]:
                dist = 0
            elif r == H - 1:
                dist = 0
            else:
                dist += 1
            d_bottom[r, c] = dist

    # Left-right
    for r in range(H):
        dist = 0
        for c in range(W):
            if frame_mask[r, c]:
                dist = 0
            elif c == 0:
                dist = 0
            else:
                dist += 1
            d_left[r, c] = dist

    # Right-left
    for r in range(H):
        dist = 0
        for c in reversed(range(W)):
            if frame_mask[r, c]:
                dist = 0
            elif c == W - 1:
                dist = 0
            else:
                dist += 1
            d_right[r, c] = dist

    return {
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }

def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — frame + distances (WO-2.1 + WO-2.2)
    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4.1-4.2: Frame & distance fields
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic geometry

    Input:
      canonical: from 02_truth.canonicalize

    Output:
      scaffold: {
        "frame_mask": np.ndarray[bool],
        "train_out_shapes": List[(H,W)],
        "has_frame": bool,
        "d_top": np.ndarray[int],
        "d_bottom": np.ndarray[int],
        "d_left": np.ndarray[int],
        "d_right": np.ndarray[int],
      }
    """
    train_out = canonical["train_out"]  # canonical grids
    frame_mask, shapes = _detect_frame(train_out, trace=trace)
    has_frame = bool(frame_mask.any())

    distances = _distance_fields(frame_mask, has_frame)

    scaffold = {
        "frame_mask": frame_mask,
        "train_out_shapes": shapes,
        "has_frame": has_frame,
        "d_top": distances["d_top"],
        "d_bottom": distances["d_bottom"],
        "d_left": distances["d_left"],
        "d_right": distances["d_right"],
    }

    if trace:
        H, W = frame_mask.shape
        logging.info(
            f"[scaffold] frame_mask shape={frame_mask.shape}, "
            f"sum={int(frame_mask.sum())}, has_frame={has_frame}"
        )
        for name in ["d_top", "d_bottom", "d_left", "d_right"]:
            D = distances[name]
            logging.info(
                f"[scaffold] {name}: shape={D.shape}, min={int(D.min())}, "
                f"max={int(D.max())}, sum={int(D.sum())}"
            )

    return scaffold
```

This integrates WO-2.1 and WO-2.2. If you prefer to keep WOs separate, `_distance_fields` can be added in WO-2.2 into the existing `build`.

---

### 5. `run.py` behavior & changes

No changes to `run.py` are needed:

```python
scaffold  = build_scaffold(canonical, trace=trace)
```

Now `build_scaffold` returns `scaffold` with distances included. The pipeline then proceeds to `size_choice` (still unimplemented).

`run.py` remains a minimal orchestrator.

---

### 6. Receipts for WO-2.2

Under `--trace`, `build` logs:

* `frame_mask` stats (from WO-2.1),
* For each distance field (`d_top`, `d_bottom`, `d_left`, `d_right`):

  * `shape`,
  * `min`, `max`,
  * `sum`.

Reviewer can check:

* `min` must be 0 for all four fields,
* For a no-frame case:

  * `d_top[r, c] == r`,
  * `d_bottom[r, c] == H-1-r`, etc. (the sums should match known formulas; golden may give reference sums),
* For a frame case:

  * All frame cells `frame_mask == True` should have 0 in all four directions,
  * As you move away from frame/border, distances increase monotonically along lines.

---

### 7. Reviewer instructions (WO-2.2 checkpoint)

**Commands:**

1. A border-frame task (from `M1_M2_checkpoints.md`):

   ```bash
   python run.py --task-id <frame_task_id> \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

2. A task with no stable frame (like `00576224`):

   ```bash
   python run.py --task-id 00576224 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

**Expected behavior:**

* For the border-frame task:

  * `frame_mask.sum() > 0`, `has_frame=True`.
  * For all four distance fields:

    * `min == 0`,
    * logged `max` and `sum` match the rough golden stats in `M1_M2_checkpoints.md` (or at least are plausible).
  * Pipeline then hits `NotImplementedError` at `size_choice.choose`.

* For `00576224`:

  * `frame_mask.sum()` may be 0, `has_frame=False`.
  * Distances should match the simple border formulas:

    * For 6×6, `d_top` min=0, max=5, sum = 0+1+2+3+4+5 repeated across columns, etc.
  * Again, pipeline stops at `size_choice`.

**Legit gap vs bug:**

* **Legit gap:**

  * Distances logged as described,
  * `size_choice` still unimplemented.

* **Bug:**

  * `min` of any distance field ≠ 0,
  * `shape` of distance fields does not match the grid size,
  * `frame_mask` ignored (distances non-zero at frame cells),
  * Using `train_in` or `test_in` instead of `train_out` for frame + distances, diverging from spec.

**Math/implementation alignment:**

* Distances must be computed on the canonical grid and respect the directional definition (up/down/left/right to frame/border).

---

### 8. Optimization / CPU considerations

* Grids are ≤ 30×30; four O(H·W) passes are negligible on CPU.
* No heaps, no Dijkstra; no need to call external graph libs.
* This is exact and simple; no reason to approximate or simplify.

---
