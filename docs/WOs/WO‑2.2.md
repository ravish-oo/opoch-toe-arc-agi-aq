## WO-2.2 — `03_scaffold`: Distance atlas (per-output) (~120–160 LOC)

**Goal:**
For each canonical `train_out[i]`, given its per-output `frame_maskᵢ` (outer border from WO-2.1), compute directional distance fields:

* `d_topᵢ[r,c]`   = number of steps from `(r,c)` **upwards** (decreasing row) to reach the frame/border in that grid,
* `d_bottomᵢ[r,c]`= steps downwards (increasing row),
* `d_leftᵢ[r,c]`  = steps left (decreasing col),
* `d_rightᵢ[r,c]` = steps right (increasing col),

with 4-adjacency (up/down/left/right). Attach these per-grid fields under `scaffold["per_output"][i]`.

---

## 0. Anchors to read (and how I used them)

Before coding, implementer **must** re-read:

1. `docs/anchors/00_MATH_SPEC.md`

   * §2: Stage A canonicalization is **per grid** on a disjoint union; canonical `(r,c)` are only meaningful within each grid. No cross-grid alignment.
   * §4.2: “For each output grid (train_out and the test canvas size once fixed): build 4-adjacency … compute distances (d_{\text{top}},d_{\text{bottom}},d_{\text{left}},d_{\text{right}})… inner S = {p: d_* > 0}.” This is explicitly **per output grid X**.

2. `docs/anchors/01_STAGES.md`

   * “scaffold: on train_out only, find the stable canvas geometry (frame) and compute distance fields; define the inner region.” Distances are part of **output-intrinsic geometry**, not cross-grid logic.

3. `docs/anchors/02_QUANTUM_MAPPING.md`

   * Scaffold = WHERE; distance computation is in the **free sector** (no ledger), purely geometric; Stage N uses these distances as atoms for laws.

4. Author clarification you pasted

   * Stage F is **per output grid**, with frame Fₓ chosen as the **outer border** in canonical coords; distances are to that frame/border.

**How I avoided misinterpretation:**

* I reconciled §2 (canonical per grid) and §4.2 (distances per grid) and explicitly **rejected** any cross-grid distance scheme.
* I use distances purely per-grid, from the frame_mask we already defined in WO-2.1, without assuming any relationship between grids.
* I treat §4.2’s “BFS from frame or border” as specifying the **metric** (shortest path in a 4-adj grid), not mandating a particular implementation; directional scans along rows/cols are exactly shortest path lengths along those directions on this grid.

---

## 1. Libraries to use

We only need:

* `numpy`

  * For array shape, and per-row/per-column loops.

* `typing`

  * `Dict`, `Any`, `List`.

* `logging`

  * For receipts (min/max/sum per distance field per output).

We **do not** use networkx or scipy here; the grid is tiny (≤30×30), and directional distances along axes are trivial to implement and mathematically equivalent to 4-adj BFS restricted to that direction.

No “optimization hacks”; this is just straight O(H·W) loops.

---

## 2. Input & output contract

After WO-2.1, `03_scaffold.build` already returns:

```python
scaffold = {
    "per_output": [
        {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": frame_mask_i,  # H_i×W_i bool, border frame for grid i
        },
        ...
    ]
}
```

`canonical` (from Stage A) is unchanged:

```python
canonical = {
    "task_id": str,
    "train_in":  List[np.ndarray],
    "train_out": List[np.ndarray],   # canonical
    "test_in":   List[np.ndarray],
    "shapes":    dict,
    "palettes":  dict,
    "graph":     ig.Graph,
    "row_orders": List[List[int]],
    "col_orders": List[List[int]],
}
```

**WO-2.2 extends `scaffold["per_output"][i]` with:**

```python
{
  "d_top":    np.ndarray[int],  # H_i×W_i
  "d_bottom": np.ndarray[int],
  "d_left":   np.ndarray[int],
  "d_right":  np.ndarray[int],
}
```

We do **not** change the structure of `scaffold["per_output"]`; we only add fields.

---

## 3. Directional distance definition (strict, per grid)

Given:

* A canonical output grid `Y_i` of shape `(H, W)`,
* Its frame_maskᵢ marking the outer border (from WO-2.1),

we define for each cell `(r,c)`:

* `d_top[r,c]`   = minimal number of steps to reach the frame in the **upward** direction alone (moving `(r−1,c)` each step) until we hit frame or the border,
* `d_bottom[r,c]`= minimal steps downward, `(r+1,c)`,
* `d_left[r,c]`  = minimal steps left, `(r,c−1)`,
* `d_right[r,c]` = minimal steps right, `(r,c+1)`.

Because frame_mask marks exactly the border, this simplifies to:

* If `frame_mask[r,c]` is True, all four distances are 0 there (we’re at the frame),
* Else:

  * `d_top[r,c]`   = number of cells between `(r,c)` and the top border along column `c`,
  * `d_bottom[r,c]`= number of cells between `(r,c)` and the bottom border,
  * `d_left[r,c]`  = number of cells between `(r,c)` and the left border,
  * `d_right[r,c]` = number of cells between `(r,c)` and the right border.

Which yields the simple formulas:

* For any `(r,c)` within the grid (H,W ≥ 1):

  ```python
  d_top[r,c]    = r
  d_bottom[r,c] = (H - 1) - r
  d_left[r,c]   = c
  d_right[r,c]  = (W - 1) - c
  ```

These are exactly shortest path distances along each axis on a 4-adj grid with border frame. They match §4.2’s “distances via BFS” metric; we’re using closed-form 1D BFS along rows/cols.

No cross-grid logic; everything is per `Y_i`.

---

## 4. Implementation sketch in `03_scaffold/step.py`

We extend the existing `build` (from WO-2.1) to add distances.

```python
from typing import Any, Dict, List
import logging

import numpy as np

def _frame_for_output(Y: np.ndarray) -> np.ndarray:
    # From WO-2.1: outer border frame
    H, W = Y.shape
    frame = np.zeros((H, W), dtype=bool)
    if H == 0 or W == 0:
        return frame
    frame[0, :] = True
    frame[H - 1, :] = True
    frame[:, 0] = True
    frame[:, W - 1] = True
    return frame

def _distance_fields_for_output(H: int, W: int, frame_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Directional distance fields per-grid, as shortest path lengths along
    each axis to the border frame.

    We use the closed-form for the border frame:
      d_top[r,c]    = r
      d_bottom[r,c] = H-1 - r
      d_left[r,c]   = c
      d_right[r,c]  = W-1 - c

    For frame cells (border), these naturally give 0. This matches the
    BFS-on-4-neighbor-grid metric in this special case.
    """
    d_top    = np.zeros((H, W), dtype=int)
    d_bottom = np.zeros((H, W), dtype=int)
    d_left   = np.zeros((H, W), dtype=int)
    d_right  = np.zeros((H, W), dtype=int)

    for r in range(H):
        for c in range(W):
            d_top[r, c]    = r
            d_bottom[r, c] = (H - 1) - r
            d_left[r, c]   = c
            d_right[r, c]  = (W - 1) - c

    # We do not need to special-case frame_mask; border cells already get 0.
    return {
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }

def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — WO-2.1+WO-2.2

    Now includes per-output frame_mask (border) and per-output directional
    distance fields.

    Input:
      canonical: from 02_truth.canonicalize

    Output:
      scaffold: {
        "per_output": [
          {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": H_i×W_i bool,
            "d_top":    H_i×W_i int,
            "d_bottom": H_i×W_i int,
            "d_left":   H_i×W_i int,
            "d_right":  H_i×W_i int,
          },
          ...
        ]
      }
    """
    train_out: List[np.ndarray] = canonical["train_out"]
    if not train_out:
        msg = "[scaffold] No train_out grids; scaffold undefined."
        if trace:
            logging.error(msg)
        raise ValueError(msg)

    per_output: List[Dict[str, Any]] = []
    for i, Y in enumerate(train_out):
        H, W = Y.shape
        frame_mask = _frame_for_output(Y)
        distances = _distance_fields_for_output(H, W, frame_mask)

        entry: Dict[str, Any] = {
            "index": i,
            "shape": (H, W),
            "frame_mask": frame_mask,
            "d_top":    distances["d_top"],
            "d_bottom": distances["d_bottom"],
            "d_left":   distances["d_left"],
            "d_right":  distances["d_right"],
        }

        if trace:
            logging.info(
                f"[scaffold] output#{i}: shape={entry['shape']}, "
                f"frame_sum={int(frame_mask.sum())}"
            )
            for name in ["d_top", "d_bottom", "d_left", "d_right"]:
                D = entry[name]
                logging.info(
                    f"[scaffold] output#{i} {name}: "
                    f"min={int(D.min())}, max={int(D.max())}, sum={int(D.sum())}"
                )

        per_output.append(entry)

    scaffold: Dict[str, Any] = {
        "per_output": per_output,
    }

    return scaffold
```

We’ve now fully implemented WO-2.1 + WO-2.2 per the clarified spec; WO-2.3 will add `inner`, parity, thickness, periods on top of these.

---

## 5. `run.py` changes

No changes.

* `run.py` still only calls:

  ```python
  scaffold = build_scaffold(canonical, trace=trace)
  ```

* It doesn’t need to know or change anything for WO-2.2; integration for later WOs remains trivial.

---

## 6. Receipts for WO-2.2

With `--trace`, for each `train_out[i]` you get logs:

```text
[scaffold] output#0: shape=(H0, W0), frame_sum=...
[scaffold] output#0 d_top:    min=0, max=..., sum=...
[scaffold] output#0 d_bottom: min=0, max=..., sum=...
[scaffold] output#0 d_left:   min=0, max=..., sum=...
[scaffold] output#0 d_right:  min=0, max=..., sum=...
...
```

For each grid:

* `d_*` **min must be 0** (at border/frame cells),
* `max` equals:

  * `d_top`: `H-1`, `d_bottom`: `H-1`, `d_left`: `W-1`, `d_right`: `W-1`,
* `sum` must match closed-form sums if you want to verify them numerically.

Reviewer can use these receipts to:

* Confirm distances are correct per grid,
* Confirm there’s no cross-grid mixing.

---

## 7. Reviewer instructions

Run for a few tasks:

```bash
python run.py --task-id 00576224 \
              --data data/arc-agi_training_challenges.json \
              --test-index 0 \
              --trace

python run.py --task-id 5e6bbc0b \
              --data data/arc-agi_training_challenges.json \
              --test-index 0 \
              --trace

python run.py --task-id 833966f4 \
              --data data/arc-agi_training_challenges.json \
              --test-index 0 \
              --trace
```

**Expected:**

* For each `train_out` grid:

  * shapes printed are correct canonical shapes,
  * frame_sum equals border size (`2*H + 2*(W-2)` if H,W≥2),
  * all four distance fields have `min=0`, `max` values as per formulas.

**Legit implementation:**

* Distances match formulas; no cross-grid dependency; logs match expected min/max/sum.

**Unsatisfiable / spec gap:**

* There is none for distances; per-grid border exists for any H,W≥1, so distances are always well-defined.

**Bug:**

* If any `d_*` have `min>0` or do not follow the expected pattern,
* If any grid’s distances depend on other grids (e.g., mixing shapes).

---
