## WO-2.1 — `03_scaffold`: Frame detector (per-output, border-based) (~80–120 LOC)

### Intent

For **each canonical training output grid** `train_out[i]`, compute a **frame_maskᵢ** that marks its **outer border** (top row, bottom row, left column, right column) in canonical coordinates. This gives us the **canvas frame** per grid; distances and inner region come later.

We store it as:

```python
scaffold = {
    "per_output": [
        {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": frame_mask_i,  # H_i×W_i bool
        },
        ...
    ]
}
```

No distances, no inner yet. Only per-grid border masks.

---

## 0. Anchors the implementer must read

Before coding this WO, the implementer must read:

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §2 (Stage A) to recall that we are in **canonical coordinates per grid** (disjoint components, no cross-grid alignment).
   * §4.2 (“For each output grid (train_out and the test canvas size once fixed) build adjacency and distances…”) — this is explicitly **per output grid**.

2. ` @docs/anchors/01_STAGES.md `

   * “scaffold”:

     > “on **train_out only**, find the stable canvas geometry (frame) and compute distance fields; define the inner region.”
     > Here “frame” is now clarified (by author AI) to mean the **canvas frame** of each grid (its border), not a global cross-output intersection.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * Section “scaffold”:

     * “scaffold — Where the story lives (output-intrinsic space)”
     * Distances from **frame or border**; Stage F is pure geometry, Stage N handles “cells identical across train_out”.

4. Author clarification you pasted

   * Stage F is **per output grid**, and
   * the simplest, universal frame Fₓ is the **outer border** of that grid in canonical coords.

I consciously **ignore** the old global F = {cells identical across all train_out} for Stage F, because the author explicitly moved that into Stage N (law-level), not scaffold.

---

## 1. Libraries to use

Only standard, well-documented libs:

* `numpy`

  * For shapes and creating boolean masks.

* `typing`

  * `Dict`, `Any`, `List`.

* `logging`

  * For trace-level receipts.

No BFS, no graph libraries, no heuristics in this WO.

---

## 2. Input & output contract

### Input

`03_scaffold.step.build(canonical, trace=False)` receives:

```python
canonical = {
    "task_id":  str,
    "train_in":  List[np.ndarray],  # H×W canonical per input
    "train_out": List[np.ndarray],  # H×W canonical per output
    "test_in":   List[np.ndarray],
    "shapes":    dict,
    "palettes":  dict,
    "graph":     ig.Graph,
    "row_orders": List[List[int]],
    "col_orders": List[List[int]],
}
```

We **only care about** `canonical["train_out"]` in this WO.

### Output (from WO-2.1)

We return a `scaffold` dict:

```python
scaffold = {
    "per_output": [
        {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": np.ndarray[bool],   # outer border of grid i
        }
        for i, Y_i in enumerate(canonical["train_out"])
    ]
}
```

Later WOs will extend each entry with distances, inner, parity, thickness, period hints, etc., and may add `scaffold["aggregated"]`. This WO only creates frame masks.

---

## 3. Algorithm (strict, no hacks)

For each canonical training output grid `Y_i`:

* Let `H, W = Y_i.shape`.
* Define `frame_mask_i` as a boolean H×W array where:

  ```python
  frame[r, c] = True if r == 0 or r == H-1 or c == 0 or c == W-1
                else False
  ```

That’s it.

Why this is spec-correct and **not a new invention**:

* From 00_MATH_SPEC §4.2: distances are defined “for each output grid X” from “frame or border”; the clarification says we choose the **border** as the frame Fₓ for geometry.
* From 01_STAGES + 02_QUANTUM_MAPPING: scaffold is **output-intrinsic** geometry; law-level invariants (“cells identical across train_out”) are discovered in Stage N via atoms (`d_*` + color).
* The author explicitly reclassified the old global F into Stage N. We’re following that.

We do **not** compare grids to each other. No cross-output logic here. All per-grid.

---

## 4. Implementation sketch (`03_scaffold/step.py`)

```python
# 03_scaffold/step.py
from typing import Any, Dict, List
import logging

import numpy as np

def _frame_for_output(Y: np.ndarray) -> np.ndarray:
    """
    Per-grid frame: the outer border of the canonical output grid.

    Stage F is output-intrinsic (per grid), not global across train_out.
    The border frame is purely geometric and later atoms/distance fields
    will use it. Law-level invariants like "all border cells are 8" are
    discovered in Stage N, not baked here.

    Anchors:
      - 00_MATH_SPEC.md §4.2: For each output grid, build adjacency and distances
      - 01_STAGES.md: scaffold = geometry on train_out
      - 02_QUANTUM_MAPPING.md: WHERE vs WHAT separation
    """
    H, W = Y.shape
    frame = np.zeros((H, W), dtype=bool)
    if H == 0 or W == 0:
        return frame
    frame[0, :] = True
    frame[H - 1, :] = True
    frame[:, 0] = True
    frame[:, W - 1] = True
    return frame

def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — WO-2.1 Frame detector (per-output, border-based)

    Input:
      canonical: dict from 02_truth.canonicalize, containing canonical train_out grids.

    Output:
      scaffold: {
        "per_output": [
          {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": H_i×W_i bool (outer border),
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
        entry = {
            "index": i,
            "shape": (H, W),
            "frame_mask": frame_mask,
        }
        if trace:
            logging.info(
                f"[scaffold] output#{i}: shape={entry['shape']}, "
                f"frame_sum={int(frame_mask.sum())}"
            )
        per_output.append(entry)

    scaffold: Dict[str, Any] = {
        "per_output": per_output,
    }

    return scaffold
```

---

## 5. `run.py` changes

None.

* `run.py` continues to be:

  ```python
  scaffold = build_scaffold(canonical, trace=trace)
  ```

* `run.py` doesn’t inspect scaffold internals; later WOs (2.2, 2.3) will extend `scaffold`.

No “god function”, no extra logic in `run.py`. Integration for later WOs stays trivial: they just read/extend `scaffold["per_output"]`.

---

## 6. Receipts for WO-2.1

With `--trace`, you’ll see per-output log lines like:

```text
[scaffold] output#0: shape=(5, 6), frame_sum=18
[scaffold] output#1: shape=(7, 7), frame_sum=24
...
```

For each grid:

* `frame_sum` should equal the number of border cells: `2*H + 2*(W-2)` (if H,W ≥ 2).

Reviewer can use these logs as receipts to confirm:

* The right number of outputs (one per train_out) is present,
* Each has correct `shape`,
* Border size matches expectation.

If you want JSON receipts later, we can add a small dump in the golden harness, but WO-2.1 just needs these logs.

---

## 7. Reviewer instructions

To test WO-2.1:

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

* `present` + `truth` behave as already verified for 00576224 (golden).
* `scaffold`:

  * For each training output grid:

    * Logs one line with `shape` (per grid’s H×W in canonical coords),
    * `frame_sum > 0` and equals `2*H + 2*(W-2)` if H,W≥2,
    * No exceptions, no cross-output shape assumptions.

**Legit implementation vs unsatisfiable task:**

* There is no “unsatisfiable” case here: every grid has a border, so frame is always well-defined.
* If `frame_sum` is 0 for any grid with H,W>0, that’s an implementation bug in `_frame_for_output`.
* If `shape` values don’t match the canonical train_out shapes, that’s a Stage A or scaffolding bug.

Later, scaffold goldens for each task (like `*_scaffold.json`) should match:

* `per_output[i].shape`,
* `frame_mask` border structure,
* `frame_sum` values.

If a golden still expects a **different frame definition** (e.g., non-border), that means the golden predates this clarified spec and needs to be regenerated to match the new, consistent definition.

---
