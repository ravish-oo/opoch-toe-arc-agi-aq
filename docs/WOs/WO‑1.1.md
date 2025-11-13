## WO-1.1 — `01_present/step.py` JSON loader (80–120 LOC)

**Goal:**
Implement `01_present.step.load` so that it:

* Parses a single ARC task from `arc-agi_training_challenges.json`,
* Produces `train_in`, `train_out`, `test_in` as `np.ndarray` with integer colors,
* Validates palette ∈ {0..9} and H,W ≤ 30,
* Emits a **present object** that the later stages can use,
* Matches the **Milestone 1 golden** for task `00576224`. 

This WO must be **over-specific** but strictly grounded in anchors:

* `00_MATH_SPEC.md` §1 (task representation),
* `01_STAGES.md` “present”,
* `02_QUANTUM_MAPPING.md` “present = load all grids into awareness”.

---

### 0. Anchors to read before coding

**Mandatory reading for implementer:**

1. `docs/anchors/00_MATH_SPEC.md`

   * §1 “Representing the task” (what a grid is: H×W cells, colors 0..9).

2. `docs/anchors/01_STAGES.md`

   * Section “present”: describes that we just load data into awareness, no Π, no invariants.

3. `docs/anchors/02_QUANTUM_MAPPING.md`

   * Part that maps “present” to “load the whole problem into awareness at once”.

4. `M1_M2_checkpoints.md` — Milestone 1 golden for `00576224`. 

   * This is the **ground truth** for what `present.load` must output for that task (arrays, shapes, palettes).

---

### 1. Libraries to use (mature & standard)

For this WO we need:

* `json` (stdlib)

  * Used by `run.py` to read the big `arc-agi_training_challenges.json` once.
  * **No custom JSON parsing** logic in `present`; it just receives `raw_task: dict`.

* `numpy`

  * `np.array(list_of_lists, dtype=np.int8)` to convert grids to arrays.
  * `np.unique()` to compute palettes.
  * `np.shape` / `.shape` to get H,W.

* `typing`

  * For type hints `Dict`, `Any`.

* `logging`

  * For trace logs and simple receipts.

We do **not** implement any algorithm beyond mapping Python lists to `numpy` arrays and validating simple constraints.

---

### 2. Input & output contract for `present.load`

#### Input

From `run.py` (WO-0.1):

```python
task_bundle = {
    "task_id": task_id,           # string like "00576224"
    "raw_task": raw_task_dict,    # one entry from arc-agi_training_challenges.json
}
```

Where `raw_task_dict` has this structure (from training JSON + golden):

```json
{
  "train": [
    { "input":  [[...], [...], ...],
      "output": [[...], [...], ...] },
    ...
  ],
  "test": [
    { "input": [[...], [...], ...] },
    ...
  ]
}
```

We only support the canonical ARC structure: list of train pairs, list of test inputs. WO-1.1 does **not** need to handle multiple test outputs etc. The anchor math spec assumes exactly this: `train_in`, `train_out`, `test_in`. 

#### Output: `present` object

Return a simple dict:

```python
present = {
    "task_id": task_id: str,
    "train_in":  List[np.ndarray[int8]]   # each H×W
    "train_out": List[np.ndarray[int8]]   # each H×W
    "test_in":   List[np.ndarray[int8]],  # usually length 1
    "shapes": {
        "train_in":  List[[H_i, W_i]],
        "train_out": List[[H_i, W_i]],
        "test_in":   [H_test, W_test],
    },
    "palettes": {
        "train_in":  List[List[int]],   # sorted unique colors in each train_in grid
        "train_out": List[List[int]],   # sorted unique colors in each train_out grid
        "test_in":   List[List[int]],   # sorted unique colors in each test_in grid
    }
}
```

This structure is exactly what the Milestone 1 golden shows for `00576224`. 
Later stages (`truth`, `scaffold`, etc.) can treat `present` as opaque and just read the arrays they need.

---

### 3. Implementation details (strict and explicit)

Inside `01_present/step.py`:

#### 3.1 Grid conversion

For each training example:

```python
import numpy as np

def _to_grid(arr_like):
    g = np.array(arr_like, dtype=np.int8)   # 2D
    if g.ndim != 2:
        raise ValueError("Grids must be 2D")
    H, W = g.shape
    if H <= 0 or W <= 0 or H > 30 or W > 30:
        raise ValueError(f"Invalid grid size H={H}, W={W} (must be 1..30).")
    if g.min() < 0 or g.max() > 9:
        raise ValueError("Palette out of range: must be in 0..9.")
    return g
```

**Palette & shapes:**

```python
def _shapes_and_palette(grids):
    shapes = []
    palettes = []
    for g in grids:
        H, W = g.shape
        shapes.append([int(H), int(W)])
        colors = np.unique(g)
        palettes.append(sorted(int(c) for c in colors.tolist()))
    return shapes, palettes
```

#### 3.2 Putting it together in `load`

```python
from typing import Any, Dict
import logging
import numpy as np

def load(task_bundle: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: present (awareness)
    Anchor:
      - 01_STAGES.md: present
      - 00_MATH_SPEC.md §1: Representing the task
      - 02_QUANTUM_MAPPING.md: 'present' = load all grids into awareness
      - M1_M2_checkpoints.md: Milestone 1 golden for task 00576224

    Input:
      task_bundle: {
        "task_id": str,
        "raw_task": dict from arc-agi_training_challenges.json
      }
      trace: if True, log shapes and palettes.

    Output:
      present: {
        "task_id": str,
        "train_in":  [np.ndarray[int8], ...],
        "train_out": [np.ndarray[int8], ...],
        "test_in":   [np.ndarray[int8], ...],
        "shapes": {...},
        "palettes": {...},
      }
    """
    task_id = task_bundle["task_id"]
    raw_task = task_bundle["raw_task"]

    train_in_arrays = []
    train_out_arrays = []
    for pair in raw_task["train"]:
        g_in  = _to_grid(pair["input"])
        g_out = _to_grid(pair["output"])
        train_in_arrays.append(g_in)
        train_out_arrays.append(g_out)

    test_in_arrays = []
    for t in raw_task["test"]:
        g_test = _to_grid(t["input"])
        test_in_arrays.append(g_test)

    # shapes & palettes
    shapes_train_in, palettes_train_in   = _shapes_and_palette(train_in_arrays)
    shapes_train_out, palettes_train_out = _shapes_and_palette(train_out_arrays)
    shapes_test_in, palettes_test_in     = _shapes_and_palette(test_in_arrays)

    present = {
        "task_id": task_id,
        "train_in":  train_in_arrays,
        "train_out": train_out_arrays,
        "test_in":   test_in_arrays,
        "shapes": {
            "train_in":  shapes_train_in,
            "train_out": shapes_train_out,
            "test_in":   shapes_test_in[0] if len(shapes_test_in) == 1 else shapes_test_in,
        },
        "palettes": {
            "train_in":  palettes_train_in,
            "train_out": palettes_train_out,
            "test_in":   palettes_test_in,
        },
    }

    if trace:
        logging.info(f"[present] task_id={task_id}")
        logging.info(f"[present] train_in shapes={shapes_train_in}")
        logging.info(f"[present] train_out shapes={shapes_train_out}")
        logging.info(f"[present] test_in shapes={shapes_test_in}")
        logging.info(f"[present] train_in palettes={palettes_train_in}")
        logging.info(f"[present] train_out palettes={palettes_train_out}")
        logging.info(f"[present] test_in palettes={palettes_test_in}")

    return present
```

This is well within 80–120 LOC and uses only numpy for conversions and validation.

---

### 4. `run.py` changes for this WO

`run.py` stays **minimal**, but now:

* `01_present.step.load` is implemented, so `run_task` will successfully return a `present` object to `truth.canonicalize`.
* `truth.canonicalize` is still a stub and will raise `NotImplementedError`.

The only optional tweak:

* In `run_task`, you might want to **log** something after `present` returns if `trace=True`, but that’s optional because `present.load` already logs shapes/palettes.

No new logic in `run.py` beyond what we already built for WO-0.1.

---

### 5. Receipts for WO-1.1

For this milestone, receipts = shapes & palettes + the actual array contents.

* The **log lines** in `present.load` when `trace=True` serve as quick receipts: they show shapes and palettes.
* For deep verification, the reviewer should compare output against **Milestone 1 golden** for task `00576224`. 

You may, optionally in this WO:

* Add a helper in `present.load` (for `trace=True`) that writes a JSON file like `traces/present_00576224.json` with:

  ```json
  {
    "task_id": "00576224",
    "train_in":  [[[...]], [[...]], ...],
    "train_out": [[[...]], [[...]], ...],
    "test_in":   [[[...]]],
    "shapes": { ... },
    "palettes": { ... }
  }
  ```

  So the reviewer can diff it directly against the golden JSON in `M1_M2_checkpoints.md`.

---

### 6. Reviewer instructions (Milestone 1 checkpoint)

**Commands to run:**

```bash
python run.py --task-id 00576224 --data data/arc-agi_training_challenges.json --trace
```

(May also try a couple more tasks like `1caeab9d`, `4c4377c9` for sanity, but the golden is for `00576224`.)

**What to expect:**

* `present.load` should **succeed** and log shapes & palettes for train_in, train_out, test_in.
* `truth.canonicalize` will then be called and raise `NotImplementedError`. So the run will terminate there, which is fine for this milestone.

**Checkpoint against golden (critical):**

For `task_id="00576224"`, `present.load` must produce exactly:

* `train_in` grids:

  ```python
  [[[7, 9],
    [4, 3]],
   [[8, 6],
    [6, 4]]]
  ```

* `train_out` grids:

  ```python
  [[[7, 9, 7, 9, 7, 9],
    [4, 3, 4, 3, 4, 3],
    [9, 7, 9, 7, 9, 7],
    [3, 4, 3, 4, 3, 4],
    [7, 9, 7, 9, 7, 9],
    [4, 3, 4, 3, 4, 3]],
   [[8, 6, 8, 6, 8, 6],
    [6, 4, 6, 4, 6, 4],
    [6, 8, 6, 8, 6, 8],
    [4, 6, 4, 6, 4, 6],
    [8, 6, 8, 6, 8, 6],
    [6, 4, 6, 4, 6, 4]]]
  ```

* `test_in`:

  ```python
  [[[3, 2],
    [7, 8]]]
  ```

* `shapes` and `palettes` as in the golden JSON. 

If any of these differ (including shape order or color content), **WO-1.1 is incorrect**.

**Legit gap vs bug:**

* **Legit WO-1.1 gap:**

  * `present.load` produces the correct arrays/shapes/palettes for `00576224`, but the pipeline crashes at `truth.canonicalize` (NotImplementedError). That is expected until WO-1.2.

* **Bug / spec mismatch:**

  * Grids differ from golden in any value.
  * H,W not 2×2 and 6×6 as per golden.
  * Palettes are not the sorted unique color sets per grid as in golden.
  * `present.load` attempts canonicalization or any math beyond loading (that belongs to `truth`).

---

### 7. Optimization / CPU concerns

* This WO does only list→numpy conversion and simple validations.
* CPU cost is negligible even for the full training file.
* No “simplifying” approximations are allowed or needed here; we must load all grids exactly.

---
