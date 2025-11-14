## WO-2.1 — `03_scaffold`: frame detector (≈80–120 LOC inside `build`)

**Goal:**
In canonical coordinates, compute the **frame mask** on **train_out**:

> A cell position (p) is in the global frame (F) iff there exists a color (k) such that (c^{(i)}_{\text{out}}(p)=k) for **all** training outputs (i).
> (00_MATH_SPEC.md §4.1)

In code: `frame_mask[r, c] = True` if all canonical `train_out[i][r, c]` have identical color.

**Acceptance:**

* On border-frame tasks (see M1_M2_checkpoints for which), `frame_mask.sum() > 0`.
* For any task, shapes are logged and consistent; build() returns a `scaffold` object instead of raising.

---

### 0. Anchors to read before coding

Mandatory:

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §4 “Stage F — Frame & distances (global relational coordinates)”

     * 4.1 Frame detection: definition of (F) as positions with same color across all train_out in canonical coords.

2. ` @docs/anchors/01_STAGES.md `

   * Section “scaffold”: describes this stage as computing WHERE: frame + distance fields + inner region, **output-intrinsic**, using train_out only.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * The “scaffold / WHERE” part: this is “space of the law,” not rules themselves.

4. ` @goldens/00576224/M1_M2_checkpoints.md `

   * Milestone 2 notes for **frame checks**: it tells you which tasks are “border-frame tasks” you should use as test cases, and what counts we expect qualitatively (non-empty frame, plausible size).

---

### 1. Libraries to use (no wheel reinvention)

For this WO we only need:

* `numpy`

  * To stack and compare multiple `train_out` grids efficiently:

    * `np.stack`, `np.all`, `np.equal`.

* `typing`

  * For `Dict`, `Any`, `List`.

* `logging`

  * For receipts (frame size, shapes) when `trace=True`.

We **do not** use any graph library or BFS here. Distances and inner region are WO-2.2/2.3. No algorithms beyond simple numpy broadcasting.

---

### 2. Input & output contract for `03_scaffold.build`

We assume WO-1.3 canonicalization is done, so `canonical` from `02_truth.canonicalize` has:

```python
canonical = {
    "task_id":    str,
    "train_in":   List[np.ndarray],  # canonical H×W per train_in
    "train_out":  List[np.ndarray],  # canonical H×W per train_out
    "test_in":    List[np.ndarray],  # canonical H×W (len == 1)
    "shapes":     dict,              # from present (original shapes)
    "palettes":   dict,
    "graph":      ig.Graph,          # union graph
    "row_orders": List[List[int]],   # per-grid old->canonical row indices
    "col_orders": List[List[int]],   # per-grid old->canonical col indices
}
```

For WO-2.1, `build` should return a `scaffold` dict:

```python
scaffold = {
    "frame_mask": np.ndarray[bool],  # H_out × W_out frame mask in canonical coords
    # placeholders for later WOs:
    "train_out_shapes": List[Tuple[int,int]],   # H,W per train_out canonical
    "has_frame": bool,
    # distance fields & inner region will be added in WO-2.2/2.3
}
```

We use **only** `canonical["train_out"]` here; no test_in, no train_in.

---

### 3. Frame detection logic (strictly per spec)

From `00_MATH_SPEC.md` §4.1:

> (F = {p : \exists k, c_{\text{out}}^{(i)}(p)=k\ \forall i}).

In code:

* Let `outs = canonical["train_out"]`, a `List[np.ndarray]`.
* Ensure **all train_out have the same canonical shape** `(H,W)`; if not, we conservatively define a **global** frame on the **overlap** or simply treat `frame_mask` as all False for WO-2.1 (safer until S0 size logic uses per-grid structures). For now, we assume most ARC tasks have same train_out shape, and we log a warning otherwise.

Implementation:

```python
import numpy as np

def _detect_frame(train_out: List[np.ndarray], trace: bool=False):
    # shapes
    shapes = [g.shape for g in train_out]
    unique_shapes = sorted(set(shapes))
    if len(unique_shapes) != 1:
        # For now: no global frame if train_out shapes differ (spec-wise: F undefined globally)
        # Later WOs may refine this; here we just log.
        if trace:
            logging.warning(f"[scaffold] train_out shapes differ: {unique_shapes}, "
                            "treating global frame as empty for now.")
        H, W = unique_shapes[0]  # arbitrary
        return np.zeros((H, W), dtype=bool), shapes

    H, W = unique_shapes[0]
    # Stack into 3D array: T × H × W
    stack = np.stack(train_out, axis=0)  # shape: (T,H,W)

    # All equal along axis 0 at each (r,c):
    # For T >= 1, we can compare everything to first slice.
    base = stack[0]
    equal_all = np.all(stack == base, axis=0)    # True if all entries equal base at (r,c)

    frame_mask = equal_all.astype(bool)

    return frame_mask, shapes
```

This is exactly the spec: if every train_out has the same color at `(r,c)`, then frame_mask[r,c] is True.

---

### 4. Implementation sketch in `03_scaffold/step.py`

```python
# 03_scaffold/step.py
from typing import Any, Dict, List, Tuple
import logging

import numpy as np

def _detect_frame(train_out: List[np.ndarray], trace: bool = False):
    shapes = [g.shape for g in train_out]
    unique_shapes = sorted(set(shapes))
    if len(unique_shapes) != 1:
        if trace:
            logging.warning(
                f"[scaffold] train_out shapes differ: {unique_shapes}; "
                "global frame_mask set to all False (no common positions)."
            )
        # Choose a canonical reference shape (first train_out) for mask size
        H, W = shapes[0]
        return np.zeros((H, W), dtype=bool), shapes

    H, W = unique_shapes[0]
    stack = np.stack(train_out, axis=0)  # T×H×W
    base = stack[0]
    equal_all = np.all(stack == base, axis=0)  # H×W bool
    frame_mask = equal_all.astype(bool)
    return frame_mask, shapes

def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — frame only (WO-2.1)
    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4.1: Frame detection from training outputs
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold (train_out-only)

    Input:
      canonical: dict from 02_truth.canonicalize with canonical train_out grids.

    Output:
      scaffold: {
        "frame_mask": np.ndarray[bool] (H×W),
        "train_out_shapes": List[(H,W)],
        "has_frame": bool,
      }
      Distance fields and inner region will be added in later WOs.
    """
    train_out = canonical["train_out"]  # list of canonical np.ndarray

    frame_mask, shapes = _detect_frame(train_out, trace=trace)
    has_frame = bool(frame_mask.any())

    scaffold = {
        "frame_mask": frame_mask,
        "train_out_shapes": shapes,
        "has_frame": has_frame,
    }

    if trace:
        H, W = frame_mask.shape
        logging.info(
            f"[scaffold] frame_mask shape={frame_mask.shape}, "
            f"sum={int(frame_mask.sum())}, has_frame={has_frame}"
        )

    return scaffold
```

This is well within 80–120 LOC and purely numpy.

---

### 5. `run.py` behavior after WO-2.1

`run.py` remains minimal and unchanged:

```python
present   = load_present(task_bundle, trace=trace)
canonical = canonicalize(present, trace=trace)
scaffold  = build_scaffold(canonical, trace=trace)
out_size  = choose_size(canonical, scaffold, trace=trace)
...
```

With WO-2.1 done:

* `present` works (M1),
* `truth` works (M1),
* `scaffold` now returns a real `scaffold` dict with `frame_mask`,
* The pipeline then moves to `04_size_choice.choose`, which still raises `NotImplementedError`.

No new logic in `run.py`.

---

### 6. Receipts for WO-2.1

With `--trace`, `build` logs:

* frame_mask shape: `(H_out, W_out)` in canonical coords,
* frame_mask sum: number of True positions (size of frame),
* `has_frame` flag.

Example log:

```text
[scaffold] frame_mask shape=(6, 6), sum=20, has_frame=True
```

Reviewer uses this to:

* Confirm frame_mask is non-empty on a border-frame task (per `M1_M2_checkpoints.md`).
* Confirm H,W match `train_out` canonical shapes.

---

### 7. Reviewer instructions (WO-2.1 checkpoint)

**Commands:**

1. Choose a known **border-frame task** from `M1_M2_checkpoints.md` (it will specify a `task_id`, e.g., `"frame_border_task_id"`):

   ```bash
   python run.py --task-id frame_border_task_id \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

2. Also run on `00576224` for sanity (even if it doesn’t have a strong frame):

   ```bash
   python run.py --task-id 00576224 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

**What to expect:**

* For the border-frame task:

  * `present` logs shapes/palettes as in M1.
  * `truth` logs union graph info and canonical hash as in M1.3.
  * `scaffold` logs `frame_mask shape=..., sum>0, has_frame=True`.
  * Then `size_choice` raises `NotImplementedError` (still unimplemented).

* For `00576224`:

  * `present` and `truth` as before.
  * `scaffold` may produce `frame_mask.sum()==0` if there is no globally invariant color per position; that’s allowed.
  * `has_frame=False`, sum=0.
  * Then `size_choice` raises `NotImplementedError`.

**Legit gap vs bug:**

* **Legit gap:**

  * `scaffold` logs correctly; `size_choice` not implemented.
  * On a border-frame task, `frame_mask.sum()>0` and shape matches canonical train_out shape.

* **Bug:**

  * `frame_mask.shape` doesn’t match train_out shapes.
  * `frame_mask` always zero on a task whose golden says a stable border/frame exists.
  * `build` is using `train_in` or `test_in`, instead of `train_out`.
  * Any attempt to mix non-canonical indices or revert to pre-canonical coordinate systems.

**Math/spec consistency:**

* Reviewer checks `frame_mask` is computed exactly as in spec: same color across **all** train_out at each position, in canonical gauge.
* No thresholds or heuristics: either always same color → frame, otherwise not.

---

### 8. Optimization / CPU considerations

* `np.stack` + `np.all` on at most a handful of 30×30 grids is trivial on CPU.
* No need for approximations or tuning; we always compute the exact frame mask.
* We do not pre-filter by border or anything like that; full grid is cheap.

---