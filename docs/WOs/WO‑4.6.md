## WO-4.6 — `05_laws/atoms.py`: Component rigid/affine transforms (G25–G26)

**Goal:** For each color, detect when connected components in `train_out` are related by grid symmetries:

* D4 (8 rotations/reflections),
* Integer isotropic scale `s` (positive, s·bbox fits in grid),
* Translation `t`.

Keep **only** exact matches of pixel sets. Then, for each cell, expose local coordinate flags relative to its component’s canonical frame.

These are G-atoms; they sit on top of C-atoms.

---

### 0. Anchors to read

Before implementing, read:

* ` @docs/anchors/00_MATH_SPEC.md `

  * §5.1 **C. Connectivity & shape** (components, bbox, ring, etc.)
  * §5.1 **G. Component transforms** (D4 × scales, exact matches, local coords flags)
* ` @docs/anchors/01_STAGES.md `

  * Stage 05_laws: atoms then miner; all on canonical grids.
* ` @docs/anchors/02_QUANTUM_MAPPING.md `

  * G-atoms are “seeing” symmetries, not paid steps.

---

### 1. Libraries

We reuse the same stack as C-atoms:

```python
import numpy as np
from scipy import ndimage
from skimage.measure import label as sk_label, regionprops
```

* `ndimage.label` / `skimage.measure.label` can be reused if we don’t carry label maps in C-atoms.
* No custom BFS/DFS. No heuristics.

---

### 2. Input / output contract

#### 2.1 Input

In `05_laws/atoms.py` define:

```python
def compute_G_atoms(
    grid: np.ndarray,        # H×W, colors 0..9, canonical coords
    C_atoms: dict,           # result from compute_C_atoms on this grid
) -> dict:
    ...
```

`C_atoms` must contain, for this grid:

```python
{
  "components": {
    k: [  # list per color k
      {
        "label": int,                 # component label in some label map
        "bbox": (r_min, c_min, r_max, c_max),
        "area": int,
        # other C stats...
      },
      ...
    ]
  }
}
```

If C_atoms doesn’t include enough to recover per-component pixel sets (labels or masks), we are allowed to recompute labels from `grid` using `ndimage.label` (per color).

#### 2.2 Output

Return:

```python
{
  "templates": {
    # per color k, list of template shapes (index 0..T_k-1)
    k: [
      {
        "bbox": (r_min, c_min, r_max, c_max),
        "mask": np.ndarray[(h,w), bool],     # canonical template patch
      },
      ...
    ]
  },

  "component_to_template": {
    # per color k: mapping comp_idx -> template_idx and transform params
    k: [
      {
        "template_idx": int,        # 0..T_k-1
        "scale": int,               # s >= 1
        "d4_op": str,               # one of {"id","rot90","rot180","rot270","flip_h","flip_v","flip_d1","flip_d2"}
        # translation is implicit from bbox; no need to store t explicitly
      },
      ...
    ]
  },

  # per-cell flags
  "template_id": np.ndarray[(H,W), int],  # -1 if not in any component
  "local_r":     np.ndarray[(H,W), int],  # row offset within component bbox
  "local_c":     np.ndarray[(H,W), int],  # col offset within component bbox
}
```

Notes:

* `template_id[r,c]` is the template index of the component that cell belongs to, or -1 if background.
* `local_r, local_c` are **per-component** local coords: `r - r_min`, `c - c_min` for that component’s bbox. This is the “canonical frame” for that component. If a group of components share a template, they share shape (up to D4×scale), and their local_r/local_c are consistent relative to their own bbox.

This is enough for type keys and laws to express “same shape up to rigid/scale transform” plus local position.

---

### 3. Exact semantics and algorithm

#### 3.1 Recover per-color components and masks

If C_atoms already has a label map, reuse it; otherwise:

```python
H, W = grid.shape
colors = np.unique(grid)
components = C_atoms["components"]

# Build per-color label maps on the fly
label_maps = {}  # k -> (labels, num_labels)
for k in colors:
    mask = (grid == k).astype(np.uint8)
    if mask.any():
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=int)  # 4-connectivity
        labels, num = ndimage.label(mask, structure=structure)
    else:
        labels, num = np.zeros_like(mask), 0
    label_maps[k] = (labels, num)
```

For each component entry in `components[k]`, bbox `(r_min,c_min,r_max,c_max)` and `label`, we can reconstruct its patch:

```python
labels, _ = label_maps[k]
r0, c0, r1, c1 = r_min, c_min, r_max, c_max
comp_mask_patch = (labels[r0:r1+1, c0:c1+1] == label)
h_comp = r1 - r0 + 1
w_comp = c1 - c0 + 1
```

#### 3.2 Templates per color (D4×scale equivalence)

For each color `k`:

1. Maintain a list of `templates[k]` (canonical shapes) as:

   ```python
   templates_k = []  # list of dicts {"bbox":..., "mask":...}
   comp_to_template = []  # per component (same order as components[k])
   ```

2. For each component `comp` in `components[k]` (iterate in any deterministic order):

   * Extract `comp_mask_patch` as above; let `(h, w)` be its size.
   * Try to find an existing template `T` in `templates_k` such that component is a D4×scale copy of `T`.

   For each template `T`:

   * Let `mask_T` have shape `(hT, wT)`.

   * Candidate integer scale `s` must satisfy:

     ```python
     if h % hT != 0 or w % wT != 0:
         continue
     s_h = h // hT
     s_w = w // wT
     if s_h != s_w:  # isotropic scaling only
         continue
     s = s_h
     ```

   * Build scaled template via Kronecker product:

     ```python
     scaled_T = np.kron(mask_T.astype(np.uint8),
                        np.ones((s, s), dtype=np.uint8)).astype(bool)
     # scaled_T shape == (h, w)
     ```

   * Enumerate the 8 D4 ops on `scaled_T`:

     ```python
     def d4_variants(patch):
         # patch is (h,w)
         yield "id", patch
         yield "rot90", np.rot90(patch, k=1)
         yield "rot180", np.rot90(patch, k=2)
         yield "rot270", np.rot90(patch, k=3)
         yield "flip_h", np.flipud(patch)
         yield "flip_v", np.fliplr(patch)
         yield "flip_d1", np.transpose(patch)  # main diag
         yield "flip_d2", np.fliplr(np.flipud(np.transpose(patch)))  # other diag
     ```

   * For each op:

     ```python
     for op_name, cand in d4_variants(scaled_T):
         if cand.shape != comp_mask_patch.shape:
             continue
         if np.array_equal(cand, comp_mask_patch):
             # Found template: same shape up to D4×scale
             comp_to_template.append({
                 "template_idx": idx_T,
                 "scale": s,
                 "d4_op": op_name,
             })
             matched = True
             break
     if matched:
         break  # no need to try other templates
     ```

3. If no existing template matches:

   * Create a new template:

     ```python
     new_idx = len(templates_k)
     templates_k.append({
         "bbox": (0, 0, h-1, w-1),  # canonical frame is the patch itself
         "mask": comp_mask_patch.copy(),
     })
     comp_to_template.append({
         "template_idx": new_idx,
         "scale": 1,
         "d4_op": "id",
     })
     ```

4. After processing all components for color `k`, store:

   ```python
   templates[k] = templates_k
   component_to_template[k] = comp_to_template
   ```

This is a full, exhaustive enumeration of D4×scale equivalence classes per color, with exact set equality.

No heuristics: every component is either its own template or a hard match.

#### 3.3 Per-cell local coords flags

We now populate three per-cell arrays for the whole grid:

```python
template_id = -np.ones((H,W), dtype=int)
local_r     = np.zeros((H,W), dtype=int)
local_c     = np.zeros((H,W), dtype=int)
```

For each color `k`:

* Get `labels_k, _ = label_maps[k]`.
* For each component index `ci, comp in enumerate(components[k])`:

  ```python
  t_idx = component_to_template[k][ci]["template_idx"]
  r_min, c_min, r_max, c_max = comp["bbox"]
  # For every pixel belonging to this component:
  mask_ci = (labels_k == comp["label"])

  # Local coords = offset inside component's bbox
  rs, cs = np.where(mask_ci)
  # set per-cell flags
  template_id[rs, cs] = t_idx
  local_r[rs, cs] = rs - r_min
  local_c[rs, cs] = cs - c_min
  ```

Thus:

* Every component’s pixels share a template id (for that color) and have local coordinates within their own bbox frame. If two components share a template, they have the same shape up to D4×scale, and `local_r/c` tell you where you are inside that shape’s frame.

This matches “local coords flags relative to its component’s canonical frame when transported”: canonical frame = bbox, transported by D4×scale×t, but local coords are still relative to that frame.

---

### 4. Brainstem `run.py` changes

None.

Pipeline remains:

```python
present   = load_present(...)
canonical = canonicalize_truth(...)
scaffold  = build_scaffold(...)
size_ch   = choose_size(...)
laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
```

`05_laws/step.py` is responsible for:

* Calling `compute_C_atoms` and then `compute_G_atoms` for each train_out grid.

`run.py` stays minimal.

---

### 5. Receipts / trace

Add:

```python
def trace_G_atoms(G_atoms: dict, grid: np.ndarray) -> None:
    H, W = grid.shape
    print("[G-atoms] grid shape:", H, "x", W)
    for k, templates_k in G_atoms["templates"].items():
        print(f"  color {k}: {len(templates_k)} templates")
        break  # only show first color

    print("[G-atoms] template_id unique values:",
          np.unique(G_atoms["template_id"]))
```

`05_laws/step.py` can call this with `trace=True` for the first train_out.

---

### 6. Reviewer instructions

Pick 2–3 ARC tasks with obvious copy/rotate/scale patterns:

* Task A: component horizontally flipped across a line.
* Task B: component rotated 90°.
* Task C: upscaled copy (e.g. 2× blow-up of a small shape).

Run:

```bash
python run.py --task tasks/<task_id>.json --trace
```

Check:

* For a grid where two same-color components are congruent up to D4 (no scale):

  * `templates[k]` should have 1 template for that color.
  * Both components’ pixels should have same `template_id`, `scale=1`, `d4_op` matching their relation.

* For a grid where a small shape has a scaled copy:

  * `templates[k]` still: 1 template (the small shape).
  * Larger component’s `scale` should be >1.
  * `template_id` equal for pixels of both the small and big component.

* If you visually see two components **not** related by any D4×scale, they must belong to different templates.

Any mismatch is an implementation bug, not a spec issue.

---
