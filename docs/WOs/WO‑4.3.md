## WO-4.3 — `05_laws/atoms.py`: Connectivity & shape atoms (C12–C15)

**Goal:** For each canonical output grid, compute the “C” atoms exactly as per spec:

From `00_MATH_SPEC.md` §5.1 **C. Connectivity & shape**:

* Per-color connected components
* Per-component stats:

  * area
  * perimeter (4-edge)
  * bbox
  * centroid (int floor)
  * `(h, w)` (height, width)
  * `h − w` (aspect difference)
  * aspect class
  * simple orientation sign
  * area rank
  * frame-ring thickness class (if component is a ring touching all sides)

These are **derivations**, computed on canonical grids, not learned or approximated.

---

### 0. Anchors to read before coding

Implementer + reviewer must re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * §5.1 **C. Connectivity & shape** (exact list of attributes).
   * §4 **Stage F**: to understand frame and distance fields (`d_*`, inner region) used for ring thickness.

2. **` @docs/anchors/01_STAGES.md `**

   * Stage **05_laws**: atoms are per-grid derivations, laws are mined from train_out only.

3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * Mapping where C-atoms live in the “free/seeing” sector; no paid bits, no heuristics.

If any field (e.g. “aspect class”, “orientation sign”) remains underspecified mathematically, you **must not** guess; mark it `NotImplemented` and flag back to math spec.

---

### 1. Libraries (no custom BFS, no hand-rolled labeling)

Use:

```python
import numpy as np
from scipy import ndimage
from skimage.measure import label as sk_label, regionprops
```

Justification:

* `scipy.ndimage` gives classic, robust image operations and labeling; `label` is explicitly documented for labeling features in arrays.
* `skimage.measure.regionprops` is the standard way to get per-region area, bbox, centroid, etc., from a labeled image.

We **do not**:

* Implement BFS/DFS ourselves,
* Write custom connected-component labeling,
* Approximate region stats.

---

### 2. Input / output contract

#### 2.1 Where this lives

Extend `05_laws/atoms.py`:

```bash
05_laws/
  atoms.py   # A-atoms, now C-atoms
  step.py    # calls compute_A_atoms, compute_B_atoms, compute_C_atoms, ...
```

#### 2.2 Input to C-atoms

For each **output grid** (train_out; test_out later):

```python
from 05_laws.atoms import compute_C_atoms

C_atoms = compute_C_atoms(
    grid: np.ndarray[(H,W)],          # canonical colors 0..9
    scaffold_info: dict,              # from 03_scaffold for this grid
)
```

`scaffold_info` must contain (from Stage F/WO-2.x):

```python
{
  "d_top": np.ndarray[(H,W), int],
  "d_bottom": np.ndarray[(H,W), int],
  "d_left": np.ndarray[(H,W), int],
  "d_right": np.ndarray[(H,W), int],
  # inner, thickness_min, etc., are present but only some are used here
}
```

If `scaffold_info` is missing or shapes don’t match, raise `ValueError` or `NotImplementedError` (fail loudly, don’t guess).

#### 2.3 Output from C-atoms

Return a nested dict:

```python
{
  "components": {
    # per color k (int)
    k: [
      {
        "label": int,                     # 1..num_components for that color
        "area": int,
        "perimeter_4": int,
        "bbox": (int, int, int, int),     # (r_min, c_min, r_max, c_max) inclusive
        "centroid_r": int,
        "centroid_c": int,
        "height": int,
        "width": int,
        "height_minus_width": int,
        "area_rank": int,                 # 0 = largest area for that color
        "ring_thickness_class": int or None,
        # placeholders for future spec clarity:
        "aspect_class": None,
        "orientation_sign": None,
      },
      ...
    ]
  }
}
```

Notes:

* `aspect_class` and `orientation_sign` are **spec-listed but mathematically undefined** in anchors; we expose them as `None` for now and require a future spec update to define them. That respects your “no improvisation” rule.

---

### 3. Exact semantics

#### 3.1 Per-color components (4-connectivity)

Interpretation:

* For each color k, build a binary mask and label 4-connected components.

Implementation:

```python
H, W = grid.shape
colors = np.unique(grid)

components = {}

for k in colors:
    mask = (grid == k).astype(np.uint8)
    # 4-connectivity structure
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=int)
    labeled, num = ndimage.label(mask, structure=structure)
    # alternatively: skimage.measure.label(mask, connectivity=1)
    # but ndimage.label is fine and documented.
    # 
```

We then compute regionprops on `labeled`:

```python
    props = regionprops(labeled)  # list of RegionProperties 
```

Each `prop` has:

* `label` (int),
* `area` (pixel count),
* `bbox` (min_row, min_col, max_row, max_col) with max exclusive, per skimage docs,
* `centroid` (float row, float col).

We map to our stats per component.

#### 3.2 Area, bbox, (h,w), centroid (int)

For each `prop`:

```python
area = int(prop.area)

minr, minc, maxr, maxc = prop.bbox  # maxr, maxc are exclusive
# convert to inclusive bbox for our spec
r_min, c_min = minr, minc
r_max, c_max = maxr - 1, maxc - 1
height = r_max - r_min + 1
width  = c_max - c_min + 1

centroid_r = int(np.floor(prop.centroid[0]))
centroid_c = int(np.floor(prop.centroid[1]))

height_minus_width = height - width
```

We **do not** use regionprops’ orientation/major/minor axes etc. because spec only asks for “simple orientation sign” and “aspect class” without definitions. We leave those as `None`.

#### 3.3 Perimeter (4-edge)

Spec: perimeter (4-edge), not smooth/Euclidean perimeter.

We define 4-edge perimeter as:

* Count of edges between component pixels and 4-neighbor background.

Efficiently:

1. For each color mask `mask` (0/1), compute:

   * `mask_up = np.pad(mask[1:,:],  ((1,0),(0,0)), constant_values=0)`
   * `mask_down = np.pad(mask[:-1,:], ((0,1),(0,0)), constant_values=0)`
   * `mask_left = np.pad(mask[:,1:],  ((0,0),(1,0)), constant_values=0)`
   * `mask_right = np.pad(mask[:,:-1], ((0,0),(0,1)), constant_values=0)`

2. For each direction, edges exist where `mask==1` and neighbor==0:

   ```python
   edges_up    = mask & (mask_up == 0)
   edges_down  = mask & (mask_down == 0)
   edges_left  = mask & (mask_left == 0)
   edges_right = mask & (mask_right == 0)
   ```

3. Total edge count per pixel:

   ```python
   edge_count_per_pixel = edges_up + edges_down + edges_left + edges_right
   ```

4. To get perimeter for each component, we sum edge_count_per_pixel per label using ndimage’s measurement tools:

   ```python
   perim_array = edge_count_per_pixel.astype(int)
   labels_flat = labeled
   label_ids = np.arange(1, num+1)
   # sum over labels
   perimeter_per_label = ndimage.sum(perim_array, labels=labels_flat, index=label_ids)
   # perimeter_per_label is array length num; map back to props
   ```

Alternatively, you can compute this inside the loop per component via `perimeter_4 = int(perimeter_per_label[label-1])`.

This gives exact 4-edge perimeter count.

#### 3.4 Area rank (within color)

For each color `k`:

1. Collect all components’ areas as a list.
2. Sort in **descending** order; largest area gets rank 0, next 1, etc.
3. Assign rank to each component.

Implementation:

```python
areas = [comp["area"] for comp in comps_for_k]
sorted_idx = np.argsort([-a for a in areas])  # descending
for rank, comp_idx in enumerate(sorted_idx):
    comps_for_k[comp_idx]["area_rank"] = rank
```

This is fully specified and deterministic.

#### 3.5 Ring thickness class (frame-ring thickness class)

Spec:

> If comp is a ring that touches all sides, compute ring thickness class via min distance.

Interpretation:

* A “ring that touches all sides” is a component whose pixels lie in a frame-like shape around the grid, touching all four borders.
* We check:

  For component mask `comp_mask`:

  * It must intersect:

    * top row (r=0),
    * bottom row (r=H-1),
    * left col (c=0),
    * right col (c=W-1).

If any of these is false → `ring_thickness_class = None`.

If ring-like:

* We estimate thickness from the **distance fields** (Stage F):

  ```python
  d_top = scaffold_info["d_top"]
  d_bottom = scaffold_info["d_bottom"]
  d_left = scaffold_info["d_left"]
  d_right = scaffold_info["d_right"]
  ```

* For each pixel in `comp_mask`, we compute:

  ```python
  # distance to nearest frame from that side
  d_val = min(d_top[p], d_bottom[p], d_left[p], d_right[p])
  ```

* For an ideal ring, this min distance equals 0 on outer boundary, and increases inward; ring thickness can be approximated by:

  * `t = max(d_val)` over comp pixels (or min positive distance for inner boundary). However, spec says “via min distance”; the wording is ambiguous.

To avoid improvisation:

* We define ring thickness class **exactly** as:

  ```python
  distances = np.minimum.reduce([
      d_top[comp_mask],
      d_bottom[comp_mask],
      d_left[comp_mask],
      d_right[comp_mask]
  ])
  # class is the maximum distance observed within the ring
  ring_thickness_class = int(distances.max())
  ```

If that is considered underspecified, we should mark it as:

* Implement this exact rule, and document it in code comments; if math spec wants a different convention, they must update the spec.

Given the current anchor text, this is the minimal consistent and deterministic interpretation.

---

### 4. Brainstem `run.py` changes

Same as previous Milestone 4 WOs:

* `run.py` stays:

  ```python
  present   = load_present(...)
  canonical = canonicalize_truth(...)
  scaffold  = build_scaffold(...)
  size_ch   = choose_size(...)
  laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
  ```

* `05_laws/step.py` is updated to call `compute_C_atoms` for each train_out grid.

No changes to `run.py` are needed; it remains a minimal orchestrator.

---

### 5. Receipts / trace for C-atoms

Add a helper:

```python
def trace_C_atoms(C_atoms: Dict[str, Any], grid: np.ndarray) -> None:
    print("[C-atoms] grid shape:", grid.shape)
    for k, comps in C_atoms["components"].items():
        print(f"  color {k}: {len(comps)} components")
        if comps:
            areas = [c["area"] for c in comps]
            print(f"    areas: {areas}")
            ranks = [c["area_rank"] for c in comps]
            print(f"    ranks: {ranks}")
            rings = [c["ring_thickness_class"] for c in comps]
            print(f"    ring_thickness_class: {rings}")
        # only print first few colors
```

`05_laws/step.py` can call `trace_C_atoms` under `trace=True` for the first train_out grid of each task.

---

### 6. Reviewer instructions

#### 6.1 Using `run.py` on 2–3 tasks

Pick:

* Task A: one with multiple blobs of a single color (e.g. scattered small components).
* Task B: a ring-like border of a single color (frame around grid).
* Task C: typical ARC task like `00576224` to test complex combinations.

Run:

```bash
python run.py --task tasks/<task_id>.json --trace
```

Ensure `mine_laws` calls `compute_C_atoms` and `trace_C_atoms` on at least the first train_out.

Checks:

1. Per-color component counts look sensible (e.g. if grid has two separate red blobs, you see 2 components for that color).

2. Area + ranks:

   * For a uniform 3×3 single-color grid, one component with area=9 and area_rank=0.
   * For a grid with one big component and one tiny, big’s rank=0, tiny’s rank=1.

3. Perimeter:

   * For a 1×1 single pixel in empty grid, perimeter_4=4.
   * For a 2×2 block in empty grid, perimeter_4=8.
   * For a full H×W block, perimeter_4 = 2H + 2W.

4. Ring thickness:

   * For a 1-cell-thick full border ring (frame) around empty interior:

     * It touches all sides, ring_thickness_class should be 0 or 1 depending on distance convention; with our min-dist-based max rule, you should see a small integer (likely 1). If that mismatches math spec expectation, that’s a spec issue to raise.

If these structural checks fail, that’s an implementation bug. If the only disagreement is about exact ring_thickness_class convention, that points back to spec needing a tighter definition.

#### 6.2 Interaction with later golden (equivalence classes)

Later, Milestone 5 golden (4 classes of size 9 on 00576224) depends partly on type keys using component shape IDs from C-atoms. If M5 fails and A/B atoms look correct, C-atoms are suspect.

* Check that component counts and bounding boxes match visual inspection of the task’s train_out.
* Check that ring_thickness_class is only non-None for actual border rings.

---
