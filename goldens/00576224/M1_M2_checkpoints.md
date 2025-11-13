## 1) Golden for Milestone 1 – WO-1.1 `present`

This is **before** canonicalization. It’s just what your loader should produce as numpy arrays.

```json
{
  "task_id": "00576224",
  "train_in": [
    [[7, 9],
     [4, 3]],
    [[8, 6],
     [6, 4]]
  ],
  "train_out": [
    [[7, 9, 7, 9, 7, 9],
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
     [6, 4, 6, 4, 6, 4]]
  ],
  "test_in": [
    [[3, 2],
     [7, 8]]
  ],
  "shapes": {
    "train_in":  [[2, 2], [2, 2]],
    "train_out": [[6, 6], [6, 6]],
    "test_in":   [2, 2]
  },
  "palettes": {
    "train_in":  [[3, 4, 7, 9], [4, 6, 8]],
    "train_out": [[3, 4, 7, 9], [4, 6, 8]],
    "test_in":   [[2, 3, 7, 8]]
  }
}
```

Notes:

* Palettes are sorted unique colors per grid.
* This is exactly what a correct loader should output, up to using `int8` vs `int32`.

If your implementer’s `present.step.load()` doesn’t produce exactly these values, something is wrong at the very first layer.

---

## 2) Golden for Milestone 2 – WO-2.3 `scaffold`

This is **after** canonicalization, but everything I give you here is **gauge-invariant**:

* Frame emptiness,
* Inner region size,
* Distance summaries to borders,
* Parity facts (midrow/midcol).

Because we BFS from borders and this task has no frame across train_out, these quantities are the same in any canonical gauge.

For each train_out grid, we assume:

* Height H = 6
* Width W  = 6

Distances (4-adjacency, BFS from each border separately):

* `d_top(r, c)    = r`
* `d_bottom(r, c) = 5 - r`
* `d_left(r, c)   = c`
* `d_right(r, c)  = 5 - c`

Inner region S is where all four distances > 0, i.e. rows 1..4 and cols 1..4, so |S| = 16.

Frame F is empty, because **no cell position shares the same color across both train_outs**.

Here is a compact JSON golden:

```json
{
  "task_id": "00576224",
  "per_output": [
    {
      "H": 6,
      "W": 6,
      "frame_count": 0,
      "inner_count": 16,
      "d_top":    { "min": 0, "max": 5, "sum": 90 },
      "d_bottom": { "min": 0, "max": 5, "sum": 90 },
      "d_left":   { "min": 0, "max": 5, "sum": 90 },
      "d_right":  { "min": 0, "max": 5, "sum": 90 }
    },
    {
      "H": 6,
      "W": 6,
      "frame_count": 0,
      "inner_count": 16,
      "d_top":    { "min": 0, "max": 5, "sum": 90 },
      "d_bottom": { "min": 0, "max": 5, "sum": 90 },
      "d_left":   { "min": 0, "max": 5, "sum": 90 },
      "d_right":  { "min": 0, "max": 5, "sum": 90 }
    }
  ],
  "global": {
    "has_midrow": false,
    "has_midcol": false
  }
}
```

A few sanity checks behind these numbers:

* For each distance field:

  * There are 36 cells.
  * For `d_top`, values per row r are r, repeated 6 times:
    sum = 6 * (0 + 1 + 2 + 3 + 4 + 5) = 6 * 15 = 90.
    min = 0, max = 5.
  * `d_bottom` is symmetric: 5−r; same sum.
  * Same logic for `d_left` and `d_right` over columns.

* `frame_count = 0`:
  For every position (r,c), `(train_out0[r,c], train_out1[r,c])` is never `(k,k)`. We checked the color 4 coincidence explicitly and it never aligns in both grids.

* `inner_count = 16`:
  S consists of all cells with r ∈ {1,2,3,4} and c ∈ {1,2,3,4}. That’s 4×4.

* `has_midrow` / `has_midcol` = false:
  H = W = 6 (even). There is no row r such that `d_top(r) = d_bottom(r)` for all cells in that row; that only happens when H is odd. So per spec, no midrow/midcol parity flag.

Your `03_scaffold.step.build()` should produce distance arrays and masks whose summaries match these exactly. Reviewer/tester can:

* Compute `frame_count` from the frame mask,
* Count `inner_count`,
* Aggregate d_top/d_bottom/d_left/d_right to min/max/sum,
* Check midrow/midcol booleans.

If any of these differ, something is wrong in BFS / F definition / inner logic.

---