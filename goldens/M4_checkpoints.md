Given that, here’s what I suggest for the three tasks we’ve already locked:

* `00576224`
* `5e6bbc0b`
* `833966f4`

I’ll propose **which atoms to gate** and **which tasks to use them on**. Then we can turn those into JSON goldens like we did for scaffold/size.

---

## Key idea: what is safe to golden for M4?

Stage 4 atoms are many. Some are **gauge-sensitive** (dependent on row/col indices, canonicalization quirks) and should not be goldened without running the exact code (e.g. 3×3 hashes, block coords, detailed mod-classes).

The safest atoms to golden across implementations are:

1. **Global palette & pixel counts** (E19–E23):

   * Per-color pixel counts on each train_out.
   * Palette sets per grid.
     These are invariant under any row/col permutations.

2. **Per-color component counts and component sizes** (C12–C15):

   * Number of connected components per color.
   * Areas (sizes) of those components, sorted.
     This is invariant under graph isomorphism (Π only relabels; it does not change adjacency).

3. **Minimal row periods** (D16):

   * For each row, minimal period along the row.
   * Summarized as a histogram: `{period → count_of_rows}`.
     This is safe as long as “period” is defined purely from the row sequence itself, which it is. Row permutations don’t change the multiset of periods.

I’d avoid goldening:

* 3×3 / 5×5 hashes: too sensitive to row/col index semantics.
* Block coords / tiling flags: depend on how you slice by index, which can vary across canonicalization choices.
* Detailed mod-class arrays: also index-sensitive.

We will **gate M4** via a small set of **E**, **C**, and **D** atoms, on tasks where they are clear.

---

## Task-by-task: what to golden

### 1) Task `833966f4` (5×1 column swap)

This is the cleanest and smallest; perfect to gate **C** and **E** and simple **row periods**.

#### Train_out grids

* Output 1: `[0, 9, 1, 8, 6]`
* Output 2: `[3, 4, 6, 8, 2]`

#### E: per-color pixel counts & palette

For train_out[0]:

* Colors present: {0, 1, 6, 8, 9}
* Each appears exactly once:

  ```json
  "pixel_counts": { "0": 1, "1": 1, "6": 1, "8": 1, "9": 1 }
  ```

For train_out[1]:

* Colors present: {2, 3, 4, 6, 8}
* Each appears exactly once:

  ```json
  "pixel_counts": { "2": 1, "3": 1, "4": 1, "6": 1, "8": 1 }
  ```

These must match exactly.

#### C: per-color component counts

Using N4-connectivity:

* Every pixel is isolated in 1D, so each color has exactly 1 component.

For train_out[0]:

```json
"component_counts": { "0": 1, "1": 1, "6": 1, "8": 1, "9": 1 }
```

For train_out[1]:

```json
"component_counts": { "2": 1, "3": 1, "4": 1, "6": 1, "8": 1 }
```

Areas of each component are all 1, so the multiset of component areas per color is just `[1]`.

#### D: minimal row periods

Each row is of length 1, so minimal period is 1.

* Row_period_histogram: `{1: 5}` for both train_outs.
* Col_period_histogram: `{1: 1}` (single column).

**Golden suggestion for `833966f4` (M4 atoms)**:

```json
{
  "task_id": "833966f4",
  "atoms_summary": {
    "train_out": [
      {
        "index": 0,
        "pixel_counts": { "0": 1, "1": 1, "6": 1, "8": 1, "9": 1 },
        "component_counts": { "0": 1, "1": 1, "6": 1, "8": 1, "9": 1 },
        "row_period_hist": { "1": 5 }
      },
      {
        "index": 1,
        "pixel_counts": { "2": 1, "3": 1, "4": 1, "6": 1, "8": 1 },
        "component_counts": { "2": 1, "3": 1, "4": 1, "6": 1, "8": 1 },
        "row_period_hist": { "1": 5 }
      }
    ]
  }
}
```

Tester rule: after WO-4.3/4.4, aggregate those summaries and compare.

---

### 2) Task `00576224` (2×2 pattern tiling 3×3)

This is ideal to gate **pixel counts** and **row periods**.

Recall the two train_outs (6×6):

* Output 1 uses colors {3,4,7,9} in a perfectly repeated tiling.
* Output 2 uses {4,6,8} in a different tiling.

#### E: per-color pixel counts

We already computed:

Train_out[0]:

* Colors: 3,4,7,9
* Each appears exactly 9 times.

```json
"pixel_counts": { "3": 9, "4": 9, "7": 9, "9": 9 }
```

Train_out[1]:

* Colors: 4,6,8
* Counts: 4 → 9, 8 → 9, 6 → 18

```json
"pixel_counts": { "4": 9, "6": 18, "8": 9 }
```

#### D: minimal row periods

Check each row:

* Output 1 rows are like `[7,9,7,9,7,9]` or `[4,3,4,3,4,3]`, etc.
  All rows have period 2 (pattern ABABAB).

* Output 2 rows also all have period 2 (e.g. `8,6,8,6,8,6`, `6,4,6,4,6,4`).

So for both train_out grids:

```json
"row_period_hist": { "2": 6 }
```

(6 rows, each period 2.)

We **don’t** try to golden column periods here; they are more subtle and not needed for gating.

**Golden suggestion for `00576224` (M4 atoms)**:

```json
{
  "task_id": "00576224",
  "atoms_summary": {
    "train_out": [
      {
        "index": 0,
        "pixel_counts": { "3": 9, "4": 9, "7": 9, "9": 9 },
        "row_period_hist": { "2": 6 }
      },
      {
        "index": 1,
        "pixel_counts": { "4": 9, "6": 18, "8": 9 },
        "row_period_hist": { "2": 6 }
      }
    ]
  }
}
```

Tester rule: after WO-4.4, compute `pixel_counts` and `row_period_hist` and compare.

---

### 3) Task `5e6bbc0b`

For this one, M4 becomes tricky:

* Mixed sizes (5×6, 7×7, 4×8, 3×3).
* More colors (0,1,2,3,4,6,8,9).
* Complex patterns in rows/cols; row periods vary.

I can compute per-color pixel counts by hand, but with the complexity and risk of arithmetic slip, I’d **not** use this task to gate M4 atoms. You already use it for:

* M1 present,
* M2 scaffold,
* M3 size_choice,
* Final output.

That’s enough. Let’s **not** add M4 goldens for 5e6bbc0b, to avoid introducing fragile hand-computed numbers.

---

## Recommended gating plan for Milestone 4

Use these two tasks:

1. **`833966f4`** – gate:

   * `pixel_counts` per train_out,
   * `component_counts` per train_out,
   * `row_period_hist` per train_out.

2. **`00576224`** – gate:

   * `pixel_counts` per train_out,
   * `row_period_hist` per train_out.

Skip M4 goldens for `5e6bbc0b`.

This gives your reviewer/tester:

* A small 1D case that exercises atoms C & E and trivial periods.
* A periodic 2D tiling case that exercises pixel counting and period detection.

And all numbers I listed above are fully determined by the spec and robust against canonicalization, so they’re safe to use as goldens.

#### 74dd1130
{
  "task_id": "74dd1130",
  "atoms_summary": {
    "train_out": [
      {
        "index": 0,
        "pixel_counts": {
          "5": 4,
          "9": 3,
          "8": 2
        },
        "component_counts_nonzero": {
          "5": 2,
          "8": 2,
          "9": 2
        },
        "row_period_hist": {
          "3": 3
        },
        "col_period_hist": {
          "3": 3
        }
      },
      {
        "index": 1,
        "pixel_counts": {
          "2": 4,
          "5": 4,
          "6": 1
        },
        "component_counts_nonzero": {
          "2": 1,
          "5": 2,
          "6": 1
        },
        "row_period_hist": {
          "3": 2,
          "2": 1
        },
        "col_period_hist": {
          "3": 2,
          "1": 1
        }
      },
      {
        "index": 2,
        "pixel_counts": {
          "2": 4,
          "6": 3,
          "1": 2
        },
        "component_counts_nonzero": {
          "1": 1,
          "2": 2,
          "6": 2
        },
        "row_period_hist": {
          "1": 1,
          "2": 1,
          "3": 1
        },
        "col_period_hist": {
          "3": 2,
          "2": 1
        }
      },
      {
        "index": 3,
        "pixel_counts": {
          "2": 4,
          "1": 3,
          "5": 2
        },
        "component_counts_nonzero": {
          "1": 2,
          "2": 2,
          "5": 2
        },
        "row_period_hist": {
          "3": 2,
          "2": 1
        },
        "col_period_hist": {
          "3": 2,
          "2": 1
        }
      }
    ]
  }
}
How your tester should compute and compare

For each train_out[i]:

pixel_counts

Count occurrences of each color in the 3×3 grid.

Store as {color → count}.

component_counts_nonzero

For each color k > 0, run N4-connected-component labeling.

Count how many connected components of that color exist.

Store as {color → component_count}.

row_period_hist

For each row (length 3), find minimal p ∈ {1,2,3} such that
row[j] == row[j % p] for all j.

Build a histogram {p → number_of_rows_with_period_p}.

col_period_hist

Same as above but per column.