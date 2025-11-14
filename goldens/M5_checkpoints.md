For **Milestone 5 (laws)** on task **00576224**, there is *one* thing I can guarantee exactly and in a way that is invariant to canonical labeling and other implementation details:

> The relational equalities (periodicity / repetition) will partition the 6×6 **test canvas** into exactly **4 equivalence classes**, each of size **9**.

That reflects the 2×2 tiling / parity structure of the grid:

* Cells with same `(r mod 2, c mod 2)` end up tied together by equalities such as `(δ_r=0, δ_c=2)` and `(δ_r=2, δ_c=0)` applied repeatedly.
* There are 4 such parity patterns, each appearing in 9 positions over a 6×6 grid, so the final equivalence partition is:

  * 4 classes,
  * each class size 9.

I am **not** going to assert anything about:

* how many unary fixes (type ⇒ color) you mine,
* how many forbids you emit,
* the exact coordinates of each class,

because those depend subtly on details of the atom key and canonicalization. Instead I specify a **permutation-invariant summary** that must hold no matter how you index rows/cols, as long as you follow the spec.

So your golden for Milestone 5 on this task should only check these structural facts.

Here is a JSON shape you can use:

```json
{
  "task_id": "00576224",
  "laws_summary": {
    "equiv_classes": {
      "count": 4,
      "size_histogram": {
        "9": 4
      }
    }
  }
}
```

### How the tester should use this

After `05_laws.step.mine(...)` has run and you’ve built the DSU / union-find structure of equalities *on the test canvas*:

1. Collect all equivalence classes (each as a set of cells).
2. Compute:

   * `equiv_class_count = number_of_classes`
   * `equiv_class_sizes_hist = { size → how_many_classes_have_that_size }`
3. Compare those two numbers to the golden:

* `equiv_class_count` must be `4`.
* `equiv_class_sizes_hist` must be exactly `{9: 4}`.

You should **not** compare:

* exact member coordinates,
* number of fixes,
* number of forbids,

for this particular task’s Milestone 5 golden. Those are allowed to vary as long as they remain consistent with the spec, and they will be exercised more precisely on other golden tasks where we can pin them down safely.

### More golden atoms
{
  "task_id": "46f33fce",
  "atoms_summary": {
    "train_out": [
      {
        "index": 0,
        "pixel_counts": {
          "0": 304,
          "1": 16,
          "2": 32,
          "3": 16,
          "4": 16,
          "8": 16
        },
        "component_counts_nonzero": {
          "1": 1,
          "2": 1,
          "3": 1,
          "4": 1,
          "8": 1
        },
        "row_period_hist": {
          "1": 4,
          "16": 4,
          "20": 12
        }
      },
      {
        "index": 1,
        "pixel_counts": {
          "0": 304,
          "1": 16,
          "2": 16,
          "3": 32,
          "4": 32
        },
        "component_counts_nonzero": {
          "1": 1,
          "2": 1,
          "3": 2,
          "4": 2
        },
        "row_period_hist": {
          "12": 4,
          "16": 4,
          "20": 12
        }
      },
      {
        "index": 2,
        "pixel_counts": {
          "0": 320,
          "1": 32,
          "2": 16,
          "3": 16,
          "4": 16
        },
        "component_counts_nonzero": {
          "1": 1,
          "2": 1,
          "3": 1,
          "4": 1
        },
        "row_period_hist": {
          "1": 12,
          "20": 8
        }
      }
    ]
  }
}
How the tester should use this
After WO-4.1–4.4 for 46f33fce:
For each train_out[i]:
Compute pixel_counts[color] = number of cells with that color.
Compute component_counts_nonzero[color] via N4 connectivity for colors > 0.
Compute row_period_hist: for each row, find the minimal period p (1..W) such that row[j] == row[j % p] for all j, then tally counts of each p.
Compare these three summaries per grid index with the golden above.
If they match (up to your JSON normalisation), your A/C/D/E atoms for this task are implemented correctly.