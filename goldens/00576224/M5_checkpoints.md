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

