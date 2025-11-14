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
---
### Goldens for M5
* For most tasks, **full M5 structure** (which cells are tied to which) is too implementation-sensitive to be a safe golden.
* But there **are** a few simple, spec-level quantities we can safely golden that will still catch real bugs:

  * number of unary fixes (`fix_count`), and
  * for some tasks, the equivalence-class histogram (`equiv_classes`) which we already did on 2 tasks.

For **`05f2a901` specifically**, there is one thing I can guarantee with 100% certainty:

> There are **no unary fixes**. So `fix_count` must be 0 on the test canvas.

Let me explain why, from the spec.

---

## Why `fix_count = 0` is safe for 05f2a901

Unary fixes in WO-5.2:

* You build a map `T → multiset(colors across train_out)`, where `T` is the type key.
* A unary fix is emitted only if:

  1. For **every** train_out, at **every** occurrence of this type `T`, the color is **identical** (call it `k`).
  2. There exists at least one training where those `T` cells are **non-background** (anti-spurious).

The type key `T(p)` includes:

* Distances (`d_top,bottom,left,right`)
* Position-like info (`r±c`, mod classes, block coords)
* 3×3 hash
* Period flags
* Component shape ID

Now look at the three `train_out` grids for `05f2a901`:

* The **2-component** moves: in each example, the big 2-blob is in a different place and has a different shape.
* The **8-component** also moves between examples (and is 0 in parts of some grids).

This means:

* Any cell that is non-background in one output (2 or 8) either:

  * is background (0) at that geometry in another train_out, or
  * has a **different** component shape / distance profile in another example, hence gets a **different** type key `T`.

So there is **no type `T`** that:

* Appears with a fixed non-zero color in all train_out,
* And passes the anti-spurious condition.

All the “always same color” types are background only, and anti-spurious rejects those. So Stage N cannot produce any unary fixes.

**Conclusion for 05f2a901:**

* `fix_count` over the test canvas **must be 0** for any spec-compliant implementation.

That’s a very strong and simple golden.

---

## Other M5 goldens we already have or can safely use

Across all the tasks we’ve prepared:

1. **`833966f4`** (5×1 column swap)

   * We already goldened **equiv_classes**: 5 classes of size 1.
   * M5 fact:

     * `fix_count = 0`
     * `equiv_classes.count = 5`
     * `equiv_classes.size_histogram = { 1: 5 }`

2. **`00576224`** (periodic tiling)

   * We already goldened **equiv_classes**: 4 classes of size 9.
   * M5 fact:

     * `fix_count = 0`
     * `equiv_classes.count = 4`
     * `equiv_classes.size_histogram = { 9: 4 }`

   Reasoning for `fix_count = 0`: second train_out has no color in common with the first at any fixed geometry (no cross-grid position is non-zero in both with same color), so no type can be non-background and constant across all train_out.

3. **`0d3d703e`** (pure palette permutation)

   * Geometry is identical across train_out; colors change by permutation.
   * For any type `T`, the color at `T` changes across examples (because palette permutes), so “always-same color” fails for all non-zero colors.
   * Background-only types are killed by anti-spurious.
   * Hence:

     * `fix_count = 0`.

   For equiv_classes: rows are 3 distinct colors, no repetition inside row or column, so relational equalities can’t pair distinct cells without violating “colors equal at all train_out”. That suggests:

   * `equiv_classes.count = 9`, `size_histogram = {1:9}` would be safe, but given how subtle Δ definitions can be, I’d keep the golden to `fix_count = 0` here and not enforce classes.

4. **`05f2a901`** (component translate)

   * As argued:

     * `fix_count = 0`.
   * I would *not* fix an equiv_class histogram here, because the miner might or might not use “same component class” or other Δs in a way that merges some test cells, and both would still be spec-compliant.

For `46f33fce` and `74dd1130`, we should **not** golden M5 structure or fix counts. Those are blow-up and transpose patterns, and different valid choices of T and Δ could legitimately produce different equality structures.

---

## What I recommend concretely for M5 goldens

For your current golden tasks, M5 gating can be:

* **`833966f4`**

  * `fix_count = 0`
  * `equiv_class_count = 5`
  * `size_histogram = { "1": 5 }`

* **`00576224`**

  * `fix_count = 0`
  * `equiv_class_count = 4`
  * `size_histogram = { "9": 4 }`

* **`0d3d703e`**

  * `fix_count = 0`

* **`05f2a901`**

  * `fix_count = 0`

And **no M5 goldens** for:

* `5e6bbc0b`
* `46f33fce`
* `74dd1130`

This gives you:

* Strong DSU structure tests on 2 tasks (equiv classes).
* Strong “no spurious unary fixes” tests on 4 tasks, which is exactly where a lot of subtle bugs tend to creep in.

If you’d like, I can now write out a small `laws_summary` JSON for `05f2a901` and `0d3d703e` in the same style as the other structural_facts goldens, with just the `fix_count` field.
