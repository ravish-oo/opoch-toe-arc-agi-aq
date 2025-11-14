## WO-2.3 — `03_scaffold`: Inner region & global facts (per-output + aggregated)

(~80–120 LOC inside `03_scaffold/step.py`)

**Goal (per anchors + clarified Stage F):**

For each canonical training output grid (X^{(i)}_{\text{out}}):

* Use the per-grid border frame (F_{X^{(i)}}) and distance fields (d_{\text{top}}^i, d_{\text{bottom}}^i, d_{\text{left}}^i, d_{\text{right}}^i) (from WO-2.2),
* Define its **inner region** (S_i),
* Compute **local facts**: midrow / midcol flags, thickness candidates, and simple period hints,
* Then derive **aggregated hints** across all outputs that S0 can safely use to filter size candidates.

---

### 0. Anchors to read before coding

**Must read:**

1. ` @docs/anchors/00_MATH_SPEC.md `

   * §4.2 “Distance fields” (especially the inner region):

     > “Inner region: (S = { p : d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}} > 0}).”
   * §3.2 “Structural disambiguation”:

     * Uses **midrow/midcol**, **frame thickness**, and **periodicity** inside the inner region to filter size maps:

       * Parity: “If all train_out have a midrow … H' must be odd; similarly for W' and midcol.”
       * Periodicity: “If train_out rows (or cols) have least period (p) everywhere inside the inner region, then (p \mid H') (resp. (p \mid W')).”

2. ` @docs/anchors/01_STAGES.md `

   * “scaffold”: after frame and distances, we “define the inner region” and compute facts used by `size_choice`.

3. ` @docs/anchors/02_QUANTUM_MAPPING.md `

   * “scaffold” section:

     * “Compute distance fields to frame/border (pure geometry).
       Define the inner region S.
       These are input to size_choice; no law/mining here.”

4. Clarified Stage F spec (your last message)

   * Frame is per grid (border), geometry only,
   * Cross-grid “identical color” invariants go to Stage N, not F.

**How I used these:**

* I took §4.2 literally for inner: `inner = (d_top>0)&(d_bottom>0)&(d_left>0)&(d_right>0)` per grid. No variations.
* I treated midrow/midcol and period hints purely as *geometric/structural properties* of each output grid’s inner region, not as color patterns across grids.
* I refused to infer any cross-grid law here; that’s Stage N’s job per the clarified TOE.

---

### 1. Packages / libraries

We need:

* `numpy`

  * For boolean masks, min/max, sums, row/col slicing.

* `typing`

  * `Dict`, `Any`, `List`, `Tuple`.

* `logging`

  * For receipts.

We do **not**:

* Use networkx, scipy, or any BFS library here (distances are already computed in WO-2.2),
* Implement any algorithms except simple per-grid scans and array ops.

---

### 2. Input & output contract (building on WO-2.1/2.2)

After WO-2.1 and WO-2.2, `03_scaffold.build(canonical, trace)` returns:

```python
scaffold = {
  "per_output": [
    {
      "index": i,
      "shape": (H_i, W_i),
      "frame_mask": np.ndarray[bool],   # F_Xᵢ = border mask
      "d_top":    np.ndarray[int],
      "d_bottom": np.ndarray[int],
      "d_left":   np.ndarray[int],
      "d_right":  np.ndarray[int],
      # no inner / facts yet
    }
    for each train_out[i]
  ]
}
```

WO-2.3 will **extend** each per-output entry with:

```python
{
  "inner":        np.ndarray[bool],
  "has_midrow":   bool,
  "has_midcol":   bool,
  "thickness": {
      "min": int,        # minimal ring width from inner to frame
  },
  "row_period":   Optional[int],
  "col_period":   Optional[int],
}
```

and add:

```python
scaffold["aggregated"] = {
  "thickness_min": Optional[int],  # min thickness across per_output entries
  "row_period":    Optional[int],  # common row period across outputs, else None
  "col_period":    Optional[int],  # common col period across outputs, else None
  "has_midrow_all": bool,          # True if every per_output[i]["has_midrow"] is True
  "has_midcol_all": bool,          # same for cols
}
```

No other new fields; we’re not inventing new structures.

---

### 3. Inner region per output grid (exact spec)

Given for grid `i`:

* `d_topᵢ`, `d_bottomᵢ`, `d_leftᵢ`, `d_rightᵢ` (Hᵢ×Wᵢ arrays of ints),

we define its inner mask:

```python
inner_i = (d_top_i > 0) & (d_bottom_i > 0) & (d_left_i > 0) & (d_right_i > 0)
```

This is exactly (S_X = {p : d_{\text{top}},d_{\text{bottom}},d_{\text{left}},d_{\text{right}}>0}) from §4.2.

No shortcuts, no “almost 0” or tolerance. Strict `> 0`.

Edge behavior:

* If the grid is too small (e.g. H<3 or W<3), `inner_i` will be **all False** (no cell is >0 away from every border); that’s fine and consistent with the definition.

---

### 4. Local facts per output grid

Given `frame_mask_i`, `d_*_i`, `inner_i`, we compute:

#### 4.1 Parity flags: has_midrowᵢ / has_midcolᵢ

Based on §3.2:

> “If all train_out have a midrow (∃ r : d_top(r,*) = d_bottom(r,*)) then H' must be odd.”

Per grid:

```python
has_midrow_i = any(np.all(d_top_i[r, :] == d_bottom_i[r, :]) for r in range(H_i))
has_midcol_i = any(np.all(d_left_i[:, c] == d_right_i[:, c]) for c in range(W_i))
```

Notes:

* We use **strict equality** `==` as spec says: row is midrow if its top-distance equals its bottom-distance at every column.
* We do **not** approximate or pick “closest”; if there is no row with exact equality, `has_midrow_i=False`. Same for `has_midcol_i`.

#### 4.2 Thickness candidate per grid (min ring width)

Clarified spec: “min ring width from inner to frame”.

Given border frame F_X and distances:

* For any inner cell `(r,c) ∈ S_X`, the **distance to the frame** along each axis is:

  ```python
  local_min = min(d_top_i[r,c], d_bottom_i[r,c], d_left_i[r,c], d_right_i[c])
  ```

* The **minimum ring thickness** over this grid is:

  ```python
  if inner_i.any():
      thickness_min_i = int(np.min(np.minimum.reduce(
          [d_top_i[inner_i], d_bottom_i[inner_i], d_left_i[inner_i], d_right_i[inner_i]]
      )))
  else:
      thickness_min_i = None  # no inner region, so no well-defined ring thickness
  ```

This matches the intuitive “how many steps of frame do I have between the inner region and the border?” and is a direct reading of “min ring width”.

Again, no hacks:

* If there is no inner cell, we **do not guess**; we set `thickness_min_i = None` and let S0 handle that later (e.g. by not using thickness for that grid).

#### 4.3 Simple row/col period hints per grid (inside inner)

From §3.2 (periodicity condition):

> “If train_out rows (or cols) have least period (p) everywhere inside the inner region, then (p \mid H') (resp. (p \mid W')).”

We implement this strictly:

* For **each grid i**:

  * For row periods:

    * For each row r where `inner_i[r,:]` has at least 1 True, extract the row segment inside the inner region:

      ```python
      row_vals = Y_i[r, inner_i[r, :]]  # 1D vector of colors
      ```

    * Compute its **least period** `p_r` (in indices) exactly:

      ```python
      def least_period(seq: np.ndarray) -> int:
          n = len(seq)
          for p in range(1, n+1):
              if n % p != 0:
                  continue
              # pattern of length p repeats exactly
              if np.all(seq == np.resize(seq[:p], n)):
                  return p
          return n
      ```

    * Collect all such `p_r` over rows that have inner cells.

    * If **all** those rows share the same `p_r = p_row_i` and `p_row_i >= 2` (we ignore trivial period=1), then:

      * `row_period_i = p_row_i`
      * Else: `row_period_i = None`.

  * For column periods: analogous, but over columns `c` and `Y_i[inner_i[:,c], c]`.

We do **not** try to unify rows/cols with different patterns; we treat that as “no reliable period hint” (`None`). That’s consistent with the spec: only when the least period is well-defined everywhere do we use it in S0.

---

## 7. Aggregated global hints for S0

Once we have per-output info:

```python
per_output[i] has:
  "thickness_min" = thickness_min_i,
  "row_period"    = row_period_i,
  "col_period"    = col_period_i,
  "has_midrow"    = has_midrow_i,
  "has_midcol"    = has_midcol_i,
```

we derive `scaffold["aggregated"]`:

```python
thickness_candidates = [t for t in thickness_min_list if t is not None]
thickness_min_global = min(thickness_candidates) if thickness_candidates else None

# row_period_global: only if all non-None row_period_i are equal
row_period_values = [p for p in row_period_list if p is not None]
if row_period_values and len(set(row_period_values)) == 1:
    row_period_global = row_period_values[0]
else:
    row_period_global = None

# similarly for col_period
col_period_values = [p for p in col_period_list if p is not None]
if col_period_values and len(set(col_period_values)) == 1:
    col_period_global = col_period_values[0]
else:
    col_period_global = None

has_midrow_all = all(per_output[i]["has_midrow"] for i in outputs)
has_midcol_all = all(per_output[i]["has_midcol"] for i in outputs)

scaffold["aggregated"] = {
    "thickness_min": thickness_min_global,
    "row_period":    row_period_global,
    "col_period":    col_period_global,
    "has_midrow_all": has_midrow_all,
    "has_midcol_all": has_midcol_all,
}
```

This matches §3.2:

* If *all* train_out have a midrow → `has_midrow_all=True`; S0 can enforce `H'` odd.
* If *all* train_out have the same row-period p_row → `row_period=p_row`; S0 can check divisibility.
* If any grid breaks the pattern (e.g., different period or no inner), `row_period` becomes `None` and S0 simply does not use that filter (no guessing).

No heuristics; strictly “all or nothing”.

---

## 8. `run.py` changes

None.

* `run.py` remains:

  ```python
  scaffold = build_scaffold(canonical, trace=trace)
  ```

* After WO-2.1/2.2/2.3, `build_scaffold` is responsible for returning the full `scaffold` dict; `run.py` just passes it to `size_choice`.

---

## 9. Receipts for WO-2.3

Under `--trace`, for each `train_out[i]`, log e.g.:

```text
[scaffold] output#i inner: shape=(H_i, W_i), sum=<N_inner_cells>
[scaffold] output#i has_midrow=<True/False>, has_midcol=<True/False>, thickness_min=<int or None>, row_period=<p or None>, col_period=<p or None>
[scaffold] aggregated: thickness_min=..., row_period=..., col_period=..., has_midrow_all=..., has_midcol_all=...
```

Reviewer can:

* Check that `inner` is zero where expected (tiny grids) and positive where expected (border-thick tasks),
* Check that `has_midrow` matches the distance geometry (rows where `d_top == d_bottom`),
* Check thickness_min matches visual border thickness on simple cases,
* Check period hints match obvious stripe / tiling tasks.

---

## 10. Reviewer instructions & how to judge correctness

**Run commands (example):**

1. Border-frame + inner region task (from your M1_M2 goldens):

   ```bash
   python run.py --task-id <frame_task_id> \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

   Expect:

   * Each per_output entry has non-zero `inner` area,
   * `thickness_min` ≈ visual border thickness,
   * `has_midrow_all` True if there’s a visible symmetric midrow, etc.

2. Stripe/period task:

   ```bash
   python run.py --task-id <stripe_task_id> \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

   Expect:

   * `inner` covers the stripe region,
   * `row_period` or `col_period` equals the visible stripe period (2, 3, etc.),
   * `has_midrow_all` may be True/False depending on pattern symmetry.

3. Non-framed / messy task:

   ```bash
   python run.py --task-id 00576224 \
                 --data data/arc-agi_training_challenges.json \
                 --test-index 0 \
                 --trace
   ```

   Expect:

   * `inner` is the strict interior (no frame from Stage F except border),
   * `thickness_min` = 1 for a 1-pixel border, or `None` if no interior,
   * `row_period`/`col_period` likely `None` if rows/cols inside inner do not share a common period.

**Legit vs bug:**

* **Legit implementation:**

  * Uses only `train_out` and their known distances,
  * Handles every grid per definitions above,
  * Logs consistent stats (min=0, inner cells only where all d_* > 0, thickness as min distance from inner to frame, periods as exact repeats).

* **Spec gap scenario:**

  * If you later decide you want a richer notion of “frame thickness” than `min(min(d_* on inner))`, or a different period rule, that’s an update to anchors. Until then, we do not “tune” formulas; we stick to the simplest exact interpretation of “min ring width” and “least period in inner region”.

* **Bug (implementation error):**

  * If `inner` is not defined as `(d_top>0)&...` exactly,
  * If `has_midrow` is computed from something other than `d_top == d_bottom`,
  * If thickness or period hints are computed using guessed or partial equality (e.g., tolerances, “almost equal”),
  * If any of these use input or test_out (they must use train_out only in this stage per 01_STAGES and 00_MATH_SPEC).

---
