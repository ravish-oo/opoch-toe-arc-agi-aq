## WO-3.2 — `04_size_choice`: Screens from scaffold (S0 structural filters)

(~80–140 LOC inside `04_size_choice/step.py` + no structural changes to `run.py` beyond what WO-3.1 already did)

**Goal (per anchors):**

Take the **size map candidates** from WO-3.1 and use **only scaffold facts from train_out** to decide which candidate test size `(H_out, W_out)` survives for the **test input**.

Screening criteria (from `00_MATH_SPEC.md` §3.2):

1. **Feasibility:** frame thickness and inner region must fit:
   [
   H' \ge h_{\text{inner}} + 2t,\quad W' \ge w_{\text{inner}} + 2t
   ]
   for every training output where thickness and inner are defined.
2. **Parity:** if all train_out have a midrow then H' must be odd; similar for midcol → W' odd.
3. **Periodicity:** if row/col periods inside inner are well-defined and common, they must divide H' and W'.
4. **Tiling constants:** for tile maps, if frame thickness t is detected, offsets must satisfy (\delta_H,\delta_W \in {0,2t}).
5. **Crude capacity:** *only* if such information is recorded in scaffold (currently it is not, so we skip this explicitly).

Policy:

* If **1** survivor → choose it: `status="OK"`, set `(H_out,W_out)`.
* If **0** survivors → infeasible: `status="IIS"`, `(H_out,W_out) = (None,None)`.
* If **>1** → ambiguous: `status="AMBIGUOUS_SIZE"`, do **not** pick one arbitrarily.

---

### 0. Anchors to read before coding

Implementer + reviewer should re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * Section **3. Stage S0 — Output canvas size**

     * Especially **3.2 Structural disambiguation (train_out-only)**:

       * Feasibility (frame thickness + inner region),
       * Parity (midrow/midcol),
       * Periodicity (row/col periods divide H',W'),
       * Tiling constants rule (`δ_H,δ_W ∈ {0,2t}`),
       * Crude capacity (only “if recorded”).

2. **` @docs/anchors/01_STAGES.md `**

   * Stage **04_size_choice**:

     * size_choice uses **only scaffold facts from train_out** to screen candidates and pick test size,
     * No peeking at test_out.

3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * S0 mapping:

     * size_choice is a **free** step: no paid bits, no optimization beyond deterministic checks,
     * It must be deterministic and use only train_out scaffold.

We are not adding new interpretations; we are implementing exactly what those sections say.

---

### 1. Libraries

We do not implement algorithms; we use mature, basic pieces:

* `numpy`

  * `np.asarray`, `.shape`, `np.where`, `.min()`, `.max()`, `%` (mod), etc.
* `typing`

  * `Dict`, `Any`, `List`, `Tuple`, `Optional`, `Literal` if you want.

No graph libs, no optimization libs in this WO. It is pure arithmetic and mask operations.

---

### 2. Input / output contract

#### 2.1 Input (assumed from earlier milestones)

`04_size_choice/step.py`:

```python
def choose(
    canonical: dict,
    scaffold: dict,
    trace: bool = False,
) -> dict:
    ...
```

We assume:

* `canonical["train_in"]`: list of np.ndarray (H_in×W_in) for each training input.

* `canonical["train_out"]`: list of np.ndarray (H_out×W_out) for each training output.

* `canonical["test_in"]`: list of np.ndarray for test inputs.

  * **Spec choice**: For now we assume **exactly one** test input:

    * If `len(canonical["test_in"]) != 1`, raise:

      ```python
      raise NotImplementedError(
          "Multiple test inputs not yet supported in size_choice spec; extend anchors before implementing."
      )
      ```

    This is consistent with current anchors and avoids improvisation.

* `scaffold` as produced by `03_scaffold.step.build` (WO-2.1–2.3):

  ```python
  scaffold = {
    "per_output": [
      {
        "frame_mask": np.ndarray[bool],         # from WO-2.1
        "d_top": np.ndarray[int],               # from WO-2.2
        "d_bottom": np.ndarray[int],
        "d_left": np.ndarray[int],
        "d_right": np.ndarray[int],
        "inner": np.ndarray[bool],              # from WO-2.3
        "thickness_min": Optional[int],
        "row_period": Optional[int],
        "col_period": Optional[int],
        "has_midrow": bool,
        "has_midcol": bool,
      },
      ...
    ],
    "aggregated": {
      "thickness_min": Optional[int],
      "row_period": Optional[int],
      "col_period": Optional[int],
      "has_midrow_all": bool,
      "has_midcol_all": bool,
    },
  }
  ```

We also rely on **WO-3.1** providing an internal helper (or equivalent):

```python
from 04_size_choice.step import enumerate_size_maps
# or a private `_enumerate...` function; adapt name as used.

size_data = enumerate_size_maps(canonical)
# size_data["candidates"] is a list of candidates with .family, .params, .fits_all
```

If this function does not exist yet, the implementer should **add it now** (per WO-3.1). If it is named differently, adapt.

#### 2.2 Output (final S0 result)

`choose(...)` returns:

```python
SizeChoiceResult = {
  "status": str,              # "OK", "IIS", or "AMBIGUOUS_SIZE"
  "H_out": Optional[int],     # chosen test H_out for this task
  "W_out": Optional[int],     # chosen test W_out for this task
  "train_size_pairs": List[dict],  # as in WO-3.1
  "candidates": List[dict],        # all candidates from WO-3.1
  "survivors": List[dict],         # subset of candidates that passed screens
}
```

Each `candidate` has at least:

```python
{
  "family": Literal["identity", "swap", "factor", "affine", "tile"], # "constant" optional in future
  "params": dict,
  "fits_all": bool,
  # plus "reproductions" etc. from WO-3.1
}
```

Each `survivor` adds:

```python
{
  "family": ...,
  "params": ...,
  "fits_all": True,
  "H_out_test": int,   # H' for test input
  "W_out_test": int,   # W' for test input
}
```

For this WO:

* `status` is one of `"OK"`, `"IIS"`, `"AMBIGUOUS_SIZE"`.
* If `status=="OK"`, then:

  * `len(survivors) == 1`,
  * `H_out`, `W_out` equal `survivors[0]["H_out_test"]`, `["W_out_test"]`.
* If `"IIS"` or `"AMBIGUOUS_SIZE"`, then `H_out=W_out=None`.

---

### 3. Computing candidate test sizes

Add a helper:

```python
def _apply_candidate_to_test_size(
    candidate: dict,
    H_test: int,
    W_test: int,
) -> Tuple[int, int]:
    ...
```

Behavior, grounded in WO-3.1’s families:

* `identity`:

  * `H' = H_test`, `W' = W_test`.
* `swap`:

  * `H' = W_test`, `W' = H_test`.
* `factor` with `{"r_H": r_H, "r_W": r_W}`:

  * `H' = r_H * H_test`, `W' = r_W * W_test`.
* `affine` with `{"M": [[M11,M12],[M21,M22]], "b": [b1,b2]}`:

  * `H' = M11*H_test + M12*W_test + b1`,
  * `W' = M21*H_test + M22*W_test + b2`.
* `tile` with `{"n_v": n_v, "n_h": n_h, "delta_H": δ_H, "delta_W": δ_W}`:

  * `H' = n_v * H_test + δ_H`,
  * `W' = n_h * W_test + δ_W`.

If any other `family` value appears:

```python
raise NotImplementedError(f"Unknown size map family: {candidate['family']}")
```

After computing `(H',W')`:

* If `H' <= 0` or `W' <= 0`, **reject this candidate immediately** (cannot be a valid grid size).

---

### 4. Scaffold-based screens (exact rules)

Implement:

```python
def _passes_scaffold_screens(
    H_out_test: int,
    W_out_test: int,
    candidate: dict,
    scaffold: dict,
) -> bool:
    ...
```

Subscreens:

#### 4.1 Feasibility: thickness + inner region

From `00_MATH_SPEC.md` §3.2:

> For every training output’s learned frame thickness (t) and inner region (h_inner, w_inner):
>
> H' ≥ h_inner + 2t, W' ≥ w_inner + 2t.

Use:

* `per_output = scaffold["per_output"]`.

For each `info` in `per_output`:

* `t_i = info["thickness_min"]`
* `inner_i = info["inner"]` (bool mask)

If `t_i is None` or `inner_i` has no `True`:

* This output provides no binding constraint; **skip** it.

Else:

* Use numpy to get inner bounding box:

  ```python
  rows, cols = np.where(inner_i)
  r_min, r_max = rows.min(), rows.max()
  c_min, c_max = cols.min(), cols.max()
  h_inner_i = int(r_max - r_min + 1)
  w_inner_i = int(c_max - c_min + 1)
  ```

* Check:

  ```python
  if H_out_test < h_inner_i + 2 * t_i:
      return False
  if W_out_test < w_inner_i + 2 * t_i:
      return False
  ```

If all outputs pass these inequalities, the candidate passes feasibility.

#### 4.2 Parity: midrow / midcol

From `00_MATH_SPEC.md` §3.2:

> If all train_out have a midrow then H' must be odd; similarly for W' and midcol.

We use `scaffold["aggregated"]`:

```python
agg = scaffold["aggregated"]
if agg["has_midrow_all"]:
    if H_out_test % 2 == 0:
        return False
if agg["has_midcol_all"]:
    if W_out_test % 2 == 0:
        return False
```

If no midrow_all or midcol_all, no parity constraint.

#### 4.3 Periodicity: periods divide H', W'

From `00_MATH_SPEC.md` §3.2:

> If train_out rows (or cols) have least period p everywhere inside inner region, then (p | H') (resp. (p | W')).

Use:

```python
row_p = agg["row_period"]  # could be None
col_p = agg["col_period"]
```

Rules:

* If `row_p is not None` and `row_p > 0`:

  * require `H_out_test % row_p == 0`, else `return False`.
* If `col_p is not None` and `col_p > 0`:

  * require `W_out_test % col_p == 0`, else `return False`.

If both pass (or are None), continue.

#### 4.4 Tiling constants for tile maps

From `00_MATH_SPEC.md` §3.2:

> For maps (H' = n_v H + δ_H, W' = n_h W + δ_W), require the same integers (n_v, n_h, δ_H, δ_W) fit all trainings. If frame thickness (t) is detected, enforce (δ_H, δ_W ∈ {0, 2t}) only.

WO-3.1 already ensures the same integers fit all training pairs via intersection. Here we add the `δ ∈ {0,2t}` constraint at S0:

* Let `t_global = agg["thickness_min"]`.

Rule:

* If `candidate["family"] == "tile"` and `t_global is not None`:

  * read `δ_H = candidate["params"]["delta_H"]`, `δ_W = ...`.
  * allowable offsets:

    ```python
    allowed = {0, 2 * t_global}
    if δ_H not in allowed or δ_W not in allowed:
        return False
    ```

If `t_global is None`, we do **not** apply this δ-screen; this is exactly “if frame thickness (t) is detected” clause.

#### 4.5 Crude capacity (anchor says “if recorded”)

Currently, `WO-2.3` defines:

* `thickness_min`, `row_period`, `col_period`, `has_midrow_all`, `has_midcol_all`.

It does **not** define any explicit “capacity” fields (e.g., total interior area, min required area, etc.).

To remain strictly anchored and avoid inventing scaffold structures:

* **We do not implement any capacity screen yet.**
* You may leave a commented placeholder:

  ```python
  # NOTE: capacity-based screens are not implemented;
  # scaffold anchors do not define capacity fields yet.
  ```

If in future anchors add explicit capacity info to `scaffold["aggregated"]`, WO-3.2 can be extended.

---

### 5. Main `choose` flow

Update `04_size_choice/step.py`:

```python
def choose(canonical: Dict[str, Any], scaffold: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    # 1. enumerate size map candidates (WO-3.1)
    size_data = enumerate_size_maps(canonical)
    train_size_pairs = size_data["train_size_pairs"]
    candidates = size_data["candidates"]

    # 2. get test input size (single test only)
    test_in_list = canonical.get("test_in", [])
    if len(test_in_list) != 1:
        raise NotImplementedError(
            "Multiple test inputs not yet supported in size_choice; extend spec first."
        )
    H_test, W_test = test_in_list[0].shape

    survivors: List[Dict[str, Any]] = []

    # 3. apply each candidate to test size and run scaffold screens
    for cand in candidates:
        if not cand.get("fits_all", False):
            continue  # paranoia: WO-3.1 should have filtered already

        H_out_test, W_out_test = _apply_candidate_to_test_size(cand, H_test, W_test)
        if H_out_test <= 0 or W_out_test <= 0:
            continue

        if _passes_scaffold_screens(H_out_test, W_out_test, cand, scaffold):
            survivor = dict(cand)
            survivor["H_out_test"] = int(H_out_test)
            survivor["W_out_test"] = int(W_out_test)
            survivors.append(survivor)

    # 4. decide status
    if len(survivors) == 0:
        status = "IIS"
        H_out = None
        W_out = None
    elif len(survivors) == 1:
        status = "OK"
        H_out = survivors[0]["H_out_test"]
        W_out = survivors[0]["W_out_test"]
    else:
        status = "AMBIGUOUS_SIZE"
        H_out = None
        W_out = None

    result = {
        "status": status,
        "H_out": H_out,
        "W_out": W_out,
        "train_size_pairs": train_size_pairs,
        "candidates": candidates,
        "survivors": survivors,
    }

    if trace:
        _trace_size_choice(result)

    return result
```

Trace helper:

```python
def _trace_size_choice(result: Dict[str, Any]) -> None:
    print("[size_choice] status:", result["status"])
    print("[size_choice] H_out, W_out:", result["H_out"], result["W_out"])
    print("[size_choice] num_train_pairs:", len(result["train_size_pairs"]))
    print("[size_choice] num_candidates:", len(result["candidates"]))
    print("[size_choice] num_survivors:", len(result["survivors"]))
    for s in result["survivors"]:
        print("  survivor:", s["family"], s.get("params", {}), "->", s["H_out_test"], "x", s["W_out_test"])
```

---

### 6. `run.py` / brainstem changes

**No new structure** in `run.py` beyond what WO-3.1 already needed.

Assuming you already had:

```python
from 01_present.step import load as load_present
from 02_truth.step import canonicalize as canonicalize_truth
from 03_scaffold.step import build as build_scaffold
from 04_size_choice.step import choose as choose_size

def main(...):
    present = load_present(task_bundle, trace=trace)
    canonical = canonicalize_truth(present, trace=trace)
    scaffold = build_scaffold(canonical, trace=trace)
    size_choice = choose_size(canonical, scaffold, trace=trace)
    # later WOs will use size_choice["H_out"], size_choice["W_out"]
```

You **do not** add any further logic here. `run.py` remains a minimal orchestrator. Later stages (laws, minimal_act) will depend on `size_choice["H_out"], ["W_out"]`, not on any richer logic in `run.py`.

---

### 7. Receipts / trace for reviewer

Under `--trace`, after `choose_size`, the console should show:

* `status` (`OK`, `IIS`, `AMBIGUOUS_SIZE`),
* `(H_out,W_out)`,
* counts of candidates and survivors,
* each survivor’s family, params, and predicted test size.

For task **`00576224`** (M3 golden):

* From `M3_checkpoints.md`:

  * Exactly one candidate: factor `(r_H=3, r_W=3)`,
  * That candidate survives all screens,
  * `status="OK"`, `H_out=6`, `W_out=6`.

So with `python run.py --task 00576224.json --trace`, you should see something like:

```text
[size_choice] status: OK
[size_choice] H_out, W_out: 6 6
[size_choice] num_train_pairs: 3
[size_choice] num_candidates: 1
[size_choice] num_survivors: 1
  survivor: factor {'r_H': 3, 'r_W': 3} -> 6 x 6
```

You don’t have to match text, but the **values** must match the golden JSON in `M3_checkpoints.md`:

* `status="OK"`,
* `H_out=6`, `W_out=6`,
* `survivor_count=1`,
* survivor is a factor map with `(r_H=3, r_W=3)`.

---

### 8. Reviewer instructions (2–3 tasks)

#### Task A: Golden `00576224` (factor 3×3)

1. Run:

   ```bash
   python run.py --task tasks/00576224.json --trace
   ```

2. Check:

   * Only one candidate survives (factor map `(r_H=3, r_W=3)`).
   * `status == "OK"`.
   * `H_out == 6`, `W_out == 6`.
   * survivors list has exactly one element with `"family": "factor"` and `"H_out_test": 6`, `"W_out_test": 6`.

   Any extra survivor or different status is an implementation bug.

#### Task B: Synthetic identity case

Create a small task where:

* All `train_in[i]` have shape `(H,W)`,
* All `train_out[i]` also shape `(H,W)`,
* Scaffold has no special periods or midlines (inner exists, thickness_min maybe 1, but row_period/col_period=None, has_midrow_all=False, has_midcol_all=False).

Expectation:

* WO-3.1 must provide an `identity` candidate that fits all.
* WO-3.2 should:

  * compute `(H_out,W_out) = (H_test,W_test)`,
  * screens should **not** reject it (feasibility satisfied),
  * if no other family fits and survives, you get `status="OK"` and `(H_out,W_out)=(H,W)`.

If some period/midline is set in scaffold, check that parity / divisibility logic applies as per spec.

#### Task C: Ambiguous mock (for debugging)

Create a synthetic scenario (not necessarily real ARC) where:

* Both factor and swap are candidates that fit all training pairs,
* Scaffold aggregated hints do not distinguish them (no periods, no midlines, no thickness constraints strong enough).

Then WO-3.2 should yield:

* `len(survivors) > 1`,
* `status="AMBIGUOUS_SIZE"`,
* `H_out=W_out=None`.

This demonstrates that the code respects the “strict uniqueness policy”: it does **not** arbitrarily pick a survivor when multiple are structurally consistent.

#### Spotting legit gaps vs errors

* If no candidate survives and `status="IIS"`:

  * Check candidate enumeration and scaffold values. If the math truly yields no size consistent with all structural constraints, this is a **legit unsatisfiable** case.
  * If you can exhibit a candidate size that passes all the formal inequalities and parity/period rules but code rejected it, that is a **bug**.

* If there are multiple survivors but status is `"OK"` and some `(H_out,W_out)` is chosen, that is a **spec violation**. It must be `"AMBIGUOUS_SIZE"` in that case.

---
=======UPDATED=======
Yeah, I see exactly what’s going on here. Let’s clear it completely and then rewrite WO-3.2 in a way that is:
	•	Fully aligned with TOE (pure A0 at S0),
	•	Consistent with your updated scaffolding (per-output border frame),
	•	And mathematically precise: no minted constraints, no false IIS, no fake ambiguity.

I’ll go point by point, then give a clean replacement spec for WO-3.2 you can drop into 00_MATH_SPEC.

⸻

1.⁠ ⁠What the critique is saying in simple terms

The snippet you pasted is pointing out three problems in the current S0 (size_choice) spec:
	1.	Periodicity:
You’re treating “least period = full row length” (i.e. no real repetition) as if it were a genuine periodicity law and then enforcing p | H' or p | W' on test size. That’s minting a fake structural constraint from mere string length.
	2.	Thickness / inner+2t:
You’re using “inner height + 2·thickness” as a hard global feasibility condition at S0, i.e., as a pure geometry constraint. But thickness is law-level padding (part of how the law chooses to place the pattern), not a purely geometric necessity. Using it at S0 can wrongly reject valid sizes.
	3.	Candidate semantics:
The spec mixes up “candidate map family” and “candidate size”. If three different families (factor, tile, affine) all yield the same (H',W') for the test input, they are Π-equivalent explanations. But S0 counts them as “multiple survivors” and emits AMBIGUOUS_SIZE, even though the size is unique.

From the TOE/A0 perspective:
	•	S0 is a seeing stage: it should only encode geometric truths that are robust and unavoidable, not minted patterns or padding choices.
	•	Anything that depends on content (like “this particular tiling of the inner region”) belongs to laws (Stage N / solver), not to Stage F/S0 geometry.
	•	AMBIGUOUS_SIZE should be about canvas size, not about multiple human “explanations”.

⸻

2.⁠ ⁠Clarifying each issue and what changes

2.1 Periodicity – how to make it A0-clean

Problem:
Current pipeline:
	•	You compute a “least period p” for rows/cols inside innerᵢ, using standard period-finding.
	•	If p equals the full length (no actual repetition), you still record p.
	•	S0 then enforces: “p must divide H’ or W’”.

So a non-repeating row of length 10 produces p=10, and you force 10 | H' on the test output even though the training output never showed any actual repeating structure. That’s minting a pattern from nothing.

A0-compatible fix:

Define true period in S0 as:
	•	For a row (or column) segment of length L, let p be the smallest integer such that:
	•	the pattern repeats at least twice, i.e. 2·p ≤ L, and
	•	the sequence is p-periodic over its domain (all equal in each congruence class mod p).

Then:
	•	If no such p < L/2 exists, you treat the row/col as non-periodic for S0.
	•	You do not enforce any p | H' / p | W' from rows that have no genuine repetition.

On the shape level:
	•	For each training output, you gather real periods (p_rowᵢ, p_colᵢ) inside innerᵢ, where repeats actually occur.
	•	Only if:
	•	the same p_row appears for all train_out,
	•	and there is at least one training grid where the repetition is visible (more than 2 cycles),
you enforce: p_row | H’.
	•	Similarly for p_col and W’.

This way, S0 only enforces periodicity when there is a true repeated pattern, not when “period = full length” is just a representation artifact.

⸻

2.2 Thickness & inner+2t – moving it out of “hard geometry”

Problem:
Current spec has something like:

For each training, learn frame thickness t and inner dims (h_inner, w_inner).
Then require for the test size:
H' ≥ h_inner + 2t, W' ≥ w_inner + 2t.

This acts as if:
	•	The law must always place the inner pattern with exactly t cells of padding on all sides,
	•	And that this padding pattern is a geometric necessity for the canvas size.

But in many ARC tasks:
	•	Padding is law-level: the law might:
	•	leave variable margin,
	•	change thickness with size,
	•	or let the pattern “float” in a larger canvas.

So at S0 (pure “seeing”), using inner+2t as a hard feasibility condition is too strong. It bakes in a specific padding layout as if it were geometry.

A0-compatible fix:

At S0, use thickness only in one-way, sanity form:
	•	You can record:
	•	t_min: minimum observed thickness across train_out,
	•	inner_min: minimum h_inner, w_inner across train_out.

But for test size you only require:
	•	Weak feasibility:
H' ≥ inner_min and W' ≥ w_inner_min.
Optionally: H' ≥ h_out_min and W' ≥ w_out_min (you can’t have a test output smaller than any training output in most ARC behaviors, but even this is arguable).

You do not enforce H' = h_inner + 2·t style equalities at S0. You leave:
	•	The exact margin/placement of the inner region,
	•	The exact thickness of frame,

to Stage N + solver (they’ll determine whether a particular margin is needed to match the law). That’s the “pay only when you have to” principle: don’t constrain geometry before the law demands it.

⸻

2.3 Candidate semantics – size vs explanation family

Problem:
S0 currently says:
	•	“Each candidate (H’,W’) must pass screens.
Policy: 1 survivor → choose; 0 → IIS; >1 → AMBIGUOUS_SIZE.”

But:
	•	WO-3.1 enumerates families (identity, factor, tile, affine, constant).
	•	For the test input, multiple families can produce the same (H',W').
	•	If you treat “candidate” = “family”, you will often end with multiple survivors that all generate the same size – and incorrectly signal AMBIGUOUS_SIZE.

From Π’s viewpoint:
	•	Different explanatory families that yield the same (H',W') are minted differences: they are equivalent descriptions of the same geometric fact.
	•	S0’s job is to fix the size, not to pick a unique causal story.

A0-compatible fix:

In S0, define:
	•	A candidate as a distinct size pair (H',W'), not a map family.
	•	When you enumerate families, apply them to the test input and collect all produced (H',W').
	•	Deduplicate to get a set of distinct size pairs.

Then:
	•	Apply the screens (parity, real periods, weak feasibility) per size.
	•	After screening, look at the set of surviving sizes, not families:
	•	If 1 size remains → choose it.
	•	If 0 → IIS (no size consistent with training geometry).
	•	If >1 sizes remain → AMBIGUOUS_SIZE with that set.

We do not care whether the size came from factor vs affine vs tile; that’s explanatory semantics, which is minting at this stage.

⸻

3.⁠ ⁠Revised WO-3.2 (size_choice) spec

Here’s a clean, complete rewrite of WO-3.2 you can paste into 00_MATH_SPEC.

WO-3.2 Screens from scaffold (revised, A0-compatible)

Inputs:
	•	Training size pairs:
(H_{\text{in}}^{(i)}, W_{\text{in}}^{(i)}) → (H_{\text{out}}^{(i)}, W_{\text{out}}^{(i)}).
	•	From WO-3.1: a set of map families (identity, swap, affine, factor, tile, constant) that reproduce all training size pairs.
	•	From WO-2.3 aggregated scaffold:
	•	has_midrow_all, has_midcol_all,
	•	row_period_real, col_period_real (true periods only),
	•	inner_min = (h_inner_min, w_inner_min) (optional).

Step 1: derive candidate sizes from map families
	•	For the test input size (H_{\text{in}}^{\text{test}}, W_{\text{in}}^{\text{test}}), apply each map family that is consistent with training to get a size pair (H',W').
	•	Collect all such size pairs and deduplicate:
\mathrm{Sizes} = \{(H'_j, W'_j)\}.

We ignore which family produced which size; all families that yield the same size are Π-equivalent for S0.

Step 2: define real periods

From WO-2.3:
	•	For each training output and each row in its inner region, compute row period p_row as:
	•	A smallest integer p such that the row segment is p-periodic and 2·p ≤ L (L = segment length).
	•	If no such p exists, mark this row as “non-periodic”.
	•	From all inners across training outputs:
	•	If there exists a single p_row_real such that:
	•	for every training output, all inners that show repetition share that same p_row_real,
	•	and at least one training has at least 2 cycles (2·p_row_real ≤ L),
then store this as row_period_real.
	•	Otherwise, set row_period_real = None.
	•	Do the same for columns to get col_period_real.

Step 3: screening each candidate size (H',W')

For each candidate size (H',W'):
	1.	Parity screen:
	•	If has_midrow_all is true (every training output has a midrow), require H' to be odd.
	•	If has_midcol_all is true, require W' to be odd.
Otherwise, no parity restriction.
	2.	Periodicity screen (real only):
	•	If row_period_real is not None, require:
row\_period\_real \mid H'.
	•	If col_period_real is not None, require:
col\_period\_real \mid W'.
If there is no real repetition (i.e., row_period_real=None or col_period_real=None), impose no periodicity constraint in that dimension.
	3.	Weak feasibility (inner capacity):
	•	Optionally compute inner_min = (h_inner_min, w_inner_min) as minimum inner dims observed across train_out (based on border frame).
	•	Require:
H' \ge h\{\text{inner\_min}},\quad W' \ge w\{\text{inner\_min}}.
This ensures the test canvas is not smaller than the smallest observed inner pattern; do not enforce H' = h_inner + 2·t or any exact thickness layout here.
	4.	Additional capacity checks (optional):
	•	You can add coarse checks like:
	•	H' ≥ max_i H_{\text{out}}^{(i)}, W' ≥ max_i W_{\text{out}}^{(i)},
if desired, but do not encode say “exact outer ring thickness t” at S0.

If a candidate fails any screen → drop that size from Sizes.

Step 4: final policy
	•	After screening:
	•	If Sizes is empty → IIS_SIZE for S0; no output size is geometrically consistent with training geometry.
	•	If Sizes contains exactly one size (H^\,W^\) → pick that as test_out canvas size.
	•	If Sizes contains more than one distinct size → AMBIGUOUS_SIZE, return the set Sizes.

Note: If you want a stricter policy for the benchmark (e.g., never allow AMBIG_SIZE), that must be enforced by requiring that the training author gave enough examples to collapse Sizes to a singleton.

⸻

4.⁠ ⁠How this fits TOE and “previous version (before N₀)”

You said:

we are executing the previous version before N0, so be mindful

Got it: we’re still in the TU/LP pipeline, not invoking the “global N₀” story. In that context:
	•	A0 is implemented at S0 as:
	•	Don’t invent structure:
	•	No fake periods from full row length,
	•	No hard thickness laws at the geometry stage,
	•	No distinction between different map families that yield the same size.
	•	Stage F is purely per-grid geometry (border frame + distances).
	•	Stage N + solver handle all content/law-level structure.

With the revised WO-3.2 above:
	•	S0 will only rule out sizes when there is genuine geometric evidence that they are impossible.
	•	It will not bake in “accidental” finite-sequence properties or padding choices.
	•	AMBIGUOUS_SIZE will only signal true geometric ambiguity about canvas size.

That gives you a clean, A0-aligned size_choice stage that the rest of your TU/LP implementation can rely on—even before we talk about N₀.

If you’d like, we can next walk one of your known problem cases (like 5e6bbc0b) through this revised S0 to check that:
	•	It produces the intended size,
	•	It no longer throws false AMBIGUOUS_SIZE or IIS.