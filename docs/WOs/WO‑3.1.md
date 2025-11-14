## WO-3.1 — `04_size_choice`: Size map candidates (enumeration only)

(~140–200 LOC inside `04_size_choice/step.py` + tiny `run.py` wiring)

**Goal (per anchors):**

Given canonicalized training grids, enumerate all **integer-exact size map families** that reproduce **all** training size pairs from `(H_in, W_in)` → `(H_out, W_out)`:

* Identity / swap
* Integer affine: `[H'; W'] = M·[H; W] + b`
* Factor maps: `H' = r_H·H`, `W' = r_W·W`
* Tile / concat: `H' = n_v·H + δ_H`, `W' = n_h·W + δ_W`
* Constant size (when all training outputs share the same size)

This WO **only** enumerates candidates and attaches an explicit reproduction table. Screening with scaffold facts and final size choice happens in **WO-3.2**.

---

### 0. Anchors to read before coding

Implementer and reviewer should re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * Section **3. Stage S0 — Output canvas size**

     * Especially **3.1 Candidate size maps**:

       * Identity / swap
       * Integer affine maps
       * Factor maps
       * Tiling / concat maps
       * Constant size
     * And the note that each must **fit all training size pairs exactly**.

2. **` @docs/anchors/01_STAGES.md `**

   * Stage **04_size_choice**:

     * “size_choice — How big the next canvas is”
     * It learns the input→output size relation from training pairs and uses scaffold only for screening (which is **next WO**).

3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * The S0 / size_choice part where:

     * size_choice is a **free** step (no paid bits),
     * It must not peek at test_out,
     * It must be **deterministic**.

I am using these literally: no extra families, no changed semantics, no “smart” guessing.

---

### 1. Packages / libraries

We do **not** implement algorithms by hand; we wire mature building blocks:

* `numpy`

  * `np.asarray`, `.shape`, integer arithmetic, broadcasting
* `itertools` (stdlib, mature)

  * `itertools.product`, `combinations` for small cartesian products
* `typing`

  * `Dict`, `List`, `Tuple`, `Literal`, `Optional`, `TypedDict` if you want stricter types

No numeric optimization, no custom linear solving. For the affine family we use `numpy.linalg.lstsq` to obtain candidate integer matrices/vectors and then **check** exactness.

---

### 2. Input / output contract for this WO

#### 2.1. Assumed input (from previous milestones)

`04_size_choice/step.py` exports:

```python
def choose(
    canonical: dict,
    scaffold: dict,
    trace: bool = False,
) -> dict:
    ...
```

For WO-3.1 we **only** use:

* `canonical["train_in"]`:

  * List of H×W numpy int arrays in canonical gauge.
* `canonical["train_out"]`:

  * List of H'×W' numpy int arrays in canonical gauge.

We **do not** use `scaffold` in this WO (that is WO-3.2’s job). You may assert its presence but not inspect it.

Assumption is consistent with earlier WOs and 01_STAGES: `truth` has already canonicalized grids and `scaffold` has been built for `train_out`.

#### 2.2. Output structure (partial for this WO)

`choose(...)` returns a dict:

```python
SizeChoiceResult = {
    "status": str,          # for this WO: always "CANDIDATES_ONLY"
    "H_out": None,          # will be set in WO-3.2
    "W_out": None,          # will be set in WO-3.2
    "candidates": List[dict],
    "train_size_pairs": List[dict],  # for receipts/debugging
}
```

Where each entry in `"train_size_pairs"` is:

```python
{
  "H_in": int,
  "W_in": int,
  "H_out": int,
  "W_out": int,
}
```

And each candidate in `"candidates"` is a dict:

```python
{
  "family": Literal["identity", "swap", "factor", "affine", "tile", "constant"],

  # parameters depend on family:
  "params": dict,

  # bookkeeping / receipts:
  "fits_all": bool,        # must be True to be kept
  "reproductions": List[dict],  # one per training pair
}
```

Each `"reproductions"` entry is:

```python
{
  "H_in": int, "W_in": int,
  "H_out_pred": int, "W_out_pred": int,
  "H_out_true": int, "W_out_true": int,
  "match": bool,  # H_out_pred == H_out_true and W_out_pred == W_out_true
}
```

For WO-3.1:

* You **only keep** candidates with `fits_all == True`.
* You **do not** pick a final size; leave `status="CANDIDATES_ONLY"`, `H_out=None`, `W_out=None`.

WO-3.2 will:

* Take this result,
* Apply scaffold-based screens,
* Set `status` and `(H_out, W_out)` accordingly.

---

### 3. Extracting training size pairs

Implement in `04_size_choice/step.py`:

```python
def _collect_train_size_pairs(canonical: dict) -> List[dict]:
    ...
```

Algorithm:

1. Read `train_in_list = canonical["train_in"]` and `train_out_list = canonical["train_out"]`.
2. Assert `len(train_in_list) == len(train_out_list) > 0`. If 0, raise `ValueError("No training pairs")`.
3. For each index `i`:

   * `H_in, W_in = train_in_list[i].shape`
   * `H_out, W_out = train_out_list[i].shape`
   * Append dict `{H_in, W_in, H_out, W_out}`.
4. Return the list.

**No shortcuts:** exact shapes only; no “min/max over training”. This is straight from the anchor spec.

---

### 4. Candidate families (exact semantics)

For all families, a candidate is valid only if it reproduces **all** training size pairs exactly.

#### 4.1 Identity

Condition:

* For all training pairs: `H_out == H_in` and `W_out == W_in`.

If true, produce:

```python
{
  "family": "identity",
  "params": {},
  ...
}
```

Reproduction:

* For each pair, `H_out_pred = H_in`, `W_out_pred = W_in`.

#### 4.2 Swap

Condition:

* For all training pairs: `H_out == W_in` and `W_out == H_in`.

If true, candidate:

```python
{
  "family": "swap",
  "params": {},
  ...
}
```

Reproduction uses `H_out_pred = W_in`, `W_out_pred = H_in`.

#### 4.3 Factor maps

Form:

* `H_out = r_H * H_in`,
* `W_out = r_W * W_in`,
  with **integer** `r_H`, `r_W`.

Algorithm:

1. For each pair:

   * If `H_in == 0` or `W_in == 0`, raise `ValueError("Zero dimension in training; unsupported.")` (fail loudly).
   * Check `H_out % H_in == 0` and `W_out % W_in == 0`.

     * If any fails, there is **no factor candidate at all**. Return `None` for factor.
2. Collect `r_H_i = H_out // H_in`, `r_W_i = W_out // W_in`.
3. If all `r_H_i` are equal to a single `r_H` and all `r_W_i` equal to `r_W`, define:

```python
params = {"r_H": int(r_H), "r_W": int(r_W)}
```

4. Build reproduction table by applying these factors back to all training inputs. Set `fits_all=True` only if every pair is an exact match.

Note: This is strictly integer. No floats, no rounding.

#### 4.4 Integer affine maps

Form (from anchors):

[
\begin{bmatrix}
H'\ W'
\end{bmatrix}
=============

M \begin{bmatrix}H\ W\end{bmatrix}

* b
  ]

where (M) is a 2×2 integer matrix and (b) a 2×1 integer vector.

We **do not** brute-force all possibilities; we use linear algebra then exact integer checks.

Algorithm:

1. If `len(train_size_pairs) < 3`:

   * The system is underdetermined (6 unknowns from a 2×2 int matrix and 2-vector).
   * For this WO, **emit no affine candidate**. This is consistent with anchors: we are not required to find an affine map when there isn’t enough data to determine one uniquely.
2. Else:

   * Build the linear system for all training pairs:

     For each pair (i):
     [
     H'*i = M*{11} H_i + M_{12} W_i + b_1
     ]
     [
     W'*i = M*{21} H_i + M_{22} W_i + b_2
     ]

     Unknown vector:
     [
     u = (M_{11}, M_{12}, M_{21}, M_{22}, b_1, b_2)^\top \in \mathbb{R}^6
     ]

     Build `A` (2T×6) and `y` (2T) using numpy.

   * Use `numpy.linalg.lstsq(A, y, rcond=None)` to get a candidate `u_real`.

   * Check residual:

     * If `max_abs_residual > 1e-9`, discard affine (no candidate).

   * Round:

     * `u_int = np.rint(u_real).astype(int)`.
     * If any `abs(u_int - u_real) > 1e-9`, discard (not truly integer).

   * Extract:

     * `M = [[M11, M12],[M21,M22]]`, `b = [b1,b2]`.

   * Apply to **all** training inputs:

     * For each `(H_in, W_in)` compute:

       ```python
       H_pred = M11*H_in + M12*W_in + b1
       W_pred = M21*H_in + M22*W_in + b2
       ```

     * If **every** `(H_pred, W_pred)` equals `(H_out, W_out)` exactly (no tolerance), accept one affine candidate with:

       ```python
       params = {
         "M": [[M11, M12], [M21, M22]],
         "b": [b1, b2],
       }
       ```

   * If any mismatch: no affine candidate.

No bounds/heuristics on M,b beyond what comes from the data. We’re using exact integers via solving then checking; no hacks.

#### 4.5 Tile / concat maps

Form (from anchors):

* `H_out = n_v * H_in + δ_H`,
* `W_out = n_h * W_in + δ_W`,
  with integers `n_v, n_h ≥ 0` and offsets `δ_H, δ_W` (can be any integer, but naturally 0 ≤ δ < some small bound; do **not** assume that in math; implementation will infer exactly).

Algorithm (brute but bounded; grids ≤30 so safe on CPU):

For H dimension:

1. For each training pair (i), we want integer solutions of:
   [
   H'_i = n_v H_i + \delta_H
   ]

2. For each pair `i`, we can enumerate all possible `(n_v, δ_H)` that satisfy this exactly:

   * For `n_v` from 0 up to `H_out // max(1, H_in)` + 1:

     * Compute `δ_H = H_out - n_v * H_in`.
     * Keep if `δ_H` is integer (it is) and no extra condition. (We do **not** restrict δ_H to `[0,H)`; anchor spec does not.)
   * Put all `(n_v, δ_H)` for pair `i` into a set.

3. Intersect the sets across all training pairs:

   * `common_H = set_for_pair0 ∩ set_for_pair1 ∩ ...`

Repeat the same for W:

* Enumerate `(n_h, δ_W)` pairs per training pair and intersect to get `common_W`.

Finally:

* For each `(n_v, δ_H)` in `common_H` and `(n_h, δ_W)` in `common_W`:

  * This defines a candidate tile map.
  * Build reproduction table and keep only those with exact matches everywhere.
* To avoid blowing up the candidate list:

  * The intersections will usually be tiny; but if `len(common_H) * len(common_W)` is huge (unlikely given ARC sizes), you can **hard cap** e.g. at a reasonable number and raise `RuntimeError("Too many tile candidates")`. That is a loud failure, not a hack.

Candidate structure:

```python
{
  "family": "tile",
  "params": {
    "n_v": int,
    "n_h": int,
    "delta_H": int,
    "delta_W": int,
  },
  ...
}
```

#### 4.6 Constant size

Condition:

* All `H_out` the same across training pairs, call it `H_const`.
* All `W_out` the same across training pairs, call it `W_const`.

If so, candidate:

```python
{
  "family": "constant",
  "params": {
    "H_const": int(H_const),
    "W_const": int(W_const),
  },
  ...
}
```

Reproduction uses the same `H_const, W_const` for every training pair.

---

### 5. Implementation outline in `04_size_choice/step.py`

Suggested structure:

```python
import numpy as np
from itertools import product
from typing import Dict, List, Any

def _collect_train_size_pairs(canonical: Dict[str, Any]) -> List[Dict[str, int]]:
    ...

def _enumerate_identity_candidate(train_sizes: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    ...

def _enumerate_swap_candidate(...):
    ...

def _enumerate_factor_candidate(...):
    ...

def _enumerate_affine_candidate(...):
    ...

def _enumerate_tile_candidates(...):
    ...

def _enumerate_constant_candidate(...):
    ...

def enumerate_size_maps(canonical: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns {
      "train_size_pairs": [...],
      "candidates": [...],
    }
    """
    ...

def choose(canonical: Dict[str, Any], scaffold: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    For WO-3.1: enumerate candidates only.
    """
    size_data = enumerate_size_maps(canonical)
    result = {
        "status": "CANDIDATES_ONLY",
        "H_out": None,
        "W_out": None,
        "train_size_pairs": size_data["train_size_pairs"],
        "candidates": size_data["candidates"],
    }
    if trace:
        # print or log a concise summary for receipts
        _trace_size_choice(result)
    return result

def _trace_size_choice(result: Dict[str, Any]) -> None:
    ...
```

Note: **No screening**, no final `(H_out, W_out)` decision here. That is WO-3.2.

---

### 6. `run.py` / brainstem changes

Keep `run.py` minimal, no god function.

Assume you currently have something like:

```python
from 01_present.step import load as load_present
from 02_truth.step import canonicalize as canonicalize_truth
from 03_scaffold.step import build as build_scaffold
# size_choice not yet wired or stubbed

def main(...):
    present = load_present(task_bundle, trace=trace)
    canonical = canonicalize_truth(present, trace=trace)
    scaffold = build_scaffold(canonical, trace=trace)
    # TODO: size_choice here
```

Update to:

```python
from 04_size_choice.step import choose as choose_size
```

And in the main flow:

```python
present = load_present(task_bundle, trace=trace)
canonical = canonicalize_truth(present, trace=trace)
scaffold = build_scaffold(canonical, trace=trace)

size_choice = choose_size(canonical, scaffold, trace=trace)

# For now (WO-3.1), we do nothing else with size_choice
# Later WOs (laws, minimal_act) will read size_choice["H_out"], size_choice["W_out"].
```

**No other logic** in `run.py`. It just wires stages and passes `trace`.

---

### 7. Receipts / trace behavior for this WO

Under `trace=True`, `choose` should print or log to stdout something like:

* Count of training size pairs
* The list of candidates: family + params + whether they fit all.

Example for task `00576224` (from `M3_checkpoints.md`):

```text
[size_choice] train_size_pairs: 3
[size_choice] candidates (family, params, fits_all):
  - factor: {'r_H': 3, 'r_W': 3}, fits_all=True
  - identity: fits_all=False (dropped)
  - swap: fits_all=False (dropped)
  - ...
[size_choice] status=CANDIDATES_ONLY
```

You don’t need to match exact wording, but the **content** must be equivalent.

---

### 8. Reviewer instructions

#### 8.1. Using `run.py` on a real ARC task

Example: `00576224` golden (from `M3_checkpoints.md`).

1. Put the official ARC JSON for `00576224` somewhere, e.g., `tasks/00576224.json`.

2. Run:

   ```bash
   python run.py --task tasks/00576224.json --trace
   ```

3. Expected behavior (WO-3.1 only):

   * Stages 01–03 behave as before.
   * Size_choice prints one **factor** candidate with `r_H=3`, `r_W=3` and `fits_all=True`.
   * `size_choice["status"] == "CANDIDATES_ONLY"`.
   * `size_choice["H_out"] is None`, `size_choice["W_out"] is None`.
   * `size_choice["train_size_pairs"]` entries match the JSON (e.g., 2→6, 2→6, etc.).

This means candidate enumeration is correct; screening is not yet in scope.

#### 8.2. Additional sanity tasks

Test 1: Identity map

* Construct a tiny synthetic ARC-like task where `train_in[i].shape == train_out[i].shape` for all i.
* Expect:

  * Identity candidate with `fits_all=True`.
  * No factor candidate unless the shapes are also clean multiples.
  * No affine/tile unless they match by chance.

Test 2: Constant map

* Make all `train_out[i]` the same size regardless of `train_in`.
* Expect:

  * `constant` candidate with `H_const, W_const` equal to that size.
  * Possibly other families if they fit, but WO-3.1 does not pick; it only enumerates.

If any reproduction table shows `match=False` entries for a candidate that is kept, that is an implementation bug violating the math spec.

#### 8.3. Identifying legit implementation gaps vs spec limits

* If there are <3 training pairs, and **no affine candidate is produced**, that is **correct**, not an error. The spec does not require guessing underdetermined affine maps.
* If factor or tile candidates are not produced, check:

  * Are the divisibility / linear equations actually satisfied? If not, there should be no such candidate.
* If `choose` sets a status other than `"CANDIDATES_ONLY"` in this milestone, or tries to pick `H_out, W_out`, that is a spec violation (WO-3.2 territory).

Any deviation from the algebra above is a math/spec mismatch, not a “reasonable alternative.”

---

