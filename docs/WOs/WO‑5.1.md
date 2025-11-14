## WO-5.1 — `05_laws/type_keys.py` (or inside `atoms.py`): Type keyer (T(p))

**Goal:** For each output grid (train_out and test canvas), compute a **type key** `T(p)` per cell as a deterministic tuple of atoms:

From ` @docs/anchors/00_MATH_SPEC.md ` §5.2:

[
T(p) = \bigl( d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}}, r\pm c, {r\bmod m}, {c\bmod m}, \text{3×3 hash}, \text{period flags}, \text{component shape ID} \bigr).
]

Type keys are used:

* To group “same-role” cells across grids,
* To mine unary fixes and relational equalities.

This WO only **defines and computes T(p)** and assigns stable integer IDs per type. Mining comes in later WOs.

---

### 0. Anchors to read

Implementer + reviewer must re-read:

1. **` @docs/anchors/00_MATH_SPEC.md `**

   * §5.1 A–G (atoms; especially A, B, D, G feeding into T(p))
   * §5.2 “Miner rules” — formal definition of `T(p)`.
2. **` @docs/anchors/01_STAGES.md `**

   * Stage 05_laws: type keys are computed on canonical grids; mining uses **train_out only**.
3. **` @docs/anchors/02_QUANTUM_MAPPING.md `**

   * “Seeing vs doing”: T(p) is pure seeing; no test_out peeking, no paid bits.

---

### 1. Libraries to use

Everything here is simple composition of existing atoms:

```python
import numpy as np
from typing import Dict, Any, Tuple, List
```

No new external libs. We rely on:

* `compute_A_atoms` (WO-4.1): distances, r±c, mod classes.
* `compute_B_atoms` (WO-4.2): 3×3 hash.
* `compute_D_atoms` (WO-4.4): row/col periods.
* `compute_G_atoms` (WO-4.6): shape / template ids.

We do **not** implement new algorithms; just wiring.

---

### 2. API and placement

Add a new module or section:

```bash
05_laws/
  type_keys.py    # or inside atoms.py if you prefer
  step.py
```

Define:

```python
def compute_type_keys_for_grid(
    A_atoms: Dict[str, Any],
    B_atoms: Dict[str, Any],
    D_atoms: Dict[str, Any],
    G_atoms: Dict[str, Any],
) -> Dict[str, Any]:
    ...
```

Inputs:

* `A_atoms` for this grid, with at least:

  ```python
  A_atoms["d_top"], A_atoms["d_bottom"],
  A_atoms["d_left"], A_atoms["d_right"],  # (H,W) int
  A_atoms["r"], A_atoms["c"],             # (H,W) int
  A_atoms["mod_r"], A_atoms["mod_c"],     # dict m->(H,W)
  ```

* `B_atoms["hash_3x3"]` (H,W) int.

* `D_atoms["row_periods"]` (H,), `D_atoms["col_periods"]` (W,).

* `G_atoms["template_id"]` (H,W) int (shape ID per pixel; -1 for background).

Output:

```python
{
  "type_tuple": np.ndarray[(H,W), dtype=object],  # Python tuples per cell
  "type_id":    np.ndarray[(H,W), int],           # small ints 0..(K-1)
  "types":      List[Tuple],                      # list of unique type tuples in stable order
}
```

The stable order is defined deterministically (no dependence on dict iteration or hash seeds).

---

### 3. Exact semantics of T(p)

For each cell (r,c), T(p) is a tuple:

1. Distances:

   ```python
   dt = d_top[r,c]
   db = d_bottom[r,c]
   dl = d_left[r,c]
   dr = d_right[r,c]
   ```

2. Diagonal coords:

   ```python
   rp = A_atoms["r_plus_c"][r,c]
   rm = A_atoms["r_minus_c"][r,c]
   ```

3. Mod classes (grid-aware m set):

   * The set of m is defined in A-atoms:

     ```python
     ms = sorted(A_atoms["mod_r"].keys())  # same as spec: {2..min(6,max(H,W))} ∪ divisors(H) ∪ divisors(W)
     mod_r_vals = tuple(A_atoms["mod_r"][m][r,c] for m in ms)
     mod_c_vals = tuple(A_atoms["mod_c"][m][r,c] for m in ms)
     ```

4. 3×3 hash:

   ```python
   h3 = B_atoms["hash_3x3"][r,c]
   ```

5. Period flags (row/col):

   * We use the raw minimal period at row r and column c:

     ```python
     row_p = D_atoms["row_periods"][r]   # 1..W
     col_p = D_atoms["col_periods"][c]   # 1..H
     ```

   * Pack as `(row_p, col_p)` in the tuple. If you prefer, you can also keep them separate; spec just says “period flags”.

6. Component shape ID:

   * From G-atoms, we already computed `template_id` per cell:

     ```python
     shape_id = int(G_atoms["template_id"][r,c])  # -1 if not in any component
     ```

Putting it all together:

```python
T_p = (
    int(dt), int(db), int(dl), int(dr),
    int(rp), int(rm),
    mod_r_vals,
    mod_c_vals,
    int(h3),
    int(row_p), int(col_p),
    shape_id,
)
```

You can use nested tuples for mods; they are hashable.

No test_out peeking: all of these come from **handoff atoms** on this grid; for test canvas we compute atoms from the chosen size, not from any predicted colors.

---

### 4. Computing type tuples and stable type IDs

Implementation sketch in `compute_type_keys_for_grid`:

```python
def compute_type_keys_for_grid(A_atoms, B_atoms, D_atoms, G_atoms):
    H, W = A_atoms["r"].shape

    d_top    = A_atoms["d_top"]
    d_bottom = A_atoms["d_bottom"]
    d_left   = A_atoms["d_left"]
    d_right  = A_atoms["d_right"]
    r_plus_c = A_atoms["r_plus_c"]
    r_minus_c = A_atoms["r_minus_c"]
    mod_r = A_atoms["mod_r"]
    mod_c = A_atoms["mod_c"]

    hash_3x3 = B_atoms["hash_3x3"]

    row_periods = D_atoms["row_periods"]
    col_periods = D_atoms["col_periods"]

    template_id = G_atoms["template_id"]

    ms = sorted(mod_r.keys())

    type_tuple = np.empty((H,W), dtype=object)

    for r in range(H):
        for c in range(W):
            dt = int(d_top[r,c]); db = int(d_bottom[r,c])
            dl = int(d_left[r,c]); dr = int(d_right[r,c])
            rp = int(r_plus_c[r,c]); rm = int(r_minus_c[r,c])
            mod_r_vals = tuple(int(mod_r[m][r,c]) for m in ms)
            mod_c_vals = tuple(int(mod_c[m][r,c]) for m in ms)
            h3 = int(hash_3x3[r,c])
            row_p = int(row_periods[r])
            col_p = int(col_periods[c])
            shape_id = int(template_id[r,c])
            T_p = (dt, db, dl, dr,
                   rp, rm,
                   mod_r_vals, mod_c_vals,
                   h3,
                   row_p, col_p,
                   shape_id)
            type_tuple[r,c] = T_p
```

Now we need **stable** IDs:

1. Collect all unique type tuples:

   ```python
   flat = type_tuple.ravel().tolist()
   unique_types = sorted(set(flat))  # lexicographic sort of tuples
   ```

2. Make mapping:

   ```python
   type_to_id = {t: i for i, t in enumerate(unique_types)}
   ```

3. Build `type_id` array:

   ```python
   type_id = np.zeros((H,W), dtype=int)
   for r in range(H):
       for c in range(W):
           type_id[r,c] = type_to_id[type_tuple[r,c]]
   ```

This gives:

* Deterministic `types` list (sorted),
* Stable `type_id` labels across runs, unaffected by Python hash randomization.

Return:

```python
return {
  "type_tuple": type_tuple,
  "type_id": type_id,
  "types": unique_types,
}
```

---

### 5. Brainstem `run.py` changes

None.

`run.py` remains:

```python
present   = load_present(...)
canonical = canonicalize_truth(...)
scaffold  = build_scaffold(...)
size_ch   = choose_size(...)
laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
```

`05_laws/step.py` will:

* For each train_out grid:

  * Call all atom builders (A–G),
  * Call `compute_type_keys_for_grid(...)`.
* For test canvas:

  * Build atoms on the canvas geometry (no test_out),
  * Call type keyer similarly.

`run.py` stays minimal.

---

### 6. Receipts / trace

Add a helper in `05_laws/type_keys.py`:

```python
def trace_type_keys(type_keys: Dict[str, Any]) -> None:
    type_id = type_keys["type_id"]
    types = type_keys["types"]
    print("[T] num_types:", len(types))
    # Show 5 smallest type tuples
    for i, t in enumerate(types[:5]):
        print(f"[T] type {i}: {t}")
    # Simple histogram of type usage
    unique, counts = np.unique(type_id, return_counts=True)
    print("[T] type_id histogram (first 10):", list(zip(unique[:10], counts[:10])))
```

`05_laws/step.py` can call this for the first train_out when `trace=True`.

---

### 7. Reviewer instructions

#### 7.1 Using `run.py` for sanity

Pick 2–3 tasks:

* Task A: uniform grid (all same color; simple geometry).
* Task B: periodic tiling grid (like 2×2 pattern repeated).
* Task C: task `00576224` (where we later expect 4 equivalence classes of size 9).

Run:

```bash
python run.py --task tasks/<task_id>.json --trace
```

Ensure `mine_laws` invokes `compute_type_keys_for_grid` and `trace_type_keys` at least for the first train_out.

Checks:

1. **Uniform grid**

   * Distances and r±c vary; mods too; shape_id is all one component.
   * You should see a small number of distinct types (e.g. corners, edges, interior).
   * Running twice gives **same** `num_types` and same first few types.

2. **Periodic tiling**

   * Cells that are in the same relative position inside each tile should have the **same type_id**.
   * Visual check: pick one cell in each repeated block; type_id values must match.

3. **Cross-run stability**

   * Run the same task twice in different Python processes.
   * `num_types` and the sorted list `types` should be identical (stable hashing/ordering).

Any mismatch is an implementation bug, not a spec issue.

#### 7.2 Identifying legit unsatisfiable vs bug

For WO-5.1 itself, “unsatisfiable” doesn’t happen: it’s just computing feature tuples. If:

* Shapes mismatch,
* Type counts change across runs, or
* Types obviously collapse cells that are geometrically different,

then the type keyer is not respecting the spec; that’s a bug.

---
