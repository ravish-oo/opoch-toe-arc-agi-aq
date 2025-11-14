### WO‑4.5 Input feature mirror

We now treat F24 exactly as spec says:

* A–E atoms **are** defined for inputs: same formulas, applied to input grids.
* Guardrail is about **usage** (evaluation only), not about which atoms exist.

#### 0. Anchors

Read:

* ` @docs/anchors/00_MATH_SPEC.md ` §5.1 A–F (especially F)
* ` @docs/anchors/01_STAGES.md ` Stage 05_laws
* ` @docs/anchors/02_QUANTUM_MAPPING.md ` (no input-driven laws)

#### 1. Libraries

Same stack:

```python
import numpy as np
from scipy import ndimage
from skimage.measure import label as sk_label, regionprops
```

Re-use existing atom functions:

* `compute_A_atoms(grid, scaffold_info)`
* `compute_B_atoms(grid)`
* `compute_C_atoms(grid, scaffold_info)`
* `compute_D_atoms(grid)`
* `compute_E_atoms_for_grid(grid, C_atoms)`

And re-use the **same scaffold builder** used in `03_scaffold.step` (frame + distances + inner) but applied to an arbitrary grid:

* Either:

  * Factor out a helper in `utils/scaffold_utils.py` that takes any grid and returns `d_top/bottom/left/right`, `frame_mask`, `inner`, etc.,
  * Then `03_scaffold` and F24 both call that helper.

No new algorithms.

#### 2. API

In `05_laws/atoms.py`:

```python
_input_atoms_cache: dict[int, dict] = {}
```

Define:

```python
from utils.scaffold_utils import build_scaffold_for_grid
from 05_laws.atoms import (
    compute_A_atoms,
    compute_B_atoms,
    compute_C_atoms,
    compute_D_atoms,
    compute_E_atoms_for_grid,
)

def get_input_atoms_for_test(
    canonical: dict,
    test_idx: int = 0,
) -> dict:
    """
    F24: mirror A–E atoms on inputs for evaluation of laws.

    IMPORTANT:
    - Never called during mining from train_out.
    - Only used to plug input-dependent values into laws already mined.
    """
    if test_idx in _input_atoms_cache:
        return _input_atoms_cache[test_idx]

    grid = canonical["test_in"][test_idx]  # np.ndarray(H,W)

    # Build scaffold on input grid using the SAME logic as Stage F (frame+distances+inner)
    scaffold_info = build_scaffold_for_grid(grid)

    # Now reuse the same atom functions as for outputs
    A = compute_A_atoms(H=grid.shape[0], W=grid.shape[1], scaffold_info=scaffold_info)
    B = compute_B_atoms(grid)
    C = compute_C_atoms(grid, scaffold_info=scaffold_info)
    D = compute_D_atoms(grid)
    E = compute_E_atoms_for_grid(grid, C_atoms=C)

    atoms = {"A": A, "B": B, "C": C, "D": D, "E": E}
    _input_atoms_cache[test_idx] = atoms
    return atoms
```

If `build_scaffold_for_grid` doesn’t exist yet:

* Create it by extracting the core logic of `03_scaffold.step.build` into a shared helper in `utils/` that works on any single grid.
* **Do not change** its math; just factor it.

#### 3. Guardrail in laws

In `05_laws/step.py`:

* Mining (from train_out):

  * Must **never** call `get_input_atoms_for_test`.
* Evaluation:

  * Only call F24 to supply values for already-mined laws, e.g.:

    ```python
    if law_needs_input_feature:
        input_atoms = get_input_atoms_for_test(canonical, test_idx=0)
        most_freq_input = input_atoms["E"]["most_frequent"][0]
        # use this in constraints
    ```

Any call to F24 in the mining path is a hard bug.

#### 4. run.py

Unchanged:

```python
present   = load_present(...)
canonical = canonicalize_truth(...)
scaffold  = build_scaffold(...)
size_ch   = choose_size(...)
laws      = mine_laws(canonical, scaffold, size_ch, trace=trace)
```

F24 is internal to `mine_laws`.

#### 5. Receipts / trace

In `05_laws/step.py`, when you actually use F24:

```python
if trace and used_input_features:
    input_atoms = get_input_atoms_for_test(canonical, test_idx=0)
    print("[F24] used input atoms for test_in[0]:")
    print("      palette:", input_atoms["E"]["palette"])
    print("      most_frequent:", input_atoms["E"]["most_frequent"])
    true_tilings = [k for k,v in input_atoms["D"]["tiling_flags"].items() if v]
    print("      tilings_true:", true_tilings)
```

This lets reviewer see exactly when and what input features were read.

#### 6. Reviewer instructions

* Inject a temporary test law that uses an input feature, e.g.:

  ```python
  # after mining from train_out
  input_atoms = get_input_atoms_for_test(canonical)
  bg_input = input_atoms["E"]["most_frequent"][0]
  # enforce: background color in test_out = bg_input
  ```

* Run `run.py --trace` and confirm:

  * `[F24]` trace appears **after** mining, never during.
  * A/B/C/D/E atoms look consistent on the input grid (same shapes, same semantics as on outputs).

Any F24 usage in mining = spec violation, not acceptable.

---