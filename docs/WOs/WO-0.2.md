## WO-0.2 — Stage folders + minimal stubs (40–60 LOC per file)

### 0. Anchors to read **before** coding

For this WO, the implementer must re-read:

* `docs/anchors/01_STAGES.md`
  To know what each stage *conceptually* does and what its inputs/outputs should be.

* `docs/anchors/00_MATH_SPEC.md`

  * §1 “Representing the task” (what `train_in`, `train_out`, `test_in` mean)
  * Titles of §§2–7 to see the **stage names** and their roles:

    * Stage A — Canonical labeling (awareness & gauge) → `truth`
    * Stage S0 — Output canvas size → `size_choice`
    * Stage F — Frame & distances → `scaffold`
    * Stage N — Invariants → `laws`
    * Stage D — Ledger minimization → `minimal_act`
    * Idempotence → `fixed_point`

* `docs/anchors/02_QUANTUM_MAPPING.md`
  Just to ensure the stage order and names you wire match the consciousness mapping.

**Important:** WO-0.2 is **only about structure**, not implementing any math. The contracts and names must match anchors exactly so later WOs can drop in real logic.

---

### 1. Responsibilities of WO-0.2

**Goal:**
Create the **seven stage folders** with `step.py` inside, each exposing exactly one public function with the same name the brainstem (`run.py`) expects:

* `01_present/step.py` → `load`
* `02_truth/step.py` → `canonicalize`
* `03_scaffold/step.py` → `build`
* `04_size_choice/step.py` → `choose`
* `05_laws/step.py` → `mine`
* `06_minimal_act/step.py` → `solve`
* `07_fixed_point/step.py` → `check`

All stubs:

* Have correct signatures `(… , trace: bool = False)`
* Have docstrings that reference the relevant anchor sections
* For now: they **only** log and then `raise NotImplementedError`.

This ensures:

* `run.py` imports succeed,
* Stage order is fixed,
* There is no “god” logic in `run.py`.

---

### 2. Libraries to use here

For this WO, only standard library is needed:

* `typing` (optional)
  For type hints like `dict`, `Any`. This is mature, avoids reinventing types.

* `logging` (optional inside stubs)
  To log that a stub was called when `trace=True`.

No other libs (numpy, igraph, networkx, ILP, etc.) are used here. Those belong to later WOs.

---

### 3. Stub content per stage

Each `step.py` should be **small** (≤20 LOC) and strictly match the contracts expected by `run.py`.

#### 3.1 `01_present/step.py`

Grounded in:

* `00_MATH_SPEC.md` §1 (task representation)
* `01_STAGES.md` “present”
* `02_QUANTUM_MAPPING.md` “present = load all content into awareness”

```python
# 01_present/step.py
from typing import Any, Dict
import logging

def load(task_bundle: Dict[str, Any], trace: bool = False) -> Any:
    """
    Stage: present (awareness)
    Anchor:
      - 01_STAGES.md: present
      - 00_MATH_SPEC.md §1: Representing the task
      - 02_QUANTUM_MAPPING.md: 'present' = load all grids into awareness

    Input:
      task_bundle: {
        "task_id": str,
        "raw_task": dict from arc-agi_training_challenges.json
      }
      trace: enable debug logging if True.

    Output:
      A 'present' object (opaque to run.py) that future stages will refine.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info(f"[present] load() called for task_id={task_bundle.get('task_id')}")
    raise NotImplementedError("01_present.load is not implemented yet.")
```

#### 3.2 `02_truth/step.py`

Grounded in:

* `00_MATH_SPEC.md` §2 Stage A — Canonical labeling
* `01_STAGES.md` “truth”
* `02_QUANTUM_MAPPING.md` “truth = Π / canonical gauge”

```python
# 02_truth/step.py
from typing import Any
import logging

def canonicalize(present: Any, trace: bool = False) -> Any:
    """
    Stage: truth (Π / canonical gauge)
    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2: Stage A — Canonical labeling
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π, kill minted differences

    Input:
      present: object returned by 01_present.load
      trace: enable debug logging if True.

    Output:
      A 'canonical' object (opaque to run.py) with grids in canonical gauge.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[truth] canonicalize() called")
    raise NotImplementedError("02_truth.canonicalize is not implemented yet.")
```

#### 3.3 `03_scaffold/step.py`

Grounded in:

* `00_MATH_SPEC.md` §4 Stage F — Frame & distances
* `01_STAGES.md` “scaffold”
* `02_QUANTUM_MAPPING.md` “WHERE”

```python
# 03_scaffold/step.py
from typing import Any
import logging

def build(canonical: Any, trace: bool = False) -> Any:
    """
    Stage: scaffold (WHERE)
    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4: Stage F — Frame & distances
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold

    Input:
      canonical: object from 02_truth.canonicalize
      trace: enable debug logging if True.

    Output:
      Scaffold object containing frame, distance fields, inner region (train_out-only).
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[scaffold] build() called")
    raise NotImplementedError("03_scaffold.build is not implemented yet.")
```

#### 3.4 `04_size_choice/step.py`

Grounded in:

* `00_MATH_SPEC.md` §3 Stage S0 — Output canvas size
* `01_STAGES.md` “size_choice”
* `02_QUANTUM_MAPPING.md` “size_choice”

```python
# 04_size_choice/step.py
from typing import Any, Tuple
import logging

def choose(canonical: Any, scaffold: Any, trace: bool = False) -> Tuple[int, int]:
    """
    Stage: size_choice (S0)
    Anchor:
      - 01_STAGES.md: size_choice
      - 00_MATH_SPEC.md §3: Stage S0 — Output canvas size
      - 02_QUANTUM_MAPPING.md: choose output shape consistent with laws

    Input:
      canonical: object from 02_truth.canonicalize
      scaffold: object from 03_scaffold.build (train_out-only scaffold)
      trace: enable debug logging if True.

    Output:
      (H_out, W_out) for the test canvas.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[size_choice] choose() called")
    raise NotImplementedError("04_size_choice.choose is not implemented yet.")
```

#### 3.5 `05_laws/step.py`

Grounded in:

* `00_MATH_SPEC.md` §5 Stage N — Invariants as linear constraints
* `01_STAGES.md` “laws”
* `02_QUANTUM_MAPPING.md` “WHAT”

```python
# 05_laws/step.py
from typing import Any
import logging

def mine(canonical: Any, scaffold: Any, out_size: Any, trace: bool = False) -> Any:
    """
    Stage: laws (N)
    Anchor:
      - 01_STAGES.md: laws
      - 00_MATH_SPEC.md §5: Stage N — Invariants as linear constraints
      - 02_QUANTUM_MAPPING.md: WHAT = law nucleus over scaffold

    Input:
      canonical: from 02_truth.canonicalize
      scaffold: from 03_scaffold.build
      out_size: (H_out, W_out) from 04_size_choice.choose
      trace: enable debug logging if True.

    Output:
      Invariants object encoding fixes, equalities, forbids, etc.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[laws] mine() called")
    raise NotImplementedError("05_laws.mine is not implemented yet.")
```

#### 3.6 `06_minimal_act/step.py`

Grounded in:

* `00_MATH_SPEC.md` §6 Stage D — Ledger minimization
* `01_STAGES.md` “minimal_act”
* `02_QUANTUM_MAPPING.md` “DO”

```python
# 06_minimal_act/step.py
from typing import Any
import logging

class Solution:
    """
    Placeholder for final solution.
    Later WOs will extend this to include out_grid and receipts.
    """
    def __init__(self, out_grid=None):
        self.out_grid = out_grid

def solve(canonical: Any, invariants: Any, out_size: Any, trace: bool = False) -> Solution:
    """
    Stage: minimal_act (D)
    Anchor:
      - 01_STAGES.md: minimal_act
      - 00_MATH_SPEC.md §6: Stage D — Ledger minimization
      - 02_QUANTUM_MAPPING.md: DO = paid step, minimize ledger

    Input:
      canonical: from 02_truth.canonicalize
      invariants: from 05_laws.mine
      out_size: (H_out, W_out) from 04_size_choice.choose
      trace: enable debug logging if True.

    Output:
      Solution object containing at least solution.out_grid (grid for test_out).
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[minimal_act] solve() called")
    raise NotImplementedError("06_minimal_act.solve is not implemented yet.")
```

#### 3.7 `07_fixed_point/step.py`

Grounded in:

* `00_MATH_SPEC.md` §7 / idempotence discussion (N² = N)
* `01_STAGES.md` “fixed_point”
* `02_QUANTUM_MAPPING.md` “fixed point / idempotence”

```python
# 07_fixed_point/step.py
from typing import Any
import logging

def check(canonical: Any, solution: Any, trace: bool = False) -> None:
    """
    Stage: fixed_point (N² = N)
    Anchor:
      - 01_STAGES.md: fixed_point
      - 00_MATH_SPEC.md §7: Idempotence (N² = N)
      - 02_QUANTUM_MAPPING.md: re-see; output must be stable

    Input:
      canonical: from 02_truth.canonicalize
      solution: from 06_minimal_act.solve (must contain out_grid when implemented)
      trace: enable debug logging if True.

    Output:
      None. In final implementation, this will re-run pipeline with test pair added
      and assert stability. For now, it is not implemented and always raises.
    """
    if trace:
        logging.info("[fixed_point] check() called")
    raise NotImplementedError("07_fixed_point.check is not implemented yet.")
```

---

### 4. `run.py` changes for this WO

With these stubs in place, `run.py` from WO-0.1 **does not need functional changes**, but we should:

* Confirm the import paths match the folders:

  ```python
  from 01_present.step     import load as load_present
  from 02_truth.step       import canonicalize
  from 03_scaffold.step    import build as build_scaffold
  from 04_size_choice.step import choose as choose_size
  from 05_laws.step        import mine as mine_laws
  from 06_minimal_act.step import solve as minimal_act
  from 07_fixed_point.step import check as fixed_point_check
  ```
* Ensure we **do not** add any new logic to `run.py`. It remains minimal, just calling these stub functions.

When you run:

```bash
python run.py --task-id 00576224 --data data/arc-agi_training_challenges.json --trace
```

You should now see logs:

* `[present] load() called for task_id=00576224`
  then a `NotImplementedError` from `01_present.load`.

That’s the expected behavior for this WO.

---

### 5. Receipts in this WO

Receipts = minimal logging per stage. For WO-0.2 we only need:

* On `--trace`, each stub logs a single line when called.
  The “receipt” is: *which stage we got to before failing*.

Reviewer uses logs to see:

* That stages are called in the right order.
* That the first failure is exactly where we haven’t implemented yet (present), not earlier.

No JSON receipts yet; those come in later WOs.

---

### 6. Reviewer instructions (with real ARC tasks)

**Commands:**

```bash
python run.py --task-id 00576224 --data data/arc-agi_training_challenges.json --trace
python run.py --task-id 1caeab9d --data data/arc-agi_training_challenges.json --trace
python run.py --task-id 4c4377c9 --data data/arc-agi_training_challenges.json --trace
```

**What to expect:**

* The program will:

  * Parse CLI args,
  * Read the JSON file,
  * Call `01_present.load` with `task_bundle`,
  * Log `[present] load() called …`,
  * Then raise `NotImplementedError("01_present.load is not implemented yet.")`.

No other stage should log or be called yet; `present` is first.

**How to detect legit gaps vs bugs:**

* **Legit WO-0.2 “gap”:**

  * `NotImplementedError` from one of the stage stubs (starting with `present`).
  * Logs show correct order up to that point.

* **Bug:**

  * Import errors (ModuleNotFoundError on any `01_present.step`, etc.).
  * Mis-ordered calls (e.g., calling `truth` before `present`).
  * `run.py` contains any non-trivial logic that belongs to later stages.
  * Crashes before calling `present.load` (e.g., wrong keys in `task_bundle`).

**Math/implementation consistency:**

* At this WO, the only thing the math spec demands is:

  * The **stage names and order** are fixed and consistent with anchors.
  * `run.py` does not “pre-solve” anything.
* Reviewer must confirm that no stage does work outside its spec; for now, every stage stub does **nothing** but log and raise.

**Note on real ARC results:**

* We do **not** compare final grids to ground truth yet; the law/atoms/ILP are not implemented.
* Another agent will later provide goldens; at that point, `run.py` must stay unchanged and stage implementations will evolve.

---

### 7. Optimization / CPU considerations

Not applicable here:

* No heavy computation in stubs.
* No need for any optimization hacks.
* All compute-heavy stuff (canonical labeling, BFS, ILP) will come in later WOs; for those WOs we will call out any trade-offs explicitly.

---