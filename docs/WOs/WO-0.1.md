## WO-0.1 — Brainstem `run.py` (40–60 LOC of real code)

### 0. Anchors to read before coding

Before implementing this work order, the implementer **must** re-read:

* ` @docs/anchors/01_STAGES.md `
  To recall the seven-stage flow and the conceptual contract of each stage.

* ` @docs/anchors/00_MATH_SPEC.md `
  Sections:

  * §0 “Ground truth (TOE)”
  * §1 “Representing the task”
  * §2 “Stage A — Canonical labeling (awareness & gauge)” (just to know what `truth` is supposed to do later, not to implement it here).

* ` @docs/anchors/02_QUANTUM_MAPPING.md `
  Only the high-level mapping of “present → truth → scaffold → … → fixed_point” to consciousness. No math to implement, just to ensure ordering matches.

**Important:** WO-0.1 does **not** implement any math; it just wires the stages so later WOs can drop in their implementations without touching `run.py`.

---

### 1. Responsibilities of `run.py` in this WO

**What it must do now:**

* Provide a **single, minimal entrypoint** that:

  * Loads the ARC challenge **task JSON** (via a helper in `01_present` later),
  * Calls stages in order:
    `present → truth → scaffold → size_choice → laws → minimal_act → fixed_point`
    by importing their `step.py` functions,
  * Handles CLI flags and printing the final grid (or a placeholder until other stages are implemented).

**What it must NOT do:**

* No canonicalization logic here, no atoms, no ILP, no BFS.
  All math/intelligence lives inside the stage modules.
* No heavy “god function” with big branches; just:

  * Parse args
  * Call `run_task(task_id, data_path, trace)`
  * That function calls each stage’s `step.*` function
  * Print or return the final grid.

---

### 2. Libraries to use (only mature, well-documented ones)

For this WO, we only need:

* `argparse` (stdlib)
  For CLI parsing. Mature, standard.

* `json` (stdlib)
  Only for **very light** sanity checks or direct task JSON loading **if** a raw task file path is ever supported. The *real* parsing of ARC JSON will be in `01_present.step`.

* `pathlib` (stdlib)
  For file paths.

* `logging` (stdlib)
  For minimal debug logs (e.g., what stage we’re in) when `--trace` is set.

We **do not** introduce `numpy`, `networkx`, `igraph`, or ILP libs in this WO. Those are used in later WOs.

---

### 3. Stage function contracts (to wire, not implement)

`run.py` should assume the following functions exist (even if they currently just raise `NotImplementedError`):

```python
# 01_present/step.py
def load(task_bundle, trace: bool = False):
    """
    Input: task_bundle, a dict with at least {"task_id": str, "raw_task": dict}
    Output: present object (opaque to run.py, passed through to next stage)
    """

# 02_truth/step.py
def canonicalize(present, trace: bool = False):
    """
    Input: present
    Output: canonical object
    """

# 03_scaffold/step.py
def build(canonical, trace: bool = False):
    """
    Input: canonical
    Output: scaffold_train_out (frame, distances, inner region, etc.)
    """

# 04_size_choice/step.py
def choose(canonical, scaffold_train_out, trace: bool = False):
    """
    Input: canonical, scaffold_train_out
    Output: out_size (H_out, W_out)
    """

# 05_laws/step.py
def mine(canonical, scaffold_train_out, out_size, trace: bool = False):
    """
    Input: canonical, scaffold_train_out, out_size
    Output: invariants object (fixed, forbids, equal_pairs, etc.)
    """

# 06_minimal_act/step.py
def solve(canonical, invariants, out_size, trace: bool = False):
    """
    Input: canonical, invariants, out_size
    Output: solution object with at least solution.out_grid (np.ndarray or list-of-lists)
    """

# 07_fixed_point/step.py
def check(canonical, solution, trace: bool = False):
    """
    Input: canonical, solution
    Output: None or raises if N^2 != N
    """
```

These signatures reflect the anchor docs (`01_STAGES.md` + `00_MATH_SPEC.md`) and are future-proof.

---

### 4. CLI design (non-future-dependent, grounded in data file)

We have a **single training file**:

* `data/arc-agi_training_challenges.json`
  with structure like:

  ```json
  {
    "00576224": {
      "train": [ { "input": ..., "output": ... }, ... ],
      "test": [ { "input": ... }, ... ]
    },
    ...
  }
  ```

For WO-0.1, design CLI as:

```bash
python run.py --task-id 00576224 \
              --data data/arc-agi_training_challenges.json \
              [--trace]
```

**Flags:**

* `--task-id` (required)
  ARC task ID string, e.g. `"00576224"`.

* `--data` (optional, default: `"data/arc-agi_training_challenges.json"`)
  Path to the challenges JSON. Implementer should **not** parse that deeply in `run.py`; just pass the loaded `raw_task` into `01_present.step.load`.

* `--trace` (optional boolean flag)
  If set, stages receive `trace=True` and may emit receipts to stdout / logs or to files. For WO-0.1, you don’t need actual receipts yet, just propagate the flag.

---

### 5. Execution flow for `run.py` (skeleton)

**Recommended structure:**

```python
# run.py
import argparse
import json
from pathlib import Path
import logging

from 01_present.step     import load as load_present
from 02_truth.step       import canonicalize
from 03_scaffold.step    import build as build_scaffold
from 04_size_choice.step import choose as choose_size
from 05_laws.step        import mine as mine_laws
from 06_minimal_act.step import solve as minimal_act
from 07_fixed_point.step import check as fixed_point_check

def run_task(task_id: str, data_path: str, trace: bool = False):
    # 1. Read raw task from the big JSON file
    data = json.loads(Path(data_path).read_text())
    raw_task = data[task_id]  # let this raise KeyError if missing: it's a legit error

    task_bundle = {
        "task_id": task_id,
        "raw_task": raw_task,
    }

    if trace:
        logging.info(f"[present] loading task {task_id} from {data_path}")

    present   = load_present(task_bundle, trace=trace)
    canonical = canonicalize(present, trace=trace)
    scaffold  = build_scaffold(canonical, trace=trace)
    out_size  = choose_size(canonical, scaffold, trace=trace)
    invariants= mine_laws(canonical, scaffold, out_size, trace=trace)
    solution  = minimal_act(canonical, invariants, out_size, trace=trace)
    fixed_point_check(canonical, solution, trace=trace)

    # Expect solution to have .out_grid; this will be a numpy array or list-of-lists
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, help="ARC task ID, e.g., 00576224")
    parser.add_argument("--data", default="data/arc-agi_training_challenges.json",
                        help="Path to arc-agi_training_challenges.json")
    parser.add_argument("--trace", action="store_true", help="Enable debug/receipts")

    args = parser.parse_args()

    if args.trace:
        logging.basicConfig(level=logging.INFO)

    solution = run_task(args.task_id, args.data, trace=args.trace)

    # For now, just print placeholder if earlier stages are unimplemented
    try:
        out_grid = solution.out_grid  # attribute or key; to be aligned later
        print(out_grid)
    except Exception as e:
        # Early WOs: 01_present will raise NotImplementedError.
        # Let that bubble, or print a clear message.
        logging.error("Pipeline did not return a grid (expected in later milestones).")
        raise

if __name__ == "__main__":
    main()
```

**Important:**

* `run.py` has **no algorithmic logic**, just sequencing and IO.
* It assumes each stage’s `step.py` will, for now, raise `NotImplementedError`, which is fine for WO-0.1.

---

### 6. Receipts for WO-0.1

Since no math is implemented yet, the only meaningful “receipts” are:

* Which **stage** failed and at what point,
* That the correct **task_id** and **data path** were used.

Minimal receipts behavior:

* If `--trace` is set:

  * Log a line before each stage call:

    * `[present] loading task ...`
    * `[truth] canonicalizing`
    * `[scaffold] building`
    * etc.
  * When a stage raises `NotImplementedError`, the log shows **exactly which stage** failed. That’s the “gap”.

No extra JSON dump is needed at this WO.

---

### 7. Reviewer instructions for WO-0.1

**Goal for reviewer:** ensure `run.py` wiring matches math spec + stage architecture, and no extra logic sneaks in.

1. **Anchor cross-check**

   * Verify that `run.py` calls the seven stages in the **correct order**:

     * `present → truth → scaffold → size_choice → laws → minimal_act → fixed_point`
   * Compare sequence with:

     * `01_STAGES.md` high-level stage list
     * `02_QUANTUM_MAPPING.md` “present → truth → scaffold → size_choice → laws → minimal_act → fixed_point” flow.

2. **Run on 2–3 real ARC tasks**

   Use the shared `data/arc-agi_training_challenges.json` and run e.g.:

   ```bash
   python run.py --task-id 00576224 --data data/arc-agi_training_challenges.json --trace
   python run.py --task-id 1caeab9d --data data/arc-agi_training_challenges.json --trace
   python run.py --task-id 4c4377c9 --data data/arc-agi_training_challenges.json --trace
   ```

   **What to expect at this WO:**

   * Since `01_present.step.load` is not implemented yet, the pipeline should fail *in that stage* with a `NotImplementedError` (or a placeholder exception) and show:

     * Logs up to `[present] loading task ...`,
     * Nothing from later stages.

   * If run.py throws an error **before** entering `present.load` (e.g., JSON parsing, argument issues), that’s a **WO-0.1 implementation bug**, not an unsatisfiable task.

   * If run.py reaches `truth` or later without `present` implemented, that means the stage contracts are not correctly enforced and is also a bug.

3. **How to identify legit vs invalid behavior**

   * **Legit WO-0.1 gap**:

     * “`NotImplementedError` in 01_present.step.load` for any task”
       This is expected until Milestone 1 is implemented.

   * **Invalid/bug**:

     * Crash before `present.load` (e.g., `KeyError` on task_id because `--task-id` not used properly).
     * Running **without** calling `present` at all.
     * Any math logic hardcoded in `run.py` (e.g., reading and interpreting grids with numpy here).

4. **Math/Implementation 100% match check**

   * For WO-0.1, math spec only demands:

     * Stage order,
     * That a “task bundle” representing `(train_in, train_out, test_in)` is prepared and passed to `present` (the actual parsing will be implemented in WO-1.1),
     * No decisions about laws, frames, atoms, or ledger are made in `run.py`.

   Reviewer should confirm:

   * `run.py` does **not** perform canonicalization, atom derivation, invariant mining, or any optimization.
   * `run.py` just passes `task_bundle` and `trace` down the chain.

5. **Note about real ARC results**

   * For this WO, we **do not** compare `out_grid` to ground truth yet; the math engine isn’t implemented.
   * Another agent will provide golden outputs later; the reviewer only ensures **wiring and control flow** now.
   * When those goldens exist, re-running the above commands should eventually print final grids that will be compared to provided ground truths—at that point, no changes to `run.py` should be needed.

---

### 8. Optimization / CPU concerns

Not applicable at this WO:

* `run.py` does not perform heavy compute.
* No need for optimization hacks or simplified implementations.
* All performance-sensitive work is in later WOs (canonicalization, BFS, atoms, ILP). Those WOs must respect the “no shortcuts unless truly necessary” rule; this WO is just plumbing.

---

### 9. Checklist: how WO-0.1 meets your meta-requirements

1. **Grounded in anchors?**

   * Stage order & semantics from `01_STAGES.md`, `02_QUANTUM_MAPPING.md`.
   * Task representation semantics from `00_MATH_SPEC.md` §1.
   * No extra math invented.

2. **Anchor docs referenced?**

   * Explicitly: 00, 01, 02 (see section 0).

3. **Mature libs only; no algorithm re-implementation?**

   * Uses only `argparse`, `json`, `pathlib`, `logging` (Python stdlib).
   * No custom algo code beyond wiring.

4. **run.py minimal, no god function, future integration easy?**

   * `run_task()` calls exactly one function per stage.
   * `main()` only parses args and calls `run_task()`.
   * When later WOs are done, `run.py` should not change.

5. **Receipts defined and usable for debugging?**

   * `--trace` enables logging per stage entry.
   * Reviewer can see exactly where execution stops (which stage is unimplemented).

6. **Reviewer instructions for 2–3 tasks & gap detection?**

   * Provided example commands and expectations (NotImplementedError in `present`).
   * Clear distinction between legit WO gap and implementation bug.

7. **No premature optimization / simplification?**

   * No algorithmic choices here, so no risk.
   * All CPU-time tradeoffs are deferred to later WOs, where we will explicitly reason about them.

---

If you like this format, next we can do the same level of expansion for **WO-1.1 (Present: JSON loader)**, including exactly how to reuse `json`, how to map the ARC schema to a `task_bundle` that matches the spec’s notion of `train_in/train_out/test_in`.
