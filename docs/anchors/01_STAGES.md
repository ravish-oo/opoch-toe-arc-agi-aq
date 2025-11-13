## ARC Cognition — consciousness-first solver

This repo implements a consciousness-mapped ARC solver. The code mirrors the mental flow:

**present → truth → scaffold → size_choice → laws → minimal_act → fixed_point**

Each stage is a tiny, self-contained folder with one `step.py`. The top-level `run.py` just wires stages in order and returns the final grid.

### 0. How it works (15-line tour)

1. **present**: load all train_in, train_out, and test_in into awareness.
2. **truth**: apply Π (graph canonical labeling) to kill minted differences; get a shared gauge per grid.
3. **scaffold**: on **train_out** only, find the stable canvas geometry (frame) and compute distance fields; define the inner region.
4. **size_choice**: learn the input→output size map from training pairs; screen test size using only scaffold facts from train_out.
5. **laws**: derive atom types (roles) per cell from scaffold distances and local neighborhoods; promote “always true in train_out” facts into linear constraints for the test grid.
6. **minimal_act**: do one paid step: solve a 0–1 ILP that minimizes interface cost subject to those constraints; decode to the test output.
7. **fixed_point** (optional): re-add (test_in, test_out), re-run; output must be identical (idempotent).

### 1. Repo layout

```
run.py
01_present/       step.py
02_truth/         step.py
03_scaffold/      step.py  atoms.py
04_size_choice/   step.py  screens.py
05_laws/          step.py  atoms.py  relations.py
06_minimal_act/   step.py  ilp_backend.py
07_fixed_point/   step.py
utils/            graph.py  (tiny reused helpers only)
README.md
```

Folder names = consciousness stages. Function names inside are mechanical (`load`, `canonicalize`, `build`, `choose`, `mine`, `solve`, `check`).

### 2. Dependencies (mature, off-the-shelf)

* `python-igraph` (canonical labeling / colored graphs; fast and simple)
* `numpy`
* `networkx` (grid graph, BFS/DFS; alternatively in utils/graph.py)
* `scipy` (optional: morphology / labeling if you prefer)
* `ortools` (CP-SAT 0–1 ILP; or `pulp` with CBC if you insist on MILP)
* `tqdm` (optional progress)

We reuse these; no new algorithms are re-implemented.

### 3. Entry point

```bash
python run.py --task path/to/task.json
```

`run.py` returns the final `test_out` grid (HxW ints 0..9). Everything else is internal.

### 4. Stage contracts (minimal)

* `01_present.step.load(task_bundle) -> present`
* `02_truth.step.canonicalize(present) -> canonical`
* `03_scaffold.step.build(canonical) -> scaffold_train_out`
  (frame mask(s), distance fields, inner mask(s), learned small thickness/period facts)
* `04_size_choice.step.choose(canonical, scaffold_train_out) -> (H_out, W_out)`
* `05_laws.step.mine(canon, scaffold_train_out, (H_out, W_out)) -> invariants`
  (`fixed`, `forbids`, `equal_pairs`)
* `06_minimal_act.step.solve(canon, invariants, (H_out, W_out)) -> solution`
  (`out_grid`, `objective_value`)
* `07_fixed_point.step.check(canon, solution) -> None` (assert or return)

### 5. Receipts

Off by default. Any stage can accept `trace=True` to emit debug artifacts (JSON). Flip on only when diagnosing a mismatch.

---

## Stagewise testing without CI (ground-truth “goldens”)

You want “solve by hand” expectations per stage for a few real tasks. Do exactly this:

### A. Choose 5 representative ARC patterns

1. **Constant frame + midline** (border ring + center bar)
2. **Stripe periodicity** (row/col mod classes)
3. **Diagonal mirror** (r±c classes)
4. **Component translate/copy** (per-component rigid move)
5. **Size-change tiling/blow-up** (input→output size mapping)

Pick actual ARC tasks that match these; write down the true `test_out` and your expected intermediates.

### B. Capture “goldens” per stage

For each task create `tests/goldens/<task_id>/` containing:

```
present.json          # normalized grids (train_in/out, test_in)
truth.json            # canonical row/col orders per grid
scaffold.json         # frame mask(s), inner masks, d_top/bottom/left/right stats
size_choice.json      # chosen (H_out, W_out) + rejected candidates
laws.json             # fixed {cell->k}, equal_pairs list size, forbids count
minimal_act.json      # ILP objective value, solution checksum (hash)
final_grid.txt        # expected decoded HxW ints
```

### C. One test driver

`tests/run_goldens.py`:

* runs each stage in order with `trace=True`
* dumps current artifacts to a temp dir
* diffs against the golden files (byte or JSON-norm diff)
* prints a one-screen pass/fail per stage

No CI. Just one script you run locally.

### D. How to produce goldens quickly

* For **present/truth**, outputs are deterministic from inputs.
* For **scaffold**, your hand expectation is just: frame thickness values, inner rectangle bounds, and a quick checksum over the distance fields (sum/min/max).
* For **size_choice**, list the candidates and the picked pair.
* For **laws**, record counts: how many fixed, how many forbids, how many equality ties; plus 10 sample entries.
* For **minimal_act**, record the final objective value and a hash of the 0–1 matrix; and dump `final_grid.txt`.

This makes mismatches obvious without giant blobs.

---

## Invariant “atoms” (fixed list)

They’re **derivations**, not detectors:

* Distances: `d_top, d_bottom, d_left, d_right` (from frame or border via BFS)
* Parity / midlines: booleans `midrow = (d_top==d_bottom)`, `midcol = ...`
* Mod classes: `r % m, c % m` for m in {2,3,4}
* Diagonals: `r-c`, `r+c` (bounded range)
* Neighbor counts: N4/N8 counts per color
* 3×3 neighborhood hash (base-11 packing)
* Component id per color (via BFS/label)

Stage **laws** mines invariants by promoting “always true across train_out” facts into:

* `fixed`: set x[p,k]=1
* `equal_pairs`: tie x[p,*] == x[q,*] for relational pairs (bounded offsets, same diag/class/component)
* `forbids` (optional): x[p,k]=0 where safe

What’s unspecified is finished by **minimal_act** (interface-minimizing ILP).

---

## Libraries we use concretely

* **Canonical labeling**: `igraph.Graph.canonical_permutation(colors=...)`
* **Distances & components**: `networkx` BFS / `scipy.ndimage.label` (either is fine)
* **ILP**: OR-Tools CP-SAT: BoolVar `x[p,k]`; objective = sum of BoolVars `diff[p,q,k]` with `diff == XOR(x[p,k], x[q,k])`

---

## Code size & effort (realistic)

* Orchestrator + stage stubs: ~60–100 LOC
* present + truth (canonicalization glue): ~150–250 LOC
* scaffold (frame, BFS distances, inner): ~200–300 LOC
* size_choice (candidates + screens): ~150–220 LOC
* laws (atoms + miner): ~300–450 LOC
* minimal_act (ILP build/solve/decode): ~200–300 LOC
* fixed_point (optional): ~40–80 LOC
* test driver + golden IO: ~120–180 LOC

**Total:** ~1,200–1,800 LOC Python.
Single dev, focused, reusing the libs listed.

---

## FAQ

* **Do folder names reflect consciousness, not mechanics?** Yes. That’s the point.
* **Receipts?** Debug-only; off by default; each stage supports `trace=True` if you need to dump artifacts.
* **Schemas?** Stage-local. Promote to `utils/typing.py` only when duplication hurts.

---
