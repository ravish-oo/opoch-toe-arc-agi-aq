# IMPLEMENTATION_PLAN.md

## Purpose

Ship an ARC solver whose code **mirrors consciousness**:

**present → truth → scaffold → size_choice → laws → minimal_act → fixed_point**

Design goals:

* **100% coverage** on finite ARC grids by construction (no “small constant” shortcuts),
* **Non‑future‑dependent** milestones (each compiles & runs),
* **≤300 LOC** per work order,
* **Deterministic, receipt‑optional** (debug traces are opt‑in),
* Use **mature Python libs** only (no new algorithms).

---

## Repo layout (consciousness‑first)

```
arc_cognition/
  run.py                          # brainstem: wires stages, zero logic
  01_present/                     # load everything into awareness
    step.py
    README.md
  02_truth/                       # Π: canonical gauge (no minted differences)
    step.py
  03_scaffold/                    # WHERE: stable canvas geometry (train_out only)
    step.py
    atoms.py
  04_size_choice/                 # choose test output size using 03 + train sizes
    step.py
    screens.py
  05_laws/                        # WHAT: atom derivations + invariant miner
    step.py
    atoms.py
    relations.py
  06_minimal_act/                 # DO: single paid step (TU LP: tree-TV)
    step.py
    gh_signer.py
  07_fixed_point/                 # N² = N check
    step.py
  utils/                          # tiny shared helpers (only when reused)
    graph.py
  README.md
```

**Import direction:** forward only (01→…→07).
**Receipts:** off by default; `trace=True` returns small JSONs.

---

## Libraries

* **Canonical labeling:** `python-igraph` (colored canonical permutation)
* **Graphs/BFS:** `networkx` or `utils/graph.py` (simple BFS)
* **Labeling/components:** `scipy.ndimage.label` (optional), or BFS
* **LP solver:** any simplex/Interior LP via `scipy.optimize` / `cvxpy` / `pulp` (LP mode)
* **MILP solver:** OR‑Tools CP‑SAT (or similar) for:
  * Full‑TV fallback when TU certification fails
  * Uniqueness probe (Hamming‑cut)
* **Numpy** throughout

> TU-certified tree-TV uses LP; when TU certification fails or LP is non-integral, we fallback to full-TV MILP for exactness.

---

## Milestone 0 — Skeleton & Brainstem (≤150 LOC total) ✅ COMPLETED

**Goal:** runnable skeleton wired to consciousness flow.

### WO‑0.1 `run.py` brainstem (40–60 LOC) ✅ COMPLETED

* Orchestrate calls: `present → truth → scaffold → size_choice → laws → minimal_act → fixed_point(opt)`.
* CLI: `--task PATH`.
  **Acceptance:** `python run.py --task sample.json` prints “stage not implemented” from the first stage.

### WO‑0.2 Stage folders + minimal stubs (40–60 LOC writing) ✅ COMPLETED

* Create `01..07` folders with `step.py` exposing `load`, `canonicalize`, `build`, `choose`, `mine`, `solve`, `check`.
  **Acceptance:** imports succeed; `run.py` starts.

---

## Milestone 1 — present + truth (canonical gauge) (≤450 LOC) ✅ COMPLETED

**Goal:** load ARC JSON; canonicalize all grids in a shared gauge.

### WO‑1.1 Present: JSON loader (80–120 LOC) ✅ COMPLETED

* Parse ARC task into: `train_in`, `train_out`, `test_in` as `np.ndarray[int8]`.
* Validate: palette in 0..9; H,W ≤ 30.
  **Acceptance:** prints shapes/palette; deterministic.

### WO‑1.2 Truth: union graph (120–160 LOC) ✅ COMPLETED

* Disjoint components; per grid:

  * vertices: **cells**, **row_node_r**, **col_node_c**,
  * vertex colors: `("cell", color)`, `("row_node",100)`, `("col_node",101)`,
  * edges: N4 cell adj + (cell↔row_node) + (cell↔col_node).
* **No grid_id** in colors.
  **Acceptance:** |V|,|E| reasonable; dump a checksum.

### WO‑1.3 Truth: canonical labeling (120–160 LOC)  ✅ COMPLETED

* `igraph.Graph.canonical_permutation` with colored vertices.
* Extract per‑grid **canonical row/col order** from perm on row/col gadgets; tie‑break stable on (canon_index, original_index).
* Remap each grid array into canonical coordinates.
  **Acceptance:** repeated runs produce identical arrays; dump a hash.

---

## Milestone 2 — scaffold (WHERE) on train_out ✅ COMPLETED

**Goal:** Build the output-intrinsic scaffold for each training output grid: frame (canvas border), distance atlas, and inner region, plus simple global hints for S0.

### WO-2.1 Frame detector (per-output, border-based) ✅ COMPLETED

For each canonical `train_out[i]`, compute a **frame_maskᵢ** that marks the **outer border** of that grid (top/bottom rows and left/right columns). Store these as `scaffold["per_output"][i]["frame_mask"]` along with each grid’s `(Hᵢ,Wᵢ)`.

### WO-2.2 Distance atlas via BFS / directional scan (per-output) ✅ COMPLETED

For each `train_out[i]`, build 4-adjacency on its cells and compute directional distance fields `d_topᵢ, d_bottomᵢ, d_leftᵢ, d_rightᵢ` from its own frame_maskᵢ (or border if needed). Attach these fields under `scaffold["per_output"][i]`; verify `min=0` and monotone behavior along rows/cols.

### WO-2.3 Inner region & global facts (per-output + aggregated) ✅ COMPLETED

For each `train_out[i]`, define `innerᵢ = (d_topᵢ>0)&(d_bottomᵢ>0)&(d_leftᵢ>0)&(d_rightᵢ>0)`, and compute local parity flags (`has_midrowᵢ/has_midcolᵢ`), thickness candidates (min ring width from inner to frame), and simple row/col period hints inside `innerᵢ`. Then combine these into `scaffold["aggregated"]` (global thickness_min, row_period, col_period, has_midrow_all, has_midcol_all) to be used later by S0.

---

## Milestone 3 — size_choice (test output size) ✅ COMPLETED

**Goal:** enumerate finite size maps from training size pairs; screen with scaffold facts (train_out‑only).

### WO‑3.1 Size map candidates (120–160 LOC) ✅ COMPLETED

* Families (integer‑exact; reproduce **all** training pairs):

  * Identity / swap,
  * Integer affine: `H'W' = M·(H,W) + b` with entries bounded by max observed magnitudes,
  * Factor maps: `H'=r_H·H`, `W'=r_W·W`,
  * Tile/concat: `H'=n_v·H+δ_H`, `W'=n_h·W+δ_W`,
  * Constant size (if train_out are constant).
* **Strict uniqueness policy**: do not pick one if ≥2 fit.
  **Acceptance:** lists candidates; reproductions table exact.

### WO‑3.2 Screens from scaffold (80–120 LOC) ✅ COMPLETED

* Use **only** scaffold facts from train_out to screen candidates for **test** size:

  * Parity (midline requires odd),
  * Periods divide `H',W'`,
  * Frame thickness fits within `H',W'`, inner region non‑negative,
  * Crude capacity checks (if recorded).
* If 1 survivor → choose; if 0 → IIS; if >1 → AMBIGUOUS_SIZE (return set).
  **Acceptance:** picks unique size on known tasks.

---

## Milestone 4 — laws: atom derivations (grid‑aware) (≤600 LOC)

**Goal:** implement the **frozen**, grid‑aware atom menu.

> **Grid‑aware policy:** any former finite range like `{2..6}` becomes **bounded by grid**: divisors or up to `min(H,W)` as appropriate. This preserves 100% coverage without blow‑ups on tiny boards.

### WO‑4.1 Coordinates & distances (A1–A7) (120–160 LOC) ✅ COMPLETED

* Provide arrays for `H,W; r,c; r±c; d_top,bottom,left,right; midrow/midcol`.
* Mod classes: **all divisors** of `H` and of `W` (or `2..min(H,W)` if simpler).
* Block coords: block sizes `b` over **divisors** (tiling‑valid) of `H,W`.
  **Acceptance:** shapes correct; hashes stable.

### WO‑4.2 Local texture (B8–B11) (150–220 LOC) ✅ COMPLETED

* N4/N8 neighbor counts per color (conv or shifts),
* 3×3 full hash (base‑11, pad sentinel=10),
* 5×5 **ring** signature (perimeter only),
* Row/col run‑lengths (length at each cell).
  **Acceptance:** quick checks on small grids; vectorized/fast enough.

### WO‑4.3 Connectivity & shape (C12–C15) (180–240 LOC) ✅ COMPLETED

* Per‑color components via `ndimage.label` or BFS,
* Stats per comp: area, perimeter, bbox, centroid (int), `(height,width)`, `(height-width)`, simple orientation class (sign of Δ moments),
* Rank components by area within color,
* If comp is a ring that touches all sides, compute ring thickness class via min distance.
  **Acceptance:** stats consistent; sample prints.

### WO‑4.4 Repetition & palette/global (D16–D18, E19–E23) (120–180 LOC) ✅ COMPLETED

* Minimal **row** and **column** periods (≤ W or H),
* 2D tiling flags for **factor** pairs `(b_r,b_c)` of `(H,W)`,
* Per‑color pixel counts, component counts, palette set, missing/most/least,
* Input↔output color permutation (bijective map when exists),
* Cyclic color class over active palette.
  **Acceptance:** values correct on crafted examples.

### WO‑4.5 Input feature mirror (F24) (≤30 LOC)

* Reuse A–E **on inputs** to **evaluate** predicates on `test_in` **only when** a mined law references an input feature.
* **Guardrail (explicit):** **F24 does not create new laws.** It only supplies values to instantiate predicates for laws already mined from train_out.
  **Acceptance:** a simple test: a law that says “midrow color equals most‑frequent input color” must **mine from train_out** and merely **read** input stats at test time.

### WO‑4.6 Component rigid/affine transforms (G25–G26) (180–240 LOC)

* Enumerate `D4 × scales` where **scale s** ranges over positive integers that keep the transformed component **inside** the output grid.
* Choose `(A,t)` only if exact pixel set equality holds across paired components in train_out (no fuzzy matches).
* For each cell, derive local coords flags relative to its component’s canonical frame when transported.
  **Acceptance:** detects known translate/flip/rotate/scale patterns; transforms finite per task.

---

## Milestone 5 — laws: invariant miner (WHAT) (≤450 LOC)

**Goal:** promote “always true across train_out” into linear constraints for the test.

### WO‑5.1 Type keyer (100–140 LOC)

* Define **type key** `T(p)` as a tuple of selected atoms (distances, divisors‑mod classes, r±c, 3×3 hash, period class, optional component id).
* Stable hashing & iteration; **no** test_out peeking.
  **Acceptance:** count of unique types; sample keys printed.

### WO‑5.2 Unary fixes (120–160 LOC)

* Map `T → multiset(colors across train_out)`.
* **Strong “always” test:** must be **identical** in every train_out at every occurrence.
* **Anti‑spurious filter:** there exists at least one training where those cells are **non‑zero** (or otherwise not trivially background) to avoid tautologies.
* Emit: for each test cell of type `T`, set `x[p,k]=1`.
  **Acceptance:** prints count of fixed cells; 10 examples.

### WO‑5.3 Relational equalities (140–180 LOC)

* Bounded Δ set driven by grid:

  * `δ_r ∈ [−(H−1)..+(H−1)]`, `δ_c ∈ [−(W−1)..+(W−1)]` **clipped** to in‑bounds on **all** train_out,
  * same `r+c` or same `r−c`,
  * same component class,
  * same 3×3 hash class.
* **Always‑equal** test: for every matched pair in every train_out, colors equal.
* Deduplicate to **equivalence classes** via union‑find; emit pairwise equalities to a canonical rep.
  **Acceptance:** number of classes & pairs; sample printed.

### WO‑5.4 Safe forbids (60–100 LOC) (optional)

* For type `T`, if color `k` **never** appears in any train_out at any `T` cell, and this **cannot** clash with any fix/equality, emit `x[p,k]=0`.
  **Acceptance:** forbids count; IIS check if over‑aggressive.

---

## Milestone 6 — minimal_act: TU LP (tree‑TV) + MILP fallback (≤530 LOC)

**Goal:** build **TU‑LP** (tree‑TV) as the fast path; fallback to **full‑TV MILP** when TU certification fails or LP is non‑integral.

### WO‑6.1 Variables & assignment (120–160 LOC)

* Continuous `x[p,k] ∈ [0,1]`; for each cell `p`: `∑_k x[p,k] = 1`.
* Apply **fixes** (set `x[p,k]=1`) and **forbids** (`x[p,k]=0`).
* For each **equivalence class** `E = {p1..pm}`, enforce `x[p_i,k] = x[p_1,k]` for all `k`.
  **Acceptance:** model builds; counts printed.

### WO‑6.2 Tree‑TV constraints & objective (100–140 LOC)

* Build a **canonical BFS spanning tree** `T` over the test output grid, rooted at top‑left (canonical order tiebreak).
* For each tree edge `(p,q)` and color `k`, add slacks:

  ```
  s[p,q,k] ≥ x[p,k] − x[q,k]
  s[p,q,k] ≥ x[q,k] − x[p,k]
  s[p,q,k] ≥ 0
  ```
* **Objective:** minimize `Σ_{(p,q)∈T, k} s[p,q,k]`.
  **Acceptance:** objective reported; solves on a small case.

### WO‑6.3 TU signer + LP solve + uniqueness probe (≤300 LOC total)

* Run **Ghouila–Houri signer** on the final LP constraint matrix (assignment + equalities + tree‑TV).

  * If signer **passes**: solve as LP with integrality tolerance `τ = 1e−9`.

    * If LP solution is **integral** (all `x[p,k] ∈ {0,1}` within τ):
      * Optionally run **uniqueness probe** (MILP Hamming‑cut) to detect AMBIGUOUS_SOLN:
        * Add cut: `Σ x[p,k]*I[x*=1] ≤ (#cells) − 1`.
        * Re‑solve with same objective.
        * If infeasible or objective increases → **unique solution**.
        * If feasible with same objective → **AMBIGUOUS_SOLN** (return witness #2 and diff mask).
      * If unique or you accept ambiguity, **accept LP solution** (no fallback needed).

    * If LP solution is **non‑integral**: go to **WO‑6.5**.

  * If signer **fails**: go to **WO‑6.5**.

**Acceptance:** signer status logged; LP integral solutions accepted; non‑integral or signer‑fail triggers WO‑6.5.

### WO‑6.4 Decode (60–100 LOC)

* Build `out_grid` with `argmax_k x[p,k]` (redundant but safe with τ).
  **Acceptance:** matches expected on sample tasks.

### WO‑6.5 Full‑TV MILP fallback (120–180 LOC)

**Trigger:** Called from WO‑6.3 when:
* TU signer fails, **or**
* LP solution is non‑integral.

**Behavior:**

Build a new MILP model over the same test canvas:

* **Variables:**
  * `x[p,k] ∈ {0,1}` (Boolean) for each cell `p` and color `k`.
  * TV slacks `s[(p,q),k] ≥ 0` for **all** 4‑neighbor edges `(p,q)` and colors `k`.

* **Constraints:**

  1. **Assignment:** `Σ_k x[p,k] = 1` for all cells `p`.
  2. **Fixes:** for each fixed constraint from Stage N, set `x[p,k]=1`.
  3. **Forbids:** for each forbid, set `x[p,k]=0`.
  4. **Equalities:** for each equivalence class `E`, enforce `x[p_i,k] = x[p_1,k]` for all `p_i ∈ E`, all `k`.
  5. **Full‑TV on all edges:** for every 4‑neighbor edge `(p,q)` and color `k`:
     ```
     s[(p,q),k] ≥ x[p,k] − x[q,k]
     s[(p,q),k] ≥ x[q,k] − x[p,k]
     ```

* **Objective:**
  ```
  min Σ_{(p,q)∈E} Σ_k s[(p,q),k]
  ```
  where `E` is the **full 4‑neighbor edge set** (not just tree).

* Use a standard **MILP/CP‑SAT solver** (e.g., OR‑Tools CP‑SAT) to get the global optimum.

* Decode: `out_grid[p] = argmax_k x[p,k]` (they're 0/1 by construction).

**Acceptance:** On a crafted non‑TU or non‑integral case, WO‑6.3 triggers WO‑6.5; full‑TV MILP returns integral optimal solution; objective equals sum of all interface cuts; decoded grid becomes final answer for Stage 7.

---

## Milestone 7 — fixed_point (N² = N) (≤120 LOC)

**Goal:** append the solved pair and confirm idempotence.

### WO‑7.1 `07_fixed_point/step.py` (80–120 LOC)

* Create a new bundle with `(test_in, out_grid)` appended to training.
* Re‑run `run()` or the stage chain; ensure bit‑for‑bit identical `out_grid`.
* On mismatch: dump a minimal repro (which stage diverged) and **raise** (bug).
  **Acceptance:** passes on goldens.

> **Note:** This WO was missing in the earlier plan; now explicit.

---

## Milestone 8 — stagewise goldens (manual, no CI) (≤250 LOC)

**Goal:** lock stage outputs for 5 real tasks; diff on demand.

### WO‑8.1 Golden artifacts & dumper (120–160 LOC)

* For each chosen task, store small JSONs:

  * `truth` (row/col orders hashes),
  * `scaffold` (frame count, inner count, distance sums),
  * `size_choice` (candidates + chosen),
  * `laws` (counts of fixes/equalities/forbids, 10 samples),
  * `lp` (TU signer result, objective, integrality violations),
  * `final_grid` (hash).
* Add `--trace` flag to each stage to dump these.

### WO‑8.2 Comparator (80–120 LOC)

* Script runs the pipeline with `trace=True`, diffs current vs golden JSONs, prints per‑stage PASS/FAIL.
  **Acceptance:** all 5 tasks pass locally.

---

## Guardrails & Non‑Goals

* **F24 (input atoms):** used **only** to evaluate predicates on `test_in` when a **mined** law references input features. **Do not** mine laws from input alone.
* **No peeking at test_out.** All laws mined from train_out; test_out only produced by LP.
* **Optional flows:** Connectivity flows are **off by default**. They may be enabled later **only** if train_out shows a color forms exactly one connected component and there are non‑empty seeds, and they must be implemented **on the tree** to preserve TU.
* **Receipts:** off by default; `trace=True` enables tiny JSONs for debugging.
* **No randomization** anywhere.

---

## Effort & LOC (single dev)

* M0–M1: 1.5–2 days (~300–450 LOC)
* M2–M3: 1.5–2 days (~350–560 LOC)
* M4: 2–3 days (~450–600 LOC)
* M5: 2 days (~300–450 LOC)
* M6: 1.5–2 days (~370–530 LOC)
* M7–M8: 1–1.5 days (~160–250 LOC)

**Total:** ~10–13 focused days; ~1,930–2,840 LOC; each WO ≤300 LOC and independently runnable.

---

## Acceptance Matrix (per milestone)

* **M1:** canonical grids stable across runs (hash equal).
* **M2:** frame & distance atlas correct on known cases.
* **M3:** size choice unique or AMBIGUOUS_SIZE/IIS signaled.
* **M4:** atom arrays returned; grid‑aware ranges honored; no heuristics.
* **M5:** rules satisfy “always” with **zero exceptions**; anti‑spurious filter applied.
* **M6:** TU signer passes OR fallback to full‑TV MILP; solution integral; uniqueness probe handled.
* **M7:** idempotence holds.
* **M8:** goldens diff‑clean.

---

## Why this plan is non‑future‑dependent

Each milestone depends **only** on earlier artifacts (and not on stubs of later code). Every WO compiles and runs in isolation and produces tangible outputs (arrays, JSONs, hashes). You can implement in order, verifying with the goldens at each step.

---