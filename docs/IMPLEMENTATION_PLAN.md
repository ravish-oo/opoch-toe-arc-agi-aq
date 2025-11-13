Sequential “consciousness” milestones with **non-future-dependent, smaller work orders** that each compiles and runs. Here’s a concrete plan you can hand to Claude Code today.

# Milestone 0 — Repo + brainstem (≤150 LOC total)

**Goal:** runnable skeleton wired to the consciousness flow, no stubs beyond `NotImplementedError`.

**WO-0.1** Brainstem `run.py` (40–60 LOC)

* Wires: `present → truth → scaffold → size_choice → laws → minimal_act → fixed_point(opt)`.
* CLI `--task path.json` that prints final grid.
  **Acceptance:** `python run.py --task sample.json` prints placeholder “not implemented” from first stage.

**WO-0.2** Stage folders + READMEs (40–60 LOC writing)

* Create `01_present/... 07_fixed_point/...` with `step.py` files that each define a function (`load`, `canonicalize`, `build`, `choose`, `mine`, `solve`, `check`).
  **Acceptance:** repo tree matches plan; imports succeed.

---

# Milestone 1 — present + truth (canonical gauge) (≤450 LOC)

**Goal:** load ARC JSON and canonicalize grids; returns canonical train_in/out and test_in.

**WO-1.1** Present: JSON loader (80–120 LOC)

* Parse ARC task JSON → `train_in: List[HxW np.int8]`, `train_out`, `test_in`.
* Validate palette 0..9, sizes ≤ 30.
  **Acceptance:** prints counts, shapes; deterministic.

**WO-1.2** Truth: disjoint-union graph build (120–160 LOC)

* Build colored graph (cells + row/col nodes; adj, row_edge, col_edge).
* Use `python-igraph` (or `networkx` + `nauty` binding if you prefer).
  **Acceptance:** returns graph with |V|, |E| logged; colors assigned.

**WO-1.3** Truth: canonical labeling + per-grid row/col order (120–160 LOC)

* Get canonical permutation; sort row/col nodes → maps R_X, C_X.
* Remap each grid’s array to canonical order.
  **Acceptance:** repeated runs yield identical canonical arrays; dump small checksum.

---

# Milestone 2 — scaffold (WHERE) on train_out (≤350 LOC)

**Goal:** find frame and distance atlas; define inner region.

**WO-2.1** Frame detector (80–120 LOC)

* Frame mask = positions whose color is identical across all train_out.
  **Acceptance:** on sample tasks, frame pixels count >0 when expected.

**WO-2.2** Distance atlas via BFS (120–160 LOC)

* If frame exists: multi-source BFS from frame; else from border.
* Produce `d_top, d_bottom, d_left, d_right`.
  **Acceptance:** min=0 at frame/border; monotone inward; checksum logged.

**WO-2.3** Inner region + scaffold facts (80–120 LOC)

* `inner = (d_top>0)&(d_bottom>0)&(d_left>0)&(d_right>0)`.
* Record parity flags, candidate thickness, simple period hints from train_out.
  **Acceptance:** inner nonempty when expected; facts JSON dump.

---

# Milestone 3 — size_choice (test output size) (≤280 LOC)

**Goal:** enumerate finite size maps from train pairs; screen with scaffold facts (train_out-only).

**WO-3.1** Size map candidates (120–160 LOC)

* Families: identity/swap; affine with small ints; factor (r_H,r_W∈ℤ⁺); concat/tiling; constant out.
* Fit only those that match **all** train pairs.
  **Acceptance:** prints candidate list for sample tasks.

**WO-3.2** Screens from scaffold (80–120 LOC)

* Check parity (midline requires odd), periods divide W/H, frame thickness fits inner, component capacity plausible.
* Pick unique survivor; else AMBIGUOUS_SIZE/IIS.
  **Acceptance:** one chosen size on known tasks.

---

# Milestone 4 — laws: atoms (derivations) (≤600 LOC)

**Goal:** implement the **fixed 26-item atom menu** as pure functions. Keep each file ≤250 LOC by grouping.

**WO-4.1** Coordinates & distances (A1–A7) (120–160 LOC)

* `r,c, r±c, r%{2..6}, c%{2..6}, block coords`, distances & midline flags.

**WO-4.2** Local texture (B8–B11) (150–220 LOC)

* N4/N8 counts per color; 3×3 hash (base-11, pad=10); 5×5 ring signature; row/col run lengths.

**WO-4.3** Connectivity & shape (C12–C15) (180–240 LOC)

* Per-color components (`scipy.ndimage.label`), stats (area, perimeter, bbox, centroid, aspect class, simple orientation), rank by area, frame-ring thickness class.

**WO-4.4** Repetition/tiling + palette/global (D16–D18, E19–E23) (120–180 LOC)

* Row/col minimal periods; 2D tileable (2..6) block flags; per-color counts & component counts; palette set; bijective color perm; cyclic class.

**WO-4.5** Input feature mirror (F24) (≤30 LOC)

* Reuse same derivations on train_in/test_in on demand.

**WO-4.6** Component rigid transform (G25–G26) (180–240 LOC)

* Enumerate D4×{1,2,3}×translate; verify exact set match; local coords flags.

**Acceptance:** each function returns arrays or dicts with checksums; no heuristics; deterministic.

---

# Milestone 5 — laws: invariant miner (WHAT) (≤450 LOC)

**Goal:** promote “always true across train_out” into constraints for test ILP.

**WO-5.1** Type keyer (100–140 LOC)

* Build type key `T(p)` as a tuple/hash from selected atoms (distances, mod, r±c, 3×3 hash, component id, period class).
  **Acceptance:** keys stable across runs; count of unique types printed.

**WO-5.2** Unary fixes (120–160 LOC)

* Map `T → multiset(colors)`. If singleton `{k}` across all train_out, add `fixed[p]=k` for test cells of same type.
  **Acceptance:** prints number of fixed cells; show 10 examples.

**WO-5.3** Relational equalities (140–180 LOC)

* Bounded Δ set: row offsets δ∈[-6..6], col offsets, same diag class (r+c, r−c), same component class, same 3×3.
* If always equal in train_out, add tie `(p,q)` for test when types & Δ realized.
  **Acceptance:** prints number of equality pairs; examples.

**WO-5.4** Safe forbids (optional) (60–100 LOC)

* Only forbid `x[p,k]=0` when consistent with fixes/ties; otherwise skip.
  **Acceptance:** forbids count reported; IIS check if too aggressive.

---

# Milestone 6 — minimal_act: ILP build/solve/decode (≤350 LOC)

**Goal:** build 0–1 ILP for test canvas; minimize interface; decode grid.

**WO-6.1** CP-SAT model (120–160 LOC)

* BoolVar `x[p,k]`; assignment `Σ_k x[p,k]=1`.
* For each tie `(p,q)`: `x[p,k]==x[q,k]` ∀k.
* For each fix/forbid: set directly.
  **Acceptance:** model builds; var/constraint counts printed.

**WO-6.2** Interface objective (100–140 LOC)

* For each 4-nbr edge `(p,q),k` add Bool `diff[p,q,k] == XOR(x[p,k], x[q,k])`.
* Objective = `Σ diff`.
  **Acceptance:** model solves on small case; objective reported.

**WO-6.3** Decode & return grid (60–100 LOC)

* Pick `argmax_k x[p,k]` (they’re 0/1 from CP-SAT).
* Return `out_grid`; also return objective value.
  **Acceptance:** final grid matches known sample tasks.

---

# Milestone 7 — goldens (manual stagewise checks) (≤250 LOC)

**Goal:** single script runs 5 chosen tasks, dumping/validating **per-stage** artifacts.

**WO-7.1** Golden schema + dump (120–160 LOC)

* For each stage, save small JSONs: scaffold stats, chosen size, counts of fixed/ties/forbids, ILP objective, final grid hash.
  **WO-7.2** Comparator (80–120 LOC)
* Runs pipeline with `trace=True`; diffs current vs golden; prints per-stage pass/fail.
  **Acceptance:** all 5 pass on your machine.

---

# Milestone 8 — polish (≤200 LOC)

**WO-8.1** Errors & messages

* AMBIGUOUS_SIZE / AMBIGUOUS_SOLN / IIS with minimal witness snippets.
  **WO-8.2** README final pass, sample commands, timings.

---

## Guardrails for Claude Code

* Each **work order ≤300 LOC**; after each WO, the repo must run end-to-end up to that stage (earlier stages compiled).
* **No new algorithms**: use `python-igraph`, `networkx/scipy`, `ortools`.
* **Deterministic outputs**: no RNG.
* **No peeking**: miner reads only train_out; size screens use only scaffold(train_out); test is used to **evaluate** predicates and solve ILP.
* **Trace toggles**: `trace=True` returns small dumps; default is silent/fast.
