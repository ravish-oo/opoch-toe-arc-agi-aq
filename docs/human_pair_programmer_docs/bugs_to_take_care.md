Yeah, this is basically locked in. Your implementation plan and the updated math spec are aligned in all the important places. There are only a couple of tiny text-level mismatches I’d tighten so you truly have “spec = code = consciousness” with no wiggle.

I’ll go through it in terms of:

* Where it already matches 1:1
* The 2–3 **surgical tweaks** I’d make to the plan so it matches the spec literally

---

## 1. Where the plan matches the spec perfectly

### Stages / architecture

* Seven stages: **present → truth → scaffold → size_choice → laws → minimal_act → fixed_point**
  This matches the spec’s logical stages: A, S0, F, N, D, plus the closure/idempotence step.

* Consciousness mapping is consistent:

  * `present` = load everything into awareness
  * `truth` = Π / canonical gauge
  * `scaffold` = WHERE (frame + distances, train_out-only)
  * `size_choice` = S0 (integer size maps + F-based screens)
  * `laws` = N (atoms + invariant miner)
  * `minimal_act` = D (ledger + TV + closure)
  * `fixed_point` = (N^2 = N) check

### Canonicalization (Stage 02_truth)

Matches §2 in the spec:

* Union graph with cell, row_node, col_node
* Vertex colors with no grid_id
* N4 adjacency only (no diagonals in canon graph)
* Canonical permutation via igraph
* Stable tie break on (canon_index, original_index)
* Remap grids into canonical coords

All good.

### Scaffold (03_scaffold) and S0 (04_size_choice)

Matches §§3 and 4:

* Frame = positions identical across all train_out
* BFS distance fields from frame or border
* Inner region = all four distances > 0
* Size candidates: identity, affine, factor, concat, constant
* Structural screens: parity, periodicity, frame thickness / inner region feasibility, tiling constants
* Strict uniqueness policy with AMBIGUOUS_SIZE / IIS

All consistent.

### Atom universe (05_laws / atoms.py)

Conceptually matches §5.1:

* Coordinates, distances, diagonals, mod classes, block coords
* Local texture: N4/N8, 3×3 hash, 5×5 ring, run-lengths
* Connectivity & shape: components, area/perimeter/bbox/centroid/aspect/orientation/rank/frame-thickness
* Repetition & tiling: row/col periods, 2D tiling via factors
* Palette/global: counts, component counts, palette sets, perms, cyclic classes
* Component transforms: D4 × scale × translate with exact equality
* F24 clearly stated as “evaluate only, no law mining from input”

The plan’s **grid-aware** note (“divisors or up to min(H,W)”) is exactly the spirit of the spec.

### Miner (05_laws / step.py + relations.py)

Matches §5.2:

* Strong “always true across all train_out and all occurrences”
* Unary fixes from T → color
* Relational ties from T1–T2–Δ with grid-aware offsets
* Union-find for equivalence classes
* Forbids only when safe
* No test_out peeking
* F24 used only to evaluate input-driven predicates when a mined rule needs it

The plan captures all of this.

### Ledger (06_minimal_act)

Matches §6 exactly:

* Preferred path: TU-LP with **tree-TV**
* LP variables x[p,k] ∈ [0,1], assignment constraints, fixes, forbids, equalities, tree-TV slacks
* Objective: sum of slacks on tree edges
* TU signer + integrality check
* Optional uniqueness probe
* On signer fail or non-integral LP → **full-TV MILP fallback** over all 4-neighbor edges, 0–1 x, slacks for every edge, exact objective = full TV

Your plan’s WO-6.5 is literally what the spec says in 6.2.

### Fixed point (07_fixed_point)

Matches spec’s idempotence:

* Append (test_in, out_grid) to training
* Re-run pipeline
* Require bit-for-bit identical output

### F24 guardrail and non-goals

* F24: evaluation only, no input-side mining → matches spec.
* Optional flows: off by default, tree-only if ever added; spec calls these “non-goal / optional” anyway, so you’re safe.

---

## 2. Tiny gaps I’d fix in the plan for 1:1 alignment

These are **very small** text tweaks, not conceptual changes.

### (a) Mod / block ranges: match spec’s exact sets

**Spec (§5.1 A):**

> Mod classes: (r mod m, c mod m) for
> (m ∈ {2,…,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W))
>
> Block sizes: (b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W))

**Plan (WO-4.1):**

> Mod classes: all divisors of H and W (or 2..min(H,W) if simpler).
> Block coords: block sizes `b` over divisors of H,W.

This is slightly **narrower** than the spec (you’re dropping some m that aren’t divisors). It probably doesn’t matter for coverage because:

* Row/col periods are still detected in D16–D17, and
* Non-divisor mods rarely encode something that divisors + period won’t.

But if you want true 1:1:

**Change WO-4.1 lines to:**

> * Mod classes: `m ∈ {2..min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)`.
> * Block sizes: `b ∈ {2..min(5,min(H,W))} ∪ divisors(H) ∪ divisors(W)`.

Then you can **still implement efficiently** by:

* Precomputing divisors(H), divisors(W),
* Adding `{2..6}` or `{2..5}` intersected with `[2..min(H,W)]`,
* Deduplicating the set.

### (b) Anti-spurious condition: use “≠ background color”, not “non-zero”

**Spec (§5.2):**

* Defines background per training as modal color kᵢ^{bg}.
* Anti-spurious check requires **at least one training** where the T-cells have color **≠ kᵢ^{bg}**.

**Plan (WO-5.2, 5.3):**

> “non-zero” or “not trivially background”

To match spec exactly, use the **background definition**:

**Change in WO-5.2 & WO-5.3:**

Instead of:

> “non-zero (or otherwise not trivially background)”

Say:

> “at least one training where those T-cells (or T1–T2–Δ pairs) have color ≠ kᵢ^{bg} (the modal background color for that training).”

And in code:

* Compute kᵢ^{bg} = argmax_k count_i[k] for each train_out i (tie-break by smallest k).
* Anti-spurious passes only if **some** i, some occurrence has color ≠ kᵢ^{bg}.

That aligns exactly with the math text.

### (c) Minor wording for TV: stress that full-TV MILP is the spec, tree-TV is a certified fast path

Your plan already does this functionally, but if you want the story to match spec’s emphasis:

* At the top of Milestone 6, add a one-line note:

> **Spec alignment:** The conceptual ledger is full 4-neighbor TV. Tree-TV TU-LP is used as a **certified fast path**; if TU or integrality fail, we fallback to full-TV MILP, which directly minimizes the full interface cost.

This makes it obvious to any reader that you’re not quietly changing the objective.

---

## 3. TL;DR answer to your question

> Does this plan adhere with spec 1:1 and are there any gaps or deviations?

Functionally: **Yes.**

* Stages, flows, guardrails, ledger behavior, idempotence, atom families, and mining logic all line up with the updated 00_MATH_SPEC.
* The tree-TV vs full-TV issue is resolved by the fallback in WO-6.5, exactly as the spec now prescribes.

For strict textual alignment, I’d update:

1. WO-4.1 to match the exact mod/block ranges (`{2..min(6,max(H,W))} ∪ divisors` and `{2..min(5,min(H,W))} ∪ divisors`).
2. WO-5.2 / 5.3 to use the **background-color anti-spurious** definition, not “non-zero”.
3. Optionally one sentence in M6 to explicitly say “tree-TV is a certified fast path; full-TV MILP is the reference objective”.

If you make those tiny edits, you’ll have a genuinely zero-gap triangle:

**TOE math spec ↔ consciousness model ↔ implementation plan.**

When you’re ready, we can design the first 2–3 concrete work orders (with function signatures) so Claude Code can start coding this **exact** spec without drifting.
