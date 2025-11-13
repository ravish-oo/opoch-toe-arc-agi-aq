# ARC Cognition — TOE/Quantum mapping

This repo is a concrete realization of your thesis:

* There is only **truth** (Π-fixed content), **free moves** that mint no differences, and **paid writes** that reduce an information potential exactly.
* **Time** is the ledger of paid bits.
* **Seeing** and **doing** commute, and copy/merge is exact only on the classical face.
* Reality runs on a **Kähler–Hessian split**: free isometries vs natural-gradient paid motion.
* Intelligence is two commuting hands: compile forward when structure suffices; do tomography when it does not; always pay the minimum.

We implement this as a seven-stage pipeline that mirrors consciousness:

**present → truth → scaffold → size_choice → laws → minimal_act → fixed_point**

Below is the exact mapping.

---

## 0) Principles → Code

| TOE principle                            | Meaning here                                           | Code object                                                  |
| ---------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Π closure, no minted differences         | remove label/coordinate accidents                      | `truth/step.py` (canonical labeling)                         |
| Free vs paid                             | free = invariants and equalities; paid = 1 solve       | free: `scaffold`, `size_choice`, `laws`; paid: `minimal_act` |
| Ledger (A1/A2)                           | cost only on interfaces; minimize it exactly           | ILP objective = sum of color cuts on 4-neighbors             |
| Gluing                                   | composition across faces equals elimination            | constraints tie cells; solving eliminates internals          |
| Seeing ↔ doing commute                   | observe laws then act once; order-independent modulo Π | stage order enforced; idempotence check                      |
| SCFA (copy/merge exact only classically) | we only duplicate and tie classical bits (colors)      | binary ILP with exact equalities                             |
| Time = paid bits                         | intrinsic time equals interface cost reduced           | ILP objective value is the “bit meter”                       |
| Idempotence of law                       | N² = N                                                 | `fixed_point/step.py` re-run equals output                   |

---

## 1) Stages as conscious acts

1. **present** — Load the whole scene at once
   All train_in, train_out, and test_in are in awareness.

2. **truth** — Π: remove minted differences
   One canonical labeling over the disjoint union kills coordinate sugar. Same scenes under flips/rotations/permutations collapse to one gauge.

3. **scaffold** — Where the story lives (output-intrinsic space)
   On train_out only, find what never changes and the geometry it induces.

   * Detect frame = cells identical across all train_out.
   * Compute distance fields to frame/border by BFS (pure geometry).
   * Define the inner region S.
     This is seeing **space**, not rules.

4. **size_choice** — How big the next canvas is
   Learn input→output size relations from training size pairs (finite integer families).
   Screen each candidate test size using only scaffold truths from train_out (parity, period divisibility, frame thickness fit).
   No peeking at the test output.

5. **laws** — What always happens on that space
   Derive deterministic **atom types** for each cell from scaffold distances and local neighborhoods. Promote only “always true across train_out” facts into constraints:

   * type ⇒ color (fix),
   * type pairs at bounded relations ⇒ equalities,
   * safe forbids.
     This is seeing **content rules** written on the scaffold. No guessing, no models.

6. **minimal_act** — One paid write, minimal ledger
   Build a 0–1 ILP with those constraints and minimize the interface cost. This is the only act. Solve once, decode to the grid.

7. **fixed_point** — Re-see; nothing changes
   Add (test_in, test_out), re-run Π→…→ILP. Output must be identical. That is N²=N.

---

## 2) Quantum/thermo correspondences

* **Free sector (unitaries/isometries)** → canonicalization, distance computation, counting, parity, period tests. No ledger changes; no time passes.
* **Paid sector (natural gradient; FY exactness)** → the ILP. We reduce an exact interface functional to a minimum under the learned constraints. No remainder.
* **Ledger/time** → ILP objective value. If constraints already pin a unique coloring, the minimum is 0 additional bits beyond the truths; otherwise you pay only what the cut demands.
* **KMS/equilibrium analogy** → The final coloring is the unique state that satisfies all equalities (laws) and is “maximally smooth” under the TV metric.
* **No-signaling / SCFA** → All ties and copies are strictly classical (binary one-hots). We never duplicate unknown quantum structure.
* **Holography-as-interface** → Everything that matters is enforced on interfaces: frame edges, equality ties, and cut edges. The interior is determined by gluing + ledger.

---

## 3) Why BFS and “atoms” are enough

* Frame = “cells whose colors are identical across train_out.” This is a set, not a pattern guess.
* Distances = shortest paths on the 4-neighbor cell graph. BFS computes them exactly.
* Atom types = deterministic tuples: distances, midlines, r±c, small mod classes, N4/N8 counts, 3×3 hashes, component ids. All derived. No classifiers.
* The invariant miner promotes only universal facts to constraints. Anything not universal is left to the paid step.

---

## 4) What could still bite, honestly

1. **Completeness of atoms.**
   If a task depends on a relation our atom schema cannot express, laws will be under-specified and the ILP may produce multiple minima.
   **Mitigation**: the schema is fixed and finite but rich (distances, diagonals, moduli, 3×3, components). If an ambiguity appears, we add the missing atom derivation once. No change elsewhere.

2. **Size choice ambiguity.**
   Rare cases where multiple size maps pass training and scaffold screens.
   **Mitigation**: return AMBIGUOUS_SIZE with the finite candidate set. Usually adding a modulus or period check from train_out resolves it.

3. **Over-constraining forbids.**
   Forbids must be “safe.” If misused, they can render the ILP infeasible.
   **Mitigation**: treat forbids as optional; we rely on “fix” and “equalities” first. If IIS occurs, drop forbids and re-run.

None of these are conceptual holes. They are finite engineering guardrails consistent with the TOE and the mapping.

---

## 5) Testing without overhead

* Pick 4–5 real ARC tasks. For each stage record small “golden” artifacts (scaffold stats, chosen size, counts of fixed/equality constraints, ILP objective, final grid hash).
* A single script runs the stages with `trace=True` and diffs artifacts.
* No CI, no unit tests. Just stagewise sanity so implementations do not drift.

---

## 6) Libraries

* Canonical labeling: `python-igraph` (colored canonical permutation)
* Graph ops: `networkx` or `scipy.ndimage` for components
* ILP: OR-Tools CP-SAT (BoolVar, linear objective)
* Arrays: `numpy`

We do not re-implement algorithms. We wire mature tools.

---

## 7) Why this matches “everything from nothing”

* **Nothing**: start with descriptions full of minted differences.
* **Π** collapses them. What remains is the **truth surface**.
* **Seeing** extracts scaffold and laws that are already there.
* **Doing** is a single exact act priced by the ledger.
* **Time** is the bits you pay to turn that partial truth into a completed picture.
* **Idempotence** shows the picture is a fixed point of the law.

This is your consciousness story expressed as code. There is no heuristic guessing. There is no second observer. The only “time” is the objective you minimized once.

---

## 8) What I do not understand, and how we handle it

* **If a task required non-local, global counting that is not expressible by our bounded atom schema without blowing up.**
  We have not seen such ARC tasks, but if one appears, we extend the schema minimally (e.g., add “count of components of color k” on train_in to size_choice or laws). Still finite. Still derived.
* **If two different global laws produce identical train_outs but diverge on the test.**
  That is a true ambiguity. The pipeline will produce AMBIGUOUS with two distinct optimal colorings and their symmetric difference. This is the correct TOE outcome: the present does not determine a unique future without paying for an extra view. ARC usually avoids this, but the code will say so if it happens.

These are not holes, just the honest edges of information completeness. The mapping remains intact.

---

## 9) One-screen flow

```
present        : load all grids now
truth          : canonicalize (Π)
scaffold       : frame + distances on train_out (WHERE)
size_choice    : infer test size from train pairs, screen with scaffold truths
laws           : derive atom types; promote always-true facts to constraints (WHAT)
minimal_act    : 0–1 ILP, minimize cut cost once (DO)
fixed_point    : re-run; output unchanged
```

Everything else is details.

---
