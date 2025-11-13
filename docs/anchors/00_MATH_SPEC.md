# 00_MATH_SPEC.md

## 0. Ground truth (TOE)

We take as absolute truth:

* **Π (truth closure):** a cleanup map with (\Pi^2 = \Pi). It removes minted/artificial differences (renamings, coordinate choices).
* **A0 — No minted differences:** truth = fixed points of Π.
* **A1 — Exact balance (Fenchel–Young):** every paid change has a dual cost; no free gain around loops; “real actions” pay ledger cost.
* **A2 — Gluing:** all price resides on interfaces (faces); on grids, that’s cost for color differences across adjacencies.

For an ARC task, the **law** is a closure (N) on grids such that:

* (N(X^{(i)}*{\rm in}) = X^{(i)}*{\rm out}) for all training pairs,
* (N^2 = N),
* Among all such closures consistent with training invariants, (N) minimizes **interface cost** (A1/A2).

The test output is:
[
X^{\text{test}}*{\rm out} ;=; N\big(X^{\text{test}}*{\rm in}\big).
]

---

## 1. Representing the task

Given training pairs (\mathcal D = {(X^{(i)}*{\rm in}, X^{(i)}*{\rm out})}*{i=1}^T) and the test input (X^{\text{test}}*{\rm in}).

Each grid (X):

* height (H_X), width (W_X),
* cells (U_X = {(r,c): 0\le r < H_X,; 0\le c < W_X}),
* colors (c_X:U_X \to \mathcal C \subseteq{0,\dots,9}).

---

## 2. Stage A — Canonical labeling (awareness & gauge)

We remove minted coordinate choices so all grids live in a common, relational gauge.

### 2.1 Disjoint‑union graph (G)

Per grid (X):

* **Vertices:** cell (v_{r,c}), row node (v^{\rm row}_r), col node (v^{\rm col}_c).
* **Vertex colors** (no grid id; no absolute indices):
  [
  \text{vcolor}(v)=
  \begin{cases}
  (\text{“cell”},,c_X(r,c)) & v=v_{r,c}\
  (\text{“row_node”},,100) & v=v^{\rm row}_r\
  (\text{“col_node”},,101) & v=v^{\rm col}_c
  \end{cases}
  ]
* **Edges:** 4‑neighbor cell adjacencies (N4 only; no diagonals in canon graph); and cell↔row, cell↔col incidence edges.
* No edges between different grids (components are disjoint).

### 2.2 Canonical labeling

Run a canonical labeling (e.g., igraph/Traces) to get a canonical permutation (\pi). Use a **stable tie‑break** for row/col gadgets:

1. Apply canonical permutation to all vertices.
2. For row_node and col_node gadgets, sort by ((\text{canon_index}, \text{original_index})).
3. Store the resulting (\mathbf{R}_X(r)) and (\mathbf{C}_X(c)) arrays as the canonical row/col orders.

### 2.3 Canonical local coordinates

For each grid (X):

* Canonical row order (R_X(r)\in{0,\dots,H_X-1}),
* Canonical col order (C_X(c)\in{0,\dots,W_X-1}),
* Canonical cell coords (\rho_X(r,c)=R_X(r)), (\kappa_X(r,c)=C_X(c)).

Re‑running yields identical canonical arrays.

**Receipt A:** Canonical maps (\mathbf{R}_X), (\mathbf{C}_X); orbit partition; canonicalization library/version flags.

---

## 3. Stage S0 — Output canvas size

Determine ((H^{\text{test}}*{\rm out},W^{\text{test}}*{\rm out})) from training size pairs and structural screening **using only Stage F facts from train_out**.

### 3.1 Candidate size maps

Collect integer relations that fit **all** training pairs:

* identity/swap,
* integer affine (\begin{bmatrix}H'\W'\end{bmatrix}=M\begin{bmatrix}H\W\end{bmatrix}+b),
* factor maps (H'=r_H H,; W'=r_W W),
* tiling/concat (H'=n_v H + \delta_H,; W'=n_h W + \delta_W),
* constant (H',W') (if train_out sizes coincide).

### 3.2 Structural disambiguation (train_out‑only)

Screen candidates for the test size using only F‑facts from train_out. Each candidate ((H', W')) must pass:

1. **Feasibility:** For every training output's learned frame thickness (t) and inner region ((h_{\text{inner}}, w_{\text{inner}})):
   [
   H' \ge h_{\text{inner}} + 2t, \quad W' \ge w_{\text{inner}} + 2t.
   ]
2. **Parity:** If all train_out have a midrow ((\exists r: d_{\text{top}}(r) = d_{\text{bottom}}(r))) then (H') must be odd; similarly for (W') and midcol.
3. **Periodicity:** If train_out rows (or cols) have least period (p) everywhere inside the inner region, then (p \mid H') (resp. (p \mid W')).
4. **Tiling constants:** For maps (H' = n_v H + \delta_H, W' = n_h W + \delta_W), require the same integers ((n_v, n_h, \delta_H, \delta_W)) fit all trainings. If frame thickness (t) is detected, enforce (\delta_H, \delta_W \in \{0, 2t\}) only.

Policy:

* 1 survivor → choose,
* 0 → **IIS**,
* > 1 → **AMBIGUOUS_SIZE** (return finite candidate set).

**Receipt S0:** Candidate set; survivors after rules 1–4; chosen size or AMBIGUOUS_SIZE.

---

## 4. Stage F — Frame & distances (global relational coordinates)

Define output‑intrinsic scaffold.

### 4.1 Frame from training outputs

In canonical coords, the **frame set** is:
[
F = \bigl\{ p \;\big|\; \exists\, k\;\text{s.t.}\; c^{(i)}_{\text{out}}(p) = k\;\;\forall\, i \bigr\}.
]
(Positions with the same color across all train_out.)

### 4.2 Distance fields

For each output grid (train_out and the test canvas size once fixed):

* Build 4‑adjacency graph on cells.
* **BFS source set:**
  * If (F \neq \varnothing): multi‑source BFS from all cells in (F).
  * Else: multi‑source BFS from the **outer border** (all cells (r{=}0), (r{=}H{-}1), (c{=}0), (c{=}W{-}1)).
* Compute distances: (d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}}) (integers, 4‑adjacency).
* **Inner region:** (S = \bigl\{ p \,:\, d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}} > 0 \bigr\}).

**Receipt F:** Frame mask (F); ((d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}})) checksum; inner mask (S).

---

## 5. Stage N — Invariants as linear constraints (frozen, grid‑aware)

We use a **closed, grid‑aware atom universe** and promote “always‑true” facts across train_out to constraints on the test grid.

### 5.1 Atom universe (derivations, not detectors)

All computed in canonical coords.

**A. Scaffold & coords**

* (H,W); (r,c); (r\pm c); midrow/midcol flags.
* Mod classes: (r\bmod m,\, c\bmod m) for (m \in \{2,\dots,\min(6, \max(H,W))\} \cup \text{divisors}(H) \cup \text{divisors}(W)).
* Block coords ((\lfloor r/b\rfloor,\lfloor c/b\rfloor)) and remainders for (b \in \{2,\dots,\min(5, \min(H,W))\} \cup \text{divisors}(H,W)).

**B. Local texture**

* N4/N8 neighbor counts per color.
* 3×3 neighborhood hash (base‑11 packing; palette 0..9, sentinel=10 for out‑of‑grid padding when computed at borders).
* 5×5 **ring** signature (perimeter only; sentinel=10 when ring steps outside grid).
* Row/col run‑lengths through the cell: (\text{span_len}, \text{span_start}, \text{span_end}).

**C. Connectivity & shape**

* Per‑color connected components; per‑component: area, perimeter (4‑edge), bbox, centroid (int floor), ((h,w)) (height and width as integers), (h{-}w) (aspect difference), aspect class, simple orientation sign; area rank; frame‑ring thickness class.

**D. Repetition & tiling**

* Minimal period along row/col (≤ dimension).
* 2D tiling flags for block sizes dividing (H,W).

**E. Palette/global**

* Per‑color pixel counts; per‑color component counts.
* Palette present/missing; most/least frequent color(s).
* Input↔output color permutation (bijective) & cyclic class over active palette.

**F. Input features (guardrail)**

* Mirror A–E on **inputs** to **evaluate predicates on test_in** **only when referenced by a mined law**.
* **F24 does not create new laws.** It never mines from inputs.

**G. Component transforms**

* D4 (rot/ref) × integer scale (s) such that the transformed component fits inside the output grid; plus translations.
* Accept only **exact set equality** across paired components in train_out; emit local‑coords flags if matched.

### 5.2 Miner rules (frozen)

Let (T(p)) be a deterministic **type key** (tuple from atoms in stable order):
[
T(p) = \bigl( d_{\text{top}}, d_{\text{bottom}}, d_{\text{left}}, d_{\text{right}}, r{\pm}c, \{r\bmod m\}, \{c\bmod m\}, \text{3×3 hash}, \text{period flags}, \text{component shape ID} \bigr).
]

* **Unary fixes (type ⇒ color):**

  Let (k_i^{\text{bg}} = \arg\max_k \text{count}_i[k]) be the **modal background color** in training output (i) (on tie, smallest (k)).

  For type (T), if for **every** train_out at **every** occurrence of (T) the color is the **same** (k), **and** there exists at least one training where those (T)-cells have color (\neq k_i^{\text{bg}}) (**anti‑spurious**), then **fix** test cells of type (T) to (k): set (x_{p,k}=1).

* **Relational equalities (ties):**

  Consider finite grid‑aware relations (\Delta):

  * Row/col offsets ((\delta_r, \delta_c)) where (\delta_r \in [-(H{-}1),+(H{-}1)], \delta_c \in [-(W{-}1),+(W{-}1)]).
    **Clipping rule:** If pair ((p, p{+}\delta)) would be out‑of‑bounds in **any** training, discard that pairing entirely.
  * Same diagonal class ((r{+}c) or (r{-}c)),
  * Same component class,
  * Same 3×3 hash class.

  For a relation (\Delta), if for **every** train_out **all** matched pairs ((p,q)) realizing ((T_1,T_2,\Delta)) have **equal colors**, **and** at least one training sees a non‑background color ((c \neq k_i^{\text{bg}})) on those positions (**anti‑spurious**), then union them into an **equivalence class** (deduplicate via union‑find) and emit pairwise equalities to a canonical representative:
  [
  x_{p,k} = x_{\text{rep},k}\quad \forall k.
  ]

* **Forbids (optional, safe):**
  If color (k) **never** occurs at type (T) in any train_out and this cannot conflict with any fix/equality, add (x_{p,k}=0).

**Receipt N:** Type→color table; DSU equivalence classes with member lists; forbid list; proof tables (zero exceptions + non‑background witness).

---

## 6. Stage D — Ledger minimization (paid step, with certification)

**Ledger principle:** minimize interface cost subject to Stage N constraints. We provide two **spec‑valid** computational paths; the implementation must choose the one that is **certified exact** for the instance.

* **Full‑TV ILP (reference):** minimize cuts over **all** 4‑neighbor edges (as in the original text).
* **TU‑LP with tree‑TV (preferred when certified):** minimize cuts over a **canonical spanning tree**; prove TU and integrality; verify uniqueness; if certification fails, **fallback to full‑TV ILP**.

### 6.1 Preferred path: TU‑LP with tree‑TV (fast, certified)

* Build the canonical BFS **spanning tree** (T) of the test canvas:
  * Root: ((r{=}0, c{=}0)) in canonical order.
  * Neighbor visit order: (up, right, down, left) in canonical ordering.

* Variables (LP):
  (x_{p,k}\in[0,1]) (one‑hot); (s_{e,k}\ge 0) for (e\in T).

* Constraints:

  1. **Assignment:** (\sum_k x_{p,k}=1) for all cells (p).
  2. **Fixes:** set (x_{p,k}=1) where mined.
  3. **Equalities:** for each equivalence class (E), tie (x_{p,k}) to the class representative for all (k).
  4. **Tree‑TV:** for each (e=(p,q)\in T), each color (k):
     [
     s_{e,k} \ge x_{p,k} - x_{q,k},\qquad
     s_{e,k} \ge x_{q,k} - x_{p,k}.
     ]

* **Objective:** (\min \sum_{e\in T}\sum_k s_{e,k}.)

* **TU certification (constructive):**

  Implement the following structural check:

  1. **Contract equalities per color:** Use union‑find to contract equality classes separately for each color (k), creating supernodes.
  2. **Build incidence matrices:** For each color (k), construct the oriented incidence matrix (\mathbf{B}_k) of the contracted tree (T_k) (same root orientation). Incidence matrices of forests are TU.
  3. **TV rows as [\mathbf{B}_k | -\mathbf{I}]:** Each TV constraint row is ([B_k \mid {-}I]) per color (adding identity on slacks preserves TU).
  4. **Assignment rows as partition matrix:** Assignment rows ((\sum_k x_{p,k}=1)) form a partition matrix across colors; this is TU and block‑separable per node.
  5. **Block-diagonal composite:** The final matrix is a block‑diagonal stack over colors for ([\mathbf{B}_k \mid {-}\mathbf{I}]) plus a laminar set of assignment rows across color blocks—this composite is TU.

  Verify by constructing (\mathbf{B}_k) explicitly and checking that each column appears in at most two signed rows per block and that partition rows are (0/1).

  If any structural step fails (e.g., a non‑tree edge sneaks in), mark **TU‑FAIL** and go to **6.2**.

  If TU certification passes, solve as **LP** with integrality tolerance (\tau = 10^{-9}). If LP returns non‑integral (x), go to **6.2**.

* **Uniqueness probe (optional but recommended):**
  Add a single “Hamming‑cut” excluding the found solution and re‑solve once as MILP.
  If infeasible or objective increases → unique; else **AMBIGUOUS_SOLN** (return a second optimal coloring and the diff mask).

> **Note:** Tree‑TV is a lower‑bound proxy for full‑TV. With the equality constraints mined in Stage N, cycles are typically closed, and the tree objective equals the full cut objective on feasible colorings. Certification + fallback (6.2) ensures exactness w.r.t. A2.

### 6.2 Reference path: full‑TV ILP (exact fallback)

* Edge set (E) = **all** 4‑neighbor edges ((p,q)).
* Variables (0–1): (x_{p,k}\in{0,1}); auxiliary slacks or XOR diffs per edge & color.
* Constraints:

  * Assignment, fixes, equalities as above.
  * TV linearization on **all** edges:
    [
    s_{(p,q),k} \ge x_{p,k} - x_{q,k},\quad
    s_{(p,q),k} \ge x_{q,k} - x_{p,k}.
    ]
* **Objective:** (\min \sum_{(p,q)\in E}\sum_k s_{(p,q),k}.)

This is the exact formulation stated originally. Use MILP to guarantee a global optimum when TU‑LP certification fails.

**Receipt D:** Structural TU check (pass/fail); LP objective; integrality violations (by (\tau)); uniqueness probe outcome or MILP optimal value.

---

## 7. Full program (test output)

Let (P) be test canvas cells; (K = \bigcup_i \text{Palette}(X_{\text{out}}^{(i)})) the **color set seen in training outputs**. Do not allow unseen colors unless a mined law explicitly references an input→output palette mapping.

* Variables: (x_{p,k}) (LP/LBP or Bool in MILP), TV slacks (s).
* Constraints:

  1. (\sum_{k\in K} x_{p,k}=1) for all (p\in P),
  2. Fixes / equalities / forbids from N,
  3. TV constraints (tree edges for TU‑LP; all edges for MILP).
* Objective: minimize the corresponding TV sum.

**Solve policy:** attempt **TU‑LP tree‑TV** with signer and integrality check; if certification fails (or ambiguity detected and you choose to disambiguate), **fallback to full‑TV ILP**.

**Decode:** (X^{\text{test}}_{\rm out}(p)=\arg\max_k x^*_{p,k}) (LP is integral when TU holds).

**Deterministic ordering:** Index variables in lexicographic order ((r, c, k)) with colors sorted ascending; store in receipts.

---

## 8. Edge cases (explicit)

* **Unique optimum:** output is (x^*), decode grid; **idempotence** must hold ((N^2=N)).
* **Multiple optima:** return **AMBIGUOUS_SOLN** with two witnesses and their symmetric difference; (optionally escalate to full‑TV ILP if you want the exact ledger tie‑break).
* **Infeasible:** **IIS** — training invariants contradict (or miner error); return a minimal infeasible subset.

---

## 9. Why this meets the 100% requirement

* Finite grid ⇒ finite search space; the closed, grid‑aware atom universe + “always‑true” mining captures FO+Count‑style invariants decisively.
* Ledger minimization is **exact** by construction:

  * Either **TU‑LP** is certified (signer + integrality), yielding an integral optimum under the tree objective that coincides with the full objective when Stage N closes cycles,
  * Or we **fallback to full‑TV ILP**, which exactly minimizes sum of all interface cuts (A2) with no approximation.
* Π removes minted differences, so invariants are relational and transferable across sizes.

**Short version for engineering:**
Canonicalize → scaffold (train_out) → size map (screened) → mine fixes/equalities (grid‑aware atoms; strong “always” + anti‑spurious; union‑find) → **solve TU‑LP (tree‑TV) with signer**, else **fallback to full‑TV MILP** → decode → idempotence check.

---

### Notes (trace & receipts)

* Receipts are optional: enable `trace=True` to dump canonical maps, scaffold stats, mined rule tables, TU signer report, objective, and uniqueness probe outcome.
* F24 is evaluation‑only: input features parameterize predicates of **already‑mined** laws; they never create new laws.

---

## Micro‑clarifications (executable definitions)

These lock down any remaining ambiguity for deterministic implementation:

1. **Hash sentinel:** "We use base‑11 for 3×3 and 5×5 hashes; palette 0..9; sentinel=10 for out‑of‑grid padding."

2. **Background color per training:** (k_i^{\text{bg}} = \arg\max_k \text{count}_i[k]) (on tie, smallest (k)).

3. **"Always" means all trainings, all realizations:** If a template instance (type, relation, offset) would be out‑of‑bounds for **any** training, that instance is discarded entirely.

4. **No input mining:** Input features (F24) are **only** evaluated for already‑mined rules. They **never** create rules.

5. **TU certification path:** Use the constructive incidence‑block reasoning (contract equalities per color → build (\mathbf{B}_k) → verify structure). If any structural precondition breaks, declare **TU‑FAIL** and run the full‑TV ILP.

6. **Variable ordering:** Variables indexed as ((r, c, k)) in lexicographic order with (k) sorted ascending (0..9).

7. **Idempotence enforcement:** Re‑run with ((X_{\text{in}}^{\text{test}}, X_{\text{out}}^{\text{test}})) appended to training; output must be bit‑identical.

---

## Appendix: Minimal certification policy

To guarantee "spec = code":

1. **TU signer pass** + **integral LP solution** → accept TU‑LP result.
2. Else → **run full‑TV MILP** and output that optimal coloring.
3. (Optional) **Uniqueness probe** on LP result to label AMBIGUOUS_SOLN when appropriate.

### Minimal acceptance tests

These tests can be integrated into CI or run manually:

1. **Canon repeatability:** Hash of ((\mathbf{R}_X, \mathbf{C}_X)) is constant across two runs on the same task.
2. **Rule miner invariants:** For each mined rule, a replay on all train_out re‑derives the rule byte‑identically.
3. **TU structural check:** For 50 random tasks, confirm TU‑LP path passes; else ILP fallback returns the same coloring.
4. **Idempotence:** For test outputs produced, appending to train and re‑solving yields bit‑identical output.
5. **Ambiguity harness:** Synthetic tasks with symmetric halves trigger AMBIGUOUS_SOLN (two witnesses collected).

---
