"""
05_laws: WHAT — derive atom types and mine invariants from train_out.

Stage: laws
Promotes "always true" facts into linear constraints (fixes, equalities, forbids).
"""

from typing import Any, Dict, List
import logging

from .atoms import compute_A_atoms, trace_A_atoms, compute_B_atoms, trace_B_atoms


def mine(canonical: Any, scaffold: Any, out_size: Any, trace: bool = False) -> Any:
    """
    Stage: laws (N) — WO-4.1 + WO-4.2: compute A+B atoms for train_out

    Anchor:
      - 01_STAGES.md: laws
      - 00_MATH_SPEC.md §5: Stage N — Invariants as linear constraints
      - 02_QUANTUM_MAPPING.md: WHAT = law nucleus over scaffold

    Input:
      canonical: from 02_truth.canonicalize
      scaffold: from 03_scaffold.build
      out_size: dict with status, H_out, W_out from 04_size_choice.choose
      trace: enable debug logging if True.

    Output:
      Invariants object encoding fixes, equalities, forbids, etc.
      For WO-4.1+4.2: returns A+B atoms for each train_out; mining not yet implemented.
    """
    if trace:
        logging.info("[laws] mine() called (WO-4.1+WO-4.2: A+B atoms)")

    # WO-4.1+4.2: Compute A+B atoms for each train_out
    # Later WOs will add C–G atoms and actual invariant mining
    train_out_A_atoms: List[Dict[str, Any]] = []
    train_out_B_atoms: List[Dict[str, Any]] = []

    per_output = scaffold["per_output"]
    train_out_grids = canonical["train_out"]

    for i, scaffold_info in enumerate(per_output):
        H, W = scaffold_info["shape"]
        grid = train_out_grids[i]

        # WO-4.1: Compute A-atoms for this train_out grid
        A_atoms = compute_A_atoms(H, W, scaffold_info)
        train_out_A_atoms.append(A_atoms)

        # WO-4.2: Compute B-atoms for this train_out grid
        B_atoms = compute_B_atoms(grid)
        train_out_B_atoms.append(B_atoms)

        # Trace for first grid if requested
        if trace and i == 0:
            logging.info(f"[laws] train_out#{i} A-atoms:")
            trace_A_atoms(A_atoms)
            logging.info(f"[laws] train_out#{i} B-atoms:")
            trace_B_atoms(B_atoms, grid)

    # WO-4.1+4.2: Return partial result (just A+B atoms)
    # Later WOs will compute test_out atoms and mine invariants
    result = {
        "train_out_A_atoms": train_out_A_atoms,
        "train_out_B_atoms": train_out_B_atoms,
        "invariants": None,  # Not yet implemented
    }

    if trace:
        logging.info(f"[laws] computed A+B atoms for {len(train_out_A_atoms)} train_out grids")

    return result
