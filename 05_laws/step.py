"""
05_laws: WHAT — derive atom types and mine invariants from train_out.

Stage: laws
Promotes "always true" facts into linear constraints (fixes, equalities, forbids).
"""

from typing import Any, Dict, List
import logging

from .atoms import compute_A_atoms, trace_A_atoms


def mine(canonical: Any, scaffold: Any, out_size: Any, trace: bool = False) -> Any:
    """
    Stage: laws (N) — WO-4.1: compute A-atoms for train_out

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
      For WO-4.1: returns A-atoms for each train_out; mining not yet implemented.
    """
    if trace:
        logging.info("[laws] mine() called (WO-4.1: A-atoms only)")

    # WO-4.1: Compute A-atoms for each train_out
    # Later WOs will add B–G atoms and actual invariant mining
    train_out_A_atoms: List[Dict[str, Any]] = []

    per_output = scaffold["per_output"]

    for i, scaffold_info in enumerate(per_output):
        H, W = scaffold_info["shape"]

        # Compute A-atoms for this train_out grid
        A_atoms = compute_A_atoms(H, W, scaffold_info)

        train_out_A_atoms.append(A_atoms)

        # Trace for first 2 grids if requested
        if trace and i < 2:
            logging.info(f"[laws] train_out#{i} A-atoms:")
            trace_A_atoms(A_atoms)

    # WO-4.1: Return partial result (just A-atoms)
    # Later WOs will compute test_out A-atoms and mine invariants
    result = {
        "train_out_A_atoms": train_out_A_atoms,
        "invariants": None,  # Not yet implemented
    }

    if trace:
        logging.info(f"[laws] computed A-atoms for {len(train_out_A_atoms)} train_out grids")

    return result
