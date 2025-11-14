"""
05_laws: WHAT — derive atom types and mine invariants from train_out.

Stage: laws
Promotes "always true" facts into linear constraints (fixes, equalities, forbids).
"""

from typing import Any, Dict, List
import logging

from .atoms import (
    compute_A_atoms, trace_A_atoms,
    compute_B_atoms, trace_B_atoms,
    compute_C_atoms, trace_C_atoms,
    compute_D_atoms, compute_E_atoms_for_grid,
    compute_global_palette_mapping, trace_D_E_atoms,
    compute_G_atoms, trace_G_atoms,
    compute_type_keys_for_grid, trace_type_keys
)


def mine(canonical: Any, scaffold: Any, out_size: Any, trace: bool = False) -> Any:
    """
    Stage: laws (N) — WO-4.1/4.2/4.3/4.4/4.6/5.1: compute A+B+C+D+E+G+type_keys atoms for train_out

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
      For WO-4.1/4.2/4.3/4.4/4.6/5.1: returns A+B+C+D+E+G+type_keys atoms for each train_out;
      mining not yet implemented.
    """
    if trace:
        logging.info("[laws] mine() called (WO-4.1/4.2/4.3/4.4/4.6/5.1: A+B+C+D+E+G+type_keys atoms)")

    # WO-4.1/4.2/4.3/4.4/4.6/5.1: Compute A+B+C+D+E+G+type_keys atoms for each train_out
    # Later WOs will add actual invariant mining
    train_out_A_atoms: List[Dict[str, Any]] = []
    train_out_B_atoms: List[Dict[str, Any]] = []
    train_out_C_atoms: List[Dict[str, Any]] = []
    train_out_D_atoms: List[Dict[str, Any]] = []
    train_out_E_atoms: List[Dict[str, Any]] = []
    train_out_G_atoms: List[Dict[str, Any]] = []
    train_out_type_keys: List[Dict[str, Any]] = []

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

        # WO-4.3: Compute C-atoms for this train_out grid
        C_atoms = compute_C_atoms(grid, scaffold_info)
        train_out_C_atoms.append(C_atoms)

        # WO-4.4: Compute D-atoms for this train_out grid
        D_atoms = compute_D_atoms(grid)
        train_out_D_atoms.append(D_atoms)

        # WO-4.4: Compute E-atoms for this train_out grid
        E_atoms = compute_E_atoms_for_grid(grid, C_atoms)
        train_out_E_atoms.append(E_atoms)

        # WO-4.6: Compute G-atoms for this train_out grid
        G_atoms = compute_G_atoms(grid, C_atoms)
        train_out_G_atoms.append(G_atoms)

        # WO-5.1: Compute type keys for this train_out grid
        type_keys = compute_type_keys_for_grid(A_atoms, B_atoms, D_atoms, G_atoms)
        train_out_type_keys.append(type_keys)

        # Trace for first grid if requested
        if trace and i == 0:
            logging.info(f"[laws] train_out#{i} A-atoms:")
            trace_A_atoms(A_atoms)
            logging.info(f"[laws] train_out#{i} B-atoms:")
            trace_B_atoms(B_atoms, grid)
            logging.info(f"[laws] train_out#{i} C-atoms:")
            trace_C_atoms(C_atoms, grid)
            logging.info(f"[laws] train_out#{i} D+E-atoms:")
            trace_D_E_atoms(D_atoms, E_atoms, grid)
            logging.info(f"[laws] train_out#{i} G-atoms:")
            trace_G_atoms(G_atoms, grid)
            logging.info(f"[laws] train_out#{i} Type keys:")
            trace_type_keys(type_keys)

    # WO-4.4: Compute task-level global palette mapping
    train_in_grids = canonical["train_in"]
    global_palette_mapping = compute_global_palette_mapping(train_in_grids, train_out_grids)

    if trace:
        logging.info("[laws] Global palette mapping:")
        trace_D_E_atoms({}, {}, train_out_grids[0], global_map=global_palette_mapping)

    # WO-4.1/4.2/4.3/4.4/4.6/5.1: Return partial result (just A+B+C+D+E+G+type_keys atoms)
    # Later WOs will compute test_out atoms and mine invariants
    result = {
        "train_out_A_atoms": train_out_A_atoms,
        "train_out_B_atoms": train_out_B_atoms,
        "train_out_C_atoms": train_out_C_atoms,
        "train_out_D_atoms": train_out_D_atoms,
        "train_out_E_atoms": train_out_E_atoms,
        "train_out_G_atoms": train_out_G_atoms,
        "train_out_type_keys": train_out_type_keys,
        "global_palette_mapping": global_palette_mapping,
        "invariants": None,  # Not yet implemented
    }

    if trace:
        logging.info(f"[laws] computed A+B+C+D+E+G+type_keys atoms for {len(train_out_A_atoms)} train_out grids")

    return result
