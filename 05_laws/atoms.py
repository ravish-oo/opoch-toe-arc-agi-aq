"""
05_laws/atoms.py — A-atoms: Coordinates & distances (WO-4.1)

Anchor:
  - 00_MATH_SPEC.md §5.1: Atom universe (A. Scaffold & coords)

Implements:
  - Basic coords (H,W,r,c,r±c)
  - Distance fields (reused from scaffold)
  - Midrow/midcol flags
  - Grid-aware mod classes
  - Grid-aware block coords
"""

import numpy as np
from typing import Dict, Any, Optional


def _divisors(n: int) -> np.ndarray:
    """
    Compute all divisors of n (including 1 and n).

    For H,W <= 30 (ARC grids), simple iteration is fine.

    Returns:
        np.ndarray of divisors in ascending order
    """
    if n < 1:
        raise ValueError(f"Cannot compute divisors of {n} (must be >= 1)")

    divs = []
    for d in range(1, n + 1):
        if n % d == 0:
            divs.append(d)

    return np.array(divs, dtype=int)


def compute_A_atoms(
    H: int,
    W: int,
    scaffold_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute A-atoms (coordinates & distances) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 A: Scaffold & coords

    Input:
      H, W: grid dimensions in canonical coords
      scaffold_info: per-output dict from 03_scaffold containing:
        - d_top, d_bottom, d_left, d_right: np.ndarray[(H,W), int]
        - has_midrow, has_midcol: bool (not used; we recompute from distances)

    Output:
      Dict containing:
        - H, W: int scalars
        - r, c: np.ndarray[(H,W), int] coordinate grids
        - r_plus_c, r_minus_c: np.ndarray[(H,W), int]
        - d_top/bottom/left/right: np.ndarray[(H,W), int] (from scaffold)
        - midrow_flag, midcol_flag: np.ndarray[(H,W), bool]
        - mod_r, mod_c: Dict[int, np.ndarray[(H,W), int]]
        - block_row, block_col: Dict[int, np.ndarray[(H,W), int]]

    Raises:
      NotImplementedError: if scaffold_info is None (test_out not yet supported)
      ValueError: if distance field shapes don't match (H,W)
    """
    # WO-4.1: Fail loudly if scaffold_info missing (test_out case)
    if scaffold_info is None:
        raise NotImplementedError(
            "A-atoms for test_out require scaffold distances; "
            "build them first in 03_scaffold."
        )

    # ========== 3.1 Basic coord arrays ==========
    r = np.arange(H, dtype=int)[:, None]  # shape (H,1)
    c = np.arange(W, dtype=int)[None, :]  # shape (1,W)

    r_grid = np.broadcast_to(r, (H, W)).copy()
    c_grid = np.broadcast_to(c, (H, W)).copy()

    r_plus_c = r_grid + c_grid
    r_minus_c = r_grid - c_grid

    # ========== 3.2 Distances (reuse from scaffold) ==========
    d_top = np.asarray(scaffold_info["d_top"], dtype=int)
    d_bottom = np.asarray(scaffold_info["d_bottom"], dtype=int)
    d_left = np.asarray(scaffold_info["d_left"], dtype=int)
    d_right = np.asarray(scaffold_info["d_right"], dtype=int)

    # Validate shapes
    if d_top.shape != (H, W):
        raise ValueError(
            f"d_top shape {d_top.shape} does not match grid shape ({H},{W})"
        )
    if d_bottom.shape != (H, W):
        raise ValueError(
            f"d_bottom shape {d_bottom.shape} does not match grid shape ({H},{W})"
        )
    if d_left.shape != (H, W):
        raise ValueError(
            f"d_left shape {d_left.shape} does not match grid shape ({H},{W})"
        )
    if d_right.shape != (H, W):
        raise ValueError(
            f"d_right shape {d_right.shape} does not match grid shape ({H},{W})"
        )

    # ========== 3.3 Midrow / midcol flags ==========
    # Spec: midrow cells are where d_top == d_bottom
    #       midcol cells are where d_left == d_right
    midrow_flag = (d_top == d_bottom)
    midcol_flag = (d_left == d_right)

    # ========== 4.1 Mod classes: exact m set ==========
    # Spec §5.1 A: m ∈ {2,…,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)
    max_dim = max(H, W)
    base_ms = list(range(2, min(6, max_dim) + 1))

    div_H = _divisors(H)
    div_W = _divisors(W)

    m_set = set(base_ms)
    m_set.update(div_H.tolist())
    m_set.update(div_W.tolist())
    m_set = {m for m in m_set if m >= 2}  # filter out mod 1 (useless)

    mod_r = {}
    mod_c = {}

    for m in sorted(m_set):
        mod_r[m] = r_grid % m
        mod_c[m] = c_grid % m

    # ========== 4.2 Block coords: exact b set ==========
    # Spec §5.1 A: b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W)
    # divisors(H,W) = divisors that divide BOTH H and W
    min_dim = min(H, W)
    base_bs = list(range(2, min(5, min_dim) + 1))

    div_H = _divisors(H)
    div_W = _divisors(W)
    div_HW = np.intersect1d(div_H, div_W)

    b_set = set(base_bs)
    b_set.update(div_HW.tolist())
    b_set = {b for b in b_set if b >= 2}  # filter out block size 1 (useless)

    block_row = {}
    block_col = {}

    for b in sorted(b_set):
        block_row[b] = r_grid // b
        block_col[b] = c_grid // b

    # ========== Return A-atoms dict ==========
    return {
        "H": H,
        "W": W,
        "r": r_grid,
        "c": c_grid,
        "r_plus_c": r_plus_c,
        "r_minus_c": r_minus_c,
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
        "midrow_flag": midrow_flag,
        "midcol_flag": midcol_flag,
        "mod_r": mod_r,
        "mod_c": mod_c,
        "block_row": block_row,
        "block_col": block_col,
    }


def trace_A_atoms(A_atoms: Dict[str, Any]) -> None:
    """
    Print trace summary of A-atoms for debugging.

    Called from 05_laws/step.py when trace=True.
    """
    H = A_atoms["H"]
    W = A_atoms["W"]

    print(f"[A-atoms] H,W = {H},{W}")
    print(f"[A-atoms] r_plus_c range: [{A_atoms['r_plus_c'].min()}, {A_atoms['r_plus_c'].max()}]")
    print(f"[A-atoms] r_minus_c range: [{A_atoms['r_minus_c'].min()}, {A_atoms['r_minus_c'].max()}]")
    print(f"[A-atoms] midrow_flag cells: {A_atoms['midrow_flag'].sum()}")
    print(f"[A-atoms] midcol_flag cells: {A_atoms['midcol_flag'].sum()}")
    print(f"[A-atoms] mod m keys: {sorted(A_atoms['mod_r'].keys())}")
    print(f"[A-atoms] block b keys: {sorted(A_atoms['block_row'].keys())}")
