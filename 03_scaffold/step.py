"""
03_scaffold: WHERE — stable canvas geometry from train_out only.

Stage: scaffold
Computes frame, distance fields, inner region, and global structural facts.
"""

from typing import Any, Dict, List
import logging

import numpy as np


def _frame_for_output(Y: np.ndarray) -> np.ndarray:
    """
    Per-grid frame: the outer border of the canonical output grid.

    Stage F is output-intrinsic (per grid), not global across train_out.
    The border frame is purely geometric and later atoms/distance fields
    will use it. Law-level invariants like "all border cells are 8" are
    discovered in Stage N, not baked here.

    Anchors:
      - 00_MATH_SPEC.md §4.1: Frame (per-grid border)
      - 00_MATH_SPEC.md §4.2: For each output grid, build adjacency and distances
      - 01_STAGES.md: scaffold = geometry on train_out
      - 02_QUANTUM_MAPPING.md: WHERE vs WHAT separation

    Input:
      Y: canonical output grid (H×W np.ndarray)

    Output:
      frame: H×W boolean array, True at outer border positions
    """
    H, W = Y.shape
    frame = np.zeros((H, W), dtype=bool)

    if H == 0 or W == 0:
        return frame

    # Mark outer border: top/bottom rows and left/right columns
    frame[0, :] = True       # top row
    frame[H - 1, :] = True   # bottom row
    frame[:, 0] = True       # left column
    frame[:, W - 1] = True   # right column

    return frame


def _distance_fields_for_output(H: int, W: int, frame_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Directional distance fields per-grid, as shortest path lengths along
    each axis to the border frame.

    Since frame_mask marks the outer border (from WO-2.1), we use closed-form
    formulas for distances to border along each direction:
      d_top[r,c]    = r              (distance to row 0)
      d_bottom[r,c] = (H-1) - r      (distance to row H-1)
      d_left[r,c]   = c              (distance to col 0)
      d_right[r,c]  = (W-1) - c      (distance to col W-1)

    These are directional distances along axes, equivalent to 1D BFS along
    rows/cols. They match the BFS-on-4-neighbor-grid metric for the border frame.

    Anchors:
      - 00_MATH_SPEC.md §4.2: "compute distances d_top, d_bottom, d_left, d_right"
      - 01_STAGES.md: scaffold = output-intrinsic geometry
      - 02_QUANTUM_MAPPING.md: distance computation is in the free sector

    Input:
      H, W: grid shape
      frame_mask: H×W bool array (outer border, from WO-2.1)

    Output:
      dict with keys "d_top", "d_bottom", "d_left", "d_right", each H×W int array
    """
    d_top = np.zeros((H, W), dtype=int)
    d_bottom = np.zeros((H, W), dtype=int)
    d_left = np.zeros((H, W), dtype=int)
    d_right = np.zeros((H, W), dtype=int)

    # Closed-form directional distances to border
    for r in range(H):
        for c in range(W):
            d_top[r, c] = r
            d_bottom[r, c] = (H - 1) - r
            d_left[r, c] = c
            d_right[r, c] = (W - 1) - c

    # Border cells naturally get 0 for all distances (no special-casing needed)
    return {
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }


def _inner_region(d_top: np.ndarray, d_bottom: np.ndarray,
                  d_left: np.ndarray, d_right: np.ndarray) -> np.ndarray:
    """
    Inner region: cells with all four directional distances > 0.

    From 00_MATH_SPEC.md §4.2: S = {p : d_top, d_bottom, d_left, d_right > 0}.

    Input:
      d_top, d_bottom, d_left, d_right: H×W int arrays

    Output:
      inner: H×W bool array, True where all four distances are > 0
    """
    return (d_top > 0) & (d_bottom > 0) & (d_left > 0) & (d_right > 0)


def _parity_flags(d_top: np.ndarray, d_bottom: np.ndarray,
                  d_left: np.ndarray, d_right: np.ndarray) -> tuple:
    """
    Detect midrow/midcol parity flags for S0 screens.

    From 00_MATH_SPEC.md §3.2:
      - has_midrow: ∃ row r where d_top[r,:] == d_bottom[r,:] at EVERY column
      - has_midcol: ∃ col c where d_left[:,c] == d_right[:,c] at EVERY row

    Input:
      d_top, d_bottom, d_left, d_right: H×W int arrays

    Output:
      (has_midrow, has_midcol): tuple of bools
    """
    H, W = d_top.shape
    has_midrow = False
    has_midcol = False

    # Check for midrow: any row where d_top == d_bottom at every column
    for r in range(H):
        if np.all(d_top[r, :] == d_bottom[r, :]):
            has_midrow = True
            break

    # Check for midcol: any col where d_left == d_right at every row
    for c in range(W):
        if np.all(d_left[:, c] == d_right[:, c]):
            has_midcol = True
            break

    return has_midrow, has_midcol


def _thickness_for_output(inner: np.ndarray, d_top: np.ndarray,
                          d_bottom: np.ndarray, d_left: np.ndarray,
                          d_right: np.ndarray) -> Dict[str, Any]:
    """
    Compute frame thickness (min ring width from inner to frame).

    From 00_MATH_SPEC.md §3.2: minimum distance from any inner cell to the frame.

    For each inner cell, the distance to frame is min(d_top, d_bottom, d_left, d_right).
    The thickness is the minimum of these over all inner cells.

    Input:
      inner: H×W bool (inner region mask)
      d_top, d_bottom, d_left, d_right: H×W int arrays

    Output:
      {"min": int or None}
    """
    if not inner.any():
        return {"min": None}

    # For each inner cell, min distance to frame along any direction
    min_distances = np.minimum.reduce([
        d_top[inner],
        d_bottom[inner],
        d_left[inner],
        d_right[inner]
    ])

    thickness_min = int(min_distances.min())
    return {"min": thickness_min}


def _least_period(seq: np.ndarray) -> int:
    """
    Compute the least period of a 1D sequence.

    A sequence has period p if it exactly repeats a pattern of length p.

    Input:
      seq: 1D numpy array

    Output:
      p: minimal period (1 to len(seq))
    """
    n = len(seq)
    if n == 0:
        return 0

    for p in range(1, n + 1):
        if n % p != 0:
            continue
        # Check if pattern of length p repeats exactly
        pattern = seq[:p]
        if np.all(seq == np.tile(pattern, n // p)):
            return p

    return n  # Trivial period (entire sequence)


def _period_hints_for_output(Y: np.ndarray, inner: np.ndarray) -> Dict[str, Any]:
    """
    Detect minimal row/col periods inside inner region for S0 screens.

    From 00_MATH_SPEC.md §3.2:
      - If all rows (with inner cells) share the same least period p ≥ 2, set row_period = p
      - Otherwise row_period = None
      - Same for columns

    Input:
      Y: H×W canonical output grid (color values)
      inner: H×W bool (inner region mask)

    Output:
      {"row_period": int or None, "col_period": int or None}
    """
    H, W = Y.shape

    # Row periods (real repetition only)
    row_periods = []
    for r in range(H):
        if inner[r, :].any():
            # Extract row segment inside inner region
            row_vals = Y[r, inner[r, :]]
            if len(row_vals) > 0:
                p = _least_period(row_vals)
                # Only count as real period if it repeats at least twice
                # i.e., p >= 2 AND 2*p <= len(sequence)
                # If p == len, it's not a repeating pattern, just a sequence appearing once
                if p >= 2 and 2 * p <= len(row_vals):
                    row_periods.append(p)

    # If all rows with inner cells share same period, use it
    if row_periods and len(set(row_periods)) == 1:
        row_period = row_periods[0]
    else:
        row_period = None

    # Column periods (real repetition only)
    col_periods = []
    for c in range(W):
        if inner[:, c].any():
            # Extract col segment inside inner region
            col_vals = Y[inner[:, c], c]
            if len(col_vals) > 0:
                p = _least_period(col_vals)
                # Only count as real period if it repeats at least twice
                # i.e., p >= 2 AND 2*p <= len(sequence)
                # If p == len, it's not a repeating pattern, just a sequence appearing once
                if p >= 2 and 2 * p <= len(col_vals):
                    col_periods.append(p)

    # If all cols with inner cells share same period, use it
    if col_periods and len(set(col_periods)) == 1:
        col_period = col_periods[0]
    else:
        col_period = None

    return {"row_period": row_period, "col_period": col_period}


def _aggregate_scaffold(per_output: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate global hints from per-output scaffolds for S0 screening.

    From 00_MATH_SPEC.md §3.2:
      - thickness_min: min over non-None thickness values
      - row_period: only if all non-None values agree
      - col_period: only if all non-None values agree
      - has_midrow_all: True iff ALL outputs have midrow
      - has_midcol_all: True iff ALL outputs have midcol

    Input:
      per_output: list of per-output scaffold entries

    Output:
      aggregated hints dict
    """
    # Thickness: min over non-None values
    thickness_values = [
        entry["thickness"]["min"]
        for entry in per_output
        if entry["thickness"]["min"] is not None
    ]
    thickness_min = min(thickness_values) if thickness_values else None

    # Row period: only if all non-None values are equal
    row_period_values = [
        entry["row_period"]
        for entry in per_output
        if entry["row_period"] is not None
    ]
    if row_period_values and len(set(row_period_values)) == 1:
        row_period = row_period_values[0]
    else:
        row_period = None

    # Col period: only if all non-None values are equal
    col_period_values = [
        entry["col_period"]
        for entry in per_output
        if entry["col_period"] is not None
    ]
    if col_period_values and len(set(col_period_values)) == 1:
        col_period = col_period_values[0]
    else:
        col_period = None

    # Midrow/midcol: True iff ALL outputs have it
    has_midrow_all = all(entry["has_midrow"] for entry in per_output)
    has_midcol_all = all(entry["has_midcol"] for entry in per_output)

    return {
        "thickness_min": thickness_min,
        "row_period": row_period,
        "col_period": col_period,
        "has_midrow_all": has_midrow_all,
        "has_midcol_all": has_midcol_all,
    }


def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — WO-2.1 + WO-2.2 + WO-2.3

    Anchors:
      - 00_MATH_SPEC.md §4: Stage F — Frame & distances (per-grid scaffold geometry)
      - 00_MATH_SPEC.md §3.2: Structural disambiguation (midrow, thickness, periods)
      - 01_STAGES.md: scaffold
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold

    Input:
      canonical: object from 02_truth.canonicalize, containing:
        - train_out: List[np.ndarray] (canonical output grids)
        - other fields (not used in WO-2.1/2.2/2.3)
      trace: enable debug logging if True

    Output:
      scaffold: {
        "per_output": [
          {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": H_i×W_i bool (outer border),
            "d_top":    H_i×W_i int,
            "d_bottom": H_i×W_i int,
            "d_left":   H_i×W_i int,
            "d_right":  H_i×W_i int,
            "inner":    H_i×W_i bool (WO-2.3),
            "has_midrow": bool (WO-2.3),
            "has_midcol": bool (WO-2.3),
            "thickness": {"min": int or None} (WO-2.3),
            "row_period": int or None (WO-2.3),
            "col_period": int or None (WO-2.3),
          },
          ...
        ],
        "aggregated": {
          "thickness_min": int or None,
          "row_period": int or None,
          "col_period": int or None,
          "has_midrow_all": bool,
          "has_midcol_all": bool,
        }
      }

    WO-2.1: per-grid border frames.
    WO-2.2: directional distance fields per-grid.
    WO-2.3: inner region, parity flags, thickness, periods, aggregated hints.
    """
    if trace:
        logging.info("[scaffold] build() called (WO-2.1+WO-2.2+WO-2.3: complete scaffold)")

    train_out: List[np.ndarray] = canonical["train_out"]

    if not train_out:
        msg = "[scaffold] No train_out grids; scaffold undefined."
        if trace:
            logging.error(msg)
        raise ValueError(msg)

    # Build per-output scaffold entries
    per_output: List[Dict[str, Any]] = []

    for i, Y in enumerate(train_out):
        H, W = Y.shape

        # WO-2.1: Frame (outer border)
        frame_mask = _frame_for_output(Y)

        # WO-2.2: Distance fields (directional to border)
        distances = _distance_fields_for_output(H, W, frame_mask)
        d_top = distances["d_top"]
        d_bottom = distances["d_bottom"]
        d_left = distances["d_left"]
        d_right = distances["d_right"]

        # WO-2.3: Inner region
        inner = _inner_region(d_top, d_bottom, d_left, d_right)

        # WO-2.3: Parity flags
        has_midrow, has_midcol = _parity_flags(d_top, d_bottom, d_left, d_right)

        # WO-2.3: Thickness
        thickness = _thickness_for_output(inner, d_top, d_bottom, d_left, d_right)

        # WO-2.3: Period hints
        period_hints = _period_hints_for_output(Y, inner)

        entry = {
            "index": i,
            "shape": (H, W),
            "frame_mask": frame_mask,
            "d_top": d_top,
            "d_bottom": d_bottom,
            "d_left": d_left,
            "d_right": d_right,
            "inner": inner,
            "has_midrow": has_midrow,
            "has_midcol": has_midcol,
            "thickness": thickness,
            "row_period": period_hints["row_period"],
            "col_period": period_hints["col_period"],
        }

        if trace:
            frame_sum = int(frame_mask.sum())
            inner_sum = int(inner.sum())
            logging.info(
                f"[scaffold] output#{i}: shape={entry['shape']}, "
                f"frame_sum={frame_sum}, inner_sum={inner_sum}"
            )
            # Log distance field stats for verification
            for name in ["d_top", "d_bottom", "d_left", "d_right"]:
                D = entry[name]
                logging.info(
                    f"[scaffold] output#{i} {name}: "
                    f"min={int(D.min())}, max={int(D.max())}, sum={int(D.sum())}"
                )
            # Log WO-2.3 facts
            logging.info(
                f"[scaffold] output#{i} has_midrow={has_midrow}, "
                f"has_midcol={has_midcol}, "
                f"thickness_min={thickness['min']}, "
                f"row_period={period_hints['row_period']}, "
                f"col_period={period_hints['col_period']}"
            )

        per_output.append(entry)

    # WO-2.3: Aggregate global hints for S0
    aggregated = _aggregate_scaffold(per_output)

    if trace:
        logging.info(
            f"[scaffold] aggregated: "
            f"thickness_min={aggregated['thickness_min']}, "
            f"row_period={aggregated['row_period']}, "
            f"col_period={aggregated['col_period']}, "
            f"has_midrow_all={aggregated['has_midrow_all']}, "
            f"has_midcol_all={aggregated['has_midcol_all']}"
        )

    scaffold: Dict[str, Any] = {
        "per_output": per_output,
        "aggregated": aggregated,
    }

    return scaffold
