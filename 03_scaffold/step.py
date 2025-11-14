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


def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — WO-2.1 + WO-2.2

    Anchors:
      - 00_MATH_SPEC.md §4: Stage F — Frame & distances (per-grid scaffold geometry)
      - 01_STAGES.md: scaffold
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold

    Input:
      canonical: object from 02_truth.canonicalize, containing:
        - train_out: List[np.ndarray] (canonical output grids)
        - other fields (not used in WO-2.1/2.2)
      trace: enable debug logging if True

    Output:
      scaffold: {
        "per_output": [
          {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": H_i×W_i bool (outer border),
            "d_top":    H_i×W_i int (WO-2.2),
            "d_bottom": H_i×W_i int (WO-2.2),
            "d_left":   H_i×W_i int (WO-2.2),
            "d_right":  H_i×W_i int (WO-2.2),
          },
          ...
        ]
      }

    WO-2.1: per-grid border frames.
    WO-2.2: directional distance fields per-grid.
    WO-2.3 will add: inner region, parity flags, thickness, periods, aggregated hints.
    """
    if trace:
        logging.info("[scaffold] build() called (WO-2.1+WO-2.2: frame + distances)")

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

        entry = {
            "index": i,
            "shape": (H, W),
            "frame_mask": frame_mask,
            "d_top": distances["d_top"],
            "d_bottom": distances["d_bottom"],
            "d_left": distances["d_left"],
            "d_right": distances["d_right"],
        }

        if trace:
            frame_sum = int(frame_mask.sum())
            logging.info(
                f"[scaffold] output#{i}: shape={entry['shape']}, "
                f"frame_sum={frame_sum}"
            )
            # Log distance field stats for verification
            for name in ["d_top", "d_bottom", "d_left", "d_right"]:
                D = entry[name]
                logging.info(
                    f"[scaffold] output#{i} {name}: "
                    f"min={int(D.min())}, max={int(D.max())}, sum={int(D.sum())}"
                )

        per_output.append(entry)

    scaffold: Dict[str, Any] = {
        "per_output": per_output,
    }

    return scaffold
