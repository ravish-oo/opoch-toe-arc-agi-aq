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


def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — WO-2.1 Frame detector (per-output, border-based)

    Anchors:
      - 00_MATH_SPEC.md §4: Stage F — Frame & distances (per-grid scaffold geometry)
      - 01_STAGES.md: scaffold
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold

    Input:
      canonical: object from 02_truth.canonicalize, containing:
        - train_out: List[np.ndarray] (canonical output grids)
        - other fields (not used in WO-2.1)
      trace: enable debug logging if True

    Output:
      scaffold: {
        "per_output": [
          {
            "index": i,
            "shape": (H_i, W_i),
            "frame_mask": H_i×W_i bool (outer border),
          },
          ...
        ]
      }

    WO-2.1 only creates per-grid border frames.
    WO-2.2 will add distance fields.
    WO-2.3 will add inner region, parity flags, thickness, periods, and aggregated hints.
    """
    if trace:
        logging.info("[scaffold] build() called (WO-2.1: per-output frame detector)")

    train_out: List[np.ndarray] = canonical["train_out"]

    if not train_out:
        msg = "[scaffold] No train_out grids; scaffold undefined."
        if trace:
            logging.error(msg)
        raise ValueError(msg)

    # Build per-output scaffold entries (WO-2.1: only frame_mask)
    per_output: List[Dict[str, Any]] = []

    for i, Y in enumerate(train_out):
        H, W = Y.shape
        frame_mask = _frame_for_output(Y)

        entry = {
            "index": i,
            "shape": (H, W),
            "frame_mask": frame_mask,
        }

        if trace:
            frame_sum = int(frame_mask.sum())
            logging.info(
                f"[scaffold] output#{i}: shape={entry['shape']}, "
                f"frame_sum={frame_sum}"
            )

        per_output.append(entry)

    scaffold: Dict[str, Any] = {
        "per_output": per_output,
    }

    return scaffold
