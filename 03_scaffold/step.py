"""
03_scaffold: WHERE — stable canvas geometry from train_out only.

Stage: scaffold
Computes frame, distance fields, inner region, and global structural facts.
"""

from typing import Any, Dict, List, Tuple
import logging

import numpy as np


def _detect_frame(
    train_out: List[np.ndarray], trace: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Detect frame mask: positions with identical color across all train_out.

    Anchor:
      - 00_MATH_SPEC.md §4.1: F = {p : exists k, c_out^(i)(p)=k for all i}

    Args:
      train_out: list of canonical train_out grids (np.ndarray)
      trace: enable logging if True

    Returns:
      frame_mask: bool array (H×W), True where all train_out have same color
      shapes: list of (H,W) tuples per train_out grid
    """
    shapes = [g.shape for g in train_out]
    unique_shapes = sorted(set(shapes))

    if len(unique_shapes) != 1:
        # No common positions if train_out shapes differ
        if trace:
            logging.warning(
                f"[scaffold] train_out shapes differ: {unique_shapes}; "
                "global frame_mask set to all False (no common positions)."
            )
        # Use first shape as reference
        H, W = shapes[0]
        return np.zeros((H, W), dtype=bool), shapes

    H, W = unique_shapes[0]
    # Stack into 3D array: T × H × W
    stack = np.stack(train_out, axis=0)  # shape: (T, H, W)

    # Check if all entries equal at each (r,c) position
    base = stack[0]
    equal_all = np.all(stack == base, axis=0)  # H×W bool array

    frame_mask = equal_all.astype(bool)

    return frame_mask, shapes


def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — frame detector (WO-2.1)

    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4.1: Frame detection from training outputs
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold (train_out-only)

    Input:
      canonical: dict from 02_truth.canonicalize with canonical train_out grids
      trace: enable debug logging if True

    Output:
      scaffold: dict containing:
        - frame_mask: np.ndarray[bool] (H×W), True where all train_out identical
        - train_out_shapes: List[(H,W)], shapes of each train_out grid
        - has_frame: bool, True if any position has identical color across all train_out

      Note: Distance fields and inner region will be added in WO-2.2/2.3.
    """
    if trace:
        logging.info("[scaffold] building output-intrinsic scaffold")

    train_out = canonical["train_out"]  # List[np.ndarray] in canonical coords

    # WO-2.1: Detect frame mask
    frame_mask, shapes = _detect_frame(train_out, trace=trace)
    has_frame = bool(frame_mask.any())

    scaffold = {
        "frame_mask": frame_mask,
        "train_out_shapes": shapes,
        "has_frame": has_frame,
    }

    if trace:
        logging.info(
            f"[scaffold] frame_mask shape={frame_mask.shape}, "
            f"sum={int(frame_mask.sum())}, has_frame={has_frame}"
        )

    return scaffold
