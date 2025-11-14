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


def _distance_fields(
    frame_mask: np.ndarray, has_frame: bool
) -> Dict[str, np.ndarray]:
    """
    Compute directional distance fields to frame or border.

    Anchor:
      - 00_MATH_SPEC.md §4.2: Distance fields d_top, d_bottom, d_left, d_right

    Args:
      frame_mask: bool array (H×W), True at frame positions
      has_frame: whether any frame exists

    Returns:
      dict with keys "d_top", "d_bottom", "d_left", "d_right", each np.ndarray[int] (H×W)
    """
    H, W = frame_mask.shape

    d_top = np.zeros((H, W), dtype=int)
    d_bottom = np.zeros((H, W), dtype=int)
    d_left = np.zeros((H, W), dtype=int)
    d_right = np.zeros((H, W), dtype=int)

    if not has_frame:
        # Distances to borders only (simple formulas)
        for r in range(H):
            for c in range(W):
                d_top[r, c] = r
                d_bottom[r, c] = H - 1 - r
                d_left[r, c] = c
                d_right[r, c] = W - 1 - c
        return {
            "d_top": d_top,
            "d_bottom": d_bottom,
            "d_left": d_left,
            "d_right": d_right,
        }

    # With frame: directional scans with reset at frame cells

    # Top-down: d_top
    for c in range(W):
        dist = 0
        for r in range(H):
            if frame_mask[r, c]:
                dist = 0
            elif r == 0:
                dist = 0  # border
            else:
                dist += 1
            d_top[r, c] = dist

    # Bottom-up: d_bottom
    for c in range(W):
        dist = 0
        for r in reversed(range(H)):
            if frame_mask[r, c]:
                dist = 0
            elif r == H - 1:
                dist = 0  # border
            else:
                dist += 1
            d_bottom[r, c] = dist

    # Left-right: d_left
    for r in range(H):
        dist = 0
        for c in range(W):
            if frame_mask[r, c]:
                dist = 0
            elif c == 0:
                dist = 0  # border
            else:
                dist += 1
            d_left[r, c] = dist

    # Right-left: d_right
    for r in range(H):
        dist = 0
        for c in reversed(range(W)):
            if frame_mask[r, c]:
                dist = 0
            elif c == W - 1:
                dist = 0  # border
            else:
                dist += 1
            d_right[r, c] = dist

    return {
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }


def build(canonical: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: scaffold (WHERE) — frame + distances (WO-2.1 + WO-2.2)

    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4.1-4.2: Frame & distance fields
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic geometry

    Input:
      canonical: dict from 02_truth.canonicalize with canonical train_out grids
      trace: enable debug logging if True

    Output:
      scaffold: dict containing:
        - frame_mask: np.ndarray[bool] (H×W), True where all train_out identical
        - train_out_shapes: List[(H,W)], shapes of each train_out grid
        - has_frame: bool, True if any position has identical color across all train_out
        - d_top: np.ndarray[int] (H×W), distance to frame/border upward
        - d_bottom: np.ndarray[int] (H×W), distance to frame/border downward
        - d_left: np.ndarray[int] (H×W), distance to frame/border leftward
        - d_right: np.ndarray[int] (H×W), distance to frame/border rightward

      Note: Inner region and global facts will be added in WO-2.3.
    """
    if trace:
        logging.info("[scaffold] building output-intrinsic scaffold")

    train_out = canonical["train_out"]  # List[np.ndarray] in canonical coords

    # WO-2.1: Detect frame mask
    frame_mask, shapes = _detect_frame(train_out, trace=trace)
    has_frame = bool(frame_mask.any())

    # WO-2.2: Compute distance fields
    distances = _distance_fields(frame_mask, has_frame)

    scaffold = {
        "frame_mask": frame_mask,
        "train_out_shapes": shapes,
        "has_frame": has_frame,
        "d_top": distances["d_top"],
        "d_bottom": distances["d_bottom"],
        "d_left": distances["d_left"],
        "d_right": distances["d_right"],
    }

    if trace:
        logging.info(
            f"[scaffold] frame_mask shape={frame_mask.shape}, "
            f"sum={int(frame_mask.sum())}, has_frame={has_frame}"
        )
        for name in ["d_top", "d_bottom", "d_left", "d_right"]:
            D = distances[name]
            logging.info(
                f"[scaffold] {name}: shape={D.shape}, min={int(D.min())}, "
                f"max={int(D.max())}, sum={int(D.sum())}"
            )

    return scaffold
