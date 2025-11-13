"""
01_present: Load everything into awareness.

Stage: present
Loads ARC task JSON into normalized structures (train_in, train_out, test_in).
"""

from typing import Any, Dict, List
import logging
import numpy as np


def _to_grid(arr_like) -> np.ndarray:
    """
    Convert a list-of-lists to a validated numpy grid.

    Validates:
    - Must be 2D
    - H, W must be in range [1, 30]
    - All colors must be in range [0, 9]

    Returns:
        np.ndarray[int8] of shape (H, W)

    Raises:
        ValueError if validation fails
    """
    g = np.array(arr_like, dtype=np.int8)

    if g.ndim != 2:
        raise ValueError(f"Grid must be 2D, got {g.ndim}D")

    H, W = g.shape
    if H <= 0 or W <= 0 or H > 30 or W > 30:
        raise ValueError(f"Invalid grid size H={H}, W={W} (must be 1..30)")

    if g.min() < 0 or g.max() > 9:
        raise ValueError(f"Palette out of range: min={g.min()}, max={g.max()} (must be 0..9)")

    return g


def _shapes_and_palette(grids: List[np.ndarray]):
    """
    Extract shapes and palettes from a list of grids.

    Args:
        grids: list of np.ndarray grids

    Returns:
        (shapes, palettes) where:
        - shapes: [[H, W], ...] for each grid
        - palettes: [[sorted colors], ...] for each grid
    """
    shapes = []
    palettes = []

    for g in grids:
        H, W = g.shape
        shapes.append([int(H), int(W)])

        colors = np.unique(g)
        palettes.append(sorted(int(c) for c in colors.tolist()))

    return shapes, palettes


def load(task_bundle: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: present (awareness)

    Anchor:
      - 01_STAGES.md: present
      - 00_MATH_SPEC.md §1: Representing the task
      - 02_QUANTUM_MAPPING.md: 'present' = load all grids into awareness

    Input:
      task_bundle: {
        "task_id": str,
        "raw_task": dict from arc-agi_training_challenges.json
      }
      trace: if True, log shapes and palettes.

    Output:
      present: {
        "task_id": str,
        "train_in":  [np.ndarray[int8], ...],
        "train_out": [np.ndarray[int8], ...],
        "test_in":   [np.ndarray[int8], ...],
        "shapes": {
          "train_in":  [[H, W], ...],
          "train_out": [[H, W], ...],
          "test_in":   [H, W]  # single test input
        },
        "palettes": {
          "train_in":  [[sorted colors], ...],
          "train_out": [[sorted colors], ...],
          "test_in":   [[sorted colors], ...]
        }
      }
    """
    task_id = task_bundle["task_id"]
    raw_task = task_bundle["raw_task"]

    if trace:
        logging.info(f"[present] load() called for task_id={task_id}")

    # Parse training pairs
    train_in_arrays = []
    train_out_arrays = []
    for pair in raw_task["train"]:
        g_in = _to_grid(pair["input"])
        g_out = _to_grid(pair["output"])
        train_in_arrays.append(g_in)
        train_out_arrays.append(g_out)

    # Parse test inputs
    test_in_arrays = []
    for t in raw_task["test"]:
        g_test = _to_grid(t["input"])
        test_in_arrays.append(g_test)

    # Extract shapes and palettes
    shapes_train_in, palettes_train_in = _shapes_and_palette(train_in_arrays)
    shapes_train_out, palettes_train_out = _shapes_and_palette(train_out_arrays)
    shapes_test_in, palettes_test_in = _shapes_and_palette(test_in_arrays)

    # Build present object
    present = {
        "task_id": task_id,
        "train_in": train_in_arrays,
        "train_out": train_out_arrays,
        "test_in": test_in_arrays,
        "shapes": {
            "train_in": shapes_train_in,
            "train_out": shapes_train_out,
            "test_in": shapes_test_in[0] if len(shapes_test_in) == 1 else shapes_test_in,
        },
        "palettes": {
            "train_in": palettes_train_in,
            "train_out": palettes_train_out,
            "test_in": palettes_test_in,
        },
    }

    if trace:
        logging.info(f"[present] train_in shapes={shapes_train_in}")
        logging.info(f"[present] train_out shapes={shapes_train_out}")
        logging.info(f"[present] test_in shapes={shapes_test_in}")
        logging.info(f"[present] train_in palettes={palettes_train_in}")
        logging.info(f"[present] train_out palettes={palettes_train_out}")
        logging.info(f"[present] test_in palettes={palettes_test_in}")

    return present
