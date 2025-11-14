"""
Shared scaffold geometry helpers.

Per-grid geometric computation factored out from 03_scaffold for reuse in F24.

Anchors:
  - 00_MATH_SPEC.md §4: Stage F — Frame & distances (per-grid scaffold geometry)
  - 00_MATH_SPEC.md §5.1 F: Input features (guardrail) — mirror A–E on inputs
"""

import numpy as np
from typing import Dict


def build_scaffold_for_grid(grid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build per-grid scaffold geometry (frame + distance fields).

    This is the core geometric computation used by both:
      - 03_scaffold/step.py (on train_out for S0 and Stage N)
      - 05_laws/atoms.py F24 (on test_in for input feature evaluation)

    Input:
      grid: H×W numpy array (canonical grid, any values)

    Output:
      scaffold_info: {
        "d_top": H×W int array (distance to top border),
        "d_bottom": H×W int array (distance to bottom border),
        "d_left": H×W int array (distance to left border),
        "d_right": H×W int array (distance to right border),
      }

    Note:
      - Frame mask not returned (implicit: outer border)
      - Inner/parity/thickness/periods not needed for A–E atoms
      - Those are aggregated across train_out in 03_scaffold for S0 screens
    """
    H, W = grid.shape

    # Frame is implicit: outer border (r=0, r=H-1, c=0, c=W-1)
    # No need to return it; distance fields encode it (border cells have distance 0)

    # Directional distances to border (closed-form)
    # From 00_MATH_SPEC.md §4.2 and WO-2.2
    d_top = np.zeros((H, W), dtype=int)
    d_bottom = np.zeros((H, W), dtype=int)
    d_left = np.zeros((H, W), dtype=int)
    d_right = np.zeros((H, W), dtype=int)

    for r in range(H):
        for c in range(W):
            d_top[r, c] = r
            d_bottom[r, c] = (H - 1) - r
            d_left[r, c] = c
            d_right[r, c] = (W - 1) - c

    return {
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }
