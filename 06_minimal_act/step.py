"""
06_minimal_act: DO — single paid step via TU-LP (tree-TV) or MILP fallback.

Stage: minimal_act
Minimizes interface cost subject to mined constraints.
"""

from typing import Any
import logging


class Solution:
    """
    Placeholder for final solution.
    Later WOs will extend this to include out_grid and receipts.
    """
    def __init__(self, out_grid=None):
        self.out_grid = out_grid


def solve(canonical: Any, invariants: Any, out_size: Any, trace: bool = False) -> Solution:
    """
    Stage: minimal_act (D)

    Anchor:
      - 01_STAGES.md: minimal_act
      - 00_MATH_SPEC.md §6: Stage D — Ledger minimization
      - 02_QUANTUM_MAPPING.md: DO = paid step, minimize ledger

    Input:
      canonical: from 02_truth.canonicalize
      invariants: from 05_laws.mine
      out_size: (H_out, W_out) from 04_size_choice.choose
      trace: enable debug logging if True.

    Output:
      Solution object containing at least solution.out_grid (grid for test_out).
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[minimal_act] solve() called")
    raise NotImplementedError("06_minimal_act.solve is not implemented yet.")
