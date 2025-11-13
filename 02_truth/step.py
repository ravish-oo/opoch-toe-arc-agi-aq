"""
02_truth: Π — remove minted differences via canonical labeling.

Stage: truth
Applies graph canonical labeling to establish a shared gauge across all grids.
"""

from typing import Any
import logging


def canonicalize(present: Any, trace: bool = False) -> Any:
    """
    Stage: truth (Π / canonical gauge)

    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2: Stage A — Canonical labeling
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π, kill minted differences

    Input:
      present: object returned by 01_present.load
      trace: enable debug logging if True.

    Output:
      A 'canonical' object (opaque to run.py) with grids in canonical gauge.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[truth] canonicalize() called")
    raise NotImplementedError("02_truth.canonicalize is not implemented yet.")
