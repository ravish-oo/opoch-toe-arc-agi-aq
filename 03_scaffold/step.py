"""
03_scaffold: WHERE — stable canvas geometry from train_out only.

Stage: scaffold
Computes frame, distance fields, inner region, and global structural facts.
"""

from typing import Any
import logging


def build(canonical: Any, trace: bool = False) -> Any:
    """
    Stage: scaffold (WHERE)

    Anchor:
      - 01_STAGES.md: scaffold
      - 00_MATH_SPEC.md §4: Stage F — Frame & distances
      - 02_QUANTUM_MAPPING.md: WHERE = output-intrinsic scaffold

    Input:
      canonical: object from 02_truth.canonicalize
      trace: enable debug logging if True.

    Output:
      Scaffold object containing frame, distance fields, inner region (train_out-only).
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[scaffold] build() called")
    raise NotImplementedError("03_scaffold.build is not implemented yet.")
