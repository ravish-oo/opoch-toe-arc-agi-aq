"""
07_fixed_point: N² = N — idempotence check.

Stage: fixed_point
Re-runs pipeline with (test_in, test_out) appended to verify law closure.
"""

from typing import Any
import logging


def check(canonical: Any, solution: Any, trace: bool = False) -> None:
    """
    Stage: fixed_point (N² = N)

    Anchor:
      - 01_STAGES.md: fixed_point
      - 00_MATH_SPEC.md §7: Idempotence (N² = N)
      - 02_QUANTUM_MAPPING.md: re-see; output must be stable

    Input:
      canonical: from 02_truth.canonicalize
      solution: from 06_minimal_act.solve (must contain out_grid when implemented)
      trace: enable debug logging if True.

    Output:
      None. In final implementation, this will re-run pipeline with test pair added
      and assert stability. For now, it is not implemented and always raises.
    """
    if trace:
        logging.info("[fixed_point] check() called")
    raise NotImplementedError("07_fixed_point.check is not implemented yet.")
