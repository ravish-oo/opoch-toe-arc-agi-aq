"""
05_laws: WHAT — derive atom types and mine invariants from train_out.

Stage: laws
Promotes "always true" facts into linear constraints (fixes, equalities, forbids).
"""

from typing import Any
import logging


def mine(canonical: Any, scaffold: Any, out_size: Any, trace: bool = False) -> Any:
    """
    Stage: laws (N)

    Anchor:
      - 01_STAGES.md: laws
      - 00_MATH_SPEC.md §5: Stage N — Invariants as linear constraints
      - 02_QUANTUM_MAPPING.md: WHAT = law nucleus over scaffold

    Input:
      canonical: from 02_truth.canonicalize
      scaffold: from 03_scaffold.build
      out_size: (H_out, W_out) from 04_size_choice.choose
      trace: enable debug logging if True.

    Output:
      Invariants object encoding fixes, equalities, forbids, etc.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[laws] mine() called")
    raise NotImplementedError("05_laws.mine is not implemented yet.")
