"""
04_size_choice: Infer test output size from training pairs + scaffold screening.

Stage: size_choice
Learns size maps from training, screens candidates with scaffold facts.
"""

from typing import Any, Tuple
import logging


def choose(canonical: Any, scaffold: Any, trace: bool = False) -> Tuple[int, int]:
    """
    Stage: size_choice (S0)

    Anchor:
      - 01_STAGES.md: size_choice
      - 00_MATH_SPEC.md §3: Stage S0 — Output canvas size
      - 02_QUANTUM_MAPPING.md: choose output shape consistent with laws

    Input:
      canonical: object from 02_truth.canonicalize
      scaffold: object from 03_scaffold.build (train_out-only scaffold)
      trace: enable debug logging if True.

    Output:
      (H_out, W_out) for the test canvas.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info("[size_choice] choose() called")
    raise NotImplementedError("04_size_choice.choose is not implemented yet.")
