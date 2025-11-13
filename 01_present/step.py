"""
01_present: Load everything into awareness.

Stage: present
Loads ARC task JSON into normalized structures (train_in, train_out, test_in).
"""

from typing import Any, Dict
import logging


def load(task_bundle: Dict[str, Any], trace: bool = False) -> Any:
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
      trace: enable debug logging if True.

    Output:
      A 'present' object (opaque to run.py) that future stages will refine.
      For now, this function is not implemented and always raises.
    """
    if trace:
        logging.info(f"[present] load() called for task_id={task_bundle.get('task_id')}")
    raise NotImplementedError("01_present.load is not implemented yet.")
