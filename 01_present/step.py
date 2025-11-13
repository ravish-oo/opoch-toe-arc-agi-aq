"""
01_present: Load everything into awareness.

Loads ARC task JSON into normalized structures (train_in, train_out, test_in).
"""


def load(task_bundle, trace: bool = False):
    """
    Load ARC task bundle into present object.

    Input:
        task_bundle: dict with {"task_id": str, "raw_task": dict}
        trace: enable debug receipts

    Output:
        present object (opaque to run.py; contains train_in, train_out, test_in)
    """
    raise NotImplementedError("01_present/step.py:load() — WO-1.1 not yet implemented")
