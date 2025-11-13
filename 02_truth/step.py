"""
02_truth: Π — remove minted differences via canonical labeling.

Applies graph canonical labeling to establish a shared gauge across all grids.
"""


def canonicalize(present, trace: bool = False):
    """
    Apply Π (canonical labeling) to remove coordinate accidents.

    Input:
        present: output from 01_present.step.load
        trace: enable debug receipts

    Output:
        canonical object (grids in canonical coordinates)
    """
    raise NotImplementedError("02_truth/step.py:canonicalize() — WO-1.2/1.3 not yet implemented")
