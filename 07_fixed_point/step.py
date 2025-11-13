"""
07_fixed_point: N² = N — idempotence check.

Re-runs pipeline with (test_in, test_out) appended to verify law closure.
"""


def check(canonical, solution, trace: bool = False):
    """
    Verify idempotence: re-run with test output appended should yield identical result.

    Input:
        canonical: output from 02_truth.step.canonicalize
        solution: output from 06_minimal_act.step.solve
        trace: enable debug receipts

    Output:
        None (raises on mismatch)
    """
    raise NotImplementedError("07_fixed_point/step.py:check() — WO-7.1 not yet implemented")
