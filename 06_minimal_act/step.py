"""
06_minimal_act: DO — single paid step via TU-LP (tree-TV) or MILP fallback.

Minimizes interface cost subject to mined constraints.
"""


def solve(canonical, invariants, out_size, trace: bool = False):
    """
    Solve for test output via ledger minimization (TU-LP or MILP fallback).

    Input:
        canonical: output from 02_truth.step.canonicalize
        invariants: output from 05_laws.step.mine
        out_size: (H_out, W_out) from 04_size_choice.step.choose
        trace: enable debug receipts

    Output:
        solution object with at least .out_grid attribute (np.ndarray or list-of-lists)
    """
    raise NotImplementedError("06_minimal_act/step.py:solve() — WO-6.* not yet implemented")
