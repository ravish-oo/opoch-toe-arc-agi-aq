"""
05_laws: WHAT — derive atom types and mine invariants from train_out.

Promotes "always true" facts into linear constraints (fixes, equalities, forbids).
"""


def mine(canonical, scaffold_train_out, out_size, trace: bool = False):
    """
    Mine invariants from train_out: type→color fixes + relational equalities.

    Input:
        canonical: output from 02_truth.step.canonicalize
        scaffold_train_out: output from 03_scaffold.step.build
        out_size: (H_out, W_out) from 04_size_choice.step.choose
        trace: enable debug receipts

    Output:
        invariants object (fixed, forbids, equal_pairs, etc.)
    """
    raise NotImplementedError("05_laws/step.py:mine() — WO-4.*/5.* not yet implemented")
