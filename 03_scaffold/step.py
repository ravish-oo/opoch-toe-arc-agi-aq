"""
03_scaffold: WHERE — stable canvas geometry from train_out only.

Computes frame, distance fields, inner region, and global structural facts.
"""


def build(canonical, trace: bool = False):
    """
    Build scaffold from train_out: frame + distance atlas + inner region.

    Input:
        canonical: output from 02_truth.step.canonicalize
        trace: enable debug receipts

    Output:
        scaffold_train_out object (frame mask, distance fields, inner mask, etc.)
    """
    raise NotImplementedError("03_scaffold/step.py:build() — WO-2.1/2.2/2.3 not yet implemented")
