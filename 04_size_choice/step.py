"""
04_size_choice: Infer test output size from training pairs + scaffold screening.

Learns size maps from training, screens candidates with scaffold facts.
"""


def choose(canonical, scaffold_train_out, trace: bool = False):
    """
    Choose test output size via size map candidates + scaffold screening.

    Input:
        canonical: output from 02_truth.step.canonicalize
        scaffold_train_out: output from 03_scaffold.step.build
        trace: enable debug receipts

    Output:
        out_size: (H_out, W_out) tuple
    """
    raise NotImplementedError("04_size_choice/step.py:choose() — WO-3.1/3.2 not yet implemented")
