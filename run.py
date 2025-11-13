#!/usr/bin/env python3
"""
run.py — Brainstem orchestrator for ARC cognition solver.

Wires the seven consciousness stages:
  present → truth → scaffold → size_choice → laws → minimal_act → fixed_point

Zero algorithmic logic; pure sequencing.
"""

import argparse
import json
import logging
from pathlib import Path

# Import stage functions (forward dependencies only)
# Note: Python module names cannot start with digits, so we use importlib
import importlib.util

def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    spec = importlib.util.spec_from_file_location(
        f"{stage_name}.step",
        Path(__file__).parent / stage_name / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load stage modules
_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_size_choice = _import_stage_step("04_size_choice")
_laws = _import_stage_step("05_laws")
_minimal_act = _import_stage_step("06_minimal_act")
_fixed_point = _import_stage_step("07_fixed_point")

# Extract stage functions
load_present = _present.load
canonicalize = _truth.canonicalize
build_scaffold = _scaffold.build
choose_size = _size_choice.choose
mine_laws = _laws.mine
minimal_act = _minimal_act.solve
fixed_point_check = _fixed_point.check


def run_task(task_id: str, data_path: str, trace: bool = False):
    """
    Execute the full 7-stage pipeline on a single ARC task.

    Args:
        task_id: ARC task ID (e.g., "00576224")
        data_path: path to arc-agi_training_challenges.json
        trace: enable debug logging and receipts

    Returns:
        solution object with .out_grid attribute

    Raises:
        KeyError: if task_id not found in data file
        NotImplementedError: from unimplemented stages (expected until milestones complete)
    """
    # 1. Load raw task from JSON
    if trace:
        logging.info(f"Loading task {task_id} from {data_path}")

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = json.loads(data_file.read_text())

    if task_id not in data:
        raise KeyError(f"Task ID '{task_id}' not found in {data_path}")

    raw_task = data[task_id]

    task_bundle = {
        "task_id": task_id,
        "raw_task": raw_task,
    }

    # 2. Execute seven stages in order
    if trace:
        logging.info("[present] loading task into awareness")
    present = load_present(task_bundle, trace=trace)

    if trace:
        logging.info("[truth] applying Π (canonical labeling)")
    canonical = canonicalize(present, trace=trace)

    if trace:
        logging.info("[scaffold] building output-intrinsic geometry")
    scaffold = build_scaffold(canonical, trace=trace)

    if trace:
        logging.info("[size_choice] inferring test output size")
    out_size = choose_size(canonical, scaffold, trace=trace)

    if trace:
        logging.info("[laws] mining invariants from train_out")
    invariants = mine_laws(canonical, scaffold, out_size, trace=trace)

    if trace:
        logging.info("[minimal_act] solving ledger minimization")
    solution = minimal_act(canonical, invariants, out_size, trace=trace)

    if trace:
        logging.info("[fixed_point] checking N² = N")
    fixed_point_check(canonical, solution, trace=trace)

    return solution


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARC cognition solver — consciousness-first architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --task-id 00576224
  python run.py --task-id 1caeab9d --data data/arc-agi_training_challenges.json --trace
        """,
    )

    parser.add_argument(
        "--task-id",
        required=True,
        help="ARC task ID (e.g., 00576224)",
    )

    parser.add_argument(
        "--data",
        default="data/arc-agi_training_challenges.json",
        help="Path to ARC challenges JSON (default: data/arc-agi_training_challenges.json)",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable debug logging and receipts",
    )

    args = parser.parse_args()

    # Configure logging
    if args.trace:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

    try:
        solution = run_task(args.task_id, args.data, trace=args.trace)

        # Print final output grid
        # (solution.out_grid will be implemented in later milestones)
        try:
            out_grid = solution.out_grid
            print(out_grid)
        except AttributeError:
            # Early milestones: solution might not have out_grid yet
            print("Pipeline executed (output grid not yet available)")

    except NotImplementedError as e:
        logging.error(f"Pipeline gap: {e}")
        raise

    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
