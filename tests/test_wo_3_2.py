#!/usr/bin/env python3
"""
Test script for WO-3.2: Size choice with scaffold screening.

Tests screening logic and final size selection against golden checkpoints.
"""

import json
import logging
from pathlib import Path
import importlib.util

# Import stage modules
def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    spec = importlib.util.spec_from_file_location(
        f"{stage_name}.step",
        Path(__file__).parent / stage_name / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_size_choice = _import_stage_step("04_size_choice")

load_present = _present.load
canonicalize = _truth.canonicalize
build_scaffold = _scaffold.build
choose_size = _size_choice.choose


def load_task(task_id: str, data_path: str = "data/arc-agi_training_challenges.json"):
    """Load raw task from JSON."""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = json.loads(data_file.read_text())
    if task_id not in data:
        raise KeyError(f"Task ID '{task_id}' not found in {data_path}")

    raw_task = data[task_id]
    task_bundle = {
        "task_id": task_id,
        "test_index": 0,
        "raw_task": raw_task,
    }
    return task_bundle


def test_task(task_id: str, golden_path: str, trace: bool = False):
    """
    Test size_choice on a single task against golden checkpoint.

    Args:
        task_id: ARC task ID
        golden_path: path to *_size_choice.json golden checkpoint
        trace: enable debug logging

    Returns:
        (passed, errors): bool and list of error messages
    """
    if trace:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing task {task_id}")
        logging.info(f"{'='*60}")

    # Load golden checkpoint
    golden_file = Path(golden_path)
    if not golden_file.exists():
        return False, [f"Golden file not found: {golden_path}"]

    golden = json.loads(golden_file.read_text())

    # Load and process task
    task_bundle = load_task(task_id)
    present = load_present(task_bundle, trace=trace)
    canonical = canonicalize(present, trace=trace)
    scaffold = build_scaffold(canonical, trace=trace)

    size_choice_result = choose_size(canonical, scaffold, trace=trace)

    errors = []

    # Extract actual values
    actual_status = size_choice_result["status"]
    actual_H_out = size_choice_result["H_out"]
    actual_W_out = size_choice_result["W_out"]
    actual_survivors = size_choice_result["survivors"]

    # Extract expected values from golden
    expected_status = golden["result"]["status"]
    expected_H_out = golden["result"]["chosen"]["H_out"]
    expected_W_out = golden["result"]["chosen"]["W_out"]
    expected_survivor_count = golden["result"]["survivor_count"]

    # Check status
    if actual_status != expected_status:
        errors.append(
            f"Status mismatch: expected '{expected_status}', got '{actual_status}'"
        )

    # Check H_out and W_out
    if actual_H_out != expected_H_out:
        errors.append(
            f"H_out mismatch: expected {expected_H_out}, got {actual_H_out}"
        )

    if actual_W_out != expected_W_out:
        errors.append(
            f"W_out mismatch: expected {expected_W_out}, got {actual_W_out}"
        )

    # Check survivor count
    if len(actual_survivors) != expected_survivor_count:
        errors.append(
            f"Survivor count mismatch: expected {expected_survivor_count}, "
            f"got {len(actual_survivors)}"
        )

    # If we have exactly 1 survivor, check its details
    if len(actual_survivors) == 1 and expected_survivor_count == 1:
        survivor = actual_survivors[0]
        expected_chosen = golden["result"]["chosen"]

        # Check survivor's predicted test size
        if survivor["H_out_test"] != expected_H_out:
            errors.append(
                f"Survivor H_out_test mismatch: expected {expected_H_out}, "
                f"got {survivor['H_out_test']}"
            )

        if survivor["W_out_test"] != expected_W_out:
            errors.append(
                f"Survivor W_out_test mismatch: expected {expected_W_out}, "
                f"got {survivor['W_out_test']}"
            )

        # Check family (golden uses "kind", implementation uses "family")
        expected_kind = expected_chosen["kind"]
        actual_family = survivor["family"]
        if actual_family != expected_kind:
            errors.append(
                f"Survivor family mismatch: expected '{expected_kind}', "
                f"got '{actual_family}'"
            )

        # Check params
        expected_params = expected_chosen["params"]
        actual_params = survivor["params"]
        if actual_params != expected_params:
            errors.append(
                f"Survivor params mismatch: expected {expected_params}, "
                f"got {actual_params}"
            )

    passed = len(errors) == 0

    if passed:
        if trace:
            logging.info(f"\n✅ Task {task_id} PASSED")
            logging.info(f"  status={actual_status}")
            logging.info(f"  H_out={actual_H_out}, W_out={actual_W_out}")
            logging.info(f"  survivors={len(actual_survivors)}")
    else:
        logging.error(f"\n❌ Task {task_id} FAILED with {len(errors)} errors:")
        for err in errors:
            logging.error(f"  - {err}")

    return passed, errors


def main():
    """Run all WO-3.2 tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info("="*60)
    logging.info("WO-3.2 TEST SUITE: Size choice with scaffold screening")
    logging.info("="*60)

    tests = [
        ("00576224", "goldens/00576224/00576224_size_choice.json"),
        ("5e6bbc0b", "goldens/5e6bbc0b/5e6bbc0b_size_choice.json"),
        ("833966f4", "goldens/833966f4/833966f4_size_choice.json"),
    ]

    results = {}
    all_passed = True

    for task_id, golden_path in tests:
        passed, errors = test_task(task_id, golden_path, trace=True)
        results[task_id] = (passed, errors)
        all_passed = all_passed and passed

    # Summary
    logging.info("\n" + "="*60)
    logging.info("TEST SUMMARY")
    logging.info("="*60)

    for task_id, (passed, errors) in results.items():
        status = "✅ PASS" if passed else f"❌ FAIL ({len(errors)} errors)"
        logging.info(f"{task_id}: {status}")

    if all_passed:
        logging.info("\n🎉 ALL TESTS PASSED! WO-3.2 implementation verified.")
        return 0
    else:
        logging.error("\n❌ SOME TESTS FAILED. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
