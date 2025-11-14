#!/usr/bin/env python3
"""
Test script for WO-2.3: Inner region and global scaffold facts.

Verifies scaffold outputs against golden checkpoints:
- goldens/5e6bbc0b/5e6bbc0b_scaffold.json
- goldens/833966f4/833966f4_scaffold.json
- goldens/00576224/00576224_scaffold.json
"""

import json
import logging
from pathlib import Path
import importlib.util
import numpy as np

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

load_present = _present.load
canonicalize = _truth.canonicalize
build_scaffold = _scaffold.build


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


def extract_scaffold_metrics(scaffold: dict):
    """
    Extract metrics from per-output scaffold dict for comparison with golden checkpoints.

    Returns:
        List of per-output dicts with metrics, aggregated parity flags
    """
    per_output_scaffolds = scaffold["per_output"]
    aggregated = scaffold["aggregated"]

    per_output = []

    # Distance field stats
    def dist_stats(d):
        return {
            "min": int(d.min()),
            "max": int(d.max()),
            "sum": int(d.sum()),
        }

    # Extract metrics for each output
    for entry in per_output_scaffolds:
        i = entry["index"]
        H, W = entry["shape"]

        frame_mask = entry["frame_mask"]
        frame_count = int(frame_mask.sum())

        inner = entry["inner"]
        inner_count = int(inner.sum())

        metrics = {
            "index": i,
            "H": H,
            "W": W,
            "frame_count": frame_count,
            "inner_count": inner_count,
            "d_top": dist_stats(entry["d_top"]),
            "d_bottom": dist_stats(entry["d_bottom"]),
            "d_left": dist_stats(entry["d_left"]),
            "d_right": dist_stats(entry["d_right"]),
            "has_midrow": entry["has_midrow"],
            "has_midcol": entry["has_midcol"],
        }
        per_output.append(metrics)

    # Global aggregated parity
    has_midrow_global = aggregated["has_midrow_all"]
    has_midcol_global = aggregated["has_midcol_all"]

    return per_output, has_midrow_global, has_midcol_global


def test_task(task_id: str, golden_path: str, trace: bool = False):
    """
    Test scaffold stage for a single task against golden checkpoint.

    Args:
        task_id: ARC task ID
        golden_path: path to *_scaffold.json golden checkpoint
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

    # Extract actual metrics
    per_output, has_midrow, has_midcol = extract_scaffold_metrics(scaffold)

    errors = []

    # Compare per-output metrics
    expected_per_output = golden["per_output"]

    if len(per_output) != len(expected_per_output):
        errors.append(
            f"Train count mismatch: expected {len(expected_per_output)}, got {len(per_output)}"
        )
        return False, errors

    for i, (actual, expected) in enumerate(zip(per_output, expected_per_output)):
        if trace:
            logging.info(f"\nTrain example {i}:")

        # Check dimensions
        if actual["H"] != expected["H"] or actual["W"] != expected["W"]:
            errors.append(
                f"  Train {i}: Shape mismatch: expected ({expected['H']}, {expected['W']}), "
                f"got ({actual['H']}, {actual['W']})"
            )

        # Check frame count
        if actual["frame_count"] != expected["frame_count"]:
            errors.append(
                f"  Train {i}: frame_count mismatch: expected {expected['frame_count']}, "
                f"got {actual['frame_count']}"
            )

        # Check inner count (WO-2.3)
        if actual["inner_count"] != expected["inner_count"]:
            errors.append(
                f"  Train {i}: inner_count mismatch: expected {expected['inner_count']}, "
                f"got {actual['inner_count']}"
            )
        else:
            if trace:
                logging.info(f"  ✓ inner_count = {actual['inner_count']}")

        # Check distance fields
        for direction in ["d_top", "d_bottom", "d_left", "d_right"]:
            actual_stats = actual[direction]
            expected_stats = expected[direction]

            for stat in ["min", "max", "sum"]:
                if actual_stats[stat] != expected_stats[stat]:
                    errors.append(
                        f"  Train {i}: {direction}.{stat} mismatch: "
                        f"expected {expected_stats[stat]}, got {actual_stats[stat]}"
                    )

        # Check parity flags (WO-2.3)
        # Some golden files have per-output parity, some have global
        if "has_midrow" in expected:
            if actual["has_midrow"] != expected["has_midrow"]:
                errors.append(
                    f"  Train {i}: has_midrow mismatch: expected {expected['has_midrow']}, "
                    f"got {actual['has_midrow']}"
                )
            else:
                if trace:
                    logging.info(f"  ✓ has_midrow = {actual['has_midrow']}")

            if actual["has_midcol"] != expected["has_midcol"]:
                errors.append(
                    f"  Train {i}: has_midcol mismatch: expected {expected['has_midcol']}, "
                    f"got {actual['has_midcol']}"
                )
            else:
                if trace:
                    logging.info(f"  ✓ has_midcol = {actual['has_midcol']}")

    # Check global parity flags if present
    if "global" in golden:
        if has_midrow != golden["global"]["has_midrow"]:
            errors.append(
                f"  Global has_midrow mismatch: expected {golden['global']['has_midrow']}, "
                f"got {has_midrow}"
            )
        else:
            if trace:
                logging.info(f"\n  ✓ Global has_midrow = {has_midrow}")

        if has_midcol != golden["global"]["has_midcol"]:
            errors.append(
                f"  Global has_midcol mismatch: expected {golden['global']['has_midcol']}, "
                f"got {has_midcol}"
            )
        else:
            if trace:
                logging.info(f"  ✓ Global has_midcol = {has_midcol}")

    passed = len(errors) == 0

    if passed:
        if trace:
            logging.info(f"\n✅ Task {task_id} PASSED")
    else:
        logging.error(f"\n❌ Task {task_id} FAILED with {len(errors)} errors:")
        for err in errors:
            logging.error(err)

    return passed, errors


def main():
    """Run all WO-2.3 tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info("="*60)
    logging.info("WO-2.3 TEST SUITE: Inner region and global scaffold facts")
    logging.info("="*60)

    tests = [
        ("5e6bbc0b", "goldens/5e6bbc0b/5e6bbc0b_scaffold.json"),
        ("833966f4", "goldens/833966f4/833966f4_scaffold.json"),
        ("00576224", "goldens/00576224/00576224_scaffold.json"),
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
        logging.info("\n🎉 ALL TESTS PASSED! WO-2.3 implementation verified.")
        return 0
    else:
        logging.error("\n❌ SOME TESTS FAILED. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
