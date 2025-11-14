#!/usr/bin/env python3
"""
Test script for WO-3.1: Size map candidate enumeration.

Tests all 6 families: identity, swap, factor, affine, tile, constant.
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


def test_task(task_id: str, trace: bool = False):
    """
    Test size_choice on a single task.

    Returns:
        size_choice result dict
    """
    if trace:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing task {task_id}")
        logging.info(f"{'='*60}")

    task_bundle = load_task(task_id)
    present = load_present(task_bundle, trace=trace)
    canonical = canonicalize(present, trace=trace)
    scaffold = build_scaffold(canonical, trace=trace)

    size_choice_result = choose_size(canonical, scaffold, trace=trace)

    return size_choice_result


def verify_wo_3_1_structure(result: dict, task_id: str) -> list:
    """
    Verify WO-3.1 output structure.

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []

    # Check required keys
    required_keys = ["status", "H_out", "W_out", "train_size_pairs", "candidates"]
    for key in required_keys:
        if key not in result:
            errors.append(f"Missing key: {key}")

    # Check status is CANDIDATES_ONLY
    if result.get("status") != "CANDIDATES_ONLY":
        errors.append(f"status should be 'CANDIDATES_ONLY', got {result.get('status')}")

    # Check H_out and W_out are None
    if result.get("H_out") is not None:
        errors.append(f"H_out should be None, got {result.get('H_out')}")

    if result.get("W_out") is not None:
        errors.append(f"W_out should be None, got {result.get('W_out')}")

    # Check train_size_pairs is non-empty list
    if not isinstance(result.get("train_size_pairs"), list):
        errors.append("train_size_pairs should be a list")
    elif len(result["train_size_pairs"]) == 0:
        errors.append("train_size_pairs should be non-empty")

    # Check candidates structure
    if not isinstance(result.get("candidates"), list):
        errors.append("candidates should be a list")
    else:
        for i, cand in enumerate(result["candidates"]):
            if "family" not in cand:
                errors.append(f"Candidate {i}: missing 'family'")
            if "params" not in cand:
                errors.append(f"Candidate {i}: missing 'params'")
            if "fits_all" not in cand:
                errors.append(f"Candidate {i}: missing 'fits_all'")
            elif cand["fits_all"] != True:
                errors.append(f"Candidate {i}: fits_all should be True, got {cand['fits_all']}")
            if "reproductions" not in cand:
                errors.append(f"Candidate {i}: missing 'reproductions'")
            else:
                # Check all reproductions have match=True
                for j, repro in enumerate(cand["reproductions"]):
                    if not repro.get("match", False):
                        errors.append(
                            f"Candidate {i} (family={cand.get('family')}), "
                            f"reproduction {j}: match should be True"
                        )

    return errors


def main():
    """Run all WO-3.1 tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info("="*60)
    logging.info("WO-3.1 TEST SUITE: Size map candidate enumeration")
    logging.info("="*60)

    tests = [
        ("00576224", "Factor map (3x expansion)"),
        ("5e6bbc0b", "Multi-shape outputs"),
        ("833966f4", "Thin grids (5x1)"),
    ]

    all_passed = True

    for task_id, description in tests:
        logging.info(f"\n{'='*60}")
        logging.info(f"Task {task_id}: {description}")
        logging.info(f"{'='*60}")

        try:
            result = test_task(task_id, trace=True)

            # Verify structure
            errors = verify_wo_3_1_structure(result, task_id)

            if errors:
                logging.error(f"\n❌ Task {task_id} FAILED:")
                for err in errors:
                    logging.error(f"  - {err}")
                all_passed = False
            else:
                # Additional checks based on task
                num_candidates = len(result["candidates"])
                families = [c["family"] for c in result["candidates"]]

                logging.info(f"\n✅ Task {task_id} structure PASSED")
                logging.info(f"  - {num_candidates} candidates found")
                logging.info(f"  - Families: {set(families)}")

                # Task-specific validations
                if task_id == "00576224":
                    # Should have factor candidate with r_H=3, r_W=3
                    factor_cands = [c for c in result["candidates"] if c["family"] == "factor"]
                    if not factor_cands:
                        logging.error("  ⚠️  Expected factor candidate not found")
                        all_passed = False
                    elif factor_cands[0]["params"] != {"r_H": 3, "r_W": 3}:
                        logging.error(f"  ⚠️  Factor params incorrect: {factor_cands[0]['params']}")
                        all_passed = False
                    else:
                        logging.info("  ✓ Factor candidate correct: r_H=3, r_W=3")

                    # Should have constant candidate (both outputs 6x6)
                    const_cands = [c for c in result["candidates"] if c["family"] == "constant"]
                    if const_cands:
                        logging.info(f"  ✓ Constant candidate: {const_cands[0]['params']}")

        except Exception as e:
            logging.error(f"\n❌ Task {task_id} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Summary
    logging.info("\n" + "="*60)
    logging.info("TEST SUMMARY")
    logging.info("="*60)

    if all_passed:
        logging.info("🎉 ALL TESTS PASSED! WO-3.1 implementation verified.")
        return 0
    else:
        logging.error("❌ SOME TESTS FAILED. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
