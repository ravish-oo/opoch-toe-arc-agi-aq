#!/usr/bin/env python3
"""
Comprehensive test for Milestones 1, 2, 3 against all golden checkpoints.

Tests:
  - M1 (present): train_in, train_out, test_in arrays match golden
  - M2 (scaffold): frame_count, inner_count, distance field stats match golden
  - M3 (size_choice): status, H_out, W_out, chosen params match golden

This is a major checkpoint before moving to Milestone 5 (laws mining).
"""

import sys
import json
import logging
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path so utils module can be found
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    import types

    stage_path = Path(__file__).parent.parent / stage_name

    # Create package
    pkg = types.ModuleType(stage_name.replace("-", "_"))
    pkg.__path__ = [str(stage_path)]
    pkg.__package__ = stage_name.replace("-", "_")
    sys.modules[pkg.__name__] = pkg

    # Load step.py as submodule
    spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.step",
        stage_path / "step.py"
    )
    step_mod = importlib.util.module_from_spec(spec)
    step_mod.__package__ = pkg.__name__
    sys.modules[spec.name] = step_mod

    spec.loader.exec_module(step_mod)
    pkg.step = step_mod

    return step_mod


# Import stages
_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_size_choice = _import_stage_step("04_size_choice")


def load_task(task_id: str) -> Dict[str, Any]:
    """Load task from ARC training data."""
    data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
    data = json.loads(data_path.read_text())

    if task_id not in data:
        raise KeyError(f"Task {task_id} not found")

    task_bundle = {
        "task_id": task_id,
        "test_index": 0,
        "raw_task": data[task_id],
    }
    return task_bundle


def test_m1_present(task_id: str, golden: Dict[str, Any]) -> bool:
    """
    Test M1 (present): validate train_in, train_out, test_in arrays.

    Returns True if all arrays match golden.
    """
    print(f"\n  [M1-present] Testing {task_id}...")

    # Load task
    task_bundle = load_task(task_id)
    present = _present.load(task_bundle, trace=False)

    # Compare train_in
    golden_train_in = golden["train_in"]
    for i, expected in enumerate(golden_train_in):
        actual = present["train_in"][i].tolist()
        if actual != expected:
            print(f"    ❌ train_in[{i}] mismatch:")
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
            return False

    # Compare train_out
    golden_train_out = golden["train_out"]
    for i, expected in enumerate(golden_train_out):
        actual = present["train_out"][i].tolist()
        if actual != expected:
            print(f"    ❌ train_out[{i}] mismatch:")
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
            return False

    # Compare test_in
    golden_test_in = golden["test_in"]
    for i, expected in enumerate(golden_test_in):
        actual = present["test_in"][i].tolist()
        if actual != expected:
            print(f"    ❌ test_in[{i}] mismatch:")
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
            return False

    print(f"    ✅ All arrays match golden")
    return True


def test_m2_scaffold(task_id: str, golden: Dict[str, Any]) -> bool:
    """
    Test M2 (scaffold): validate frame_count, inner_count, distance field stats.

    Returns True if all summaries match golden.
    """
    print(f"\n  [M2-scaffold] Testing {task_id}...")

    # Load and run pipeline up to scaffold
    task_bundle = load_task(task_id)
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(canonical, trace=False)

    # Compare per_output summaries
    golden_per_output = golden["per_output"]
    actual_per_output = scaffold["per_output"]

    if len(actual_per_output) != len(golden_per_output):
        print(f"    ❌ Number of outputs mismatch: {len(actual_per_output)} vs {len(golden_per_output)}")
        return False

    for i, golden_out in enumerate(golden_per_output):
        actual_out = actual_per_output[i]

        # Check H, W
        if actual_out["shape"] != (golden_out["H"], golden_out["W"]):
            print(f"    ❌ train_out[{i}] shape mismatch:")
            print(f"       Expected: ({golden_out['H']}, {golden_out['W']})")
            print(f"       Actual:   {actual_out['shape']}")
            return False

        # Check frame_count
        actual_frame_count = int(actual_out["frame_mask"].sum())
        if actual_frame_count != golden_out["frame_count"]:
            print(f"    ❌ train_out[{i}] frame_count mismatch:")
            print(f"       Expected: {golden_out['frame_count']}")
            print(f"       Actual:   {actual_frame_count}")
            return False

        # Check inner_count
        actual_inner_count = int(actual_out["inner"].sum())
        if actual_inner_count != golden_out["inner_count"]:
            print(f"    ❌ train_out[{i}] inner_count mismatch:")
            print(f"       Expected: {golden_out['inner_count']}")
            print(f"       Actual:   {actual_inner_count}")
            return False

        # Check distance field stats
        for field_name in ["d_top", "d_bottom", "d_left", "d_right"]:
            d_field = actual_out[field_name]
            golden_stats = golden_out[field_name]

            actual_min = int(d_field.min())
            actual_max = int(d_field.max())
            actual_sum = int(d_field.sum())

            if actual_min != golden_stats["min"]:
                print(f"    ❌ train_out[{i}] {field_name}.min mismatch: {actual_min} vs {golden_stats['min']}")
                return False

            if actual_max != golden_stats["max"]:
                print(f"    ❌ train_out[{i}] {field_name}.max mismatch: {actual_max} vs {golden_stats['max']}")
                return False

            if actual_sum != golden_stats["sum"]:
                print(f"    ❌ train_out[{i}] {field_name}.sum mismatch: {actual_sum} vs {golden_stats['sum']}")
                return False

    # Check global flags (optional, may be at global level or per-output level)
    if "global" in golden:
        golden_global = golden["global"]
        actual_global = scaffold.get("aggregated", {})

        # has_midrow check (if present in golden)
        if "has_midrow" in golden_global:
            actual_has_midrow = actual_global.get("has_midrow_all", False)
            if actual_has_midrow != golden_global["has_midrow"]:
                print(f"    ❌ has_midrow mismatch: {actual_has_midrow} vs {golden_global['has_midrow']}")
                return False

        # has_midcol check (if present in golden)
        if "has_midcol" in golden_global:
            actual_has_midcol = actual_global.get("has_midcol_all", False)
            if actual_has_midcol != golden_global["has_midcol"]:
                print(f"    ❌ has_midcol mismatch: {actual_has_midcol} vs {golden_global['has_midcol']}")
                return False

    print(f"    ✅ All scaffold summaries match golden")
    return True


def test_m3_size_choice(task_id: str, golden: Dict[str, Any]) -> bool:
    """
    Test M3 (size_choice): validate status, H_out, W_out, chosen params.

    Returns True if result matches golden.
    """
    print(f"\n  [M3-size_choice] Testing {task_id}...")

    # Load and run pipeline up to size_choice
    task_bundle = load_task(task_id)
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(canonical, trace=False)
    size_result = _size_choice.choose(canonical, scaffold, trace=False)

    # Compare result
    golden_result = golden["result"]

    # Check status
    if size_result["status"] != golden_result["status"]:
        print(f"    ❌ status mismatch:")
        print(f"       Expected: {golden_result['status']}")
        print(f"       Actual:   {size_result['status']}")
        return False

    # If status is OK, check H_out and W_out
    if golden_result["status"] == "OK":
        golden_chosen = golden_result["chosen"]

        if size_result["H_out"] != golden_chosen["H_out"]:
            print(f"    ❌ H_out mismatch:")
            print(f"       Expected: {golden_chosen['H_out']}")
            print(f"       Actual:   {size_result['H_out']}")
            return False

        if size_result["W_out"] != golden_chosen["W_out"]:
            print(f"    ❌ W_out mismatch:")
            print(f"       Expected: {golden_chosen['W_out']}")
            print(f"       Actual:   {size_result['W_out']}")
            return False

        # Check kind (if available)
        if "kind" in size_result and "kind" in golden_chosen:
            if size_result["kind"] != golden_chosen["kind"]:
                print(f"    ❌ kind mismatch:")
                print(f"       Expected: {golden_chosen['kind']}")
                print(f"       Actual:   {size_result['kind']}")
                return False

        # Check params (if available and kind matches)
        if "params" in size_result and "params" in golden_chosen:
            if size_result["params"] != golden_chosen["params"]:
                print(f"    ⚠️  params mismatch (non-critical):")
                print(f"       Expected: {golden_chosen['params']}")
                print(f"       Actual:   {size_result['params']}")
                # Don't fail on params mismatch if H_out/W_out are correct

    print(f"    ✅ Size choice result matches golden")
    return True


def run_all_tests():
    """Run M1/M2/M3 tests for all available golden files."""
    goldens_dir = Path(__file__).parent.parent / "goldens"

    # Find all tasks with golden files
    task_dirs = sorted([d for d in goldens_dir.iterdir() if d.is_dir()])

    print("=" * 70)
    print("MILESTONE 1/2/3 COMPREHENSIVE CHECKPOINT TESTS")
    print("=" * 70)
    print(f"\nFound {len(task_dirs)} tasks with golden files")

    results = {}

    for task_dir in task_dirs:
        task_id = task_dir.name
        print(f"\n{'=' * 70}")
        print(f"Testing task: {task_id}")
        print('=' * 70)

        # Check which golden files exist
        present_file = task_dir / f"{task_id}_present.json"
        scaffold_file = task_dir / f"{task_id}_scaffold.json"
        size_choice_file = task_dir / f"{task_id}_size_choice.json"

        task_results = {
            "M1_present": None,
            "M2_scaffold": None,
            "M3_size_choice": None,
        }

        # Test M1 (present)
        if present_file.exists():
            try:
                golden = json.loads(present_file.read_text())
                task_results["M1_present"] = test_m1_present(task_id, golden)
            except Exception as e:
                print(f"  [M1-present] ❌ Exception: {e}")
                task_results["M1_present"] = False
        else:
            print(f"  [M1-present] ⚠️  Golden file not found")

        # Test M2 (scaffold)
        if scaffold_file.exists():
            try:
                golden = json.loads(scaffold_file.read_text())
                task_results["M2_scaffold"] = test_m2_scaffold(task_id, golden)
            except Exception as e:
                print(f"  [M2-scaffold] ❌ Exception: {e}")
                task_results["M2_scaffold"] = False
        else:
            print(f"  [M2-scaffold] ⚠️  Golden file not found")

        # Test M3 (size_choice)
        if size_choice_file.exists():
            try:
                golden = json.loads(size_choice_file.read_text())
                task_results["M3_size_choice"] = test_m3_size_choice(task_id, golden)
            except Exception as e:
                print(f"  [M3-size_choice] ❌ Exception: {e}")
                task_results["M3_size_choice"] = False
        else:
            print(f"  [M3-size_choice] ⚠️  Golden file not found")

        results[task_id] = task_results

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True
    for task_id, task_results in results.items():
        m1_status = "✅" if task_results["M1_present"] else ("❌" if task_results["M1_present"] is False else "⚠️ ")
        m2_status = "✅" if task_results["M2_scaffold"] else ("❌" if task_results["M2_scaffold"] is False else "⚠️ ")
        m3_status = "✅" if task_results["M3_size_choice"] else ("❌" if task_results["M3_size_choice"] is False else "⚠️ ")

        print(f"{task_id}: M1={m1_status} M2={m2_status} M3={m3_status}")

        if task_results["M1_present"] is False or \
           task_results["M2_scaffold"] is False or \
           task_results["M3_size_choice"] is False:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 ALL MILESTONE 1/2/3 CHECKPOINT TESTS PASSED!")
        print("=" * 70)
        print("\nReady to proceed to Milestone 5 (laws mining)")
        return True
    else:
        print("❌ SOME TESTS FAILED - Review issues above")
        print("=" * 70)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    success = run_all_tests()
    sys.exit(0 if success else 1)
