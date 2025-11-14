#!/usr/bin/env python3
"""
Test WO-4.1, WO-4.2, WO-4.3 (A+B+C atoms) against M4 golden checkpoints.

M4 goldens test:
- pixel_counts (E-atoms, not yet implemented)
- component_counts_nonzero (C-atoms, WO-4.3)
- row_period_hist (D-atoms, not yet implemented)

This test validates C-atoms only (component counts per color).
"""

import sys
import json
import logging
import importlib.util
from pathlib import Path


def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    spec = importlib.util.spec_from_file_location(
        f"{stage_name}.step",
        Path(__file__).parent.parent / stage_name / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import stages
_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_size_choice = _import_stage_step("04_size_choice")
_laws = _import_stage_step("05_laws")


def load_task(task_id: str):
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


def compute_component_counts_nonzero(C_atoms):
    """
    Extract component_counts_nonzero from C-atoms.

    Returns: dict mapping color → number of components (for colors > 0)
    """
    components = C_atoms["components"]
    counts = {}
    for k, comps in components.items():
        if k > 0 and len(comps) > 0:  # only non-background colors with components
            counts[str(k)] = len(comps)
    return counts


def test_task_833966f4():
    """Test simple 5×1 case with component counts."""
    print("\n=== Testing 833966f4 ===")

    golden = {
        "train_out": [
            {
                "index": 0,
                "component_counts": {"0": 1, "1": 1, "6": 1, "8": 1, "9": 1}
            },
            {
                "index": 1,
                "component_counts": {"2": 1, "3": 1, "4": 1, "6": 1, "8": 1}
            }
        ]
    }

    task_bundle = load_task("833966f4")
    present = _present.load(task_bundle, trace=False)
    truth = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(truth, trace=False)
    size_choice = _size_choice.mine(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    C_atoms_list = laws["train_out_C_atoms"]

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        C_atoms = C_atoms_list[i]
        actual = compute_component_counts_nonzero(C_atoms)

        # Compare (note: golden has all colors including 0, we only count >0)
        expected_nonzero = {k: v for k, v in expected["component_counts"].items() if k != "0"}

        if actual == expected_nonzero:
            print(f"  ✅ train_out[{i}]: component_counts match")
        else:
            print(f"  ❌ train_out[{i}]: component_counts MISMATCH")
            print(f"     Expected: {expected_nonzero}")
            print(f"     Actual:   {actual}")
            passed = False

    return passed


def test_task_74dd1130():
    """Test complex 3×3 case with multiple train_outs."""
    print("\n=== Testing 74dd1130 ===")

    golden = {
        "train_out": [
            {
                "index": 0,
                "component_counts_nonzero": {"5": 2, "8": 2, "9": 2}
            },
            {
                "index": 1,
                "component_counts_nonzero": {"2": 1, "5": 2, "6": 1}
            },
            {
                "index": 2,
                "component_counts_nonzero": {"1": 1, "2": 2, "6": 2}
            },
            {
                "index": 3,
                "component_counts_nonzero": {"1": 2, "2": 2, "5": 2}
            }
        ]
    }

    task_bundle = load_task("74dd1130")
    present = _present.load(task_bundle, trace=False)
    truth = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(truth, trace=False)
    size_choice = _size_choice.mine(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    C_atoms_list = laws["train_out_C_atoms"]

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        C_atoms = C_atoms_list[i]
        actual = compute_component_counts_nonzero(C_atoms)

        if actual == expected["component_counts_nonzero"]:
            print(f"  ✅ train_out[{i}]: component_counts_nonzero match")
        else:
            print(f"  ❌ train_out[{i}]: component_counts_nonzero MISMATCH")
            print(f"     Expected: {expected['component_counts_nonzero']}")
            print(f"     Actual:   {actual}")
            passed = False

    return passed


def test_task_0d3d703e():
    """Test palette permutation case with 4 train_outs."""
    print("\n=== Testing 0d3d703e ===")

    golden = {
        "train_out": [
            {
                "index": 0,
                "component_counts_nonzero": {"1": 1, "9": 1, "2": 1}
            },
            {
                "index": 1,
                "component_counts_nonzero": {"6": 1, "4": 1, "9": 1}
            },
            {
                "index": 2,
                "component_counts_nonzero": {"8": 1, "3": 1, "6": 1}
            },
            {
                "index": 3,
                "component_counts_nonzero": {"4": 1, "5": 1, "6": 1}
            }
        ]
    }

    task_bundle = load_task("0d3d703e")
    present = _present.load(task_bundle, trace=False)
    truth = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(truth, trace=False)
    size_choice = _size_choice.mine(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    C_atoms_list = laws["train_out_C_atoms"]

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        C_atoms = C_atoms_list[i]
        actual = compute_component_counts_nonzero(C_atoms)

        if actual == expected["component_counts_nonzero"]:
            print(f"  ✅ train_out[{i}]: component_counts_nonzero match")
        else:
            print(f"  ❌ train_out[{i}]: component_counts_nonzero MISMATCH")
            print(f"     Expected: {expected['component_counts_nonzero']}")
            print(f"     Actual:   {actual}")
            passed = False

    return passed


def test_task_46f33fce():
    """Test larger 20×20 case from M5 goldens (applicable to M4 C-atoms)."""
    print("\n=== Testing 46f33fce ===")

    golden = {
        "train_out": [
            {
                "index": 0,
                "component_counts_nonzero": {"1": 1, "2": 1, "3": 1, "4": 1, "8": 1}
            },
            {
                "index": 1,
                "component_counts_nonzero": {"1": 1, "2": 1, "3": 2, "4": 2}
            },
            {
                "index": 2,
                "component_counts_nonzero": {"1": 1, "2": 1, "3": 1, "4": 1}
            }
        ]
    }

    task_bundle = load_task("46f33fce")
    present = _present.load(task_bundle, trace=False)
    truth = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(truth, trace=False)
    size_choice = _size_choice.mine(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    C_atoms_list = laws["train_out_C_atoms"]

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        C_atoms = C_atoms_list[i]
        actual = compute_component_counts_nonzero(C_atoms)

        if actual == expected["component_counts_nonzero"]:
            print(f"  ✅ train_out[{i}]: component_counts_nonzero match")
        else:
            print(f"  ❌ train_out[{i}]: component_counts_nonzero MISMATCH")
            print(f"     Expected: {expected['component_counts_nonzero']}")
            print(f"     Actual:   {actual}")
            passed = False

    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("M4 Golden Checkpoint Tests (C-atoms: component counts)")
    print("=" * 60)

    results = []

    results.append(("833966f4", test_task_833966f4()))
    results.append(("74dd1130", test_task_74dd1130()))
    results.append(("0d3d703e", test_task_0d3d703e()))
    results.append(("46f33fce", test_task_46f33fce()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for task_id, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{task_id}: {status}")

    all_passed = all(p for _, p in results)

    if all_passed:
        print("\n🎉 All M4 golden checkpoint tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED")
        sys.exit(1)
