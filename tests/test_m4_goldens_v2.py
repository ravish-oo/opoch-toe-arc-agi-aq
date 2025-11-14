#!/usr/bin/env python3
"""
Test WO-4.1, WO-4.2, WO-4.3 (A+B+C atoms) against M4 golden checkpoints.

M4 goldens test:
- component_counts_nonzero (C-atoms, WO-4.3)
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import stages as packages
import importlib
import types

def load_stage_as_package(stage_dir_name):
    """Load a stage directory as a package to handle relative imports."""
    stage_path = Path(__file__).parent.parent / stage_dir_name

    # Create package
    pkg = types.ModuleType(stage_dir_name.replace("-", "_"))
    pkg.__path__ = [str(stage_path)]
    pkg.__package__ = stage_dir_name.replace("-", "_")
    sys.modules[pkg.__name__] = pkg

    # Load step.py as submodule
    spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.step",
        stage_path / "step.py"
    )
    step_mod = importlib.util.module_from_spec(spec)
    step_mod.__package__ = pkg.__name__
    sys.modules[spec.name] = step_mod

    # For 05_laws, also load atoms
    if stage_dir_name == "05_laws":
        atoms_spec = importlib.util.spec_from_file_location(
            f"{pkg.__name__}.atoms",
            stage_path / "atoms.py"
        )
        atoms_mod = importlib.util.module_from_spec(atoms_spec)
        atoms_mod.__package__ = pkg.__name__
        sys.modules[atoms_spec.name] = atoms_mod
        atoms_spec.loader.exec_module(atoms_mod)
        pkg.atoms = atoms_mod

    spec.loader.exec_module(step_mod)
    pkg.step = step_mod

    return step_mod

# Load stages
_present_step = load_stage_as_package("01_present")
_truth_step = load_stage_as_package("02_truth")
_scaffold_step = load_stage_as_package("03_scaffold")
_size_choice_step = load_stage_as_package("04_size_choice")
_laws_step = load_stage_as_package("05_laws")


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
    present = _present_step.load(task_bundle, trace=False)
    truth = _truth_step.canonicalize(present, trace=False)
    scaffold = _scaffold_step.build(truth, trace=False)
    size_choice = _size_choice_step.choose(truth, scaffold, trace=False)
    laws = _laws_step.mine(truth, scaffold, size_choice, trace=False)

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
    present = _present_step.load(task_bundle, trace=False)
    truth = _truth_step.canonicalize(present, trace=False)
    scaffold = _scaffold_step.build(truth, trace=False)
    size_choice = _size_choice_step.choose(truth, scaffold, trace=False)
    laws = _laws_step.mine(truth, scaffold, size_choice, trace=False)

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
    present = _present_step.load(task_bundle, trace=False)
    truth = _truth_step.canonicalize(present, trace=False)
    scaffold = _scaffold_step.build(truth, trace=False)
    size_choice = _size_choice_step.choose(truth, scaffold, trace=False)
    laws = _laws_step.mine(truth, scaffold, size_choice, trace=False)

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
    present = _present_step.load(task_bundle, trace=False)
    truth = _truth_step.canonicalize(present, trace=False)
    scaffold = _scaffold_step.build(truth, trace=False)
    size_choice = _size_choice_step.choose(truth, scaffold, trace=False)
    laws = _laws_step.mine(truth, scaffold, size_choice, trace=False)

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

    try:
        results.append(("833966f4", test_task_833966f4()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("833966f4", False))

    try:
        results.append(("74dd1130", test_task_74dd1130()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("74dd1130", False))

    try:
        results.append(("0d3d703e", test_task_0d3d703e()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("0d3d703e", False))

    try:
        results.append(("46f33fce", test_task_46f33fce()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        results.append(("46f33fce", False))

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
