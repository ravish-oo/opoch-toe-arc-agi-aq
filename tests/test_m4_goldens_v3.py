#!/usr/bin/env python3
"""
Test WO-4.1, WO-4.2, WO-4.3 (A+B+C atoms) against M4 golden checkpoints.

This version de-canonicalizes grids to compare component counts on original representation.
"""

import sys
import json
import logging
import importlib.util
import numpy as np
from pathlib import Path
from scipy import ndimage


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

    # For 05_laws, also load atoms
    if stage_name == "05_laws":
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


def compute_component_counts_on_original(canonical_grid, row_order, col_order):
    """
    Compute component counts on de-canonicalized (original) grid.

    Args:
        canonical_grid: Grid in canonical coordinates
        row_order: R_X array (old_r -> canon_r mapping)
        col_order: C_X array (old_c -> canon_c mapping)

    Returns: dict mapping color -> number of components (for colors > 0)
    """
    # De-canonicalize: original[old_r, old_c] = canonical[R_X[old_r], C_X[old_c]]
    R_X = np.array(row_order)
    C_X = np.array(col_order)

    H, W = canonical_grid.shape
    original_grid = np.zeros_like(canonical_grid)

    for old_r in range(H):
        canon_r = R_X[old_r]
        for old_c in range(W):
            canon_c = C_X[old_c]
            original_grid[old_r, old_c] = canonical_grid[canon_r, canon_c]

    # Compute components using 4-connectivity
    structure_4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=int)

    colors = np.unique(original_grid)
    counts = {}

    for k in colors:
        if k == 0:
            continue  # skip background

        mask = (original_grid == k).astype(np.uint8)
        labeled, num_features = ndimage.label(mask, structure=structure_4)

        if num_features > 0:
            counts[str(k)] = num_features

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
    size_choice = _size_choice.choose(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    num_train_in = len(present["train_in"])

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        overall_idx = num_train_in + i
        canonical_grid = truth["train_out"][i]
        row_order = truth["row_orders"][overall_idx]
        col_order = truth["col_orders"][overall_idx]

        actual = compute_component_counts_on_original(canonical_grid, row_order, col_order)

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
    size_choice = _size_choice.choose(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    num_train_in = len(present["train_in"])

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        overall_idx = num_train_in + i
        canonical_grid = truth["train_out"][i]
        row_order = truth["row_orders"][overall_idx]
        col_order = truth["col_orders"][overall_idx]

        actual = compute_component_counts_on_original(canonical_grid, row_order, col_order)

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
    size_choice = _size_choice.choose(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    num_train_in = len(present["train_in"])

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        overall_idx = num_train_in + i
        canonical_grid = truth["train_out"][i]
        row_order = truth["row_orders"][overall_idx]
        col_order = truth["col_orders"][overall_idx]

        actual = compute_component_counts_on_original(canonical_grid, row_order, col_order)

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
    size_choice = _size_choice.choose(truth, scaffold, trace=False)
    laws = _laws.mine(truth, scaffold, size_choice, trace=False)

    num_train_in = len(present["train_in"])

    passed = True
    for i, expected in enumerate(golden["train_out"]):
        overall_idx = num_train_in + i
        canonical_grid = truth["train_out"][i]
        row_order = truth["row_orders"][overall_idx]
        col_order = truth["col_orders"][overall_idx]

        actual = compute_component_counts_on_original(canonical_grid, row_order, col_order)

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
    print("De-canonicalizing grids to compare on original representation")
    print("=" * 60)

    results = []

    try:
        results.append(("833966f4", test_task_833966f4()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("833966f4", False))

    try:
        results.append(("74dd1130", test_task_74dd1130()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("74dd1130", False))

    try:
        results.append(("0d3d703e", test_task_0d3d703e()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("0d3d703e", False))

    try:
        results.append(("46f33fce", test_task_46f33fce()))
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
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
