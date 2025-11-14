#!/usr/bin/env python3
"""
Test script for WO-4.4: D-atoms (Repetition & tiling) and E-atoms (Palette/global)

Validates:
  - Row/col minimal periods computed correctly
  - 2D tiling flags for divisor pairs
  - Per-grid palette stats (pixel counts, most/least frequent)
  - Per-color component counts
  - Task-level global palette mapping (bijection when exists)
"""

import logging
import importlib.util
from pathlib import Path
import json


def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    spec = importlib.util.spec_from_file_location(
        f"{stage_name}.step",
        Path(__file__).parent / stage_name / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import stages
_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_laws = _import_stage_step("05_laws")


def load_task(task_id: str):
    """Load task from training data."""
    data_path = Path("data/arc-agi_training_challenges.json")
    data = json.loads(data_path.read_text())

    if task_id not in data:
        raise KeyError(f"Task {task_id} not found")

    task_bundle = {
        "task_id": task_id,
        "test_index": 0,
        "raw_task": data[task_id],
    }
    return task_bundle


def test_D_E_atoms_on_task(task_id: str):
    """
    Test D and E atoms computation on a single task.

    Args:
        task_id: ARC task ID
    """
    print(f"\n{'=' * 60}")
    print(f"Testing task {task_id}")
    print(f"{'=' * 60}")

    # Run pipeline through laws
    task_bundle = load_task(task_id)
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(canonical, trace=False)
    laws_result = _laws.mine(canonical, scaffold, None, trace=False)

    # Check first train_out
    D_atoms = laws_result["train_out_D_atoms"][0]
    E_atoms = laws_result["train_out_E_atoms"][0]
    global_map = laws_result["global_palette_mapping"]
    grid = canonical["train_out"][0]
    H, W = grid.shape

    print(f"Grid shape: {H}×{W}")

    # ========== Validate D-atoms ==========
    assert "row_periods" in D_atoms
    assert "col_periods" in D_atoms
    assert "tiling_flags" in D_atoms

    # Validate shapes
    assert D_atoms["row_periods"].shape == (H,), \
        f"row_periods shape should be ({H},), got {D_atoms['row_periods'].shape}"
    assert D_atoms["col_periods"].shape == (W,), \
        f"col_periods shape should be ({W},), got {D_atoms['col_periods'].shape}"

    # Validate period ranges
    assert D_atoms["row_periods"].min() >= 1
    assert D_atoms["row_periods"].max() <= W
    assert D_atoms["col_periods"].min() >= 1
    assert D_atoms["col_periods"].max() <= H

    # Validate tiling flags are for divisor pairs only
    for (b_r, b_c), flag in D_atoms["tiling_flags"].items():
        assert H % b_r == 0, f"b_r={b_r} should divide H={H}"
        assert W % b_c == 0, f"b_c={b_c} should divide W={W}"
        assert isinstance(flag, (bool, np.bool_)), f"tiling_flag should be bool"

    # Full grid tiling should always be True
    assert D_atoms["tiling_flags"][(H, W)] is True, "Full grid tiling should be True"

    print(f"✅ D-atoms valid:")
    print(f"  - row_periods: {D_atoms['row_periods'].tolist()}")
    print(f"  - col_periods: {D_atoms['col_periods'].tolist()}")
    true_tilings = [(b_r, b_c) for (b_r, b_c), v in D_atoms["tiling_flags"].items() if v]
    print(f"  - tiling_flags (True): {true_tilings}")

    # ========== Validate E-atoms ==========
    assert "pixel_counts" in E_atoms
    assert "component_counts" in E_atoms
    assert "palette" in E_atoms
    assert "missing" in E_atoms
    assert "most_frequent" in E_atoms
    assert "least_frequent" in E_atoms

    # Validate pixel counts
    assert E_atoms["pixel_counts"].shape == (10,), "pixel_counts should have length 10"
    assert E_atoms["pixel_counts"].sum() == H * W, \
        f"pixel_counts sum should equal grid size ({H*W})"

    # Validate palette consistency
    for k in E_atoms["palette"]:
        assert E_atoms["pixel_counts"][k] > 0, f"Palette color {k} should have count > 0"
    for k in E_atoms["missing"]:
        assert E_atoms["pixel_counts"][k] == 0, f"Missing color {k} should have count == 0"

    # Validate palette + missing = {0..9}
    assert set(E_atoms["palette"]) | set(E_atoms["missing"]) == set(range(10)), \
        "Palette ∪ missing should equal {0..9}"
    assert len(set(E_atoms["palette"]) & set(E_atoms["missing"])) == 0, \
        "Palette ∩ missing should be empty"

    # Validate most/least frequent
    if len(E_atoms["palette"]) > 0:
        max_count = E_atoms["pixel_counts"].max()
        for k in E_atoms["most_frequent"]:
            assert E_atoms["pixel_counts"][k] == max_count, \
                f"Most frequent color {k} should have max count {max_count}"

        positive_counts = [E_atoms["pixel_counts"][k] for k in E_atoms["palette"]]
        min_count = min(positive_counts)
        for k in E_atoms["least_frequent"]:
            assert E_atoms["pixel_counts"][k] == min_count, \
                f"Least frequent color {k} should have min count {min_count}"

    # Validate component counts
    for k in range(10):
        assert k in E_atoms["component_counts"], f"component_counts missing key {k}"
        assert E_atoms["component_counts"][k] >= 0, \
            f"component_counts[{k}] should be non-negative"

    print(f"✅ E-atoms valid:")
    print(f"  - palette: {E_atoms['palette']}")
    print(f"  - most_frequent: {E_atoms['most_frequent']}")
    print(f"  - least_frequent: {E_atoms['least_frequent']}")

    # ========== Validate global palette mapping ==========
    assert "has_bijection" in global_map
    assert "is_permutation" in global_map
    assert "perm" in global_map
    assert "cycles" in global_map
    assert "color_mapping" in global_map

    if global_map["has_bijection"]:
        if global_map["is_permutation"]:
            # Case A: True permutation (K_in == K_out)
            assert global_map["perm"] is not None, "Case A should have perm"
            assert global_map["cycles"] is not None, "Case A should have cycles"
            assert global_map["color_mapping"] is None, "Case A shouldn't use color_mapping"

            # Validate perm is dict[int, int]
            for cin, cout in global_map["perm"].items():
                assert isinstance(cin, int) and isinstance(cout, int)

            # Validate cycles cover all palette_in
            all_cycle_elements = set()
            for cycle in global_map["cycles"]:
                all_cycle_elements.update(cycle)
            assert all_cycle_elements == set(global_map["perm"].keys()), \
                "Cycles should cover all keys in perm"

            print(f"✅ Global mapping: Case A (permutation)")
            print(f"  - perm: {global_map['perm']}")
            print(f"  - cycles: {global_map['cycles']}")
        else:
            # Case B: Disjoint palettes (K_in ≠ K_out)
            assert global_map["perm"] is None, "Case B shouldn't use perm"
            assert global_map["cycles"] is None, "Case B shouldn't have cycles"
            assert global_map["color_mapping"] is not None, "Case B should have color_mapping"

            # Validate color_mapping is dict[int, int]
            for cin, cout in global_map["color_mapping"].items():
                assert isinstance(cin, int) and isinstance(cout, int)

            print(f"✅ Global mapping: Case B (disjoint palettes)")
            print(f"  - color_mapping: {global_map['color_mapping']}")
    else:
        # No bijection
        assert global_map["is_permutation"] is False, "No bijection means not a permutation"
        assert global_map["perm"] is None
        assert global_map["cycles"] is None
        assert global_map["color_mapping"] is None
        print(f"✅ Global mapping: no bijection (as expected)")


import numpy as np


def main():
    """Run WO-4.4 acceptance tests."""
    logging.basicConfig(level=logging.WARNING)

    print("WO-4.4 ACCEPTANCE TESTS: D-atoms (repetition/tiling) + E-atoms (palette/global)")

    # Test case 1: 00576224 (6×6 grid)
    test_D_E_atoms_on_task("00576224")

    # Test case 2: 0520fde7 (3×3 small grid)
    test_D_E_atoms_on_task("0520fde7")

    # Test case 3: 025d127b (8×9 asymmetric)
    test_D_E_atoms_on_task("025d127b")

    print(f"\n{'=' * 60}")
    print("🎉 ALL WO-4.4 TESTS PASSED!")
    print(f"{'=' * 60}")
    print("\nValidated:")
    print("  ✅ Row/col minimal periods (1 <= p <= dimension)")
    print("  ✅ 2D tiling flags for divisor pairs")
    print("  ✅ Pixel counts sum to grid size")
    print("  ✅ Palette + missing = {0..9}")
    print("  ✅ Most/least frequent colors correct")
    print("  ✅ Component counts non-negative")
    print("  ✅ Global palette mapping structure valid")


if __name__ == "__main__":
    main()
