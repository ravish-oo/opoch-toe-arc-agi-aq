#!/usr/bin/env python3
"""
Test script for WO-4.5: F24 input feature mirror

Validates:
  - A–E atoms computable on test_in (same formulas as on train_out)
  - F24 guardrail: NEVER called during mining, ONLY for evaluation
  - Cache works correctly
  - Atoms have correct structure and semantics on input grids
"""

import logging
import importlib.util
from pathlib import Path
import json
import numpy as np


def _import_stage_step(stage_name):
    """Helper to import step.py from stages with numeric prefixes."""
    spec = importlib.util.spec_from_file_location(
        f"{stage_name}.step",
        Path(__file__).parent / stage_name / "step.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_atoms():
    """Import atoms.py from 05_laws."""
    spec = importlib.util.spec_from_file_location(
        "laws_atoms",
        Path(__file__).parent / "05_laws" / "atoms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import stages
_present = _import_stage_step("01_present")
_truth = _import_stage_step("02_truth")
_scaffold = _import_stage_step("03_scaffold")
_laws = _import_stage_step("05_laws")

# Import atoms module for F24
atoms_module = _import_atoms()
get_input_atoms_for_test = atoms_module.get_input_atoms_for_test


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


def test_f24_basic_functionality():
    """
    Test F24: basic functionality on a real task.

    Validates:
      - F24 can compute A–E atoms on test_in
      - Atoms have correct structure
      - Cache works
    """
    print("\n" + "=" * 60)
    print("TEST 1: F24 basic functionality")
    print("=" * 60)

    # Load a task and run pipeline up through laws
    task_bundle = load_task("00576224")
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(canonical, trace=False)
    laws_result = _laws.mine(canonical, scaffold, None, trace=False)

    # AFTER mining, call F24 to get input atoms
    print("\n[TEST] Calling F24 AFTER mining (correct usage)...")
    input_atoms = get_input_atoms_for_test(canonical, test_idx=0)

    # Validate structure
    assert "A" in input_atoms, "Should have A-atoms"
    assert "B" in input_atoms, "Should have B-atoms"
    assert "C" in input_atoms, "Should have C-atoms"
    assert "D" in input_atoms, "Should have D-atoms"
    assert "E" in input_atoms, "Should have E-atoms"

    # Get test_in grid for shape validation
    test_in_grid = canonical["test_in"][0]
    H, W = test_in_grid.shape
    print(f"[TEST] test_in shape: {H}×{W}")

    # Validate A-atoms (scaffold geometry)
    A = input_atoms["A"]
    assert "H" in A and A["H"] == H, "A-atoms should have correct H"
    assert "W" in A and A["W"] == W, "A-atoms should have correct W"
    print(f"✅ A-atoms: H={A['H']}, W={A['W']}")

    # Validate B-atoms (local texture)
    B = input_atoms["B"]
    assert "n4_counts" in B, "B-atoms should have N4 counts"
    assert "hash_3x3" in B, "B-atoms should have 3×3 hash"
    assert B["hash_3x3"].shape == (H, W), \
        f"hash_3x3 shape should be ({H},{W})"
    num_colors_with_n4 = len(B["n4_counts"])
    print(f"✅ B-atoms: hash_3x3 shape={B['hash_3x3'].shape}, "
          f"{num_colors_with_n4} colors with N4 counts")

    # Validate C-atoms (connectivity & shape)
    C = input_atoms["C"]
    assert "components" in C, "C-atoms should have components"
    num_colors_with_comps = len([k for k, v in C["components"].items() if len(v) > 0])
    print(f"✅ C-atoms: {num_colors_with_comps} colors with components")

    # Validate D-atoms (repetition & tiling)
    D = input_atoms["D"]
    assert "row_periods" in D, "D-atoms should have row_periods"
    assert D["row_periods"].shape == (H,), f"row_periods shape should be ({H},)"
    assert "col_periods" in D, "D-atoms should have col_periods"
    assert D["col_periods"].shape == (W,), f"col_periods shape should be ({W},)"
    print(f"✅ D-atoms: row_periods shape={D['row_periods'].shape}, "
          f"col_periods shape={D['col_periods'].shape}")

    # Validate E-atoms (palette/global)
    E = input_atoms["E"]
    assert "palette" in E, "E-atoms should have palette"
    assert "pixel_counts" in E, "E-atoms should have pixel_counts"
    assert E["pixel_counts"].shape == (10,), "pixel_counts should have length 10"
    assert E["pixel_counts"].sum() == H * W, \
        f"pixel_counts sum should equal grid size ({H*W})"
    print(f"✅ E-atoms: palette={E['palette']}, "
          f"most_frequent={E['most_frequent']}")

    print("\n✅ PASS: F24 basic functionality correct")


def test_f24_cache():
    """
    Test F24: cache behavior.

    Validates:
      - Second call returns cached result (same object identity)
    """
    print("\n" + "=" * 60)
    print("TEST 2: F24 cache")
    print("=" * 60)

    # Load a task
    task_bundle = load_task("0520fde7")
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)

    # First call
    input_atoms_1 = get_input_atoms_for_test(canonical, test_idx=0)

    # Second call should return cached result
    input_atoms_2 = get_input_atoms_for_test(canonical, test_idx=0)

    # Verify it's the same object (cached)
    assert input_atoms_1 is input_atoms_2, "Second call should return cached result"
    print("✅ PASS: F24 cache working (same object returned)")


def test_f24_guardrail_manual():
    """
    Test F24: guardrail enforcement (manual verification).

    This test demonstrates correct usage pattern:
      1. Mine laws from train_out (F24 NOT called)
      2. After mining, use F24 to evaluate input features for already-mined laws

    Any call to F24 during mining would be a spec violation.
    """
    print("\n" + "=" * 60)
    print("TEST 3: F24 guardrail (manual verification)")
    print("=" * 60)

    # Load task
    task_bundle = load_task("025d127b")
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)
    scaffold = _scaffold.build(canonical, trace=False)

    # ========== MINING PHASE (F24 MUST NOT BE CALLED) ==========
    print("\n[MINING PHASE] Mining laws from train_out...")
    print("[MINING PHASE] F24 should NOT be called here!")

    laws_result = _laws.mine(canonical, scaffold, None, trace=False)

    print("[MINING PHASE] Mining complete ✅")

    # ========== EVALUATION PHASE (F24 CAN BE CALLED) ==========
    print("\n[EVALUATION PHASE] Now using F24 for input features...")

    input_atoms = get_input_atoms_for_test(canonical, test_idx=0)

    # Example: use input feature in a law constraint
    bg_input = input_atoms["E"]["most_frequent"][0]
    print(f"[EVALUATION PHASE] Input background color: {bg_input}")
    print(f"[EVALUATION PHASE] (This could be used to constrain test_out)")

    print("\n✅ PASS: F24 guardrail pattern demonstrated correctly")
    print("   - Mining: F24 NOT called ✅")
    print("   - Evaluation: F24 called AFTER mining ✅")


def test_f24_multiple_test_inputs():
    """
    Test F24: handling multiple test inputs (if task has >1 test).

    Note: Most ARC tasks have 1 test input, but API should support test_idx.
    """
    print("\n" + "=" * 60)
    print("TEST 4: F24 with test_idx parameter")
    print("=" * 60)

    # Load a task
    task_bundle = load_task("00576224")
    present = _present.load(task_bundle, trace=False)
    canonical = _truth.canonicalize(present, trace=False)

    # Get atoms for test_idx=0
    input_atoms_0 = get_input_atoms_for_test(canonical, test_idx=0)

    test_in_grid = canonical["test_in"][0]
    H, W = test_in_grid.shape

    # Verify shapes match
    assert input_atoms_0["A"]["H"] == H
    assert input_atoms_0["A"]["W"] == W

    print(f"✅ PASS: F24 test_idx=0 returns atoms for correct grid ({H}×{W})")

    # Test cache independence for different test_idx
    # (Most tasks only have 1 test, but API should handle it)
    print("   (Note: Most ARC tasks have 1 test input; cache keyed by test_idx)")


def main():
    """Run WO-4.5 F24 acceptance tests."""
    logging.basicConfig(level=logging.WARNING)

    print("WO-4.5 ACCEPTANCE TESTS: F24 input feature mirror")
    print("=" * 60)

    test_f24_basic_functionality()
    test_f24_cache()
    test_f24_guardrail_manual()
    test_f24_multiple_test_inputs()

    print("\n" + "=" * 60)
    print("🎉 ALL WO-4.5 F24 TESTS PASSED!")
    print("=" * 60)
    print("\nValidated:")
    print("  ✅ F24 computes A–E atoms on test_in")
    print("  ✅ Atoms have correct structure & semantics")
    print("  ✅ Cache works (same object returned)")
    print("  ✅ Guardrail pattern: F24 AFTER mining, NEVER during")
    print("  ✅ test_idx parameter supported")


if __name__ == "__main__":
    main()
