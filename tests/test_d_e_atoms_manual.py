#!/usr/bin/env python3
"""
Manual test for WO-4.4: D-atoms and E-atoms verification

Tests basic semantics on synthetic grids:
  - Periods: minimal period for rows/cols
  - Tiling: exact tiling by divisor blocks
  - Palette: pixel counts, most/least frequent
  - Global mapping: bijective color permutation
"""

import numpy as np
from pathlib import Path
import importlib.util


def _import_atoms():
    """Import atoms.py from 05_laws."""
    spec = importlib.util.spec_from_file_location(
        "laws_atoms",
        Path(__file__).parent / "05_laws" / "atoms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atoms = _import_atoms()
compute_D_atoms = atoms.compute_D_atoms
compute_E_atoms_for_grid = atoms.compute_E_atoms_for_grid
compute_global_palette_mapping = atoms.compute_global_palette_mapping


def test_periods_simple():
    """Test minimal period detection on simple sequences."""
    print("\n" + "=" * 60)
    print("TEST 1: Minimal periods (simple sequences)")
    print("=" * 60)

    # Row 0: [1,2,1,2,1,2] -> period 2
    # Row 1: [3,3,3,3,3,3] -> period 1
    # Row 2: [4,5,6,4,5,6] -> period 3
    grid = np.array([
        [1, 2, 1, 2, 1, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 5, 6, 4, 5, 6]
    ], dtype=int)

    D = compute_D_atoms(grid)

    print(f"Row periods: {D['row_periods'].tolist()}")
    assert D['row_periods'][0] == 2, f"Row 0 period should be 2, got {D['row_periods'][0]}"
    assert D['row_periods'][1] == 1, f"Row 1 period should be 1, got {D['row_periods'][1]}"
    assert D['row_periods'][2] == 3, f"Row 2 period should be 3, got {D['row_periods'][2]}"

    # Col 0: [1,3,4] -> period 3 (full length, no repetition)
    # Col 1: [2,3,5] -> period 3
    print(f"Col periods: {D['col_periods'].tolist()}")

    print("✅ PASS: Period detection correct")


def test_tiling_exact():
    """Test exact tiling detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Exact tiling (2×2 tile on 4×6)")
    print("=" * 60)

    # Create 4×6 grid tiled by 2×2 blocks
    tile = np.array([[1, 2],
                     [3, 4]], dtype=int)

    grid = np.tile(tile, (2, 3))  # 4×6 grid
    print(f"Grid shape: {grid.shape}")
    print(f"Grid:\n{grid}")

    D = compute_D_atoms(grid)

    # Check (2,2) tiling should be True
    assert D['tiling_flags'][(2, 2)] is True, "Should detect 2×2 tiling"

    # (4,6) should also be True (full grid is a tile)
    assert D['tiling_flags'][(4, 6)] is True, "Full grid tiling should be True"

    # (1,1) should be False (individual pixels differ)
    assert D['tiling_flags'][(1, 1)] is False, "1×1 tiling should be False"

    true_tilings = [(b_r, b_c) for (b_r, b_c), v in D['tiling_flags'].items() if v]
    print(f"True tilings: {true_tilings}")

    print("✅ PASS: Tiling detection correct")


def test_palette_stats():
    """Test palette statistics."""
    print("\n" + "=" * 60)
    print("TEST 3: Palette statistics")
    print("=" * 60)

    # Grid with colors: 1(3×), 2(2×), 3(5×), 4(2×)
    grid = np.array([
        [1, 1, 1, 2],
        [3, 3, 3, 2],
        [3, 3, 4, 4]
    ], dtype=int)

    E = compute_E_atoms_for_grid(grid, C_atoms=None)

    print(f"Palette: {E['palette']}")
    print(f"Pixel counts: {E['pixel_counts'].tolist()}")

    # Most frequent: 3 (5 pixels)
    assert E['most_frequent'] == [3], f"Most frequent should be [3], got {E['most_frequent']}"

    # Least frequent: 2 and 4 (both 2 pixels)
    assert set(E['least_frequent']) == {2, 4}, \
        f"Least frequent should be {{2, 4}}, got {set(E['least_frequent'])}"

    # Palette should be [1,2,3,4]
    assert E['palette'] == [1, 2, 3, 4], f"Palette mismatch"

    # Missing should be [0,5,6,7,8,9]
    assert E['missing'] == [0, 5, 6, 7, 8, 9], f"Missing colors mismatch"

    print("✅ PASS: Palette stats correct")


def test_global_mapping_identity():
    """Test global mapping: identity (train_in == train_out) - Case A."""
    print("\n" + "=" * 60)
    print("TEST 4: Global mapping - identity (Case A: K_in == K_out)")
    print("=" * 60)

    # Two training pairs where input == output
    train_in = [
        np.array([[1, 2], [3, 4]], dtype=int),
        np.array([[5, 6]], dtype=int)
    ]
    train_out = [
        np.array([[1, 2], [3, 4]], dtype=int),
        np.array([[5, 6]], dtype=int)
    ]

    mapping = compute_global_palette_mapping(train_in, train_out)

    print(f"has_bijection: {mapping['has_bijection']}")
    print(f"is_permutation: {mapping['is_permutation']}")
    print(f"perm: {mapping['perm']}")
    print(f"cycles: {mapping['cycles']}")

    assert mapping['has_bijection'] is True, "Should detect bijection"
    assert mapping['is_permutation'] is True, "Case A: K_in == K_out, should be permutation"
    assert mapping['color_mapping'] is None, "Case A doesn't use color_mapping"

    # Identity mapping
    for k in [1, 2, 3, 4, 5, 6]:
        assert mapping['perm'][k] == k, f"Identity mapping broken for {k}"

    # Cycles should all be singletons
    for cycle in mapping['cycles']:
        assert len(cycle) == 1, f"Identity should have singleton cycles, got {cycle}"

    print("✅ PASS: Case A (permutation) identity correct")


def test_global_mapping_permutation():
    """Test global mapping: simple permutation - Case A."""
    print("\n" + "=" * 60)
    print("TEST 5: Global mapping - permutation (1→2, 2→3, 3→1) Case A")
    print("=" * 60)

    # Permutation: 1→2, 2→3, 3→1 (cycle of length 3)
    # K_in == K_out == {1,2,3}
    train_in = [
        np.array([[1, 2, 3]], dtype=int),
        np.array([[1, 1, 2, 2, 3, 3]], dtype=int)
    ]
    train_out = [
        np.array([[2, 3, 1]], dtype=int),
        np.array([[2, 2, 3, 3, 1, 1]], dtype=int)
    ]

    mapping = compute_global_palette_mapping(train_in, train_out)

    print(f"has_bijection: {mapping['has_bijection']}")
    print(f"is_permutation: {mapping['is_permutation']}")
    print(f"perm: {mapping['perm']}")
    print(f"cycles: {mapping['cycles']}")

    assert mapping['has_bijection'] is True, "Should detect bijection"
    assert mapping['is_permutation'] is True, "Case A: K_in == K_out, should be permutation"
    assert mapping['perm'][1] == 2, "1 should map to 2"
    assert mapping['perm'][2] == 3, "2 should map to 3"
    assert mapping['perm'][3] == 1, "3 should map to 1"
    assert mapping['color_mapping'] is None, "Case A doesn't use color_mapping"

    # Should have one cycle of length 3
    assert len(mapping['cycles']) == 1, f"Should have 1 cycle, got {len(mapping['cycles'])}"
    assert set(mapping['cycles'][0]) == {1, 2, 3}, "Cycle should be {1,2,3}"

    print("✅ PASS: Case A (permutation) cycle correct")


def test_global_mapping_inconsistent():
    """Test global mapping: inconsistent (no bijection)."""
    print("\n" + "=" * 60)
    print("TEST 6: Global mapping - inconsistent (1→2 and 1→3)")
    print("=" * 60)

    # Inconsistent: first pair maps 1→2, second pair maps 1→3
    train_in = [
        np.array([[1]], dtype=int),
        np.array([[1]], dtype=int)
    ]
    train_out = [
        np.array([[2]], dtype=int),
        np.array([[3]], dtype=int)
    ]

    mapping = compute_global_palette_mapping(train_in, train_out)

    print(f"has_bijection: {mapping['has_bijection']}")

    assert mapping['has_bijection'] is False, "Should detect inconsistency"
    assert mapping['is_permutation'] is False, "is_permutation should be False"
    assert mapping['perm'] is None, "perm should be None"
    assert mapping['cycles'] is None, "cycles should be None"
    assert mapping['color_mapping'] is None, "color_mapping should be None"

    print("✅ PASS: Inconsistency detected correctly")


def test_global_mapping_shape_mismatch():
    """Test global mapping: shape mismatch (no bijection possible)."""
    print("\n" + "=" * 60)
    print("TEST 7: Global mapping - shape mismatch")
    print("=" * 60)

    # Shapes differ: cannot do position-wise mapping
    train_in = [
        np.array([[1, 2]], dtype=int),  # 1×2
    ]
    train_out = [
        np.array([[3], [4]], dtype=int),  # 2×1
    ]

    mapping = compute_global_palette_mapping(train_in, train_out)

    print(f"has_bijection: {mapping['has_bijection']}")

    assert mapping['has_bijection'] is False, "Shape mismatch should prevent bijection"
    assert mapping['is_permutation'] is False, "is_permutation should be False"
    assert mapping['perm'] is None, "perm should be None"
    assert mapping['cycles'] is None, "cycles should be None"
    assert mapping['color_mapping'] is None, "color_mapping should be None"

    print("✅ PASS: Shape mismatch handled correctly")


def test_global_mapping_disjoint_palettes():
    """Test global mapping: disjoint palettes - Case B (non-permutation bijection)."""
    print("\n" + "=" * 60)
    print("TEST 8: Global mapping - disjoint palettes (Case B: K_in ≠ K_out)")
    print("=" * 60)

    # Disjoint palettes: K_in = {1,3,5}, K_out = {2,4,6}
    # This is a bijection but NOT a permutation (different sets)
    # Cycles are NOT defined for disjoint palettes
    train_in = [
        np.array([[1, 3, 5]], dtype=int),
        np.array([[1, 1, 3, 3, 5, 5]], dtype=int)
    ]
    train_out = [
        np.array([[2, 4, 6]], dtype=int),
        np.array([[2, 2, 4, 4, 6, 6]], dtype=int)
    ]

    mapping = compute_global_palette_mapping(train_in, train_out)

    print(f"has_bijection: {mapping['has_bijection']}")
    print(f"is_permutation: {mapping['is_permutation']}")
    print(f"color_mapping: {mapping['color_mapping']}")
    print(f"perm: {mapping['perm']}")
    print(f"cycles: {mapping['cycles']}")

    assert mapping['has_bijection'] is True, "Should detect bijection"
    assert mapping['is_permutation'] is False, "Case B: K_in ≠ K_out, should NOT be permutation"
    assert mapping['perm'] is None, "Case B doesn't use perm"
    assert mapping['cycles'] is None, "Cycles NOT defined for disjoint palettes (Case B)"
    assert mapping['color_mapping'] is not None, "Case B should use color_mapping"

    # Validate mapping: 1→2, 3→4, 5→6
    assert mapping['color_mapping'][1] == 2, "1 should map to 2"
    assert mapping['color_mapping'][3] == 4, "3 should map to 4"
    assert mapping['color_mapping'][5] == 6, "5 should map to 6"

    print("✅ PASS: Case B (disjoint palettes) correctly handled - no cycles computed")


if __name__ == "__main__":
    test_periods_simple()
    test_tiling_exact()
    test_palette_stats()
    test_global_mapping_identity()
    test_global_mapping_permutation()
    test_global_mapping_inconsistent()
    test_global_mapping_shape_mismatch()
    test_global_mapping_disjoint_palettes()

    print("\n" + "=" * 60)
    print("🎉 ALL MANUAL D+E ATOMS TESTS PASSED!")
    print("=" * 60)
