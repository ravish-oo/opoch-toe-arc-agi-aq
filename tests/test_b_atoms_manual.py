#!/usr/bin/env python3
"""
Manual test for WO-4.2: B-atoms verification

Tests basic semantics on small synthetic grids.
"""

import numpy as np
import sys
from pathlib import Path
import importlib.util

# Import 05_laws/atoms.py
def _import_atoms():
    spec = importlib.util.spec_from_file_location(
        "laws_atoms",
        Path(__file__).parent / "05_laws" / "atoms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

atoms = _import_atoms()
compute_B_atoms = atoms.compute_B_atoms


def test_uniform_grid():
    """Test on uniform 3×3 grid (all same color)."""
    print("\n" + "="*60)
    print("TEST 1: Uniform 3×3 grid (all color 5)")
    print("="*60)

    grid = np.full((3, 3), 5, dtype=int)
    print("Grid:")
    print(grid)

    B = compute_B_atoms(grid)

    # N4 counts for color 5:
    # Center cell should have 4 neighbors of color 5
    # Corner cells should have 2 neighbors
    # Edge cells should have 3 neighbors
    print(f"\nn4_counts[5]:")
    print(B["n4_counts"][5])
    print(f"Center (1,1): {B['n4_counts'][5][1,1]} (expect 4)")
    print(f"Corner (0,0): {B['n4_counts'][5][0,0]} (expect 2)")
    print(f"Edge (0,1): {B['n4_counts'][5][0,1]} (expect 3)")

    # N8 counts
    print(f"\nn8_counts[5]:")
    print(B["n8_counts"][5])
    print(f"Center (1,1): {B['n8_counts'][5][1,1]} (expect 8)")
    print(f"Corner (0,0): {B['n8_counts'][5][0,0]} (expect 3)")

    # 3×3 hash: interior cells should have same hash (all 5s)
    # border cells will differ due to sentinel=10
    print(f"\nhash_3x3 values:")
    print(B["hash_3x3"])
    print(f"Center hash: {B['hash_3x3'][1,1]}")
    print(f"Corner hash: {B['hash_3x3'][0,0]} (should differ due to sentinel)")

    # Run-lengths: all should be 3 (full row/col)
    print(f"\nrow_span_len:")
    print(B["row_span_len"])
    print("All should be 3")

    print(f"\ncol_span_len:")
    print(B["col_span_len"])
    print("All should be 3")


def test_striped_grid():
    """Test on row-striped 4×4 grid."""
    print("\n" + "="*60)
    print("TEST 2: Row-striped 4×4 grid")
    print("="*60)

    grid = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2]
    ], dtype=int)

    print("Grid:")
    print(grid)

    B = compute_B_atoms(grid)

    # Row run-lengths: all should be 4
    print(f"\nrow_span_len:")
    print(B["row_span_len"])
    print("All should be 4")

    # Col run-lengths: top half = 2, bottom half = 2
    print(f"\ncol_span_len:")
    print(B["col_span_len"])
    print("All should be 2")

    # Col span_start/end
    print(f"\ncol_span_start:")
    print(B["col_span_start"])
    print("Top half should be 0, bottom half should be 2")

    print(f"\ncol_span_end:")
    print(B["col_span_end"])
    print("Top half should be 1, bottom half should be 3")


def test_complex_pattern():
    """Test on pattern with multiple runs."""
    print("\n" + "="*60)
    print("TEST 3: Complex pattern with multiple runs")
    print("="*60)

    grid = np.array([
        [1, 1, 2, 2, 2, 3],
    ], dtype=int)

    print("Grid (1 row):")
    print(grid)

    B = compute_B_atoms(grid)

    # Row spans: [0-1]→2, [2-4]→3, [5]→1
    print(f"\nrow_span_len:")
    print(B["row_span_len"])
    print("Expected: [2, 2, 3, 3, 3, 1]")

    print(f"\nrow_span_start:")
    print(B["row_span_start"])
    print("Expected: [0, 0, 2, 2, 2, 5]")

    print(f"\nrow_span_end:")
    print(B["row_span_end"])
    print("Expected: [1, 1, 4, 4, 4, 5]")

    # Verify
    expected_len = [2, 2, 3, 3, 3, 1]
    expected_start = [0, 0, 2, 2, 2, 5]
    expected_end = [1, 1, 4, 4, 4, 5]

    if np.array_equal(B["row_span_len"][0], expected_len):
        print("\n✅ row_span_len CORRECT")
    else:
        print(f"\n❌ row_span_len WRONG: got {B['row_span_len'][0].tolist()}")

    if np.array_equal(B["row_span_start"][0], expected_start):
        print("✅ row_span_start CORRECT")
    else:
        print(f"❌ row_span_start WRONG: got {B['row_span_start'][0].tolist()}")

    if np.array_equal(B["row_span_end"][0], expected_end):
        print("✅ row_span_end CORRECT")
    else:
        print(f"❌ row_span_end WRONG: got {B['row_span_end'][0].tolist()}")


if __name__ == "__main__":
    test_uniform_grid()
    test_striped_grid()
    test_complex_pattern()

    print("\n" + "="*60)
    print("Manual B-atoms tests complete!")
    print("="*60)
