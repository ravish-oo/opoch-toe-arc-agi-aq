#!/usr/bin/env python3
"""
Manual test for WO-4.3: C-atoms verification

Tests basic semantics on synthetic grids:
  - Single pixel: area=1, perimeter=4
  - 2×2 block: area=4, perimeter=8
  - Full grid: perimeter=2H+2W
  - Ring detection and thickness
  - Area ranking
"""

import numpy as np
import sys
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
compute_C_atoms = atoms.compute_C_atoms


def _make_scaffold_info(H, W):
    """Create minimal scaffold_info with distance fields."""
    # Distance from borders (simple: distance to nearest edge)
    d_top = np.arange(H, dtype=int)[:, None].repeat(W, axis=1)
    d_bottom = np.arange(H - 1, -1, -1, dtype=int)[:, None].repeat(W, axis=1)
    d_left = np.arange(W, dtype=int)[None, :].repeat(H, axis=0)
    d_right = np.arange(W - 1, -1, -1, dtype=int)[None, :].repeat(H, axis=0)

    return {
        "shape": (H, W),
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
    }


def test_single_pixel():
    """Test single pixel: area=1, perimeter=4."""
    print("\n" + "=" * 60)
    print("TEST 1: Single pixel (1×1 grid, color 5)")
    print("=" * 60)

    grid = np.array([[5]], dtype=int)
    scaffold_info = _make_scaffold_info(1, 1)

    C = compute_C_atoms(grid, scaffold_info)

    # Should have 1 component for color 5
    assert 5 in C["components"]
    comps = C["components"][5]
    assert len(comps) == 1

    comp = comps[0]
    print(f"Component: {comp}")

    # Checks
    assert comp["area"] == 1, f"Expected area=1, got {comp['area']}"
    assert comp["perimeter_4"] == 4, f"Expected perimeter=4, got {comp['perimeter_4']}"
    assert comp["height"] == 1
    assert comp["width"] == 1
    assert comp["area_rank"] == 0  # only component

    print("✅ PASS: area=1, perimeter=4, height=1, width=1")


def test_2x2_block():
    """Test 2×2 block: area=4, perimeter=8."""
    print("\n" + "=" * 60)
    print("TEST 2: 2×2 block (all color 3)")
    print("=" * 60)

    grid = np.array([[3, 3],
                     [3, 3]], dtype=int)
    scaffold_info = _make_scaffold_info(2, 2)

    C = compute_C_atoms(grid, scaffold_info)

    comps = C["components"][3]
    assert len(comps) == 1

    comp = comps[0]
    print(f"Component: {comp}")

    # Checks
    assert comp["area"] == 4, f"Expected area=4, got {comp['area']}"
    assert comp["perimeter_4"] == 8, f"Expected perimeter=8, got {comp['perimeter_4']}"
    assert comp["height"] == 2
    assert comp["width"] == 2
    assert comp["height_minus_width"] == 0

    print("✅ PASS: area=4, perimeter=8, height=2, width=2")


def test_full_grid():
    """Test full 5×7 grid: perimeter=2H+2W=24."""
    print("\n" + "=" * 60)
    print("TEST 3: Full 5×7 grid (all color 1)")
    print("=" * 60)

    H, W = 5, 7
    grid = np.full((H, W), 1, dtype=int)
    scaffold_info = _make_scaffold_info(H, W)

    C = compute_C_atoms(grid, scaffold_info)

    comps = C["components"][1]
    assert len(comps) == 1

    comp = comps[0]
    print(f"Component: {comp}")

    expected_perim = 2 * H + 2 * W
    assert comp["area"] == H * W
    assert comp["perimeter_4"] == expected_perim, \
        f"Expected perimeter={expected_perim}, got {comp['perimeter_4']}"

    # Ring thickness: full grid touches all sides, min distance is 0 at edges
    # max(min distances) should be small (center has min=min(H/2, W/2))
    assert comp["ring_thickness_class"] is not None, "Full grid should be detected as ring"

    print(f"✅ PASS: area={H*W}, perimeter={expected_perim}, ring_thickness_class={comp['ring_thickness_class']}")


def test_border_ring():
    """Test 1-pixel-thick border ring around empty 5×5 grid."""
    print("\n" + "=" * 60)
    print("TEST 4: Border ring (1-pixel thick frame on 5×5)")
    print("=" * 60)

    grid = np.array([
        [2, 2, 2, 2, 2],
        [2, 0, 0, 0, 2],
        [2, 0, 0, 0, 2],
        [2, 0, 0, 0, 2],
        [2, 2, 2, 2, 2]
    ], dtype=int)
    scaffold_info = _make_scaffold_info(5, 5)

    C = compute_C_atoms(grid, scaffold_info)

    # Color 2: should be 1 component (ring)
    comps_2 = C["components"][2]
    assert len(comps_2) == 1

    comp = comps_2[0]
    print(f"Ring component: {comp}")

    expected_area = 16  # 5*5 - 3*3
    # Perimeter: outer (20 edges: 4 sides of 5×5) + inner (12 edges: 4 sides of 3×3)
    expected_perim = 20 + 12  # = 32

    assert comp["area"] == expected_area, f"Expected area={expected_area}, got {comp['area']}"
    assert comp["perimeter_4"] == expected_perim, \
        f"Expected perimeter={expected_perim}, got {comp['perimeter_4']}"

    # Ring touches all 4 sides
    assert comp["ring_thickness_class"] is not None, "Border ring should be detected"
    # With our distance calculation, ring thickness should be 1 (1 cell thick)
    print(f"  ring_thickness_class: {comp['ring_thickness_class']}")

    print(f"✅ PASS: border ring detected, area={expected_area}, perimeter={expected_perim}")


def test_multiple_components_ranking():
    """Test area ranking: multiple blobs of same color."""
    print("\n" + "=" * 60)
    print("TEST 5: Multiple components, area ranking")
    print("=" * 60)

    # 3 separate blobs of color 7: sizes 1, 4, 2
    grid = np.array([
        [7, 0, 0, 0, 0],
        [0, 0, 7, 7, 0],
        [0, 0, 7, 7, 0],
        [0, 0, 0, 0, 0],
        [0, 7, 7, 0, 0]
    ], dtype=int)
    scaffold_info = _make_scaffold_info(5, 5)

    C = compute_C_atoms(grid, scaffold_info)

    comps = C["components"][7]
    assert len(comps) == 3, f"Expected 3 components, got {len(comps)}"

    # Sort by area_rank to check ordering
    comps_sorted = sorted(comps, key=lambda c: c["area_rank"])
    areas = [c["area"] for c in comps_sorted]

    print(f"Components (sorted by rank): {[(c['area'], c['area_rank']) for c in comps_sorted]}")

    # Rank 0 should be largest
    assert areas[0] == 4, f"Rank 0 should have area=4, got {areas[0]}"
    assert areas[1] == 2, f"Rank 1 should have area=2, got {areas[1]}"
    assert areas[2] == 1, f"Rank 2 should have area=1, got {areas[2]}"

    print(f"✅ PASS: area ranking correct: {areas}")


def test_no_ring_detection():
    """Test that non-ring component gets ring_thickness_class=None."""
    print("\n" + "=" * 60)
    print("TEST 6: Non-ring component (doesn't touch all sides)")
    print("=" * 60)

    # Component touches top and left, but not bottom or right
    grid = np.array([
        [4, 4, 0],
        [4, 4, 0],
        [0, 0, 0]
    ], dtype=int)
    scaffold_info = _make_scaffold_info(3, 3)

    C = compute_C_atoms(grid, scaffold_info)

    comps = C["components"][4]
    assert len(comps) == 1

    comp = comps[0]
    print(f"Component: {comp}")

    # Should NOT be detected as ring (doesn't touch all 4 sides)
    assert comp["ring_thickness_class"] is None, \
        "Component should not be detected as ring"

    print("✅ PASS: non-ring component correctly has ring_thickness_class=None")


if __name__ == "__main__":
    test_single_pixel()
    test_2x2_block()
    test_full_grid()
    test_border_ring()
    test_multiple_components_ranking()
    test_no_ring_detection()

    print("\n" + "=" * 60)
    print("🎉 ALL MANUAL C-ATOMS TESTS PASSED!")
    print("=" * 60)
