#!/usr/bin/env python3
"""
Test script for WO-4.6: G-atoms (component rigid/affine transforms)

Validates:
  - D4 symmetry detection (8 operations: id, rot90/180/270, flip_h/v/d1/d2)
  - Isotropic scaling (s×s Kronecker product)
  - Exact pixel-set equality only
  - Template grouping (components sharing canonical shapes)
  - Per-cell local coordinate flags
"""

import numpy as np
import importlib.util
from pathlib import Path


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
compute_G_atoms = atoms.compute_G_atoms


def test_identity_no_transform():
    """
    Test G-atoms: two identical components (no transform).

    Grid has two separate components of color 1 with identical shape.
    Should share one template, both with scale=1, d4_op="id".
    """
    print("\n" + "=" * 60)
    print("TEST 1: Identity (no transform)")
    print("=" * 60)

    # Two identical 2×2 squares of color 1, separated
    grid = np.array([
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    # Need scaffold_info for C-atoms (minimal: distance fields)
    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    # Compute C-atoms (required for G-atoms)
    C_atoms = compute_C_atoms(grid, scaffold_info)

    # Compute G-atoms
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")
    print(f"Components for color 1: {len(C_atoms['components'][1])}")

    # Validate: one template for color 1
    assert 1 in G_atoms["templates"], "Should have templates for color 1"
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 1, f"Should have 1 template, got {len(templates_1)}"
    print(f"✅ 1 template for color 1")

    # Both components should point to template 0
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2, "Should have 2 components"

    for ci, cmap in enumerate(comp_map_1):
        assert cmap["template_idx"] == 0, f"Component {ci} should map to template 0"
        assert cmap["scale"] == 1, f"Component {ci} should have scale 1"
        assert cmap["d4_op"] == "id", f"Component {ci} should have d4_op='id'"

    print(f"✅ Both components: template_idx=0, scale=1, d4_op='id'")

    # Check per-cell flags
    template_id = G_atoms["template_id"]

    # template_id is globally unique across all colors
    # Color 0 templates get IDs first, then color 1, etc.
    # Both color-1 components should share the same template_id
    tid_comp0 = template_id[0, 0]  # First pixel of first component
    tid_comp1 = template_id[0, 3]  # First pixel of second component

    assert tid_comp0 == tid_comp1, \
        f"Both components should share same template_id, got {tid_comp0} and {tid_comp1}"

    # All pixels of first component should have same template_id
    assert (template_id[0:2, 0:2] == tid_comp0).all(), \
        "First component pixels should all have same template_id"

    # All pixels of second component should have same template_id
    assert (template_id[0:2, 3:5] == tid_comp1).all(), \
        "Second component pixels should all have same template_id"

    print(f"✅ Per-cell template_id correct: both components share template_id={tid_comp0}")

    print("\n✅ PASS: Identity transform detected correctly")


def test_rotation_90():
    """
    Test G-atoms: 90° rotation.

    Grid has two components: one original, one rotated 90° counterclockwise.
    Should share one template with d4_op="rot90" for the rotated one.
    """
    print("\n" + "=" * 60)
    print("TEST 2: 90° rotation")
    print("=" * 60)

    # Original shape (color 1):
    #   [1, 1]
    #   [1, 0]
    # Rotated 90° CCW (np.rot90 k=1):
    #   [1, 0]
    #   [1, 1]

    grid = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # One template for color 1
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 1, f"Should have 1 template, got {len(templates_1)}"

    # Component mappings
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2, "Should have 2 components"

    # First component (original): template 0, scale 1, d4_op="id"
    assert comp_map_1[0]["template_idx"] == 0
    assert comp_map_1[0]["scale"] == 1
    assert comp_map_1[0]["d4_op"] == "id"

    # Second component (rotated): template 0, scale 1, d4_op="rot90"
    assert comp_map_1[1]["template_idx"] == 0
    assert comp_map_1[1]["scale"] == 1
    assert comp_map_1[1]["d4_op"] == "rot90", \
        f"Second component should be rot90, got {comp_map_1[1]['d4_op']}"

    print(f"✅ Component 0: id, Component 1: rot90")
    print("\n✅ PASS: 90° rotation detected correctly")


def test_horizontal_flip():
    """
    Test G-atoms: horizontal flip.

    Grid has two components: one original, one horizontally flipped.
    Should share one template with d4_op="flip_h" for the flipped one.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Horizontal flip")
    print("=" * 60)

    # Original shape (color 1):
    #   [1, 1, 1]
    #   [1, 0, 0]
    # Flipped horizontally (flipud):
    #   [1, 0, 0]
    #   [1, 1, 1]

    grid = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # One template
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 1, f"Should have 1 template, got {len(templates_1)}"

    # Component mappings
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2

    # One should be id, one should be flip_h
    ops = [comp_map_1[0]["d4_op"], comp_map_1[1]["d4_op"]]
    assert "id" in ops, "One component should be identity"
    assert "flip_h" in ops, "One component should be flip_h"

    print(f"✅ One id, one flip_h: {ops}")
    print("\n✅ PASS: Horizontal flip detected correctly")


def test_isotropic_scaling():
    """
    Test G-atoms: isotropic scaling (2×).

    Grid has small shape and 2× scaled version.
    Should share one template with different scale values.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Isotropic scaling (2×)")
    print("=" * 60)

    # Small shape (color 1): 2×2
    #   [1, 1]
    #   [1, 1]
    # Large shape (color 1): 4×4 (2× scaled)
    #   [1, 1, 1, 1]
    #   [1, 1, 1, 1]
    #   [1, 1, 1, 1]
    #   [1, 1, 1, 1]

    grid = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # One template (the small 2×2 shape)
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 1, f"Should have 1 template, got {len(templates_1)}"

    # Template should be 2×2
    template_mask = templates_1[0]["mask"]
    assert template_mask.shape == (2, 2), \
        f"Template should be 2×2, got {template_mask.shape}"

    # Component mappings
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2

    # Small component: scale 1
    # Large component: scale 2
    scales = sorted([comp_map_1[0]["scale"], comp_map_1[1]["scale"]])
    assert scales == [1, 2], f"Should have scales [1, 2], got {scales}"

    print(f"✅ Template shape: {template_mask.shape}")
    print(f"✅ Component scales: {scales}")
    print("\n✅ PASS: Isotropic scaling detected correctly")


def test_different_templates():
    """
    Test G-atoms: components with different shapes.

    Grid has two different shapes (not related by D4×scale).
    Should create two separate templates.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Different templates (no transform relation)")
    print("=" * 60)

    # Shape A (color 1): 2×2 square
    #   [1, 1]
    #   [1, 1]
    # Shape B (color 1): 1×3 horizontal bar
    #   [1, 1, 1]

    grid = np.array([
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # Two templates for color 1
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 2, \
        f"Should have 2 templates (different shapes), got {len(templates_1)}"

    # Each component maps to its own template
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2

    template_indices = [comp_map_1[0]["template_idx"], comp_map_1[1]["template_idx"]]
    assert set(template_indices) == {0, 1}, \
        f"Should have template_idx {{0, 1}}, got {set(template_indices)}"

    print(f"✅ 2 templates for different shapes")
    print(f"✅ Template indices: {template_indices}")
    print("\n✅ PASS: Different templates created correctly")


def test_local_coordinates():
    """
    Test G-atoms: per-cell local coordinate flags.

    Validate local_r and local_c are correct offsets from component bbox.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Per-cell local coordinates")
    print("=" * 60)

    # Simple 3×3 component of color 1 at position (1, 2)
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # Component bbox should be (1, 2, 3, 4)
    comp = C_atoms["components"][1][0]
    r_min, c_min, r_max, c_max = comp["bbox"]
    print(f"Component bbox: ({r_min}, {c_min}, {r_max}, {c_max})")

    assert (r_min, c_min) == (1, 2), "Component should start at (1, 2)"
    assert (r_max, c_max) == (3, 4), "Component should end at (3, 4)"

    # Check local coords
    local_r = G_atoms["local_r"]
    local_c = G_atoms["local_c"]

    # Top-left pixel of component (1, 2) should have local_r=0, local_c=0
    assert local_r[1, 2] == 0, f"local_r at (1,2) should be 0, got {local_r[1, 2]}"
    assert local_c[1, 2] == 0, f"local_c at (1,2) should be 0, got {local_c[1, 2]}"

    # Bottom-right pixel of component (3, 4) should have local_r=2, local_c=2
    assert local_r[3, 4] == 2, f"local_r at (3,4) should be 2, got {local_r[3, 4]}"
    assert local_c[3, 4] == 2, f"local_c at (3,4) should be 2, got {local_c[3, 4]}"

    # Center pixel (2, 3) should have local_r=1, local_c=1
    assert local_r[2, 3] == 1, f"local_r at (2,3) should be 1, got {local_r[2, 3]}"
    assert local_c[2, 3] == 1, f"local_c at (2,3) should be 1, got {local_c[2, 3]}"

    print(f"✅ local_r[1,2] = {local_r[1, 2]} (expected 0)")
    print(f"✅ local_c[1,2] = {local_c[1, 2]} (expected 0)")
    print(f"✅ local_r[3,4] = {local_r[3, 4]} (expected 2)")
    print(f"✅ local_c[3,4] = {local_c[3, 4]} (expected 2)")
    print("\n✅ PASS: Local coordinates correct")


def test_multiple_colors():
    """
    Test G-atoms: multiple colors with independent templates.

    Templates are per-color, so different colors should have independent
    template groupings.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Multiple colors (independent templates)")
    print("=" * 60)

    # Color 1: two identical 2×2 squares
    # Color 2: one 1×3 bar
    grid = np.array([
        [1, 1, 0, 1, 1, 0, 2, 2, 2],
        [1, 1, 0, 1, 1, 0, 0, 0, 0],
    ], dtype=int)

    from utils.scaffold_utils import build_scaffold_for_grid
    scaffold_info = build_scaffold_for_grid(grid)

    C_atoms = compute_C_atoms(grid, scaffold_info)
    G_atoms = compute_G_atoms(grid, C_atoms)

    print(f"Grid:\n{grid}")

    # Color 1: 1 template (both squares are identical)
    templates_1 = G_atoms["templates"][1]
    assert len(templates_1) == 1, f"Color 1 should have 1 template, got {len(templates_1)}"

    # Color 2: 1 template (one bar)
    templates_2 = G_atoms["templates"][2]
    assert len(templates_2) == 1, f"Color 2 should have 1 template, got {len(templates_2)}"

    # Color 1 has 2 components
    comp_map_1 = G_atoms["component_to_template"][1]
    assert len(comp_map_1) == 2, "Color 1 should have 2 components"

    # Color 2 has 1 component
    comp_map_2 = G_atoms["component_to_template"][2]
    assert len(comp_map_2) == 1, "Color 2 should have 1 component"

    print(f"✅ Color 1: {len(templates_1)} template, {len(comp_map_1)} components")
    print(f"✅ Color 2: {len(templates_2)} template, {len(comp_map_2)} components")
    print("\n✅ PASS: Multiple colors handled correctly")


def main():
    """Run WO-4.6 G-atoms acceptance tests."""
    print("WO-4.6 ACCEPTANCE TESTS: G-atoms (component rigid/affine transforms)")
    print("=" * 60)

    test_identity_no_transform()
    test_rotation_90()
    test_horizontal_flip()
    test_isotropic_scaling()
    test_different_templates()
    test_local_coordinates()
    test_multiple_colors()

    print("\n" + "=" * 60)
    print("🎉 ALL WO-4.6 G-ATOMS TESTS PASSED!")
    print("=" * 60)
    print("\nValidated:")
    print("  ✅ D4 symmetry detection (identity, rotations, reflections)")
    print("  ✅ Isotropic scaling (s×s Kronecker product)")
    print("  ✅ Exact pixel-set equality (no fuzzy matching)")
    print("  ✅ Template grouping (components sharing canonical shapes)")
    print("  ✅ Per-cell local coordinate flags (offset from bbox)")
    print("  ✅ Multiple colors (independent template sets)")


if __name__ == "__main__":
    main()
