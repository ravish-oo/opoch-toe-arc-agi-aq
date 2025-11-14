"""
05_laws/atoms.py — Atom derivations for laws

Anchor:
  - 00_MATH_SPEC.md §5.1: Atom universe

Implements:
  - WO-4.1: A-atoms (coordinates & distances)
  - WO-4.2: B-atoms (local texture)
  - WO-4.3: C-atoms (connectivity & shape)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy import ndimage
from skimage.measure import regionprops


def _divisors(n: int) -> np.ndarray:
    """
    Compute all divisors of n (including 1 and n).

    For H,W <= 30 (ARC grids), simple iteration is fine.

    Returns:
        np.ndarray of divisors in ascending order
    """
    if n < 1:
        raise ValueError(f"Cannot compute divisors of {n} (must be >= 1)")

    divs = []
    for d in range(1, n + 1):
        if n % d == 0:
            divs.append(d)

    return np.array(divs, dtype=int)


def compute_A_atoms(
    H: int,
    W: int,
    scaffold_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute A-atoms (coordinates & distances) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 A: Scaffold & coords

    Input:
      H, W: grid dimensions in canonical coords
      scaffold_info: per-output dict from 03_scaffold containing:
        - d_top, d_bottom, d_left, d_right: np.ndarray[(H,W), int]
        - has_midrow, has_midcol: bool (not used; we recompute from distances)

    Output:
      Dict containing:
        - H, W: int scalars
        - r, c: np.ndarray[(H,W), int] coordinate grids
        - r_plus_c, r_minus_c: np.ndarray[(H,W), int]
        - d_top/bottom/left/right: np.ndarray[(H,W), int] (from scaffold)
        - midrow_flag, midcol_flag: np.ndarray[(H,W), bool]
        - mod_r, mod_c: Dict[int, np.ndarray[(H,W), int]]
        - block_row, block_col: Dict[int, np.ndarray[(H,W), int]]

    Raises:
      NotImplementedError: if scaffold_info is None (test_out not yet supported)
      ValueError: if distance field shapes don't match (H,W)
    """
    # WO-4.1: Fail loudly if scaffold_info missing (test_out case)
    if scaffold_info is None:
        raise NotImplementedError(
            "A-atoms for test_out require scaffold distances; "
            "build them first in 03_scaffold."
        )

    # ========== 3.1 Basic coord arrays ==========
    r = np.arange(H, dtype=int)[:, None]  # shape (H,1)
    c = np.arange(W, dtype=int)[None, :]  # shape (1,W)

    r_grid = np.broadcast_to(r, (H, W)).copy()
    c_grid = np.broadcast_to(c, (H, W)).copy()

    r_plus_c = r_grid + c_grid
    r_minus_c = r_grid - c_grid

    # ========== 3.2 Distances (reuse from scaffold) ==========
    d_top = np.asarray(scaffold_info["d_top"], dtype=int)
    d_bottom = np.asarray(scaffold_info["d_bottom"], dtype=int)
    d_left = np.asarray(scaffold_info["d_left"], dtype=int)
    d_right = np.asarray(scaffold_info["d_right"], dtype=int)

    # Validate shapes
    if d_top.shape != (H, W):
        raise ValueError(
            f"d_top shape {d_top.shape} does not match grid shape ({H},{W})"
        )
    if d_bottom.shape != (H, W):
        raise ValueError(
            f"d_bottom shape {d_bottom.shape} does not match grid shape ({H},{W})"
        )
    if d_left.shape != (H, W):
        raise ValueError(
            f"d_left shape {d_left.shape} does not match grid shape ({H},{W})"
        )
    if d_right.shape != (H, W):
        raise ValueError(
            f"d_right shape {d_right.shape} does not match grid shape ({H},{W})"
        )

    # ========== 3.3 Midrow / midcol flags ==========
    # Spec: midrow cells are where d_top == d_bottom
    #       midcol cells are where d_left == d_right
    midrow_flag = (d_top == d_bottom)
    midcol_flag = (d_left == d_right)

    # ========== 4.1 Mod classes: exact m set ==========
    # Spec §5.1 A: m ∈ {2,…,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)
    max_dim = max(H, W)
    base_ms = list(range(2, min(6, max_dim) + 1))

    div_H = _divisors(H)
    div_W = _divisors(W)

    m_set = set(base_ms)
    m_set.update(div_H.tolist())
    m_set.update(div_W.tolist())
    m_set = {m for m in m_set if m >= 2}  # filter out mod 1 (useless)

    mod_r = {}
    mod_c = {}

    for m in sorted(m_set):
        mod_r[m] = r_grid % m
        mod_c[m] = c_grid % m

    # ========== 4.2 Block coords: exact b set ==========
    # Spec §5.1 A: b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W)
    # divisors(H,W) = divisors that divide BOTH H and W
    min_dim = min(H, W)
    base_bs = list(range(2, min(5, min_dim) + 1))

    div_H = _divisors(H)
    div_W = _divisors(W)
    div_HW = np.intersect1d(div_H, div_W)

    b_set = set(base_bs)
    b_set.update(div_HW.tolist())
    b_set = {b for b in b_set if b >= 2}  # filter out block size 1 (useless)

    block_row = {}
    block_col = {}

    for b in sorted(b_set):
        block_row[b] = r_grid // b
        block_col[b] = c_grid // b

    # ========== Return A-atoms dict ==========
    return {
        "H": H,
        "W": W,
        "r": r_grid,
        "c": c_grid,
        "r_plus_c": r_plus_c,
        "r_minus_c": r_minus_c,
        "d_top": d_top,
        "d_bottom": d_bottom,
        "d_left": d_left,
        "d_right": d_right,
        "midrow_flag": midrow_flag,
        "midcol_flag": midcol_flag,
        "mod_r": mod_r,
        "mod_c": mod_c,
        "block_row": block_row,
        "block_col": block_col,
    }


def trace_A_atoms(A_atoms: Dict[str, Any]) -> None:
    """
    Print trace summary of A-atoms for debugging.

    Called from 05_laws/step.py when trace=True.
    """
    H = A_atoms["H"]
    W = A_atoms["W"]

    print(f"[A-atoms] H,W = {H},{W}")
    print(f"[A-atoms] r_plus_c range: [{A_atoms['r_plus_c'].min()}, {A_atoms['r_plus_c'].max()}]")
    print(f"[A-atoms] r_minus_c range: [{A_atoms['r_minus_c'].min()}, {A_atoms['r_minus_c'].max()}]")
    print(f"[A-atoms] midrow_flag cells: {A_atoms['midrow_flag'].sum()}")
    print(f"[A-atoms] midcol_flag cells: {A_atoms['midcol_flag'].sum()}")
    print(f"[A-atoms] mod m keys: {sorted(A_atoms['mod_r'].keys())}")
    print(f"[A-atoms] block b keys: {sorted(A_atoms['block_row'].keys())}")


# ========== WO-4.2: B-atoms (Local Texture) ==========


def compute_B_atoms(grid: np.ndarray) -> Dict[str, Any]:
    """
    Compute B-atoms (local texture) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 B: Local texture

    Input:
      grid: np.ndarray[(H,W)] with colors 0..9 in canonical coords

    Output:
      Dict containing:
        - n4_counts: Dict[int, np.ndarray[(H,W), int]] - N4 neighbor counts per color
        - n8_counts: Dict[int, np.ndarray[(H,W), int]] - N8 neighbor counts per color
        - hash_3x3: np.ndarray[(H,W), int] - base-11 encoding of 3×3 neighborhood
        - ring_5x5: np.ndarray[(H,W), int] - base-11 encoding of 5×5 perimeter
        - row_span_len/start/end: np.ndarray[(H,W), int] - run-lengths along rows
        - col_span_len/start/end: np.ndarray[(H,W), int] - run-lengths along columns
    """
    H, W = grid.shape

    # ========== 3.1 N4 / N8 neighbor counts per color ==========
    # Grid-aware: only compute for colors present in this grid
    colors = np.unique(grid)

    # Define convolution kernels
    kernel_n4 = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=int)

    kernel_n8 = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=int)

    n4_counts = {}
    n8_counts = {}

    for k in colors:
        mask_k = (grid == k).astype(int)
        n4_counts[int(k)] = ndimage.convolve(mask_k, kernel_n4, mode="constant", cval=0)
        n8_counts[int(k)] = ndimage.convolve(mask_k, kernel_n8, mode="constant", cval=0)

    # ========== 3.2 3×3 full hash (base-11, sentinel=10) ==========
    SENT = 10

    # Pad grid with sentinel
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=SENT)

    # 9 positions in row-major order relative to center
    offsets_3x3 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Collect shifted views
    windows_3x3 = []
    for dr, dc in offsets_3x3:
        r0 = 1 + dr
        r1 = r0 + H
        c0 = 1 + dc
        c1 = c0 + W
        windows_3x3.append(padded[r0:r1, c0:c1])

    # Encode in base-11
    hash_3x3 = np.zeros((H, W), dtype=int)
    for w in windows_3x3:
        hash_3x3 = hash_3x3 * 11 + w

    # ========== 3.3 5×5 ring signature (perimeter only, sentinel=10) ==========
    # Pad grid with sentinel=10, pad 2 in each direction
    padded5 = np.pad(grid, pad_width=2, mode="constant", constant_values=SENT)

    # 16 perimeter positions of 5×5 window in fixed order:
    # Top row (5), Right col (3), Bottom row (5), Left col (3)
    offsets_ring = [
        # Top row
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        # Right col (excluding corners)
        (-1, 2), (0, 2), (1, 2),
        # Bottom row
        (2, 2), (2, 1), (2, 0), (2, -1), (2, -2),
        # Left col (excluding corners)
        (1, -2), (0, -2), (-1, -2),
    ]

    # Collect shifted views for ring
    ring_5x5 = np.zeros((H, W), dtype=int)
    for dr, dc in offsets_ring:
        r0 = 2 + dr
        r1 = r0 + H
        c0 = 2 + dc
        c1 = c0 + W
        patch = padded5[r0:r1, c0:c1]
        ring_5x5 = ring_5x5 * 11 + patch

    # ========== 3.4 Row / col run-lengths through each cell ==========
    # Row run-lengths
    row_span_len = np.zeros((H, W), dtype=int)
    row_span_start = np.zeros((H, W), dtype=int)
    row_span_end = np.zeros((H, W), dtype=int)

    for r in range(H):
        row_vals = grid[r, :]  # shape (W,)
        c = 0
        while c < W:
            k = row_vals[c]
            # Find end of this same-color run
            c2 = c + 1
            while c2 < W and row_vals[c2] == k:
                c2 += 1
            # Span is [c, c2)
            span_start = c
            span_end = c2 - 1
            span_len = span_end - span_start + 1
            row_span_start[r, span_start:span_end + 1] = span_start
            row_span_end[r, span_start:span_end + 1] = span_end
            row_span_len[r, span_start:span_end + 1] = span_len
            c = c2

    # Column run-lengths
    col_span_len = np.zeros((H, W), dtype=int)
    col_span_start = np.zeros((H, W), dtype=int)
    col_span_end = np.zeros((H, W), dtype=int)

    for c in range(W):
        col_vals = grid[:, c]  # shape (H,)
        r = 0
        while r < H:
            k = col_vals[r]
            # Find end of this same-color run
            r2 = r + 1
            while r2 < H and col_vals[r2] == k:
                r2 += 1
            # Span is [r, r2)
            span_start = r
            span_end = r2 - 1
            span_len = span_end - span_start + 1
            col_span_start[span_start:span_end + 1, c] = span_start
            col_span_end[span_start:span_end + 1, c] = span_end
            col_span_len[span_start:span_end + 1, c] = span_len
            r = r2

    # ========== Return B-atoms dict ==========
    return {
        "n4_counts": n4_counts,
        "n8_counts": n8_counts,
        "hash_3x3": hash_3x3,
        "ring_5x5": ring_5x5,
        "row_span_len": row_span_len,
        "row_span_start": row_span_start,
        "row_span_end": row_span_end,
        "col_span_len": col_span_len,
        "col_span_start": col_span_start,
        "col_span_end": col_span_end,
    }


def trace_B_atoms(B_atoms: Dict[str, Any], grid: np.ndarray) -> None:
    """
    Print trace summary of B-atoms for debugging.

    Called from 05_laws/step.py when trace=True.
    """
    H, W = grid.shape
    colors = np.unique(grid)

    print(f"[B-atoms] H,W = {H},{W}")
    print(f"[B-atoms] colors present: {colors.tolist()}")

    # Show one example color for neighbor counts
    if len(colors) > 0:
        example_k = int(colors[0])
        print(f"[B-atoms] n4_counts[{example_k}] sum = {B_atoms['n4_counts'][example_k].sum()}")
        print(f"[B-atoms] n8_counts[{example_k}] sum = {B_atoms['n8_counts'][example_k].sum()}")

    print(f"[B-atoms] hash_3x3 range: [{B_atoms['hash_3x3'].min()}, {B_atoms['hash_3x3'].max()}]")
    print(f"[B-atoms] ring_5x5 range: [{B_atoms['ring_5x5'].min()}, {B_atoms['ring_5x5'].max()}]")
    print(f"[B-atoms] row_span_len range: [{B_atoms['row_span_len'].min()}, {B_atoms['row_span_len'].max()}]")
    print(f"[B-atoms] col_span_len range: [{B_atoms['col_span_len'].min()}, {B_atoms['col_span_len'].max()}]")


# ========== WO-4.3: C-atoms (Connectivity & shape) ==========


def compute_C_atoms(grid: np.ndarray, scaffold_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute C-atoms (connectivity & shape) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 C: Connectivity & shape

    Input:
      grid: np.ndarray[(H,W)] with colors 0..9 in canonical coords
      scaffold_info: per-output dict from 03_scaffold containing:
        - d_top, d_bottom, d_left, d_right: np.ndarray[(H,W), int]

    Output:
      Dict containing:
        - components: Dict[int, List[Dict]] - per-color list of component stats
          Each component dict contains:
            - label: int (1..num_components for that color)
            - area: int
            - perimeter_4: int (4-edge perimeter)
            - bbox: (int, int, int, int) (r_min, c_min, r_max, c_max) inclusive
            - centroid_r: int (floor)
            - centroid_c: int (floor)
            - height: int
            - width: int
            - height_minus_width: int
            - area_rank: int (0 = largest area for that color)
            - ring_thickness_class: int or None (only if touches all 4 sides)
            - aspect_class: None (spec-listed but undefined; placeholder)
            - orientation_sign: None (spec-listed but undefined; placeholder)

    Raises:
      ValueError: if scaffold_info is None or shapes don't match
    """
    H, W = grid.shape

    # WO-4.3: Fail loudly if scaffold_info missing
    if scaffold_info is None:
        raise ValueError(
            "C-atoms require scaffold_info with distance fields; "
            "cannot compute ring_thickness_class without them."
        )

    # Validate distance field shapes
    d_top = np.asarray(scaffold_info["d_top"], dtype=int)
    d_bottom = np.asarray(scaffold_info["d_bottom"], dtype=int)
    d_left = np.asarray(scaffold_info["d_left"], dtype=int)
    d_right = np.asarray(scaffold_info["d_right"], dtype=int)

    if d_top.shape != (H, W):
        raise ValueError(f"d_top shape {d_top.shape} != grid shape ({H},{W})")
    if d_bottom.shape != (H, W):
        raise ValueError(f"d_bottom shape {d_bottom.shape} != grid shape ({H},{W})")
    if d_left.shape != (H, W):
        raise ValueError(f"d_left shape {d_left.shape} != grid shape ({H},{W})")
    if d_right.shape != (H, W):
        raise ValueError(f"d_right shape {d_right.shape} != grid shape ({H},{W})")

    # ========== 3.1 Per-color components (4-connectivity) ==========
    colors = np.unique(grid)

    # 4-connectivity structure
    structure_4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=int)

    components = {}

    for k in colors:
        mask = (grid == k).astype(np.uint8)

        # Label 4-connected components
        labeled, num_features = ndimage.label(mask, structure=structure_4)

        if num_features == 0:
            components[int(k)] = []
            continue

        # ========== 3.2 Compute 4-edge perimeter for this color ==========
        # Perimeter = count of edges between component pixels and background
        # For each direction, check if neighbor is different

        # Pad mask to handle borders (pad with 0 = background)
        mask_padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

        # Shift in 4 directions
        mask_up = mask_padded[:-2, 1:-1]     # shift down (neighbor above)
        mask_down = mask_padded[2:, 1:-1]    # shift up (neighbor below)
        mask_left = mask_padded[1:-1, :-2]   # shift right (neighbor left)
        mask_right = mask_padded[1:-1, 2:]   # shift left (neighbor right)

        # Edge exists where mask==1 and neighbor==0
        edges_up = mask & (mask_up == 0)
        edges_down = mask & (mask_down == 0)
        edges_left = mask & (mask_left == 0)
        edges_right = mask & (mask_right == 0)

        # Total edge count per pixel
        edge_count_per_pixel = (edges_up + edges_down + edges_left + edges_right).astype(int)

        # Sum edge counts per component label
        label_ids = np.arange(1, num_features + 1)
        perimeter_per_label = ndimage.sum(edge_count_per_pixel, labels=labeled, index=label_ids)

        # ========== 3.3 Use regionprops to extract component stats ==========
        props = regionprops(labeled)

        comps_for_k = []

        for prop in props:
            label_id = prop.label
            area = int(prop.area)

            # bbox: (min_row, min_col, max_row, max_col) with max exclusive
            minr, minc, maxr, maxc = prop.bbox
            r_min, c_min = minr, minc
            r_max, c_max = maxr - 1, maxc - 1  # convert to inclusive

            height = r_max - r_min + 1
            width = c_max - c_min + 1

            # centroid (int floor)
            centroid_r = int(np.floor(prop.centroid[0]))
            centroid_c = int(np.floor(prop.centroid[1]))

            height_minus_width = height - width

            # Perimeter from our 4-edge calculation
            perimeter_4 = int(perimeter_per_label[label_id - 1])

            # ========== 3.4 Ring thickness class ==========
            # Check if component touches all 4 sides
            comp_mask = (labeled == label_id)

            touches_top = comp_mask[0, :].any()
            touches_bottom = comp_mask[H - 1, :].any()
            touches_left = comp_mask[:, 0].any()
            touches_right = comp_mask[:, W - 1].any()

            ring_thickness_class = None
            if touches_top and touches_bottom and touches_left and touches_right:
                # Component is a ring touching all sides
                # Compute thickness via min distance to frame
                distances_min = np.minimum.reduce([
                    d_top[comp_mask],
                    d_bottom[comp_mask],
                    d_left[comp_mask],
                    d_right[comp_mask]
                ])
                ring_thickness_class = int(distances_min.max())

            comps_for_k.append({
                "label": label_id,
                "area": area,
                "perimeter_4": perimeter_4,
                "bbox": (r_min, c_min, r_max, c_max),
                "centroid_r": centroid_r,
                "centroid_c": centroid_c,
                "height": height,
                "width": width,
                "height_minus_width": height_minus_width,
                "area_rank": -1,  # will be set below
                "ring_thickness_class": ring_thickness_class,
                "aspect_class": None,  # spec-listed but undefined
                "orientation_sign": None,  # spec-listed but undefined
            })

        # ========== 3.5 Area rank (within color, 0 = largest) ==========
        areas = [comp["area"] for comp in comps_for_k]
        sorted_idx = np.argsort([-a for a in areas])  # descending order
        for rank, comp_idx in enumerate(sorted_idx):
            comps_for_k[comp_idx]["area_rank"] = rank

        components[int(k)] = comps_for_k

    # ========== Return C-atoms dict ==========
    return {
        "components": components,
    }


def trace_C_atoms(C_atoms: Dict[str, Any], grid: np.ndarray) -> None:
    """
    Print trace summary of C-atoms for debugging.

    Called from 05_laws/step.py when trace=True.
    """
    H, W = grid.shape
    colors = np.unique(grid)

    print(f"[C-atoms] H,W = {H},{W}")
    print(f"[C-atoms] colors present: {colors.tolist()}")

    components = C_atoms["components"]
    print(f"[C-atoms] components per color:")

    for k in sorted(components.keys()):
        comps = components[k]
        print(f"  color {k}: {len(comps)} component(s)")
        if comps:
            areas = [c["area"] for c in comps]
            perims = [c["perimeter_4"] for c in comps]
            ranks = [c["area_rank"] for c in comps]
            rings = [c["ring_thickness_class"] for c in comps]

            print(f"    areas: {areas}")
            print(f"    perimeters_4: {perims}")
            print(f"    area_ranks: {ranks}")
            print(f"    ring_thickness_class: {rings}")
