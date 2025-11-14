"""
05_laws/atoms.py — Atom derivations for laws

Anchor:
  - 00_MATH_SPEC.md §5.1: Atom universe

Implements:
  - WO-4.1: A-atoms (coordinates & distances)
  - WO-4.2: B-atoms (local texture)
  - WO-4.3: C-atoms (connectivity & shape)
  - WO-4.4: D-atoms (repetition & tiling) + E-atoms (palette/global)
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
    # CRITICAL: Compute EXACT grid-aware mod set from spec, do NOT hand-pick subset
    # Spec §5.1 A line 163: m ∈ {2,...,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)
    # This set is used in type keyer T(p) — must match spec exactly for correct mining
    max_dim = max(H, W)
    base_ms = list(range(2, min(6, max_dim) + 1))  # {2,...,min(6,max(H,W))}

    div_H = _divisors(H)
    div_W = _divisors(W)

    m_set = set(base_ms)           # Start with {2,...,min(6,max(H,W))}
    m_set.update(div_H.tolist())   # ∪ divisors(H)
    m_set.update(div_W.tolist())   # ∪ divisors(W)
    m_set = {m for m in m_set if m >= 2}  # filter out mod 1 (trivial)

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


# ========== WO-4.4: D-atoms (Repetition & tiling) ==========


def _least_period_1d(seq: np.ndarray) -> int:
    """
    Return smallest p (1 <= p <= L) such that seq is p-periodic.

    A sequence is p-periodic if seq[i] == seq[i+p] for all valid i.
    Equivalently: seq[:-p] == seq[p:].

    This is the RAW least period; S0 (size_choice) applies its own
    "real repetition" filter (2p <= L) when aggregating.

    Anchor:
      - 00_MATH_SPEC.md §5.1 D: Minimal period along row/col (≤ dimension)
    """
    L = len(seq)
    for p in range(1, L + 1):
        if p == L:
            # Full length is always a valid period
            return L
        # Check if seq is p-periodic: seq[:-p] == seq[p:]
        if np.array_equal(seq[:-p], seq[p:]):
            return p
    # Should never reach here due to p==L case above
    return L


def compute_D_atoms(grid: np.ndarray) -> Dict[str, Any]:
    """
    Compute D-atoms (repetition & tiling) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 D: Repetition & tiling

    Input:
      grid: np.ndarray[(H,W)] with colors 0..9 in canonical coords

    Output:
      Dict containing:
        - row_periods: np.ndarray[(H,), int] - minimal period for each row
        - col_periods: np.ndarray[(W,), int] - minimal period for each column
        - tiling_flags: Dict[(b_r, b_c), bool] - True if grid is exactly tiled
          by b_r×b_c blocks, where b_r divides H and b_c divides W
    """
    H, W = grid.shape

    # ========== 3.1 Minimal row / column periods ==========
    row_periods = np.zeros(H, dtype=int)
    col_periods = np.zeros(W, dtype=int)

    for r in range(H):
        row_periods[r] = _least_period_1d(grid[r, :])

    for c in range(W):
        col_periods[c] = _least_period_1d(grid[:, c])

    # ========== 3.2 2D tiling flags for factor pairs ==========
    # Compute divisors of H and W
    div_H = [d for d in range(1, H + 1) if H % d == 0]
    div_W = [d for d in range(1, W + 1) if W % d == 0]

    tiling_flags = {}

    for b_r in div_H:
        for b_c in div_W:
            # Extract canonical tile at top-left (0,0)
            tile = grid[0:b_r, 0:b_c].copy()

            # Check if all blocks match this tile
            ok = True
            for r0 in range(0, H, b_r):
                if not ok:
                    break
                for c0 in range(0, W, b_c):
                    block = grid[r0:r0 + b_r, c0:c0 + b_c]
                    if not np.array_equal(block, tile):
                        ok = False
                        break

            tiling_flags[(b_r, b_c)] = ok

    # ========== Return D-atoms dict ==========
    return {
        "row_periods": row_periods,
        "col_periods": col_periods,
        "tiling_flags": tiling_flags,
    }


# ========== WO-4.4: E-atoms (Palette/global) ==========


def compute_E_atoms_for_grid(
    grid: np.ndarray,
    C_atoms: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute E-atoms (palette/global) for a single output grid.

    Anchor:
      - 00_MATH_SPEC.md §5.1 E: Palette/global

    Input:
      grid: np.ndarray[(H,W)] with colors 0..9 in canonical coords
      C_atoms: optional C-atoms dict from compute_C_atoms (to reuse component counts)

    Output:
      Dict containing:
        - pixel_counts: np.ndarray[10, int] - count of each color 0..9
        - component_counts: Dict[int, int] - per-color component count
        - palette: List[int] - colors present (count > 0)
        - missing: List[int] - colors absent (count == 0)
        - most_frequent: List[int] - ALL colors achieving max count (>0)
        - least_frequent: List[int] - ALL colors achieving min positive count
    """
    # ========== 4.1 Per-color pixel counts ==========
    flat = grid.ravel()
    counts = np.bincount(flat, minlength=10)  # length 10, counts[k] = # pixels of color k

    # ========== 4.2 Palette present/missing ==========
    palette = [k for k in range(10) if counts[k] > 0]
    missing = [k for k in range(10) if counts[k] == 0]

    # ========== 4.3 Most/least frequent colors ==========
    # Most frequent: ALL colors achieving max count (>0)
    if len(palette) > 0:
        max_count = counts.max()
        most_frequent = [k for k in range(10) if counts[k] == max_count and counts[k] > 0]
    else:
        most_frequent = []

    # Least frequent: ALL colors achieving min positive count
    positive_counts = [(k, counts[k]) for k in range(10) if counts[k] > 0]
    if positive_counts:
        min_pos = min(c for k, c in positive_counts)
        least_frequent = [k for k, c in positive_counts if c == min_pos]
    else:
        least_frequent = []

    # ========== 4.4 Per-color component counts ==========
    # Reuse from C_atoms if available, else recompute
    if C_atoms is not None and "components" in C_atoms:
        component_counts = {k: len(C_atoms["components"].get(k, [])) for k in range(10)}
    else:
        # Recompute via ndimage.label
        component_counts = {}
        structure_4 = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=int)
        for k in range(10):
            mask = (grid == k).astype(np.uint8)
            if mask.any():
                _, num_features = ndimage.label(mask, structure=structure_4)
                component_counts[k] = int(num_features)
            else:
                component_counts[k] = 0

    # ========== Return E-atoms dict ==========
    return {
        "pixel_counts": counts,
        "component_counts": component_counts,
        "palette": palette,
        "missing": missing,
        "most_frequent": most_frequent,
        "least_frequent": least_frequent,
    }


def compute_global_palette_mapping(
    train_in: List[np.ndarray],
    train_out: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Compute task-level input↔output color mapping.

    Anchor:
      - 00_MATH_SPEC.md §5.1 E: Input↔output color permutation (bijective)
        & cyclic class over active palette

    Distinguishes two cases:

    Case A — True color permutation (gauge):
      - K_in == K_out (same palette)
      - f: K → K is a bijection on one set
      - Cycles ARE defined (permutation on single set)
      - This is gauge transformation (relabeling)

    Case B — Disjoint palettes (non-gauge color mapping):
      - K_in ≠ K_out (different palettes)
      - f: K_in → K_out is bijective but acts on two different sets
      - Cycles are NOT defined (no single set to permute)
      - This is real content transformation (not gauge)

    Input:
      train_in: list of input grids (canonical coords)
      train_out: list of output grids (canonical coords)

    Output:
      Dict containing:
        - has_bijection: bool - True if a consistent bijection exists
        - is_permutation: bool - True only if K_in == K_out (Case A)
        - perm: Dict[int, int] or None - only for Case A (true permutation)
        - cycles: List[List[int]] or None - only for Case A (cycle decomposition)
        - color_mapping: Dict[int, int] or None - only for Case B (disjoint palettes)

    Algorithm:
      1. For each train pair, check cell-by-cell if shapes match
      2. Build forward (cin -> cout) and reverse (cout -> cin) dicts
      3. Check consistency: each cin maps to exactly one cout, and vice versa
      4. If K_in == K_out: Case A (permutation + cycles)
      5. If K_in ≠ K_out: Case B (color_mapping, no cycles)
    """
    if len(train_in) != len(train_out):
        # Spec doesn't cover this case; fail loudly
        raise ValueError(
            f"train_in and train_out lengths mismatch: "
            f"{len(train_in)} != {len(train_out)}"
        )

    fwd = {}  # color_in -> color_out
    rev = {}  # color_out -> color_in
    consistent = True

    for i in range(len(train_in)):
        gi = train_in[i]
        go = train_out[i]

        # Check if shapes match for position-wise mapping
        if gi.shape != go.shape:
            # Cannot do position-wise mapping; no bijection possible
            consistent = False
            break

        # Flatten and check cell-by-cell
        gi_flat = gi.ravel()
        go_flat = go.ravel()

        for cin, cout in zip(gi_flat, go_flat):
            cin = int(cin)
            cout = int(cout)

            # Check forward consistency
            if cin in fwd:
                if fwd[cin] != cout:
                    consistent = False
                    break
            else:
                fwd[cin] = cout

            # Check reverse consistency (for bijectivity)
            if cout in rev:
                if rev[cout] != cin:
                    consistent = False
                    break
            else:
                rev[cout] = cin

        if not consistent:
            break

    if not consistent:
        return {
            "has_bijection": False,
            "is_permutation": False,
            "perm": None,
            "cycles": None,
            "color_mapping": None,
        }

    # Check bijectivity on active palettes
    palette_in = sorted(fwd.keys())
    palette_out = sorted(rev.keys())

    # For bijection: |palette_in| == |palette_out|
    if len(palette_in) != len(palette_out):
        return {
            "has_bijection": False,
            "is_permutation": False,
            "perm": None,
            "cycles": None,
            "color_mapping": None,
        }

    # ========== Distinguish Case A vs Case B ==========
    # Case A: True permutation (K_in == K_out)
    # Case B: Disjoint palettes (K_in ≠ K_out)

    if set(palette_in) == set(palette_out):
        # ========== Case A: True color permutation (gauge) ==========
        # K_in == K_out, so mapping is a permutation on ONE set
        # Cycles ARE defined

        # Build explicit permutation dict
        perm = {cin: fwd[cin] for cin in palette_in}

        # Compute cycle decomposition
        visited = set()
        cycles = []

        for start in palette_in:
            if start in visited:
                continue

            cycle = []
            x = start
            while x not in visited:
                visited.add(x)
                cycle.append(x)
                x = perm.get(x, x)  # follow permutation within same set

            if len(cycle) > 0:
                cycles.append(cycle)

        return {
            "has_bijection": True,
            "is_permutation": True,  # Case A
            "perm": perm,
            "cycles": cycles,
            "color_mapping": None,  # Not used in Case A
        }

    else:
        # ========== Case B: Disjoint palettes (non-gauge mapping) ==========
        # K_in ≠ K_out, so this is NOT a permutation on one set
        # Cycles are NOT defined (no single set to permute)
        # This is real content transformation (law), not gauge

        # Build color mapping (not permutation)
        color_mapping = {cin: fwd[cin] for cin in palette_in}

        return {
            "has_bijection": True,
            "is_permutation": False,  # Case B
            "perm": None,  # Not a permutation
            "cycles": None,  # Cycles not defined for disjoint palettes
            "color_mapping": color_mapping,
        }


def trace_D_E_atoms(
    D_atoms: Dict[str, Any],
    E_grid: Dict[str, Any],
    grid: np.ndarray,
    global_map: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print trace summary of D-atoms and E-atoms for debugging.

    Called from 05_laws/step.py when trace=True.
    """
    # If D_atoms and E_grid are provided (not empty), trace them
    if D_atoms and E_grid:
        H, W = grid.shape

        print(f"[D-atoms] H,W = {H},{W}")
        print(f"[D-atoms] row_periods: {D_atoms['row_periods'].tolist()}")
        print(f"[D-atoms] col_periods: {D_atoms['col_periods'].tolist()}")

        # Show only tiling flags that are True
        true_tilings = [(b_r, b_c) for (b_r, b_c), v in D_atoms["tiling_flags"].items() if v]
        print(f"[D-atoms] tiling_flags (True): {true_tilings}")

        print(f"[E-atoms:grid] palette: {E_grid['palette']}")
        print(f"[E-atoms:grid] pixel_counts: {E_grid['pixel_counts'].tolist()}")

        # Show only non-zero component counts
        nonzero_comps = {k: v for k, v in E_grid["component_counts"].items() if v > 0}
        print(f"[E-atoms:grid] component_counts (nonzero): {nonzero_comps}")

        print(f"[E-atoms:grid] most_frequent: {E_grid['most_frequent']}")
        print(f"[E-atoms:grid] least_frequent: {E_grid['least_frequent']}")

    # Trace global map if provided
    if global_map is not None:
        print(f"[E-atoms:global] has_bijection: {global_map['has_bijection']}")
        if global_map["has_bijection"]:
            print(f"[E-atoms:global] is_permutation: {global_map['is_permutation']}")
            if global_map["is_permutation"]:
                # Case A: True permutation (K_in == K_out)
                print(f"  perm: {global_map['perm']}")
                print(f"  cycles: {global_map['cycles']}")
            else:
                # Case B: Disjoint palettes (K_in ≠ K_out)
                print(f"  color_mapping: {global_map['color_mapping']}")
                print(f"  (Cycles not defined for disjoint palettes)")


# ============================================================================
# F24: Input feature mirror (WO-4.5)
# ============================================================================

# Module-level cache for input atoms (test_in is immutable per test_idx)
_input_atoms_cache: Dict[int, Dict[str, Any]] = {}


def get_input_atoms_for_test(
    canonical: Dict[str, Any],
    test_idx: int = 0,
) -> Dict[str, Any]:
    """
    F24: Mirror A–E atoms on inputs for evaluation of laws.

    Anchors:
      - 00_MATH_SPEC.md §5.1 F: "Mirror A–E on **inputs** to **evaluate predicates
        on test_in** **only when referenced by a mined law**. **F24 does not create
        new laws.** It never mines from inputs."
      - 01_STAGES.md: Stage 05_laws
      - 02_QUANTUM_MAPPING.md: (no input-driven laws)

    IMPORTANT GUARDRAIL:
      - NEVER called during mining from train_out.
      - ONLY used to plug input-dependent values into laws already mined.
      - Any call to this function in the mining path is a spec violation.

    Implementation:
      - Build scaffold on input grid using the SAME logic as Stage F (distances).
      - Reuse the same atom functions as for outputs: compute_A/B/C/D/E_atoms.
      - Cache result (test_in is immutable).

    Input:
      canonical: from 02_truth.canonicalize, containing:
        - test_in: List[np.ndarray] (canonical input grids)
      test_idx: which test input to compute atoms for (default 0)

    Output:
      atoms: {
        "A": A-atoms dict (scaffold geometry),
        "B": B-atoms dict (local texture),
        "C": C-atoms dict (connectivity & shape),
        "D": D-atoms dict (repetition & tiling),
        "E": E-atoms dict (palette/global),
      }

    Raises:
      IndexError: if test_idx out of range
      ValueError: if scaffold build fails
    """
    # Check cache first
    if test_idx in _input_atoms_cache:
        return _input_atoms_cache[test_idx]

    # Get test_in grid
    test_in_grids = canonical.get("test_in", [])
    if test_idx >= len(test_in_grids):
        raise IndexError(
            f"test_idx={test_idx} out of range for test_in "
            f"(len={len(test_in_grids)})"
        )

    grid = test_in_grids[test_idx]
    H, W = grid.shape

    # Build scaffold on input grid using the SAME logic as Stage F
    # Import here to avoid circular dependency
    from utils.scaffold_utils import build_scaffold_for_grid

    scaffold_info = build_scaffold_for_grid(grid)

    # Now reuse the same atom functions as for outputs
    # A-atoms: scaffold geometry
    A_atoms = compute_A_atoms(H=H, W=W, scaffold_info=scaffold_info)

    # B-atoms: local texture
    B_atoms = compute_B_atoms(grid)

    # C-atoms: connectivity & shape
    C_atoms = compute_C_atoms(grid, scaffold_info=scaffold_info)

    # D-atoms: repetition & tiling
    D_atoms = compute_D_atoms(grid)

    # E-atoms: palette/global (per-grid only, no task-level mapping)
    E_atoms = compute_E_atoms_for_grid(grid, C_atoms=C_atoms)

    atoms = {
        "A": A_atoms,
        "B": B_atoms,
        "C": C_atoms,
        "D": D_atoms,
        "E": E_atoms,
    }

    # Cache result
    _input_atoms_cache[test_idx] = atoms

    return atoms


# ============================================================================
# G-atoms: Component rigid/affine transforms (WO-4.6)
# ============================================================================


def _d4_variants(patch: np.ndarray):
    """
    Generate all 8 D4 (dihedral group) transforms of a 2D patch.

    D4 consists of:
      - 4 rotations (0°, 90°, 180°, 270°)
      - 4 reflections (horizontal, vertical, main diagonal, anti-diagonal)

    Yields:
      (op_name: str, transformed_patch: np.ndarray)
    """
    # Identity
    yield "id", patch

    # Rotations (counterclockwise)
    yield "rot90", np.rot90(patch, k=1)
    yield "rot180", np.rot90(patch, k=2)
    yield "rot270", np.rot90(patch, k=3)

    # Reflections
    yield "flip_h", np.flipud(patch)         # flip horizontal axis
    yield "flip_v", np.fliplr(patch)         # flip vertical axis
    yield "flip_d1", np.transpose(patch)     # flip main diagonal
    yield "flip_d2", np.fliplr(np.flipud(np.transpose(patch)))  # flip anti-diagonal


def compute_G_atoms(
    grid: np.ndarray,
    C_atoms: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute G-atoms (component rigid/affine transforms) for a single output grid.

    Anchors:
      - 00_MATH_SPEC.md §5.1 G: Component transforms
        "D4 (rot/ref) × integer scale (s) such that the transformed component
        fits inside the output grid; plus translations. Accept only **exact set
        equality** across paired components in train_out."
      - IMPLEMENTATION_PLAN.md WO-4.6

    Algorithm:
      1. Recover per-color component masks via ndimage.label (4-connectivity)
      2. For each color, group components into "templates" via D4×scale equivalence:
         - For each component, try to match against existing templates
         - Match = template scaled isotropically by s + D4 transform equals component exactly
         - If no match, create new template (component is its own canonical shape)
      3. Populate per-cell flags: template_id, local_r, local_c (offset from bbox)

    Input:
      grid: H×W canonical output grid (colors 0..9)
      C_atoms: result from compute_C_atoms on this grid, containing:
        {
          "components": {
            k: [
              {"label": int, "bbox": (r_min, c_min, r_max, c_max), ...},
              ...
            ]
          }
        }

    Output:
      {
        "templates": {
          k: [  # per color k, list of canonical template shapes
            {"bbox": (r0, c0, r1, c1), "mask": np.ndarray[(h,w), bool]},
            ...
          ]
        },
        "component_to_template": {
          k: [  # per component (same order as C_atoms["components"][k])
            {"template_idx": int, "scale": int, "d4_op": str},
            ...
          ]
        },
        "template_id": np.ndarray[(H,W), int],  # -1 if background
        "local_r": np.ndarray[(H,W), int],      # row offset in component bbox
        "local_c": np.ndarray[(H,W), int],      # col offset in component bbox
      }

    Raises:
      ValueError: if C_atoms missing required fields
    """
    H, W = grid.shape

    if C_atoms is None or "components" not in C_atoms:
        raise ValueError("G-atoms require C_atoms with 'components' field")

    components = C_atoms["components"]
    colors = np.unique(grid)

    # ========== 1. Recover per-color component label maps ==========
    # C_atoms has label values per component, but not full label array
    # Recompute via ndimage.label for mask extraction
    label_maps = {}  # k -> (labels: np.ndarray, num_labels: int)

    structure_4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=int)

    for k in colors:
        mask = (grid == k).astype(np.uint8)
        if mask.any():
            labels, num = ndimage.label(mask, structure=structure_4)
        else:
            labels, num = np.zeros((H, W), dtype=int), 0
        label_maps[int(k)] = (labels, num)

    # ========== 2. Build templates via D4×scale equivalence (per color) ==========
    templates = {}              # k -> list of template dicts
    component_to_template = {}  # k -> list of transform params (per component)

    for k in colors:
        k = int(k)
        if k not in components or len(components[k]) == 0:
            templates[k] = []
            component_to_template[k] = []
            continue

        labels_k, _ = label_maps[k]
        comps_k = components[k]

        templates_k = []
        comp_to_template_k = []

        for comp in comps_k:
            # Extract component mask patch
            r_min, c_min, r_max, c_max = comp["bbox"]
            label_val = comp["label"]

            # Component mask in bbox frame
            r0, c0, r1, c1 = r_min, c_min, r_max, c_max
            comp_labels_patch = labels_k[r0:r1 + 1, c0:c1 + 1]
            comp_mask_patch = (comp_labels_patch == label_val)

            h_comp, w_comp = comp_mask_patch.shape

            # Try to match against existing templates
            matched = False

            for t_idx, template in enumerate(templates_k):
                mask_T = template["mask"]
                hT, wT = mask_T.shape

                # Check if component can be a scaled version of template
                # Isotropic scaling only: s_h == s_w
                if h_comp % hT != 0 or w_comp % wT != 0:
                    continue

                s_h = h_comp // hT
                s_w = w_comp // wT

                if s_h != s_w:
                    continue  # Not isotropic

                s = s_h

                # Scale template via Kronecker product
                scaled_T = np.kron(mask_T.astype(np.uint8),
                                   np.ones((s, s), dtype=np.uint8)).astype(bool)

                # Try all D4 transforms on scaled template
                for op_name, transformed in _d4_variants(scaled_T):
                    if transformed.shape != comp_mask_patch.shape:
                        continue

                    if np.array_equal(transformed, comp_mask_patch):
                        # Exact match found!
                        comp_to_template_k.append({
                            "template_idx": t_idx,
                            "scale": s,
                            "d4_op": op_name,
                        })
                        matched = True
                        break  # Stop trying D4 ops for this template

                if matched:
                    break  # Stop trying other templates

            if not matched:
                # Create new template (component is its own canonical shape)
                new_idx = len(templates_k)
                templates_k.append({
                    "bbox": (0, 0, h_comp - 1, w_comp - 1),  # canonical frame
                    "mask": comp_mask_patch.copy(),
                })
                comp_to_template_k.append({
                    "template_idx": new_idx,
                    "scale": 1,
                    "d4_op": "id",
                })

        templates[k] = templates_k
        component_to_template[k] = comp_to_template_k

    # ========== 3. Populate per-cell flags ==========
    # template_id must be unique across ALL colors, so offset per color
    template_id = -np.ones((H, W), dtype=int)
    local_r = np.zeros((H, W), dtype=int)
    local_c = np.zeros((H, W), dtype=int)

    # Build global template ID offset for each color
    global_template_offset = {}
    next_global_id = 0
    for k in sorted(colors):  # Process in sorted order for determinism
        k = int(k)
        global_template_offset[k] = next_global_id
        if k in templates:
            next_global_id += len(templates[k])

    for k in colors:
        k = int(k)
        if k not in components or len(components[k]) == 0:
            continue

        labels_k, _ = label_maps[k]
        comps_k = components[k]
        offset = global_template_offset[k]

        for ci, comp in enumerate(comps_k):
            t_idx_local = component_to_template[k][ci]["template_idx"]
            t_idx_global = offset + t_idx_local  # Make globally unique

            r_min, c_min, r_max, c_max = comp["bbox"]
            label_val = comp["label"]

            # Find all pixels belonging to this component
            mask_ci = (labels_k == label_val)
            rs, cs = np.where(mask_ci)

            # Set per-cell flags
            template_id[rs, cs] = t_idx_global
            local_r[rs, cs] = rs - r_min
            local_c[rs, cs] = cs - c_min

    return {
        "templates": templates,
        "component_to_template": component_to_template,
        "template_id": template_id,
        "local_r": local_r,
        "local_c": local_c,
    }


def trace_G_atoms(G_atoms: Dict[str, Any], grid: np.ndarray) -> None:
    """
    Trace G-atoms for debugging.

    Shows template counts per color and unique template_id values.
    """
    H, W = grid.shape
    print(f"[G-atoms] grid shape: {H}×{W}")

    # Show template counts per color
    for k, templates_k in G_atoms["templates"].items():
        if len(templates_k) > 0:
            print(f"[G-atoms] color {k}: {len(templates_k)} template(s)")
            for ti, template in enumerate(templates_k):
                h, w = template["mask"].shape
                print(f"  template {ti}: shape ({h}×{w})")

    # Show unique template_id values (excluding -1 for background)
    unique_tids = np.unique(G_atoms["template_id"])
    unique_tids = unique_tids[unique_tids >= 0]  # Filter out background
    print(f"[G-atoms] unique template_id values: {unique_tids.tolist()}")


# ============================================================================
# Type Keys: T(p) per cell (WO-5.1)
# ============================================================================


def compute_type_keys_for_grid(
    A_atoms: Dict[str, Any],
    B_atoms: Dict[str, Any],
    D_atoms: Dict[str, Any],
    G_atoms: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute type key T(p) for each cell as a deterministic tuple of atoms.

    Anchors:
      - 00_MATH_SPEC.md §5.2: Type keyer
        "T(p) = (d_top, d_bottom, d_left, d_right, r±c, {r mod m}, {c mod m},
        3×3 hash, period flags, component shape ID)"
      - 01_STAGES.md: Stage 05_laws
      - IMPLEMENTATION_PLAN.md WO-5.1

    Type keys are used to:
      - Group "same-role" cells across grids
      - Mine unary fixes and relational equalities

    This function only DEFINES and COMPUTES T(p) and assigns stable integer IDs.
    Mining comes in later WOs (5.2, 5.3).

    Input:
      A_atoms: from compute_A_atoms, containing:
        - d_top, d_bottom, d_left, d_right: (H,W) int
        - r_plus_c, r_minus_c: (H,W) int
        - mod_r, mod_c: dict[int -> (H,W) int]
      B_atoms: from compute_B_atoms, containing:
        - hash_3x3: (H,W) int
      D_atoms: from compute_D_atoms, containing:
        - row_periods: (H,) int
        - col_periods: (W,) int
      G_atoms: from compute_G_atoms, containing:
        - template_id: (H,W) int (-1 for background)

    Output:
      {
        "type_tuple": np.ndarray[(H,W), object],  # Python tuples per cell
        "type_id": np.ndarray[(H,W), int],        # stable IDs 0..(K-1)
        "types": List[Tuple],                     # unique type tuples in sorted order
      }

    Raises:
      ValueError: if required fields missing
    """
    # Validate inputs
    if "d_top" not in A_atoms:
        raise ValueError("A_atoms missing required field 'd_top'")
    if "hash_3x3" not in B_atoms:
        raise ValueError("B_atoms missing required field 'hash_3x3'")
    if "row_periods" not in D_atoms:
        raise ValueError("D_atoms missing required field 'row_periods'")
    if "template_id" not in G_atoms:
        raise ValueError("G_atoms missing required field 'template_id'")

    H, W = A_atoms["d_top"].shape

    # Extract atom arrays
    d_top = A_atoms["d_top"]
    d_bottom = A_atoms["d_bottom"]
    d_left = A_atoms["d_left"]
    d_right = A_atoms["d_right"]
    r_plus_c = A_atoms["r_plus_c"]
    r_minus_c = A_atoms["r_minus_c"]
    mod_r = A_atoms["mod_r"]
    mod_c = A_atoms["mod_c"]

    hash_3x3 = B_atoms["hash_3x3"]

    row_periods = D_atoms["row_periods"]
    col_periods = D_atoms["col_periods"]

    template_id = G_atoms["template_id"]

    # Get sorted mod class set (grid-aware, deterministic)
    # CRITICAL: Use ALL m from A-atoms, do NOT subset
    # Spec §5.1 A line 163: m ∈ {2,...,min(6,max(H,W))} ∪ divisors(H) ∪ divisors(W)
    ms = sorted(mod_r.keys())

    # Build type tuple for each cell
    # CRITICAL: Include ALL 12 fields from spec §5.2 line 205, do NOT hand-pick
    type_tuple = np.empty((H, W), dtype=object)

    for r in range(H):
        for c in range(W):
            # 1. Distances (4 ints) — §5.1 A via Stage F
            dt = int(d_top[r, c])
            db = int(d_bottom[r, c])
            dl = int(d_left[r, c])
            dr = int(d_right[r, c])

            # 2. Diagonal coords (2 ints) — §5.1 A
            rp = int(r_plus_c[r, c])
            rm = int(r_minus_c[r, c])

            # 3. Mod classes (2 tuples) — §5.1 A
            # Both mod_r AND mod_c required (independent residue classes)
            mod_r_vals = tuple(int(mod_r[m][r, c]) for m in ms)
            mod_c_vals = tuple(int(mod_c[m][r, c]) for m in ms)

            # 4. 3×3 hash (1 int) — §5.1 B
            h3 = int(hash_3x3[r, c])

            # 5. Period flags (2 ints) — §5.1 D
            row_p = int(row_periods[r])
            col_p = int(col_periods[c])

            # 6. Component shape ID (1 int, -1 OK for background) — §5.1 G
            shape_id = int(template_id[r, c])

            # Assemble type tuple — EXACT ORDER from spec §5.2 line 205
            # T(p) = (d_top, d_bottom, d_left, d_right, r±c, {r mod m}, {c mod m},
            #         3×3 hash, period flags, component shape ID)
            T_p = (
                dt, db, dl, dr,           # Distances
                rp, rm,                   # Diagonal coords
                mod_r_vals, mod_c_vals,   # Mod classes (BOTH tuples)
                h3,                       # Hash
                row_p, col_p,             # Periods
                shape_id,                 # Template ID
            )

            type_tuple[r, c] = T_p

    # Compute stable type IDs via lexicographic sort
    # 1. Collect unique types
    flat = type_tuple.ravel().tolist()
    unique_types = sorted(set(flat))  # Lexicographic sort (deterministic)

    # 2. Build mapping
    type_to_id = {t: i for i, t in enumerate(unique_types)}

    # 3. Build type_id array
    type_id = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            type_id[r, c] = type_to_id[type_tuple[r, c]]

    return {
        "type_tuple": type_tuple,
        "type_id": type_id,
        "types": unique_types,
    }


def trace_type_keys(type_keys: Dict[str, Any]) -> None:
    """
    Trace type keys for debugging.

    Shows:
      - Number of unique types
      - First 5 type tuples
      - Histogram of type_id usage
    """
    type_id = type_keys["type_id"]
    types = type_keys["types"]

    print(f"[T] num_types: {len(types)}")

    # Show first 5 type tuples
    for i, t in enumerate(types[:5]):
        print(f"[T] type {i}: {t}")

    # Histogram of type usage
    unique, counts = np.unique(type_id, return_counts=True)
    print(f"[T] type_id histogram (first 10): {list(zip(unique[:10].tolist(), counts[:10].tolist()))}")
