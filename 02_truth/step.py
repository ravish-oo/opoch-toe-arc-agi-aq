"""
02_truth: Π — remove minted differences via canonical labeling.

Stage: truth
Builds disjoint-union graph and applies canonical labeling to establish shared gauge.
"""

from typing import Any, Dict, List, Tuple
import logging
import hashlib

import numpy as np
import igraph as ig


def canonicalize(present: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: truth (Π / canonical gauge) — union graph + canonical labeling

    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2: Stage A — Canonical labeling (awareness & gauge)
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π / kill minted differences

    Input:
      present: object from 01_present.load, containing train_in, train_out, test_in
      trace: if True, log vertex/edge counts, component count, and canonical hash

    Output:
      canonical: present + union graph G + canonicalized grids
        {
          "task_id": str,
          "train_in": [np.ndarray, ...],    # canonicalized
          "train_out": [np.ndarray, ...],   # canonicalized
          "test_in": [np.ndarray, ...],     # canonicalized
          "shapes": {...},
          "palettes": {...},
          "graph": ig.Graph,                # disjoint union of all grids
          "row_orders": [[int], ...],       # old→canonical row mapping per grid
          "col_orders": [[int], ...],       # old→canonical col mapping per grid
        }
    """
    if trace:
        logging.info("[truth] canonicalize() called")

    # Collect all grids in order: train_in, train_out, test_in
    grids: List[np.ndarray] = (
        present["train_in"]
        + present["train_out"]
        + present["test_in"]
    )

    num_train_in = len(present["train_in"])
    num_train_out = len(present["train_out"])

    # ========== WO-1.2: Build union graph ==========

    # Compute vertex count and offsets per grid
    vert_offsets: List[int] = []
    total_vertices = 0
    grid_shapes: List[Tuple[int, int]] = []

    for g in grids:
        H, W = int(g.shape[0]), int(g.shape[1])
        n_cells = H * W
        n_rows = H
        n_cols = W
        n_verts = n_cells + n_rows + n_cols
        vert_offsets.append(total_vertices)
        total_vertices += n_verts
        grid_shapes.append((H, W))

    # Initialize edge list and color codes
    edges: List[Tuple[int, int]] = []
    color_codes: List[int] = [0] * total_vertices

    # Build vertices and edges per grid
    for grid_idx, g in enumerate(grids):
        H, W = grid_shapes[grid_idx]
        base = vert_offsets[grid_idx]

        # Index layout (per grid, with base offset):
        # cells:     base + r*W + c         (0 to H*W-1)
        # row_nodes: base + H*W + r         (H*W to H*W+H-1)
        # col_nodes: base + H*W + H + c     (H*W+H to H*W+H+W-1)

        def cell_index(r: int, c: int) -> int:
            return base + r * W + c

        def row_node_index(r: int) -> int:
            return base + H * W + r

        def col_node_index(c: int) -> int:
            return base + H * W + H + c

        # Set color codes for cells (0-9), row_nodes (100), col_nodes (101)
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                color_codes[v] = int(g[r, c])  # 0..9 (NO grid_id!)

        for r in range(H):
            v = row_node_index(r)
            color_codes[v] = 100  # row_node marker

        for c in range(W):
            v = col_node_index(c)
            color_codes[v] = 101  # col_node marker

        # N4 adjacency edges: connect (r,c) to (r+1,c) down and (r,c+1) right
        # This yields undirected 4-adjacency without duplicates
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                if r + 1 < H:  # down neighbor
                    edges.append((v, cell_index(r + 1, c)))
                if c + 1 < W:  # right neighbor
                    edges.append((v, cell_index(r, c + 1)))

        # Incidence edges: cell↔row_node, cell↔col_node
        for r in range(H):
            for c in range(W):
                v = cell_index(r, c)
                edges.append((v, row_node_index(r)))
                edges.append((v, col_node_index(c)))

    # Build igraph Graph (undirected)
    g_all = ig.Graph(n=total_vertices, edges=edges, directed=False)
    g_all.vs["color_code"] = color_codes

    if trace:
        comp = g_all.components()
        n_comp = len(comp)
        num_grids = len(grids)
        logging.info(
            f"[truth] union graph built: |V|={g_all.vcount()}, |E|={g_all.ecount()}, "
            f"components={n_comp} (expected {num_grids} grids)"
        )

    # ========== WO-1.3: Canonical labeling ==========

    # Compute canonical permutation by vertex color
    perm = g_all.canonical_permutation(color=g_all.vs["color_code"])
    perm_list = list(perm)  # perm_list[canon_idx] = old_vertex_index

    # Build inverse: inv[old_idx] = canonical index
    n = g_all.vcount()
    inv = [0] * n
    for canon_idx, old_idx in enumerate(perm_list):
        inv[old_idx] = canon_idx

    # Extract canonical row/col orders per grid
    row_orders: List[List[int]] = []
    col_orders: List[List[int]] = []

    for grid_idx in range(len(grids)):
        H, W = grid_shapes[grid_idx]
        base = vert_offsets[grid_idx]

        def row_node_index(r: int) -> int:
            return base + H * W + r

        def col_node_index(c: int) -> int:
            return base + H * W + H + c

        # Rows: collect (canonical_index, original_r) pairs
        row_pairs = []
        for r in range(H):
            v = row_node_index(r)
            canon_idx_row = inv[v]
            row_pairs.append((canon_idx_row, r))

        # Sort by (canon_idx, original_r) for stable tie-break
        row_pairs.sort(key=lambda x: (x[0], x[1]))

        # Build R_X: old_r → canonical_r
        R_X = [0] * H
        for canon_r, (_, old_r) in enumerate(row_pairs):
            R_X[old_r] = canon_r
        row_orders.append(R_X)

        # Columns: collect (canonical_index, original_c) pairs
        col_pairs = []
        for c in range(W):
            v = col_node_index(c)
            canon_idx_col = inv[v]
            col_pairs.append((canon_idx_col, c))

        # Sort by (canon_idx, original_c) for stable tie-break
        col_pairs.sort(key=lambda x: (x[0], x[1]))

        # Build C_X: old_c → canonical_c
        C_X = [0] * W
        for canon_c, (_, old_c) in enumerate(col_pairs):
            C_X[old_c] = canon_c
        col_orders.append(C_X)

    # Remap grids to canonical coordinates
    canonical_grids: List[np.ndarray] = []
    for grid_idx, G in enumerate(grids):
        H, W = grid_shapes[grid_idx]
        R_X = np.array(row_orders[grid_idx], dtype=int)
        C_X = np.array(col_orders[grid_idx], dtype=int)

        # Compute inverse mapping: canonical→old
        inv_rows = np.argsort(R_X)  # inv_rows[new_r] = old_r
        inv_cols = np.argsort(C_X)  # inv_cols[new_c] = old_c

        # Remap grid
        G_canon = G[inv_rows, :][:, inv_cols]
        canonical_grids.append(G_canon)

    # Split canonical grids back into train_in, train_out, test_in
    canonical_train_in = canonical_grids[0:num_train_in]
    canonical_train_out = canonical_grids[num_train_in:num_train_in + num_train_out]
    canonical_test_in = canonical_grids[num_train_in + num_train_out:]

    # Build canonical object
    canonical: Dict[str, Any] = {
        "task_id": present["task_id"],
        "train_in": canonical_train_in,
        "train_out": canonical_train_out,
        "test_in": canonical_test_in,
        "shapes": present["shapes"],     # shapes unchanged by canonicalization
        "palettes": present["palettes"], # palettes unchanged
        "graph": g_all,
        "row_orders": row_orders,
        "col_orders": col_orders,
    }

    # Compute determinism hash
    if trace:
        h = hashlib.sha256()
        for G in canonical_train_in + canonical_train_out + canonical_test_in:
            h.update(G.tobytes())
        digest = h.hexdigest()[:16]
        logging.info(f"[truth] canonicalization hash={digest}")

    return canonical
