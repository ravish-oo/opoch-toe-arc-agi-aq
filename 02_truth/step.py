"""
02_truth: Π — remove minted differences via canonical labeling.

Stage: truth
Builds disjoint-union graph for canonical labeling.
"""

from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import igraph as ig


def canonicalize(present: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
    """
    Stage: truth (Π / canonical gauge) — union graph construction

    Anchor:
      - 01_STAGES.md: truth
      - 00_MATH_SPEC.md §2.1: Disjoint-union graph G
      - 02_QUANTUM_MAPPING.md: 'truth' = apply Π / build canonicalizable structure

    Input:
      present: object from 01_present.load, containing train_in, train_out, test_in
      trace: if True, log vertex/edge counts and component count

    Output:
      canonical: present + union graph G. No canonical row/col orders yet.
        {
          "task_id": str,
          "train_in": [np.ndarray, ...],
          "train_out": [np.ndarray, ...],
          "test_in": [np.ndarray, ...],
          "shapes": {...},
          "palettes": {...},
          "graph": ig.Graph,       # disjoint union of all grids
          "row_orders": None,      # to be filled by WO-1.3
          "col_orders": None,      # to be filled by WO-1.3
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

    # Build canonical object
    canonical: Dict[str, Any] = {
        "task_id": present["task_id"],
        "train_in": present["train_in"],
        "train_out": present["train_out"],
        "test_in": present["test_in"],
        "shapes": present["shapes"],
        "palettes": present["palettes"],
        "graph": g_all,
        # Placeholders for WO-1.3 (canonical labeling)
        "row_orders": None,
        "col_orders": None,
    }

    return canonical
