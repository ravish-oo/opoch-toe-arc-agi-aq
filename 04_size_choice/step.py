"""
04_size_choice: Infer test output size from training pairs + scaffold screening.

Stage: size_choice
Learns size maps from training, screens candidates with scaffold facts.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np


def _collect_train_size_pairs(canonical: Dict[str, Any]) -> List[Dict[str, int]]:
    """
    Extract (H_in, W_in, H_out, W_out) for each training pair.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Candidate size maps must reproduce all training pairs

    Input:
      canonical: from 02_truth.canonicalize, with train_in and train_out lists

    Output:
      List of dicts with keys: H_in, W_in, H_out, W_out
    """
    train_in = canonical["train_in"]
    train_out = canonical["train_out"]

    if len(train_in) == 0 or len(train_out) == 0:
        raise ValueError("[size_choice] No training pairs; cannot infer size map.")

    if len(train_in) != len(train_out):
        raise ValueError(
            f"[size_choice] Mismatched training counts: "
            f"{len(train_in)} inputs vs {len(train_out)} outputs."
        )

    pairs = []
    for i, (grid_in, grid_out) in enumerate(zip(train_in, train_out)):
        H_in, W_in = grid_in.shape
        H_out, W_out = grid_out.shape
        pairs.append({
            "H_in": int(H_in),
            "W_in": int(W_in),
            "H_out": int(H_out),
            "W_out": int(W_out),
        })

    return pairs


def _build_reproductions(
    train_pairs: List[Dict[str, int]],
    predict_fn: callable
) -> tuple:
    """
    Build reproduction table by applying predict_fn to each training pair.

    Returns: (reproductions: List[dict], fits_all: bool)
    """
    reproductions = []
    fits_all = True

    for pair in train_pairs:
        H_in = pair["H_in"]
        W_in = pair["W_in"]
        H_out_true = pair["H_out"]
        W_out_true = pair["W_out"]

        H_out_pred, W_out_pred = predict_fn(H_in, W_in)

        match = (H_out_pred == H_out_true) and (W_out_pred == W_out_true)
        if not match:
            fits_all = False

        reproductions.append({
            "H_in": H_in,
            "W_in": W_in,
            "H_out_pred": int(H_out_pred),
            "W_out_pred": int(W_out_pred),
            "H_out_true": H_out_true,
            "W_out_true": W_out_true,
            "match": match,
        })

    return reproductions, fits_all


def _enumerate_identity_candidate(
    train_pairs: List[Dict[str, int]]
) -> Optional[Dict[str, Any]]:
    """
    Identity: H_out == H_in, W_out == W_in for all pairs.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Identity / swap family
    """
    def predict(H_in, W_in):
        return H_in, W_in

    reproductions, fits_all = _build_reproductions(train_pairs, predict)

    if not fits_all:
        return None

    return {
        "family": "identity",
        "params": {},
        "fits_all": True,
        "reproductions": reproductions,
    }


def _enumerate_swap_candidate(
    train_pairs: List[Dict[str, int]]
) -> Optional[Dict[str, Any]]:
    """
    Swap: H_out == W_in, W_out == H_in for all pairs.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Identity / swap family
    """
    def predict(H_in, W_in):
        return W_in, H_in

    reproductions, fits_all = _build_reproductions(train_pairs, predict)

    if not fits_all:
        return None

    return {
        "family": "swap",
        "params": {},
        "fits_all": True,
        "reproductions": reproductions,
    }


def _enumerate_factor_candidate(
    train_pairs: List[Dict[str, int]]
) -> Optional[Dict[str, Any]]:
    """
    Factor maps: H_out = r_H * H_in, W_out = r_W * W_in (integer r_H, r_W).

    Anchors:
      - 00_MATH_SPEC.md §3.1: Factor maps family
    """
    # Check divisibility and extract factors
    r_H_values = []
    r_W_values = []

    for pair in train_pairs:
        H_in = pair["H_in"]
        W_in = pair["W_in"]
        H_out = pair["H_out"]
        W_out = pair["W_out"]

        if H_in == 0 or W_in == 0:
            raise ValueError(
                f"[size_choice] Zero dimension in training pair: "
                f"H_in={H_in}, W_in={W_in}; unsupported."
            )

        # Check if H_out and W_out are exact multiples
        if H_out % H_in != 0 or W_out % W_in != 0:
            return None  # Not a factor map

        r_H = H_out // H_in
        r_W = W_out // W_in

        r_H_values.append(r_H)
        r_W_values.append(r_W)

    # Check if all factors are consistent
    if len(set(r_H_values)) != 1 or len(set(r_W_values)) != 1:
        return None  # Inconsistent factors

    r_H = r_H_values[0]
    r_W = r_W_values[0]

    # Build reproductions
    def predict(H_in, W_in):
        return r_H * H_in, r_W * W_in

    reproductions, fits_all = _build_reproductions(train_pairs, predict)

    # Should always fit if we got here, but verify
    if not fits_all:
        return None

    return {
        "family": "factor",
        "params": {"r_H": int(r_H), "r_W": int(r_W)},
        "fits_all": True,
        "reproductions": reproductions,
    }


def _enumerate_affine_candidate(
    train_pairs: List[Dict[str, int]]
) -> Optional[Dict[str, Any]]:
    """
    Integer affine maps: [H'; W'] = M·[H; W] + b
    where M is 2×2 int matrix, b is 2-vector.

    Uses numpy.linalg.lstsq to find candidate, then verifies integer exactness.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Integer affine family
    """
    # Need at least 3 pairs for 6 unknowns (M11, M12, M21, M22, b1, b2)
    if len(train_pairs) < 3:
        return None  # Underdetermined

    # Build linear system: A·u = y
    # u = [M11, M12, M21, M22, b1, b2]^T
    # For each pair i:
    #   H_out_i = M11*H_in_i + M12*W_in_i + b1
    #   W_out_i = M21*H_in_i + M22*W_in_i + b2

    num_pairs = len(train_pairs)
    A = np.zeros((2 * num_pairs, 6), dtype=float)
    y = np.zeros(2 * num_pairs, dtype=float)

    for i, pair in enumerate(train_pairs):
        H_in = pair["H_in"]
        W_in = pair["W_in"]
        H_out = pair["H_out"]
        W_out = pair["W_out"]

        # Row for H equation
        A[2*i, :] = [H_in, W_in, 0, 0, 1, 0]
        y[2*i] = H_out

        # Row for W equation
        A[2*i + 1, :] = [0, 0, H_in, W_in, 0, 1]
        y[2*i + 1] = W_out

    # Solve least squares
    u_real, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    # Check residual
    y_pred = A @ u_real
    max_residual = np.abs(y - y_pred).max()
    if max_residual > 1e-9:
        return None  # No exact affine solution

    # Round to integers
    u_int = np.rint(u_real).astype(int)

    # Verify closeness to integers
    if np.abs(u_int - u_real).max() > 1e-9:
        return None  # Not truly integer

    # Extract M and b
    M11, M12, M21, M22, b1, b2 = u_int
    M = [[int(M11), int(M12)], [int(M21), int(M22)]]
    b = [int(b1), int(b2)]

    # Build reproductions
    def predict(H_in, W_in):
        H_pred = M[0][0] * H_in + M[0][1] * W_in + b[0]
        W_pred = M[1][0] * H_in + M[1][1] * W_in + b[1]
        return H_pred, W_pred

    reproductions, fits_all = _build_reproductions(train_pairs, predict)

    if not fits_all:
        return None  # Shouldn't happen if math is correct, but safety check

    return {
        "family": "affine",
        "params": {"M": M, "b": b},
        "fits_all": True,
        "reproductions": reproductions,
    }


def _enumerate_tile_candidates(
    train_pairs: List[Dict[str, int]]
) -> List[Dict[str, Any]]:
    """
    Tile/concat maps: H_out = n_v * H_in + δ_H, W_out = n_h * W_in + δ_W

    Uses set intersection to find (n_v, δ_H) and (n_h, δ_W) that work for ALL pairs.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Tile/concat family
    """
    # Enumerate (n_v, δ_H) for each pair, then intersect
    common_H = None
    for pair in train_pairs:
        H_in = pair["H_in"]
        H_out = pair["H_out"]

        candidates_H = set()
        # Enumerate n_v from 0 to reasonable upper bound
        max_n_v = (H_out // max(1, H_in)) + 2  # +2 for safety
        for n_v in range(0, max_n_v + 1):
            delta_H = H_out - n_v * H_in
            candidates_H.add((n_v, delta_H))

        if common_H is None:
            common_H = candidates_H
        else:
            common_H = common_H.intersection(candidates_H)

        if not common_H:
            break  # No common (n_v, δ_H) found

    # Enumerate (n_h, δ_W) for each pair, then intersect
    common_W = None
    for pair in train_pairs:
        W_in = pair["W_in"]
        W_out = pair["W_out"]

        candidates_W = set()
        max_n_h = (W_out // max(1, W_in)) + 2
        for n_h in range(0, max_n_h + 1):
            delta_W = W_out - n_h * W_in
            candidates_W.add((n_h, delta_W))

        if common_W is None:
            common_W = candidates_W
        else:
            common_W = common_W.intersection(candidates_W)

        if not common_W:
            break

    if not common_H or not common_W:
        return []  # No tile candidates

    # Build candidates from cartesian product
    tile_candidates = []
    for (n_v, delta_H) in common_H:
        for (n_h, delta_W) in common_W:
            def predict(H_in, W_in, nv=n_v, nh=n_h, dH=delta_H, dW=delta_W):
                return nv * H_in + dH, nh * W_in + dW

            reproductions, fits_all = _build_reproductions(train_pairs, predict)

            if fits_all:
                tile_candidates.append({
                    "family": "tile",
                    "params": {
                        "n_v": int(n_v),
                        "n_h": int(n_h),
                        "delta_H": int(delta_H),
                        "delta_W": int(delta_W),
                    },
                    "fits_all": True,
                    "reproductions": reproductions,
                })

    return tile_candidates


def _enumerate_constant_candidate(
    train_pairs: List[Dict[str, int]]
) -> Optional[Dict[str, Any]]:
    """
    Constant size: all train_out have same (H_out, W_out).

    Anchors:
      - 00_MATH_SPEC.md §3.1: Constant size family
    """
    H_out_values = [pair["H_out"] for pair in train_pairs]
    W_out_values = [pair["W_out"] for pair in train_pairs]

    if len(set(H_out_values)) != 1 or len(set(W_out_values)) != 1:
        return None  # Not constant

    H_const = H_out_values[0]
    W_const = W_out_values[0]

    # Build reproductions
    def predict(H_in, W_in):
        return H_const, W_const

    reproductions, fits_all = _build_reproductions(train_pairs, predict)

    # Should always fit if we got here
    if not fits_all:
        return None

    return {
        "family": "constant",
        "params": {"H_const": int(H_const), "W_const": int(W_const)},
        "fits_all": True,
        "reproductions": reproductions,
    }


def enumerate_size_maps(canonical: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enumerate all integer-exact size map families that fit training pairs.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Candidate size maps

    Returns:
      {
        "train_size_pairs": [...],
        "candidates": [...]
      }
    """
    train_pairs = _collect_train_size_pairs(canonical)

    candidates = []

    # 1. Identity
    cand = _enumerate_identity_candidate(train_pairs)
    if cand:
        candidates.append(cand)

    # 2. Swap
    cand = _enumerate_swap_candidate(train_pairs)
    if cand:
        candidates.append(cand)

    # 3. Factor
    cand = _enumerate_factor_candidate(train_pairs)
    if cand:
        candidates.append(cand)

    # 4. Affine
    cand = _enumerate_affine_candidate(train_pairs)
    if cand:
        candidates.append(cand)

    # 5. Tile (can return multiple)
    tile_cands = _enumerate_tile_candidates(train_pairs)
    candidates.extend(tile_cands)

    # 6. Constant
    cand = _enumerate_constant_candidate(train_pairs)
    if cand:
        candidates.append(cand)

    return {
        "train_size_pairs": train_pairs,
        "candidates": candidates,
    }


def _apply_candidate_to_test_size(
    candidate: Dict[str, Any],
    H_test: int,
    W_test: int
) -> tuple:
    """
    Apply a size map candidate to test input size to get predicted output size.

    Anchors:
      - 00_MATH_SPEC.md §3.1: Size map families

    Input:
      candidate: from WO-3.1 with family and params
      H_test, W_test: test input dimensions

    Output:
      (H_out_test, W_out_test): predicted test output size
    """
    family = candidate["family"]
    params = candidate["params"]

    if family == "identity":
        return H_test, W_test

    elif family == "swap":
        return W_test, H_test

    elif family == "factor":
        r_H = params["r_H"]
        r_W = params["r_W"]
        return r_H * H_test, r_W * W_test

    elif family == "affine":
        M = params["M"]
        b = params["b"]
        H_out = M[0][0] * H_test + M[0][1] * W_test + b[0]
        W_out = M[1][0] * H_test + M[1][1] * W_test + b[1]
        return H_out, W_out

    elif family == "tile":
        n_v = params["n_v"]
        n_h = params["n_h"]
        delta_H = params["delta_H"]
        delta_W = params["delta_W"]
        return n_v * H_test + delta_H, n_h * W_test + delta_W

    elif family == "constant":
        H_const = params["H_const"]
        W_const = params["W_const"]
        return H_const, W_const

    else:
        raise NotImplementedError(f"Unknown size map family: {family}")


def _passes_scaffold_screens(
    H_out_test: int,
    W_out_test: int,
    scaffold: Dict[str, Any]
) -> bool:
    """
    Check if candidate test size passes all scaffold-based structural screens.

    Revised per A0-compatible WO-3.2:
      - Only real repetition periods (2*p <= len)
      - Weak feasibility (no 2*t constraint)
      - No tiling constants (law-level, not geometry)
      - Screens apply to sizes, not families

    Anchors:
      - 00_MATH_SPEC.md §3.2 (revised): Structural disambiguation

    Screens:
      1. Parity: midrow/midcol → odd dimension
      2. Real periodicity: only if 2*p <= len(sequence)
      3. Weak feasibility: H' >= h_inner_min, W' >= w_inner_min

    Input:
      H_out_test, W_out_test: candidate test output size
      scaffold: from 03_scaffold.build

    Output:
      True if passes all screens, False otherwise
    """
    per_output = scaffold["per_output"]
    aggregated = scaffold["aggregated"]

    # Screen 1: Parity (midrow/midcol)
    # If all train_out have midrow → H_out_test must be odd
    # If all train_out have midcol → W_out_test must be odd
    if aggregated["has_midrow_all"]:
        if H_out_test % 2 == 0:
            return False

    if aggregated["has_midcol_all"]:
        if W_out_test % 2 == 0:
            return False

    # Screen 2: Real periodicity (only periods with 2*p <= len)
    # If row_period exists (and is real), H_out_test must be divisible by it
    # If col_period exists (and is real), W_out_test must be divisible by it
    row_period = aggregated["row_period"]
    if row_period is not None and row_period > 0:
        if H_out_test % row_period != 0:
            return False

    col_period = aggregated["col_period"]
    if col_period is not None and col_period > 0:
        if W_out_test % col_period != 0:
            return False

    # Screen 3: Weak feasibility (inner capacity, no 2*t)
    # Require H' >= h_inner_min, W' >= w_inner_min
    # Do NOT enforce H' >= h_inner + 2*t (that's law-level padding, not geometry)
    h_inner_min = None
    w_inner_min = None

    for info in per_output:
        inner = info["inner"]

        if not inner.any():
            continue  # No inner region for this output

        # Get inner bounding box
        rows, cols = np.where(inner)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        h_inner = int(r_max - r_min + 1)
        w_inner = int(c_max - c_min + 1)

        # Track minimum inner dimensions across all outputs
        if h_inner_min is None or h_inner < h_inner_min:
            h_inner_min = h_inner
        if w_inner_min is None or w_inner < w_inner_min:
            w_inner_min = w_inner

    # Apply weak feasibility if we have inner dimensions
    if h_inner_min is not None and H_out_test < h_inner_min:
        return False
    if w_inner_min is not None and W_out_test < w_inner_min:
        return False

    # NOTE: Tiling constants (δ ∈ {0, 2t}) removed - that's law-level, not geometry.
    # NOTE: Exact thickness constraints removed - padding is law-level choice.

    return True


def _select_canonical_representative(families: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select canonical representative from Π-equivalent families.

    Preference order (most preferred first):
      1. factor - covers scaling including identity case (r=1)
      2. affine - general linear
      3. identity - special case
      4. swap - dimension swap
      5. tile - concat/tiling
      6. constant - fallback

    This ensures deterministic representative selection when multiple families
    produce the same output size (Π-equivalent descriptions).
    """
    if not families:
        raise ValueError("Cannot select representative from empty family list")

    # Define preference order
    preference = ["factor", "affine", "identity", "swap", "tile", "constant"]

    # Find highest-preference family
    for preferred_family in preference:
        for fam in families:
            if fam["family"] == preferred_family:
                return fam

    # Fallback: return first family (should not reach here with complete preference list)
    return families[0]


def choose(
    canonical: Dict[str, Any],
    scaffold: Dict[str, Any],
    trace: bool = False
) -> Dict[str, Any]:
    """
    Stage: size_choice (S0) — WO-3.1 + WO-3.2

    Anchors:
      - 00_MATH_SPEC.md §3: Stage S0 — Output canvas size
      - 01_STAGES.md: size_choice
      - 02_QUANTUM_MAPPING.md: size_choice is free (no paid bits)

    Input:
      canonical: from 02_truth.canonicalize
      scaffold: from 03_scaffold.build
      trace: enable debug logging if True

    Output:
      {
        "status": "OK" | "IIS" | "AMBIGUOUS_SIZE",
        "H_out": int or None,
        "W_out": int or None,
        "train_size_pairs": [...],
        "candidates": [...],
        "survivors": [...]
      }

    WO-3.1: Enumerate all integer-exact size map candidates.
    WO-3.2: Apply scaffold screens and choose final (H_out, W_out).
    """
    if trace:
        logging.info("[size_choice] choose() called (WO-3.1+WO-3.2: candidates + screens)")

    # WO-3.1: Enumerate candidates
    size_data = enumerate_size_maps(canonical)
    train_size_pairs = size_data["train_size_pairs"]
    candidates = size_data["candidates"]

    # WO-3.2: Get test input size (single test only)
    test_in_list = canonical.get("test_in", [])
    if len(test_in_list) != 1:
        raise NotImplementedError(
            "[size_choice] Multiple test inputs not yet supported; extend spec first."
        )
    H_test, W_test = test_in_list[0].shape

    # WO-3.2 (Revised): Deduplicate by size, then screen sizes (not families)
    # Step 1: Apply all candidates to test size and collect size→families mapping
    size_to_families = {}

    for cand in candidates:
        if not cand.get("fits_all", False):
            continue  # Paranoia: WO-3.1 should have filtered already

        # Apply candidate to test input size
        H_out_test, W_out_test = _apply_candidate_to_test_size(cand, H_test, W_test)

        # Reject if invalid dimensions
        if H_out_test <= 0 or W_out_test <= 0:
            continue

        # Deduplicate by size
        size_key = (int(H_out_test), int(W_out_test))
        if size_key not in size_to_families:
            size_to_families[size_key] = []
        size_to_families[size_key].append(cand)

    # Step 2: Screen unique sizes (not families)
    # From Π viewpoint, different families yielding same (H',W') are minted differences
    surviving_sizes = {}  # {(H', W'): [families]}

    for (H_out_test, W_out_test), families in size_to_families.items():
        # Apply scaffold screens to the SIZE, not individual families
        if _passes_scaffold_screens(H_out_test, W_out_test, scaffold):
            surviving_sizes[(H_out_test, W_out_test)] = families

    # Step 3: Decide status based on NUMBER OF UNIQUE SIZES
    # 1 size → "OK"
    # 0 sizes → "IIS"
    # >1 sizes → "AMBIGUOUS_SIZE"
    num_unique_sizes = len(surviving_sizes)

    if num_unique_sizes == 0:
        status = "IIS"
        H_out = None
        W_out = None
    elif num_unique_sizes == 1:
        status = "OK"
        (H_out, W_out) = list(surviving_sizes.keys())[0]
    else:
        status = "AMBIGUOUS_SIZE"
        H_out = None
        W_out = None

    # Build survivors list for output (one representative per unique size)
    # Π-equivalence: multiple families → same size are minted differences
    survivors = []
    for (H_out_test, W_out_test), families in surviving_sizes.items():
        if families:
            # Select canonical representative using preference order
            representative = _select_canonical_representative(families)
            survivor = dict(representative)
            survivor["H_out_test"] = int(H_out_test)
            survivor["W_out_test"] = int(W_out_test)
            survivors.append(survivor)

    result = {
        "status": status,
        "H_out": H_out,
        "W_out": W_out,
        "train_size_pairs": train_size_pairs,
        "candidates": candidates,
        "survivors": survivors,
    }

    if trace:
        num_pairs = len(result["train_size_pairs"])
        num_candidates = len(result["candidates"])
        num_unique_sizes_before = len(size_to_families)
        num_unique_sizes_after = len(surviving_sizes)
        num_survivors = len(result["survivors"])

        logging.info(f"[size_choice] train_size_pairs: {num_pairs}")
        logging.info(f"[size_choice] candidates found: {num_candidates}")
        logging.info(f"[size_choice] unique sizes (before screens): {num_unique_sizes_before}")
        logging.info(f"[size_choice] unique sizes (after screens): {num_unique_sizes_after}")
        logging.info(f"[size_choice] survivors (one representative per size): {num_survivors}")

        # Show surviving sizes and their Π-equivalent families
        for (H_out_test, W_out_test), families in surviving_sizes.items():
            family_names = [f["family"] for f in families]
            representative = _select_canonical_representative(families)["family"]
            logging.info(f"[size_choice]   - size {H_out_test}×{W_out_test}: {len(families)} Π-equivalent families {family_names}, canonical representative={representative}")

        logging.info(f"[size_choice] status={result['status']}")
        logging.info(f"[size_choice] H_out={result['H_out']}, W_out={result['W_out']}")

    return result
