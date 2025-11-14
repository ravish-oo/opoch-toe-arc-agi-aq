#!/usr/bin/env python3
"""
Task type detector for ARC-AGI training tasks.

Finds candidate tasks for 4 categories:
1. DIAGONAL_OR_FLIP - geometric transforms
2. COMPONENT_TRANSLATE_OR_COPY - shape translation/duplication
3. PALETTE_PERMUTATION - color relabeling
4. BLOWUP_SCALING - pixel→block scaling
"""

import json
from pathlib import Path
from collections import deque


# ============================================================================
# 1. DIAGONAL_OR_FLIP
# ============================================================================

def all_equal(A, B):
    """Check if two grids are identical."""
    if len(A) != len(B):
        return False
    if len(A) > 0 and len(A[0]) != len(B[0]):
        return False
    return all(rowA == rowB for rowA, rowB in zip(A, B))


def hflip(grid):
    """Horizontal flip."""
    return [row[::-1] for row in grid]


def vflip(grid):
    """Vertical flip."""
    return grid[::-1]


def transpose(grid):
    """Transpose (swap rows/cols)."""
    H, W = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(H)] for c in range(W)]


def rot90(grid):
    """Rotate 90 degrees clockwise."""
    return transpose(vflip(grid))


def rot180(grid):
    """Rotate 180 degrees."""
    return vflip(hflip(grid))


def rot270(grid):
    """Rotate 270 degrees clockwise."""
    return vflip(transpose(grid))


TRANSFORMS = {
    "HFLIP": hflip,
    "VFLIP": vflip,
    "TRANSPOSE": transpose,
    "ROT90": rot90,
    "ROT180": rot180,
    "ROT270": rot270,
}


def find_diagonal_or_flip_tasks(tasks_json):
    """Find tasks that are pure geometric transforms."""
    result = {name: [] for name in TRANSFORMS.keys()}

    for tid, task in tasks_json.items():
        trains = task["train"]

        # Size guard to avoid huge tasks
        if any(len(t["input"]) > 10 or (len(t["input"][0]) > 10 if t["input"] else False)
               for t in trains):
            continue

        valid_transforms = []
        for tname, T in TRANSFORMS.items():
            ok = True
            for pair in trains:
                inp = pair["input"]
                out = pair["output"]
                try:
                    if not all_equal(T(inp), out):
                        ok = False
                        break
                except Exception:
                    ok = False
                    break

            if ok:
                valid_transforms.append(tname)

        # If exactly 1 transform works, it's a clean candidate
        if len(valid_transforms) == 1:
            result[valid_transforms[0]].append(tid)

    return result


# ============================================================================
# 2. COMPONENT_TRANSLATE_OR_COPY
# ============================================================================

def bfs_components(grid, color):
    """Find all connected components (N4) of a given color."""
    H = len(grid)
    if H == 0:
        return []
    W = len(grid[0])

    seen = [[False]*W for _ in range(H)]
    comps = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] == color and not seen[r][c]:
                q = deque([(r, c)])
                seen[r][c] = True
                comp = []

                while q:
                    rr, cc = q.popleft()
                    comp.append((rr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W and
                            not seen[nr][nc] and grid[nr][nc] == color):
                            seen[nr][nc] = True
                            q.append((nr, nc))

                comps.append(comp)

    return comps


def is_translation(A, B):
    """Check if component B is a translation of component A."""
    if len(A) != len(B):
        return False

    A = sorted(A)
    B = sorted(B)

    dr = B[0][0] - A[0][0]
    dc = B[0][1] - A[0][1]

    A_shifted = sorted([(r + dr, c + dc) for r, c in A])
    return A_shifted == B


def find_component_translate_tasks(tasks_json):
    """Find tasks where components are translated."""
    candidates = []

    for tid, task in tasks_json.items():
        trains = task["train"]

        # Size guard
        if any(len(t["input"]) > 15 or (len(t["input"][0]) > 15 if t["input"] else False)
               for t in trains):
            continue

        # Try to find a color k that works in all train pairs
        colors_any = set()
        for pair in trains:
            if pair["input"]:
                colors_any |= {v for row in pair["input"] for v in row}

        ok_task = False
        for k in colors_any:
            k_ok_for_all_pairs = True

            for pair in trains:
                inp, out = pair["input"], pair["output"]

                # Counts should be equal for pure translate
                cnt_in = sum(v == k for row in inp for v in row)
                cnt_out = sum(v == k for row in out for v in row)

                if cnt_in == 0 or cnt_out == 0 or cnt_in != cnt_out:
                    k_ok_for_all_pairs = False
                    break

                comps_in = bfs_components(inp, k)
                comps_out = bfs_components(out, k)

                # Quick reject if component counts differ
                if len(comps_in) != len(comps_out):
                    k_ok_for_all_pairs = False
                    break

                # Check each input comp matches some output comp by translation
                used = [False] * len(comps_out)
                for cin in comps_in:
                    match_found = False
                    for j, cout in enumerate(comps_out):
                        if used[j]:
                            continue
                        if is_translation(cin, cout):
                            used[j] = True
                            match_found = True
                            break

                    if not match_found:
                        k_ok_for_all_pairs = False
                        break

                if not k_ok_for_all_pairs:
                    break

            if k_ok_for_all_pairs:
                ok_task = True
                break

        if ok_task:
            candidates.append(tid)

    return candidates


# ============================================================================
# 3. PALETTE_PERMUTATION
# ============================================================================

def find_palette_permutation_tasks(tasks_json):
    """Find tasks that are pure color relabeling."""
    candidates = []

    for tid, task in tasks_json.items():
        trains = task["train"]
        global_map = None
        ok_task = True

        for pair in trains:
            inp, out = pair["input"], pair["output"]
            H = len(inp)
            if H == 0:
                ok_task = False
                break
            W = len(inp[0])

            if len(out) != H or (len(out[0]) if out else 0) != W:
                ok_task = False
                break

            local_map = {}
            used_out = {}

            for r in range(H):
                for c in range(W):
                    a = inp[r][c]
                    b = out[r][c]

                    if a not in local_map:
                        # Injective constraint
                        if b in used_out and used_out[b] != a:
                            ok_task = False
                            break
                        local_map[a] = b
                        used_out[b] = a
                    else:
                        if local_map[a] != b:
                            ok_task = False
                            break

                if not ok_task:
                    break

            if not ok_task:
                break

            # Merge into global_map
            if global_map is None:
                global_map = local_map
            else:
                for a, b in local_map.items():
                    if a in global_map and global_map[a] != b:
                        ok_task = False
                        break
                    global_map[a] = b

        if ok_task and global_map is not None:
            candidates.append(tid)

    return candidates


# ============================================================================
# 4. BLOWUP_SCALING
# ============================================================================

def is_uniform_block(grid, r0, r1, c0, c1):
    """Check if all cells in block have same value."""
    val = grid[r0][c0]
    for r in range(r0, r1):
        for c in range(c0, c1):
            if grid[r][c] != val:
                return False
    return True


def find_blowup_scaling_tasks(tasks_json):
    """Find tasks where output is scaled-up version of input."""
    candidates = []

    for tid, task in tasks_json.items():
        trains = task["train"]
        scale_r = None
        scale_c = None
        ok_task = True

        for pair in trains:
            inp, out = pair["input"], pair["output"]
            H_in = len(inp)
            if H_in == 0:
                ok_task = False
                break
            W_in = len(inp[0])

            H_out = len(out)
            if H_out == 0:
                ok_task = False
                break
            W_out = len(out[0])

            # Divisibility check
            if H_out % H_in != 0 or W_out % W_in != 0:
                ok_task = False
                break

            s_r = H_out // H_in
            s_c = W_out // W_in

            if scale_r is None:
                scale_r, scale_c = s_r, s_c
            else:
                if scale_r != s_r or scale_c != s_c:
                    ok_task = False
                    break

            # Uniform block check
            for ri in range(H_in):
                for ci in range(W_in):
                    r0 = ri * s_r
                    r1 = (ri + 1) * s_r
                    c0 = ci * s_c
                    c1 = (ci + 1) * s_c

                    if not is_uniform_block(out, r0, r1, c0, c1):
                        ok_task = False
                        break

                if not ok_task:
                    break

            if not ok_task:
                break

        if ok_task and scale_r is not None and scale_r > 1:
            candidates.append((tid, scale_r, scale_c))

    return candidates


# ============================================================================
# MAIN
# ============================================================================

def main():
    data_path = Path("data/arc-agi_training_challenges.json")

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    tasks = json.loads(data_path.read_text())

    print("=" * 70)
    print("ARC-AGI TASK TYPE DETECTOR")
    print("=" * 70)
    print()

    # 1. DIAGONAL_OR_FLIP
    print("1. DIAGONAL_OR_FLIP tasks")
    print("-" * 70)
    flip_tasks = find_diagonal_or_flip_tasks(tasks)
    for tname, ids in flip_tasks.items():
        if ids:
            print(f"  {tname}: {len(ids)} tasks")
            print(f"    IDs: {ids[:10]}")  # Show first 10
    print()

    # 2. COMPONENT_TRANSLATE
    print("2. COMPONENT_TRANSLATE_OR_COPY tasks")
    print("-" * 70)
    translate_tasks = find_component_translate_tasks(tasks)
    print(f"  Found: {len(translate_tasks)} tasks")
    print(f"  IDs: {translate_tasks[:10]}")  # Show first 10
    print()

    # 3. PALETTE_PERMUTATION
    print("3. PALETTE_PERMUTATION tasks")
    print("-" * 70)
    palette_tasks = find_palette_permutation_tasks(tasks)
    print(f"  Found: {len(palette_tasks)} tasks")
    print(f"  IDs: {palette_tasks[:10]}")  # Show first 10
    print()

    # 4. BLOWUP_SCALING
    print("4. BLOWUP_SCALING tasks")
    print("-" * 70)
    blowup_tasks = find_blowup_scaling_tasks(tasks)
    print(f"  Found: {len(blowup_tasks)} tasks")
    for tid, sr, sc in blowup_tasks[:10]:  # Show first 10
        print(f"    {tid}: scale ({sr}×{sc})")
    print()

    # Summary with specific recommendations
    print("=" * 70)
    print("RECOMMENDED CANDIDATES")
    print("=" * 70)

    # Pick best from each category
    print("\n1. DIAGONAL_OR_FLIP:")
    for tname in ["TRANSPOSE", "ROT90", "ROT270", "HFLIP", "VFLIP"]:
        if flip_tasks[tname]:
            print(f"   {tname}: {flip_tasks[tname][0]} (and {len(flip_tasks[tname])-1} more)")
            break

    print("\n2. COMPONENT_TRANSLATE_OR_COPY:")
    if translate_tasks:
        print(f"   {translate_tasks[0]} (and {len(translate_tasks)-1} more)")

    print("\n3. PALETTE_PERMUTATION:")
    if palette_tasks:
        print(f"   {palette_tasks[0]} (and {len(palette_tasks)-1} more)")

    print("\n4. BLOWUP_SCALING:")
    # Prefer 2x2 or 3x3 uniform scaling
    uniform_blowups = [(tid, sr, sc) for tid, sr, sc in blowup_tasks if sr == sc and sr in [2, 3]]
    if uniform_blowups:
        tid, sr, sc = uniform_blowups[0]
        print(f"   {tid}: {sr}×{sc} scaling (and {len(uniform_blowups)-1} more)")
    elif blowup_tasks:
        tid, sr, sc = blowup_tasks[0]
        print(f"   {tid}: {sr}×{sc} scaling (and {len(blowup_tasks)-1} more)")

    print()


if __name__ == "__main__":
    main()
