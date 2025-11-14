# WO-4.5 Implementation Summary: F24 Input Feature Mirror

## Overview

Implemented F24 (input feature mirror) per `00_MATH_SPEC.md §5.1 F`:
- A–E atoms ARE defined for inputs (same formulas as outputs)
- Guardrail is about **USAGE** (evaluation-only), not about which atoms exist
- F24 never mines from inputs, only evaluates features for already-mined laws

## Anchors

- `00_MATH_SPEC.md §5.1 F`: "Mirror A–E on **inputs** to **evaluate predicates on test_in** **only when referenced by a mined law**. **F24 does not create new laws.**"
- `01_STAGES.md`: Stage 05_laws
- `02_QUANTUM_MAPPING.md`: (no input-driven laws)

## Implementation

### 1. Shared Scaffold Helper (`utils/scaffold_utils.py`, ~65 LOC)

**File**: `utils/scaffold_utils.py`

Created `build_scaffold_for_grid(grid)` to compute per-grid geometry:
- Input: single grid (H×W numpy array)
- Output: `{d_top, d_bottom, d_left, d_right}` (distance fields)
- Pure geometric computation, no color semantics
- Reused by both `03_scaffold` (on train_out) and F24 (on test_in)

**Key design**:
- Minimal: only returns what A/C atoms need (distance fields)
- No inner/parity/thickness/periods (those are for S0 aggregation across train_out)
- Same math as original scaffold implementation (closed-form distances to border)

### 2. Refactored 03_scaffold (`03_scaffold/step.py`, ~10 LOC changed)

**Changes**:
- Import `build_scaffold_for_grid` from `utils/scaffold_utils`
- Replace `_distance_fields_for_output()` call with shared helper
- Remove old `_distance_fields_for_output()` function (~47 LOC removed)

**No functional change**: Existing tests (WO-4.4) pass unchanged.

### 3. F24 Implementation (`05_laws/atoms.py`, ~100 LOC)

**File**: `05_laws/atoms.py` (appended to end)

Added:
- Module-level cache: `_input_atoms_cache: Dict[int, Dict] = {}`
- Function: `get_input_atoms_for_test(canonical, test_idx=0)`

**API**:
```python
def get_input_atoms_for_test(
    canonical: Dict[str, Any],
    test_idx: int = 0,
) -> Dict[str, Any]:
    """
    F24: Mirror A–E atoms on inputs for evaluation of laws.

    GUARDRAIL:
      - NEVER called during mining from train_out
      - ONLY used to evaluate input features for already-mined laws

    Returns:
      {
        "A": A-atoms (scaffold geometry),
        "B": B-atoms (local texture),
        "C": C-atoms (connectivity & shape),
        "D": D-atoms (repetition & tiling),
        "E": E-atoms (palette/global),
      }
    """
```

**Implementation**:
1. Check cache (test_in is immutable)
2. Build scaffold on input grid via `build_scaffold_for_grid(grid)`
3. Reuse existing atom functions:
   - `compute_A_atoms(H, W, scaffold_info)`
   - `compute_B_atoms(grid)`
   - `compute_C_atoms(grid, scaffold_info)`
   - `compute_D_atoms(grid)`
   - `compute_E_atoms_for_grid(grid, C_atoms)`
4. Cache and return result

**Guardrail enforcement**:
- Docstring explicitly states: "NEVER called during mining"
- Any call in mining path = spec violation

### 4. Tests (`test_wo_4_5_f24.py`, ~271 LOC)

**File**: `test_wo_4_5_f24.py`

Four test cases:

1. **Basic functionality**: F24 computes A–E atoms on test_in, validates structure
2. **Cache**: Second call returns same object (cached)
3. **Guardrail pattern**: Demonstrates mining (no F24) → evaluation (F24 usage)
4. **test_idx parameter**: Validates API supports multiple test inputs

**All tests pass** ✅

### 5. Demo (`demo_f24_trace.py`, ~132 LOC)

**File**: `demo_f24_trace.py`

Demonstrates:
- Mining phase: F24 NOT called
- Evaluation phase: F24 called to get input features
- Trace pattern: `[F24]` prefix appears AFTER mining
- Example law using input feature (background color from input)

**Output shows**:
```
[MINING] Mining laws from train_out...
[MINING] (F24 is NOT called during mining)
[MINING] Mining complete

[EVALUATION] Now evaluating input features for law constraints...
[F24] used input atoms for test_in[0]:
      palette: [2, 3, 7, 8]
      most_frequent: [2, 3, 7, 8]
      tilings_true: [(2, 2)]
```

## Files Modified/Created

### Created:
1. `utils/scaffold_utils.py` (~65 LOC)
2. `test_wo_4_5_f24.py` (~271 LOC)
3. `demo_f24_trace.py` (~132 LOC)
4. `docs/WO_4_5_SUMMARY.md` (this file)

### Modified:
1. `03_scaffold/step.py`: Refactored to use shared helper (~10 LOC changed, ~47 LOC removed)
2. `05_laws/atoms.py`: Added F24 implementation (~100 LOC added)

**Total**: ~468 LOC added, ~47 LOC removed, net +421 LOC

## Validation

### Test Results:
```bash
$ python test_wo_4_5_f24.py
🎉 ALL WO-4.5 F24 TESTS PASSED!

Validated:
  ✅ F24 computes A–E atoms on test_in
  ✅ Atoms have correct structure & semantics
  ✅ Cache works (same object returned)
  ✅ Guardrail pattern: F24 AFTER mining, NEVER during
  ✅ test_idx parameter supported
```

### Existing Tests:
```bash
$ python test_wo_4_4.py
🎉 ALL WO-4.4 TESTS PASSED!
```
(Verifies scaffold refactor didn't break existing functionality)

### Demo:
```bash
$ python demo_f24_trace.py
✅ DEMO COMPLETE

Key observations:
  1. F24 called AFTER mining (never during) ✅
  2. [F24] trace appears only in evaluation phase ✅
  3. Input atoms have same structure as output atoms ✅
  4. Input features used to parameterize already-mined laws ✅
```

## Design Decisions

### 1. Minimal Scaffold Helper
**Decision**: Only return distance fields, not inner/parity/thickness/periods.

**Rationale**:
- A/C atoms only need distance fields
- Inner/parity/thickness/periods are for S0 screening across train_out
- F24 operates on single input grid, doesn't need aggregated hints
- Keeps helper focused and reusable

### 2. Module-Level Cache
**Decision**: Cache at module level, not in function closure.

**Rationale**:
- test_in is immutable (canonical coords fixed after Stage A)
- Avoids recomputation on repeated evaluation
- Simple dict keyed by test_idx
- Clear separation: cache is global state, function is pure logic

### 3. Import in Function Body
**Decision**: Import `build_scaffold_for_grid` inside `get_input_atoms_for_test`, not at module level.

**Rationale**:
- Avoids circular dependency (atoms.py ← utils/scaffold_utils.py)
- Function is called rarely (only when law uses input features)
- Import cost amortized by cache

## Future Work (Not in WO-4.5)

When a law actually needs input features (future WO):

1. Add trace to `05_laws/step.py`:
   ```python
   if trace and used_input_features:
       input_atoms = get_input_atoms_for_test(canonical, test_idx=0)
       print("[F24] used input atoms for test_in[0]:")
       print(f"      palette: {input_atoms['E']['palette']}")
       # ... etc
   ```

2. Use input features in constraints:
   ```python
   bg_input = input_atoms["E"]["most_frequent"][0]
   # Add constraint: background in test_out = bg_input
   ```

Current implementation provides the **infrastructure** (F24 ready to use), but no laws currently reference input features.

## Spec Compliance

✅ **00_MATH_SPEC.md §5.1 F**: "Mirror A–E on **inputs** to **evaluate predicates on test_in** **only when referenced by a mined law**."
- A–E atoms computable on inputs ✅
- Same formulas as outputs ✅
- Evaluation-only (not mining) ✅

✅ **F24 does not create new laws**: Guardrail enforced via documentation and usage pattern ✅

✅ **No improvisation**: Reuses existing atom functions, shared scaffold logic ✅

✅ **Fail loudly**: Clear errors if test_idx out of range or scaffold build fails ✅

## Summary

F24 is now fully operational:
- Infrastructure: shared scaffold helper, cache, atom computation
- Guardrail: clear separation (mining vs evaluation)
- Testing: 4 tests + demo validate correctness
- Ready for future laws that reference input features

**No breaking changes**: existing tests pass, scaffold refactor is internal-only.
