#!/usr/bin/env python3
"""Verify component counts on canonical grids for 74dd1130."""

import sys
import json
import numpy as np
from pathlib import Path
import importlib
import types
from scipy import ndimage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def load_stage_as_package(stage_dir_name):
    """Load a stage directory as a package to handle relative imports."""
    stage_path = Path(__file__).parent / stage_dir_name

    # Create package
    pkg = types.ModuleType(stage_dir_name.replace("-", "_"))
    pkg.__path__ = [str(stage_path)]
    pkg.__package__ = stage_dir_name.replace("-", "_")
    sys.modules[pkg.__name__] = pkg

    # Load step.py as submodule
    spec = importlib.util.spec_from_file_location(
        f"{pkg.__name__}.step",
        stage_path / "step.py"
    )
    step_mod = importlib.util.module_from_spec(spec)
    step_mod.__package__ = pkg.__name__
    sys.modules[spec.name] = step_mod

    spec.loader.exec_module(step_mod)
    pkg.step = step_mod

    return step_mod

# Load stages
_present_step = load_stage_as_package("01_present")
_truth_step = load_stage_as_package("02_truth")

# Load task
data_path = Path("data/arc-agi_training_challenges.json")
data = json.loads(data_path.read_text())

task_bundle = {
    "task_id": "74dd1130",
    "test_index": 0,
    "raw_task": data["74dd1130"],
}

# Process through pipeline
present = _present_step.load(task_bundle, trace=False)
canonical = _truth_step.canonicalize(present, trace=False)

print("=== Component counts on CANONICAL grids for 74dd1130 ===\n")

# 4-connectivity structure
structure_4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=int)

for i, grid in enumerate(canonical["train_out"]):
    print(f"train_out[{i}]:")
    print(grid)

    colors = np.unique(grid)
    component_counts = {}

    for k in colors:
        if k == 0:
            continue  # skip background

        mask = (grid == k).astype(np.uint8)
        labeled, num_features = ndimage.label(mask, structure=structure_4)
        component_counts[str(k)] = num_features

    print(f"Component counts: {component_counts}\n")
