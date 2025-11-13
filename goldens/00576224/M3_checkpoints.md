* Exactly **one candidate map**: factor map `(r_H=3, r_W=3)`.
* That candidate **survives all screens**.
* `status = "OK"`, with `H_out = 6`, `W_out = 6`.

Here is a JSON you can use directly as the golden for size_choice:

```json
{
  "task_id": "00576224",
  "train_size_pairs": [
    {
      "H_in": 2,
      "W_in": 2,
      "H_out": 6,
      "W_out": 6
    },
    {
      "H_in": 2,
      "W_in": 2,
      "H_out": 6,
      "W_out": 6
    }
  ],
  "candidates": [
    {
      "kind": "factor",
      "formula": {
        "H_out": "r_H * H_in",
        "W_out": "r_W * W_in"
      },
      "params": {
        "r_H": 3,
        "r_W": 3
      },
      "fits_all_training_pairs": true
    }
  ],
  "screens": {
    "train_out_only": true,
    "checks": [
      {
        "name": "feasibility_inner_region",
        "passed": true
      },
      {
        "name": "parity_midrow_midcol",
        "passed": true
      },
      {
        "name": "periodicity_divides_HW",
        "passed": true
      },
      {
        "name": "frame_thickness_capacity",
        "passed": true
      }
    ]
  },
  "result": {
    "status": "OK",
    "chosen": {
      "kind": "factor",
      "params": {
        "r_H": 3,
        "r_W": 3
      },
      "H_out": 6,
      "W_out": 6
    },
    "survivor_count": 1
  }
}
```

**How your tester should use this:**

* After `04_size_choice.step.choose(...)`, your code should produce a structure that can be normalized to:

  * One factor candidate with `r_H=3, r_W=3`,
  * Status `"OK"`,
  * `H_out=6, W_out=6`.
* If later you decide to also enumerate a “constant 6×6” candidate, you’ll get ambiguity and the golden will need to change. For now, this is the cleanest and safest spec-consistent choice.
