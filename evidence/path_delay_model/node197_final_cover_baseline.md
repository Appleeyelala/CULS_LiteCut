# baseline: final cover proof

Target mapper node 196; active phase refs [0, 0, 0]. The listed covers are selected (nRefAct>0); mapper-subject DFS inside their cut boundaries reaches the target as an internal node. The accompanying witness is independently reconstructed with the existing AIG parser, using the exact serialized input for this run. No mapper marks or references are modified.

| Output serialized AIG var | Verified original function | Phase | Supergate | Cell | Cut leaves | Arrival | Internal nodes | Witness to target |
|---|---|---|---|---|---|---:|---|---|
| 211 | 211 | 0 | 1837 | OAI21xp33_ASAP7_75t_L | [195, 196, 209] | 95.2699966 | [197, 210, 211] | [211, 210, 197] |
| 211 | 211 | 1 | 1567 | AOI21xp33_ASAP7_75t_L | [195, 196, 209] | 94.4199982 | [197, 210, 211] | [211, 210, 197] |
| 212 | 212 | 0 | 2037 | NAND3xp33_ASAP7_75t_L | [195, 196, 209] | 93.8099976 | [197, 210, 212] | [212, 210, 197] |
| 212 | 212 | 1 | 1303 | NOR3xp33_ASAP7_75t_L | [195, 196, 209] | 95.8399963 | [197, 210, 212] | [212, 210, 197] |

```json
[
  {
    "type": "cover",
    "output_aiger_var": 211,
    "output_mapper_id": 210,
    "phase": 0,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      195,
      196,
      209
    ],
    "leaf_mapper_ids": [
      194,
      195,
      208
    ],
    "arrival": {
      "rise": 95.2699966,
      "fall": 95.2699966,
      "worst": 95.2699966
    },
    "super": {
      "id": 1837,
      "root_cell": "OAI21xp33_ASAP7_75t_L",
      "formula": "",
      "area": 1.16999996,
      "n_gates": 1,
      "n_fanins": 3,
      "fanout_limit": 2,
      "arcs": [
        {
          "pin": 0,
          "rr": 16.3600006,
          "fr": 16.3600006,
          "rf": 16.3600006,
          "ff": 16.3600006
        },
        {
          "pin": 1,
          "rr": 15.6300001,
          "fr": 15.6300001,
          "rf": 15.6300001,
          "ff": 15.6300001
        },
        {
          "pin": 2,
          "rr": 13.2799997,
          "fr": 13.2799997,
          "rf": 13.2799997,
          "ff": 13.2799997
        }
      ],
      "implementation": {
        "cell": "OAI21xp33_ASAP7_75t_L",
        "fanins": [
          {
            "leaf_pin": 0
          },
          {
            "leaf_pin": 1
          },
          {
            "leaf_pin": 2
          }
        ]
      }
    },
    "verified_original_output": 211,
    "aiger_internal_witness": [
      211,
      210,
      197
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 211,
    "output_mapper_id": 210,
    "phase": 1,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      195,
      196,
      209
    ],
    "leaf_mapper_ids": [
      194,
      195,
      208
    ],
    "arrival": {
      "rise": 94.4199982,
      "fall": 94.4199982,
      "worst": 94.4199982
    },
    "super": {
      "id": 1567,
      "root_cell": "AOI21xp33_ASAP7_75t_L",
      "formula": "",
      "area": 1.16999996,
      "n_gates": 1,
      "n_fanins": 3,
      "fanout_limit": 2,
      "arcs": [
        {
          "pin": 0,
          "rr": 16.0200005,
          "fr": 16.0200005,
          "rf": 16.0200005,
          "ff": 16.0200005
        },
        {
          "pin": 1,
          "rr": 16.1800003,
          "fr": 16.1800003,
          "rf": 16.1800003,
          "ff": 16.1800003
        },
        {
          "pin": 2,
          "rr": 14.6199999,
          "fr": 14.6199999,
          "rf": 14.6199999,
          "ff": 14.6199999
        }
      ],
      "implementation": {
        "cell": "AOI21xp33_ASAP7_75t_L",
        "fanins": [
          {
            "leaf_pin": 0
          },
          {
            "leaf_pin": 1
          },
          {
            "leaf_pin": 2
          }
        ]
      }
    },
    "verified_original_output": 211,
    "aiger_internal_witness": [
      211,
      210,
      197
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 212,
    "output_mapper_id": 211,
    "phase": 0,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      195,
      196,
      209
    ],
    "leaf_mapper_ids": [
      194,
      195,
      208
    ],
    "arrival": {
      "rise": 93.8099976,
      "fall": 93.8099976,
      "worst": 93.8099976
    },
    "super": {
      "id": 2037,
      "root_cell": "NAND3xp33_ASAP7_75t_L",
      "formula": "",
      "area": 1.16999996,
      "n_gates": 1,
      "n_fanins": 3,
      "fanout_limit": 2,
      "arcs": [
        {
          "pin": 0,
          "rr": 15,
          "fr": 15,
          "rf": 15,
          "ff": 15
        },
        {
          "pin": 1,
          "rr": 16.1900005,
          "fr": 16.1900005,
          "rf": 16.1900005,
          "ff": 16.1900005
        },
        {
          "pin": 2,
          "rr": 16.75,
          "fr": 16.75,
          "rf": 16.75,
          "ff": 16.75
        }
      ],
      "implementation": {
        "cell": "NAND3xp33_ASAP7_75t_L",
        "fanins": [
          {
            "leaf_pin": 0
          },
          {
            "leaf_pin": 1
          },
          {
            "leaf_pin": 2
          }
        ]
      }
    },
    "verified_original_output": 212,
    "aiger_internal_witness": [
      212,
      210,
      197
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 212,
    "output_mapper_id": 211,
    "phase": 1,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      195,
      196,
      209
    ],
    "leaf_mapper_ids": [
      194,
      195,
      208
    ],
    "arrival": {
      "rise": 95.8399963,
      "fall": 95.8399963,
      "worst": 95.8399963
    },
    "super": {
      "id": 1303,
      "root_cell": "NOR3xp33_ASAP7_75t_L",
      "formula": "",
      "area": 1.16999996,
      "n_gates": 1,
      "n_fanins": 3,
      "fanout_limit": 2,
      "arcs": [
        {
          "pin": 0,
          "rr": 17.6000004,
          "fr": 17.6000004,
          "rf": 17.6000004,
          "ff": 17.6000004
        },
        {
          "pin": 1,
          "rr": 16.2000008,
          "fr": 16.2000008,
          "rf": 16.2000008,
          "ff": 16.2000008
        },
        {
          "pin": 2,
          "rr": 18.0300007,
          "fr": 18.0300007,
          "rf": 18.0300007,
          "ff": 18.0300007
        }
      ],
      "implementation": {
        "cell": "NOR3xp33_ASAP7_75t_L",
        "fanins": [
          {
            "leaf_pin": 1
          },
          {
            "leaf_pin": 0
          },
          {
            "leaf_pin": 2
          }
        ]
      }
    },
    "verified_original_output": 212,
    "aiger_internal_witness": [
      212,
      210,
      197
    ]
  }
]
```
