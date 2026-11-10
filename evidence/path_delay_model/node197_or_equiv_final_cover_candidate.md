# candidate: final cover proof

Target mapper node 194; active phase refs [0, 0, 0]. The listed covers are selected (nRefAct>0); mapper-subject DFS inside their cut boundaries reaches the target as an internal node. The accompanying witness is independently reconstructed with the existing AIG parser, using the exact serialized input for this run. No mapper marks or references are modified.

| Output serialized AIG var | Verified original function | Phase | Supergate | Cell | Cut leaves | Arrival | Internal nodes | Witness to target |
|---|---|---|---|---|---|---:|---|---|
| 305 | NOT_PROVEN | 1 | 1567 | AOI21xp33_ASAP7_75t_L | [277, 278, 235] | 91.7099991 | [287, 305] | [305, 287] |
| 313 | 211 | 1 | 1567 | AOI21xp33_ASAP7_75t_L | [277, 278, 235] | 91.7099991 | [287, 305, 313] | [313, 305, 287] |
| 314 | 212 | 0 | 2037 | NAND3xp33_ASAP7_75t_L | [277, 278, 235] | 91.7200012 | [287, 305, 314] | [314, 305, 287] |
| 314 | 212 | 1 | 1301 | NOR3xp33_ASAP7_75t_L | [277, 278, 235] | 93.7799911 | [287, 305, 314] | [314, 305, 287] |
| 322 | 213 | 0 | 1619 | A2O1A1Ixp33_ASAP7_75t_L | [277, 278, 305, 314] | 108.5 | [287, 313, 322] | [322, 313, 287] |

```json
[
  {
    "type": "cover",
    "output_aiger_var": 305,
    "output_mapper_id": 207,
    "phase": 1,
    "refs": [
      0,
      1,
      1
    ],
    "contains_target_internal": true,
    "leaves": [
      277,
      278,
      235
    ],
    "leaf_mapper_ids": [
      192,
      193,
      206
    ],
    "arrival": {
      "rise": 91.7099991,
      "fall": 91.7099991,
      "worst": 91.7099991
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
    "verified_original_output": "NOT_PROVEN",
    "aiger_internal_witness": [
      305,
      287
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 313,
    "output_mapper_id": 208,
    "phase": 1,
    "refs": [
      0,
      1,
      1
    ],
    "contains_target_internal": true,
    "leaves": [
      277,
      278,
      235
    ],
    "leaf_mapper_ids": [
      192,
      193,
      206
    ],
    "arrival": {
      "rise": 91.7099991,
      "fall": 91.7099991,
      "worst": 91.7099991
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
      313,
      305,
      287
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 314,
    "output_mapper_id": 209,
    "phase": 0,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      277,
      278,
      235
    ],
    "leaf_mapper_ids": [
      192,
      193,
      206
    ],
    "arrival": {
      "rise": 91.7200012,
      "fall": 91.7200012,
      "worst": 91.7200012
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
      314,
      305,
      287
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 314,
    "output_mapper_id": 209,
    "phase": 1,
    "refs": [
      1,
      1,
      2
    ],
    "contains_target_internal": true,
    "leaves": [
      277,
      278,
      235
    ],
    "leaf_mapper_ids": [
      192,
      193,
      206
    ],
    "arrival": {
      "rise": 93.7799911,
      "fall": 93.7799911,
      "worst": 93.7799911
    },
    "super": {
      "id": 1301,
      "root_cell": "NOR3xp33_ASAP7_75t_L",
      "formula": "",
      "area": 1.16999996,
      "n_gates": 1,
      "n_fanins": 3,
      "fanout_limit": 2,
      "arcs": [
        {
          "pin": 0,
          "rr": 16.2000008,
          "fr": 16.2000008,
          "rf": 16.2000008,
          "ff": 16.2000008
        },
        {
          "pin": 1,
          "rr": 17.6000004,
          "fr": 17.6000004,
          "rf": 17.6000004,
          "ff": 17.6000004
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
      314,
      305,
      287
    ]
  },
  {
    "type": "cover",
    "output_aiger_var": 322,
    "output_mapper_id": 210,
    "phase": 0,
    "refs": [
      9,
      11,
      20
    ],
    "contains_target_internal": true,
    "leaves": [
      277,
      278,
      305,
      314
    ],
    "leaf_mapper_ids": [
      192,
      193,
      207,
      209
    ],
    "arrival": {
      "rise": 108.5,
      "fall": 108.5,
      "worst": 108.5
    },
    "super": {
      "id": 1619,
      "root_cell": "A2O1A1Ixp33_ASAP7_75t_L",
      "formula": "",
      "area": 0,
      "n_gates": 1,
      "n_fanins": 4,
      "fanout_limit": 1,
      "arcs": [
        {
          "pin": 0,
          "rr": 18.8299999,
          "fr": 18.8299999,
          "rf": 18.8299999,
          "ff": 18.8299999
        },
        {
          "pin": 1,
          "rr": 19.2399998,
          "fr": 19.2399998,
          "rf": 19.2399998,
          "ff": 19.2399998
        },
        {
          "pin": 2,
          "rr": 16.7900009,
          "fr": 16.7900009,
          "rf": 16.7900009,
          "ff": 16.7900009
        },
        {
          "pin": 3,
          "rr": 15.5900002,
          "fr": 15.5900002,
          "rf": 15.5900002,
          "ff": 15.5900002
        }
      ],
      "implementation": {
        "cell": "A2O1A1Ixp33_ASAP7_75t_L",
        "fanins": [
          {
            "leaf_pin": 0
          },
          {
            "leaf_pin": 1
          },
          {
            "leaf_pin": 2
          },
          {
            "leaf_pin": 3
          }
        ]
      }
    },
    "verified_original_output": 213,
    "aiger_internal_witness": [
      322,
      313,
      287
    ]
  }
]
```
