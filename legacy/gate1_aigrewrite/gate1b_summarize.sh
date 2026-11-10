#!/usr/bin/env bash
set -Eeuo pipefail
OUT=/data/wushangge/gate1a_AIGRewrite/gate1b_out

jq -s '
  [ .[] as $design |
    range(0;20) as $sample |
    $design.records[(($sample * (($design.records|length) - 1) / 19)|floor)] |
    . + {
      design:$design.design,
      standard_mffc_size:(.standard_mffc_nodes|length),
      standard_mffc_boundary_size:(.standard_mffc_boundary|length),
      cut_cone_size:(.cut_cone_nodes|length),
      saved_region_size:(.saved_region_nodes|length)
    }
  ] as $records |
  def exact: (.saved_region_nodes == .standard_mffc_nodes and .cut_leaves == .standard_mffc_boundary);
  def strict_inside: (.saved_subset_of_mffc and (.saved_region_nodes != .standard_mffc_nodes));
  {
    methodology:{records_per_design:20,total_records:($records|length),selection:"20 deterministic evenly spaced records from 512 captured multi-subgraph cuts per design"},
    records:$records,
    statistics:{
      exact_mffc_boundary_count:([$records[]|select(exact)]|length),
      exact_mffc_boundary_percent:(([$records[]|select(exact)]|length)*100/($records|length)),
      cut_strictly_inside_mffc_count:([$records[]|select((exact|not) and strict_inside)]|length),
      cut_strictly_inside_mffc_percent:(([$records[]|select((exact|not) and strict_inside)]|length)*100/($records|length)),
      other_count:([$records[]|select((exact|not) and (strict_inside|not))]|length),
      other_percent:(([$records[]|select((exact|not) and (strict_inside|not))]|length)*100/($records|length)),
      cut_cone_subset_of_mffc_percent:(([$records[]|select(.cut_cone_subset_of_mffc)]|length)*100/($records|length)),
      saved_subset_of_mffc_percent:(([$records[]|select(.saved_subset_of_mffc)]|length)*100/($records|length)),
      saved_equals_cut_cone_percent:(([$records[]|select(.saved_equals_cut_cone)]|length)*100/($records|length)),
      gpu_saved_size_mismatches:([$records[]|select((.saved_region_nodes|length) != .gpu_saved_size)]|length)
    }
  }
' "$OUT/dft_orig_alignment.json" "$OUT/idft_orig_alignment.json" "$OUT/vga_lcd_orig_alignment.json" \
  > "$OUT/gate1b_statistics.json"
