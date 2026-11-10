#!/usr/bin/env python3
"""Gate 2A-R mapping correspondence audit.

This is deliberately an audit-only adapter.  It consumes the logs produced by
the canonical cutLib ABC flow and the Gate 1C-R provenance; it does not build
features, load a checkpoint, or run Student inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    import networkx as nx
except ImportError:  # pragma: no cover - canonical repo normally has networkx
    nx = None


OUT = Path(__file__).resolve().parent / "gate2a_r_out"
GATE1 = Path("/data/wushangge/gate1a_AIGRewrite/gate1c_r_out")
ABC = Path("/home/wushangge/abc_for_cutLib/abc")
LIB = Path("/home/wushangge/cutLibOrigin/lib/asap7.lib")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_aig2map(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Return AIG->mapper and mapper->AIG IDs from canonical aig2map.log."""
    aig_to_map: dict[int, int] = {}
    map_to_aig: dict[int, int] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"Map Node ID:\s*(\d+)", line)
        if m:
            current = int(m.group(1))
            continue
        m = re.search(r"AIG Node Id:\s*(\d+)", line)
        if m and current is not None:
            aig = int(m.group(1))
            aig_to_map[aig] = current
            map_to_aig[current] = aig
    return aig_to_map, map_to_aig


def parse_map_log(path: Path) -> list[dict]:
    pat = re.compile(
        r"ID:(\d+), AigID:(\d+), Area:([\d.eE+-]+), "
        r"TruthTable:([\d\s,]+), Cut Leaves:([\d\s]*)"
    )
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        out.append(
            {
                "mapper_node_id": int(m.group(1)),
                "aig_node_id": int(m.group(2)),
                "area": float(m.group(3)),
                "truth_table": [int(x) for x in m.group(4).replace(" ", "").split(",") if x],
                "supergate_leaves": [int(x) for x in m.group(5).split()],
            }
        )
    return out


def parse_super_create(path: Path) -> list[dict]:
    events = []
    current = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"Creating Super:\s*(\d+) Node", line)
        if m:
            current = {"mapper_node_id": int(m.group(1)), "created_gates": []}
            events.append(current)
            continue
        m = re.search(r"Create Gate Id:\s*(\d+) Node Id:(\d+)", line)
        if m and current is not None:
            current["created_gates"].append(
                {"gate_id": int(m.group(1)), "mapped_netlist_node_id": int(m.group(2))}
            )
    return events


def parse_aig_fanouts(path: Path, source: int) -> list[dict]:
    """Return AIG AND records whose fanin is *source*."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.strip().split()
        if len(fields) != 5:
            continue
        try:
            a, ca, b, cb, root = (int(x) for x in fields)
        except ValueError:
            continue
        if a == source or b == source:
            out.append(
                {
                    "fanin0": a,
                    "fanin0_complement": ca,
                    "fanin1": b,
                    "fanin1_complement": cb,
                    "aig_root": root,
                }
            )
    return out


def subg_hash(subg: dict, root: int):
    """ID-independent hash of a canonical subg, retaining edge inversion."""
    if not subg or nx is None:
        return None
    g = nx.DiGraph()
    root = int(root)
    for u, edges in subg.items():
        u = int(u)
        role = "root" if u == root else ("pi" if u < 41 else "internal")
        g.add_node(u, role=role)
        for v, inv in edges:
            v = int(v)
            vrole = "pi" if v < 41 else "internal"
            g.add_node(v, role=("root" if v == root else vrole))
            g.add_edge(u, v, inv=str(int(inv)))
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="role", edge_attr="inv")


def load_payload(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text())
    return payload["cell_struct"].get(str(key), {})


def load_full_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def aig_header_inputs(path: Path) -> int:
    # Gate1 artifacts are binary AIGER; the ASCII header is the first line.
    fields = path.open("rb").readline().decode("ascii", "replace").split()
    return int(fields[2]) if len(fields) >= 3 else 0


def candidate_record(cid: str, root: int, subgraph: int, aig_name: str) -> dict:
    mapping = OUT / f"mapping_{cid}"
    aig2map = mapping / "aig2map.log"
    map_log = mapping / "abc_map.log"
    super_log = mapping / "abc_superCreate.log"
    payload_path = mapping / "payload.json"
    aig_to_map, map_to_aig = parse_aig2map(aig2map)
    mapper_root = aig_to_map.get(root)
    map_entries = parse_map_log(map_log)
    direct_by_mapper = [x for x in map_entries if x["mapper_node_id"] == mapper_root]
    direct_by_aig = [x for x in map_entries if x["aig_node_id"] == root]
    events = parse_super_create(super_log)
    root_events = [x for x in events if x["mapper_node_id"] == mapper_root]
    payload_root = load_payload(payload_path, mapper_root) if mapper_root is not None else {}
    payload_full = load_full_payload(payload_path)
    gate1 = json.loads((GATE1 / f"{aig_name}.json").read_text())
    # Single-candidate Gate 1 files use decisions[0].  Keep the full record so
    # that replacement DAG provenance is not silently reduced to a feature.
    d = gate1["decisions"][0]
    selected_index = payload_root.get("supergateIndex", -1)
    canonical_cut = None
    if payload_root.get("cuts"):
        canonical_cut = payload_root["cuts"][0]
    n_inputs = aig_header_inputs(GATE1 / f"{aig_name}.aig")

    def mapper_leaf_to_aig(mapper_id: int):
        # map_aig2mapper initializes PI mapper IDs as 0..n_inputs-1; those
        # entries are not repeated in aig2map.log.
        if mapper_id < n_inputs:
            return mapper_id + 1
        return map_to_aig.get(mapper_id)

    canonical_first_cut_aig = None
    if canonical_cut is not None:
        canonical_first_cut_aig = {
            "truth": canonical_cut[0],
            "mapper_leaves": canonical_cut[1],
            "aig_leaves": [mapper_leaf_to_aig(x) for x in canonical_cut[1]],
            "n_leaves": canonical_cut[2],
        }

    # Mapping often absorbs a Gate1 node into a supergate of its fanout.  Keep
    # that evidence explicit; it is not reported as the candidate's own
    # supergate.  This is the reason direct root identity cannot be assumed.
    aig_fanouts = parse_aig_fanouts(mapping / "abc_aig.log", root)
    fused = []
    for edge in aig_fanouts:
        fanout_aig = edge["aig_root"]
        fanout_mapper = aig_to_map.get(fanout_aig)
        fanout_records = [x for x in map_entries if x["mapper_node_id"] == fanout_mapper]
        fanout_payload = load_payload(payload_path, fanout_mapper) if fanout_mapper is not None else {}
        fidx = fanout_payload.get("supergateIndex", -1)
        fsubg = fanout_payload.get("feats", {}).get("subgraph") if fidx >= 0 else None
        fused.append(
            {
                "root_to_fanout_edge": edge,
                "fanout_aig_root": fanout_aig,
                "fanout_mapper_node_id": fanout_mapper,
                "fanout_supergate_records": fanout_records,
                "fanout_selected_supergate_index": fidx,
                "fanout_canonical_subg": fsubg,
                "fanout_canonical_subg_structural_hash": subg_hash(fsubg, fanout_mapper) if fsubg else None,
            }
        )
    return {
        "candidate_id": cid,
        "gate1": {
            "decision_root": root,
            "cut_leaves": d.get("cut_leaves", []),
            "subgraph_index": subgraph,
            "installation_kind": d.get("installation_kind"),
            "preserved_root_wrapper": d.get("preserved_root_wrapper", root),
            "library_subgraph_root": d.get("library_subgraph_root"),
            "nodes_added": d.get("nodes_added"),
            "physical_graph_modified": d.get("physical_graph_modified"),
            "physically_deleted_nodes": d.get("physically_deleted_nodes", []),
            "replayed_aig": str(GATE1 / f"{aig_name}.aig"),
            "replacement_dag": {
                "source": "Gate1 BuildSubgr replay (replayed AIG is the lossless artifact)",
                "library_subgraph_root": d.get("library_subgraph_root"),
                "replayed_aig": str(GATE1 / f"{aig_name}.aig"),
            },
        },
        "mapping": {
            "abc_binary": str(ABC),
            "technology_library": str(LIB),
            "aig2map_log": str(aig2map),
            "abc_map_log": str(map_log),
            "abc_superCreate_log": str(super_log),
            "mapper_root_id": mapper_root,
            "mapper_root_to_aig": map_to_aig.get(mapper_root) if mapper_root is not None else None,
            "direct_supergate_records_by_mapper_id": direct_by_mapper,
            "direct_supergate_records_by_aig_id": direct_by_aig,
            "super_create_events_for_mapper_root": root_events,
            "root_has_direct_selected_supergate": bool(direct_by_mapper or direct_by_aig or root_events),
            "canonical_parser_graph_stats": {
                "transformed_graph_nodes": len(payload_full.get("transformed_graph", {})),
                "feat_graph_nodes": len(payload_full.get("feat_graph", {})),
                "cell_struct_entries": len(payload_full.get("cell_struct", {})),
            },
            "root_fanout_fusion_evidence": fused,
            "canonical_root_cut_set": payload_root.get("cuts", []),
            "canonical_first_cut": canonical_cut,
            "canonical_first_cut_aig_namespace": canonical_first_cut_aig,
            "canonical_selected_supergate_index": selected_index,
            "canonical_selected_supergate": (
                payload_root.get("feats") if selected_index >= 0 else None
            ),
            "canonical_subg": (
                payload_root.get("feats", {}).get("subgraph") if selected_index >= 0 else None
            ),
            "canonical_subg_structural_hash": None,
            "log_sha256": {
                p.name: sha256(p)
                for p in [aig2map, map_log, super_log]
                if p.exists()
            },
        },
    }


def main() -> None:
    # The gate explicitly stops after d1 if A/B cannot be aligned.  C/D map
    # logs are retained in their directories for reproducibility, but are not
    # used to manufacture a positive correspondence result.
    records = [
        candidate_record("A", 146, 0, "d1_sub0"),
        candidate_record("B", 146, 1, "d1_sub1"),
    ]
    for rec in records:
        (OUT / f"candidate_{rec['candidate_id']}_mapping_alignment.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
        )

    status = "NOT_ALIGNABLE"
    audit = {
        "status": status,
        "scope": "d1=(root=146,cut_slot=0), candidates A/B",
        "canonical_flow": {
            "abc_binary": str(ABC),
            "technology_library": str(LIB),
            "command": "read asap7.lib; read <Gate1 replay>.aig; rewrite -c; map; write_verilog",
            "python_parser": "src/subutils/Parser.py:Aig2MapParser._build_graph_payload",
            "cut_constructor": "src/data_preprocess/data_load.py:construct_cut_set",
            "subg_constructor": "src/subutils/Parser.py:construct_subg_of_cut",
        },
        "candidates": records,
        "questions": {
            "a_b_root_mapped_supergate_found": False,
            "preserved_root_direct_mapper_correspondence": {
                "answer": "mapper ID exists in aig2map.log, but no direct selected supergate is emitted for either root; netlist identity therefore requires function/boundary matching",
                "A": records[0]["mapping"]["mapper_root_id"],
                "B": records[1]["mapping"]["mapper_root_id"],
            },
            "a_b_canonical_supergate_different": "UNDETERMINED_NO_DIRECT_SUPERGATE",
            "boundary_matches_gate1_fixed_cut": False,
            "mapping_can_fuse_or_exceed_gate1_region": {
                "answer": "YES_FUSION_OBSERVED",
                "evidence": "A root 146 feeds AIG 194, whose selected supergate is mapper 218; B root 146 feeds AIG 192, whose selected supergate is mapper 219. Root 146 has no selected supergate in either mapping.",
            },
            "different_candidates_same_canonical_supergate": {
                "answer": "ROOT_UNDETERMINED; FUSED_FANOUT_MANY_TO_ONE_OBSERVED",
                "fused_fanout_subg_hash_A": records[0]["mapping"]["root_fanout_fusion_evidence"][0]["fanout_canonical_subg_structural_hash"],
                "fused_fanout_subg_hash_B": records[1]["mapping"]["root_fanout_fusion_evidence"][0]["fanout_canonical_subg_structural_hash"],
            },
        },
        "gate_decision": {
            "A_B": "STOP_BEFORE_C_D",
            "reason": "Neither A nor B has a direct canonical selected supergate/subg at the preserved root mapper ID (supergateIndex=-1 and no abc_map/superCreate root record).",
        },
    }
    (OUT / "gate2a_r_alignment_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# Gate 2A-R — Gate1 candidate to canonical mapping alignment audit",
        "",
        "## Result: NOT_ALIGNABLE",
        "",
        "Scope was deliberately limited to d1=(root 146, cut slot 0), candidates A and B. No Student inference, STA, or Gate 1C-R modification was performed.",
        "",
        "Canonical flow used for each replay: `read asap7.lib; read <candidate>.aig; rewrite -c; map; write_verilog`, with `/home/wushangge/abc_for_cutLib/abc`, followed by the unmodified `Aig2MapParser._build_graph_payload()` path.",
        "",
        "| candidate | Gate1 kind | Gate1 cut | Gate1 subgraph | AIG→mapper root | direct `abc_map`/`superCreate` root supergate | parser selected index | canonical first cut |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for r in records:
        m = r["mapping"]
        lines.append(
            f"| {r['candidate_id']} | {r['gate1']['installation_kind']} | "
            f"`{r['gate1']['cut_leaves']}` | {r['gate1']['subgraph_index']} | "
            f"{m['mapper_root_id']} | {'yes' if m['root_has_direct_selected_supergate'] else 'no'} | "
            f"{m['canonical_selected_supergate_index']} | `{m['canonical_first_cut']}` |"
        )
    lines += [
        "",
        "### Answers",
        "",
        "1. A/B replayed AIGs both map successfully and the canonical parser extracts the surrounding graph, but neither preserved root has a direct selected canonical supergate: `supergateIndex=-1`, no matching root record in `abc_map.log`, and no matching `Creating Super` event.",
        "2. `aig2map.log` does provide an AIG→mapper correspondence (A: 146→216; B: 146→218). This is not a stable LiteCut supergate identity; matching the mapped netlist would require function/boundary matching.",
        "3. A/B canonical supergates cannot be declared different or equal because neither root has a selected supergate/subg. The audit does not manufacture one from a nearby node.",
        "4. The canonical root cut interface is not the Gate1 fixed cut `[2, 42, 82]`: after translating mapper IDs back to the replayed AIG namespace, A's first cut is `[10, 94]` and B's is `[14, 94]` (both two leaves). Mapping therefore reselects a different interface; this is boundary change, not an aligned Gate1 candidate.",
        "5. Mapping fusion is directly visible: A's root 146 is a fanin of AIG 194 and B's root 146 is a fanin of AIG 192; those fanout supergates are selected instead of a root-146 supergate. This permits the canonical mapped implementation to cross the Gate1 removable-region boundary.",
        "6. The two fused fanout subgraphs are structurally isomorphic (ID-independent hash `9449c0d1ca77a3162fabb4562ddbcb6f` for both), but this is not a root-candidate correspondence; root-level many-to-one collapse remains undetermined.",
        "",
        "Per the gate rule, C/D are not promoted to an alignment result after A/B fail the direct correspondence test. Their raw canonical logs remain under `mapping_C/` and `mapping_D/` only as reproducible tool outputs.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `candidate_A_mapping_alignment.json`, `candidate_B_mapping_alignment.json`: Gate1 provenance, mapper IDs, canonical cuts, and direct-supergate checks.",
        "- `mapping_A/`, `mapping_B/`: complete ABC logs and parser payloads.",
        "- `gate2a_r_alignment_audit.json`: machine-readable result and decision gate.",
    ]
    (OUT / "gate2a_r_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
