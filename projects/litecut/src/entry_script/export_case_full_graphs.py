from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from subutils.Parser import Aig2MapParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export full transformed_graph and feat_graph for cases mentioned in "
            "the baseline neighbor analysis report."
        )
    )
    parser.add_argument(
        "--analysis_json",
        type=str,
        default="./log/cutlib_baseline_neighbor_analysis.json",
        help="Path to the baseline neighbor analysis JSON report",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/zhoulingfeng/data/open-sourceDataset",
        help="Root directory that contains dataset subdirectories with .aig files",
    )
    parser.add_argument(
        "--map_root",
        type=str,
        default="/data/zhoulingfeng/data_cutLib/cutLibData2",
        help="Root directory that contains ABC mapping logs and save_json_* outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/case_full_graphs",
        help="Directory to store exported full-graph artifacts",
    )
    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=None,
        help="Optional explicit case ids like OpenABC/ac97_ctrl_orig; overrides analysis_json collection",
    )
    parser.add_argument(
        "--include_queries",
        action="store_true",
        default=True,
        help="Collect query case ids from the analysis report",
    )
    parser.add_argument(
        "--include_neighbors",
        action="store_true",
        default=True,
        help="Collect neighbor case ids from the analysis report",
    )
    parser.add_argument(
        "--neighbor_limit",
        type=int,
        default=None,
        help="Only take the first N neighbors from each query; default uses all neighbors",
    )
    parser.add_argument(
        "--save_json_graph",
        action="store_true",
        help="Also save transformed_graph.json and feat_graph.json besides full_graph.pt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exports",
    )
    return parser.parse_args()


def _load_case_ids_from_report(
    analysis_json: str,
    include_queries: bool,
    include_neighbors: bool,
    neighbor_limit: int | None,
) -> list[str]:
    with open(analysis_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    case_ids: list[str] = []
    for item in report.get("query_results", []):
        if include_queries:
            query_case_id = (item.get("query") or {}).get("case_id")
            if query_case_id:
                case_ids.append(query_case_id)
        if include_neighbors:
            neighbors = item.get("neighbors") or []
            if neighbor_limit is not None:
                neighbors = neighbors[:neighbor_limit]
            for neighbor in neighbors:
                neighbor_case_id = neighbor.get("case_id")
                if neighbor_case_id:
                    case_ids.append(neighbor_case_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        deduped.append(case_id)
    return deduped


def _resolve_dataset_dir(dataset_root: str, dataset: str) -> str:
    candidate = os.path.join(dataset_root, dataset)
    if os.path.isdir(candidate):
        return candidate

    for entry in os.scandir(dataset_root):
        if entry.is_dir() and entry.name.lower() == dataset.lower():
            return entry.path

    raise FileNotFoundError(f"Dataset directory not found for {dataset} under {dataset_root}")


def main() -> None:
    args = parse_args()

    case_ids = args.case_ids
    if not case_ids:
        case_ids = _load_case_ids_from_report(
            analysis_json=args.analysis_json,
            include_queries=args.include_queries,
            include_neighbors=args.include_neighbors,
            neighbor_limit=args.neighbor_limit,
        )

    if not case_ids:
        raise RuntimeError("No case ids were collected for export")

    dataset_to_designs: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        if "/" not in case_id:
            raise ValueError(f"Invalid case id format: {case_id}")
        dataset, design = case_id.split("/", 1)
        dataset_to_designs[dataset].append(design)

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset, designs in sorted(dataset_to_designs.items()):
        dataset_dir = _resolve_dataset_dir(args.dataset_root, dataset)
        parser = Aig2MapParser(os.path.join(args.map_root, dataset), "./log/export_case_full_graphs.log")
        export_root = os.path.join(args.output_dir, dataset)

        print(f"[{dataset}] exporting {len(designs)} case(s) to {export_root}")
        for design in designs:
            aig_path = os.path.join(dataset_dir, f"{design}.aig")
            if not os.path.isfile(aig_path):
                raise FileNotFoundError(f"AIG file not found: {aig_path}")
            parser.export_full_graph(
                aig_path,
                export_root,
                save_json_graph=args.save_json_graph,
                overwrite=args.overwrite,
            )
            print(f"  saved {dataset}/{design}")


if __name__ == "__main__":
    main()
