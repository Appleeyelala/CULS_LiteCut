from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

SCALAR_KEYS = [
    "format_version",
    "model_family",
    "shared_hidden_dim",
    "num_experts",
    "gate_temperature",
    "hidden_activation",
    "group_fusion_policy",
    "expert_routing_policy",
    "output_denorm",
    "input_dim",
    "output_mu",
    "output_inv_sigma",
    "clamp_min",
    "clamp_max",
]
VECTOR_KEYS = [
    "group_dims",
    "input_mu",
    "input_inv_sigma",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Student checkpoint payload for ABC runtime loading")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint containing abc_inference_payload")
    parser.add_argument("--output", required=True, help="Path to output runtime model text file")
    return parser.parse_args()


def _load_payload(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    payload = checkpoint.get("abc_inference_payload")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain abc_inference_payload")
    if "state_dict" not in payload or not isinstance(payload["state_dict"], dict):
        raise ValueError("abc_inference_payload is missing state_dict")
    return payload


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported scalar type for export: {type(value).__name__}")


def _format_array(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_format_array(item) for item in value) + "]"
    return _format_scalar(value)


def _iter_export_lines(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in SCALAR_KEYS:
        if key not in payload:
            raise ValueError(f"abc_inference_payload missing required key: {key}")
        lines.append(f"{key}={_format_scalar(payload[key])}")
    for key in VECTOR_KEYS:
        if key not in payload:
            raise ValueError(f"abc_inference_payload missing required key: {key}")
        lines.append(f"{key}={_format_array(payload[key])}")
    for key, value in sorted(payload["state_dict"].items()):
        lines.append(f"state_dict.{key}={_format_array(value)}")
    return lines


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(checkpoint_path)
    lines = _iter_export_lines(payload)
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote ABC Student runtime model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
