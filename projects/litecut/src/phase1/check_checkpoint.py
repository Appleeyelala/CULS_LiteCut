"""Inspect a LiteCut checkpoint without running model inference."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

def inspect(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    result = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
              "has_abc_inference_payload": False, "prediction_status": "unavailable"}
    if not isinstance(payload, dict):
        result["reason"] = "checkpoint_not_mapping"; return result
    abc = payload.get("abc_inference_payload")
    if isinstance(abc, dict):
        result.update({"has_abc_inference_payload": True, "model_family": abc.get("model_family"),
                       "input_dim": abc.get("input_dim"), "prediction_status": "payload_available"})
    else:
        result["reason"] = "missing_abc_inference_payload"
    return result

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("checkpoint", type=Path); p.add_argument("--output", type=Path)
    a = p.parse_args(); text = json.dumps(inspect(a.checkpoint), indent=2)
    if a.output: a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(text + "\n", encoding="utf-8")
    print(text)
if __name__ == "__main__": main()
