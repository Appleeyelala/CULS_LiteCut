"""Replot selected MoE heatmaps with clean style for paper figures."""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def apply_style() -> None:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({
        "font.size": 16,
        "font.weight": "bold",
        "font.family": "times new roman",
        "pdf.fonttype": 42,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "text.usetex": False,
        "text.parse_math": False,
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
    })


def _clean_bucket_label(label: str) -> str:
    """'Q1 [5.0, 18.3]' -> '[5.0, 18.3]', 'Q2 (18.3, 29.6]' -> '(18.3, 29.6]'"""
    cleaned = re.sub(r"^Q\d+\s*", "", label)
    return cleaned


def plot_clean_heatmap(csv_path: Path, output_prefix: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if int(r["count"]) > 0]
    if not rows:
        return

    expert_keys = [k for k in rows[0] if k.startswith("expert_")]
    values = np.asarray([[float(r[k]) for k in expert_keys] for r in rows], dtype=np.float64)
    bucket_labels = [_clean_bucket_label(str(r["bucket"])) for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0), constrained_layout=True)
    value_max = float(np.max(values)) if values.size else 0.0
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(1.0, value_max))

    # No title
    ax.set_xlabel("Expert ID", fontsize=18, fontweight="bold")
    ax.set_ylabel("Delay Bucket", fontsize=18, fontweight="bold")
    ax.set_xticks(np.arange(len(expert_keys)))
    ax.set_xticklabels([k.replace("expert_", "E") for k in expert_keys], fontweight="bold", fontsize=15)
    ax.set_yticks(np.arange(len(bucket_labels)))
    ax.set_yticklabels(bucket_labels, fontweight="bold", fontsize=14)

    for ri in range(values.shape[0]):
        for ci in range(values.shape[1]):
            ax.text(ci, ri, f"{values[ri, ci]:.2f}", ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")

    # Colorbar: no label text
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    colorbar.ax.tick_params(labelsize=13)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"CurPID: {os.getpid()}")
    apply_style()

    cases = {
        "ACE_b18_1_comb": "moe_fixed/save_json_ACE_b18_1_comb/student_moe_heatmap_bycase.csv",
        "forgeEDA_cabac_ucontext": "moe_fixed/save_json_forgeEDA_cabac_ucontext/student_moe_heatmap_bycase.csv",
        "OpenABC_ac97_ctrl": "moe_fixed/save_json_OpenABC_ac97_ctrl_orig/student_moe_heatmap_bycase.csv",
    }

    output_dir = Path("moe_fixed/paper_figures")
    for name, csv_path in cases.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"{_now()} WARNING: {csv_path} not found, skipping {name}")
            continue
        out = output_dir / f"heatmap_{name}"
        print(f"{_now()} Plotting {name} -> {out}")
        plot_clean_heatmap(csv_path, out)

    print(f"{_now()} Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
