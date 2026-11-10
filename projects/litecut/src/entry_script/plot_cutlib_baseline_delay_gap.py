from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.stats import gamma as _gamma_dist
from scipy.stats import gaussian_kde as _gaussian_kde


def _set_font_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "font.weight": "bold",
            "font.family": "times new roman",
            "pdf.fonttype": 42,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.parse_math"] = False
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "dejavusans"


_set_font_rcparams()

def format_case_label(case_name: str, cut_root: str, design_idx: int) -> str:
    return f"Cut {cut_root}\nDesign {design_idx}"


def apply_publication_style() -> str:
    if importlib.util.find_spec("scienceplots") is not None:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
        _set_font_rcparams()
        return "scienceplots"

    if importlib.util.find_spec("seaborn") is not None:
        import seaborn as sns

        sns.set_theme(
            context="paper",
            style="whitegrid",
            palette="colorblind",
            font_scale=1.05,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.alpha": 0.28,
                "grid.linestyle": "--",
            },
        )
        _set_font_rcparams()
        return "seaborn"

    _set_font_rcparams()
    return "matplotlib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot delay gaps from cutlib baseline neighbor analysis to show that "
            "structurally identical or similar cuts can still have very different delays."
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="./log/cutlib_baseline_neighbor_analysis.json",
        help="Path to cutlib baseline neighbor analysis JSON",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="./log/cutlib_baseline_delay_gap.pdf",
        help="Path to output figure PDF",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./log/cutlib_baseline_delay_gap_summary.csv",
        help="Path to output summary CSV",
    )
    parser.add_argument(
        "--top_n_queries",
        type=int,
        default=5,
        help="Plot the top-n queries with the most same-structure neighbors",
    )
    parser.add_argument(
        "--min_iqr_span",
        "--min_ratio_span",
        dest="min_iqr_span",
        type=float,
        default=2.0,
        help="Filter out queries whose Q3/Q1 delay ratio span is smaller than this threshold",
    )
    parser.add_argument(
        "--require_both_sides",
        action="store_true",
        default=True,
        help="Only keep queries whose same-structure ratios exist on both sides of 1.0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["box", "normal", "compare"],
        default="box",
        help="box: boxplot of delay ratios; normal: histogram + Gaussian fit; compare: histogram with Gaussian/LogNormal/Gamma/KDE overlays",
    )
    parser.add_argument(
        "--single_index",
        type=int,
        default=None,
        help="If set with --mode normal, render only this 1-based group index in a standalone figure (no subplot title)",
    )
    return parser.parse_args()


def load_analysis(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_signature_delays(report: dict) -> dict[str, list[float]]:
    signature_to_delays: dict[str, list[float]] = defaultdict(list)
    for item in report.get("query_results", []):
        query = item["query"]
        signature_to_delays[str(query["cell_struct_signature"])].append(float(query["delay"]))
        for neighbor in item.get("neighbors", []):
            signature_to_delays[str(neighbor["cell_struct_signature"])].append(float(neighbor["delay"]))
    return signature_to_delays


def collect_query_groups(report: dict) -> list[dict]:
    groups = []
    for idx, item in enumerate(report.get("query_results", []), start=1):
        query = item["query"]
        same_signature = str(query["cell_struct_signature"])
        neighbors = [
            neighbor
            for neighbor in item.get("neighbors", [])
            if str(neighbor.get("cell_struct_signature")) == same_signature
        ]
        query_delay = float(query["delay"])
        neighbor_delays = sorted(float(neighbor["delay"]) for neighbor in neighbors)
        neighbor_ratios = [delay / max(query_delay, 1e-12) for delay in neighbor_delays]
        groups.append(
            {
                "label": f"Q{idx}",
                "case_id": query["case_id"],
                "cell_struct_key": query["cell_struct_key"],
                "cell_struct_signature": same_signature,
                "query_delay": query_delay,
                "query_ratio": 1.0,
                "neighbor_delays": neighbor_delays,
                "neighbor_ratios": neighbor_ratios,
                "neighbor_min": min(neighbor_delays) if neighbor_delays else query_delay,
                "neighbor_max": max(neighbor_delays) if neighbor_delays else query_delay,
                "neighbor_median": float(np.median(neighbor_delays)) if neighbor_delays else query_delay,
                "ratio_vs_query": (
                    max(max(neighbor_delays), float(query["delay"]))
                    / max(min(min(neighbor_delays), float(query["delay"])), 1e-12)
                    if neighbor_delays
                    else 1.0
                ),
            }
        )
    return groups


def summarize_signature_groups(signature_to_delays: dict[str, list[float]]) -> list[dict]:
    summary = []
    for signature, delays in signature_to_delays.items():
        if len(delays) < 2:
            continue
        delay_min = min(delays)
        delay_max = max(delays)
        summary.append(
            {
                "signature": signature,
                "count": len(delays),
                "delay_min": delay_min,
                "delay_max": delay_max,
                "delay_median": float(np.median(delays)),
                "delay_ratio": delay_max / max(delay_min, 1e-12),
            }
        )
    summary.sort(key=lambda row: (-row["delay_ratio"], -row["count"], row["signature"]))
    return summary


def collect_top_query_groups(
    report: dict,
    top_n_queries: int,
    min_iqr_span: float | None = None,
    require_both_sides: bool = True,
    min_ratio_span: float | None = None,
) -> list[dict]:
    if min_iqr_span is None:
        min_iqr_span = 2.0 if min_ratio_span is None else min_ratio_span

    query_groups = []
    for item in report.get("query_results", []):
        query = item["query"]
        case_name = query["case_id"].split("/")[-1]
        query_delay = float(query["delay"])
        query_signature = str(query["cell_struct_signature"])
        same_signature_neighbors = [
            neighbor
            for neighbor in item.get("neighbors", [])
            if str(neighbor.get("cell_struct_signature")) == query_signature
        ]
        neighbor_ratios_only = [
            float(neighbor["delay"]) / max(query_delay, 1e-12) for neighbor in same_signature_neighbors
        ]
        if require_both_sides and neighbor_ratios_only:
            has_smaller = min(neighbor_ratios_only) < 1.0
            has_larger = max(neighbor_ratios_only) > 1.0
            if not (has_smaller and has_larger):
                continue

        ratios = [1.0] + [float(neighbor["delay"]) / max(query_delay, 1e-12) for neighbor in same_signature_neighbors]
        ratios_arr = np.asarray(ratios, dtype=np.float64)

        def _area_of(rec):
            return (rec.get("cell_struct_summary") or {}).get("area")

        all_neighbors = item.get("neighbors", [])
        delays_all = [query_delay] + [float(n["delay"]) for n in all_neighbors]
        areas_all = [_area_of(query)] + [_area_of(n) for n in all_neighbors]
        ratio_min = float(ratios_arr.min())
        ratio_q1 = float(np.percentile(ratios_arr, 25))
        ratio_q3 = float(np.percentile(ratios_arr, 75))
        ratio_max = float(ratios_arr.max())
        iqr_ratio_span = ratio_q3 / max(ratio_q1, 1e-12)
        if iqr_ratio_span < min_iqr_span:
            continue
        query_groups.append(
            {
                "case_name": case_name,
                "cut_root": str(query["cell_struct_key"]),
                "query_delay": query_delay,
                "same_structure_count": len(same_signature_neighbors),
                "sample_count": len(ratios),
                "ratios": ratios,
                "ratio_min": ratio_min,
                "ratio_q1": ratio_q1,
                "ratio_median": float(np.median(ratios_arr)),
                "ratio_q3": ratio_q3,
                "ratio_max": ratio_max,
                "iqr_ratio_span": iqr_ratio_span,
                "delays_all": delays_all,
                "areas_all": areas_all,
            }
        )

    query_groups.sort(
        key=lambda row: (
            -row["same_structure_count"],
            -row["iqr_ratio_span"],
            -row["sample_count"],
            row["case_name"],
            row["cut_root"],
        )
    )
    return query_groups[:top_n_queries]


def save_summary_csv(path: str, query_groups: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "case_name,cut_root,same_structure_count,sample_count,"
            "ratio_min,ratio_q1,ratio_median,ratio_q3,ratio_max,iqr_ratio_span\n"
        )
        for row in query_groups:
            f.write(
                f"{row['case_name']},{row['cut_root']},{row['same_structure_count']},{row['sample_count']},"
                f"{row['ratio_min']},{row['ratio_q1']},{row['ratio_median']},{row['ratio_q3']},"
                f"{row['ratio_max']},{row['iqr_ratio_span']}\n"
            )


def _filter_outliers_iqr(values: list[float], k: float = 1.5) -> list[float]:
    """Remove extreme outliers using IQR method (values outside [Q1 - k*IQR, Q3 + k*IQR])."""
    arr = np.asarray(values, dtype=np.float64)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return arr[(arr >= lower) & (arr <= upper)].tolist()


def plot_case_panel(ax, query_groups: list[dict]) -> None:
    box_face = "#91CAE8"
    box_edge = "#000000"
    accent = "#000000"
    flier = "#C77C2E"

    positions = np.arange(1, len(query_groups) + 1)
    ratio_groups = [_filter_outliers_iqr(row["ratios"]) for row in query_groups]

    ax.boxplot(
        ratio_groups,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": accent, "linewidth": 1.6},
        boxprops={"facecolor": box_face, "edgecolor": box_edge, "linewidth": 1.2, "alpha": 0.92},
        whiskerprops={"color": box_edge, "linewidth": 1.2},
        capprops={"color": box_edge, "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": flier,
            "markeredgecolor": flier,
            "markersize": 3.6,
            "alpha": 0.7,
        },
    )

    ax.set_ylabel("Delay Ratio", fontsize=18, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [format_case_label(row["case_name"], row["cut_root"], i) for i, row in enumerate(query_groups, start=1)],
        rotation=0,
        fontweight="bold",
        fontsize=15,
    )
    ax.tick_params(axis="y", labelsize=14)
    sfmt = ScalarFormatter(useOffset=False)
    sfmt.set_useMathText(False)
    ax.yaxis.set_major_formatter(sfmt)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _fit_distributions(arr: np.ndarray, scale: float) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """Fit Gaussian / LogNormal / Gamma / KDE to a non-negative sample array.

    Returns dict[name] = (xs, ys, label). ys are scaled to histogram counts via `scale`
    (= n_samples * bin_width), so they overlay directly on a count-based histogram.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    n = arr.size
    if n < 2:
        return out

    sigma = float(arr.std(ddof=1))
    mu = float(arr.mean())
    x_lo = max(0.0, float(arr.min()) - 0.5 * (sigma if sigma > 0 else 1.0))
    x_hi = float(arr.max()) + 0.5 * (sigma if sigma > 0 else 1.0)
    xs = np.linspace(x_lo, x_hi, 400)
    xs_pos = np.linspace(max(x_lo, 1e-9), x_hi, 400)

    # Gaussian
    if sigma > 0:
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        out["Gaussian"] = (xs, pdf * scale, f"Normal(μ={mu:.1f}, σ={sigma:.1f})")

    # Log-Normal (fit log(arr) ~ Normal); requires strictly positive samples
    pos = arr[arr > 0]
    if pos.size >= 2:
        log_arr = np.log(pos)
        mu_log = float(log_arr.mean())
        sigma_log = float(log_arr.std(ddof=1))
        if sigma_log > 0:
            pdf_ln = (1.0 / (xs_pos * sigma_log * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((np.log(xs_pos) - mu_log) / sigma_log) ** 2
            )
            out["LogNormal"] = (
                xs_pos,
                pdf_ln * scale,
                f"LogN(med={np.exp(mu_log):.1f}, σ_log={sigma_log:.2f})",
            )

    # Gamma (force loc=0 since delays are positive)
    try:
        shape_k, _loc, theta = _gamma_dist.fit(arr, floc=0)
        if shape_k > 0 and theta > 0:
            pdf_g = _gamma_dist.pdf(xs_pos, shape_k, loc=0, scale=theta)
            out["Gamma"] = (xs_pos, pdf_g * scale, f"Gamma(k={shape_k:.2f}, θ={theta:.1f})")
    except Exception:
        pass

    # KDE with reflection at 0 boundary: kde over [-arr, arr], then density(x>=0)*2
    if n >= 2 and arr.std(ddof=1) > 0:
        reflected = np.concatenate([-arr, arr])
        kde = _gaussian_kde(reflected)
        pdf_k = kde(xs_pos) * 2.0
        out["KDE"] = (xs_pos, pdf_k * scale, f"KDE(N={n})")

    return out


def plot_compare_panel(ax, group: dict, global_idx: int, show_title: bool = True) -> None:
    bar_color = "#6888F5"
    curve_colors = {
        "Gaussian": "#D77071",
        "LogNormal": "#2E8B57",
        "Gamma": "#8E44AD",
        "KDE": "#E67E22",
    }

    raw_delays = [r * group["query_delay"] for r in group["ratios"]]
    delays = _filter_outliers_iqr(raw_delays)
    if len(delays) < 2:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        if show_title:
            ax.set_title(f"Cut {group['cut_root']}\nDesign {global_idx}", fontweight="bold")
        return

    arr = np.asarray(delays, dtype=np.float64)
    n_bins = max(6, min(20, int(np.sqrt(arr.size) * 1.5)))
    counts, edges, _ = ax.hist(
        arr, bins=n_bins, density=False, color=bar_color, edgecolor="#000000", linewidth=0.6, alpha=0.85
    )
    bin_width = float(edges[1] - edges[0]) if len(edges) >= 2 else 1.0
    scale = arr.size * bin_width

    fits = _fit_distributions(arr, scale)
    for name, (xs, ys, label) in fits.items():
        ax.plot(xs, ys, color=curve_colors.get(name, "#000000"), linewidth=1.8, alpha=0.9, label=label)

    if fits:
        ax.legend(loc="upper right", fontsize=11, frameon=False)

    if show_title:
        ax.set_title(f"Cut {group['cut_root']}\nDesign {global_idx}", fontweight="bold")
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    sfmt = ScalarFormatter(useOffset=False)
    sfmt.set_useMathText(False)
    ax.xaxis.set_major_formatter(sfmt)
    ax.yaxis.set_major_formatter(sfmt)


def plot_normal_panel(ax, group: dict, global_idx: int, show_title: bool = True) -> None:
    bar_color = "#6888F5"
    line_color = "#D77071"

    raw_delays = [r * group["query_delay"] for r in group["ratios"]]
    delays = _filter_outliers_iqr(raw_delays)
    if len(delays) < 2:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        if show_title:
            ax.set_title(f"Cut {group['cut_root']}\nDesign {global_idx}", fontweight="bold")
        return

    arr = np.asarray(delays, dtype=np.float64)
    n_bins = max(6, min(20, int(np.sqrt(arr.size) * 1.5)))
    counts, edges, _ = ax.hist(
        arr, bins=n_bins, density=False, color=bar_color, edgecolor="#000000", linewidth=0.6, alpha=0.9
    )
    bin_width = float(edges[1] - edges[0]) if len(edges) >= 2 else 1.0
    scale = arr.size * bin_width

    ax_right = ax.twinx()
    if arr.size >= 2 and arr.std(ddof=1) > 0:
        try:
            shape_k, _loc, theta = _gamma_dist.fit(arr, floc=0)
        except Exception:
            shape_k, theta = 0.0, 0.0
        if shape_k > 0 and theta > 0:
            x_lo = max(0.0, float(arr.min()) - 0.5 * theta)
            x_hi = float(arr.max()) + 0.5 * theta
            xs = np.linspace(max(x_lo, 1e-9), x_hi, 400)
            pdf = _gamma_dist.pdf(xs, shape_k, loc=0, scale=theta)
            ax_right.plot(xs, pdf * scale, color=line_color, linewidth=2.0, label=f"Gamma(k={shape_k:.2f}, θ={theta:.1f})")
            ax_right.legend(loc="upper right", fontsize=13, frameon=False)
    ax_right.set_ylim(bottom=0)
    ax_right.set_yticks([])
    ax_right.spines["right"].set_visible(False)

    if show_title:
        ax.set_title(f"Cut {group['cut_root']}\nDesign {global_idx}", fontweight="bold")
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("Count")
    x_right = float(arr.max()) + 0.5 * (theta if (shape_k > 0 and theta > 0) else (float(arr.std(ddof=1)) or 1.0))
    ax.set_xlim(left=-0.04 * x_right, right=x_right)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    sfmt = ScalarFormatter(useOffset=False)
    sfmt.set_useMathText(False)
    ax.xaxis.set_major_formatter(sfmt)
    ax.yaxis.set_major_formatter(sfmt)
    ax_right.yaxis.set_major_formatter(sfmt)


def build_figure(
    report: dict,
    output_png: str,
    output_csv: str,
    top_n_queries: int,
    min_iqr_span: float,
    require_both_sides: bool,
    mode: str = "box",
    single_index: int | None = None,
) -> list[dict]:
    apply_publication_style()
    query_groups = collect_top_query_groups(report, top_n_queries, min_iqr_span, require_both_sides)

    if mode in ("normal", "compare"):
        panel_fn = plot_normal_panel if mode == "normal" else plot_compare_panel
        if single_index is not None:
            idx = single_index - 1
            if idx < 0 or idx >= len(query_groups):
                raise IndexError(f"single_index {single_index} out of range (have {len(query_groups)} groups)")
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), constrained_layout=True)
            panel_fn(ax, query_groups[idx], single_index, show_title=False)
        else:
            selected = query_groups[:9]
            fig, axes = plt.subplots(3, 3, figsize=(15.0, 13.0), constrained_layout=True)
            flat_axes = axes.flatten()
            for i, (ax, group) in enumerate(zip(flat_axes, selected), start=1):
                panel_fn(ax, group, i)
            for j in range(len(selected), 9):
                flat_axes[j].set_visible(False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.2), constrained_layout=True)
        plot_case_panel(ax, query_groups)

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    save_summary_csv(output_csv, query_groups)
    return query_groups


def main() -> None:
    args = parse_args()
    report = load_analysis(args.input_json)
    query_groups = build_figure(
        report,
        args.output_png,
        args.output_csv,
        args.top_n_queries,
        args.min_iqr_span,
        args.require_both_sides,
        args.mode,
        args.single_index,
    )

    print(f"Saved figure to {args.output_png}")
    print(f"Saved summary to {args.output_csv}")
    if query_groups:
        top = query_groups[0]
        print(
            "Top same-structure-heavy query:",
            f"case={top['case_name']}, cut_root={top['cut_root']}, "
            f"same_structure_count={top['same_structure_count']}, sample_count={top['sample_count']}, "
            f"ratio range=[{top['ratio_min']:.6f}, {top['ratio_max']:.6f}], "
            f"IQR span=[{top['ratio_q1']:.6f}, {top['ratio_q3']:.6f}], "
            f"iqr_ratio_span={top['iqr_ratio_span']:.2f}x",
        )
    for row in query_groups[: min(len(query_groups), 10)]:
        print(
            f"{row['case_name']}:{row['cut_root']}: same_structure_count={row['same_structure_count']}, "
            f"sample_count={row['sample_count']}, "
            f"ratio range=[{row['ratio_min']:.6f}, {row['ratio_max']:.6f}], "
            f"IQR span=[{row['ratio_q1']:.6f}, {row['ratio_q3']:.6f}], "
            f"iqr_ratio_span={row['iqr_ratio_span']:.2f}x",
        )


if __name__ == "__main__":
    main()
