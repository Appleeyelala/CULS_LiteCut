"""Measure per-cut inference time (us) for DeepCut / Teacher / Student / Student-nodistill."""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

from validate.eval_prediction_accuracy import (
    FEATURE_INPUT_DIM,
    load_case_data,
    load_student_model,
    load_teacher_model,
    standardize_with_params,
)


def _now() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _model_cpu_memory_bytes(model: torch.nn.Module) -> int:
    """Return total bytes of all parameters + buffers when stored on CPU."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total


def _time_forward(forward_fn, device, warmup=5, repeats=50):
    """Run forward_fn warmup+repeats times, return mean seconds of the repeats."""
    use_cuda = device != "cpu"
    for _ in range(warmup):
        forward_fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn()
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--picks_csv", required=True, help="main_table_picks.csv")
    p.add_argument("--accuracy_csv", required=True, help="prediction_accuracy_picks.csv")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--student_nodistill_ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=2048)
    return p.parse_args()


def main() -> int:
    print(f"CurPID: {os.getpid()}")
    args = parse_args()

    # --- 读取主表获取 rt_deepcut ---
    with open(args.picks_csv, encoding="utf-8") as f:
        picks = list(csv.DictReader(f))
    rt_deepcut_map = {(r["dataset"], r["case"]): float(r["rt_deepcut"]) for r in picks}

    # --- 读取精度表 ---
    with open(args.accuracy_csv, encoding="utf-8") as f:
        acc_rows = list(csv.DictReader(f))
    print(f"{_now()} Loaded {len(acc_rows)} rows from {args.accuracy_csv}")

    # --- 加载 norm params ---
    ckpt = torch.load(args.student_ckpt, map_location="cpu")
    x_mu = np.asarray(ckpt["x_mu"], dtype=np.float32)
    x_inv = np.asarray(ckpt["x_inv_sigma"], dtype=np.float32)

    # --- 加载模型 ---
    print(f"{_now()} Loading Teacher...")
    teacher_model, _ = load_teacher_model(args.teacher_ckpt, args.device)
    print(f"{_now()} Loading Student...")
    student_model, _ = load_student_model(args.student_ckpt, args.device)
    nodistill_model = None
    if args.student_nodistill_ckpt and os.path.isfile(args.student_nodistill_ckpt):
        print(f"{_now()} Loading Student (no-distill)...")
        nodistill_model, _ = load_student_model(args.student_nodistill_ckpt, args.device)

    # --- 模型 CPU 内存占用 ---
    teacher_mem = _model_cpu_memory_bytes(teacher_model)
    student_mem = _model_cpu_memory_bytes(student_model)
    nodistill_mem = _model_cpu_memory_bytes(nodistill_model) if nodistill_model else 0
    print(f"{_now()} Model CPU memory footprint:")
    print(f"{_now()}   Teacher:            {teacher_mem:>10d} bytes  ({teacher_mem / 1024:.2f} KB)")
    print(f"{_now()}   Student:            {student_mem:>10d} bytes  ({student_mem / 1024:.2f} KB)")
    if nodistill_model:
        print(f"{_now()}   Student (no-distill):{nodistill_mem:>10d} bytes  ({nodistill_mem / 1024:.2f} KB)")

    # --- 逐 case 计时 ---
    for row in acc_rows:
        ds, case = row["dataset"], row["case"]
        n_cuts = int(row["n_cuts"])
        print(f"{_now()} {ds}/{case}  n_cuts={n_cuts}")

        # DeepCut
        rt_dc = rt_deepcut_map.get((ds, case), 0.0)
        row["deepcut_us_per_cut"] = f"{rt_dc * 1e6 / n_cuts:.4f}" if n_cuts else ""

        # 加载数据
        x_raw, _ = load_case_data(args.dataset_path, ds, case)
        x_scaled = standardize_with_params(x_raw, x_mu, x_inv)
        x_t = torch.from_numpy(x_scaled).float().to(args.device)

        # wl_hash for Teacher
        wl_hash = np.zeros(n_cuts, dtype=np.int64)
        try:
            from subutils.extraction_utils import _load_single_json_case
            suffix = f"{case}_orig" if ds == "OpenABC" else case
            jp = os.path.join(args.dataset_path, f"save_json_{ds}", suffix, "data_2.json")
            _, xf, _, _, _ = _load_single_json_case(jp, include_graph=False)
            if xf is not None and xf.shape[1] > FEATURE_INPUT_DIM:
                wl_hash = xf[:, -1].astype(np.int64)
        except Exception:
            pass
        wl_t = torch.from_numpy(wl_hash).long().to(args.device)

        bs = args.batch_size

        # Teacher timing
        def _teacher_fwd(x=x_t, w=wl_t):
            with torch.inference_mode():
                for s in range(0, len(x), bs):
                    teacher_model(x[s:s+bs], w[s:s+bs], None)

        t_teacher = _time_forward(_teacher_fwd, args.device)
        row["teacher_us_per_cut"] = f"{t_teacher * 1e6 / n_cuts:.4f}"

        # Student timing
        def _student_fwd(x=x_t):
            with torch.inference_mode():
                for s in range(0, len(x), bs):
                    student_model(x[s:s+bs])

        t_student = _time_forward(_student_fwd, args.device)
        row["student_us_per_cut"] = f"{t_student * 1e6 / n_cuts:.4f}"

        # Student no-distill timing
        if nodistill_model is not None:
            def _nodistill_fwd(x=x_t):
                with torch.inference_mode():
                    for s in range(0, len(x), bs):
                        nodistill_model(x[s:s+bs])

            t_nd = _time_forward(_nodistill_fwd, args.device)
            row["student_nodistill_us_per_cut"] = f"{t_nd * 1e6 / n_cuts:.4f}"
        else:
            row["student_nodistill_us_per_cut"] = ""

        print(f"{_now()}   dc={row['deepcut_us_per_cut']}  tea={row['teacher_us_per_cut']}  "
              f"stu={row['student_us_per_cut']}  nd={row['student_nodistill_us_per_cut']} (us/cut)")

    # --- 写回 ---
    new_cols = ["deepcut_us_per_cut", "teacher_us_per_cut",
                "student_us_per_cut", "student_nodistill_us_per_cut"]
    fields = list(acc_rows[0].keys())
    for c in new_cols:
        if c not in fields:
            fields.append(c)

    with open(args.accuracy_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(acc_rows)

    print(f"{_now()} Updated {args.accuracy_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
