from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .config import ExtractCutLibStudentConfig


@dataclass(frozen=True)
class ResultDirLayout:
    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    predictions: Path
    config: Path


def build_result_dir_layout(result_root: str | Path, experiment_name: str) -> ResultDirLayout:
    root = Path(result_root) / experiment_name
    return ResultDirLayout(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        metrics=root / "metrics",
        predictions=root / "predictions",
        config=root / "config",
    )


def parse_runtime_args() -> tuple[ExtractCutLibStudentConfig, str, int]:
    parser = argparse.ArgumentParser(description="Student cutLib distillation runner")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument(
        "--with_context8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the first 8 context features in the student input",
    )
    parser.add_argument(
        "--with_distill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable teacher soft/alignment distillation losses",
    )
    parser.add_argument(
        "--with_rank_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include pairwise rank loss in the student objective",
    )
    parser.add_argument(
        "--with_moe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use soft MoE routing instead of a single regression head",
    )
    parser.add_argument("--experiment_name", type=str, default="", help="Optional experiment name for grouped outputs")
    parser.add_argument(
        "--result_root",
        type=str,
        default="",
        help="Optional root directory for grouped experiment outputs",
    )
    parser.add_argument(
        "--prepared_cache_dir",
        type=str,
        default="",
        help="Optional directory used to save/load prepared student data cache",
    )
    parser.add_argument(
        "--reuse_prepared_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to load student prepared data from cache when available",
    )
    parser.add_argument(
        "--save_prepared_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save freshly prepared student data into cache",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Training device: auto/cpu/cuda/cuda:0")
    parser.add_argument(
        "--label_outlier_filter",
        type=str,
        choices=["none", "zscore", "mad", "maximum"],
        default="none",
        help="Optional label outlier filter applied after concatenating all datasets",
    )
    parser.add_argument(
        "--label_outlier_threshold",
        type=float,
        default=3.5,
        help="Threshold used by the label outlier filter",
    )
    parser.add_argument("--cpu_worker", type=int, default=16, help="Worker count for JSON loading")

    parser.add_argument("--input_dim", type=int, default=None, help="Student input dim; auto-derived from with_context8")
    parser.add_argument("--shared_hidden_dim", type=int, default=18, help="Group encoder / fusion hidden dim")
    parser.add_argument("--teacher_pls_dim", type=int, default=18, help="PLS compressed teacher feature dim")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of expert regressors after group fusion",
    )
    parser.add_argument(
        "--gate_temperature",
        type=float,
        default=1.0,
        help="Soft fusion temperature for combining group encoders",
    )
    parser.add_argument(
        "--teacher_hidden_dim",
        type=int,
        default=None,
        help="Optional teacher hidden dim before adapter; defaults to student hidden dim",
    )

    parser.add_argument("--lambda_hard", type=float, default=1.0, help="Weight for supervised hard loss")
    parser.add_argument("--lambda_soft", type=float, default=1.0, help="Weight for teacher output distillation loss")
    parser.add_argument("--lambda_align", type=float, default=1.0, help="Weight for teacher feature align loss")
    parser.add_argument("--rank_loss_weight", type=float, default=0.0, help="Weight for pairwise rank loss")

    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--scheduler_t_max", type=int, default=20, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--scheduler_t_mult", type=int, default=1, help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Cosine scheduler eta_min")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="PyTorch DataLoader worker count")
    parser.add_argument("--log_every", type=int, default=50, help="Log training metrics every N batches")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/extract_cutlib_student",
        help="Directory to save student checkpoints",
    )
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Student checkpoint path to resume from")
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default="",
        help="Trained teacher checkpoint path used for distillation",
    )
    parser.add_argument(
        "--teacher_batch_size",
        type=int,
        default=1024,
        help="Batch size used when collecting teacher predictions/features",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='./log/cutLib_extraction.log',
        help="Path to store log",
    )

    args = parser.parse_args()
    prepared_cache_dir = args.prepared_cache_dir
    checkpoint_dir = args.checkpoint_dir
    log_path = args.log_path
    result_dir = ""

    if args.result_root and args.experiment_name:
        layout = build_result_dir_layout(args.result_root, args.experiment_name)
        result_dir = str(layout.root)
        checkpoint_dir = str(layout.checkpoints)
        log_path = str(layout.logs / "train.log")

    cfg = ExtractCutLibStudentConfig(
        dataset_path=args.dataset_path,
        with_context8=args.with_context8,
        with_distill=args.with_distill,
        with_rank_loss=args.with_rank_loss,
        with_moe=args.with_moe,
        experiment_name=args.experiment_name,
        result_dir=result_dir,
        prepared_cache_dir=prepared_cache_dir,
        reuse_prepared_cache=args.reuse_prepared_cache,
        save_prepared_cache=args.save_prepared_cache,
        random_seed=args.random_seed,
        device=args.device,
        label_outlier_filter=args.label_outlier_filter,
        label_outlier_threshold=args.label_outlier_threshold,
        cpu_worker=args.cpu_worker,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        input_dim=args.input_dim,
        shared_hidden_dim=args.shared_hidden_dim,
        teacher_pls_dim=args.teacher_pls_dim,
        num_experts=args.num_experts,
        gate_temperature=args.gate_temperature,
        teacher_hidden_dim=args.teacher_hidden_dim,
        lambda_hard=args.lambda_hard,
        lambda_soft=args.lambda_soft,
        lambda_align=args.lambda_align,
        rank_loss_weight=args.rank_loss_weight,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_eta_min=args.scheduler_eta_min,
        dataloader_num_workers=args.dataloader_num_workers,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
        log_path=log_path,
    )
    return cfg, args.teacher_checkpoint, args.teacher_batch_size


def parse_args() -> ExtractCutLibStudentConfig:
    cfg, _, _ = parse_runtime_args()
    return cfg
