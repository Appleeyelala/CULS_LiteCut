from __future__ import annotations

import argparse

from .config import ExtractCutLibConfig


def parse_args() -> ExtractCutLibConfig:
    parser = argparse.ArgumentParser(description="Teacher two-branch cutLib runner")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to teacher dataset root")
    parser.add_argument(
        "--prepared_cache_dir",
        type=str,
        default="",
        help="Optional directory used to save/load preprocessed teacher data cache",
    )
    parser.add_argument(
        "--reuse_prepared_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to load teacher data from prepared cache when available",
    )
    parser.add_argument(
        "--save_prepared_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save freshly prepared teacher data into cache",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto/cpu/cuda/cuda:0",
    )
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
    parser.add_argument(
        "--include_graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load and run the graph branch",
    )
    parser.add_argument(
        "--cpu_worker",
        type=int,
        default=16,
        help="Worker count for JSON loading ",
    )
    parser.add_argument(
        "--cpu_graph_worker",
        type=int,
        default=32,
        help="Worker count for graph tensorization",
    )
    parser.add_argument(
        "--graph_tensorize_chunk_size",
        type=int,
        default=128,
        help="Number of graph payloads processed by each graph tensorization future",
    )

    # Feature Branch
    parser.add_argument(
        "--feature_group_hidden_dim",
        type=int,
        default=32,
        help="Hidden dim for group encoders, WL encoder, and group attention scorer",
    )
    parser.add_argument(
        "--feature_fusion_hidden_dim",
        type=int,
        default=128,
        help="Output dim of feature branch fusion and branch fusion width",
    )

    # Graph Branch
    parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="Hidden dim for graph message passing")
    parser.add_argument("--gnn_num_layers", type=int, default=3, help="Number of relation-aware GNN layers")
    parser.add_argument(
        "--readout_pos_gate_hidden_dim",
        type=int,
        default=16,
        help="Hidden dim of positional gate MLP",
    )
    parser.add_argument(
        "--readout_content_hidden_dim",
        type=int,
        default=32,
        help="Hidden dim of content scorer MLP",
    )
    parser.add_argument(
        "--p_inv_init",
        type=str,
        choices=["zero", "gaussian"],
        default="zero",
        help="Initialization mode for inverter sign gate",
    )
    parser.add_argument(
        "--p_inv_init_std",
        type=float,
        default=0.02,
        help="Std used when p_inv_init=gaussian",
    )

    # Regression Head
    parser.add_argument("--regression_hidden_dim1", type=int, default=128, help="First hidden dim of regression head")
    parser.add_argument("--regression_hidden_dim2", type=int, default=64, help="Second hidden dim of regression head")
    parser.add_argument("--regression_dropout", type=float, default=0.1, help="Dropout in regression head")

    # Training
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument(
        "--rank_loss_weight",
        type=float,
        default=0.0,
        help="Weight lambda in total loss: mse_loss + lambda * rank_loss",
    )
    parser.add_argument(
        "--scheduler_t_max",
        type=int,
        default=20,
        help="CosineAnnealingWarmRestarts T_0",
    )
    parser.add_argument(
        "--scheduler_t_mult",
        type=int,
        default=1,
        help="CosineAnnealingWarmRestarts T_mult",
    )
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Cosine scheduler eta_min")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="PyTorch DataLoader worker count")
    parser.add_argument("--log_every", type=int, default=50, help="Log training metrics every N batches")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/extract_cutlib_teacher",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Checkpoint path to resume from",
    )

    args = parser.parse_args()
    return ExtractCutLibConfig(
        dataset_path=args.dataset_path,
        prepared_cache_dir=args.prepared_cache_dir,
        reuse_prepared_cache=args.reuse_prepared_cache,
        save_prepared_cache=args.save_prepared_cache,
        random_seed=args.random_seed,
        device=args.device,
        label_outlier_filter=args.label_outlier_filter,
        label_outlier_threshold=args.label_outlier_threshold,
        feature_group_hidden_dim=args.feature_group_hidden_dim,
        feature_fusion_hidden_dim=args.feature_fusion_hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        readout_pos_gate_hidden_dim=args.readout_pos_gate_hidden_dim,
        readout_content_hidden_dim=args.readout_content_hidden_dim,
        regression_hidden_dim1=args.regression_hidden_dim1,
        regression_hidden_dim2=args.regression_hidden_dim2,
        regression_dropout=args.regression_dropout,
        include_graph=args.include_graph,
        p_inv_init=args.p_inv_init,
        p_inv_init_std=args.p_inv_init_std,
        cpu_worker=args.cpu_worker,
        cpu_graph_worker=args.cpu_graph_worker,
        graph_tensorize_chunk_size=args.graph_tensorize_chunk_size,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_loss_weight=args.rank_loss_weight,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_eta_min=args.scheduler_eta_min,
        dataloader_num_workers=args.dataloader_num_workers,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
