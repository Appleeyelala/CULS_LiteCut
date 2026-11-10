from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractCutLibConfig:
    dataset_path: str
    prepared_cache_dir: str = ""
    reuse_prepared_cache: bool = False
    save_prepared_cache: bool = False
    random_seed: int = 42
    device: str = "auto"
    label_outlier_filter: str = "none"
    label_outlier_threshold: float = 3.5
    feature_group_hidden_dim: int = 32
    feature_fusion_hidden_dim: int = 128
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    readout_pos_gate_hidden_dim: int = 16
    readout_content_hidden_dim: int = 32
    regression_hidden_dim1: int = 128
    regression_hidden_dim2: int = 64
    regression_dropout: float = 0.1
    include_graph: bool = True
    p_inv_init: str = "zero"
    p_inv_init_std: float = 0.02
    cpu_worker: int = 16
    cpu_graph_worker: int = 32
    graph_tensorize_chunk_size: int = 128
    train_ratio: float = 0.9
    batch_size: int = 64
    eval_batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    rank_loss_weight: float = 0.0
    scheduler_t_max: int = 20
    scheduler_t_mult: int = 1
    scheduler_eta_min: float = 1e-6
    dataloader_num_workers: int = 0
    log_every: int = 50
    save_every: int = 1
    checkpoint_dir: str = "./checkpoints/extract_cutlib_teacher"
    resume_checkpoint: str = ""
