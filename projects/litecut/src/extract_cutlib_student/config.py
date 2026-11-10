from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractCutLibStudentConfig:
    dataset_path: str
    with_context8: bool = True
    with_distill: bool = True
    with_rank_loss: bool = False
    with_moe: bool = True
    experiment_name: str = ""
    result_dir: str = ""
    prepared_cache_dir: str = ""
    reuse_prepared_cache: bool = False
    save_prepared_cache: bool = False
    random_seed: int = 42
    device: str = "auto"
    label_outlier_filter: str = "none"
    label_outlier_threshold: float = 3.5
    cpu_worker: int = 16
    train_ratio: float = 0.9
    batch_size: int = 64
    eval_batch_size: int = 128
    input_dim: int | None = None
    shared_hidden_dim: int = 18
    teacher_pls_dim: int = 18
    num_experts: int = 4
    gate_temperature: float = 1.0
    teacher_hidden_dim: int | None = None
    lambda_hard: float = 1.0
    lambda_soft: float = 1.0
    lambda_align: float = 1.0
    rank_loss_weight: float = 0.0
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_t_max: int = 20
    scheduler_t_mult: int = 1
    scheduler_eta_min: float = 1e-6
    dataloader_num_workers: int = 0
    log_every: int = 50
    save_every: int = 1
    checkpoint_dir: str = "./checkpoints/extract_cutlib_student"
    resume_checkpoint: str = ""
    log_path: str = ""

    def __post_init__(self) -> None:
        resolved_input_dim = 19 if self.with_context8 else 11
        if self.input_dim is None:
            object.__setattr__(self, "input_dim", resolved_input_dim)
        elif self.input_dim != resolved_input_dim:
            raise ValueError(
                f"input_dim={self.input_dim} does not match with_context8={self.with_context8}; "
                f"expected {resolved_input_dim}"
            )

        if not self.with_distill:
            object.__setattr__(self, "lambda_soft", 0.0)
            object.__setattr__(self, "lambda_align", 0.0)

        if not self.with_rank_loss:
            object.__setattr__(self, "rank_loss_weight", 0.0)

        if not self.with_moe and self.num_experts < 1:
            object.__setattr__(self, "num_experts", 1)
