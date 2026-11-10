from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from extract_cutlib_teacher.logger import get_logger

from .config import ExtractCutLibStudentConfig
from .data_preprocess import StudentDataset, StudentPreparedData, collate_student_batch
from .modeling import StudentDistillLosses, StudentModel


class StudentTrainer:
    LABEL_BUCKET_SPECS = [
        ("[0, 30]", 0.0, 30.0, True),
        ("(30, 100]", 30.0, 100.0, False),
        ("(100, 300]", 100.0, 300.0, False),
        ("(300, 600]", 300.0, 600.0, False),
        ("(600, 1000]", 600.0, 1000.0, False),
        ("< 0", float("-inf"), 0.0, False),
        ("> 1000", 1000.0, float("inf"), False),
    ]
    DENORM_LOG_CLAMP_MIN = -10.0
    DENORM_LOG_CLAMP_MAX = 10.0

    def __init__(
        self,
        cfg: ExtractCutLibStudentConfig,
        model: StudentModel,
        prepared: StudentPreparedData,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or get_logger()
        self.device = self._resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.prepared = prepared
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self._denorm_clip_logged = False
        self.result_dir = Path(cfg.result_dir) if cfg.result_dir else None
        self.metrics_dir = self.result_dir / "metrics" if self.result_dir else None
        self.predictions_dir = self.result_dir / "predictions" if self.result_dir else None
        self.config_dir = self.result_dir / "config" if self.result_dir else None

        self.y_mu = torch.from_numpy(prepared.y_mu).float().to(self.device)
        self.y_inv_sigma = torch.from_numpy(prepared.y_inv_sigma).float().to(self.device)
        self.model.set_input_stats(
            torch.from_numpy(prepared.x_mu).float().to(self.device),
            torch.from_numpy(prepared.x_inv_sigma).float().to(self.device),
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, cfg.scheduler_t_max),
            T_mult=max(1, cfg.scheduler_t_mult),
            eta_min=cfg.scheduler_eta_min,
        )

        self.train_loader, self.val_loader = self._build_dataloaders()
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self._ensure_output_dirs()
        self._write_run_config()

        if cfg.resume_checkpoint:
            self.load_checkpoint(cfg.resume_checkpoint)

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        num_samples = self.prepared.x_scaled.shape[0]
        perm = np.random.permutation(num_samples)
        train_size = max(1, int(num_samples * self.cfg.train_ratio))
        if num_samples > 1:
            train_size = min(train_size, num_samples - 1)
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
        if len(val_indices) == 0:
            val_indices = train_indices[:1]

        train_dataset = StudentDataset(self.prepared, train_indices)
        val_dataset = StudentDataset(self.prepared, val_indices)

        loader_num_workers = max(0, int(self.cfg.dataloader_num_workers))
        train_loader_kwargs: dict[str, Any] = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": loader_num_workers,
            "collate_fn": collate_student_batch,
        }
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": self.cfg.eval_batch_size,
            "shuffle": False,
            "num_workers": loader_num_workers,
            "collate_fn": collate_student_batch,
        }
        if loader_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 1
            train_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 1
            val_loader_kwargs["persistent_workers"] = True

        return DataLoader(train_dataset, **train_loader_kwargs), DataLoader(val_dataset, **val_loader_kwargs)

    def denormalize_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        y_log = y_scaled / self.y_inv_sigma + self.y_mu
        y_log_clamped = torch.clamp(y_log, min=self.DENORM_LOG_CLAMP_MIN, max=self.DENORM_LOG_CLAMP_MAX)
        if not self._denorm_clip_logged and not torch.equal(y_log, y_log_clamped):
            self.logger.warning(
                "Clamp y_log before expm1 for numerical stability: raw_range=[%.4f, %.4f], clamp_range=[%.1f, %.1f]",
                float(y_log.min().detach().cpu().item()),
                float(y_log.max().detach().cpu().item()),
                self.DENORM_LOG_CLAMP_MIN,
                self.DENORM_LOG_CLAMP_MAX,
            )
            self._denorm_clip_logged = True
        return torch.expm1(y_log_clamped)

    @staticmethod
    def _finite_pair_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.isfinite(y_true) & np.isfinite(y_pred)

    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = StudentTrainer._finite_pair_mask(y_true, y_pred)
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        if y_true.size == 0:
            return 0.0
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = StudentTrainer._finite_pair_mask(y_true, y_pred)
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        if y_true.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _compute_mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = StudentTrainer._finite_pair_mask(y_true, y_pred)
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        if y_true.size == 0:
            return 0.0
        denom = np.maximum(np.abs(y_true), 1e-8)
        return float(np.mean(np.abs(y_pred - y_true) / denom))

    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 100.0 * self._compute_mean_relative_error(y_true, y_pred)

    def _compute_epoch_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": self._compute_rmse(y_true, y_pred),
            "r2": self._compute_r2(y_true, y_pred),
            "mean_relative_error": self._compute_mean_relative_error(y_true, y_pred),
            "mape": self._compute_mape(y_true, y_pred),
            "num_samples": int(np.asarray(y_true).reshape(-1).size),
        }

    def _compute_bucket_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str | int]]:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        bucket_rows = []

        for label, lower, upper, include_lower in self.LABEL_BUCKET_SPECS:
            if include_lower:
                mask = (y_true >= lower) & (y_true <= upper)
            else:
                mask = (y_true > lower) & (y_true <= upper)

            count = int(mask.sum())
            if count == 0:
                bucket_rows.append(
                    {
                        "bucket": label,
                        "count": 0,
                        "rmse": 0.0,
                        "r2": 0.0,
                        "mean_relative_error": 0.0,
                        "mape": 0.0,
                    }
                )
                continue

            bucket_rows.append(
                {
                    "bucket": label,
                    "count": count,
                    "rmse": self._compute_rmse(y_true[mask], y_pred[mask]),
                    "r2": self._compute_r2(y_true[mask], y_pred[mask]),
                    "mean_relative_error": self._compute_mean_relative_error(y_true[mask], y_pred[mask]),
                    "mape": self._compute_mape(y_true[mask], y_pred[mask]),
                }
            )

        return bucket_rows

    def _format_bucket_metrics(self, title: str, bucket_rows: list[dict[str, float | str | int]]) -> str:
        header = f"{title}\n" + "-" * len(title)
        lines = [header, f"{'bucket':<14} {'count':>10} {'rmse':>14} {'r2':>12} {'mean_rel_err':>16} {'mape(%)':>12}"]
        for row in bucket_rows:
            lines.append(
                f"{str(row['bucket']):<14} "
                f"{int(row['count']):>10d} "
                f"{float(row['rmse']):>14.6f} "
                f"{float(row['r2']):>12.6f} "
                f"{float(row['mean_relative_error']):>16.6f} "
                f"{float(row['mape']):>12.4f}"
            )
        return "\n".join(lines)

    def _ensure_output_dirs(self) -> None:
        if self.metrics_dir is not None:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        if self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
        if self.config_dir is not None:
            self.config_dir.mkdir(parents=True, exist_ok=True)

    def _write_run_config(self) -> None:
        if self.config_dir is None:
            return
        with open(self.config_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2, ensure_ascii=False)

    def _require_teacher_targets(self, batch: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.cfg.with_distill:
            return None, None
        teacher_y_scaled = batch["teacher_y_scaled"]
        teacher_hidden = batch["teacher_hidden"]
        if teacher_y_scaled is None or teacher_hidden is None:
            raise ValueError(
                "StudentTrainer requires prepared.teacher_y_scaled and prepared.teacher_hidden. "
                "Run attach_teacher_pls_targets(...) and attach teacher predictions before training."
            )
        return teacher_y_scaled.to(self.device), teacher_hidden.to(self.device)

    def _compute_losses(
        self,
        batch: dict[str, Any],
    ) -> tuple[StudentDistillLosses, torch.Tensor]:
        x_scaled = batch["x_scaled"].to(self.device)
        y_scaled = batch["y_scaled"].to(self.device)
        teacher_y_scaled, teacher_hidden = self._require_teacher_targets(batch)

        output = self.model(x_scaled)
        losses = self.model.compute_distill_losses(
            output=output,
            y_true=y_scaled,
            teacher_prediction=teacher_y_scaled,
            teacher_hidden=teacher_hidden,
            lambda_hard=self.cfg.lambda_hard,
            lambda_soft=self.cfg.lambda_soft,
            lambda_align=self.cfg.lambda_align,
            rank_loss_weight=self.cfg.rank_loss_weight,
        )
        return losses, output.prediction

    def _write_predictions_csv(self, path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        data = np.column_stack([y_true.reshape(-1), y_pred.reshape(-1)])
        np.savetxt(path, data, delimiter=",", header="y_true,y_pred", comments="")

    def _write_final_metrics(self, metrics: dict[str, Any]) -> None:
        if self.metrics_dir is None:
            return
        payload = dict(metrics)
        payload["config"] = asdict(self.cfg)
        with open(self.metrics_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def train_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.model.train()
        y_raw = batch["y_raw"].to(self.device)
        y_scaled = batch["y_scaled"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        losses, y_pred = self._compute_losses(batch)
        if not torch.isfinite(losses.total):
            raise FloatingPointError(
                "Encountered non-finite student loss: "
                f"total={losses.total.item()} hard={losses.hard.item()} soft={losses.soft.item()} "
                f"align={losses.align.item()} rank={losses.rank.item()}"
            )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        y_pred_raw = self.denormalize_y(y_pred.detach())
        y_raw_np = y_raw.detach().cpu().numpy().reshape(-1)
        y_pred_raw_np = y_pred_raw.detach().cpu().numpy().reshape(-1)
        return {
            "loss": float(losses.total.detach().item()),
            "hard_loss": float(losses.hard.detach().item()),
            "soft_loss": float(losses.soft.detach().item()),
            "align_loss": float(losses.align.detach().item()),
            "rank_loss": float(losses.rank.detach().item()),
            "rmse": float(self._compute_rmse(y_raw_np, y_pred_raw_np)),
            "r2": float(self._compute_r2(y_raw_np, y_pred_raw_np)),
            "y_raw": y_raw_np,
            "y_pred_raw": y_pred_raw_np,
            "y_scaled": y_scaled.detach().cpu().numpy().reshape(-1),
            "y_pred_scaled": y_pred.detach().cpu().numpy().reshape(-1),
        }

    def eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.model.eval()
        y_raw = batch["y_raw"].to(self.device)
        y_scaled = batch["y_scaled"].to(self.device)
        with torch.no_grad():
            losses, y_pred = self._compute_losses(batch)
            y_pred_raw = self.denormalize_y(y_pred)

        y_raw_np = y_raw.detach().cpu().numpy().reshape(-1)
        y_pred_raw_np = y_pred_raw.detach().cpu().numpy().reshape(-1)
        return {
            "loss": float(losses.total.detach().item()),
            "hard_loss": float(losses.hard.detach().item()),
            "soft_loss": float(losses.soft.detach().item()),
            "align_loss": float(losses.align.detach().item()),
            "rank_loss": float(losses.rank.detach().item()),
            "rmse": float(self._compute_rmse(y_raw_np, y_pred_raw_np)),
            "r2": float(self._compute_r2(y_raw_np, y_pred_raw_np)),
            "y_raw": y_raw_np,
            "y_pred_raw": y_pred_raw_np,
            "y_scaled": y_scaled.detach().cpu().numpy().reshape(-1),
            "y_pred_scaled": y_pred.detach().cpu().numpy().reshape(-1),
        }

    def _run_epoch(self, epoch: int) -> dict[str, float]:
        train_losses = []
        train_hard_losses = []
        train_soft_losses = []
        train_align_losses = []
        train_rank_losses = []
        train_y_raw = []
        train_y_pred_raw = []

        for batch_idx, batch in enumerate(self.train_loader, start=0):
            metrics = self.train_batch(batch)
            self.global_step += 1
            train_losses.append(metrics["loss"])
            train_hard_losses.append(metrics["hard_loss"])
            train_soft_losses.append(metrics["soft_loss"])
            train_align_losses.append(metrics["align_loss"])
            train_rank_losses.append(metrics["rank_loss"])
            train_y_raw.append(metrics["y_raw"])
            train_y_pred_raw.append(metrics["y_pred_raw"])
            if batch_idx % max(1, self.cfg.log_every) == 0:
                self.logger.info(
                    "Train Step | epoch=%d step=%d loss=%.6f hard=%.6f soft=%.6f align=%.6f rank=%.6f rmse=%.6f r2=%.6f lr=%.6e",
                    epoch,
                    batch_idx,
                    metrics["loss"],
                    metrics["hard_loss"],
                    metrics["soft_loss"],
                    metrics["align_loss"],
                    metrics["rank_loss"],
                    metrics["rmse"],
                    metrics["r2"],
                    self.optimizer.param_groups[0]["lr"],
                )
            if batch_idx % max(1, self.cfg.log_every * 50) == 0:
                self.logger.info(
                    "Train Sample Predictions\n------------------------\ny_true=%s\ny_pred=%s",
                    metrics["y_raw"],
                    metrics["y_pred_raw"],
                )
                self.logger.info(
                    "Train Sample Predictions (Scaled)\n---------------------------------\ny_true_scaled=%s\ny_pred_scaled=%s",
                    metrics["y_scaled"],
                    metrics["y_pred_scaled"],
                )

        val_losses = []
        val_hard_losses = []
        val_soft_losses = []
        val_align_losses = []
        val_rank_losses = []
        val_y_raw = []
        val_y_pred_raw = []

        for batch_idx, batch in enumerate(self.val_loader, start=0):
            metrics = self.eval_batch(batch)
            val_losses.append(metrics["loss"])
            val_hard_losses.append(metrics["hard_loss"])
            val_soft_losses.append(metrics["soft_loss"])
            val_align_losses.append(metrics["align_loss"])
            val_rank_losses.append(metrics["rank_loss"])
            val_y_raw.append(metrics["y_raw"])
            val_y_pred_raw.append(metrics["y_pred_raw"])
            if batch_idx % max(1, self.cfg.log_every) == 0:
                self.logger.info(
                    "Val Step   | epoch=%d step=%d loss=%.6f hard=%.6f soft=%.6f align=%.6f rank=%.6f rmse=%.6f r2=%.6f",
                    epoch,
                    batch_idx,
                    metrics["loss"],
                    metrics["hard_loss"],
                    metrics["soft_loss"],
                    metrics["align_loss"],
                    metrics["rank_loss"],
                    metrics["rmse"],
                    metrics["r2"],
                )
            if batch_idx % max(1, self.cfg.log_every * 50) == 0:
                self.logger.info(
                    "Val Sample Predictions\n----------------------\ny_true=%s\ny_pred=%s",
                    metrics["y_raw"],
                    metrics["y_pred_raw"],
                )
                self.logger.info(
                    "Val Sample Predictions (Scaled)\n-------------------------------\ny_true_scaled=%s\ny_pred_scaled=%s",
                    metrics["y_scaled"],
                    metrics["y_pred_scaled"],
                )

        self.scheduler.step()

        train_y_true_all = np.concatenate(train_y_raw) if train_y_raw else np.empty((0,), dtype=np.float64)
        train_y_pred_all = np.concatenate(train_y_pred_raw) if train_y_pred_raw else np.empty((0,), dtype=np.float64)
        val_y_true_all = np.concatenate(val_y_raw) if val_y_raw else np.empty((0,), dtype=np.float64)
        val_y_pred_all = np.concatenate(val_y_pred_raw) if val_y_pred_raw else np.empty((0,), dtype=np.float64)

        train_epoch_metrics = self._compute_epoch_metrics(train_y_true_all, train_y_pred_all)
        val_epoch_metrics = self._compute_epoch_metrics(val_y_true_all, val_y_pred_all)
        train_bucket_metrics = self._compute_bucket_metrics(train_y_true_all, train_y_pred_all)
        val_bucket_metrics = self._compute_bucket_metrics(val_y_true_all, val_y_pred_all)

        return {
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "train_hard_loss": float(np.mean(train_hard_losses)) if train_hard_losses else 0.0,
            "train_soft_loss": float(np.mean(train_soft_losses)) if train_soft_losses else 0.0,
            "train_align_loss": float(np.mean(train_align_losses)) if train_align_losses else 0.0,
            "train_rank_loss": float(np.mean(train_rank_losses)) if train_rank_losses else 0.0,
            "train_rmse": float(train_epoch_metrics["rmse"]),
            "train_r2": float(train_epoch_metrics["r2"]),
            "train_mean_relative_error": float(train_epoch_metrics["mean_relative_error"]),
            "train_mape": float(train_epoch_metrics["mape"]),
            "train_num_samples": int(train_epoch_metrics["num_samples"]),
            "train_bucket_metrics": train_bucket_metrics,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "val_hard_loss": float(np.mean(val_hard_losses)) if val_hard_losses else 0.0,
            "val_soft_loss": float(np.mean(val_soft_losses)) if val_soft_losses else 0.0,
            "val_align_loss": float(np.mean(val_align_losses)) if val_align_losses else 0.0,
            "val_rank_loss": float(np.mean(val_rank_losses)) if val_rank_losses else 0.0,
            "val_rmse": float(val_epoch_metrics["rmse"]),
            "val_r2": float(val_epoch_metrics["r2"]),
            "val_mean_relative_error": float(val_epoch_metrics["mean_relative_error"]),
            "val_mape": float(val_epoch_metrics["mape"]),
            "val_num_samples": int(val_epoch_metrics["num_samples"]),
            "val_bucket_metrics": val_bucket_metrics,
            "train_y_true_all": train_y_true_all,
            "train_y_pred_all": train_y_pred_all,
            "val_y_true_all": val_y_true_all,
            "val_y_pred_all": val_y_pred_all,
        }

    def save_checkpoint(self, path: str, epoch: int) -> None:
        abc_inference_payload = self._build_abc_inference_payload()
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": asdict(self.cfg),
                #denorm y stats are needed for resuming training with correct denormalization
                "y_mu": self.y_mu.detach().cpu().numpy(),
                "y_inv_sigma": self.y_inv_sigma.detach().cpu().numpy(),
                #norm x stats are needed for resuming training with correct input normalization
                "x_mu": self.model.input_mu.detach().cpu().numpy(),
                "x_inv_sigma": self.model.input_inv_sigma.detach().cpu().numpy(),
                #ABC consumes only the inference subset, so save a flattened export-friendly payload too
                "abc_inference_payload": abc_inference_payload,
            },
            path,
        )

    @staticmethod
    def _tensor_to_abc_value(tensor: torch.Tensor) -> float | list[float] | list[list[float]]:
        tensor_cpu = tensor.detach().cpu().to(dtype=torch.float32)
        if tensor_cpu.ndim == 0:
            return float(tensor_cpu.item())
        return tensor_cpu.tolist()

    @staticmethod
    def _tensor_to_scalar(tensor: torch.Tensor) -> float:
        tensor_cpu = tensor.detach().cpu().to(dtype=torch.float32).reshape(-1)
        if tensor_cpu.numel() != 1:
            raise ValueError(f"Expected scalar tensor for ABC export, got shape {tuple(tensor.shape)}")
        return float(tensor_cpu.item())

    def _build_abc_inference_payload(self) -> dict[str, Any]:
        exported_state = {
            name: self._tensor_to_abc_value(param)
            for name, param in self.model.state_dict().items()
        }
        return {
            "format_version": 2,
            "model_family": "student_group_fusion_soft_moe" if self.model.with_moe else "student_group_fusion_single_head",
            "group_fusion_policy": "softmax(group_logits / gate_temperature)" if len(self.model.group_dims) > 1 else "single_group",
            "expert_routing_policy": "softmax(expert_logits / gate_temperature)" if self.model.with_moe else "single_head",
            "hidden_activation": "relu",
            "output_denorm": "expm1(clamp(y_scaled / y_inv_sigma + y_mu, clamp_min, clamp_max))",
            "input_dim": int(self.model.input_dim),
            "group_dims": [int(dim) for dim in self.model.group_dims],
            "shared_hidden_dim": int(self.model.shared_hidden_dim),
            "num_experts": int(self.model.num_experts),
            "gate_temperature": float(self.model.gate_temperature),
            "clamp_min": float(self.DENORM_LOG_CLAMP_MIN),
            "clamp_max": float(self.DENORM_LOG_CLAMP_MAX),
            "input_mu": self._tensor_to_abc_value(self.model.input_mu),
            "input_inv_sigma": self._tensor_to_abc_value(self.model.input_inv_sigma),
            "output_mu": self._tensor_to_scalar(self.y_mu),
            "output_inv_sigma": self._tensor_to_scalar(self.y_inv_sigma),
            "state_dict": exported_state,
        }

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.logger.info("Resumed student checkpoint from %s at epoch %d", path, self.start_epoch)

    def train(self) -> None:
        self.logger.info("Training student on device %s", self.device)
        self.logger.info(
            "Train samples=%d, Val samples=%d, Batch size=%d",
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            self.cfg.batch_size,
        )

        for epoch in range(self.start_epoch, self.cfg.num_epochs + 1):
            metrics = self._run_epoch(epoch)
            self.logger.info(
                "Epoch %d/%d Summary\n"
                "==================\n"
                "Train | loss=%.6f hard=%.6f soft=%.6f align=%.6f rank=%.6f rmse=%.6f r2=%.6f mean_rel_err=%.6f mape=%.4f%% samples=%d\n"
                "Val   | loss=%.6f hard=%.6f soft=%.6f align=%.6f rank=%.6f rmse=%.6f r2=%.6f mean_rel_err=%.6f mape=%.4f%% samples=%d",
                epoch,
                self.cfg.num_epochs,
                metrics["train_loss"],
                metrics["train_hard_loss"],
                metrics["train_soft_loss"],
                metrics["train_align_loss"],
                metrics["train_rank_loss"],
                metrics["train_rmse"],
                metrics["train_r2"],
                metrics["train_mean_relative_error"],
                metrics["train_mape"],
                metrics["train_num_samples"],
                metrics["val_loss"],
                metrics["val_hard_loss"],
                metrics["val_soft_loss"],
                metrics["val_align_loss"],
                metrics["val_rank_loss"],
                metrics["val_rmse"],
                metrics["val_r2"],
                metrics["val_mean_relative_error"],
                metrics["val_mape"],
                metrics["val_num_samples"],
            )
            self.logger.info(self._format_bucket_metrics("Train Bucket Metrics", metrics["train_bucket_metrics"]))
            self.logger.info(self._format_bucket_metrics("Val Bucket Metrics", metrics["val_bucket_metrics"]))

            latest_path = os.path.join(self.cfg.checkpoint_dir, "last.pt")
            if epoch % max(1, self.cfg.save_every) == 0:
                self.save_checkpoint(latest_path, epoch)

            if metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = metrics["val_loss"]
                best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
                self.save_checkpoint(best_path, epoch)
                self.logger.info("Saved new best student checkpoint to %s", best_path)
                if self.predictions_dir is not None:
                    self._write_predictions_csv(
                        self.predictions_dir / "best_val_predictions.csv",
                        metrics["val_y_true_all"],
                        metrics["val_y_pred_all"],
                    )

        self._write_final_metrics(
            {
                "best_val_loss": float(self.best_val_loss),
                "best_checkpoint": os.path.join(self.cfg.checkpoint_dir, "best.pt"),
                "last_checkpoint": os.path.join(self.cfg.checkpoint_dir, "last.pt"),
                "seed": int(self.cfg.random_seed),
                "experiment_name": self.cfg.experiment_name,
                "result_dir": self.cfg.result_dir,
                "with_context8": self.cfg.with_context8,
                "with_distill": self.cfg.with_distill,
                "with_rank_loss": self.cfg.with_rank_loss,
                "with_moe": self.cfg.with_moe,
                "input_dim": int(self.cfg.input_dim),
            }
        )
