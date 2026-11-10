from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as torch_mp
from torch.nn.parallel import parallel_apply, replicate
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

from .config import ExtractCutLibConfig
from .data_preprocess import TeacherPreparedData
from .fusion import TeacherTwoBranchModel
from .graph_branch import GraphTensorSample
from .logger import get_logger


class TeacherDataset(Dataset):
    def __init__(self, prepared: TeacherPreparedData, indices: np.ndarray, include_graph: bool) -> None:
        self.x_scaled = prepared.x_scaled
        self.x_wl_hash = prepared.x_wl_hash
        self.y_scaled = prepared.y_scaled
        self.y_raw = prepared.y_raw
        self.x_graph = prepared.x_graph
        self.indices = indices.tolist()
        self.include_graph = include_graph

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_idx = self.indices[idx]
        item = {
            "x_scaled": torch.from_numpy(self.x_scaled[sample_idx]).float(),
            "wl_hash": torch.tensor(self.x_wl_hash[sample_idx], dtype=torch.long),
            "y_scaled": torch.from_numpy(self.y_scaled[sample_idx]).float(),
            "y_raw": torch.from_numpy(self.y_raw[sample_idx]).float(),
        }
        if self.include_graph and self.x_graph is not None:
            item["graph"] = self.x_graph[sample_idx]
        else:
            item["graph"] = None
        return item


def collate_teacher_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x_scaled = torch.stack([item["x_scaled"] for item in batch], dim=0)
    wl_hash = torch.stack([item["wl_hash"] for item in batch], dim=0)
    y_scaled = torch.stack([item["y_scaled"] for item in batch], dim=0)
    y_raw = torch.stack([item["y_raw"] for item in batch], dim=0)
    graph_items = [item["graph"] for item in batch]
    graph_batch = None if all(graph is None for graph in graph_items) else tuple(graph_items)
    return {
        "x_scaled": x_scaled,
        "wl_hash": wl_hash,
        "y_scaled": y_scaled,
        "y_raw": y_raw,
        "graph_batch": graph_batch,
    }


class Trainer:
    LABEL_BUCKET_SPECS = [
        ("[0, 30]", 0.0, 30.0, True),
        ("(30, 100]", 30.0, 100.0, False),
        ("(100, 300]", 100.0, 300.0, False),
        ("(300, 600]", 300.0, 600.0, False),
        ("(600, 1000]", 600.0, 1000.0, False),
        ("< 0", float("-inf"), 0.0, False),
        ("> 1000", 1000.0, float("inf"), False),
    ]

    def __init__(
        self,
        cfg: ExtractCutLibConfig,
        model: TeacherTwoBranchModel,
        prepared: TeacherPreparedData,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or get_logger()
        self.device = self._resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.device_ids = self._resolve_parallel_device_ids(cfg.device)
        self.prepared = prepared
        self.y_mu = torch.from_numpy(prepared.y_mu).float().to(self.device)
        self.y_inv_sigma = torch.from_numpy(prepared.y_inv_sigma).float().to(self.device)
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self._parallel_debug_logged = True
        self._device_debug_logged = True
        self._configured_sharing_strategy = False

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
        # self.loss_fn = nn.HuberLoss()
        self.loss_fn = nn.MSELoss()

        self._configure_multiprocessing_sharing()
        self.train_loader, self.val_loader = self._build_dataloaders()
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        if len(self.device_ids) > 1:
            self.logger.info(
                "Enable non-DDP multi-GPU training: primary_device=%s, parallel_devices=%s, visible_gpu_count=%d",
                self.device,
                self.device_ids,
                torch.cuda.device_count(),
            )
        else:
            self.logger.info("Use single-device training on %s", self.device)

        if cfg.resume_checkpoint:
            self.load_checkpoint(cfg.resume_checkpoint)

    def _log_cuda_debug_info(self, x_scaled: torch.Tensor, wl_hash: torch.Tensor, y_scaled: torch.Tensor) -> None:
        if self._device_debug_logged:
            return

        model_param = next(self.model.parameters(), None)
        model_device = model_param.device if model_param is not None else "unknown"
        self.logger.info(
            "Device Debug | model_device=%s x_scaled_device=%s wl_hash_device=%s y_scaled_device=%s",
            model_device,
            x_scaled.device,
            wl_hash.device,
            y_scaled.device,
        )

        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            self.logger.info(
                "CUDA Debug | torch.cuda.is_available=%s visible_gpu_count=%d current_device=%d",
                torch.cuda.is_available(),
                visible_count,
                torch.cuda.current_device(),
            )
            for dev_id in range(visible_count):
                self.logger.info(
                    "CUDA Debug | gpu=%d name=%s allocated=%.2fMB reserved=%.2fMB",
                    dev_id,
                    torch.cuda.get_device_name(dev_id),
                    torch.cuda.memory_allocated(dev_id) / (1024 ** 2),
                    torch.cuda.memory_reserved(dev_id) / (1024 ** 2),
                )

        self._device_debug_logged = True

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _resolve_parallel_device_ids(self, device_str: str) -> list[int]:
        if not torch.cuda.is_available():
            return []

        if device_str == "auto":
            return list(range(torch.cuda.device_count()))

        resolved = torch.device(device_str)
        if resolved.type != "cuda":
            return []

        if resolved.index is not None:
            return [resolved.index]

        return list(range(torch.cuda.device_count()))

    def _configure_multiprocessing_sharing(self) -> None:
        if self._configured_sharing_strategy:
            return

        try:
            torch_mp.set_sharing_strategy("file_system")
            self.logger.info(
                "Set torch multiprocessing sharing strategy to file_system to reduce open-file pressure."
            )
        except RuntimeError as exc:
            self.logger.warning("Failed to set torch multiprocessing sharing strategy: %s", exc)

        self._configured_sharing_strategy = True

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

        train_dataset = TeacherDataset(self.prepared, train_indices, include_graph=self.cfg.include_graph)
        val_dataset = TeacherDataset(self.prepared, val_indices, include_graph=self.cfg.include_graph)

        loader_num_workers = max(0, int(self.cfg.dataloader_num_workers))
        if self.cfg.include_graph and self.prepared.x_graph is not None and loader_num_workers > 0:
            self.logger.warning(
                "Graph batches contain many small tensors; forcing dataloader_num_workers from %d to 0 to avoid "
                "Too many open files during multiprocessing transfer.",
                loader_num_workers,
            )
            loader_num_workers = 0

        train_loader_kwargs: dict[str, Any] = {
            "batch_size": self.cfg.batch_size,
            "shuffle": True,
            "num_workers": loader_num_workers,
            "collate_fn": collate_teacher_batch,
        }
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": self.cfg.eval_batch_size,
            "shuffle": False,
            "num_workers": loader_num_workers,
            "collate_fn": collate_teacher_batch,
        }
        if loader_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 1
            train_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 1
            val_loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            **val_loader_kwargs,
        )
        return train_loader, val_loader

    def denormalize_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        y_log = y_scaled / self.y_inv_sigma + self.y_mu
        return torch.expm1(y_log)

    @staticmethod
    def _split_graph_batch(
        graph_batch: Sequence[GraphTensorSample] | None,
        chunk_sizes: Sequence[int],
    ) -> list[tuple[GraphTensorSample, ...] | None]:
        if graph_batch is None:
            return [None] * len(chunk_sizes)

        graph_splits: list[tuple[GraphTensorSample, ...] | None] = []
        start = 0
        total_graphs = len(graph_batch)
        for size in chunk_sizes:
            end = start + int(size)
            if end > total_graphs:
                raise ValueError(
                    f"Graph batch split overflow: trying to slice [{start}:{end}] from {total_graphs} graphs."
                )
            graph_splits.append(tuple(graph_batch[start:end]))
            start = end

        if start != total_graphs:
            raise ValueError(f"Graph batch split mismatch: consumed {start} graphs, expected {total_graphs}.")
        return graph_splits

    def _forward_model(
        self,
        x_scaled: torch.Tensor,
        wl_hash: torch.Tensor,
        graph_batch: Sequence[GraphTensorSample] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.device_ids) <= 1 or x_scaled.shape[0] < 2:
            return self.model(x_scaled, wl_hash, graph_batch)

        num_chunks = min(len(self.device_ids), x_scaled.shape[0])
        if num_chunks <= 1:
            return self.model(x_scaled, wl_hash, graph_batch)

        x_chunks = torch.chunk(x_scaled, num_chunks, dim=0)
        wl_chunks = torch.chunk(wl_hash, num_chunks, dim=0)
        chunk_sizes = [int(chunk.shape[0]) for chunk in x_chunks]
        graph_chunks = self._split_graph_batch(graph_batch, chunk_sizes)

        active_device_ids = self.device_ids[:num_chunks]
        if not self._parallel_debug_logged:
            graph_chunk_sizes = [0 if chunk is None else len(chunk) for chunk in graph_chunks]
            self.logger.info(
                "Multi-GPU batch split: devices=%s, tensor_chunk_sizes=%s, graph_chunk_sizes=%s",
                active_device_ids,
                chunk_sizes,
                graph_chunk_sizes,
            )
            self._parallel_debug_logged = True

        replicas = replicate(self.model, active_device_ids)
        inputs = []
        for idx, device_id in enumerate(active_device_ids):
            device = torch.device(f"cuda:{device_id}")
            inputs.append(
                (
                    x_chunks[idx].to(device, non_blocking=True),
                    wl_chunks[idx].to(device, non_blocking=True),
                    graph_chunks[idx],
                )
            )

        if not self._device_debug_logged:
            for idx, device_id in enumerate(active_device_ids):
                self.logger.info(
                    "Parallel Input Debug | gpu=%d x_chunk_device=%s wl_chunk_device=%s graph_chunk_size=%d",
                    device_id,
                    inputs[idx][0].device,
                    inputs[idx][1].device,
                    0 if inputs[idx][2] is None else len(inputs[idx][2]),
                )

        outputs = parallel_apply(replicas, inputs, devices=active_device_ids)
        y_pred = torch.cat([output[0].to(self.device) for output in outputs], dim=0)
        z_feat = torch.cat([output[1].to(self.device) for output in outputs], dim=0)
        z_graph = torch.cat([output[2].to(self.device) for output in outputs], dim=0)
        return y_pred, z_feat, z_graph

    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
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
        if y_true.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _compute_mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
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

    @staticmethod
    def _compute_rank_loss(y_true_scaled: torch.Tensor, y_pred_scaled: torch.Tensor) -> torch.Tensor:
        y_true_vec = y_true_scaled.reshape(-1)
        y_pred_vec = y_pred_scaled.reshape(-1)
        if y_true_vec.numel() < 2:
            return y_pred_vec.new_zeros(())

        true_diff = y_true_vec.unsqueeze(1) - y_true_vec.unsqueeze(0)
        sign = torch.sign(true_diff)
        pair_mask = torch.triu(sign != 0, diagonal=1)
        if not torch.any(pair_mask):
            return y_pred_vec.new_zeros(())

        pred_diff = y_pred_vec.unsqueeze(1) - y_pred_vec.unsqueeze(0)
        pair_sign = sign[pair_mask]
        pair_pred_diff = pred_diff[pair_mask]
        return F.softplus(-pair_sign * pair_pred_diff).mean()

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

    def train_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        x_scaled = batch["x_scaled"].to(self.device)
        wl_hash = batch["wl_hash"].to(self.device)
        y_scaled = batch["y_scaled"].to(self.device)
        y_raw = batch["y_raw"].to(self.device)
        graph_batch = batch["graph_batch"] if self.cfg.include_graph else None
        # self._log_cuda_debug_info(x_scaled, wl_hash, y_scaled)

        self.optimizer.zero_grad(set_to_none=True)
        y_pred, _, _ = self._forward_model(x_scaled, wl_hash, graph_batch)
        mse_loss = self.loss_fn(y_pred, y_scaled)
        rank_loss = self._compute_rank_loss(y_scaled, y_pred)
        loss = mse_loss + self.cfg.rank_loss_weight * rank_loss
        loss.backward()
        self.optimizer.step()

        y_pred_raw = self.denormalize_y(y_pred.detach())
        y_raw_np = y_raw.detach().cpu().numpy().reshape(-1)
        y_pred_raw_np = y_pred_raw.detach().cpu().numpy().reshape(-1)
        rmse = self._compute_rmse(y_raw_np, y_pred_raw_np)
        r2 = self._compute_r2(y_raw_np, y_pred_raw_np)

        return {
            "loss": float(loss.detach().item()),
            "mse_loss": float(mse_loss.detach().item()),
            "rank_loss": float(rank_loss.detach().item()),
            "rmse": float(rmse),
            "r2": float(r2),
            "y_raw": y_raw_np,
            "y_pred_raw": y_pred_raw_np,
            "y_scaled": y_scaled.detach().cpu().numpy().reshape(-1),
            "y_pred_scaled": y_pred.detach().cpu().numpy().reshape(-1),
        }

    def _eval_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.eval()
        x_scaled = batch["x_scaled"].to(self.device)
        wl_hash = batch["wl_hash"].to(self.device)
        y_scaled = batch["y_scaled"].to(self.device)
        y_raw = batch["y_raw"].to(self.device)
        graph_batch = batch["graph_batch"] if self.cfg.include_graph else None

        with torch.no_grad():
            y_pred, _, _ = self._forward_model(x_scaled, wl_hash, graph_batch)
            mse_loss = self.loss_fn(y_pred, y_scaled)
            rank_loss = self._compute_rank_loss(y_scaled, y_pred)
            loss = mse_loss + self.cfg.rank_loss_weight * rank_loss
            y_pred_raw = self.denormalize_y(y_pred)
            y_raw_np = y_raw.detach().cpu().numpy().reshape(-1)
            y_pred_raw_np = y_pred_raw.detach().cpu().numpy().reshape(-1)
            rmse = self._compute_rmse(y_raw_np, y_pred_raw_np)
            r2 = self._compute_r2(y_raw_np, y_pred_raw_np)

        return {
            "loss": float(loss.detach().item()),
            "mse_loss": float(mse_loss.detach().item()),
            "rank_loss": float(rank_loss.detach().item()),
            "rmse": float(rmse),
            "r2": float(r2),
            "y_raw": y_raw_np,
            "y_pred_raw": y_pred_raw_np,
            "y_scaled": y_scaled.detach().cpu().numpy().reshape(-1),
            "y_pred_scaled": y_pred.detach().cpu().numpy().reshape(-1),
        }

    def _run_epoch(self, epoch: int) -> dict[str, float]:
        train_losses = []
        train_mse_losses = []
        train_rank_losses = []
        train_y_raw = []
        train_y_pred_raw = []
        for batch_idx, batch in enumerate(self.train_loader, start=0):
            metrics = self.train_batch(batch)
            self.global_step += 1
            train_losses.append(metrics["loss"])
            train_mse_losses.append(metrics["mse_loss"])
            train_rank_losses.append(metrics["rank_loss"])
            train_y_raw.append(metrics["y_raw"])
            train_y_pred_raw.append(metrics["y_pred_raw"])
            if batch_idx % max(1, self.cfg.log_every) == 0:
                self.logger.info(
                    "Train Step | epoch=%d step=%d loss=%.6f mse=%.6f rank=%.6f rmse=%.6f r2=%.6f lr=%.6e",
                    epoch,
                    batch_idx,
                    metrics["loss"],
                    metrics["mse_loss"],
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
        val_mse_losses = []
        val_rank_losses = []
        val_y_raw = []
        val_y_pred_raw = []
        for batch_idx, batch in enumerate(self.val_loader, start=0):
            metrics = self._eval_batch(batch)
            val_losses.append(metrics["loss"])
            val_mse_losses.append(metrics["mse_loss"])
            val_rank_losses.append(metrics["rank_loss"])
            val_y_raw.append(metrics["y_raw"])
            val_y_pred_raw.append(metrics["y_pred_raw"])
            if batch_idx % max(1, self.cfg.log_every) == 0:
                self.logger.info(
                    "Val Step   | epoch=%d step=%d loss=%.6f mse=%.6f rank=%.6f rmse=%.6f r2=%.6f",
                    epoch,
                    batch_idx,
                    metrics["loss"],
                    metrics["mse_loss"],
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
            "train_mse_loss": float(np.mean(train_mse_losses)) if train_mse_losses else 0.0,
            "train_rank_loss": float(np.mean(train_rank_losses)) if train_rank_losses else 0.0,
            "train_rmse": float(train_epoch_metrics["rmse"]),
            "train_r2": float(train_epoch_metrics["r2"]),
            "train_mean_relative_error": float(train_epoch_metrics["mean_relative_error"]),
            "train_mape": float(train_epoch_metrics["mape"]),
            "train_num_samples": int(train_epoch_metrics["num_samples"]),
            "train_bucket_metrics": train_bucket_metrics,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "val_mse_loss": float(np.mean(val_mse_losses)) if val_mse_losses else 0.0,
            "val_rank_loss": float(np.mean(val_rank_losses)) if val_rank_losses else 0.0,
            "val_rmse": float(val_epoch_metrics["rmse"]),
            "val_r2": float(val_epoch_metrics["r2"]),
            "val_mean_relative_error": float(val_epoch_metrics["mean_relative_error"]),
            "val_mape": float(val_epoch_metrics["mape"]),
            "val_num_samples": int(val_epoch_metrics["num_samples"]),
            "val_bucket_metrics": val_bucket_metrics,
        }

    def save_checkpoint(self, path: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": asdict(self.cfg),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.logger.info("Resumed checkpoint from %s at epoch %d", path, self.start_epoch)

    def train(self) -> None:
        self.logger.info("Training on device %s", self.device)
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
                "Train | loss=%.6f mse=%.6f rank=%.6f rmse=%.6f r2=%.6f mean_rel_err=%.6f mape=%.4f%% samples=%d\n"
                "Val   | loss=%.6f mse=%.6f rank=%.6f rmse=%.6f r2=%.6f mean_rel_err=%.6f mape=%.4f%% samples=%d",
                epoch,
                self.cfg.num_epochs,
                metrics["train_loss"],
                metrics["train_mse_loss"],
                metrics["train_rank_loss"],
                metrics["train_rmse"],
                metrics["train_r2"],
                metrics["train_mean_relative_error"],
                metrics["train_mape"],
                metrics["train_num_samples"],
                metrics["val_loss"],
                metrics["val_mse_loss"],
                metrics["val_rank_loss"],
                metrics["val_rmse"],
                metrics["val_r2"],
                metrics["val_mean_relative_error"],
                metrics["val_mape"],
                metrics["val_num_samples"],
            )
            self.logger.info(self._format_bucket_metrics("Train Bucket Metrics", metrics["train_bucket_metrics"]))
            self.logger.info(self._format_bucket_metrics("Val Bucket Metrics", metrics["val_bucket_metrics"]))

            latest_path = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
            if epoch % max(1, self.cfg.save_every) == 0:
                self.save_checkpoint(latest_path, epoch)

            if metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = metrics["val_loss"]
                best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
                self.save_checkpoint(best_path, epoch)
                self.logger.info("Saved new best checkpoint to %s", best_path)
