from __future__ import annotations

import os
import traceback
from dataclasses import asdict

import torch

from extract_cutlib_teacher.config import ExtractCutLibConfig
from extract_cutlib_teacher.fusion import TeacherTwoBranchModel
from extract_cutlib_teacher.logger import get_logger
from extract_cutlib_teacher.preprocess import seed_all

from .config import ExtractCutLibStudentConfig
from .data_preprocess import (
    adapt_prepared_data_for_config,
    attach_teacher_pls_targets,
    load_student_data,
    load_student_data_cache,
    save_student_data_cache,
)
from .modeling import StudentModel
from .trainer import StudentTrainer


def _build_teacher_model_from_checkpoint(checkpoint_path: str, device: torch.device, logger) -> TeacherTwoBranchModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher_cfg_raw = checkpoint.get("config")
    if teacher_cfg_raw is None:
        raise ValueError(f"Teacher checkpoint {checkpoint_path} does not contain `config`.")

    teacher_cfg = ExtractCutLibConfig(**teacher_cfg_raw)
    teacher_model = TeacherTwoBranchModel(teacher_cfg).to(device)
    teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher_model.eval()
    logger.info(
        "Loaded teacher checkpoint from %s with feature_fusion_hidden_dim=%d",
        checkpoint_path,
        teacher_cfg.feature_fusion_hidden_dim,
    )
    return teacher_model


def run(cfg: ExtractCutLibStudentConfig, teacher_checkpoint: str, teacher_batch_size: int = 1024) -> None:
    logger = get_logger()

    try:
        seed_all(cfg.random_seed)
        cache_available = bool(cfg.prepared_cache_dir) and os.path.isdir(cfg.prepared_cache_dir)
        if cfg.reuse_prepared_cache and cache_available:
            prepared = load_student_data_cache(cfg.prepared_cache_dir)
            prepared = adapt_prepared_data_for_config(prepared, cfg)
            logger.info("Loaded prepared student data cache from %s", cfg.prepared_cache_dir)
        else:
            prepared = load_student_data(cfg)
            logger.info(
                "Loaded student data: x_scaled=%s y_scaled=%s",
                tuple(prepared.x_scaled.shape),
                tuple(prepared.y_scaled.shape),
            )
            if cfg.save_prepared_cache and cfg.prepared_cache_dir:
                save_student_data_cache(prepared, cfg.prepared_cache_dir)
                logger.info("Saved prepared student data cache to %s", cfg.prepared_cache_dir)

        train_device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device)
        if cfg.device == "auto" and not torch.cuda.is_available():
            train_device = torch.device("cpu")

        if cfg.with_distill:
            if not teacher_checkpoint:
                raise ValueError("teacher_checkpoint is required when with_distill=true")
            teacher_model = _build_teacher_model_from_checkpoint(teacher_checkpoint, train_device, logger)
            prepared, pls = attach_teacher_pls_targets(
                prepared=prepared,
                teacher_model=teacher_model,
                latent_dim=cfg.teacher_pls_dim,
                batch_size=teacher_batch_size,
                device=train_device,
            )
            if cfg.save_prepared_cache and cfg.prepared_cache_dir:
                save_student_data_cache(prepared, cfg.prepared_cache_dir)
                logger.info("Updated student cache with teacher targets at %s", cfg.prepared_cache_dir)
            logger.info(
                "Prepared teacher targets: teacher_y_scaled=%s teacher_hidden=%s pls_components=%d",
                None if prepared.teacher_y_scaled is None else tuple(prepared.teacher_y_scaled.shape),
                None if prepared.teacher_hidden is None else tuple(prepared.teacher_hidden.shape),
                int(pls.n_components),
            )
        else:
            logger.info("with_distill=false, skip teacher checkpoint loading and target attachment")

        model = StudentModel(
            input_dim=cfg.input_dim,
            shared_hidden_dim=cfg.shared_hidden_dim,
            with_context8=cfg.with_context8,
            with_moe=cfg.with_moe,
            num_experts=cfg.num_experts,
            gate_temperature=cfg.gate_temperature,
            teacher_hidden_dim=prepared.teacher_hidden.shape[1] if prepared.teacher_hidden is not None else cfg.teacher_hidden_dim,
        )
        trainer = StudentTrainer(cfg=cfg, model=model, prepared=prepared, logger=logger)
        logger.info("Student config: %s", asdict(cfg))
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno if exc.__traceback__ else -1
        file = traceback.extract_tb(exc.__traceback__)[-1].filename if exc.__traceback__ else "unknown"
        logger.error("An error occurred at line %d in %s: %s", line, file, exc)
