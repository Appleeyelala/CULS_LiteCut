from __future__ import annotations

import os
import traceback

from .config import ExtractCutLibConfig
from .data_preprocess import load_teacher_data, load_teacher_data_cache, save_teacher_data_cache
from .fusion import TeacherTwoBranchModel
from .logger import get_logger
from .preprocess import seed_all
from .trainer import Trainer


def run(cfg: ExtractCutLibConfig) -> None:
    logger = get_logger()
    inside_logger = None

    try:
        seed_all(cfg.random_seed)
        cache_available = bool(cfg.prepared_cache_dir) and os.path.isdir(cfg.prepared_cache_dir)
        if cfg.reuse_prepared_cache and cache_available:
            prepared = load_teacher_data_cache(cfg.prepared_cache_dir, include_graph=cfg.include_graph, logger=logger)
        else:
            prepared = load_teacher_data(cfg, include_graph=cfg.include_graph, inside_logger=inside_logger)
            if cfg.save_prepared_cache and cfg.prepared_cache_dir:
                save_teacher_data_cache(prepared, cfg.prepared_cache_dir, logger=logger)
        model = TeacherTwoBranchModel(cfg)
        trainer = Trainer(cfg=cfg, model=model, prepared=prepared, logger=logger)
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno if exc.__traceback__ else -1
        file = traceback.extract_tb(exc.__traceback__)[-1].filename if exc.__traceback__ else "unknown"
        logger.error("An error occurred at line %d in %s: %s", line, file, exc)
