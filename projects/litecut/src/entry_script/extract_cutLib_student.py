import os

from extract_cutlib_student.cli import parse_runtime_args
from extract_cutlib_teacher.logger import setup_logger


def main() -> None:
    cfg, teacher_checkpoint, teacher_batch_size = parse_runtime_args()
    logger = setup_logger(cfg.log_path)

    logger.info(f"Cur PID: {os.getpid()}")
    logger.info(f"Cur PPID: {os.getppid()}")
    from extract_cutlib_student.runner_student import run

    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    run(cfg, teacher_checkpoint=teacher_checkpoint, teacher_batch_size=teacher_batch_size)


if __name__ == "__main__":
    main()
