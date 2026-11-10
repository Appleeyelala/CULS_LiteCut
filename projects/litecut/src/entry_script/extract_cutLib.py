import os

from extract_cutlib_teacher.cli import parse_args
from extract_cutlib_teacher.logger import setup_logger

def main() -> None:
    logger = setup_logger()

    logger.info(f"Cur PID: {os.getpid()}")
    logger.info(f"Cur PPID: {os.getppid()}")
    cfg = parse_args()
    # from extract_cutlib.runner import run
    from extract_cutlib_teacher.runner_teacher import run

    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    run(cfg)


if __name__ == "__main__":
    main()
