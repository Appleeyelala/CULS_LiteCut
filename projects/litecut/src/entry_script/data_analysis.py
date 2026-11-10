from __future__ import annotations

from extract_cutlib_teacher.data_preprocess import load_teacher_data
from extract_cutlib_teacher.cli import parse_args
from extract_cutlib_teacher.logger import setup_logger

import torch


def main() -> None:
    cfg = parse_args()
    prepared = load_teacher_data(cfg, include_graph=cfg.include_graph, inside_logger=None)

    # save prepared data to a file
    torch.save(prepared, "/data2/zhoulingfeng/saved_prepared_data.pt")
    y_raw = prepared.y_raw
    x_raw = prepared.x_raw
    cut_root_fanout = x_raw[:, 2]

    torch.save((cut_root_fanout, y_raw), "/data2/zhoulingfeng/saved_cut_root_fanout_and_y_raw.pt")


if __name__ == "__main__":
    setup_logger("./log/data_analysis.log", name="dataAnalysisLogger")
    main()
