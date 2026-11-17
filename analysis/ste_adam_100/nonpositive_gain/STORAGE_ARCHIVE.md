# 本次新run缓存的无损存储归档

存储说明：磁盘余量接近150 GiB且剩余输入仍约257万个AND节点时，在既有Slurm作业 65901 的步骤 2 中，以1 CPU、1 GiB、0 GPU归档了 16 个**已终态电路的LiteCut缓存文件**，释放 64.528 GiB。每个gzip解压后的原始字节SHA256均与压缩前一致；算法源码、原始结果和计时记录保持原值。并行存储I/O可能影响尚未结束电路的实测Runtime；未扣除、修正或估算这种影响，不作严格同负载性能比较。见 `STORAGE_ARCHIVE_INDEX.json` 和 `STORAGE_ARCHIVE.md`。

归档开始：`2026-11-16T23:47:55.583924+00:00`；结束：`2026-11-16T23:58:17.895307+00:00`。仅处理已经保存终态记录且状态为PASS或AREA_LIMIT_EXCEEDED的本次新run；所有原实验、AIG、模型、库、优化轨迹、Mapping/CEC结果及计时文件均保留原处。

索引逐项记录原始路径、gzip路径、原始SHA256、实测恢复SHA256、gzip SHA256、原始大小/权限/mtime及终态记录SHA256。程序先完成解压SHA256核验并持久化索引，随后才用gzip归档替代明文缓存。完整gzip保留在/data，未上传GitHub。

需要重新读取某个缓存时，使用索引中的 `archive_path` 执行 `gzip -dk <archive_path>`，恢复同目录原JSON并保留gzip；已有同名文件时不要覆盖。用 `sha256sum <source_path>` 对照 `original_sha256`。恢复需要该文件原始大小对应的可用空间；权限与mtime可按索引恢复。

这属于数据保存操作，没有执行额外的候选搜索、Replay、CEC、Mapping或训练。
