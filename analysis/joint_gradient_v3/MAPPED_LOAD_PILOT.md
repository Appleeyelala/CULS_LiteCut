# 原图 Mapping 负载修正小面板

12 个开发电路 × 8 个冻结非空组合 × 3 个原 v3 种子。逐电路比较原 v3 全局残差与增加负载修正后的代理；旧 checkpoint、LiteCut、候选域及并列规则相同。每次选择均查已有真实 ABC 标签。没有新增 Mapping 或训练。

修正使用原图已付费 Mapping、ASAP7 Liberty 和替换候选 AIG。用 512 位固定随机逻辑仿真签名筛选原图 AIG cut 边界与映射驱动的唯一同相匹配，排除近常量、歧义和无可用时序弧者。所有 PO 按原始声明顺序核验签名；内部匹配仍是启发式。对每个候选计算边界引脚使用数的变化，以原映射平均接收引脚电容估算负载变化，在 Liberty 中间输入转换时间近似延迟对电容的斜率。修正沿原映射的估计最慢前驱链到达 PO，同一组合的修正通过选择权重相加，全部 no-op 的修正为零。该方式保留原映射覆盖及传播链，不显式模拟重新选门。

36 个选择完全一致，平均 Regret 都为 1.6293%，Top-1 都为 10/36。288 个组合预测中 21 个的全局最大值变化超过 0.001 ps，最大变化 7.3594 ps。353 个有连接数变化的边界节点中有 143 个通过对齐且可获得负载斜率（按电路计数）。在至少一个输出上有非零修正的候选有 56 个。ID75 的组合 6、8 在 PO90 的修正均为零，ID179 组合 7 在 PO1035 的修正也为零，三个种子仍将它判为限制端点。该版本不接入搜索。

Slurm 66069 在读取转义端口名时提前退出，修复后作业 66070 完成，训练/测试规则不变。52 秒，进程峰值约 672 MiB，1 CPU、4 GiB 作业内存限额，无 GPU。时间仅代表本次离线面板。[运行清单](/data/wushangge/CULS_LiteCut/runs/20261118T055931Z-v3-mapped-load-interface-32939534/run.json)、[选优表](/data/wushangge/CULS_LiteCut/runs/20261118T055931Z-v3-mapped-load-interface-32939534/output/selected.csv)、[端点](/data/wushangge/CULS_LiteCut/runs/20261118T055931Z-v3-mapped-load-interface-32939534/output/endpoints.csv)、[复核](/data/wushangge/CULS_LiteCut/runs/20261118T055931Z-v3-mapped-load-interface-32939534/output/verification.json)。
