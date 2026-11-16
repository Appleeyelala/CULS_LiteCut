# STE + Adam 随机100电路交付

此目录发布已完成正式实验的整理结果，统一入口为本目录。未运行新实验、训练或修改算法；历史`analysis/benchmark905/`保持原样。

1. [汇总报告](STE_ADAM_100_SUMMARY.md)：真实Delay、完整Runtime、配对比较和结论边界。
2. [逐电路CSV](results/STE_ADAM_100_RESULTS.csv)：100行，A/B/Ours的真实指标、各自状态、Ours最终来源、最终输出CEC及候选CEC。`Ours_candidate_delay_ps`仅表示候选真实测量，未请求时留空；正式Ours指标为采用或回退后的最终结果。
3. [实际配置](config/STE_ADAM_100_CONFIG.json)与[抽样清单](config/sampled_100_manifest.json)：参数、硬件、工具哈希、源码Commit及100个原始ID/输入SHA256。
4. [实现说明](implementation/STE_ADAM_100_IMPLEMENTATION.md)：STE、Adam、GPU传播、冻结代理及真实验证边界。
5. [Mapping次数](audit/mapping_calls.csv)、[失败及回退清单](audit/failures.json)、[证据索引](audit/evidence_index.json)：逐电路记录及服务器原始文件路径、SHA256。索引的路径用于服务器核验，不是公开数据下载链接。

`original_noop`表示原图零编辑；`original_fallback`表示最终候选已验证但无真实严格改善而回退；`ste_adam_verified_improvement`表示采用真实改善的改写。`Ours_cec`为最终输出CEC；`Ours_candidate_cec=NOT_REQUESTED`表示未提交候选，和方法失败不同。全部失败/超时计数均为0。

输入AIG、模型权重、Liberty、二进制、完整日志及环境敏感配置不在本交付中。LiteCut局部预测、代理目标与真实全局Mapping Delay分开解释。
