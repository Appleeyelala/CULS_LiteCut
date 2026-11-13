# 实际 AIG 增益与时序候选安装策略

观察依据：DMA 的四个已保存梯度提案均因 root2901/cut5/subgraph4 的 `saved=2, added=4` 而 REVERTED，修复删除该 Decision 后得到 2301.566650 ps。正 AIG 节点增益是旧执行器的安装启发式，与功能合法性、mapped area 和 mapped critical delay 不是同一个条件。

新增可关闭的 `SearchConfig.allow_nonpositive_gain`（默认 false），原生命令对应 `CULS_REPLAY_ALLOW_NONPOSITIVE_GAIN=1`。它只允许**已审计显式 replay** 在通过其他检查后接受 saved≤added 的动作；不扩大原始正增益 GPU candidate mask，不改变 cuts/library、指定顺序、leaf/引用环检查、精确请求校验或 CEC。NO_OP、COMPROMISED、CYCLE_REJECT 等仍失败。实际 mapped area≤1.05 no-op 及真实 delay 排名保持原规则。

`replay_policy.json` 记录原生模式，上层要求该文件与请求一致，不能给不支持此模式的旧工具贴新策略标签。非正增益接受项保留 saved/added，原因明确为 `explicit_replay_nonpositive_gain_allowed`。默认模式与历史结果不回写；这项变更不能被描述为仅加速原算法。

`probe_install_gain.py` 对保存的四个失败向量进行固定配置实验：正增益控制含 fresh no-op、fresh 修复点和四个原向量；新策略回放相同四个向量。只有完整 ACCEPTED 才逐点执行 CEC 和统一 Liberty mapping，并报告面积可行性、所有十个计划配置和实际调用数。此为**新的真实测量**，不计作原八请求优化过程的免费反馈，也不作运行时间加速证据。

代码已实现，尚待编译、回归与该 probe 的实际证据。允许负增益本身不证明时序改善，也可能扩大结构或失去修复带来的好点；是否保留该策略由真实结果决定。
