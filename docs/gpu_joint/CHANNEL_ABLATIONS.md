# LiteCut标量、显式交互与修复的补充控制

用户研究说明§4要求贡献分离。正式九方法面板覆盖内层更新、完整梯度包与GPU，但不能单独隔离上述三项。补充三个可关闭控制，默认算法不变，2257998正式快照保持原样：

- `adaptive_no_score_batch_auto`：实际19维提取和Student推理仍执行、计时、保存；只将代理四维feature中的局部分数相对delta置零。结构特征仍保留，因此不是去掉全部局部信息，也不能用此控制证明省掉推理的速度收益。
- `adaptive_no_pairs_batch_auto`：校准前将显式邻接交互系数置零并以持久零梯度hook保持零；局部二次目标的反馈pair描述量也禁用。DAG/LSE仍会产生路径和端点耦合，不能声称删除全部跨MFFC联系。
- `adaptive_no_repair_batch_auto`：失败仍完整记录、不送部分电路测量；不把失败转成修复队列。后续同预算槽可用于别的提案，因此会改变实际搜索序列，不能把它当相同向量的加速。

`pipeline.gradient_search`只增加可选model-transform和repair回调，默认None沿用原逻辑；原Student、surrogate公式和原生安装检查不变。`FrozenCriticalityObjective`仅在显式禁用pair时抑制该反馈通道，默认True的计算保持原样。

研究协议为`config/gpu_joint_ablation_protocol.json`：按原冻结面板顺序取每个AND规模区间的第一个条目，固定65/96/625/587、三个原种子、默认+三个控制，共48方法点。选取时主面板已在运行、补充结果尚不存在，因此不称作完全预结果设计；未查看候选数/失败率/时序/时间来挑选补充输入。它不重选主面板、不重新定阈值，不建立四设计之外的泛化。

运行入口先做数值/回调回归，再在同一Slurm分配上重放冻结2257998默认和当前默认c1355四请求，要求向量、状态、真实delay/area/全部PO完全一致；另验证三个真实开关。验证标签不喂给正式研究。之后在新快照执行48点并完整审计，所有真实数据仍要求完整ACCEPTED、CEC与统一Liberty。

计算在正式64983及后处理64984之后进行。65004已COMPLETED/0:0、30:01，37/37回归、同分配真实默认兼容及四配置smoke通过，48/48正式补充点和115成功观察完成原始审计。默认对no-score/no-pairs分别2胜7平3负，delay比1.002822/1.002929，未支持总体信息通道收益；对no-repair为9胜3平0负、0.952208，支持该子集修复的质量作用。

每配置12个设计/seed点中3个没有Decision、都为no-op；默认36个直接梯度请求全部失败，最终9repair/3no-op，repair父请求为5梯度/4warmup。因此不能把修复收益全写成直接梯度作用。保留原12配对分母；无修复通常更快但质量更差。结果在 `/data/wushangge/CULS_LiteCut/runs/20261113T073402Z-channel-repair-ablation-c9634975/output/ablation_analysis/`，详细表格及限制见[FINAL_RESULTS.md](FINAL_RESULTS.md)。

另检查12个默认完整请求轨迹（选择/状态/成功全部PO、delay、area）均与正式2257998相同，见[来源核对](../../provenance/gpu_joint_ablation_source_audit.json)。这不是跨作业时间对照：主面板在eda/RTX4090，补充在eda2/A40，时间比只在各自同分配内解释。
