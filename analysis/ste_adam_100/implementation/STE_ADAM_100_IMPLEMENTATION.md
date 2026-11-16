# MFFC Softmax + STE + Adam（随机 100 电路）

遵循 [实施 Plan](../../../docs/PLAN_MFFC_STE_ADAM_随机100电路评估.md)。旧 `mffc_experiment.py`、`mffc_search.py` 和原 905 证据不变。

## 实际算法

- `global_timing/mffc_ste_experiment.py:SteExperiment` 独立准备原图真实 Mapping、strict MFFC、候选、原有 Boolean 检查及 CandidateFeasibility、冻结 Student 分数和原有全局代理。
- `choice=0` 始终代表原始 AIG，无 replay-noop。代理传入自己测量的原图 PO/Area，零编辑精确锚定该值。
- `global_timing/mffc_ste_search.py:ste_adam_search` 创建 float64 可训练 logits；只有 logits 进入 Adam。代理12个系数保持原有先验并冻结，显式 interaction 系数仍为零；DAG 的上下游传播仍联合计算。LiteCut 不接收梯度。
- `p=softmax(masked_logits/T)`，GPU 计算各 Decision 最佳非零候选相对 no-op 的 logit margin，只传最多128条正 margin 编辑；现有可行性投影从原图出发保留最多16条编辑。hard 张量直接在设备上构造。
- `v=hard-p.detach()+p`；loss 为原有 `scalar_objective`：所有 PO 的平滑归一化延迟加 `20*relu(area/original_area-1.05)^2`。每步 backward、范数裁剪5、Adam(lr=.12) 更新 logits。
- 固定40步、1 restart、优化种子20261117、温度1→.3、no-op初始偏置+2、噪声标准差.01。保存每步稀疏组合和代理值，选已访问组合的最小代理目标（包含原图），不只取末步。未追加最终 Adam 更新后的第41次投影。
- 内循环无 ABC/CEC/Mapping 或校准。非空最优组合只调用一次 `execute_one`；严格全量 ACCEPTED 后 CEC、Mapping。失败不重试、不修复。真实面积≤原图1.05倍且真实 Delay 严格降低才采用，否则返回自己付费测量的原图。

## 执行与数据

`config/ste100_protocol.json` 冻结参数和旧正式工具哈希。`scripts/run_ste100.py` 从原905清单用 `random.Random(20261116).sample(range(905),100)` 升序固定样本及输入哈希；不读取历史测量。正式只运行 A/B/Ours，独立冷进程、同节点资源、轮换顺序；A/B原配方与Evaluator保持一致。

小图沿用原固定CPU门槛，非小图使用现有CUDA Region DAG前反传播。Runtime包括自身原图Mapping、准备、推理、40步、最终验证、失败处理及进程启动/退出，不包括排队、一次性构建和汇总。

`scripts/summarize_ste100.py` 只从真实方法终态生成100行CSV、完整300终态/Mapping次数JSON及简明报告。失败Delay空白，不填预测值。原始候选失败、回退及无编辑均保留。

## 指定检查（不属于正式100统计）

源码 `21c9d1b`；Slurm `65533`，COMPLETED，31秒，4CPU/64GiB/1GPU。目录：
`/data/wushangge/CULS_LiteCut/runs/20261116T070231Z-ste-adam-smoke-778bdf85/`

仅运行 `tests/test_mffc_ste.py:SteAdamCheck` 一项合成检查：合法hard、Softmax梯度、40步真实logits更新、冻结代理、原图锚点通过；原生CUDA backward=40。随后仅三个指定Ours smoke：

|输入|Mapping数|最终候选CEC|原图/候选真实Delay ps|最终来源|
|---|---:|---|---:|---|
|OpenABC/c1355_orig.aig|2|PASS|360.111877 / 406.762024|原图回退|
|ACE/DMA_comb.aig|2|PASS|2463.366460 / 2413.334960|改写|
|forgeEDA/mig_7series_v2_1_ecc_merge_enc.aig|1|无候选；原图CEC PASS|见原始summary|原图|

c1355与DMA分别完整接受16/13个编辑。检查说明实现可运行，不说明代理普遍准确或GPU端到端优于基线。

## 本次发布的源码与记录核对

正式运行源码Commit为`9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`。当前发布前工作区的STE搜索、Ours流程、runner、汇总器及协议文件与该冻结Commit一致。算法没有改动。

- Softmax＋STE＋Adam多步优化已实现；`torch.optim.Adam([logits], lr=0.12)`仅更新float64 Candidate选择logits。每步`loss.backward()`经冻结的可微全局时序代理和Softmax回传；不对真实ABC求导。
- Student冻结、代理12参数冻结；显式interaction系数为0，Region DAG上下游仍联合传播。MFFC级CUDA ready-queue前向与反向传播保留。83个有候选电路实际使用CUDA，每个40步；17个无候选跳过模型，没有触发CPU fallback。
- 温度1.0线性降至0.3；1 restart；梯度范数裁剪5；no-op偏置+2、噪声标准差0.01；最多16个非零Decision，每步最多128条稀疏候选传CPU合法性投影。
- 最终选择访问过的代理目标最佳合法组合（包含原图），不追加第41次投影。内循环没有ABC、CEC、真实Mapping、反馈校准或修复重试。
- 实际Mapping为67个电路各2次、33个各1次，总167次。一次原图锚点测量加至多一次非空最终候选验证；成本计入Ours。逐电路证据见[Mapping表](../audit/mapping_calls.csv)。
- 80/83出现可测logits更新，67/83改变hard组合并降低代理目标；三个无可测更新的ID及真实负结果均保留。

MFFC Region是strict maximal standard MFFC分区；Decision固定`(root,cut_slot)`；Rewrite Region是cut-bounded removable cone；Candidate是replacement subgraph。LiteCut是冻结Student对19维局部特征的局部abc_delay预测，不是全局真实Delay。原AIG、GPU、replay、mapper和mapped-netlist节点ID属于不同命名空间。

历史905冻结Commit`ecc85e6ac82571caf9f1a6004098dbe976536256`采用方向导数候选块、多次真实反馈及在线校准，不能与本次STE100混合解释。当前证据未证明代理普遍准确、梯度优于其他搜索或端到端GPU加速。

冻结源码入口：

- [STE搜索](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_ste_search.py)
- [Ours流程](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_ste_experiment.py)
- [MFFC GPU代理](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_model.py)
- [CUDA Region DAG](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/framework/mffc_gpu/region_lse.cu)
