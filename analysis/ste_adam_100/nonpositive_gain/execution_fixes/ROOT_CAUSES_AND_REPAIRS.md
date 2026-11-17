# 执行异常修复与后续研究主线

本文件是历史 full905 之后的新诊断，不修改历史 PASS/FAIL/超时或 QoR。
正式历史源码：`bd87a176024d681140305d2aa21ba20defdba4bd`。
历史运行：`/data/wushangge/CULS_LiteCut/runs/20261116T170324Z-nonpositive-full905-a94b687e`。
本次只重放少量既有组合、CEC 和已保存输入的原版 STE40；没有新 Mapping、A/B 或 full905。

## 16、24：共享设备的资源问题

历史每个 Slurm GPU allocation 启动四个 shard，各有一个 cold worker；两个 allocation 共八个 worker。
证据是历史 `input/job_worker.py` 中 `range(group,8,2)` 和 `subprocess.Popen`，以及 job65900/65901 的资源记录。
16、24 原错误位于模型 `.to(cuda)` 附近，异步栈本身不能独立证明具体失败分配。

保存输入隔离复现中，两者使用未修改的模型/优化器完成 40 步：
16 峰值 allocated 251823616 bytes；24 为 608949248 bytes。
这是 PyTorch allocated 峰值，不含驱动上下文，不是 nvidia-smi 总显存。
显存竞争是强证据支持的解释；历史没有故障瞬间进程级显存快照，不能确定当时哪一个其他 PID 占满了卡。

资源修复：`global_timing/gpu_slot.py` 与 `scripts/run_nonpositive_ours.py` 在 shard parent 启动 cold case 前取得同 run、Slurm allocation 和 GPU visibility set 的互斥锁。
排队等待单独记录，不消耗 case 的 1800s timeout，也不混入 cold case runtime；总作业 wall time仍应完整报告。
这是防止显存互相挤占的保守调度修复，不声称因此获得 GPU 加速。
后续主要阶段 GPU 批处理/流水化应在显存预算内单独评估。

## 827：关联矩阵远大于 logits

候选特征张量只有 63610880 bytes，logits 单张量约 15 MiB。
但 `TimingSurrogate` 为每个 Region 的每个 port 枚举能被该 port 到达的 Decision，
显式生成 `port_incidence[19429,124240]`，含 362201325 个非零项。
float64 COO 的两条 int64 索引和一条数值数组共 8692831800 bytes，约 8.10 GiB。
候选扩张、宽边界及共享可达性共同扩大了此表示；不是“40 步保存了 40 份 logits”。
历史 backward 已报告本进程 allocated 21.77 GiB，再申请 2.70 GiB 失败。
Slurm65912 在独占且初始空闲的 RTX4090 上复现同样 OOM，排除了“必须有其他任务抢显存才会失败”的解释。
Profiler捕获 `aten::t/transpose/clone` 的约8.10GiB转置复制、5.40GiB索引临时量和2.70GiB数值临时量；
这些是嵌套事件，不能直接将各event的bytes求和。峰值allocated为23377776128 bytes，失败后释放了部分临时量。
这直接定位到固定稀疏关联矩阵的通用反向实现，而不是Adam状态。

修复 `global_timing/fixed_incidence.py`：前向仍用同一 `torch.sparse.mm(A,X)`；
反向按非零项分块计算相同的 `Aᵀ grad`，不生成完整转置COO和排序临时量。
只对输入X求导，A本来就是固定buffer；仍为一阶反向，MFFC native GPU传播不变。
该修复不减少候选、不裁剪梯度、不改变精度，也不修改STE/Adam公式。
CPU/GPU稠密参考梯度、gradcheck、原MFFC模型和STE fixture已通过；保存输入验证Slurm65958已完成：16、24、827均40步PASS；峰值allocated分别0.235、0.477、9.699GiB。
16、24相对旧源码隔离运行的40步hard选择序列、各步目标值和最终proxy_best完全一致；梯度范数差异≤1.1e-17。
827旧版独占运行在第一次backward失败，新版成功完成40步，但尚未对其新组合做Replay/CEC/Mapping，因此不增加历史full905的PASS数或QoR样本。
这证明本次分块伴随解决了已复现的显存故障，不能当作完整端到端加速结果。
原始大关联表和CPU构造成本仍存在；本次未优化其表示生成。详见MEMORY_REGRESSION.json。

## 服务器与实际设备

`eda`有4张RTX4090，每张24564MiB；集群另有eda2/eda3各4张A40。
历史65900/65901均分配在eda，各占1GPU。旧任务只记录型号，未记录物理索引/UUID，不能事后把历史CUDA:0当成物理0。
本次完整旧源码隔离复现65912及修复验证65958由各自start事件记录SLURM_JOB_GPUS、CUDA_VISIBLE_DEVICES、UUID、PCI bus和初始进程。
65912使用eda物理0，PCI 00000000:67:00.0，其UUID保存在原始start事件；进程内仍显示CUDA:0。
早期65910使用eda物理1，PCI 00000000:68:00.0，但进程内同样是CUDA:0，说明重映射必须区分。
设备信息以MEMORY_REGRESSION.json中的原始start事件及来源为准。

## Replay：联合组合的边界生命周期错误

原始图的 removable/conflict 集合未覆盖顺序安装后引用计数下降导致的额外删除。
例：40 的 root12799 静态 removable 只有自身，实际安装删除了3956；后续 root12864 仍以3956为 cut leaf，因而 COMPROMISED。
480 对应64434→57769→64435；597为5038→4991→6259；772为21401→10039→21402/22936。
证据在各 case 的 raw GPU enumeration 与最终 `rewrite/raw/traces/choice.jsonl`。

修复 `CPUSolver::Rewrite`：只对显式 nonpositive joint replay，在整个安装事务中临时增加所选 roots 和 cut leaves 的引用；全部选择处理完成后释放，再清理无引用节点。
循环依赖拒绝、COMPROMISED 检查和 CEC 均保留。历史 greedy positive-gain 路径不进入这段代码。
此修复使后续候选边界存活，不通过忽略错误或减少选择项掩盖失败。

Slurm65911 新鲜回归：旧 binary 对四例安装数为15/16、15/16、15/16、14/16；
新 binary 对四例普通和 prepared replay 全为16/16且CEC PASS。
另外成功案例3、124的两条路径也均16/16且CEC PASS，共12个修复后PASS。
详见 `REPLAY_REGRESSION.json`；未测新组合 Mapping QoR。

## 六个超时：不等于 Adam 步数过多

22、59、108、541、552 已有 verified catalogue 和 feasibility，尚未写出 LiteCut 完成结果/trajectory；
合法候选分别约401万、401万、817万、238万、105万。
578有924万 raw候选，但 verified/feasibility 未完成，应归入更早的验证/候选处理区间。
外层预算1800s覆盖整个流程；不能把“最后一个阶段未完成”当成该阶段独占1800s。

`global_timing/litecut.py::score_catalogue` 逐 Decision/choice 在 CPU 进行 Boolean/library检查、图物化、NetworkX特征提取，累计所有 Python rows/features，最后再整体堆叠、CPU Student batch128及完整JSON序列化。
`projects/culs/tools/gate3e_run.py::LiteCutPredictor` 明确将模型加载到CPU。
所以候选增多会同时扩大串行工作、Python对象内存和I/O；GPU枚举加速不代表整个流程GPU化。
896个已完成阶段记录中，LiteCut占已记录阶段累计时间72.81%，STE40占2.58%；这不含9个OOM/超时的缺失阶段及未列进程开销。
现有记录不足以给六个超时逐项分摊特征、推理、JSON与I/O时间，不能编造精确瓶颈百分比。

新增Slurm65959只诊断552保存候选中均匀取出的64个Decision（393个含no-op选项），不枚举新候选或运行搜索。
带cProfile：LiteCut整体3.588s；extract_all累计1.928s，其中get_timing_stats约1.028s；
NetworkX dag_longest_path_length被调用786次、累计1.024s；Student.forward四个batch共0.005695s。
这是有工具开销的小样本，嵌套累计时间有重叠，不是六个完整case的耗时分解，也不能按候选数线性外推成实测时间。
它支持优先处理逐候选Python/NetworkX图计算，而不是只迁移Student推理。详见LITECUT_BOUNDED_PROFILE.json。

下一步应先持久化子阶段进度/计时，再针对主耗时做有界分批的候选物化、复用、紧凑存储与GPU特征/推理。
只把Student移到GPU不能解决之前的逐候选CPU图处理；只延长timeout只能增加等待预算；恢复positive-gain硬筛选会重新丢掉已恢复的合法候选。
这里尚未实施性能改造或重跑六个完整超时电路。

## 研究主线：使梯度选出的联合改写更有效

目标是 CEC PASS，真实 Mapping Delay 整体优于B；Area改善是次要目标；部署只做一次原图 Mapping，GPU加速必须改善主要阶段和完整runtime。
不把后续工作预设为重新训练预测器，也不预设Adam失效。

已有891个完整安装/CEC/Mapping的非空组合均降低了优化目标，但相对各自原图只有427改善、12平、452退化。
其中193例预测max-PO delay未下降（1e-12容差），仅平滑logsumexp目标下降，却仍被best选择；其中100例真实退化。
其余698例预测max-delay下降，仍有352真实退化，说明仅修正平滑目标选择也不能保证解决全部问题。
STE forward 已评价可行hard组合，最终best来自访问过的hard组合；不是只评价soft结果后盲目投影。

建议依次做可证伪的小规模诊断：
1. **优化目标与最终选择对齐**：检查最大PO/critical端点、次关键路径和smooth objective的差别；先对已有轨迹归因，再讨论目标或best判据，不先增加步数/重启。
2. **联合组合有效性**：检查候选梯度排序、冲突投影、16-edit预算与组合交互；原图锚点正确不代表组合增量正确。用结构、边界和执行证据区分目标问题、搜索问题和时序表示问题；最终Mapping仅作为研究评估。
3. **GPU覆盖主要开销**：优先LiteCut前处理/候选表示与聚合，再讨论已经只占少部分的Adam内核。加速验证必须保持候选域、优化规则与精度口径一致。

41个历史area超限的被选组合，预测面积比全部≤1.003012，真实面积比1.050081～1.516283；面积惩罚未感知这些违规。
提高现有面积惩罚系数不能惩罚预测中不存在的超限；需要区分局部节点数量、重写后结构与mapped cell面积。
本次不改变面积约束、优化目标、no-op偏置、学习率、步数或edit预算。
