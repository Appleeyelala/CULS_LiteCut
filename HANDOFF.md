# 最新研究：v3 排序与输出端上下文单项实验（2026-11-17）

分支 `work/joint-gradient-v3-rank-improve`，基于 `work/joint-gradient-v3-ranking`。
先读 [本轮结果](analysis/joint_gradient_v3/ranking_improvement/README.md)。
诊断确认旧global头的下游context在终端PO为0；已只替换其八维上下文为
原图根节点时序敏感性/可达性加权的候选特征，网络、损失与搜索器保持不变。
三种子开发Pairwise从54.91%降到51.82%，Top1从27.78%升到33.33%，
平均Regret从1.629%升到1.908%，因此不推进接入搜索器。原12验证电路已作为
开发集；本轮没有独立测试、新Mapping或100/905运行，旧记录不变。
19项检查通过，Slurm66052/66053/66054已完成；下一步建议先研究目标对齐，
不把这次表示缺口修补当作真实排序改善。

# 历史交接补充：joint-gradient v3 受控验证完成（2026-11-17）

先读 [v3入口](analysis/joint_gradient_v3/README.md) 和
[学习结果](analysis/joint_gradient_v3/PILOT_RESULTS.md)。分支 `work/joint-gradient-v3`。

- 已实现无需内部pin对齐的相位端口代理、GPU前反向、候选端口学习及全局残差；LiteCut冻结。
- 旧39/180结构不一致已定位为literal相位组合问题并修复；180组合＋12原图全部通过。
- 12训练＋12验证、216次新增离线Mapping全部成功，另复用24个历史组合；三种子学习未过排序/方向门槛。
- 旧12电路96个开发搜索选择已完成，无新增Mapping。结合代理＋联合梯度/B几何比1.016664且1例面积超限。
- 新发现：19/24 serialized no-op的真实Delay不同于原图，最大+32.29%；结构哈希、工具和命令语义相同。未唯一归因于编号策略，不重写历史结论。
- 未启动后续独立24、100或905；未替换生产入口。当前无待续作业。下一步建议先统一序列化/Mapping基准，尚未实施。

---

# 历史交接补充：执行修复已验证（2026-11-17）

先读 [执行修复与根因](analysis/ste_adam_100/nonpositive_gain/execution_fixes/README.md) 和 [非正gain全量结果](analysis/ste_adam_100/nonpositive_gain/README.md)。下方2026-11-16段落是原STE100历史交接，不能替代后续实验现状。

- 原STE100：正式源码`9a199bf0...`；之后已完成放宽gain后的固定100门槛和原905全量Ours，正式源码`bd87a176...`，历史结果851 PASS/41 AREA_LIMIT_EXCEEDED/7 FAIL/6 TIMEOUT。
- 本次执行修复：Replay事务保护roots/cut边界；同GPU cold workers互斥并记录设备；大COO固定关联矩阵的分块反向。未改变STE/Adam、no-op或目标。
- Slurm65911：4个历史Replay失败组合在两条路径下均16/16安装且CEC PASS，另2个成功对照通过。Slurm65912：原版16/24独占40步通过、827独占OOM。Slurm65958：修复后三者40步通过；827峰值allocated约9.70GiB。Slurm65959：552的64个保存Decision做有限函数剖析。
- 新验证没有Mapping，不能据此重写full905状态/Delay/Runtime；未新跑100/905或A/B。结果与失败探针均保留。当前无待续作业。
- 用户最新目标：CEC PASS；真实Mapping Delay整体优于Baseline B；Area改善次之；部署只做一次原图Mapping；GPU并行加速必须覆盖主要开销。主线是让梯度选出的联合组合更好，不预设转向预测器训练。
- 下一步等待用户具体任务。优先检查smooth目标与critical-delay选择的对齐、组合交互/投影，再做主要阶段GPU化；不擅自调参或启动研究实验。

---

# 对话交接报告：MFFC + LiteCut + STE Adam（2026-11-16）

## 新对话先读

当前已完成固定随机100电路的真实A/B/Ours评估，并整理代码、结果和旧中间文件。**本轮任务已经结束，没有正在运行或获准追加的科研实验。** 后续先结合实际证据讨论用户提出的问题，不自动重跑、调参、扩大到905或恢复多次Mapping反馈。

公开仓库：[Appleeyelala/CULS_LiteCut](https://github.com/Appleeyelala/CULS_LiteCut)。默认`main`是阅读入口；最新数据在[`analysis/ste100/`](analysis/ste100/README.md)。旧905资料仍在`analysis/benchmark905/`，两个实验不能混算。

建议按顺序读：本文件 → [STE100结果](docs/mffc_gpu/STE100_RESULTS.md) → [实现说明](docs/mffc_gpu/STE_ADAM_IMPLEMENTATION.md) → [原实施Plan](docs/PLAN_MFFC_STE_ADAM_随机100电路评估.md) → [逐电路CSV](analysis/ste100/STE100_RESULTS.csv)。

## 研究目标与概念

目标：通过strict maximal MFFC分区、LiteCut局部特征/延迟预测和跨MFFC候选联合梯度选择，降低真实ABC技术映射后的全局critical delay，长期用于论文研究。

- **MFFC Region**：原图互不重叠的maximal standard MFFC，负责结构覆盖、分组和Region DAG；不是一个候选。
- **Decision**：固定`(root,cut_slot)`；同一region可有多个decision。
- **Rewrite Region**：对应decision的cut-bounded removable cone，不等于maximal MFFC。
- **Candidate**：decision下的replacement subgraph；choice=0是原始AIG零编辑。
- **LiteCut score**：冻结Student对19维局部特征的`abc_delay`预测，不是PI→PO或全局真实Delay。
- 原AIG节点、GPU节点、replay序列化节点、mapper节点与mapped-netlist节点不能混用。

Gate1/2/3A～F历史证据和c1355的覆盖、候选、交互验证见[WORK_HISTORY](docs/WORK_HISTORY.md)。历史Gate3F的PO arrival使用MIO，global使用Liberty；旧撤回的PI→PO标签不再有效。最新100实验统一使用ABC_SCL_LIBERTY，不与这些旧标签合并。

## 当前实际流程

原图 → Ours自己付费的一次原图CEC/Mapping → strict MFFC/Region DAG → CULS候选枚举与必要Boolean验证 → CandidateFeasibility → 冻结LiteCut Student → 原图锚定的全局时序代理 → Softmax＋STE＋Adam 40步 → 取已访问组合中代理目标最优者（含原图） → 非空组合至多一次真实Replay → 全部ACCEPTED → 最终CEC → 一次Mapping → 按真实Delay/Area决定采用或回退原图。

空组合直接返回已测原图；无replay-noop Mapping，无中间ABC反馈、在线校准、greedy/warmup、repair或重试。Ours总Mapping次数只能为1或2，不免费借用A/B测量。

### 梯度具体作用

每个Decision的Candidate/no-op具有可训练float64 logits `z`。有效掩码后：

`p = softmax(masked_z / T)`

`v = hard - p.detach() + p`

hard由GPU上的每Decision最佳非零候选相对no-op的logit margin，经现有可行性投影构成。只向CPU传最多128条稀疏编辑，最终最多16个非零Decision；不逐步把完整D×W logits复制到CPU。

`loss = 0.02 * logsumexp((PO / original_delay) / 0.02) + 20 * relu(area / original_area - 1.05)^2`

loss通过完整代理反传到logits；Adam只更新logits。Student不接收梯度，代理12个参数冻结为原有先验，显式interaction系数为零；DAG上下游传播仍联合计算。该实现不是对真实ABC求导，也不是已完成的AGD论文复现。Smooth Tchebycheff选项存在于旧目标函数中，**本轮实际采用面积约束目标**。

固定：40步、1restart、Adam lr=.12、梯度范数裁剪5、T从1线性退火到.3、no-op初始偏置+2、噪声标准差.01、优化seed20261117。每步记录代理loss、梯度/更新范数和hard编辑，最终挑访问过的最优组合，不仅取末步；第40次更新后没有额外第41次投影。不得把这段描述改成旧方向导数搜索。

## 真实实验结果与边界

原905清单无过滤抽样：`sorted(random.Random(20261116).sample(range(905),100))`。100输入各A/B/Ours，300终态全部PASS。

A=原图Mapping；B=`strash; rewrite`一次后Mapping；Ours如上。均为同一ABC/ASAP7、同节点资源、独立冷进程；Runtime含启动、准备、推理、优化、自付原图及最终验证、失败处理和必要输出，不含排队/一次性构建/汇总。

|方法|成功|真实Delay中位数ps|完整Runtime中位数s|
|---|---:|---:|---:|
|A|100/100|3123.414675|2.172834|
|B|100/100|3076.719850|2.702031|
|Ours|100/100|3090.977665|13.175791|

不同电路原始Delay中位数不能直接计算配对收益：

|配对（均100对）|Ours/基线 Delay几何平均比|Ours胜/平/负|Ours/基线 Runtime比中位数|
|---|---:|---:|---:|
|A|0.959272|31/69/0|5.281315|
|B|1.001070|35/27/38|4.045126|

结论：相对A约降低4.07%，相对B约增加0.107%；没有整体优于B，端到端仍更慢。不能由本对照宣称同算法GPU相对CPU的因果加速。

- 83个有候选电路均使用CUDA、各40步，共3320次原生DAG backward；17个无候选电路直接返回原图。本样本没有触发CPU代理fallback。
- 80/83出现可测logits更新；67/83改变hard组合且代理最优目标下降。
- ID242、637梯度范数为0；836约8.59e-43～1.97e-41，未产生可测float64 logits变化。没有针对这些负例调参或补跑。
- 67个非空最终组合完整ACCEPTED、CEC PASS：31个真实Delay改善、2个相等、34个变差，均未超过面积限制。后36个回退原图。
- 另16个有候选电路仍选择no-op，加17个无候选，共33个只做1次Mapping；67个做2次，Ours合计167次。
- 代理下降与真实改善不等价；当前证据不支持代理普遍准确、跨电路泛化、梯度优于其他搜索或端到端GPU加速。

## 版本与文件入口

正式实验冻结源码：`9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`。
原实现提交`21c9d1b`，原结果提交`3860d60`；里程碑tag：`mffc-ste-adam-100-results-20261116`。后续交接/清理/发布提交不代表重跑或新算法。

|文件|职责|
|---|---|
|`global_timing/mffc_ste_search.py`|logits、Softmax/STE、Adam、合法hard投影、最佳已访问组合|
|`global_timing/mffc_ste_experiment.py`|独立Ours准备、原图锚点、唯一最终请求、真实回退判定|
|`global_timing/mffc_model.py`、`mffc_gpu.py`、`framework/mffc_gpu/region_lse.cu`|既有GPU可微Region DAG|
|`global_timing/surrogate.py`|局部特征到全局PO/Area代理及scalar_objective|
|`global_timing/litecut.py`、`feasibility.py`、`execution.py`、`evaluation.py`|局部分数、可行性、安装、CEC/统一Mapping|
|`scripts/run_ste100.py`、`summarize_ste100.py`|固定抽样、冷进程三方法评估及真实结果汇总|
|`config/ste100_protocol.json`|参数、工具hash及计量协议|
|`analysis/ste100/`|可直接在GitHub读取的CSV、状态、梯度轨迹及精简方法记录|
|`provenance/ste_adam100_results.json`|原文件位置及SHA256|

最新数据的`FILE_MANIFEST.json`记录每个来源和转换。顶层交付及83份trajectory/proxy_best是原始字节；cases摘要和300方法records省略逐PO数组/路径文本，choices向量改为可恢复的稀疏表示。完整原件仍在服务器；精简未更改任何真实global Delay/Runtime或筛掉负例。

## 服务器复现定位（不要自动执行）

- 源码：`/home/wushangge/CULS_LiteCut`。
- 数据集入口：`/home/wushangge/CULS_LiteCut/open-sourceDataset`；真实实验用manifest记录的`/data`输入副本。
- 数据/运行根：`/data/wushangge/CULS_LiteCut`。
- 正式run：`/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a`。
- 原始结果：上述run的`output/`；逐电路`case_NNNN_Ours/work/summary.json`、`optimization/trajectory.json`、`final/eval_0000/observation.json`。
- 实际命令：run的`input/command.json`；配置及工具：`input/ste100_preflight.json`；环境：`input/session_65534.json`、`run.json`；stdout/stderr：`logs/slurm-65534.*`。
- 正式Slurm65534：COMPLETED0:0，39分9秒，eda/RTX4090，4CPU、64GiB、1GPU。
- 唯一必要检查：Slurm65533，`20261116T070231Z-ste-adam-smoke-778bdf85`；1项小GPU合成检查＋c1355/DMA/无候选三个Ours smoke。与正式100分开，未运行全仓或额外消融。

## 历史版本与清理

旧正式905源码`ecc85e6`使用方向导数候选块、多次真实反馈及在线校准，不是当前STE100。其Runtime分析见[旧905耗时报告](docs/mffc_gpu/RUNTIME_ANALYSIS_905.md)：约57%原图/no-op/候选真实验证、17.1%CPU LiteCut准备、0.72%梯度与校准；这些比例只属于旧905，不能套到STE100。新版本尚未作同算法CPU/GPU消融或新的详细阶段归因实验。

本轮按用户要求删除5个旧中间文件：3份进度/中间分析文档和2个一次性诊断脚本。清单、hash及恢复commit见[清理记录](provenance/ste100_handoff_cleanup.json)。正式入口、复用模块、最终报告、旧905数据、负结果、Git历史和全部冻结run保留；旧证明链接改指不可变Git历史版本。没有删除原始源码仓库、模型、数据集或实验结果。

继续遵守AGENTS.md及slurm-data-workflow：重计算只能在`/data/wushangge`通过Slurm，先查资源和已有作业，总量≤64CPU/256GiB/2GPU；只在新run中运行授权实验，记录commit和hash。未获用户明确委派，不启动子代理。当前没有待续的实验作业；不要把历史“待完成”文字当新任务。

## 可直接粘贴给新对话

> 请先阅读 https://github.com/Appleeyelala/CULS_LiteCut/blob/main/HANDOFF.md ，再读 docs/mffc_gpu/STE100_RESULTS.md 和 analysis/ste100/。当前固定随机100电路STE＋Adam评估已经完成，300方法点全部PASS，但未优于B且仍更慢。先根据源码和已保存数据理解现状；不要自动重跑、调参、恢复方向导数/多Mapping反馈或开展新研究实验。后续按我的具体问题继续。
