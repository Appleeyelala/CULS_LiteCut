# 未提交Rewrite的根因诊断

诊断对象：正式源码 `9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`，固定随机100电路既有运行。使用 `systematic-debugging` 的“追溯边界 → 对照模式 → 单项假设检验”流程。没有修改算法、运行ABC/CEC/Mapping、Student推理、CUDA内核或Adam搜索；未重跑100电路。本文区分原始记录直接确认的事实、源码推导、基于保存数据的独立解析核算，以及尚缺证据的假设。

**17个无候选的停止边界是GPU候选mask导出；16个有候选no-op的停止边界是候选logit未超过no-op。后者没有进入组合冲突拒绝阶段。** 零梯度只解释其中两个特征完全不可区分的电路；另一个极小梯度电路有不同原因。不能把全部33个写成“梯度不起作用”。

## 诊断方法与证据

1. Phase 1：从 `summary.selected=original_noop` 反向追溯到 `proxy_best`、40步trajectory、Boolean审计、候选manifest/catalogue、原始分区和LiteCut缓存；核对冻结代码与正式运行版本。以历史保存的40步记录核对行为，不重新启动实验。
2. Phase 2：比较67个非空电路与16个no-op，比较242/637的等特征与836的非零特征，比较有无Boolean/feasibility拒绝。
3. Phase 3：以每个记录边界的计数检验假设；用保存的特征、Region DAG和原始AIG拓扑做标量解析核算，计算no-op处每个候选对代理目标的方向导数。此数值是独立推导的辅助证据，**不是正式运行保存的候选梯度，不是新的CUDA/模型运行**。
4. Phase 4按本次授权范围未进入：未修复实现、未改参数。最小后续方案见[建议](RECOMMENDED_FIXES.md)。未记录的反事实不伪装成已验证原因。

[NO_REWRITE_CASES.csv](NO_REWRITE_CASES.csv)共33行；[SAVED_PROXY_SENSITIVITY.csv](SAVED_PROXY_SENSITIVITY.csv)列出16个有候选no-op的逐候选特征差和解析灵敏度；[TRAJECTORY_OVERVIEW.csv](TRAJECTORY_OVERVIEW.csv)覆盖83个实际优化电路。所有输入路径、SHA256和冻结源码位置见[EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)。

正式run根目录：`/data/wushangge/CULS_LiteCut/runs/20261116T070435Z-ste-adam-random100-d532863a`。下文每个ID对应 `output/case_NNNN_Ours/work/`，NNNN为四位补零ID。表格中的计数来自该电路自己的文件，不是按电路名称猜测。

## A. 17个无Candidate：候选在何处消失

调用链：[global_timing/mffc_ste_experiment.py:53](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_ste_experiment.py#L53) 先完成原图测量和strict maximal MFFC分区，然后调用 [global_timing/candidates.py:121](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/candidates.py#L121) 导出GPU候选，再调用Boolean验证。17例均有非空分区和大量3/4-leaf retained cuts，但 `gate3c_raw_manifest.json` 中 `executable_decision_count=candidate_count=0`。`catalogue.json`也为0，`cut_validation/audit.json`的raw/verified/excluded均为0；没有创建 `candidate_feasibility.json`，也没有进入LiteCut或Adam。

|ID|输入（forgeEDA/）|AND数|MFFC数|导出的eligible cuts|GPU Candidate / Boolean排除|
|---:|---|---:|---:|---:|---:|
|124|DEC_mem.aig|10554|2202|40427|0 / 0|
|188|InstMem.aig|6241|2143|23795|0 / 0|
|191|Instruction_Decode.aig|5386|1130|23722|0 / 0|
|410|cr_quantizer.aig|7928|2648|21120|0 / 0|
|412|crc32_dat112_flat.aig|5433|1811|16755|0 / 0|
|414|crc32_dat128_flat.aig|6216|2072|19186|0 / 0|
|461|dual_port_sync.aig|51601|18833|238138|0 / 0|
|518|icache_ram.aig|22284|4620|87409|0 / 0|
|565|lscc_ram_dq.aig|39302|1214|166234|0 / 0|
|586|mig_7series_v2_1_ecc_dec_fix.aig|18240|64|58543|0 / 0|
|593|mig_7series_v4_0_ecc_merge_enc.aig|17792|576|79826|0 / 0|
|638|nv_ram_rws_16x256.aig|12314|4122|46900|0 / 0|
|662|nv_ram_rwsp_256x16.aig|8786|370|36462|0 / 0|
|704|pa_f_spsram_32x46.aig|9034|1664|38566|0 / 0|
|729|picosoc_mem.aig|42817|9589|191648|0 / 0|
|792|reg_module.aig|12288|4096|12288|0 / 0|
|887|y_quantizer.aig|7928|2648|21120|0 / 0|

每行证据：`candidates/raw/gate3c_raw_manifest.json` → `candidates/catalogue.json` → `cut_validation/audit.json` → `summary.json`。manifest记录 `use_zeros=false`、`cut_set_size=8`、`candidate_rule=saved_minus_nodes_added_gt_zero`。这些是已经经过枚举保留策略的cuts，不是全部可能cuts。

源头规则见 [projects/culs/src/algorithms/rewrite.cu:383](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/projects/culs/src/algorithms/rewrite.cu#L383) 的 `EvaluateNode`：

- 只处理`used`且至少3叶的cut；枚举最大4叶、每root固定8槽，并有cut合并/过滤/支持最小化。见 [projects/culs/src/algorithms/rewrite.cu:269](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/projects/culs/src/algorithms/rewrite.cu#L269)。
- library subgraph若`match[rt]==id`，跳过根自复用候选（约第426行）。
- `saved - nodesAdded <= 0`在`fUseZeros=false`时跳过（第435行）；只有其余项才设置`candidateMasks=1`。
- 导出入口明确以`false`调用预评估，然后按mask导出候选；见 [projects/culs/src/algorithms/rewrite.cu:2106](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/projects/culs/src/algorithms/rewrite.cu#L2106)。Python正常化只保留 `candidate_count>0` 的Decision。

**已确认：**17例在当前保留cuts、library和GPU规则组合定义的候选域内没有导出的候选；不是MFFC为空，不是Boolean或CandidateFeasibility过滤掉全部候选，更不是Adam失败。positive-gain是明确限制条件，会排除零/负节点收益方案，尽管后者可能改变Mapping Delay。

**未确认：**不能说17例全部“只因positive-gain”或“原图绝对无任何延迟改进机会”。保存的JSONL只记录通过mask的subgraph及CPU侧saved_count，没有每个被拒subgraph的GPU nodesAdded、gain、self-reuse理由，也没有被cut预算淘汰的候选。因而无法拆出gain过滤、自复用跳过、cut截断的各自贡献，也不能仅凭全零mask排除枚举器实现缺陷。现有记录没有给出可归责的代码缺陷。最小下一步是对一个代表例补齐这些边界计数，暂不放宽规则或重跑100例。

## B. 16个有Candidate但no-op：共同停止边界

16例的Boolean排除数和feasibility `blocked_choices`均为0；原始候选均仍在有效选择域。各有40次原生DAG backward，13例有非零logits更新，242/637/836无可测更新。全部640个step：

- `host_candidate_rows=0`，所以 [global_timing/mffc_ste_search.py:24](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_ste_search.py#L24) 的正margin过滤没有输出任何proposal；有效非零候选最大logit未超过no-op。
- `hard_edits=[]`、visited hard组合仅1种、loss逐步完全相同，`proxy_best.step=-1`并保留原图。
- 每个候选单独激活都满足保存的静态可行性条件：no-op起点、无blocked choice，单元素集合没有两两冲突。此事实不替代真实安装/CEC。

因此“非空组合曾经出现但被best选择丢弃”“被16编辑上限挡住”“所有候选被合法性mask屏蔽”“进入CPU投影后全部被冲突拒绝”均被已有记录排除。top128截断也不是零proposal的原因：至少一个正margin存在时，排序top128一定含正margin。

投影返回no-op之前，具体信号分为以下六组（含前述17例）：

|类别|ID|证据支持的解释|
|---|---|---|
|GPU导出域为空|上述17例|停止于mask导出，具体拒绝分解尚未保存|
|代理特征不可区分|242、637|唯一候选与原图19维特征、Student分数完全相同，冻结时序与面积代理均无选择差异|
|冻结代理没有下降方向|643|两个候选的局部时序系数分别为正和0；所有组合都不能降低当前代理目标|
|灵敏度衰减至不可测更新|836|候选有负时序系数，传播到目标的灵敏度约1.68e-39；梯度非零但Adam步长低于float64可分辨尺度|
|梯度/更新很小，未激活|546、571、658、744|有代理有利方向，但记录的全程logits变化分别只有0.0121、0.1444、3.18e-5、6.25e-11，未跨越no-op margin|
|混合信号，未激活|445、468、561、574、641、650、651、778|解析显示强信号主要来自代理不利候选，代理有利候选信号弱；确有更新，但40步无正margin。缺少逐候选logit/概率轨迹，不能精确分摊偏置、退火、epsilon与预算的因果贡献|

### 逐电路梯度与更新证据

以下梯度/更新列均来自正式轨迹；`h_min`是保存数据解析所得的最有利候选no-op方向导数，与logits梯度不是同一个量。

|ID|Decision / Candidate|最大记录梯度范数|logits总变化范数|解析h_min|分类|
|---:|---:|---:|---:|---:|---|
|242|1 / 1|0.000e+00|0.000e+00|0.000e+00|IDENTICAL_FEATURES_PROXY_FLAT|
|445|12 / 15|2.435e-04|4.646e+00|-5.206e-16|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|468|21 / 41|3.188e-04|4.640e+00|-4.036e-27|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|546|1012 / 2144|3.016e-11|1.206e-02|-4.850e-11|WEAK_GRADIENT_NO_ACTIVATION|
|561|15 / 18|3.260e-02|5.659e+00|-5.292e-09|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|571|448 / 653|4.315e-10|1.444e-01|-1.320e-09|WEAK_GRADIENT_NO_ACTIVATION|
|574|22 / 27|3.710e-05|6.896e+00|-4.529e-10|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|637|1 / 1|0.000e+00|0.000e+00|0.000e+00|IDENTICAL_FEATURES_PROXY_FLAT|
|641|7 / 7|1.006e-03|3.295e+00|-8.230e-12|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|643|2 / 2|2.275e-11|7.776e-03|0.000e+00|NO_DESCENT_IN_FROZEN_PROXY|
|650|5 / 7|2.545e-04|2.680e+00|-8.595e-16|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|651|5 / 7|2.671e-04|2.680e+00|-3.950e-19|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|658|1 / 1|9.252e-14|3.181e-05|-6.229e-13|WEAK_GRADIENT_NO_ACTIVATION|
|744|83 / 120|1.678e-19|6.255e-11|-1.013e-18|WEAK_GRADIENT_NO_ACTIVATION|
|778|11 / 17|2.024e-04|5.333e+00|-1.902e-08|MIXED_PROXY_SIGNAL_NO_ACTIVATION|
|836|6 / 8|1.967e-41|0.000e+00|-8.094e-41|ATTENUATED_GRADIENT_NO_FLOAT64_UPDATE|

### 242和637：特征相同导致代理目标平坦

- 242：Decision 0，原AIG root4411、region8；原图/候选Student分数均为49.72636544822439；Weighted_Depth均为2，Node_Count均为5。
- 637：Decision 0，原AIG root5667、region1169；分数均为60.327277582958835；Weighted_Depth均为2，Node_Count均为5。
- 两例各自的原图与候选 `feature_hash`完全相同。19维相同并不代表AIG结构相同，也不代表真实Mapping相同。

[global_timing/surrogate.py:126](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/surrogate.py#L126) 构造 `[局部分数相对差, Weighted_Depth相对差, Node_Count相对差, 1]`；前三项都是0，第四项乘冻结的0权重。`local_weights=[.08,.15,0,0]`、interaction全0、area只用Node_Count差。故候选与no-op在整个代理中不可区分，损失对选择的梯度为0；不是no-op偏置把已有梯度“挡住”。独立解析还确认两者到PO的结构灵敏度非零，尤其637为约0.183，排除了“根本不连到输出”这一解释。

证据：对应 `litecut_cache/*.json`、`cut_validation/verified_catalogue.json`、`optimization/trajectory.json`。当前局部特征无法区分这两种改写，后续只能先检查差异信息如何表达；调学习率或多跑相同目标不能创造缺失信号。

### 643：当前代理本身不偏好改写

两个候选的 `q=.08*delta_local+.15*delta_Weighted_Depth` 分别为0.009209684884627527和0。冻结interaction为0，q非负会使相应port cost不减，Region DAG及PO平滑目标对cost单调不减；面积罚项原图为0且始终非负。因此在当前代理候选域内任何组合均不能严格优于no-op。记录中小幅logits更新符合降低不利候选概率的方向；不应强行把它归类为“没有探索”。这项结论仅针对冻结代理，不能推出真实Mapping无改善。

### 836：衰减发生在代理传播，Adam的epsilon进一步限制步幅

8个候选中5个q<0、3个q=0；有效mask没有屏蔽它们。用保存的Region DAG/port incidence、原图PO和特征做no-op处标量解析，三个候选region（1654、1815、2299）的结构灵敏度均约 `1.6767254e-39`，最负候选方向导数约 `-8.094416e-41`。正式轨迹logits梯度范数为 `8.593071e-43～1.966610e-41`，40次更新范数全部0。

传播每条边乘 `exp((upstream + cost - arrival)/temperature)`，见 [framework/mffc_gpu/region_lse.cu:117](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/framework/mffc_gpu/region_lse.cu#L117)。上游结构分支可以通向关键PO，但在该结构代理中仍只有极小权重；不能仅凭PO可达性或真实端点的高Delay认定梯度应当大。

运行缓存记录PyTorch `1.13.1+cu117`。现有同版本Adam源码默认 `eps=1e-8`，STE入口只显式设置lr=.12。以最大记录范数给出单坐标步幅上界，约 `.12*1.97e-41/1e-8 = 2.36e-34`；比约0.01或2量级float64 logits的分辨率低很多。已有`adam_update_norm=0`直接确认数值停滞。这里不是梯度浮点下溢到0（记录梯度非零），也没有发现CUDA backward未调用。

### 偏置、温度、Softmax与40步的作用边界

`z_noop += 2`（[global_timing/mffc_ste_search.py:48](https://github.com/Appleeyelala/CULS_LiteCut/blob/9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac/global_timing/mffc_ste_search.py#L48)）与“非零候选logit必须超过no-op”的投影共同形成启动门槛。温度从1到.3只改变softmax和反向梯度，不改变同一组logits的argmax。只有hard组合前向参与代理，所以概率变化可以完全不改变记录loss。

13例有非零梯度和实际logits变化；242/637/836即使logits不动，退火也会改变非退化softmax分布。但本次没有保存每步概率、margin或分候选梯度，不能声称测得了某个概率增长幅度或“把偏置改小即可成功”。没有用随机初始化的近似值冒充实际logits。

对照67个非空案例：全部在step9首次出现非空hard组合，之后31个step非空；同样的+2偏置、lr=.12与40步可正常越过门槛。ID117最终真实改善，ID3最终真实退化，两者都在step9激活并找到更低代理目标。这证明激活、代理优化和真实QoR必须分别诊断。

## 解析核算的公式及限制

设候选局部系数 `q_dc=.08*f0+.15*f1`；在no-op处，成本 `c_e=depth_e*depth_to_ps`。输出目标对PO的权重是归一化指数权重除以原图Delay。沿保存Region DAG反向展开：

`h_dc = q_dc * sum_{e influenced by decision d}(c_e * adjoint[target(e)] * exp((arrival[source(e)]+c_e-arrival[target(e)])/T_DAG))`

这里 `T_DAG=.005*original_delay`，输出目标温度为归一化`.02`；它们与候选Softmax温度1→.3是三个不同温度。面积在no-op处低于1.05阈值，面积罚项导数为0。真正logits导数还需 `g_dc=(p_dc/T_selection)*(h_dc-sum_k(p_dk*h_dk))`，所以不能把h直接称为保存的logits梯度。

`saved_proxy_sensitivity.py`按冻结公式读取保存数据做标量运算，只有AIG解析函数来自冻结源码，不导入torch、不执行模型或优化器。浮点归约顺序不同，数值用于量级/方向分析，不声称逐位复现GPU。16例hard恒为no-op，因而该处代理的选择导数是相关的固定分析点。

## 尚未解决的根因与最小取证

- 17例：缺每个被拒library subgraph的gain/self-reuse/cut淘汰原因，无法证明放宽positive-gain是否会增加候选，或枚举实现是否漏掉合法候选。
- 混合信号8例：缺逐候选梯度、logit margin和softmax概率，不能把总更新范数直接解释为有利候选获得足够更新。优先在一个代表例补每步最有利候选与no-op的margin、h/g和概率；不必保存完整tensor。
- 缺第40次Adam更新后的投影。本版本明确定义40次“投影→loss→更新”，没有第41次投影；不能推断第40次更新后是否有新组合，也不能把未访问组合作为历史结果。
- 已确认特征不可区分和梯度尺度问题，但没有证据确认本次实现存在“梯度没接上”或“mask写错”的程序缺陷。CPU/GPU梯度一致性或新的反事实试验本次未执行。
