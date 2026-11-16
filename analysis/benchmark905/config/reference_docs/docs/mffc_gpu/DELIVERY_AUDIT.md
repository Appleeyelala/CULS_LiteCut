# MFFC GPU 与完整905：最终逐项验收

验收日期：2026-11-14。任务依据为[Goal](../research/CODEX_GOAL_MFFC并行改造与905电路评估.md) §2、§4、§5、§6。**实现、冻结、全905评估与交付均已完成；更好的整体Delay–Runtime取舍未成立，保留负结果。** 原始实验目录与历史结果未改写，没有新增本轮禁止的random/greedy/旧版/native CULS/joint正式方法组。

## 终态、版本与证据

| 运行 | 冻结源码 | 实际终态 | 作用 |
|---|---|---|---|
| 65245 | `ecc85e6ac82571caf9f1a6004098dbe976536256`，tag `mffc-gpu-905-freeze-20261114` | COMPLETED/0:0，34427s | 905×A/B/Ours独立冷方法点，随后同快照原始审计/统计/图 |
| 65247 | `84ec94d` | COMPLETED/0:0，20s | 首次只读安装/结构/取舍补充，独立保留 |
| 65249 | `b1afd3480fedbd978203a73e3168ee1f61b4097b` | COMPLETED/0:0，18s | 含完整工具成本的最终只读交付，优先发布源 |

正式实验：`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`。原始交付为其`output/benchmark905_delivery/`。最终补充为`/data/wushangge/CULS_LiteCut/runs/20261113T200926Z-mffc-final905-tool-cost-delivery-67ebf285/output/final_mffc905_delivery/`。

原始`analysis_completion.json`=AUDITED_AND_REPORTED/formal=true；最终`delivery_completion.json`=COMPLETE_AUDITED_DELIVERY/formal=true、905/2715、original_statistics_unchanged=true。5750份真实测量、3039个成功replay、2个内部失败请求；原始hash索引61067项，补充6987项。最终独立检查23个原始/补充交付产物hash、全部原CSV列与统计未改变、精确905×3唯一覆盖，47个运行源码与冻结/data/当前工作区字节一致。12:37:53还核对了全部905个成功Ours配置实际存在，以及input/gpuls/checkpoint/native_library身份；不是仅检查已存在的配置文件。

## 逐项验收

| Goal要求 | 实际交付及核验 | 状态 |
|---|---|---|
| §2 MFFC/Candidate并行且保留全局依赖 | `mffc_model.py`的Candidate→Decision→Region/port聚合；`region_lse.cu`一个warp处理就绪MFFC，设备队列完成整个DAG前/反向，保留重复端口和重汇合；`mffc_search.py`求全局方向梯度与联合块全图重评分。不是只并行多设计/多轨迹 | 已实现、已验证 |
| §2 必要数值、回归、真实安装 | 65237原生7项、65238/39模型/约束32项、65244最终41项均无SKIP；六开发输入18点、38份测量/20个成功replay原始审计；首次编译及fixture失败保留于RUN_LOG | 已完成，开发不替代905 |
| §4.1 全905与冻结 | `benchmark905_manifest.json` SHA `8c93bf7988c1707efd4b392663062eef651e9e679329f0056e802250671cf935`；65242全输入hash/接口核对、65244入口预检；仅三方法，精确2715唯一终态，无挑最佳重复 | 已完成 |
| §4.2 相同真实工具与指标 | 同一ABC/ASAP7哈希、`cec -n`及`map; topo; stime -p`，GT_META/全部GT_PO属于同一`ABC_SCL_LIBERTY`；A直接original，B单次`strash; rewrite`，Ours original/no-op另付费。未读A/B结果替Ours选best | 已核验 |
| §4.3 面积、预算、失败保留 | 固定seed20261114、4个replay请求含no-op/FAIL/repair，2轮×至多2提案、每轮16 edits，12步代理校准，area≤1.05×付费no-op，tool900/method1800s；A/B超面积raw值保留 | 已完成 |
| §4.4 完整冷E2E | CSV每点`end_to_end_runtime_s`含初始化/所需准备/所有真实验证/失败/保存，全部2715有时钟；固定eda/RTX4090、Torch/OMP/OpenBLAS/MKL1线程、2验证workers。OS页缓存未清空；queue/build/事后统计不计单方法latency | 已完成，口径明确 |
| §5.1 输入与结构 | CSV：`index,input,input_sha256,source,and_count,pi_count,po_count,regions,decisions,candidates,region_dag_layers,max_layer_width,region_ports`。94个已知无候选、0个数量未知；无候选不删除 | 已交付 |
| §5.1 三方法状态、真实QoR与配对 | CSV每个输入3行；status/delay_ps/area/cec/timing_model与冷wall；Ours的A/B改善%、delay ratio、area ratio和runtime ratio；失败delay/area/CEC保持空值 | 已交付 |
| §5.1 请求、安装、来源与设备 | CSV请求、mapping/CEC/replay调用、complete_installation_*、失败部分接受、接受rewrite/涉及MFFC、final_source/device和时钟；详细安装在/data的`installation_details.json` | 已交付 |
| §5.2.1 尝试/成功/失败/超时/分母 | A 905/905/0/0，B 905/901/4/0，Ours 905/905/0/0；配对905/901，所有方法点均保留；4个B失败不变成质量胜出 | 已交付 |
| §5.2.2 Delay | 对A/B胜平负544/360/1及354/184/363；改善中位0.355115%/0%，delay ratio几何均值0.944392/0.998084，正且可定义分母905/901；JSON保留全部配对 | 已交付 |
| §5.2.3 Runtime | 各方法all-terminal中位/P90/总记录时间；逐对baseline/Ours比中位0.135011/0.166129，未用两列中位相除冒充配对比；失败成本保留，无丢失时钟 | 已交付 |
| §5.2.4 Area | Ours905可行；A 898可行/7不可行，B828/73；对A/B面积比中位1.0；单位为固定Liberty库单位，raw与共同可行条件分开 | 已交付 |
| §5.2.5 来源/规模/结构 | JSON的`by_source`覆盖40/47/805/5/8；`by_and_size`使用冻结区间，空组N/A；`delivery_supplement.gpu_shape`有实际CPU/CUDA/none与DAG层数/宽度分组、原生事件成本 | 已交付，描述性而非因果对照 |
| §5.2.6 主要成本 | MD/JSON给inclusive阶段及各工具调用数、中位/P90/合计、缺失分母；Ours CEC4284/Mapping3944/replay命令2312，总请求3041；失败请求inclusive6.545706s、repair0，不与共享/嵌套阶段相加当独占wall占比 | 已交付 |
| §5.2.6 套件elapsed | driver33434.407486s；主Slurm34427s含原始分析，补充20/18s另列；不同于单方法时间合计，也不同于排队时间 | 已交付 |
| §5.2.7 面积可行取舍 | 三者共同成功且均可行826例，冻结容差九格关系完整保留；四类具名统计另列，前两类可重叠。对A/B“delay更好且不更慢”0/1；“更慢但delay改善”479/328 | 已交付 |
| §5.2 图表 | Delay CDF、runtime CDF、按AND规模质量—时间散点；3PNG+3PDF。PNG逐张目视检查坐标/图例/尾部，PDF均1页可读取；未删异常尾部或按结果重画 | 已检查 |
| §5.3 六个命名文件 | 根目录方案、协议、manifest、完整CSV/JSON/中文MD均存在；图在`docs/mffc_gpu/figures/`，小型来源清单在`provenance/mffc905_delivery.json`，完整原始证据留/data | 已交付 |

## 复制及数据完整性

根CSV/JSON与65249源产物逐字节一致，PNG/PDF亦逐字节复制；根Markdown补充一页中文核心结果、已确认限制、原始路径并修正图链接，原/data文稿不改。来源/目标hash及文稿变换记录在[发布清单](../../provenance/mffc905_delivery.json)。原始23产物hash检查及完整原CSV字段/统计保持不变，不等于再次执行5750个EDA测量；原始审计自身已经按日志、trace、网表字节核对并记录证据索引。

最终呈现检查另核对9个发布产物hash、144个Markdown本地链接（缺失0）、2715个唯一终态和完整冷时钟、2711个成功行的CEC/ABC_SCL_LIBERTY标记、4个B失败行的质量字段为空及905个Ours的实际面积约束；六个根目录命名交付均存在。CSV原样保留CRLF和加引号的多行失败日志，按CSV解析为2715条记录（含header共2756物理行）；Git暂存blob逐字节一致，空白检查使用`core.whitespace=cr-at-eol`，不为格式检查改写证据。此检查仅读取小型现有产物，不运行模型或EDA。

[梯度状态报告](../../GRADIENT_OPTIMIZATION_STATUS.md)保留早期08:55截点，同时新增正式905完成更新。README、IMPLEMENTATION和RUN_LOG已同步当前完成状态，旧版905及432/48、Gate3F不同timing模型维持各自范围。

## 性能与科学结论的边界

- Ours相对A有真实质量改善，但对B胜354/负363，改善算术均值−2.457263%，几何均值略好不能当稳定优势；典型冷wall显著更慢。§6允许如实交付该负结果，完成工程不等于达到论文性能目标。
- Ours final来源415梯度、359 no-op、131原图；366例跨多个MFFC、5872个final接受rewrite，不证明跨MFFC超加性、梯度独立贡献或LiteCut通道有效性。
- BRAM_SSP的2个请求共享一次gpuls CUDA heap分配失败；其最终no-op成功，不能写零请求失败。命令3.269543s与两个重叠inclusive请求时钟分开，未填成功timing；更深显存原因未验证。
- 806个实际CUDA点原生事件合计逐点中位70.09ms；其余Torch、CPU准备和EDA不是此事件时钟。没有本轮同算法CPU/GPU完整对照，不能把A/B速度比当纯GPU加速。
- 精确remapping可微性、可靠全局预测、全部路径目标、动态改图后DAG更新、AGD严格复现、当前独立消融、泛化及P&R signoff均未完成或未验证。没有为完成本轮偷偷增加新实验。
