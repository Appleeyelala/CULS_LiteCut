# MFFC 级 GPU 改造运行记录

共同运行根目录为 `/data/wushangge/CULS_LiteCut/runs/`。本轮只对照原图 A、单次 ABC rewrite B 与新版 Ours；CPU 仅作数值正确性参照。旧 905 和 432/48 方法点不作为新版全量结果。

| Slurm / 源码 | 目录 | 实际结果 |
|---|---|---|
| 65236 / ece4004 | `20261113T181219Z-mffc-native-gpu-validation-dc09f585` | FAILED/1:0，4s；CUDA 11.8 `atomic_ref<double>` 无 `fetch_add`。保留构建报告与原始日志，无 EDA 测量 |
| 65237 / 267e795 | `20261113T181347Z-mffc-native-cas-validation-48583136` | COMPLETED/0:0，9s；CAS 累加实现编译通过，7/7 测试通过，无 SKIP。RTX4090、CUDA11.8，重复依赖/重汇合、4096层链、宽512层、高fanin、混合拓扑、有限差分及前反向参考一致性。没有执行 rewrite/ABC |
| 65238 / eeb3aa3 | `20261113T182055Z-mffc-global-model-validation-344f085a` | COMPLETED/0:0，9s；32/32 无 SKIP。完整 GPU 代理的 PO/area、逐候选及12参数梯度、校准、全局方向提案与 CPU 参考一致；候选复用/原图传递自引用/相位置换/多Decision约束及旧核心回归通过。没有执行 rewrite/ABC |
| 65239 / 544601b | `20261113T182919Z-mffc-native-loop-validation-d928356f` | COMPLETED/0:0，52s；原生sm80/86/89构建、32/32数值/回归及c1355/ss_pcm/DMA的9/9 A/B/Ours冷方法点通过。20份真实测量、11个成功replay观察的完整ACCEPTED/CEC/原始GT_META/网表hash审计通过，无失败请求 |
| 65240 / 56b6afd | `20261113T183438Z-mffc-deep-empty-validation-0eb1e760` | COMPLETED/0:0，5m02s；div/hyp/mig三个输入9/9 A/B/Ours终态PASS。div/hyp使用全GPU DAG，mig无Decision/无代理；全部请求无失败，真实final均CEC/面积可行 |
| 65241 / 914b29c | `20261113T184502Z-mffc-development-evidence-audit-b33d3b0f` | COMPLETED/0:0，17s；4/4统计测试；对65239/65240保存证据各自审计20/18份实际测量、11/9个成功replay观察，均PASS，三组PNG/PDF各自生成，无新EDA |
| 65242 / bbc47e3 | `20261113T184743Z-mffc-complete905-input-freeze-3baac755` | COMPLETED/0:0，1m39s；全部905个staged AIG的SHA与头部接口/节点数核对通过；附既有strict partition/DAG/域元数据，没有重新运行模型/EDA。manifest SHA8c93bf79… |
| 65243 / 68f3984 | `20261113T185837Z-mffc-final-preflight-ab0b62fa` | FAILED/1:0，7s；40项中新增两个控制fixture缺`structure_meta`而ERROR，其余通过；入口检查未执行，没有新方法点。保留失败证据，不归因于真实GPU/CEC错误 |
| 65244 / e0f4ad4 | `20261113T190558Z-mffc-freeze-gate-validation-8e9a544c` | COMPLETED/0:0，9s；41/41无SKIP；完整905清单、工具hash、RTX4090/eda和线程/协议入口检查通过，`methods_launched=0`。修正fixture、独占原子终态写入、活作业拒绝恢复、丢失时钟N/A及预算/修复控制已验证 |

65237 产物 `build/mffc_gpu/libmffc_region.so` 的 SHA256 为 `a11ba6d727cb8fa6aafe0dee511a81bbf74d5ce654a1263233ebf347e3767cd6`，源码 `framework/mffc_gpu/region_lse.cu` 的 SHA256 为 `5003ca9f7133f4d27f230f1421cd2bdbccede8c436f0862e6c698a15e434d588`。命令、编译器、GPU和逐项测试在对应 `build/mffc_gpu/build_report.json` 与 `output/mffc_gpu_tests.json`。

65239的Ours真实delay/cold E2E为：c1355 `319.617096ps/7.580s`、ss_pcm `325.867493ps/5.830s`、DMA `2411.792720ps/17.153s`。三者final均来自完整接受的直接全局梯度提案、CEC PASS；分别18/3/16个rewrites，真实area为520.214417/213.451202/5180.449220，满足no-op1.05倍。c1355/DMA真实执行CUDA全DAG前反传播，ss_pcm按固定阈值CPU回退。A/B分别为360.111877/359.501831、350.016113/319.565643、2463.366460/2491.453120ps；Ours明显更慢，ss_pcm也不优于B，不据三例宣称总体优势。

`output/mffc_loop_validation.json`保留逐点/核验数值，完整请求在每例`work/optimization/`。一次准备的`20261113T182828Z-mffc-global-loop-smoke-71c0dff2`未提交，因后续删除GPU分支不需要的CPU层张量而使用544601b新快照；不是失败或漏计实验。大图/无候选检查已完成，正式协议待输入核验与最终冻结；尚无2715方法点终态。

65240的真实A/B/Ours delay（ps）为：div `75831.0469/67247.3594/75760.1484`，hyp `1774327.75/1772861.75/1774327.75`，mig `688.224609/688.224609/688.224609`。Ours冷E2E为87.328/149.424/6.337s，来源为直接梯度/付费原图回退/no-op。div的B area25400.459超出no-op面积阈值，raw QoR照实保留。hyp没有delay收益；不得把运行成功称性能成功。完整逐点Area/时间/设备/来源及图在65241的两个`output/development_<实验名>/`中，均明确标为开发而非全905。

正式源冻结门槛已满足：`provenance/mffc_gpu_freeze_audit.json`逐字节核对47个运行源码与65244通过快照相同；冻结后仅协议状态/身份和文档/来源变更。固定seed20261114、预算4含no-op/失败/repair、最多2轮×2提案、每轮16edits、12步校准、area≤1.05no-op、900/1800s时限。正式运行必须独立新目录、使用本地tag `mffc-gpu-905-freeze-20261114`，预检本身不是2715点实验结果。

## 正式完整905运行：进行中

65245已于2026-11-14 03:16:19（北京时间）在Compute/eda实际RUNNING，4CPU/64GiB/1GPU RTX4090、时限48h；提交时本用户其他RUNNING/PENDING为空，总请求4/64/1，未取消任何作业。源码固定为 `ecc85e6ac82571caf9f1a6004098dbe976536256`，本地tag同上。

目录：`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`。`input/{formal-command.json,freeze_identity.json,submission.json,mffc905_preflight.json,session_65245.json}`保存真实命令、冻结身份、资源核验和设备；`output/records/`逐点独占原子发布终态，原始文件在`output/case_NNNN_A|B|Ours/`。脚本完成所有905×3点后自动执行同快照的原始证据审计、统计与图表，输出`output/benchmark905_delivery/`。

当前只有部分方法点终态，不能宣布905完成或给出总体改善。以实时Slurm和`output/formal_completion.json`、`benchmark905_delivery/analysis_completion.json`及精确2715终态覆盖共同判定，不能以本条RUNNING记录代替最终证明。不得重提交仍在运行的65245，已完成点不挑最佳重跑。最终需将完整CSV/JSON/中文报告的小型交付文件放到仓库规定名称，保留所有原始失败、超时及真实runtime。

## 冻结后的纯交付分析

65246（源adc877a，`20261113T193807Z-mffc-delivery-validation-906a915b`）已COMPLETED/0:0、3s。四项安装/部分工作/丢失时钟/取舍分母测试通过；分别对两批已审计开发9点生成补充CSV/JSON/中文报告，保留原始比较统计不变，并核对每例实际资产hash。没有启动新的EDA或模型。输出在`output/supplement_<开发实验名>/`，明确`formal=false`，不计入正式905。

新增 `scripts/finalize_mffc905.py` / `global_timing/mffc_delivery.py` 只补全规定的安装成功/失败、修复成本、DAG形状与具名取舍，冻结算法/data源码不变。正式交付门槛和待核验项见 [DELIVERY_AUDIT.md](DELIVERY_AUDIT.md)。

65247已排队为`afterany:65245`，源84ec94d，目录`20261113T194518Z-mffc-final905-delivery-d3bb36f0`，2CPU/16GiB/0GPU。它仅在原始正式审计形成后核对并补充文件，不运行EDA。提交时含原作业合计6CPU/80GiB/1GPU；不能把PENDING当成正式交付完成。

随后逐项审查发现：原始CSV已有每点CEC/Mapping/Replay时钟，但MD/JSON缺少这些工具各自的总体耗时表。源510ae7b补充`tool_cost_summary`，按每点已有命令inclusive时间求n/中位/P90/合计，保留FAIL/TIMEOUT和缺失时钟，不改变原始统计或冻结算法。65248（`20261113T200701Z-mffc-tool-cost-validation-31baef17`，eda2、2CPU/16GiB/0GPU）已COMPLETED/0:0、3s，5/5无SKIP；两开发批次各9点的原始统计保持不变，真实工具成本补充生成成功，没有重跑模型/EDA。选择eda2避免与正式eda上的计时工作共享CPU。

新的开发补充在上述65248的`output/supplement_<开发实验名>/`。例如N的三个Ours点记录CEC15次/1.482559s、Mapping14次/19.377779s；H为CEC13次/34.490238s、Mapping12次/100.938567s。它们是已有命令的inclusive和，不是额外调用数或独占wall占比。当前工作区与正式/data快照的47个冻结运行源码再次逐字节一致；新增代码仅影响交付统计。最终正式交付需使用含该成本补充的验证版本，原65247产物保持为先前交付分析，不覆盖。

含工具成本的正式交付65249已经排队，源b1afd34，目录`20261113T200926Z-mffc-final905-tool-cost-delivery-67ebf285`，`afterany:65247`、2CPU/16GiB/0GPU。顺序为65245正式运行/原始审计→65247首次补充→65249完整成本交付，分别保留产物。所有RUNNING/PENDING合计8CPU/96GiB/1GPU；这两项后处理不新增任何EDA方法点。最终优先核对65249的`output/final_mffc905_delivery/`，然后完成根目录文件、图表人工检查与逐项验收，不能仅凭排队或脚本状态宣布Goal完成。

2026-11-14 04:11:22（北京时间）只读进度：65245实际RUNNING，已独占发布218/2715个方法终态，72个输入三方法齐全，当前保存点均PASS；`formal_completion.json`尚未形成，未对该部分样本计算总体性能结论。Slurm当时MaxRSS约4.88GiB，非显存指标；无资源阻断迹象。运行仍按冻结协议继续，不重新提交已完成点。

2026-11-14 06:49:09只读检查：逐个读取当前终态JSON，933/2715个方法终态均为PASS，311个输入三方法齐全；与Slurm实时RUNNING及驱动日志一致，两个后处理仍PENDING/Dependency。完整运行和审计标记尚未形成，不计算该前缀的总体性能结论。工作区干净，/data仍约1.6TiB可用，Slurm MaxRSS约5.52GiB（非GPU显存）。检查没有重跑方法点或修改冻结源码/参数。

监控期间确认的长耗时验证实例：index108 `forgeEDA/Back_End_With_Decode.aig`的B已PASS，cold E2E为622.795256s，其中`work/evaluation/cec.command.json`记录573.063254s；原始CEC文本明确等价。相应Ours的两轮CUDA梯度已写入`optimization/gradient_{00,01}.json`，随后真实验证完成。index147 `forgeEDA/Division.aig`的Ours前两组评估均PASS，inclusive请求分别237.568647/214.597248s，第三组已完整安装24个rewrite；该方法点随后也已终态PASS。上述两个case的全部终态保存在正式run的`output/records/`，请求证据在各`case_NNNN_METHOD/work/`。它们解释了阶段性终态数暂停，并非全905性能结论；内部请求与最终方法状态仍由正式审计分开统计。

2026-11-14 07:28左右检查index375 `forgeEDA/cb_dct.aig`：Ours已PASS，cold E2E为172.311129s、CUDA后端，最终来自直接全局梯度，四个含no-op请求均PASS、无失败。`child_result.json:stages`记录LiteCut准备/推理70.841s、候选枚举17.341s、候选复用预检4.246s、模型准备4.381s；两轮全局梯度/联合块阶段0.529/0.572s，校准0.796s。此例表明剩余CPU特征/推理准备也可能明显影响总成本，不能把所有长耗时归因于CEC。LiteCut阶段包含物化、特征、加载和推理，并非单独Student前向；梯度阶段包含CPU投影/传输，不是纯GPU kernel。原始证据在`output/records/0375_Ours.json`、`output/case_0375_Ours/{child_result.json,work/summary.json}`。inclusive阶段有嵌套，不相加当wall；此例不替代全905分组统计，也不触发冻结算法/参数调整。

约07:33，index392 `forgeEDA/codec.aig`的B在ABC原生Rewrite触发`Abc_AigUpdateLevelR_int`断言，退出−6，原生调用0.214344236s、冷E2E 0.518344442s；该方法未取得CEC/mapping指标，FAIL保留。相同输入A/Ours已PASS，Ours来源为付费`original_fallback`。驱动继续后续输入，没有替换B或重跑失败点；冻结ABC/参数保持不变。详见[正式失败记录](FAILURE_NOTES.md)。历史也记录过该输入的同名断言，不据此声称新运行的根因已完全定位；本轮只引用本次实际文件。

09:23只读检查确认index562 `forgeEDA/loco_top.aig`的B为第二个方法级FAIL：同名`Abc_AigUpdateLevelR_int`断言，原生退出−6/0.214363001s，冷E2E 0.569874013s，没有进入CEC/mapping。相同输入A/Ours均PASS，Ours来源`mffc_global_gradient`；B缺失不作为质量胜出。原始`0562_B.json`、native命令与日志已读取，详情见FAILURE_NOTES；驱动继续后续输入，65245仍RUNNING，未更换工具/参数或重跑失败点。全量统计仍待905/2715终态与审计，不把当前两个失败当最终数量。

09:49逐项核实index610/611的`mrc3/mrc4`：两例B均在原生rewrite触发同名断言，返回−6，调用0.164326474/0.164231263s，冷E2E 0.519633666/0.518854996s；未进入CEC/mapping，缺失指标保留。相同输入A/Ours均PASS，mrc3的Ours为直接梯度，mrc4为no-op；不把B缺失或no-op当作梯度收益。已读取两例实际record/native命令/断言和输入hash，完整证据在FAILURE_NOTES。四项已知方法失败均为B，尚非最终失败数；65245仍RUNNING并继续后续输入，冻结算法与工具不变，没有新增或重跑方法点。

## 正式方法运行全部终态，原始证据审计进行中

2026-11-14 12:33:43只读监测确认`output/formal_completion.json`已经形成：905个输入、2715个方法点，2711 PASS、4 FAIL、0 TIMEOUT，仅A/B/Ours。驱动记录单session elapsed为33434.40748606506s；这是整批driver elapsed，不是单方法latency。四个FAIL均为上述B原生rewrite断言；A与Ours分别905 PASS，B为901 PASS/4 FAIL，失败指标保持缺失。

12:37:53另做轻量只读核对：2715个终态精确覆盖905×3且不重复，输入SHA/源码commit/Slurm ID均与冻结身份一致；全部905个成功Ours的`work/config.json`实际存在，input/gpuls/checkpoint/native_library四项身份与预检一致。核对没有执行模型、EDA、测试或重启方法点。

65245在方法全部终态后继续运行同冻结快照的原始trace/CEC/Liberty/netlist审计；截至12:49:01，`analysis_completion.json`尚未形成、错误日志为空，65247/65249仍为PENDING/Dependency。全量审计和交付尚未完成，不能仅凭2715终态宣布Goal完成或发布未审计的总体性能结论。CPU/I/O累计量持续增长；继续等待原作业，不重新提交实验。


## 最终完成：905正式实验、原始审计与成本交付

2026-11-14约12:50，65245最终COMPLETED/0:0，Slurm elapsed34427s（9:33:47），含方法运行和原始分析；driver elapsed33434.407486s。原始`output/benchmark905_delivery/analysis_completion.json`为AUDITED_AND_REPORTED/formal=true，精确905/2715、5750份真实测量、3039个成功replay观察、2个内部失败请求。原始审计hash索引包含61067项，成功观察全部精确ACCEPTED/CEC/同一Liberty/网表hash核对通过。

后处理65247为COMPLETED/0:0、20s；含工具成本的最终65249为COMPLETED/0:0、18s，源b1afd3480fedbd978203a73e3168ee1f61b4097b，产物`output/final_mffc905_delivery/`的状态COMPLETE_AUDITED_DELIVERY/formal=true、905/2715，原统计不变。12:51:32再次核对23个原始/最终交付文件hash、全部原CSV字段不变、精确覆盖、47个冻结运行源码与/data及当前工作区字节一致。905个成功Ours配置的实际存在和四项身份已于12:37:53独立核对，不靠“文件存在时才核验”推断全部存在。

最终A/Ours各905 PASS，B为901 PASS/4 FAIL，无方法TIMEOUT或丢失冷时钟；四项B缺失均保留。Ours有3041 replay请求，3039完整安装PASS/2 FAIL/0 UNKNOWN，2134非空完整组合；2 FAIL为BRAM_SSP同次gpuls batch设置2GiB CUDA heap失败，最终no-op成功、没有repair或部分接受。原始命令返回1、3.269543s，两请求inclusive重叠，详见FAILURE_NOTES；这不是两个独立失败工具调用。

相对A：905对，Delay胜平负544/360/1，改善中位0.355115%，Delay比几何均值0.944392，配对baseline/Ours时间比中位0.135011。相对B：901对，354/184/363，中位0%，几何均值0.998084，配对时间比0.166129，改善算术均值−2.457263%。A/B/Ours冷E2E中位2.322496/2.821493/18.766875s。工程与完整评估交付完成，整体Delay–Runtime优势未成立。

Ours全905 final CEC/面积可行，实际CUDA806、CPU5、无候选94；来源梯度415/no-op359/付费原图131。366个final跨多个MFFC，最终接受5872 rewrite，单例最多32个。实际CUDA事件合计逐点中位70.09ms，不是整个方法耗时或GPU因果加速。工具/阶段inclusive、来源/AND/DAG形状、826共同面积可行配对取舍均已交付，失败与重叠成本不删除。

六个命名交付均在仓库根目录。CSV/JSON和三组PNG/PDF从65249逐字节复制；根Markdown增加中文核心结论、已确认限制和证据索引并迁移图片链接，/data原文不改。三PNG逐张目视检查，三PDF各1页可读取。来源/hash见`provenance/mffc905_delivery.json`；逐项验收见DELIVERY_AUDIT。最终整理没有新增EDA、训练、方法组、按结果重试或改变冻结参数。
