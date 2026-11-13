# MFFC 级 GPU 改造运行记录

共同运行根目录为 `/data/wushangge/CULS_LiteCut/runs/`。本轮只对照原图 A、单次 ABC rewrite B 与新版 Ours；CPU 仅作数值正确性参照。旧 905 和 432/48 方法点不作为新版全量结果。

| Slurm / 源码 | 目录 | 实际结果 |
|---|---|---|
| 65236 / ece4004 | `20261113T181219Z-mffc-native-gpu-validation-dc09f585` | FAILED/1:0，4s；CUDA 11.8 `atomic_ref<double>` 无 `fetch_add`。保留构建报告与原始日志，无 EDA 测量 |
| 65237 / 267e795 | `20261113T181347Z-mffc-native-cas-validation-48583136` | COMPLETED/0:0，9s；CAS 累加实现编译通过，7/7 测试通过，无 SKIP。RTX4090、CUDA11.8，重复依赖/重汇合、4096层链、宽512层、高fanin、混合拓扑、有限差分及前反向参考一致性。没有执行 rewrite/ABC |
| 65238 / eeb3aa3 | `20261113T182055Z-mffc-global-model-validation-344f085a` | COMPLETED/0:0，9s；32/32 无 SKIP。完整 GPU 代理的 PO/area、逐候选及12参数梯度、校准、全局方向提案与 CPU 参考一致；候选复用/原图传递自引用/相位置换/多Decision约束及旧核心回归通过。没有执行 rewrite/ABC |
| 65239 / 544601b | `20261113T182919Z-mffc-native-loop-validation-d928356f` | COMPLETED/0:0，52s；原生sm80/86/89构建、32/32数值/回归及c1355/ss_pcm/DMA的9/9 A/B/Ours冷方法点通过。20份真实测量、11个成功replay观察的完整ACCEPTED/CEC/原始GT_META/网表hash审计通过，无失败请求 |

65237 产物 `build/mffc_gpu/libmffc_region.so` 的 SHA256 为 `a11ba6d727cb8fa6aafe0dee511a81bbf74d5ce654a1263233ebf347e3767cd6`，源码 `framework/mffc_gpu/region_lse.cu` 的 SHA256 为 `5003ca9f7133f4d27f230f1421cd2bdbccede8c436f0862e6c698a15e434d588`。命令、编译器、GPU和逐项测试在对应 `build/mffc_gpu/build_report.json` 与 `output/mffc_gpu_tests.json`。

65239的Ours真实delay/cold E2E为：c1355 `319.617096ps/7.580s`、ss_pcm `325.867493ps/5.830s`、DMA `2411.792720ps/17.153s`。三者final均来自完整接受的直接全局梯度提案、CEC PASS；分别18/3/16个rewrites，真实area为520.214417/213.451202/5180.449220，满足no-op1.05倍。c1355/DMA真实执行CUDA全DAG前反传播，ss_pcm按固定阈值CPU回退。A/B分别为360.111877/359.501831、350.016113/319.565643、2463.366460/2491.453120ps；Ours明显更慢，ss_pcm也不优于B，不据三例宣称总体优势。

`output/mffc_loop_validation.json`保留逐点/核验数值，完整请求在每例`work/optimization/`。一次准备的`20261113T182828Z-mffc-global-loop-smoke-71c0dff2`未提交，因后续删除GPU分支不需要的CPU层张量而使用544601b新快照；不是失败或漏计实验。正式905协议仍待大图/无候选分支检查后冻结；尚无2715方法点终态。
