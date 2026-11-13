# MFFC 级 GPU 改造运行记录

共同运行根目录为 `/data/wushangge/CULS_LiteCut/runs/`。本轮只对照原图 A、单次 ABC rewrite B 与新版 Ours；CPU 仅作数值正确性参照。旧 905 和 432/48 方法点不作为新版全量结果。

| Slurm / 源码 | 目录 | 实际结果 |
|---|---|---|
| 65236 / ece4004 | `20261113T181219Z-mffc-native-gpu-validation-dc09f585` | FAILED/1:0，4s；CUDA 11.8 `atomic_ref<double>` 无 `fetch_add`。保留构建报告与原始日志，无 EDA 测量 |
| 65237 / 267e795 | `20261113T181347Z-mffc-native-cas-validation-48583136` | COMPLETED/0:0，9s；CAS 累加实现编译通过，7/7 测试通过，无 SKIP。RTX4090、CUDA11.8，重复依赖/重汇合、4096层链、宽512层、高fanin、混合拓扑、有限差分及前反向参考一致性。没有执行 rewrite/ABC |

65237 产物 `build/mffc_gpu/libmffc_region.so` 的 SHA256 为 `a11ba6d727cb8fa6aafe0dee511a81bbf74d5ce654a1263233ebf347e3767cd6`，源码 `framework/mffc_gpu/region_lse.cu` 的 SHA256 为 `5003ca9f7133f4d27f230f1421cd2bdbccede8c436f0862e6c698a15e434d588`。命令、编译器、GPU和逐项测试在对应 `build/mffc_gpu/build_report.json` 与 `output/mffc_gpu_tests.json`。

接下来的完整代理/选择/可执行性代码尚未在此记录中宣称通过。现有 `MFFC_GPU_REVISION_PLAN.md` 是方案，正式905清单/协议仍待开发验证后冻结；尚无2715方法点终态。
