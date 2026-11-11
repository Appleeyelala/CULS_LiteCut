# 测试证据与未覆盖部分

测试按源码快照记录，不能把import、编译或`--help`成功解释为完整功能通过。全量benchmark仍在执行；本文件不是整体验收声明。

## 核心与实际工具

`20261111T132506Z-student-native-regression-7773e417/output/tests.json`记录42项核心测试全部通过，包含MFFC参照、冲突与空候选、有限差分梯度、两种目标、失败CEC、上下文cut证明、真实GPU候选/replay、常量导出、Student推理缓存、Liberty端点及路径重构。随后完整程序矩阵生成完毕，Slurm step 64341.2最终为COMPLETED；其中明确保留历史环境缺失。

`20261111T132425Z-student-feature-regression-9184e32c`的CPU矩阵为32 PASS、10个真实工具入口SKIP；不能把这些SKIP算作CPU任务完成了GPU验证。其保留输入复核另行重算c1355的417条、div的122,869条原实现/候选特征及预测，与旧25维提取流程保存的全部score记录完全相同。没有训练模型或更换权重。

`20261111T130741Z-selection-cost-regression-3cba9919`检查候选归一化、24组标量/批量坐标成本及可获得的旧独立选择。缺失完整旧选择的div和缺失旧完整catalogue的hyp明确为`NO_SAVED_REFERENCE`，不冒称做过这些完整对照。

路径补充审计在`20261111T125125Z-path-provenance-regression-1876ee41`完成：8设计、124份映射、168个成功观察引用，全部重新CEC及检查数值/网表正文不变，重构当前映射内的最大arrival链。范围和节点身份限制见模型契约。

## 全程序矩阵

最新已完成矩阵`20261111T132506Z-student-native-regression-7773e417`覆盖2,495项源码/入口：1,553项语法PASS、1,350个编译translation units、127项import PASS、17 FAIL、1 SKIP、51项CLI smoke PASS。45个文件有来源字节匹配的命名功能证据；这不是行覆盖率，其余项目明确不宣称逐文件功能覆盖。并发完成的64357重算核验另有完整报告，不回写已生成矩阵。

导入失败/跳过的实际原因如下；具体逐文件记录在`output/repository_tests/test_matrix.json`：

| 类型 | 路径或范围 | 已有证据/所需条件 |
|---|---|---|
| 命令式脚本缺参数/相对输入 | `legacy/gate1_aigrewrite/gate1c_r_io_signature.py`、`gate1c_select.py`及4个R9入口 | 无参数import仍为FAIL；指定保留输入的独立真实CLI测试已通过 |
| 缺DGL | LiteCut的`abc_pipe/`、部分`data_preprocess/`、`entry_script/`、`models/attn_sage.py`、`subutils/infer_and_collect.py`共8项 | 需要对应训练/图处理环境；Student推理不依赖这些入口 |
| 缺einops/deepgate | `models/Attention.py`、`models/CellPool_wo_lib.py` | 对应依赖及模型环境尚未恢复 |
| 原训练数据不可访问 | `projects/litecut/src/filt_data.py` | 固定读取其他用户的数据目录，当前PermissionError；未更改权限或伪造输入 |
| import触发外部写入 | `projects/litecut/src/entry_script/process_aig.py` | 隔离探针阻止写入固定历史路径，记SKIP；未直接执行该旧入口 |

历史shell/Slurm脚本只做语法及路径审计，未直接启动可能覆盖旧结果的任务；ABC变体、头文件和可选组件只有明确构建/命名测试支持时才标记相应成功。当前扫描没有发现数据库API或数据库文件，不虚构数据库测试。

所有上述目录的共同前缀为`/data/wushangge/CULS_LiteCut/runs/`。测试报告、环境包版本、失败日志和输入/二进制hash保存在各运行内，原始仓库与历史结果不改写。
