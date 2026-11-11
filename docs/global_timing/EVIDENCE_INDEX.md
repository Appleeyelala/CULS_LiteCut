# 实现与验证证据入口

本索引区分已完成的固定证据和仍在运行的全量验证。日期为2026-11-12（北京时间）；它不是整体验收通过声明。具体要求见[验收矩阵](IMPLEMENTATION_STATUS.md)，历史Gate结果见[工作历史](../WORK_HISTORY.md)。

## 代码与执行

| 内容 | 入口 |
|---|---|
| 完整真实闭环 | `global_timing/pipeline.py`、`global_timing/evaluation.py` |
| strict MFFC、owner与Region DAG | `global_timing/structure.py` |
| decision/candidate、冲突与cut证明 | `global_timing/candidates.py`、`global_timing/cut_validation.py` |
| Student特征、批量预测与缓存 | `global_timing/litecut.py`，复用`projects/litecut/` |
| 端口传播、反馈校准、联合离散选择 | `global_timing/surrogate.py`、`arrival.py`、`optimization.py` |
| 顺序安装及动态引用环拒绝 | `projects/culs/src/algorithms/rewrite.cu`、`rewrite.h` |
| 多目标比较、预测及路径分析 | `global_timing/reporting.py`、`search_analysis.py`、`path_changes.py` |
| 不可变运行准备与复现 | `scripts/prepare_run.py`、[运行说明](USAGE.md) |

## 已完成的固定证据

以下运行目录均位于`/data/wushangge/CULS_LiteCut/runs/`，保留原始日志、来源hash和环境；较晚证据不覆盖较早文件。

| 核验对象 | 运行目录与报告 | 能支持的结论 |
|---|---|---|
| 最新真实核心回归 | `20261111T182419Z-native-control-real-regression-2c5d4d0b/output/tests.json`、`repository_tests/test_matrix.json`、`audit_summary.json` | 程序版本`94909a7`：75/75核心PASS；当次2510项矩阵完整保留 |
| 当前全程序矩阵及失败分析 | `20261111T190337Z-saved-failure-evidence-audit-9b981c5c/output/repository_tests/test_matrix.json`、`tests.json`、`failure_regression.json` | `53d4541`：2512项记录，17项import FAIL及1项SKIP；110文件有命名证据。81项CPU矩阵69 PASS/12 SKIP，6项新增失败分析测试通过；未重跑原优化器或GPU回归 |
| 原生循环及独立对照失败 | 同一目录的`cycle_regression.json`、`native_control_regression.json` | 旧循环请求被拒绝，另一次13项安装CEC/mapping通过；受控native故障不妨碍其余真实测量，缺失对照不填数值；故障fixture不参与性能统计 |
| 可追溯的原生构建 | `20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/build_report.json` | gpuls/ABC的实际构建、hash与对应源码；Python后续版本须另行验证 |
| 搜索与公平比较 | `20261111T175518Z-acyclic-cohort-program-audit-333c8796/output/cohort/cohort.json`、`paired_report/paired_statistics.json`、`search_quality/` | 固定191输入、206次尝试；选定185 PASS、6 FAIL，其余714未记录；多开发版本分层保留，不能当作最终统一版本全量结果 |
| 保存映射的最大arrival链 | `20261111T160518Z-frozen-cohort-path-audit-0994e1ac/output/arrival_paths/summary.json` | 固定清单中116个成功设计的1870份映射、2218个引用，通过新CEC和数值/网表正文不变核验 |
| 路径变化与预测关联 | `20261111T182725Z-completed116-path-analysis-3a705938/output/path_changes/summary.json`、`retained_prefix_comparison.json` | 全部116设计只读复核，原39设计分析精确一致；不与191输入搜索快照拼成同一实验，不能证明唯一逻辑路径身份 |

上述搜索比较未显示稳定优于对等预算随机搜索的结果。真实CEC通过、代理预测准确、优化收益和跨设计泛化是不同结论，分别见[搜索分析](INTERIM_SEARCH_ANALYSIS.md)、[路径分析](PATH_CHANGE_ANALYSIS.md)和[限制报告](LIMITATIONS.md)。

## 全量状态与验收边界

905个唯一文件hash的组合AIG已完整盘点和暂存。两个统一版本分片的源码固定为`160b218`，可执行代码与上述`94909a7`回归相同，分别覆盖453/452输入；准备目录及后续状态见[运行记录](RUN_LOG.md)。准备清单和提交成功都不算benchmark完成。

最终比较须核对全部905个终态、同一来源与参数、各方法缺项、真实预算及失败请求。已有开发版本全量扫描和定向补验独立保存，不按结果优劣拼接为最终cohort。Slurm退出0也不等于全部case或历史程序PASS；应读取逐例和逐程序报告。当前不宣称全量验收完成。
