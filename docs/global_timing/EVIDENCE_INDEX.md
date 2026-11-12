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
| 真实核心及完整对照回归 | `20261111T182419Z-native-control-real-regression-2c5d4d0b/output/tests.json`、`repository_tests/test_matrix.json`、`audit_summary.json` | 程序版本`94909a7`：75/75核心PASS；当次2510项矩阵完整保留 |
| 统一版本c1355同配置复现 | `20261111T214331Z-consistent-c1355-exact-repeat-0e387139/output/verification.json`、`comparison/repeat_comparison.json`、`accounting.json` | 64397.0独立执行同一`160b218`优化器源码及完整参数；配置、全部数值/选择/最终选点、25项精确图结构一致，冻结输入未变。native ABC时间注释使字节不同；两模式无改善，复现不进入905输入性能cohort |
| 自然native失败后的完整案例尝试 | `20261111T184033Z-isolated-native-backend-retry-813f3b26/output/tests.json`、`suite/records.jsonl`及逐方法产物 | 75/75真实核心再次PASS；index108的native超时后继续执行，但整例7200.59 s FAIL。部分测量保留，面积约束未完成、joint未启动；不能作为完整比较 |
| 统一版本Back_End_With_Decode完整比较 | `20261111T184752Z-consistent-full-shard0-1568c1b9/output/suite/case_0108/report/comparison.csv`、`observations.csv`及原始CEC日志 | 4790.78 s完整PASS，无缺失方法；native及两最终点CEC通过。两梯度最终均来自修复，较原图慢15.122%，各四次直接梯度请求均FAIL；旧整例超时保留，路径审计尚在运行 |
| 统一版本mrc4自然native失败 | `20261111T184753Z-consistent-full-shard1-175169c2/output/suite/case_0611/native_abc/rewrite.log`及`report/method_failures.json`、`comparison.csv`、`observations.csv` | ABC reverse-level更新断言退出−6，无native指标；其余方法完成，整例PARTIAL。两梯度最终点来自修复，各四次直接梯度请求均失败；不丢弃输入、不补造缺失基线 |
| 统一版本codec自然native失败 | `20261111T184752Z-consistent-full-shard0-1568c1b9/output/suite/case_0392/native_abc/rewrite.log`及`report/method_failures.json`、`comparison.csv`、`observations.csv` | 原生rewrite断言退出−6，缺native指标、整例PARTIAL；两最终修复点CEC通过，较原图改善3.659%/2.371%，各四次直接梯度请求均FAIL。与mrc4断言位置相同，不宣称已证明相同深层根因 |
| 失败分析 | `20261111T190337Z-saved-failure-evidence-audit-9b981c5c/output/repository_tests/test_matrix.json`、`tests.json`、`failure_regression.json` | `53d4541`：2512项记录，17项import FAIL及1项SKIP；110文件有命名证据。81项CPU矩阵69 PASS/12 SKIP，6项新增失败分析测试通过；未重跑原优化器或GPU回归 |
| 原生循环及独立对照失败 | 同一目录的`cycle_regression.json`、`native_control_regression.json` | 旧循环请求被拒绝，另一次13项安装CEC/mapping通过；受控native故障不妨碍其余真实测量，缺失对照不填数值；故障fixture不参与性能统计 |
| 可追溯的原生构建 | `20261111T173355Z-acyclic-replay-tool-build-64cf764e/build/framework_tools/build_report.json` | gpuls/ABC的实际构建、hash与对应源码；Python后续版本须另行验证 |
| 全输入分区结构及当前程序矩阵 | `20261111T191749Z-full905-partition-certificates-f9e48d3a/output/partition_validation/summary.json`及逐例certificate、`repository_tests/test_matrix.json` | 全905输入的CPU strict partition独立结构检查PASS；完整owner、DAG与各端口AND距离。2515项程序记录、113文件有命名证据；86项CPU矩阵74 PASS/12 SKIP。没有rewrite、CEC、mapping或时序改善结论 |
| 搜索与公平比较 | `20261111T175518Z-acyclic-cohort-program-audit-333c8796/output/cohort/cohort.json`、`paired_report/paired_statistics.json`、`search_quality/` | 固定191输入、206次尝试；选定185 PASS、6 FAIL，其余714未记录；多开发版本分层保留，不能当作最终统一版本全量结果 |
| 保存映射的最大arrival链 | `20261111T160518Z-frozen-cohort-path-audit-0994e1ac/output/arrival_paths/summary.json` | 固定清单中116个成功设计的1870份映射、2218个引用，通过新CEC和数值/网表正文不变核验 |
| 路径变化与预测关联 | `20261111T182725Z-completed116-path-analysis-3a705938/output/path_changes/summary.json`、`retained_prefix_comparison.json` | 全部116设计只读复核，原39设计分析精确一致；不与191输入搜索快照拼成同一实验，不能证明唯一逻辑路径身份 |
| 统一版本首批路径采集 | `20261111T221147Z-consistent-path-audit-batch1-4a351887/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定89/89个设计PASS，1411份映射、1678个引用，精确benchmark目录与原清单一致；新CEC及数值/网表正文不变核验通过。benchmark源码`160b218`，与最终比较相同的64389 ABC；首批冻结cohort仍有816未记录，不是全905完成 |
| 统一版本第二批路径采集 | `20261111T225011Z-consistent-path-audit-batch2-7d3c9774/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定77/77设计PASS，1135份映射、1366个引用；精确清单、冻结hash及Slurm正常终态已核验。原166 PASS/739未记录保持不变 |
| 统一版本第三批路径采集 | `20261111T230414Z-consistent-path-audit-batch3-9f6ede4b/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定38/38设计PASS，539份映射、653个引用，包含Division；精确清单、冻结hash及Slurm正常终态已核验。原204 PASS/701未记录保持不变 |
| 统一版本第四批路径采集 | `20261111T231943Z-consistent-path-audit-batch4-c3f19ce7/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定37/37设计PASS，662份映射、773个引用；精确清单、冻结hash及Slurm正常终态已核验。原241 PASS/664未记录保持不变 |
| 统一版本第五批路径采集 | `20261111T233214Z-consistent-path-audit-batch5-33fcbf06/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定49/49个设计PASS，751份映射、898个引用，精确清单一致，新CEC及timing/网表正文不变；Slurm正常完成，选择/cohort哈希不变。原冻结290 PASS/615未记录保持不变，与首批及其他已分配批次互斥 |
| 统一版本第六批路径采集 | `20261111T235819Z-consistent-path-audit-batch6-0fe6209c/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定74/74设计审计PASS，1113份映射、1335个引用；精确清单、冻结hash及Slurm正常终态已核验。原364 PASS/541未记录保持不变 |
| 统一版本第九批路径采集 | `20261112T005832Z-consistent-path-audit-batch9-316b38d0/output/arrival_paths/summary.json`、`audit_summary.json`、`slurm_completion.json` | 固定52/52设计审计PASS，791份映射、947个引用；精确清单、冻结hash及Slurm正常终态已核验。原602 PASS/1 PARTIAL/302未记录保持不变 |

上述搜索比较未显示稳定优于对等预算随机搜索的结果。真实CEC通过、代理预测准确、优化收益和跨设计泛化是不同结论，分别见[搜索分析](INTERIM_SEARCH_ANALYSIS.md)、[路径分析](PATH_CHANGE_ANALYSIS.md)和[限制报告](LIMITATIONS.md)。

## 全量状态与验收边界

905个唯一文件hash的组合AIG已完整盘点和暂存。两个统一版本分片的源码固定为`160b218`，可执行代码与上述`94909a7`回归相同，分别覆盖453/452输入。偶数64399、奇数64397均已启动并分别通过75项真实核心测试；提交监测64396完成两次检查及提交后正常结束。目录及具体状态见[运行记录](RUN_LOG.md)。准备清单和提交成功都不算benchmark完成。

旧64340/64341自然结束时，其附属修正版64340.1/64341.1均被Slurm取消；分别保留207/186条终态，246/266个声明输入未形成终态。监测目录的`output/transition_64340_to_64399.json`、`transition_64341_to_64397.json`及各自校验和保存了两次衔接的sacct、记录hash、提交回执和启动测试。原run.json遗留的RUNNING不表示任务仍活跃，历史结果不并入最终统一版本比较。

最终比较须核对全部905个终态、同一来源与参数、各方法缺项、真实预算及失败请求。已有开发版本全量扫描和定向补验独立保存，不按结果优劣拼接为最终cohort。Slurm退出0也不等于全部case或历史程序PASS；应读取逐例和逐程序报告。当前不宣称全量验收完成。

统一版本路径审计64398、64400、64401、64402、64403、64404、64405分别选择89、77、38、37、49、74、151个有测量结果的设计，目录见[运行记录](RUN_LOG.md#统一版本路径证据分批采集)。七批精确目录互斥、并集515，benchmark源码均为`160b218`；第七批冻结514 PASS、1 PARTIAL、390未记录，包含mrc4有效测量。各批启动快照独立保留，不将未记录计数相加。首五批均已完成，共290设计、4498份映射、5368个引用；第六批随后完成，第七批仍在运行。选择清单见各自`output/path_audit_selection.json`，逐例报告见`output/arrival_paths/`；已分配不等于已完成。最终按精确benchmark目录关联，旧116设计报告不替代新结果。

第八批64406冻结550 PASS、1 PARTIAL、354未记录，排除前七批515项后新增36项，首个为index108；八批精确目录互斥、并集551。准备/分析源码`e9016eb`，选择清单见`20261112T004638Z-consistent-path-audit-batch8-b7761547/output/path_audit_selection.json`，SHA256为`fb2104a3bcbc417311185ed0aa6ea890422824dd34bdfdc9ad2de63fc38a7cbf`。来源核验通过、审计仍在运行，分配范围不增加已通过覆盖数。

第九批64407冻结602 PASS、1 PARTIAL、302未记录，排除前八批551项后新增52项，首个为index322；九批精确目录互斥、并集603。准备/分析源码`46cabc2`，选择清单为`20261112T005832Z-consistent-path-audit-batch9-316b38d0/output/path_audit_selection.json`，SHA256为`db9115336fdf856882fe73df5d3b30e0c093f503b0ec044badd1105541dc2677`。第九批随后完整结束，见上表；晚于冻结的codec结果由后续批次纳入。

第十批64408冻结636 PASS、2 PARTIAL、267未记录，排除前九批603项后新增35项，包含codec有效测量；十批精确目录互斥、并集638。准备/分析源码`b6bc67a`，选择清单为`20261112T010854Z-consistent-path-audit-batch10-1b537bfa/output/path_audit_selection.json`，SHA256为`c51e52282c6888eda0dcb1d33c839b0c0c36f7001012fb8ddd549455553ce157`。来源核验通过，审计仍在运行。

第六、第九批完成后，已确认完成的首六批加第九批共416个互斥设计、6402份映射、7650个成功引用。第十一批64481已启动，准备/分析源码`e1880f0`，排除前十批638个目录；新增范围仍待冻结清单，提交回执见`20261112T012322Z-consistent-path-audit-batch11-69a57845/input/submission.json`。
