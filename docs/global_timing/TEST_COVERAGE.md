# 测试证据与未覆盖部分

测试按源码快照记录，不能把import、编译或`--help`成功解释为完整功能通过。全量benchmark仍在执行；本文件不是整体验收声明。

## 核心与实际工具

`20261111T134714Z-final-feature-e2e-retry-aa36f9d5/output/tests.json`记录46项核心测试全部通过，包含MFFC参照、冲突与空候选、有限差分梯度、两种目标、失败CEC、上下文cut证明、真实GPU候选/replay、常量导出、Student推理缓存、Liberty端点/路径及预测来源审计。其c1355完整比较通过，div超时；核心测试通过不替代每个benchmark的完成状态。随后梯度重复计算修复的回归单独记录，不把旧测试冒充新源码验证。

重复计算修复后，`20261111T143814Z-profiled-scale-e2e-retry-8cee3d62/output/tests.json`再次记录46/46真实核心PASS（64341.4）；其大设计完整比较仍在进行。CPU保存输入一致性验证见64363及[性能检查](PERFORMANCE_AUDIT.md)。

`b288fe1`快照在`20261111T150227Z-bounded-arrival-real-regression-b71177d2/output/tests.json`通过50/50真实核心测试（64340.6）。新增内容为有界arrival反传的3项测试，以及真实gpuls在`CUDA_VISIBLE_DEVICES=-1`时明确返回正错误码并报告CUDA初始化失败的负向测试。测试使用64365重新构建且全部原生源码匹配的二进制；CPU构建阶段的39 PASS/11 SKIP由此得到独立真实工具补验。此项不证明此前ALU原生崩溃的根因已找到。

64366的新增汇总回归为57项CPU矩阵：46 PASS、11个真实工具入口SKIP。7项新测试验证不按数值选重跑、待补验不伪装全覆盖、重复/错误身份拒绝、JSONL不完整尾部、冻结快照不随live append变化、快照/结果/选择/版本篡改检测，以及输入/工具/预算不匹配拒绝。随后实际读取104输入、106次尝试，预测来源审计错误0；这些是保存结果检查，不增加新的时序测量。

`0e1cbd9`的未变化入口缓存增加3项测试。64379的60项CPU矩阵为49 PASS/11 SKIP；c1355、des_perf、div、voter共32个实际coordinate的全部候选成本及4份完整选择与保存参照精确相同。64340.8（`20261111T160210Z-cached-coordinate-real-regression-e46fd0d0`）随后通过60/60真实核心测试，c1355两模式完整比较通过。缓存只改变独立region基线的计算复用，未更改目标、候选或预算，完整证据见[性能检查](PERFORMANCE_AUDIT.md)。

`f29a906`新增工具时限声明回归，覆盖旧300 s默认值、显式900 s、配置与声明不一致及非法声明。64383的61项CPU矩阵为50 PASS/11 SKIP，并用新审计代码成功读取旧104/125输入清单。64340.9随后通过61/61真实核心测试，已完成显式900 s配置的c1355完整比较；同一suite的Division仍须等待自己的终态记录。

64383还实际复核缓存版c1355的22份映射（25个观察引用、18种AIG字节），全部重新CEC、确认全部PO/area/global与网表正文相同，并通过最大arrival链检查。此前与缓存版c1355的完整数值、选择及精确图结构一致；native ABC时间注释仍导致字节不同。它不把CPU矩阵中的11项SKIP当作真实GPU测试PASS。

`01889c4`新增6项跨配置路径分析测试。64386的67项CPU矩阵为56 PASS/11 SKIP，检查PO/PI/相位与并列集合、零delay、失败间隔、保存最终选点、缺失/重复/虚构引用、报告优先级、精确benchmark身份，以及原始CEC/路径/netlist被改动的拒绝。39设计611份真实保存映射的只读复核错误0；未再次执行映射。新增分析未改变优化器或原生工具，原61项真实核心证据仅按源码字节匹配复用，不把11项CPU SKIP改为本次GPU通过。

`ab863e8`再增加2项路径与评估前预测关联检查：拒绝错误输入/索引、重复观察、当前标签泄漏及错配delta；分组统计按设计等权，零delay和实际未变样本不冒充方向准确性。64388的69项CPU矩阵为58 PASS/11 SKIP；对同一39设计新增关联时，旧路径分析全部内容保持一致。此后`d2065ab`修改了原生rewrite的传递依赖检查，其新构建和真实工具回归单独记录如下；不能沿用旧原生源码的PASS。

原生修改的独立验证现已完成：64389重建工具，实际链接solver检查三节点环、两个fanin的传递依赖、补相及100001级合法别名链，并通过两次常量输出CEC。64341.6随后记录69/69真实核心PASS，完整c1355和index197比较PASS；同一20项旧循环选择明确CYCLE_REJECT，另一次13项修复安装CEC/mapping PASS。c1355全部数值、选择与图结构保持不变。复核记录为`output/cycle_regression.json`，详细来源见[NATIVE_REPLAY_CYCLES.md](NATIVE_REPLAY_CYCLES.md)。

`20261111T132425Z-student-feature-regression-9184e32c`的CPU矩阵为32 PASS、10个真实工具入口SKIP；不能把这些SKIP算作CPU任务完成了GPU验证。其保留输入复核另行重算c1355的417条、div的122,869条原实现/候选特征及预测，与旧25维提取流程保存的全部score记录完全相同。没有训练模型或更换权重。

`20261111T130741Z-selection-cost-regression-3cba9919`检查候选归一化、24组标量/批量坐标成本及可获得的旧独立选择。缺失完整旧选择的div和缺失旧完整catalogue的hyp明确为`NO_SAVED_REFERENCE`，不冒称做过这些完整对照。

路径补充审计在`20261111T125125Z-path-provenance-regression-1876ee41`完成：8设计、124份映射、168个成功观察引用，全部重新CEC及检查数值/网表正文不变，重构当前映射内的最大arrival链。范围和节点身份限制见模型契约。

## 全程序矩阵

最新已完成矩阵`20261111T175518Z-acyclic-cohort-program-audit-333c8796`（源码`cefb56a`，64390）覆盖2,510项源码/入口：1,613项语法PASS、1,395个编译translation units、142项import PASS、17 FAIL、1 SKIP、58项CLI smoke PASS。106个文件有来源字节匹配的命名功能证据；这不是行覆盖率，其余项目明确不宣称逐文件功能覆盖。新增的证据关联入口实际为rewrite头文件/实现、candidate与pipeline附上64341.6的循环拒绝/修复记录，且逐文件字节相同。该任务复用已执行的69项真实核心及原生CPU证据，没有再次执行GPU测试或mapping；6m50s、MaxRSS539592 KiB。

此前`20261111T173748Z-acyclic-replay-real-regression-e1510460`（源码`20e7aba`，原生代码与`d2065ab`相同，64341.6）的同规模矩阵完整保留。同一运行有69项真实核心测试、两个完整benchmark和实际循环拒绝/修复检查；原生CPU证据来自相同源码的64389。9m27s、MaxRSS810424 KiB，旧原生源码PASS不冒充本次验证，历史依赖缺项仍明确保留。

此前64388（源码`ab863e8`）的2,510项矩阵、105个有命名功能证据的文件和39设计预测/路径关联完整保留。它没有验证后续原生修改。

此前64386的同规模矩阵、67项CPU测试及首次39设计路径分析完整保留；不回写其结果来表示后续代码已经执行。

此前`20261111T162905Z-timeout-cohort-program-audit-d2c73954`（源码`b6c5838`，可执行代码与`f29a906`相同）的2,507项矩阵、102个有命名功能证据的文件和实际61项核心来源记录完整保留。同时冻结137输入/145次尝试，核验实际900 s案例及旧300 s案例的时限声明与配置一致；未新增rewrite或mapping。

此前`20261111T160210Z-cached-coordinate-real-regression-e46fd0d0`（源码`0e1cbd9`）的同规模矩阵、60项真实核心测试、c1355双目标比较、writer/Gate1工具/Gia检查完整保留，step 64340.8退出0。历史依赖缺项不因Slurm成功而改为PASS。

此前`20261111T152720Z-benchmark-cohort-audit-38c31f2a`的2,507项矩阵及57项CPU回归独立保留；该版本的汇总与审计证据只在相应源码字节匹配时复用。

此前`20261111T150227Z-bounded-arrival-real-regression-b71177d2`（源码`b288fe1`）的2,504项矩阵、100个有命名功能证据的文件与50项真实核心测试保留；同一运行已完成c1355双目标比较及writer/Gate1工具/Gia检查。后续独立选择剖析入口的参数变化另行保存实际验证，不把旧矩阵当作后续所有文件的功能证明。

此前`20261111T142809Z-native-aware-program-matrix-571ab8dd`（源码`157987c`）的2,501项矩阵、97个有命名功能证据的文件及CPU核心36 PASS/10 SKIP仍完整保留。源码有后续修改的模块使用新回归，不回写旧报告。两份矩阵中的编译TU都是按文件去重后的计数，不能把历史变体的1,379个TU再次加到总计1,395上。

历史原生程序已独立重建：`64359`保存5个ABC overlay及Gate1的完整编译记录；首次mapped-BLIF CEC入口缺少加载library而失败，不能记为不等价。`64360`复用hash核对的二进制，在新目录重新检查后5/5通过，包括reachable/unreachable pair、开启/关闭mapper观察的网表正文与pair一致，以及local arrival的109.239990/106.219986 ps对照。该reachable smoke查询名字`pi04`→`po11`，不是历史R9机制分析的`pi03`→`po11`，两者不混用。

`64340.4`实际运行重建的Gate1。原始输入为502-AND的历史c1355；AC/AD/BC/BD输出分别504/503/505/504 AND，全部与原输入CEC等价，两个请求均`installed=true`，provenance与保存的历史JSON完全相同，输出精确图结构hash相同。`NO_OP_FULL_CORRESPONDENCE`仍区分于物理替换；不能据此宣称一般的独立decision安装性质。

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
