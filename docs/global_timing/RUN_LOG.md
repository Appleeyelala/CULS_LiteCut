# 新开发阶段的运行记录

所有目录均在 `/data/wushangge/CULS_LiteCut/runs/`；原历史目录及 `evidence/` 未改写。下列“完成”仅指对应运行，不代表整体用户目标完成。

| Slurm ID | 目录后缀 | 结果与原因 |
|---|---|---|
| 64322 | `20261111T102740Z-historical-baseline-cab97f65` | FAIL：新编排脚本传错Gate3D raw trace目录；Gate3A–D完成 |
| 64323 | `20261111T102850Z-historical-baseline-826120e0` | FAIL：历史ABC通用read不识别无后缀Liberty别名；Gate3E完成 |
| 64324 | `20261111T103029Z-historical-baseline-51d97b14` | COMPLETED：Gate3A–F复跑，独立严格CEC通过，四组历史数值复现 |
| 64325 | `20261111T103249Z-framework-build-5e8e7b23` | FAIL：新增字段打乱C++旧聚合初始化顺序；已将字段移到末尾 |
| 64327 | `20261111T103732Z-framework-build-41106501` | FAIL：CULS/ABC均构建成功，测试类路径属性覆盖了unittest.run；已修复，二进制按hash复用 |
| 64328 | `20261111T104247Z-framework-regression-5c16847e` | COMPLETED：14项测试全部通过，包括真实rewrite/CEC/Liberty、326候选批量一致性 |
| 64329 | `20261111T104812Z-gradient-regression-826bf929` | COMPLETED：18项测试全部通过，新增有限差分梯度、联合选择与双目标模式 |
| 64330 | `20261111T105213Z-c1355-global-e2e-7961ed58` | 首次完整闭环产生summary；每种搜索24次预算；贪心与独立region有REVERTED，后续增加显式修复，不把部分安装当作成功 |
| 64331 | `20261111T105804Z-c1355-repair-e2e-e04ad899` | COMPLETED：20项回归；修复请求计预算后，面积约束搜索未优于baseline；负结果保留 |
| 64332 | `20261111T105821Z-benchmark-inventory-114a3573` | COMPLETED：905个组合binary AIG逐文件暂存/hash，均唯一；总20,561,771 AND |
| 64333 | `20261111T110315Z-sparse-regression-79327bd5` | COMPLETED：20项稀疏传播与真实工具回归 |
| 64334 | `20261111T111356Z-benchmark-scale-pilot-d516a378` | COMPLETED：22项回归；c1355/c5315/c7552/jpeg_hm_decode四个8预算完整比较 |
| 64335 | `20261111T111703Z-repository-test-matrix-a85876d4` | COMPLETED：2476项源码/程序清单；首轮导入探针过度拦截/dev/null，结果不替代功能测试 |
| 64336 | `20261111T111908Z-integrity-and-scale-regression-7f453a81` | COMPLETED：26项回归，增加负向CEC、ABC失败、原cut真值及独立region快捷计算对照 |
| 64337 | `20261111T112204Z-concurrent-suite-pilot-64109582` | COMPLETED_WITH_FAILURES：c1355通过；中位电路因把功能支持cut当结构边界而误拒绝 |
| 64338 | `20261111T112224Z-repository-import-regression-a19a25d0` | COMPLETED：修正/dev/null与工具搜索路径；83个import PASS，18 FAIL、27受探针约束SKIP；后续允许只读lscpu/fc-list查询 |
| 64339 | `20261111T112634Z-support-cut-regression-4ac56f58` | COMPLETED：29项回归；c1355与中位电路并发比较完成，发现8条score涉及功能支持最小化 |
| 64340/64341 | `20261111T112917Z-full-benchmark-shard0-864a4c8b` / `20261111T112919Z-full-benchmark-shard1-5bb41012` | 旧版本全量扫描进行中；出现命名接口、cut上下文/BDD规模及常量输出问题；不作为最终全量验收 |
| 64342 | `20261111T113159Z-cross-design-integrity-audit-0f3668f1` | COMPLETED：独立读取已导出cut，定位边界自由变量真值差异 |
| 64343 | `20261111T113648Z-context-proof-regression-525333c5` | COMPLETED：31项CPU测试（8个GPU入口SKIP）、4项真实ABC回归；六个保留案例10,281个cut全部证明；两个旧no-op仍因常量翻转不等价 |
| 64344 | `20261111T113959Z-constant-safe-tool-build-5e7be150` | BUILT_NOT_RUNTIME_VALIDATED：修复常量导出/未初始化状态后重新编译；尚待GPU回归 |
| 64345 | `20261111T114605Z-native-writer-and-import-tests-71839ae5` | COMPLETED：链接实际solver对象，连续两次Reset/Reorder/WriteAig的常量CEC与mapping通过；逐程序矩阵完成，导入缺失依赖仍显式记录 |
| 64346 | `20261111T115338Z-imported-native-and-cli-tests-0dd6d0c7` | COMPLETED：Gate1 io_signature、候选选择器和ABC四项Gia原生单元测试全部通过；未下载依赖 |
| 64347 | `20261111T115530Z-ci-cpu-and-existing-reports-bdf90782` | FAIL：CI父进程提前导入绘图库使子进程继承不兼容MKL线程层；调整导入顺序，保留失败日志 |
| 64348 | `20261111T115853Z-ci-cpu-and-reports-regression-c08b3bd8` | COMPLETED：32项CPU矩阵（23 PASS、9个GPU入口SKIP）；6组已保存实验生成图表；重复运行数值/选择一致，Native ABC输出字节hash因时间注释不同，进一步按精确图结构区分 |
| 64340.0 | `20261111T121113Z-corrected-tools-regression-538f79e0` | COMPLETED_WITH_EXPLICIT_OMISSIONS：独立step在原分配内运行，33项真实核心测试全部通过，c1355双目标8预算完成、都未改善基线；原生writer/Gate1/Gia及2486项矩阵完成；重复运行选择/数值/结构hash一致，ABC注释导致字节hash不同 |
| 64341.0 | `20261111T121235Z-corrected-cross-design-pilot-13f96501` | COMPLETED：33项真实回归及8/8设计完整比较通过；包括六个此前失败设计、c1355和中位设计；不是代表性随机抽样 |
| 64340.1 / 64341.1 | `20261111T121807Z-corrected-full-shard0-93ae61a2` / `20261111T121809Z-corrected-full-shard1-29165762` | RUNNING：修复后的全905输入双分片；独立step共享已有分配，旧batch未取消；各step先通过33项真实回归 |
| 64349 | `20261111T122037Z-historical-analysis-regression-324c8936` | COMPLETED：四个R9分析入口用隔离历史输入执行通过，依赖文件hash未改变；2487项程序矩阵合入来源匹配的历史Gate3/R9功能证据，120 import PASS、17 FAIL、1 SKIP |
| 64350 | `20261111T122536Z-paired-results-regression-b1f525fb` | COMPLETED：35项CPU测试（26 PASS、9个GPU入口SKIP）；8设计实际配对统计与修复版c1355重复比较通过，包含最终选点检查 |
| 64351 | `20261111T124031Z-arrival-path-observer-build-1ab72688` | COMPLETED：构建可选Liberty最大arrival路径observer；实际ABC路径重构及插桩前后网表/全部PO/面积一致性测试通过；GPU测试由64340.2补齐 |
| 64340.2 | `20261111T124507Z-arrival-path-full-ci-6caaeabb` | COMPLETED_WITH_EXPLICIT_OMISSIONS：38项真实核心回归全部通过，c1355双目标smoke、原生writer/Gate1/Gia及2492项程序矩阵完成；缺失历史训练环境明确保留 |
| 64352 | `20261111T124509Z-saved-selection-path-analysis-2bda1d0e` | FAIL：分析wrapper误按搜索method定位复用的no-op文件；原搜索结果未改动 |
| 64353 | `20261111T124930Z-saved-path-analysis-regression-c851ab7f` | FAIL：首次修复假设了错误的replay路径层数；随后改为按实际observation路径和输入hash定位，并增加回归 |
| 64354 | `20261111T125125Z-path-provenance-regression-1876ee41` | COMPLETED：39项CPU矩阵（29 PASS、10 SKIP）；8设计168个成功观察引用对应124份映射，重新CEC、全部PO/global/area、网表正文与逐arc路径重构全部通过 |
| 64355 | `20261111T130741Z-selection-cost-regression-3cba9919` | COMPLETED：41项CPU矩阵（31 PASS、10 SKIP）；c1355/des_perf/div的完整catalogue均与保存结果一致，24组坐标选项成本与标量参考逐位相同；c1355/des_perf完整选择一致。hyp归一化完成但无旧完整catalogue可比；div无旧完整选择可比，均明确标记 |
| 64340.3 | `20261111T131145Z-selection-scale-e2e-f53fa37e` | COMPLETED_WITH_FAILURES：41项真实核心回归通过；div/hyp均在1800 s单例限制超时，c1355完整比较通过；Step退出0不代表所有case通过 |
| 64356 | `20261111T131747Z-updated-program-matrix-4d80fce4` | COMPLETED_WITH_RECORDED_OMISSIONS：2494项矩阵；126 import PASS、17 FAIL、1 SKIP、51 CLI smoke PASS，45个文件有来源匹配的命名功能证据 |
| 64357 | `20261111T132425Z-student-feature-regression-9184e32c` | COMPLETED：42项CPU矩阵（32 PASS、10 SKIP）；19维快捷提取重新计算c1355全部417条、div全部122,869条score，特征和预测及其记录与旧结果完全一致；div重算31.40 s |
| 64341.2 | `20261111T132506Z-student-native-regression-7773e417` | COMPLETED_WITH_RECORDED_OMISSIONS：19维快捷提取的42项真实核心回归全部通过；2495项程序矩阵完成，127 import PASS、17 FAIL、1 SKIP、51 CLI smoke PASS |
| 64358 | `20261111T134520Z-search-quality-audit-0b22b5c7` | COMPLETED：46项CPU矩阵（36 PASS、10 SKIP）；冻结46个已记录输入（44 PASS、2 timeout），审计44个实际搜索的预测来源及最终选点来源，错误0，保存CSV/PNG/PDF；不调用rewrite/mapping |
| 64341.3 | `20261111T134714Z-final-feature-e2e-retry-aa36f9d5` | COMPLETED_WITH_FAILURES：46项真实核心测试全部通过；19维快捷提取下div仍在1800 s超时，c1355通过且两模式均未改善360.111877 ps / 513.216003 area；旧失败保留 |
| 64359 | `20261111T141136Z-native-variant-regression-e8da9407` | FAILED：5个历史ABC变体及Gate1全部编译成功；新测试驱动未在新进程载入library，mapped BLIF无法读入，5项CEC检查失败；不是已经证明不等价。Gate1仅记录BUILT_NOT_GPU_VALIDATED |
| 64360 | `20261111T142227Z-native-variant-validation-46576b83` | COMPLETED：修正测试驱动的library加载，按源文件/二进制hash复用5个已建变体，5/5实际命令测试通过；不改旧日志或重复编译 |
| 64340.4 | `20261111T142618Z-gate1-native-replay-validation-9a03312d` | COMPLETED：重建Gate1的normalize及AC/AD/BC/BD四组真实GPU replay通过；原始502-AND输入CEC、两decision完整安装、历史provenance及序列化图结构全部一致 |
| 64361 | `20261111T142532Z-saved-gradient-profile-5baf3f8e` | COMPLETED：保留div输入的首轮面积约束校准及完整梯度trace精确复现；诊断4183层DAG、51140个decision的实际CPU开销，不执行rewrite/mapping；初版误把mutable run.json列入input hashes，后续排除该项 |
| 64362 | `20261111T142809Z-native-aware-program-matrix-571ab8dd` | COMPLETED_WITH_RECORDED_OMISSIONS：46项CPU矩阵36 PASS/10 SKIP；2501项程序清单、1395个编译TU、133 import PASS/17 FAIL/1 SKIP、55 CLI smoke PASS；97文件附来源匹配的命名功能证据 |
| 64363 | `20261111T143301Z-gradient-overhead-regression-62411fb5` | COMPLETED：46项CPU矩阵36 PASS/10 SKIP；移除逐层重复扫描及逐decision张量转换后，保留div首轮校准的全部loss/参数及完整40步梯度trace精确一致；初始化2.60 s、校准99.05 s、提案105.82 s，均含profiler |
| 64341.4 | `20261111T143814Z-profiled-scale-e2e-retry-8cee3d62` | COMPLETED_WITH_FAILURES：性能修复后的46项真实核心测试全部通过；div在2022.49 s、c1355在63.85 s完成完整比较并PASS，hyp在3600.94 s超时，仍每种搜索8次预算。本快照尚未采用有界arrival求导。旧1800 s失败保留；step退出0不代表hyp通过 |
| 64364 | `20261111T144801Z-bounded-arrival-regression-0a0473d0` | COMPLETED：49项CPU矩阵39 PASS/10 SKIP；显式一阶arrival反传与原Torch参照及有限差分通过；保存div全部校准loss/参数、完整40步梯度trace精确一致。含profiler校准21.13 s、提案49.05 s；未执行rewrite/mapping |
| 64340.5 | `20261111T145230Z-alu-replay-diagnosis-ecc3c927` | COMPLETED_DIAGNOSTIC / PASS_NOT_REPRODUCED：同一旧gpuls/ALU输入的GDB重放未复现SIGSEGV，零rewrite并CEC/mapping通过；原全量失败保留，根因尚未证明 |
| 64365 | `20261111T145549Z-checked-cuda-tool-build-2623c263` | BUILT_NOT_RUNTIME_VALIDATED：50项CPU矩阵39 PASS/11 SKIP；CULS增加CUDA初始化返回值检查并初始化heap输出变量，独立ABC observer保持原源码；两工具重新构建完成，实际工具回归单独记录 |
| 64340.6 | `20261111T150227Z-bounded-arrival-real-regression-b71177d2` | COMPLETED_WITH_EXPLICIT_OMISSIONS：`b288fe1`的50项真实核心测试全部通过，含无GPU初始化错误检查；c1355完整双目标8预算比较完成，两模式仍为360.111877 ps / 513.216003 area、未改善。实际writer/Gate1工具/Gia检查通过；2504项程序矩阵完成，136 import PASS/17 FAIL/1 SKIP、56 CLI smoke PASS、1395个编译TU，100文件附来源匹配的命名功能证据 |
| 64340.7 | `20261111T151256Z-bounded-arrival-failed-case-retry-9d4f505b` | COMPLETED_WITH_FAILURES：先通过50项真实核心测试；ALU、voter、ACE sqrt、OpenABC div、dft、idft在109.86/2151.71/924.92/1393.41/1267.79/1246.78 s完整PASS，原SIGSEGV根因仍未证明。59（OpenABC hyp）在3600.74 s超时，已完成area但未完成joint；每种搜索8次预算。Slurm 2h58m52s、MaxRSS2904620 KiB；原失败/hash保留，未取消其他任务 |
| 64366 | `20261111T152720Z-benchmark-cohort-audit-38c31f2a` | COMPLETED_WITH_EXPLICIT_OMISSIONS：57项CPU矩阵46 PASS/11 SKIP；冻结106次尝试、104个输入，当前选择97 PASS/7 FAIL/801未记录，原9次失败全部保留。预测来源审计错误0，配对比较与版本分层完成；c1355跨性能修复版本的数值/选择/图结构一致。2507项程序矩阵、139 import PASS/17 FAIL/1 SKIP、57 CLI smoke PASS；未运行新rewrite/mapping |
| 64367 | `20261111T153819Z-voter-coordinate-profile-bf87713a` | COMPLETED：voter完整catalogue与保存结果一致；一个region含7007个decision、1001个边界入口、431860项入口关联。单个coordinate两个候选的批量/标量成本精确相同，记录重复遍历全部入口的cProfile；未执行完整独立选择、rewrite或mapping，不替代live E2E |
| 64379 | `20261111T155547Z-cached-coordinate-regression-0fa1eca4` | COMPLETED_WITH_GPU_OMISSIONS：`0e1cbd9`的60项CPU矩阵49 PASS/11 SKIP；c1355/des_perf/div/voter各8个coordinate的全部成本与标量参照精确相同，4份完整独立选择与保存请求一致。完整选择0.142/16.192/30.030/44.233 s；不执行rewrite/mapping，不将其计时与旧共享E2E相除作加速比 |
| 64340.8 | `20261111T160210Z-cached-coordinate-real-regression-e46fd0d0` | COMPLETED_WITH_EXPLICIT_OMISSIONS：缓存版60项真实核心测试全部通过，c1355双目标8预算比较完成，两模式仍为360.111877 ps / 513.216003 area、CEC PASS；writer/Gate1工具/Gia及2507项程序矩阵完成，139 import PASS/17 FAIL/1 SKIP、57 CLI smoke PASS、102文件有命名功能证据。Slurm 9m09s，MaxRSS457464 KiB，共享分配计时 |
| 64381 | `20261111T160518Z-frozen-cohort-path-audit-0994e1ac` | COMPLETED_WITH_INCOMPLETE_BENCHMARK_COVERAGE：固定125输入清单中116/116个成功设计通过路径审计，1870份映射、2218个引用，全部新CEC及数值/网表正文一致。原9 FAIL/780未记录保留；不重新搜索。2h16m46s、MaxRSS791656 KiB；不是全905完成 |
| 64382 | `20261111T160933Z-division-cec-diagnosis-e5e74bdc` | DIAGNOSIS_COMPLETED：对index147 Division保存的原始/native rewrite AIG，核对同一旧ABC及输入hash；原`cec -n`在外部1800 s限制下于306.23 s取得等价性证明，`cec -n -p`在300.06 s超时。诊断不改变原300 s FAIL，没有rewrite/mapping，不代表完整benchmark通过 |
| 64341.5 | `20261111T160711Z-cached-coordinate-failed-case-retry-963fafa6` | COMPLETED_WITH_FAILURES：缓存版60项真实核心测试全部通过；78（OpenABC sqrt）和125（DF_pipeline）在873.49/118.20 s完整PASS，22（ACE hyp）在3600.50 s超时，已进入joint但没有完整终态结果。每种搜索8次预算；Slurm 1h17m07s、MaxRSS2830788 KiB。旧记录/hash保留，不取消其他任务 |
| 64383 | `20261111T161958Z-tool-timeout-regression-18c44f98` | COMPLETED_WITH_GPU_OMISSIONS：`f29a906`增加显式工具时限来源检查，61项CPU矩阵50 PASS/11 SKIP；旧104/125输入冻结清单均仍有效。缓存版c1355的25个成功观察引用对应22份映射、18种AIG字节，重新CEC/全部PO与area/global/网表正文及路径检查通过。缓存前后完整数值/选择/图结构相同；native ABC时间注释使文件字节不同 |
| 64340.9 | `20261111T162117Z-declared-timeout-division-retry-50d6b379` | COMPLETED_WITH_FAILURES：61项真实核心PASS；c1355在70.68 s完整PASS。Division147在7200.51 s整例超时，虽然native CEC曾通过，也未完成全部搜索。每搜索8次预算、工具900 s，原300 s失败和306.23 s诊断独立保留。2h01m56s、MaxRSS1399728 KiB；不把局部证书当整例完成 |
| 64384 | `20261111T162905Z-timeout-cohort-program-audit-d2c73954` | COMPLETED_WITH_EXPLICIT_OMISSIONS：冻结145次尝试、137输入，选择130 PASS/7 FAIL/768未记录；包括实际900 s配置的c1355及旧300 s运行，时限/工具/输入/预算/快照核验通过。2507项程序矩阵完成，139 import PASS/17 FAIL/1 SKIP、57 CLI smoke PASS、1395个编译TU、102文件有命名功能证据。代码与`f29a906`相同，复用字节匹配的61项真实核心及其他命名测试，不新增rewrite/mapping |
| 64385 | `20261111T163916Z-frozen137-search-analysis-29b53754` | COMPLETED_WITH_EXPLICIT_OMISSIONS：分析64384同一冻结清单中的130个成功设计，预测来源审计错误0；面积约束/联合模式的对等预算随机比较为28/59/43与33/48/49胜/平/负。保留7项遗漏及5种源码分层；无稳定优势结论。生成CSV/图/来源记录，不重新拟合、rewrite或mapping |
| 64386 | `20261111T165927Z-saved-path-change-audit-c21e0dbe` | COMPLETED_WITH_EXPLICIT_OMISSIONS：`01889c4`的67项CPU矩阵56 PASS/11 SKIP；只读复核39设计611份保存映射、728个成功观察引用，错误0。冻结125输入清单，77个成功设计的路径报告尚未包含在该快照，9个非PASS、780未记录。27/39个original→no-op的代表PI/PO对已变化。2510项程序矩阵完成：142 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、1395个编译TU、105个文件有来源匹配命名功能证据。6m07s、MaxRSS513212 KiB；没有新增rewrite/mapping |
| 64387 | `20261111T170329Z-backend-catalogue-diagnosis-dcbb55ab` | COMPLETED_DIAGNOSTIC：只读处理index108 Back_End_With_Decode旧1800 s超时后保存的1,638,237条cut；当前catalogue构建7.43 s、22,477 executable decisions、43,734 regions。512条旧/新normalizer结果精确相同；旧完整catalogue不存在，明确NO_SAVED_REFERENCE。Slurm 34 s、MaxRSS3312724 KiB；未执行rewrite/mapping，不替代完整E2E |
| 64388 | `20261111T171234Z-prediction-path-regression-6ea0c287` | COMPLETED_WITH_EXPLICIT_OMISSIONS：`ab863e8`的69项CPU矩阵58 PASS/11 SKIP；严格复用64386的39设计，原始路径统计完全相同，评估前预测与精确观察的关联审计错误0。critical PO变化组归一化MAE为0.155/0.186，代表PI/PO不变组为0.027/0.031；仅为描述性关联。2510项程序矩阵完成，142 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、105个文件有命名功能证据。6m04s、MaxRSS457688 KiB |
| 64340.10 | `20261111T172127Z-selected-replay-crash-diagnosis-1e0694ff` | COMPLETED_DIAGNOSTIC / NATIVE_SIGNAL_REPRODUCED：index197的20项原选择、owner表、输入及旧二进制在GDB下重现SIGSEGV，回溯为反复调用CPUSolver::Fanin；没有新搜索或mapping |
| 64340.11 | `20261111T172508Z-replay-alias-chain-inspection-1c801ee8` | FAILED_DIAGNOSTIC：成功到达旧二进制Fanin入口，但集群GDB不支持Python脚本，未取得内存快照；环境错误与日志独立保留 |
| 64340.12 | `20261111T172704Z-replay-alias-memory-inspection-dd0019ee` | COMPLETED_DIAGNOSTIC / ALIAS_CYCLE_OBSERVED：GDB原生命令读取实际内存，确认3个不同region的原始root 6868→10577→8953→6868形成环，旧trace均ACCEPTED；100个活跃引用进入环。详见NATIVE_REPLAY_CYCLES.md；无CEC/mapping标签 |
| 64389 | `20261111T173355Z-acyclic-replay-tool-build-64cf764e` | BUILT_AND_CPU_VALIDATED：`d2065ab`加入安装前传递依赖检查和有界别名解析；69项CPU矩阵58 PASS/11 SKIP，工具构建及链接真实solver的环/长链/补相/常量输出CEC回归通过。2m55s、MaxRSS2594464 KiB；GPU验证单独记录于64341.6 |
| 64341.6 | `20261111T173748Z-acyclic-replay-real-regression-e1510460` | COMPLETED_WITH_EXPLICIT_OMISSIONS：69/69真实核心PASS，c1355/index197完整双目标比较分别56.80/165.97 s PASS。旧20项循环选择保持FAIL，明确CYCLE_REJECT root10577；另一次13项修复安装CEC/mapping PASS，未用no-op fallback。c1355完整数值/选择/图结构与保存参照一致。2510项程序矩阵完成：142 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、1395编译TU、106文件有命名功能证据。9m27s、MaxRSS810424 KiB |
| 64341.7 | `20261111T174715Z-acyclic-terminal-case-retries-7cce9a09` | RUNNING：新无环工具按108/122/185/22顺序补验。108在1337.65 s FAIL：准备与no-op通过，native CEC在900.21 s超时，后续搜索未执行；不是不等价证明。122（Core）和185（Ibtida_top）在873.16/1098.06 s完整PASS，22仍在运行。仍每搜索8次预算、7200 s单例/900 s工具限制；原失败与输入身份冻结，不重复当时live59/147、不取消旧allocation。源码`ee91a33`与已验证原生`d2065ab`相同 |
| 64390 | `20261111T175518Z-acyclic-cohort-program-audit-333c8796` | COMPLETED_WITH_EXPLICIT_OMISSIONS：冻结206次尝试、191输入，选择185 PASS/6 FAIL/714未记录；原17 FAIL及1 PARTIAL保留。预测来源审计错误0；对等预算随机比较为36/91/58与48/77/60胜/平/负，未显示稳定优势。2510项程序矩阵完成：142 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、1395编译TU、106文件有来源匹配命名证据，实际附上新循环回归。6m50s、MaxRSS539592 KiB；无新增rewrite/mapping |
| 64391 | `20261111T180508Z-hyp-saved-gradient-profile-8d6186eb` | COMPLETED：只读旧3600 s超时的ACE hyp保存数据，重构首轮3样本校准及2×20步提案；全部loss/参数/梯度trace精确相同。117509 regions、16269层；含profiler的校准90.58 s、提案113.57 s。3m40s、MaxRSS902432 KiB；无新增rewrite/mapping/时序标签，不替代完整补验 |
| 64340.13 | `20261111T181423Z-openabc-hyp-terminal-retry-ace13171` | COMPLETED：69项真实核心PASS，index59在4980.26 s完整PASS，各搜索8次预算、0失败请求；两个梯度模式均有5次直接梯度评价，最终1854999.12 ps/176674.141 area，CEC PASS。相对no-op改善0.03748%，但比原输入慢4.54659%；原生ABC为1772861.75 ps。7200 s单例/900 s工具时限，旧3600 s失败及hash保留。Slurm1h23m53s、MaxRSS2758896 KiB；共享分配计时不作独占性能比较，源码`f984581` |
| 64392 | `20261111T182248Z-native-control-failure-regression-7ef427ca` | COMPLETED_WITH_GPU_OMISSIONS：`94909a7`隔离独立native对照失败，75项CPU矩阵63 PASS/12 SKIP；检查必要基线失败中止、内部错误重新抛出、缺失对照禁止数值/路径替代及重复结果状态。11 s；真实工具另行验证 |
| 64341.8 | `20261111T182419Z-native-control-real-regression-2c5d4d0b` | COMPLETED_WITH_EXPLICIT_OMISSIONS：75/75真实核心PASS；c1355/index197完整比较69.22/225.77 s PASS，c1355数值/选择/图结构不变，循环修复仍为13项CEC PASS。独立native故障保留FAIL，其余真实测量的9份映射/12个路径引用/3个预测关联通过。2510项矩阵完成：142 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、1395编译TU、108文件有命名证据。13m17s、MaxRSS839268 KiB；故障注入不计入benchmark性能 |
| 64393 | `20261111T182725Z-completed116-path-analysis-3a705938` | COMPLETED_WITH_EXPLICIT_OMISSIONS：只读复核64381全部116设计/1870份映射/2218引用，错误0，原39设计全部分析精确一致。50/116个original→no-op代表PI/PO对变化；9个失败及780未记录保留。5m25s、MaxRSS998472 KiB；没有新增rewrite/CEC/mapping，统计与解释见PATH_CHANGE_ANALYSIS.md |
| 64340.14 | `20261111T184033Z-isolated-native-backend-retry-813f3b26` | RUNNING：75项真实核心再次PASS后，使用已验证失败隔离流程重验已终止的108；本次真实native CEC仍在900.30 s超时，FAIL保留，随后已进入greedy。整例尚未完成，不提前记为PARTIAL/PASS。每搜索8次预算、7200 s单例/900 s工具限制，没有故障注入；旧失败及源记录前缀冻结。源码`0355cae`，代码与已验证`94909a7`相同；共享既有allocation，不重复其他live输入 |
| 64394 | `20261111T190337Z-saved-failure-evidence-audit-9b981c5c` | COMPLETED_WITH_EXPLICIT_OMISSIONS：`53d4541`增加只读失败证据汇总；81项CPU矩阵69 PASS/12 SKIP，191输入冻结清单的全部预测/梯度/方法CSV及数值聚合与旧结果完全相同，审计错误0。实际7200 s Division、900 s native失败及旧受控native fixture分别复核，不混入性能统计。2512项程序矩阵完成：144 import PASS/17 FAIL/1 SKIP、58 CLI smoke PASS、1395编译TU、110文件有来源匹配命名证据。5m33s、MaxRSS400244 KiB；没有新rewrite、CEC、mapping或拟合 |
| 64395 | `20261111T191749Z-full905-partition-certificates-f9e48d3a` | COMPLETED_WITH_EXPLICIT_OMISSIONS：`16ac3cd`对完整905输入独立验证CPU strict MFFC/owner、精确DAG及每端口反向AND距离，905/905 PASS；4 worker结构阶段147.13 s。86项CPU矩阵74 PASS/12 SKIP；2515项程序矩阵完成，147 import PASS/17 FAIL/1 SKIP、59 CLI smoke PASS、1395编译TU、113文件有命名证据。5m02s、MaxRSS6091568 KiB；未运行GPU、CEC、mapping或搜索，不能替代全量benchmark比较 |
| 64396 | `20261111T194041Z-consistent-cohort-submit-guard-e5f1d667` | RUNNING：仅申请1 CPU/1 GiB（站点实际2 CPU）、无GPU的提交监测任务。严格匹配父作业64340→新分片0、64341→新分片1，自然终止后重新检查sinfo、全部用户请求及CPU/内存/GPU预算，核对冻结脚本/输入/准备元数据hash后各提交一次。资源解析、GPU类型去重及预算边界检查PASS；首次实际检查为10 CPU/129 GiB/2 GPU，两份均保持等待。禁止取消旧作业；提交结果不明时停止，不自动重试 |

## 统一版本全量比较的预先准备

`20261111T184752Z-consistent-full-shard0-1568c1b9`和`20261111T184753Z-consistent-full-shard1-175169c2`已准备，**尚未提交**。两份`input/cohort_plan.json`固定同一源码`160b218`（程序与已通过75项真实核心及全仓库矩阵的`94909a7`相同）、64389工具构建、同一输入清单及互斥的偶数453/奇数452项；并集恰为905。准备清单hash、命令与shell语法已核对，输出目录为空，不计为实验完成。

两份均固定每搜索8次请求、warmup 2、梯度20步、校准30步、整例7200 s及单工具900 s；每个分片计划8 CPU、64 GiB、1 GPU、4个独立case进程、5天。保留默认固定随机种子及两目标模式。此前不同开发版本及定向补验不并入这一次统一源码的初始全量比较，不根据结果好坏挑选重跑。

当前64340/64341各占一张GPU，因累计上限2 GPU，准备没有触发提交，也没有取消现有分配。实际提交前须重新检查集群及所有RUNNING/PENDING请求；某一分配自然释放后才能提交相应新分片。附在旧分配内的step如随父作业结束而中断，须依据Slurm和保存记录明确标注，不能补造终态或把未运行的case算作已覆盖。后续提交与结果在新记录中更新。

两份分片现交由64396监测提交。**该监测任务仍存活时，不要另行执行这两份`job.slurm`。** 具体计划与脚本在其`input/plans.json`、`input/guard.py`，每次实际配额检查保存于`output/checks/`。提交前还会再次检查，并在目标运行内独占创建`input/launch_reservation.json`和`input/pre_submission_run.json`；成功后写`input/submission.json`，监测目录另存`output/submitted_64340.json`或`submitted_64341.json`。以这些回执及Slurm状态判断是否已提交，不根据旧准备状态重复启动。监测失败或退出时先检查全部回执和Slurm，不自动重试结果不明的sbatch。若明确需要停止后续自动提交，可在监测运行根目录创建`STOP`；它不会取消已经提交的作业。

## 首次真实闭环的有限结论

`20261111T162048Z-declared-timeout-division-retry-7fb70837`仅准备了源码和Division输入，未提交任何任务。准备后检查发现suite核心测试固定要求c1355，已标为`SUPERSEDED_NOT_SUBMITTED`；新目录64340.9用默认c1355作核心fixture，实际47/147从完整manifest选取。未覆盖输入或把准备失败计为实验FAIL。

`20261111T170757Z-backend-e2e-retry-812b3360`也仅准备、未执行。随后发现并修复了真实原生引用环，因此标记为`SUPERSEDED_NOT_SUBMITTED`，index108的实际补验使用64341.7的新目录和已验证工具；没有覆盖旧准备文件。

64330的原始与no-op replay基线均为360.111877 ps、513.216003 area；ABC rewrite为359.501831 ps、529.778870 area。面积约束搜索最好记录349.582336 ps、513.449280 area；联合目标最好记录345.476715 ps、486.388794 area。两种搜索各有17次直接梯度提案实际评估，其余为基线/warmup；失败请求计入24次预算。

这些仅是一个设计、一个种子的初次结果。尚未据此宣称优于随机搜索、跨设计泛化或整体验收通过；两个确定性基线的失败请求需要修复后重新比较。原日志永久保留。

64331修复后的完整比较：LiteCut贪心359.873505 ps/533.511353 area，独立region399.386475 ps/523.013794 area；面积约束搜索最佳仍为360.111877 ps/513.216003 area，联合目标最佳360.111877 ps/510.416656 area。不能只引用64330的改善。

64334小预算跨设计pilot中，c5315梯度面积约束搜索404.531586 ps，随机414.942444 ps；c7552梯度1020.499270 ps，随机927.469604 ps；jpeg_hm_decode独立region3040.164790 ps，梯度搜索维持no-op 9142.371090 ps。预算每搜索8次、单种子；没有建立稳定的算法优势。

## 修复版的阶段结果

64341.0的8设计检查中，面积约束梯度相对相同已消耗预算的随机搜索为2胜、5平、1负；联合目标为1胜、4平、3负。这里按各自实际目标判断数值胜负，joint胜不等于delay单项更低。样本包含定向选择的失败案例，不能据此推断全数据集效果或统计显著性。

`DSP_comb`的原始delay为4208.673340 ps，零rewrite replay为5567.682620 ps，二者CEC通过；梯度面积约束结果3847.503660 ps，随机3635.908200 ps。此例说明序列化对mapper的影响可能很大；报告必须分别对照original与no-op，不能仅报相对较差no-op的收益。修复版c1355两模式仍为360.111877 ps / 513.216003 area，未改善。

修复版c1355两次完整搜索的选择、最终选点、CEC、全部PO、面积及精确序列化图结构一致；Native ABC输出时间注释导致原始文件字节不同。完整比较见64350运行的`output/paired_report/repeat_comparison.json`。同一c1355运行的12个预先预测样本global-delta MAE约71.03 ps、Pearson约−0.63；这是自适应搜索内小样本诊断，不能宣称代理准确或跨设计泛化。

## 全量扫描暴露的接口问题

1. 命名AIG与无符号名replay：旧默认CEC按名称匹配而失败。新接口先检查PI/PO数量，再明确按原AIGER位置证明，保留命名差异。
2. cut自由变量检查过强：支持最小化和原始叶子逻辑关系会造成差异。六个案例的实际上下文CEC证明全部10,281个导出decision，不把旧误报伪装成GPU布尔错误。
3. 常量导出实际错误：`CPUSolver::Reset`已转为AIGER的0=false，但旧`WriteAig`再次翻转0/1。b14_1的前两个PO从0/1变成1/0。新导出直接写solver literal，同时初始化常量/PI phase与reorder状态；旧二进制和错误输出保留供回归。

## 测量边界

新ABC只读插桩输出的PO11 **Liberty** arrival（例如GG=424.422180 ps）与历史Gate3F的PO11 **MIO** arrival（GG=194.1 ps）不同，不能替换历史列。新工具的GG global=425.052216 ps，与历史显示425.05 ps一致；GL/LG/LL也分别对应389.023682、369.597626、388.937592 ps。关键PO并列集合与原生trace各自记录，不把mapper数字ID跨运行等同为原AIG逻辑点。

新增路径observer不依赖native `stime -p`的min-slack回溯作为指定端点的最大arrival路径。它在相同SC timing状态下读取load/slew、cell arc与rise/fall相位，选择能重构端点arrival的最大前驱，验证每级累计值。64354实际复核的124份映射来自8设计、107份不同AIG字节；同一AIG可能对应不同保存的mapped正文，因而不只按AIG去重。完整证据为该运行的`output/arrival_paths/summary.json`及逐映射`paths.json`。

此处固定PI/PO索引从0开始，只比较源/端点集合及当前映射内的链。它不是原图候选到cell的逐项对齐，也没有证明某条布尔可敏化的唯一物理路径切换。

## 大设计选择开销检查

64355在CPU任务中只读取保留的实际输入，不执行rewrite/mapping。`div`有51,140个executable decisions，最大region含1,016个decision；8组坐标选项的旧标量累计耗时7.41 s，优化后0.327 s，成本逐位一致。完整独立region选择耗时39.61 s；旧全量运行在此阶段超过1800 s单例上限，没有完成的旧选择可比较。`des_perf`完整选择与已保存请求完全相同，当前22.45 s；历史90.54 s来自不同并发负载，只能作诊断。

`hyp`有117,509个region、979,378个legal cuts。共享只读region集合后，完整catalogue耗时4.52 s；512条实际cut的归一化参考耗时1.027 s、优化后0.00304 s，内容相同。此处不把子集计时外推为整个优化框架的加速，也不把纯CPU诊断记作完整benchmark通过。后续须补齐超时设计的真实E2E。

64361用保存的真实输入与反馈标签重构div首轮校准及梯度提案，损失、参数、完整trace均与保存值完全相同。30步校准104.46 s（backward占75.67 s），两次restart各20步提案127.31 s（backward占55.94 s）；均含cProfile开销。主要瓶颈位于4183层DAG的传播/求导，另有逐decision转换与one-hot构造开销。不能把共享分配内的实际超时和该CPU剖析时间直接相除作为加速比。细节见[性能检查](PERFORMANCE_AUDIT.md)。
