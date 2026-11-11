# 构建、测试与复现

## 工作区与依赖

源码在 `/home/wushangge/CULS_LiteCut`；计算、临时文件、缓存、二进制和结果均在 `/data/wushangge/CULS_LiteCut/runs/`。使用本地分支提交后准备不可变快照，不直接运行历史 `/home` 实验脚本。

当前验证环境为 Python 3.11（`/home/share/process/anaconda3/bin/python`）、GCC/G++ 11.4、CUDA 11.8、CMake 3.28，以及已有 Torch、NumPy、SciPy、pandas、networkx、scikit-learn、matplotlib。每个新运行的 `run.json` 保存具体版本、线程限制、Slurm环境、设备和依赖信息。`test_repository.py` 输出完整已安装包版本。未擅自安装或改动共享环境。

预训练Student、ASAP7和历史二进制的路径/hash见 `provenance/assets.json`。不需要重新训练Student。Teacher/DeepGate训练还需要原训练数据、DGL/DeepGate等依赖；这些入口不能因Student推理成功被标成训练PASS。

## 准备与提交

先提交源码，再执行轻量准备：

```bash
python -B scripts/prepare_run.py --name my-check \
  --purpose '具体授权的测试目的' --design OpenABC/c1355_orig.aig
```

准备工具只建目录和清单，不运行或提交任务。将下面命令写入该运行目录的 `job.slurm`，通过 `sbatch` 执行。设置工作目录、`TMPDIR`、`XDG_CACHE_HOME`、`MPLCONFIGDIR`、`PYTHONPATH` 至运行目录及其 `src`；设置 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。共享环境建议显式设置`MKL_THREADING_LAYER=GNU`，避免先导入NumPy的父进程把INTEL线程层传给Torch子进程。已完成运行目录保留完整作业脚本，可按任务审查资源和参数。

每次提交前检查 `sinfo` 和 `squeue -u wushangge`。包含PENDING在内，累计不超过64CPU、256GiB、2GPU；现有分区为Compute，但仍须当次检查。

## 入口命令

以下占位符 `RUN` 与 `BUILD_RUN` 均指具体 `/data` 运行目录，命令只能在Slurm分配内执行。

```bash
python -B RUN/src/scripts/build_framework_tools.py --run RUN --jobs 8
python -B RUN/src/scripts/test_framework_run.py --run RUN --tool-run BUILD_RUN
python -B RUN/src/scripts/run_framework_experiment.py \
  --run RUN --tool-run BUILD_RUN --evaluations 24 --warmup 6 \
  --gradient-steps 60 --calibration-steps 80 --mode both
```

构建工具复制ABC基线后加入只读Liberty端点报告及默认关闭的最大arrival路径observer。`--build-only`可在无GPU任务中编译，明确记录`BUILT_NOT_RUNTIME_VALIDATED`；之后仍须运行GPU回归。测试工具核对二进制哈希与全部对应原生源码，拒绝用旧二进制测试已改动的C/CUDA源。

底层API为 `python -m global_timing --help`。指定输入、gpuls、ABC、library、checkpoint和全新output目录。支持`area_constraint`、`joint`或`both`；`--area-limit-ratio`默认1.05，`--area-weight`默认0.5，随机种子默认20261111。路径必须不含CULS/ABC命令解析器无法转义的字符。

全数据集先用 `stage_benchmarks.py` 生成逐文件hash清单；再用 `run_benchmark_suite.py --manifest … --shard-count 2 --shard-index 0` 及另一个索引1覆盖数据。`--workers`受CPU配额限制，共享当前已分配GPU，不额外申请资源。单例超时会终止其工具后代进程并保留记录。

本地持续集成入口为：

```bash
python -B RUN/src/scripts/ci.py --run RUN --tool-run BUILD_RUN
```

它串联完整核心回归、真实c1355双目标8预算smoke、原生writer、已有Gate1工具与ABC Gia测试、全程序矩阵，并生成`ci_summary.json`。完整入口需要一个GPU分配；`--unit-only`可用于CPU快速检查，明确保留GPU测试SKIP。没有配置外部CI服务或Git钩子，完整流水线须在实际运行通过后才能作为提交的验证证据。

## 检查产物

- `config.json`、`partition/`、`candidates/`：输入与原始分区、GPU导出，保留原始ID。
- `cut_validation/`：独立边界/小BDD证明或原始逻辑上下文CEC，记录全部未证明条目；只有已证明cut进入选择。
- `litecut.json`：局部19维特征、预测、cache来源；不作为global真值。
- 每个 `eval_*/`：选择、真实顺序trace、AIG、CEC、mapping、Liberty timing、mapped netlist、命令和耗时。
- `gradient_*/`：预先预测、拟合记录、真实梯度、投影选择、实际评价进展；失败/修复/warmup均计入预算。
- `summary.json`、`report/`：六类方法、两种目标的真实比较、CSV与可导出PNG/PDF。

使用 `summarize_benchmarks.py` 聚合已有结果；`--repeat A B`分别比较文件字节、精确序列化图结构、选择与全部PO/CEC/面积，忽略墙钟时间。图结构hash仅排除符号名称和注释（包括ABC时间戳），保留节点ID和端口顺序。跨版本还须确认算法源码、输入、库和二进制哈希；不能只因最终delay相同就声称完全复现。报告分开绘制最佳面积可行delay与实际联合目标，保存候选变化及真实梯度范数。

保存选择的路径审计可在新的CPU Slurm任务中执行：

```bash
python -B RUN/src/scripts/analyze_timing_paths.py --run RUN \
  --tool-run OBSERVER_BUILD_RUN --benchmarks SAVED_CASE_DIRECTORY
```

该命令在新目录重新CEC/mapping，启用`CULS_GLOBAL_TIMING_PATH_REPORT=1`，要求原测量数值和网表正文完全一致。`output/arrival_paths/`保存各映射的`paths.json`、完整日志及各方法源/端点集合变化；旧搜索目录不改写。observer源码和二进制应来自同一已记录构建，不用新标签覆盖历史Gate3F的MIO列。

`profile_selection.py --run RUN --benchmarks SAVED_CASE_DIRECTORY`只读取已保存输入、候选与LiteCut分数，比较优化后的候选归一化、标量参考成本和历史独立region选择，不执行rewrite或mapping。`--catalogue-only`可诊断尚未完成推理的大设计。成本相等、选择相等与耗时分别报告；缺失历史选择时明确写`NO_SAVED_REFERENCE`。

加`--student-rescore`会用准备目录的checkpoint重新提取19维特征并推理，要求全部score记录与保存的旧结果完全相同。新的`extract_all(..., student_only=True)`跳过模型未使用的5个谱特征及WL hash；默认25维接口保留。缓存键包含提取器和adapter源码hash，因此旧缓存不会被无声复用为新版本的核验证据。具体测试范围及缺失依赖见[测试证据](TEST_COVERAGE.md)。

`analyze_search_quality.py --output NEW_OUTPUT --suites SAVED_SUITE_0 SAVED_SUITE_1 --expected-inputs 905 --cohort-label LABEL`在CPU Slurm任务中审计已有搜索，不执行rewrite/mapping或拟合。它冻结各suite当前完整JSONL记录，拒绝同一输入重复混入，输出预测误差、校准标签来源检查、梯度请求接受率、最终选点来源和图表。正在扫描的suite会明确显示尚未全覆盖；失败请求不填充时序数值。预测统计属于模型不断更新的自适应搜索，不解释为固定模型排序或跨设计泛化。

全量分片和补验包含重复输入时，先在新的CPU Slurm运行中建立显式选择清单：

```bash
python -B RUN/src/scripts/build_benchmark_cohort.py --manifest INVENTORY_JSON \
  --suites ORIGINAL_SUITE_0 ORIGINAL_SUITE_1 RETRY_SUITE --output RUN/output/cohort
python -B RUN/src/scripts/analyze_search_quality.py \
  --cohort RUN/output/cohort/cohort.json --expected-inputs 905 \
  --cohort-label LABEL --output RUN/output/search_quality
python -B RUN/src/scripts/summarize_benchmarks.py \
  --cohort RUN/output/cohort/cohort.json --output RUN/output/paired_report
```

顺序必须预先声明：最后列出且已有终态记录的尝试用于当前汇总，后一次失败也会替代前一次成功，绝不按delay/area选择最优重跑。清单冻结所有源JSONL完整前缀、run元数据和输入manifest，并保留全部尝试及原结果路径；尚无记录的声明任务不代表已确认正在运行，只表示本快照没有其终态。只有所有输入及所有声明任务均有终态记录，才标记覆盖完成；它仍不等于全部PASS。

读取清单时重新检查来源快照hash、选择规则，以及成功案例的输入、配置、模型/工具/库身份和结果hash。两种分析使用同一选择，保留源码版本并分别输出版本分层统计；总体数字不冒充单一最终版本的结果。新的原始记录只进入后续新清单，不改写此前快照。该流程不重新执行任何rewrite、mapping或模型校准。

`profile_gradient.py --run RUN --benchmark SAVED_CASE --mode area_constraint --iteration 2`在CPU Slurm内重构已保存的首轮校准与梯度提案。要求参数、loss及完整trace精确相同，输出逐阶段cProfile。后续轮次还依赖旧模型状态，当前入口无法重建时会明确FAIL。它不产生新的timing标签。

历史原生工具使用`build_native_variants.py --run RUN --jobs 8`重建：每个ABC overlay各自覆盖一份基线，Gate1单独链接其ABC依赖。随后使用`test_native_variants.py --run NEW_RUN --build-run NATIVE_BUILD_RUN --scope abc`，或在GPU分配内用`--scope gate1 --abc CEC_BINARY`。新验证核对全部相关源码与二进制hash，输出到新目录。mapped BLIF的CEC必须先加载同一library。`test_repository.py`支持`--native-build-run`和多个`--evidence-run`，把实际对象文件及来源匹配的命名检查加入逐程序矩阵。

## 失败与恢复

任何CEC否定/未决、工具超时、缺失时序或部分安装都保留FAIL，不写入虚构指标。结构冲突在GPU前拒绝；运行时REVERTED必须创建一个新修复请求，不能把原失败请求改写为成功。保留旧结果，在新提交、新目录中修复并重跑相关回归。缺失依赖和历史硬编码入口分别报告，不自动重放旧训练或覆盖历史目录。

`diagnose_replay.py --run RUN --tool-run BUILD_RUN --saved-case SAVED_CASE`用于在GPU Slurm分配内诊断保存的零rewrite原生失败。准备时必须用`--design`选择与原失败输入hash相同的设计；入口核对旧二进制hash、复制owner表并在GDB下重放，保存信号与backtrace。如果未复现，仍检查零选择和CEC/mapping，只记录`PASS_NOT_REPRODUCED`，不修订原case状态。诊断作业退出0仅表示生成诊断记录，应检查`output/replay_diagnosis/summary.json`中的实际分类。
