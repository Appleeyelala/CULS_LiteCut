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

构建工具复制ABC基线后仅加入只读Liberty端点报告。`--build-only`可在无GPU任务中编译，明确记录`BUILT_NOT_RUNTIME_VALIDATED`；之后仍须运行GPU回归。测试工具核对二进制哈希与全部对应原生源码，拒绝用旧二进制测试已改动的C/CUDA源。

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

## 失败与恢复

任何CEC否定/未决、工具超时、缺失时序或部分安装都保留FAIL，不写入虚构指标。结构冲突在GPU前拒绝；运行时REVERTED必须创建一个新修复请求，不能把原失败请求改写为成功。保留旧结果，在新提交、新目录中修复并重跑相关回归。缺失依赖和历史硬编码入口分别报告，不自动重放旧训练或覆盖历史目录。
