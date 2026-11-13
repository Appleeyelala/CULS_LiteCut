# 独立基线的成本与验证契约

最终面板之前固定本协议；目的为公平比较完整部署成本，不让简单方法承担它没有使用的神经推理或代理拟合。源码入口为 `global_timing/controls.py`、`control_profile.py`，统一冷进程驱动仍为 `scripts/profile_single_method.py`。

## 方法与成本

| 方法 | 真实执行内容 |
|---|---|
| original | 原图严格 CEC + 统一 Liberty mapping/timing；不创建候选域或 Student |
| native_abc | 原图 `strash; rewrite`，再逐点严格 CEC + 相同 mapping/timing；不创建候选域 |
| native_culs | 原生 CULS `rewrite` 一遍，独立 CEC + 相同 mapping/timing；不是梯度方法，也不替换成多个优化 passes |
| random_cpu / random_batch | 同一 strict partition、GPU cuts/candidates、原图 cut/context 证明和逐 candidate library truth；变密度随机采样，静态合法投影。无需 Student、全代理或 calibration |
| greedy_litecut | 同一证明/域，原 checkpoint 19维特征与冻结推理；按局部 delay 最小值投影，真实安装及显式 repair，成功后停止 |

原图、native 的完整自然成本和一次请求保留，不虚构八次搜索。candidate 方法的逻辑预算8包含已测 no-op、失败和 repair；greedy 可少于8次。random 采用旧版相同的变密度 sampler，同种子先运行至多两个随机 warmup 请求；之后 batch 版本一次提出最多4个独立随机组合，CPU版逐一执行。repair 优先并占新预算位，所有选择都重新从原图安装。

随机和 greedy 只测自身必要的 no-op，不强迫它们再测一次不用的 original；原图另作独立控制。旧/新梯度已有 profiler 额外 original 测量仍计时，不从历史 wall 中事后删去。这给简单基线更接近部署的成本，不能把其速度差全部解释为梯度计算开销。

各 candidate 方法都用 no-op 的实际 area 定义≤1.05硬约束，返回满足约束的已测最低 delay，包括 no-op。保存 greedy 的原始请求和 repair，不用最终 fallback 隐藏失败。native/original 的面积是否满足同输入的 no-op 阈值由最终对照分析确定，不为计算阈值额外执行无用搜索。

## 执行复用与隔离

`execution.py` 从已验证的旧 `Experiment` 和 batch 实现逐函数提取真实安装、repair 与完整验证逻辑；旧公开方法保留包装。`boolean_candidates.py` 的库 parser 来自保留的原 canonical 函数，Boolean checks 为与 Student adapter 共用的同一实现。原始 LiteCut/CULS快照不回写；Student cache identity 加入共享 Boolean 文件 hash。

random/native 的导入路径不加载 Torch/NumPy/NetworkX；greedy 按需加载真实 Student。每方法独立冷 child、独立输出/cache；开始前记录输入/模型/库/二进制/源码身份，完成时记录 monotonic 时间。batch 只复用枚举/原图上下文，每个成功完整安装仍各自 CEC+mapping，验证并发2并受分配限制。

主比较使用默认正 AIG 增益 sequential 安装，保持与冻结旧方法和随机基线同政策。`*_physical*` 独立保留作政策对照：64976 显示放宽后 DMA 请求虽不再失败，却改变 warmup/反馈及搜索序列，返回2396.672360 ps，比正增益版2301.566650 ps更差且更慢。先前固定配置2288.404300 ps不能替代完整搜索结果。

所有时间包含该方法实际部署所需的导入、初始化、证明/特征、传输、选择、失败/修复、CEC与mapping；输入 staging 另外计入一次。排队/源码归档分列，操作系统页缓存不清。各方法按真实完成事件计算时间截点，不能倒填未完成结果；含并发的阶段时间不能直接求和。新基线实现尚待独立回归与真实对照，协议本身不是通过证据。
