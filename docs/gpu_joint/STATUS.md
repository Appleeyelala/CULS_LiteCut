# GPU 联合梯度优化：当前进度

当前目标仍在进行，尚未完成最终验收。主线为跨MFFC梯度联合选择、单电路GPU并行、真实mapped delay与端到端runtime；原905实验保持原样，不作为新版方法的大样本验证。

## 已实现与已有证据

- 保留冻结旧STE控制；新增一次原DAG adjoint、保留邻接双线性项的局部二次提案，32条轨迹×40步，CPU/CUDA float64，支持八维既往PASS标量反馈。
- 稀疏传回argmax编辑、CPU冲突索引和hard打分复用；64932大图同工作量组件有1.70–27.59倍单次CPU/CUDA差异，选择/顺序hash一致。这不是端到端加速。
- 原生多配置命令一次枚举复用，逐配置从原图安装；每个完整组合各自CEC与Liberty mapping。64945/64951的12最终点和6个seq/batch逐请求配对一致，端到端时间仍不稳定。
- 新增按固定规模选择旧全代理STE或局部批量提案：64957数值30/30 PASS；64976真实9/9冷进程通过。c1355/ss_pcm实际仍用CPU全代理，DMA用局部近似CPU。
- 独立自然成本original/native ABC/native CULS/random/LiteCut greedy入口已实现；共享原Boolean与真实执行函数，random/native不加载不用的ML。64977为51/51回归PASS；64978为15/15真实开发方法点PASS。
- 64979通过3项反例/时间截点测试，并审计上述两组24个方法点的94次成功测量：精确完整安装、原始CEC、Liberty、网表、资产/命令hash与真实完成事件通过。时间阈值按已有协议，不由mtime推测。

## 当前真实结果与边界

64976中，正增益adaptive batch：DMA2301.566650ps（28rewrite显式修复），ss_pcm323.908447ps，c1355360.111877ps/no-op。相同分配旧版分别2380.250980、323.908447、360.111877ps。小图旧STE避免了此前ss_pcm近似版328.026855ps的退化；单次wall不能证明稳定加速。

允许非正AIG增益的64956固定四向量全部CEC/面积可行，最佳2288.404300ps；但64976中同策略自由搜索DMA返回2396.672360ps、耗时27.917s，差于正增益2301.566650ps/18.222s。该政策改变warmup和反馈序列，保留负结果，默认仍正增益，不将固定probe最佳点冒充搜索最佳点。

64978自然成本强基线：ss_pcm原生CULS同样323.908447ps/1.618s，random319.488068ps/8.381s；c1355原生CULS353.219391ps/1.668s，优于梯度no-op。DMA梯度点低于本轮随机/CULS，但更慢。两组不同分配，不作严格端到端加速配对。

## 当前待验证与交付边界

- Slurm64980正在执行大图ACE/div同算法CPU/CUDA完整八请求闭环，包含全部准备、真实安装/CEC/mapping；终态与结果尚待核对，不预言端到端收益。
- 新增初始化不更新控制只关闭内层backward/Adam，仍保留既有校准/大图adjoint；不是全无梯度方法，尚待回归与真实验证。说明见[INNER_UPDATE_CONTROL.md](INNER_UPDATE_CONTROL.md)。
- 多种子冷进程入口与前瞻最终协议已写入，固定面板16输入×3seed未执行。主方法、强基线、CPU/auto及内层更新控制将同分配顺序比较；自然成本口径见[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md)。
- 竞争性整体QoR、稳定低runtime、梯度独立价值和最终Delay–Runtime比较尚未建立；完整五项交付尚未完成。

具体源码/数学见[IMPLEMENTATION.md](IMPLEMENTATION.md)、[ADAPTIVE_SELECTOR.md](ADAPTIVE_SELECTOR.md)；所有路径、失败与实际数值见[RUN_LOG.md](RUN_LOG.md)。主报告是7e5bbc1记录的阅读截点，后续新实现和实验在本目录继续追加，不回写旧905证据。
