# GPU 联合梯度优化：当前进度

本轮实现、实验与分析交付及最终文档/版本核验已完成，详见[交付审查](GOAL_DELIVERY_AUDIT.md)。主线仍为跨MFFC梯度联合选择、单电路GPU并行、真实mapped delay与端到端runtime；原905实验保持原样，不作为新版方法的大样本验证。**整体1.5倍加速与梯度优于简单搜索的性能目标没有达到。**

## 正式结果与补充控制均已完成

64983和独立分析64984均已COMPLETED/0:0：432个方法点、1119个成功观察的原始证据审计通过。主方法/旧版delay比几何均值0.999398、P90=1.007710，达到预定QoR非劣；总体旧版/主方法中位wall比1.0355，未达到1.5。GPU在9设计27个实际CUDA且请求/工作量一致的配对上中位加速1.326246；其余21对不计CUDA贡献。主方法对random的delay比1.026663，对关闭内层更新为1.043888，梯度独立优势没有成立。

实际结果、全部baseline、时间截点和限制见[FINAL_RESULTS.md](FINAL_RESULTS.md)，复现入口见[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。下文较早条目是开发阶段证据，不覆盖本节终态。

65004已COMPLETED/0:0、30:01：37项回归、同分配默认兼容与四配置smoke通过，48/48方法点和115成功观察原始审计完成。默认/去score的delay比1.002822、默认/去pair为1.002929，均无总体收益证据；默认/禁修复为0.952208、9胜3平，支持该四设计子集的修复价值。每配置含3个无Decision/no-op点，不外推泛化。

65009（`20261113T081810Z-joint-final-delivery-d22ea3ab`、1994e57）已COMPLETED/0:0、13s：来源/阶段分组、修正图和hash清单已生成，四组PNG/PDF已查看/读取。它只复用完成审计产物，不运行EDA/模型。S/D的18/14项产物hash逐项核对，三个未改动PNG与原审查图逐字节相同。

全部正式432点的3120模块引用、两快照65个包/Student来源文件与Git匹配；补充48点420模块引用也匹配，且12个默认完整请求轨迹/真实指标与2257998相同。修复父请求另经实际summary核对：主方法28个最终repair中17来自梯度、11来自warmup，不能把28全部归梯度。

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

## 已验证范围与未建立的结论

- Slurm64980已完成大图ACE/div完整CPU/CUDA配对：327.206/169.602s，约1.929倍，八请求向量/状态/成功指标一致；最终都返回no-op，没有QoR改善。64982随后完成26开发/基线方法点的108成功测量完整原始审计。
- 初始化不更新控制只关闭内层backward/Adam，仍保留既有校准/大图adjoint；不是全无梯度方法。64981为36/36回归PASS，2seed×4方法8个真实冷child及22个成功测量审计PASS。说明见[INNER_UPDATE_CONTROL.md](INNER_UPDATE_CONTROL.md)。
- 冻结16输入×3seed×9方法的正式比较64983已完成，独立64984统计与四组PNG/PDF已生成并人工查看；正式终态与数值见本页上节，不把原905计入新版验证。自然成本口径见[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md)。
- 整体1.5倍runtime目标和梯度独立价值未得到支持；按用户研究说明§6交付负结果，不修改门槛或宣称论文优势。
- 65004和65009均已终态，补充控制与最终分组范围见[CHANNEL_ABLATIONS.md](CHANNEL_ABLATIONS.md)及FINAL_RESULTS；没有扩展成新版全905验证。

具体源码/数学见[IMPLEMENTATION.md](IMPLEMENTATION.md)、[ADAPTIVE_SELECTOR.md](ADAPTIVE_SELECTOR.md)；所有路径、失败与实际数值见[RUN_LOG.md](RUN_LOG.md)。GRADIENT_OPTIMIZATION_STATUS是9b5b065记录的15:57只读截点；后续正式/消融结论在本目录，不回写旧905证据。最终逐项交付审查见[GOAL_DELIVERY_AUDIT.md](GOAL_DELIVERY_AUDIT.md)。
