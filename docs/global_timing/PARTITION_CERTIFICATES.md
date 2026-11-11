# 全905输入的分区结构证书

64395使用源码`16ac3cd`，对已暂存清单中的全部905个唯一文件hash输入完成独立结构检查：905 PASS、0 FAIL，无抽样或遗漏。产物在`/data/wushangge/CULS_LiteCut/runs/20261111T191749Z-full905-partition-certificates-f9e48d3a/output/partition_validation/`。`summary.json`保留完整逐例记录；每个`case_XXXX/`保存`partition.json`、`owners.csv`和`certificate.json`及其hash。

| 数据来源 | 已验证输入 |
|---|---:|
| ACE | 40 |
| OpenABC | 47 |
| forgeEDA | 805 |
| iwls2022 | 5 |
| iwls2024 | 8 |

该阶段使用4个CPU worker，累计wall time为147.13 s，包含解析、构造、独立核验、写出完整分区与hash；不是单独校验函数的性能或独占资源加速比。完整作业还运行CPU测试和程序矩阵，其终态另见运行记录。

## 检查方法

生产构造器仍为未修改的`global_timing/structure.py`。独立检查器`partition_validation.py`不调用deref函数，使用原始图的fanout计数与从region root反向计算的距离验证：

1. 每个AND节点恰有一个owner，成员、root、region ID和size一致。
2. 每个成员在region内可达root；非root成员没有外部AND或PO fanout。
3. 每个边界AND仍有region外的使用者，即使删除整个region，也不能把它吸收进此cone。
4. 所有跨region原图边与去重Region DAG完全对应；topological order为完整排列，全部边朝前。
5. 反向遍历独立计算每个边界端口到root的最长AND距离，逐项匹配transfer；再把区域传播与原图节点级深度递推比较。

前两项使成员属于root的standard MFFC；第三项阻止任何外部边界节点进入，所以每个cone是完整standard MFFC。一个成员为root的MFFC仍被同一闭合region包含；这些精确root cones又构成互不重叠的全覆盖，因此都是inclusion-maximal。该论证不依赖对905个输入枚举全部嵌套cones。

新增测试把80个不同小图的结果与历史Gate3A全root oracle比较，并用反例检查可继续吸收的错误分割、内部节点暴露给PO、断开的成员、owner/edge/order错误，以及全局最长深度不变但单端口距离错误。5项新增测试通过；完整CPU矩阵86项中74 PASS、12项真实工具入口明确SKIP。

## 已验证范围

c1355仍为512 AND、62个region、128条region边、240个边界端口，最大PO AND深度29。大例Back_End_With_Decode的365700个AND和43734个region也通过；hyp的117509个region全部通过。不同namespace和不同深度指标分别保留。

这是CPU strict MFFC构造器的结构证书，不宣称CULS受cut/depth限制的GPU discovery在全部输入上相同。AND深度没有ps单位，也不是Liberty arrival；这些证书不验证candidate执行、CEC、映射后时序或优化收益，不替代正在进行的全量benchmark比较。
