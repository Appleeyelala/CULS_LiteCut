# 联合梯度与 GPU 改进：最终实验结论

本轮实现、对照实验与分析交付已完成：**与旧版的真实delay基本持平；GPU对新提案器有独立端到端收益，但整体方法未达到预定加速目标，梯度优于简单搜索的主张也未成立。** 四设计补充控制没有支持LiteCut标量或显式pair的独立收益，保留修复则有实际质量收益。这里交付完整负结果，不将工程通过写成性能目标达成。

## 实验身份与测量口径

固定16输入×3种子×9方法，共432个冷方法点全部完成，Slurm64983为COMPLETED/0:0、elapsed1:42:27。正式源码2257998，旧控制实际导入5025a6c及旧gpuls；独立分析Slurm64984为COMPLETED/0:0、elapsed7:01、10项分析测试通过。原始证据审计覆盖1119个进入观察/曲线的成功测量，错误0。失败请求不获得timing标签。

路径缩写（共同前缀 `/data/wushangge/CULS_LiteCut/runs/`）：

| 标识 | 实际目录 |
|---|---|
| F：正式实验 | `20261113T062048Z-frozen-joint-final-panel-b9c390b3` |
| S：独立正式统计/图表 | `20261113T070050Z-frozen-joint-final-synthesis-f895a375` |
| A：补充通道/修复控制 | `20261113T073402Z-channel-repair-ablation-c9634975` |
| D：最终来源分组与图表显示修正 | `20261113T081810Z-joint-final-delivery-d22ea3ab` |

主结果来自 `S/output/final_synthesis/{RESULTS.md,statistics.json,audited_records.json,methods.csv,pairs.csv,method_points.csv,provenance.json,analysis_completion.json}`。F的completion和协议/输入hash均通过独立后处理检查。A为COMPLETED/0:0、30:01，48/48方法点及115个成功观察完成原始审计；D为COMPLETED/0:0、13s，来源/阶段汇总和四组修正PNG/PDF已生成并查看。S/D分别18/14项产物hash逐项复核，四PDF均可读取、各1页；原S和F保持原样。

全部结果为同一ABC、ASAP7和 `ABC_SCL_LIBERTY`：每个完整组合先精确全ACCEPTED、CEC等价，再 `map; topo; stime -p`，从同一SC_Man读取global和全部PO rise/fall。不是MIO arrival、旧PI→PO标签或Student预测。面积上限为真实replay no-op的1.05倍；原图与序列化no-op分列。

每搜索最多8逻辑请求，含no-op、warmup、失败、修复；小图保留20步×2restarts的全代理STE，大图一次DAG adjoint后32条float64轨迹×40步、合法邻域投影。全部方法独立冷child，随机不承担不用的Student/代理，native不做候选准备。完整成本含初始化、准备、推理、拟合、传输、失败/修复、CEC/mapping、保存和一次input staging；不含排队与一次性源码归档/工具构建。详见[复现索引](REPRODUCIBILITY.md)。

## 预先门槛与真实质量—时间结果

| 项目 | 结果 | 判断 |
|---|---:|---|
| 主方法/旧版delay比几何均值，48/48配对 | 0.999398 | 达到≤1.01门槛 |
| 同一48配对delay比P90 | 1.007710 | 达到≤1.05门槛 |
| 旧版/主方法完整wall比中位数 | 1.0355 | 未达到≥1.5门槛 |
| 相对旧版真实delay胜/平/负 | 17/24/7 | 小面板近似持平，不能称稳定QoR优势 |

加速值按48个配对的 `median(t旧/t新)` 定义，不是两列绝对wall中位数相除。下表新方法绝对wall中位数略高，与配对比值略大于1并不矛盾。

相同时间的替代条件没有预定一个总体二元判定阈值，不事后选择截点宣布达标。30秒时主方法对旧版13/29/6、delay比几何均值0.999975；对随机9/21/18、1.038822。120/600秒时对旧版17/24/7、0.999398，对随机12/19/17、1.026663。所有48对在这些截点均有已验证面积可行点，没有补造no-op。

这些是固定最多8请求运行的真实完成曲线，不是让各算法在整个120/600秒内持续增加预算的实验。终止后保留已测best，不能由曲线平台推断搜索已收敛或达到最优。

| 方法 | 面积可行final/48 | 中位完整wall s | 达到预定质量/48 | 达标者条件中位时间 s |
|---|---:|---:|---:|---:|
| 冻结旧STE CPU | 48 | 19.3666 | 32 | 8.0074 |
| 新联合CPU | 48 | 24.9616 | 32 | 7.8289 |
| 新联合auto（主方法） | 48 | 19.7258 | 32 | 7.8456 |
| 关闭内层更新 | 48 | 21.1864 | 39 | 8.7274 |
| 自然成本random batch | 48 | 10.3085 | 36 | 4.3419 |
| LiteCut greedy | 48 | 11.1881 | 30 | 6.0460 |
| 原图 | 48 | 1.9175 | 21 | 1.6784 |
| native ABC rewrite | 42 | 2.4945 | 39 | 2.6587 |
| native CULS rewrite | 45 | 2.6483 | 36 | 2.8704 |

达标定义固定为面积可行no-op/native ABC较低delay的1.01倍。条件中位数的达标集合不同，不能单独用其比值宣称所有设计的time-to-quality加速。上表所有方法进程48/48成功；native面积不可行仍保留原始测量，不填充或删除。完整逐设计、每seed的delay、面积、请求、来源、时间与工具调用在S的CSV/JSON中。

## 梯度和 GPU 的独立证据

主方法对random为12/19/17，delay比几何均值1.026663；纯CPU梯度包对random也是该比值。主方法对“无内层更新”为7/17/24、1.043888。**当前面板没有支持内层梯度比简单搜索/初始化控制更有效，存在真实退化。** 无内层控制仍有代理校准、大图adjoint/反馈/投影，不能叫完全无梯度方法；独立random才是不使用时序梯度的对照。

主方法直接梯度请求仅45/161成功；其中22个成功直接点面积可行且比no-op delay低。48个最终点来源为直接梯度6、显式修复28、warmup1、no-op12、重复投影后的探索1。闭环能找到真实改善组合，但大量收益依赖修复或已有基线；不能将所有成功都归于梯度，更不能以非零梯度或loss下降替代有效性结论。

[修复父请求审查](../../provenance/gpu_joint_repair_ancestry.json)进一步确认：主方法28个最终修复点中，17个来自失败的直接梯度请求、11个来自warmup；旧版为8/23。无内层控制的31个最终修复点中，24个来自重复投影后的探索、7个来自warmup。逐项父请求均存在、先于子请求、状态为FAIL且来源相符。梯度对部分修复组合有来源关系，仍不构成相对随机的收益证明。

GPU对照为 **9设计、27个实际CUDA配对**：同分配、算法/模型/候选域/精度/配置/内层调用数相同，完整请求向量、状态、成功PO/delay/area及EDA调用数一致，CPU/auto完整wall比中位数 **1.326246**。其余21对为CPU或没有提案，未计入GPU贡献。27对的最终QoR与CPU相同；已有CULS的GPU枚举/重写能力没有当成新增梯度GPU贡献。

保存源码还通过[独立静态身份核对](../../provenance/gpu_joint_source_audit.json)：432方法点的3120个记录模块引用（20种身份）均与实际快照及所记Git对象相同；两快照全部框架Python包和指定canonical Student/feature/bridge/library共65文件也逐项匹配。该轻量检查不运行实验，不替代工具构建或原始测量审计。

这解释了“GPU确实有用但整体对旧版仅1.0355倍”：新CPU联合方法本身更贵，GPU减少了其中的部分成本；Student特征准备、CPU校准、adjoint、投影和真实EDA仍限制端到端收益。开发集div另有327.206/169.602s、1.929倍的完整CPU/CUDA配对，但最终均no-op；该单例不并入正式16设计，也不证明更早找到改善点。

### 已有内存/显存数据的范围

正式64983的batch MaxRSS为2083396K，独立分析64984为420804K；这是Slurm进程内存记录，不是GPU显存。已有64932保存了三个大图提案组件的 `torch.cuda.max_memory_allocated()`：

| 输入 | Torch活跃CUDA分配峰值 bytes | 对应组件调用 |
|---|---:|---|
| div | 1099438592 | 32轨迹×40步，CUDA3.513s |
| hyp | 44398080 | 同配置，CUDA2.564s |
| voter | 98498560 | 同配置，CUDA0.184s；只有1个MFFC |

原始路径为 `20261113T035708Z-joint-sparse-cpu-cuda-a65c4ca7/output/kernel_case_{0020,0022,0038}.json`（同上/data runs根目录），采集代码 `scripts/profile_joint_kernel.py` 在每次组件调用前reset峰值，再读取该Torch指标。它不含全部driver/context/reserved显存或原生CULS的分配，不能当作完整闭环显存峰值。Slurm虽然配置了gpumem/gpuutil TRES，但64983/64980实际会计字段未给出这两项；正式端到端显存仍未记录，不能填成0。

保存分区的结构形状另有[只读核对记录](../../provenance/gpu_joint_component_structure.json)：

| 输入 | Region数 / Region边数 | Region DAG最长层数 | 同层最大Region数 | Decisions / padded width |
|---|---|---:|---:|---|
| div | 28306 / 72644 | 4183 | 128 | 51140 / 7 |
| hyp | 117509 / 266271 | 16269 | 12991 | 2760 / 5 |
| voter | 1 / 0 | 1 | 1 | 7007 / 6 |

“层数”按最长链上的region节点数定义，不是原AIG的AND深度或真实时序。hyp的CUDA组件调用中CPU adjoint约2.445s，接近整个2.564s；div相应为0.969/3.513s。它们说明剩余CPU传播与候选张量规模必须分开考虑，不能仅按Region数或GPU kernel倍数预测总加速。这三例同时改变了DAG形状、候选数等因素，是描述性瓶颈证据，不是独立控制深度的因果实验；voter也不构成跨MFFC收益例证。

## 已完成的通道与修复控制

协议按原面板各AND规模区间的首项固定65/96/625/587、三个原seed、默认加三个开关。研究选在主面板运行期间，不能称完全预结果或未见留出设计。主面板不重选、不回写。先执行37/37回归、同分配c1355默认兼容与四配置smoke，再执行48方法点。原始115个成功观察均完整安装、CEC/Liberty通过；失败请求保留。

| 移除项 | 配对 | 默认胜/平/负 | 默认/移除后delay比几何均值 | 移除后/默认中位wall比 |
|---|---:|---|---:|---:|
| LiteCut标量delta | 12/12 | 2/7/3 | 1.002822 | 0.989273 |
| 显式邻接pair | 12/12 | 2/7/3 | 1.002929 | 1.006506 |
| 可执行性修复 | 12/12 | 9/3/0 | 0.952208 | 0.976433 |

前两项没有显示默认信息通道的总体质量收益；不据这四设计断言LiteCut或交互在所有电路无效。第三项支持修复在该子集的实际质量价值；禁止修复的搜索通常较快，但质量更差。no-score保留实际19维/Student准备成本、仅清零标量delta；no-pairs仍有DAG传播耦合；no-repair会改变后续请求和反馈，不能解释为相同组合的纯速度差异。A没有另外测native，其内部达标参考仅no-op，不与F的native参考混合。

其中mig设计没有可执行Decision，三个seed都返回no-op，因此每配置只有9个方法点具有可变候选域；12配对分母保持原协议。默认有75请求、45失败，36个直接梯度请求全部失败，最终为9修复/3no-op；9修复的父请求为5梯度/4warmup。无修复有75请求、50失败，最终为3直接梯度/9no-op。该子集的收益尤其依赖修复，不能把“48进程PASS”误读为所有请求成功。

三seed真实delay中位数（ps）可直接定位差异：

| 输入 | 默认 | no-score | no-pairs | no-repair |
|---|---:|---:|---:|---:|
| k2 | 399.364594 | 395.401672 | 399.364594 | 410.195526 |
| ALU_mod | 1556.381230 | 1576.749880 | 1559.154540 | 1698.107300 |
| multiplier_33x33_partial | 930.648376 | 930.648376 | 930.319946 | 1012.163640 |
| mig_7series_v2_1_ecc_merge_enc | 688.224609 | 688.224609 | 688.224609 | 688.224609 |

原始路径为 `A/output/ablation_analysis/{REPORT.md,records.json,paired_results.json,evidence_hashes.json,completion.json}`。另有[补充源码与实际兼容核对](../../provenance/gpu_joint_ablation_source_audit.json)：48点的420个模块记录引用与实际快照/Git一致，12个默认完整轨迹（全部请求选择、状态、成功PO/delay/area）也与F冻结2257998相同。A在eda2/A40、F在eda/RTX4090，不能跨这两个作业比较runtime；通道控制的时间比仅来自A内部同分配配对。

## 来源分组、阶段成本与真实差异

D的 `output/delivery/{source_groups.json,stage_summary.json,REPORT.md,accounting.json,provenance.json,completion.json}` 汇总正式数据。来源分组如下；各来源只有2–4个设计，不作总体泛化：

| 来源 | seed配对 | 默认/旧版delay比 | 旧版/默认中位wall比 | 默认/random delay比 |
|---|---:|---:|---:|---:|
| ACE | 9 | 1.005074 | 1.048335 | 1.039999 |
| OpenABC | 12 | 0.999932 | 1.007763 | 1.031950 |
| forgeEDA | 12 | 1.002048 | 1.061911 | 0.996610 |
| iwls2022 | 6 | 1.002577 | 0.858276 | 1.079805 |
| iwls2024 | 9 | 0.987454 | 1.108092 | 1.012605 |

AND四个规模区间的正式配对已在 `S/.../statistics.json:scale_groups_vs_legacy` 保留；不是仅选获益来源报告。新方法在iwls2022这两设计更慢，对random的质量也更差。

阶段中位数（s；括号为有该阶段记录的方法点数）：

| 方法 | 准备 | 梯度提案 | 代理校准 | CEC+mapping |
|---|---:|---:|---:|---:|
| 旧STE CPU | 4.5572 (48) | 1.1245 (42) | 0.3058 (42) | 6.2374 (48) |
| 新联合CPU | 4.4588 (48) | 6.8917 (42) | 0.3133 (42) | 7.1482 (48) |
| 新联合auto | 4.7018 (48) | 1.2567 (42) | 0.2907 (42) | 7.2512 (48) |

各阶段为inclusive且中位数不可相加，不能拼成完整wall。随机准备中位1.6251s，未记录/不适用的ML阶段不补成零。GPU把新提案成本降到接近旧版，但准备和真实EDA仍昂贵，新方法还实际做了更多成功mapping；主面板缺少整体显著加速的证据。1311次实际mapping中，1119次进入审计观察/曲线，另外192次为四种梯度方法单独测的original；后者成本全部计入wall/调用数，没有作为额外搜索收益。CEC总调用含额外cut上下文证明，不等同于mapping数。

- ALU_mod三seed中位delay：新1556.3812、旧1585.0527、random1585.3152ps，是本面板的有效局部结果；不代表各seed或各设计都胜出。
- ex27：新1254.3214、旧1180.3484、random1171.9586、native ABC1087.5774ps，说明局部近似/反馈并不保证有效方向。
- DSP：新4122.6938、random3635.9080、native CULS2837.2060ps；不能因GPU提案更快就忽略搜索质量差距。
- sasc、usb_funct和两个mig设计有大量no-op或相同结果，不能把三个seed当三个独立电路，或把无提案设计当作梯度贡献证据。

可研究的问题由本轮负结果限定：如何减少原增益政策/顺序安装造成的无效请求与修复成本；如何避免冻结局部敏感度把优化带向真实mapper的不利选择；以及怎样降低CPU特征/校准/adjoint成本。这里没有引入新的大型local-to-global模型，也没有把这些问题当成已实现方案。

最终图表在 `D/output/delivery/figures/`：逐设计quality-runtime、实际时间曲线、GPU贡献和最终来源，各有PNG/PDF。曲线顶端遮挡与y轴offset已修正并人工查看；另外三PNG与已经查看的S原图逐字节相同，PDF各1页可读取。D仅做后处理，原图和所有实验数据不覆盖；其共享Matplotlib安装有旧style键警告，输出正常且未修改环境。

| 图 | 查看 | 下载 |
|---|---|---|
| Delay–Runtime | [PNG](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/delay_runtime_by_design.png) | [PDF](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/delay_runtime_by_design.pdf) |
| 实际完成时间曲线 | [PNG](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/validated_delay_time_curves.png) | [PDF](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/validated_delay_time_curves.pdf) |
| GPU贡献 | [PNG](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/gpu_contribution_by_design.png) | [PDF](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/gpu_contribution_by_design.pdf) |
| 梯度控制与最终来源 | [PNG](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/gradient_control_and_final_sources.png) | [PDF](/data/wushangge/CULS_LiteCut/runs/20261113T081810Z-joint-final-delivery-d22ea3ab/output/delivery/figures/gradient_control_and_final_sources.pdf) |

## 结论边界

改进实现、统一主面板和GPU独立对照已有可复现证据；目前只达到对旧版的预定QoR非劣，没有达到整体1.5倍加速，也没有建立梯度胜过简单搜索的优势。这是有完整负结果的研究实现，不是已成立的论文性能结论。

16输入来自已经分析的905集合；三seed、少量来源和规模区间不建立未见泛化或显著性。当前测量仍是ABC逻辑映射/Liberty timing，没有布局布线signoff、精确可微mapper、唯一跨网表逻辑路径身份或AGD论文严格复现。内存数据按上节的CPU进程/组件Torch/未记录的整闭环显存分别解释。
