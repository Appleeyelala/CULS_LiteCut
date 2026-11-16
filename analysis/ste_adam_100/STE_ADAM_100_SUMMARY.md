# STE + Adam 随机100电路真实结果

本次仅整理已完成正式运行，未重跑或调参。源代码 Commit：`9a199bf0ea0dcf20bf55e40da8caa8b8bbff5fac`；Slurm 65534。样本为原905全集无过滤抽取100个：`sorted(random.Random(20261116).sample(range(905),100))`；优化种子20261117。三个方法各100点，共300个终态；smoke不计入。

A：原图Mapping。B：一次`strash; rewrite`后Mapping。Ours：自己付费的原图Mapping、冻结代理上的40步STE＋Adam、至多一次最终候选真实验证。

|方法|PASS / FAIL / TIMEOUT|真实Delay中位数 ps|完整Runtime中位数 s|
|---|---:|---:|---:|
|A|100 / 0 / 0|3123.414675|2.172834|
|B|100 / 0 / 0|3076.719850|2.702031|
|Ours|100 / 0 / 0|3090.977665|13.175791|

原始Delay中位数仅描述分布，不能相除作为配对收益。以下每项均基于相同100个电路逐对计算：

|配对|Ours胜 / 平 / 负|Delay Ratio几何均值|逐电路Runtime比中位数|
|---|---:|---:|---:|
|Ours/A|31 / 69 / 0|0.959272|5.281315|
|Ours/B|35 / 27 / 38|1.001070|4.045126|

Delay只来自同一ABC/ASAP7的`ABC_SCL_LIBERTY`真实全局时序（`map; topo; stime -p`），没有用LiteCut或代理预测填充。Ratio为Ours/基线；几何均值为`exp(mean(log(ratio)))`。胜平负容差为`max(2e-5 ps, 2e-6 × max(两者Delay))`。

Runtime为独立冷进程完整wall time，包含进程启动退出、输入准备、候选与模型准备、推理、优化、Ours自付原图测量、最终Replay/CEC/Mapping及必要输出；不含排队、一次性构建和汇总。三方法同机同配额，执行顺序轮换。

Ours的67个电路各2次Mapping，33个各1次，共167次，全部符合最多两次约束。83个有候选电路各40步CUDA优化，共3320次原生DAG backward；17个无候选跳过代理。80/83有可测logits更新；242、637梯度为0，836梯度极小，没有可测更新。

67个非空最终组合全部完整ACCEPTED、CEC PASS且满足1.05倍面积限制：31个真实Delay改善、2个持平、34个变差。后36个返回原图；另16个有候选但选择no-op，加17个无候选，共33个直接返回原图。最终输出CEC全部PASS；无候选验证的CEC标记NOT_REQUESTED，不伪造验证成功。

无方法失败、超时或必需指标缺失。逐电路记录保留全部回退和no-op；未请求候选的Delay留空。

相对A，Delay几何平均约降低4.07%；相对B约增加0.107%，尚未整体优于B。完整Runtime仍为A/B的约5.28/4.05倍。代理目标下降不等于真实Mapping改善，也未建立梯度优于其他搜索或同算法GPU相对CPU的加速证据。

本次STE100与历史905方向导数、多次Mapping反馈及在线校准版本独立；未修改旧905数据。详细算法见[实现说明](implementation/STE_ADAM_100_IMPLEMENTATION.md)，逐电路见[CSV](results/STE_ADAM_100_RESULTS.csv)，原始路径及SHA256见[证据索引](audit/evidence_index.json)。
