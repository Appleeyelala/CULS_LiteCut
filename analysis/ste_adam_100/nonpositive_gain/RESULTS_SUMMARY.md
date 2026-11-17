# 结果汇总

固定100候选检查终态：{'PASS': 100}；具有合法非零候选 100/100；原17例解决 17/17，GO_905=true。

候选总数：旧raw 721771 → 新raw 42880087；旧feasible 721771 → 新feasible 39307930。新合法正/零/负分别 721771/5046277/33539882。覆盖检查完整冷进程时间中位数 19.546099s，包含指定代表例的Replay/CEC探针，不是完整Ours优化Runtime。

探针实际完整安装/CEC：{'PASS': 12}。候选存在不能推断Adam会选择它，也不能推断真实Delay改善。候选域扩大增加Boolean、可行性和后续LiteCut/代理的实际工作量，本轮没有做Runtime优化。

新版905实际终态 905/905，{'PASS': 851, 'FAIL': 7, 'TIMEOUT': 6, 'AREA_LIMIT_EXCEEDED': 41}。确认零候选 0，候选域不可判定 1。旧94变化见 full905/FULL905_ZERO_CANDIDATES.md。

实际异常记录：CUDA OOM报告ID [16, 24, 827]；Replay失败ID [40, 480, 597, 772]；超时ID [22, 59, 108, 541, 552, 578]。完整错误和已完成边界见CSV及原始记录。超时前未完成Boolean/Feasibility的电路保留未知，不补零；日志报告的OOM不等同于已确认其瞬时资源根因。

选择非空组合 895 个，完整Replay接受 891 个（以非空选择为分母的比例 0.9955307262569832）。完整安装、CEC及Mapping成功的非空候选 891 个，相对自己付费原图真实Delay胜/平/负 427/12/452；候选/原图Delay比几何均值 0.9607774130060962。真实Area超1.05的数量 41，单列不伪装可行改善；没有按真实Delay回退。

Runtime中位数（有值分母见JSON）：{'measured_full_runtime_s': 119.06824380112812, 'second_mapping_abc_time_s': 1.6167946108616889, 'optimization_runtime_s': 117.54762948397547}。optimization_runtime_s = measured_full_runtime_s - second_mapping_abc_time_s；无第二次Mapping扣0，缺失实际命令时钟时不估算。此为研究记账指标，不是已测一次Mapping部署性能。905按独立冷进程分片执行，A/B为历史测量，没有重跑，也不作严格同负载Runtime配对。

固定100子集：终态 100/100，状态 {'PASS': 96, 'AREA_LIMIT_EXCEEDED': 4}；非空选择 99，完整Replay 99，完整CEC/Mapping候选 99，其中面积合规 95。旧STE100原始候选（回退前）为 67 个，面积合规 67 个；用相同的原始候选口径比较，不与旧版按Delay采用的31个混算。新100原始候选相对自付原图真实Delay胜/平/负 [37, 2, 60]；Runtime三项中位数 {'measured_full_runtime_s': 106.22525670286268, 'second_mapping_abc_time_s': 1.6169976554811, 'optimization_runtime_s': 104.67238010512665}。

100子集结果直接来自本次905同ID的实际Ours运行（不是另起100次重复实验），附历史A/B参考。失败/超时及缺失均保留，见CSV与运行状态JSON。

存储说明：磁盘余量接近150 GiB且剩余输入仍约257万个AND节点时，在既有Slurm作业 65901 的步骤 2 中，以1 CPU、1 GiB、0 GPU归档了 16 个**已终态电路的LiteCut缓存文件**，释放 64.528 GiB。每个gzip解压后的原始字节SHA256均与压缩前一致；算法源码、原始结果和计时记录保持原值。并行存储I/O可能影响尚未结束电路的实测Runtime；未扣除、修正或估算这种影响，不作严格同负载性能比较。见 `STORAGE_ARCHIVE_INDEX.json` 和 `STORAGE_ARCHIVE.md`。
