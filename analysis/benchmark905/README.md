# 最新 MFFC GPU 梯度优化：905 电路分析包

本包只整理已存在的正式实验及其后处理，不执行 CULS、ABC、CEC、Mapping、LiteCut、优化器或新性能测试。

- 算法冻结源码：`ecc85e6ac82571caf9f1a6004098dbe976536256`；正式作业65245。
- 结果来自：`/data/wushangge/CULS_LiteCut/runs/20261113T191340Z-mffc-full905-frozen-6a45b8db`。
- 含成本的后处理：`/data/wushangge/CULS_LiteCut/runs/20261113T200926Z-mffc-final905-tool-cost-delivery-67ebf285/output/final_mffc905_delivery`，后处理源码`b1afd3480fedbd978203a73e3168ee1f61b4097b`。
- 905个电路均有A/B/Ours终态，共2715点；901个电路有三方法全部成功的真实Delay/Area/CEC，另外4个B失败而A/Ours成功。
- 方法状态：A 905 PASS，B 901 PASS/4 FAIL，Ours 905 PASS；没有方法超时。失败和no-op/回退全部保留。
- Ours候选组合请求3041个：成功3039、失败2，含905个no-op；另有905次Ours自付费original测量，不混作候选请求。
- 有评估前代理预测的请求2136个；有效global预测—真实Mapping配对2134条；有效逐PO配对6486582条，可从对应JSON的数组按PO位置展开。失败预测也保留。
- 有Student局部分数的设计811个，无候选94个；4296273个Decision、8402256个physical candidates，连同no-op共12698529条局部分数。没有candidate-local真实abc_delay标签，局部预测准确性的有效真值配对为0；不能把global Mapping Delay当作局部标签。
- 已保存梯度记录1589份、代理校准777份；仅保存过的梯度norm/方向排名可分析，不伪造完整梯度张量。

## 目录与推荐读取顺序

| 目录 | 内容 |
|---|---|
| `benchmark/` | 原始完整CSV/JSON/MD与三组图，2715终态JSONL；A/B成功的全部PO/真实测量JSON |
| `optimization/` | 905个`cases/NNNN/work/summary.json`、全部observation、精确replay、gradient/calibration和评估前预测存证；Decision表及candidate域字典 |
| `prediction/` | `global_request_pairs.csv`全部3041请求，`litecut_scores.csv`全部局部分数，去重的精确19维特征，以及Student缓存身份 |
| `runtime/` | 2715方法冷wall/工具调用、各阶段时钟、保存的原生GPU profiles及计时语义 |
| `config/` | 正式协议、冻结输入/工具/源代码小快照、原始GPU环境和实现说明、无ML依赖的JSON引用读取器 |
| `audit/` | 已有audited_records/installation_details、失败、原始证据hash索引、本次来源及文件清单、完整性与缺失说明 |

先读`audit/coverage.json`、`benchmark/BENCHMARK905_RESULTS.csv`与`prediction/global_request_pairs.csv`；再按circuit_id/request_index读取详细JSON。不要用CSV物理行数代替记录数，原始benchmark CSV含加引号的多行错误文本。

## 身份与指标

`circuit_id`就是冻结清单原始index（0–904），与input及SHA256联合核对。方法固定A/B/Ours。Ours的request_index=0为no-op，其余为同一原图上的独立组合，不是累计改图。最终source和observation_index原样保留；original_fallback的选择向量是null，不冒充no-op。

Decision由本电路内`decision_index`对应原始`(root,cut_slot)`和owner MFFC；choice=0是no-op，choice=k>0通过`candidate_domain_id`及`candidate_domains.json[domain_id][k-1]`找到subgraph_index。MFFC、Decision、cut-bounded region、candidate及不同节点命名空间不可混用。

严格区分：`predicted_local_abc_delay`是冻结Student局部代理；`predicted_delay_ps`是评估前多PO可微代理；`actual_mapping_delay_ps`才是真实`ABC_SCL_LIBERTY`。真实PO数组保存在对应evaluation/result.json，预测PO数组保存在observation的prediction_before_evaluation。失败真实值缺失不填0；no-op缺预测也不填0。

有效global配对要求请求PASS、有有限的评估前predicted delay，并有同一次实际CEC PASS、ABC_SCL_LIBERTY测量。所有training_observation_indices均小于当前request_index。此数据是在线选择/反馈后的样本，不是独立未见集。

## 压缩提取规则（不损失搜索记录的JSON字段）

原始summary/observation反复包含相同真实PO数组。包内保留每份原始JSON的全部语义字段，仅把重复的original、observations、replay_noop、selected.metrics、observation.metrics及评估前存证中的重复prediction/choices替换为`{"$bundle_ref":{"file":"相对包根路径","pointer":"JSON pointer"}}`。真实metrics JSON、gradient、calibration、replay、配置均逐字节复制。全部905份summary及3041份observation已逐一解引用，并与原始对象全字段相等比较通过。

解压后用：

```python
import sys
sys.path.insert(0, '/解压目录/config')
from bundle_reader import resolve_file
summary = resolve_file('/解压目录', 'optimization/cases/0047/work/summary.json')
```

局部分数不筛选：所有physical及no-op均保留。`feature_id`为feature_columns各列CSV中相同的0基数据行号；19维原始数值文本原样保留。feature_hash由这一行19个float32的小端字节SHA256精确重算，全部4,998,334行均已与原hash核对通过；无需重复保存可推导的哈希字符串。仅省略重复的cut_validation/replacement_dag_info证明与物化图细节，以及catalogue的大节点/冲突列表；它们不是局部或全局真实timing标签。原路径、原字节SHA256和提取规则在audit清单。连续值不舍入，不重做归一化，不重新推理。

按要求原样附上的reference_docs状态说明含旧工作背景，只作实现与历史边界参考；其旧905/16电路数值不属于本包机器数据或统计样本。本包没有读取旧905或16电路运行目录来提取任何实验结果。

## 完整性、缺失与成本边界

已核对905×3唯一终态、全部905份Ours summary、请求序号/选择域、预测存证、全部已有梯度/校准文件及版本。意外缺失0项，详见`audit/MISSING_DATA.json`；未记录的局部真值、完整梯度和显存峰值明确列为未记录，不重新生成。

四项B失败为codec/loco_top/mrc3/mrc4的native rewrite断言。两个内部请求失败来自BRAM_SSP同一次gpuls CUDA heap分配错误；它们的inclusive耗时重叠，共享命令只计一次。方法最终905 PASS不等于内部请求零失败。runtime分项可重叠，冷wall优先；原生GPU事件不是整个GPU或方法耗时。最新实验没有独立的全量同算法CPU/GPU对照，不能由A/B/Ours时间比推断纯GPU因果加速。

## ZIP大小与SHA256

**完整ZIP的精确大小和SHA256见ZIP同目录的外部`README.md`、`MFFC_GPU_GRADIENT_905_ANALYSIS_BUNDLE.zip.metadata.json`和`MFFC_GPU_GRADIENT_905_ANALYSIS_BUNDLE.zip.sha256`。** 将ZIP自身的SHA256再写入ZIP内部会改变被校验文件，形成自引用，故内部README只说明外部校验位置，不填伪造或过期hash。内部`audit/CHECKSUMS.sha256`校验所有其余成员；`audit/FILE_MANIFEST.jsonl`和`SOURCE_FILES.jsonl`记录包内文件及服务器原路径/原hash。整个ZIP在形成后逐成员读取、CRC检查并核对SHA256，再生成外部大小/hash。

## 无损列式特征与终态记录读取

`prediction/feature_vectors.schema.json`列出19个单列CSV；同一数据行就是同一feature_id，表头不计入行号。所有数值文本及行顺序逐项保留，未削减特征、候选或请求。读取例子：

```python
import sys
sys.path.insert(0, '/解压目录/config')
from read_feature_columns import iter_feature_vectors, iter_terminal_records
# 按需流式读取，避免一次载入全部499.8万组向量。
for feature_id, values_19, original_feature_hash in iter_feature_vectors('/解压目录'):
    pass
```

`benchmark/method_records.jsonl`中成功终态的data.metrics也采用同样的$bundle_ref，指向已包含的真实测量JSON。`iter_terminal_records`可还原原始对象；2711份替换前的metrics均逐字段核对相等。原始记录所有其他字段不变，包括失败、状态、耗时和身份。其余summary/observation引用仍由bundle_reader读取。

压缩整理只是编码变化；统计口径与coverage.json逐字节保持不变。第一版较大ZIP及其原始整理目录仍保留，没有覆盖或删除。转换来源、逐行核验与版本见audit/compaction_provenance.json。
