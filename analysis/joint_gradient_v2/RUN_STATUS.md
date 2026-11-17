# 运行、阻碍与恢复

状态：`PARTIAL_PENDING_RESUME`。2026-11-17按用户最新要求进一步删除旧运行后，估算释放1.41TiB，共享磁盘约2.0TiB可用，见[LATEST_ONLY_CLEANUP_20261117.md](LATEST_ONLY_CLEANUP_20261117.md)。研究尚未自动续跑；尚未执行新代理/新选择的真实电路消融，门槛未评估，不是算法验收完成。下文保留此前存储中断的调查；两份未完成快照已删除，源工作树已恢复为完整、干净状态。

## 来源与已执行任务

| 工作 | 源码 | Slurm | 结果 |
|---|---|---:|---|
| 905证据审计、家族划分 | 5b92e878 | 65961 | COMPLETED；1分29秒 |
| 12×16预测冻结 | 4593837d | 65962 | COMPLETED；2分46秒；0新增Mapping |
| 首次测量启动 | 4593837d | 65964 | FAILED；启动器误读prepare命令，FileExistsError；0新增Mapping |
| 修正启动器后的测量 | 4593837d | 65967 | FAILED；3分57秒；92完整+1记录不完整Mapping |
| 联合选择单元检查v1 | 未提交校验快照 | 65965 | 10项中1项错误：测试模型漏width属性 |
| 联合选择单元检查v2 | 未提交校验快照 | 65968 | 10/10通过 |
| 加入端口代理检查v3 | 随后提交为77030289 | 65969 | 13/13通过 |
| 续跑协议检查 | f84da962 | 轻量stdlib，本地 | 1/1通过；不重复已有Mapping、不将残缺记录晋升PASS |

大计算全部在`/data`通过Slurm。主测量任务4CPU、32GiB、1GPU；检查任务2CPU、4GiB。没有重跑A/B、没有后台新任务仍在运行。未将旧方向导数或STE100结果计入本轮组合统计。

运行目录：

- 审计：`/data/wushangge/CULS_LiteCut/runs/20261117T042043Z-joint-gradient-audit-5fc79cac`
- 组合准备及部分测量：`/data/wushangge/CULS_LiteCut/runs/20261117T042705Z-joint-gradient-panel-00895eff`
- 13项检查：`/data/wushangge/CULS_LiteCut/staging/joint-gradient-unit-v3`
- 不完整快照（禁止作为可执行source）：`20261117T044007Z-joint-gradient-ablation-14048a96`、`20261117T044211Z-joint-gradient-ablation-6d53ff33`。

主测量目录`input/measurement_predictions.json`保存测量开始前的12份预测SHA256；`run.json`保存两次启动历史。修正启动器独立存为`measure_worker_v2.py`，没有覆盖失败日志。

## 存储故障

885/mask03已经产生CEC、GT_META/GT_PO和mapped.v，但`mapping.command.json`、`result.json`、`observation.json`写入失败。只保留为`INCOMPLETE_RECORD`；真实命令结束状态和耗时不补造，正式指标排除。下一次继续不重新Mapping此组合，因此全部请求仍最多180次Mapping。

本轮输出约2.1GiB。故障后`df`显示约14GiB可用、约2840万空闲inode，但源码快照创建和小文件创建仍连续`ENOSPC`。挂载为XFS、`noquota`。现有证据未定位到具体分配组/元数据或其他写入者，不能断言某个外部任务导致故障；8GiB的全局空闲量检查不足以保证这台近满文件系统能创建文件。

未删除历史运行、旧日志或用户文件，也未转移计算到`/home`。已请求释放/扩容`/data`或提供其他获准的数据运行目录；建议至少100GiB余量再继续。

## 恢复顺序

1. 解决存储并验证目录和小文件可创建。干净源工作树`/data/wushangge/CULS_LiteCut/staging/joint-gradient-source`的checkout因ENOSPC缺失`tests/test_panel_resume.py`，应先恢复其当前提交的该文件并核对clean状态，不使用残缺快照。
2. 用`prepare_run.py`从当前已提交分支准备新的完整快照，记录hash、环境、资源、目的。检查Slurm预算后提交；保留原run目录全部证据。
3. 对原12个case顺序调用新快照中的`joint_gradient_panel.py --phase resume`，使用原reference-run、原output和固定gpuls。完整6例直接跳过；885复用已完成Replay和前两次Mapping，mask03标记不完整，执行其余12个；另5例各15个。只新增87次Mapping，不能重做已有93次。
4. 独立运行`joint_gradient_ablation.py --panel <原case> --output <新run/output/case_NNNN/ablation.json>`。不得读候选测量后改选择；此入口本身不打开measurements。不同方法的选择冻结后，用汇总脚本关联已有标签。
5. 将完整与残缺面板分开计算排序、方向、搜索差距、真实QoR及面积。单独报告885的缺失，不通过重测或删除记录让数据看起来完整。
6. 根据开发消融决定修订/离线拟合或独立24验证；按计划门槛决定100及905扩展。目前这些阶段均未开始，不能默认门槛已通过。

当前代码是研究分支上的原型；没有切换生产优化入口，没有推送远端或覆盖main。下一阶段仍需要端口组合与显式候选图的更广泛对照、真实电路消融、必要的共享参数拟合和完整Runtime测量。
