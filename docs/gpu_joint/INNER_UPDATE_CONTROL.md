# 内层更新的受控消融

`adaptive_no_inner_batch_auto` 复用主方法的输入、候选域、特征、真实观察、warmup/repair预算、设备选择与最终投影，唯一关闭项为当前提案调用的内层 backward/Adam。默认主方法不变。此控制用于区分优化更新与初始化、投影、反馈或随机 fallback 的作用；不是新增搜索算法。

小图分支使用原STE的两个初始化硬状态并由完整代理评分，不做任何内层更新。大图分支依然执行一次完整DAG adjoint并可拟合已有八维成功残差，使用相同32条anchor偏置初始化，在初始状态投影；不运行内层backward、Adam，也不伪造零梯度值。trace明确`inner_gradients_computed=false`，没有`total_gradient_norm`列。

因此它**不是完全无梯度方法**：代理校准与大图外层adjoint仍保留。真正不使用梯度/Student的基线为独立random。两者合起来分析内层更新和完整梯度搜索的作用，不能只凭一个消融把所有收益归因于精确timing梯度。

未见初始化点的来源写为`initialization_without_inner_updates`，不误标`joint_selection_gradient`；若初始化都投影回已测incumbent，原pipeline会按既有规则随机探索，并保留该来源。此现象可能源于原初始化很靠近anchor，属于本控制的实际行为，不自动证明梯度方向准确。每方法仍至多8请求，包含no-op、失败、repair与探索，不增加免费mapping。

此控制的数值/设备回归及真实多种子入口尚待验证。prospective最终协议在`config/gpu_joint_final_protocol.json`，其主面板/种子/质量与runtime阈值未因开发结果改变。所有最终真实数据需要按该协议单独执行与审计。
