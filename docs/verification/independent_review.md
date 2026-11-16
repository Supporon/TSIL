# 独立代码复核记录

日期：2026-10-26。基线：`5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff`；审查对象为 `codex/paper-method-alignment` 的当前修改。

**结论：本次审查范围内没有剩余 Critical / Important 阻断项，可以作为方法对齐的本地代码交付。此结论不代表原论文真实市场实验已经复现。**

复核以原始 `methodology.tex`、`experiments.tex` 和已确认的实施计划为依据，覆盖六类描述符、固定 TSIL 损失、原始行情缓存、数据集切片、训练与逐日指标、九个发布骨干的接入、18 份配置、分阶段搜索和最终 checkpoint / Qlib 回测入口。复核者未修改生产源码；按主执行者要求补充了缺失标签与真实 GAT 因果性回归测试。

| 审查发现 | 等级 | 最终核验 |
|---|---|---|
| `_learn` 经 DropnaLabel 过滤，使 t+1 标签可用性改变 t 日关系模型的股票集合 | Critical | `src/dataset_ori_onehot.py:145` 使用完整 `_infer`；`src/model_backbone_phitp.py:355` 在完整 forward 后才筛监督残差。真实 GAT 的标签缺失、次日 close 缺失两个回归均通过，推理保留无标签 score。 |
| 关系模型搜索的 stock batch_size 实际被固定两日批次忽略 | Important | `search.py:20` 改为用户确认的 2/4/8 完整交易日搜索；配置保留选中值，manifest 标明批量单位。 |
| 多模型搜索复用单个模板，GAT / TCN 等缺少必需架构字段 | Important | `search.py:42` 从目标模型的发布 YAML 取得架构字段；LSTM、GRU 两种入口模板切换全部九模型的回归通过。 |
| MASTER Feature 增加输入维数后仍被截为 158 列，导致矩阵维数错误 | Important | `src/models/master.py:241` 在关闭 gate 时保留完整输入；九模型 Feature 前向检查通过。 |
| 不足 20 个因子日被补零，按行数取窗也可能跨成员日期缺口 | Important | `src/dataset_ori_onehot.py:175` 要求完整连续交易日窗口；暖启动及中间成员日缺失回归通过。排除暖启动端点属于公开实现约定，未称为论文明确指定的处理策略。 |

六类 raw descriptor 和批次 Frobenius 归一化后的加性残差矩损失与提供的论文公式相符。固定 beta 不在 backbone 参数中，计算损失不构造 B×B Gram。关系模型按日期分别前向；每日 IC 用于选择；末尾不足整批样本被保留。最终复核还检查了 `frozen_network_manifest` 复用已选网络、Feature 不重复搜索 beta，以及只从已选 checkpoint 执行测试的路径。

复核者最终实际运行：

```text
.venv/Scripts/python.exe -m pytest tests/test_paper_data.py tests/test_selection.py tests/test_training_alignment.py tests/test_final_evaluation.py -q
56 passed, 25 warnings in 16.19s

git diff --check
exit 0（仅 Git 换行转换提示）
```

此前亦运行数学、损失、配置等定向检查；最终全套 `94 passed` 和真实 Qlib 合成链路的主执行者证据另见 [RESULTS.md](RESULTS.md)。本次独立复核没有运行真实 CSI300 / CSI800 行情实验，也没有验证旧论文表格数值、真实收益优势或 GPU 延迟。年化、日相关标准差、市场收益定义、完整历史排除、交易阈值近似等未定约定已在 [方法对齐报告](../PAPER_ALIGNMENT_REPORT_ZH.md) 明确披露。
