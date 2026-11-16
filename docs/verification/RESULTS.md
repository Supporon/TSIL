# 最终验证记录（2026-10-26）

工作目录：`TSIL_paper_aligned_20261025`；解释器：`.venv/Scripts/python.exe`。

| 检查 | 实际结果 | 证据/边界 |
|---|---|---|
| `python -m pytest tests -q` | **94 passed，25 warnings，11.42 s，exit 0** | [完整输出](pytest_final.txt)；涵盖最终连续历史窗口和未来缺失回归 |
| `python -m scripts.verify_synthetic_pipeline` | **exit 0** | [机器可读结果](synthetic_verification.json)、[完整运行输出](synthetic_run.txt) |
| `python -m scripts.compare_baseline` | **exit 0** | [真实基线反例](before_after.json)；需要保留 Git 基线历史 |
| `python -m compileall -q src train.py search.py scripts` | **exit 0** | Python 编译检查，无需真实行情 |
| `python -m pip check` | **No broken requirements found，exit 0** | [环境与 freeze](environment.txt) |
| `git diff --check` | **exit 0** | 换行转换提示是 Git CRLF 配置提示，不是 whitespace error |
| HTML 浏览器检查 | **9 个折叠区均可开关，6 行对照表，无水平溢出** | Tabbit 实际打开本地文件；已查看截图，注意事项左侧边框 4 px |
| 独立最终复核 | **56 项受影响测试通过，无剩余 Critical/Important 阻断项** | [独立复核记录](independent_review.md)；未扩大为真实市场结论 |

pytest 的警告来自 Matplotlib/pyparsing 兼容接口、PyTorch weight_norm/Transformer 提示、单层 RNN dropout，以及原有 logger.warn 用法。它们没有被全局屏蔽，不属于失败的测试。

## 合成集成验证的范围

实际生成并使用 Qlib 二进制数据提供方：56 股、120 个业务日、另加 benchmark。不是 mock Alpha158 或假回测返回值。

- 真实 Alpha158 → CSZScoreNorm → 数据集 → LSTM 一轮训练 → 验证选择 → 保存并重载 checkpoint。
- Original、Feature、TSIL 三种形态的测试预测索引完全相同，各 504 条（56×9）；跨 split 的末端目标被剔除。
- 10 个交易日的 Top-50 / Drop-5 实际执行，成本/换手非零，实际持仓满足 100 股整手。
- 使用 Qlib Exchange 验证买/卖报价为 close×1.0001/0.9999；验证缺 close 的停牌过滤与 0.095 阈值的涨跌停代理过滤。
- 标签逐项对照独立原始 close 比值。
- 真实缩小版搜索：一个网络候选、一个 seed、momentum 的两个 alpha 候选，选择阶段 `test_evaluated=false`。随后通过正式 `train.main` checkpoint 入口输出测试 metrics.json 和回测报告。

这些结果证明接口与实现可执行，**不证明真实市场上的 IC、AR 或原论文表格成立**。本次没有 GPU 延迟数据，没有训练全量九模型真实 CSI300/CSI800 实验。

## 修复过程中实际遇到的失败

1. 基线损失反例得到 2.375，与论文的 3.0625 不符；修复后相符。
2. [审查回归的失败输出](review_regressions_before.txt)：关系模型批量设定被覆盖、多骨干模板字段缺失、MASTER Feature 输入维数不匹配；修复后对应测试通过。
3. 真实 Qlib 集成在 Joblib 1.6.0 上失败：`ParallelExt` 没有 `_backend_args`；固定 Joblib 1.4.2 后真实加载通过。
4. checkpoint 最终报告曾因 NumPy float32 JSON 序列化失败；转换为 Python 标量后正式入口通过。
5. 独立真实 GAT 反例：只改变 t+1 close/标签可用性，旧的 `_learn` 截面会改变 BBB 在 t 的预测；使用 `_infer` 并在 forward 后 mask 标签后，两种未来缺失回归均通过。
6. 暖启动原测试曾预期只有 5 个因子日仍产生预测；按完整 T=20 输入约定修正为 0，并新增缺少中间交易日的窗口回归，最终完整套件通过。

## 版本确定性

重新克隆时的基线为 `5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff`。交付包记录本地修改提交及 SHA256。收尾时额外执行的 `git ls-remote` 因连接重置/超时未成功，因此不声称已再次确认 2026-10-26 当时远程 main 未变化；该网络问题不影响已克隆基线及本地测试证据。

真实数据提供方 `qlib_data/cn_data` 在本次检验时不存在。需提供确定版本的数据后，按 README 的冻结网络/验证选择/最终测试流程重新实验。
