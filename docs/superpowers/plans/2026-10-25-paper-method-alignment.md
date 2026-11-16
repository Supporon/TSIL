# TSIL 原始论文方法一致性修复计划

> **For agentic workers:** 按任务执行，使用 test-driven-development；requesting-code-review 独立复核。此任务是恢复用户指定论文已有定义，不新增研究方法。

**Goal:** 从 GitHub 新克隆的 `5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff` 出发，使实际训练入口遵循原始方法，并保存修改前后验证证据。

**Architecture:** 原始价格/收益与 Alpha158 输入分开；按样本缓存因果描述符；训练时按当前批次整体 Frobenius 归一化、直接计算残差矩。固定 beta 属于训练器而不是 backbone。逐日验证与分阶段选择使用相同指标。

**Tech Stack:** Python / NumPy / pandas / PyTorch / Qlib / pytest。验证运行环境另行记录，不声称与论文的旧版本环境一致。

## 权威来源和决策

- 论文：工作区 `1015_TIST/ori/Elsevier_NN (1)/methodology.tex` 与 `experiments.tex`，不修改这些文件。
- 指定会话：`6acf7bea-5804-83ea-b6c1-35578698d6a1`，已通过 read_thread 和只读浏览器补全分析，作为待核实的线索。
- 旧审计清单：`1015_TIST/TIST-review-code-1015/CODE_EVIDENCE_AND_ISSUES_ZH.md`，逐项与新克隆代码复核。
- 仅本地分支 `codex/paper-method-alignment`；交付源码与补丁，未经明确授权不直接覆盖 GitHub main。
- 明确的论文要求直接修复；论文未定的数值设定、交易执行时间与历史表格来源必须注明。
- 默认复现设定（论文未给全数值）：K=[5,10,20]，W=[10,20,40]，Wc=Wp=w=20，T=20；时间戳使用 weekday/month 两组固定 OneHot；市场收益使用逐日可用成分股收益均值；交易日编号以数据提供方日历起点为0。窗口已向用户异步核对。
- 预测标签保持 t 到 t+1 原始收益；实际交易保留 Qlib 的下一交易日执行，不把事后已知收盘价当成可立即成交价格。

## Task 1 — 谓词与损失（先写失败用例）

Files: `src/temporal_predicates.py`, `src/tsil_loss.py`, `tests/test_paper_math.py`.

- [x] 用独立 NumPy 公式检查六种原始描述符；已知正弦市场返回2维sin/cos，phase使用交易日modulo，momentum使用log价格比，vol/dist使用总体标准差。
- [x] 同时重排样本及元数据，检查描述符随之重排；追加未来行情不能改变历史描述符。
- [x] 先记录旧实现失败，再实现最小修复；不从标准化Alpha158推断原始价格。
- [x] 用手算与autograd检查：

```python
e = prediction - target
expected = e.square().mean()
for name, phi in descriptors.items():
    p = phi.detach() / (phi.detach().norm() + 1e-8)
    expected += beta[name] * ((p.T @ e) / e.numel()).square().sum()
```

验证 beta=0 为 MSE、多个谓词可加、梯度与论文 Proposition 1 一致、没有可训练beta/alpha、生产损失不构造 B×B 矩阵。

## Task 2 — 原始行情与样本对齐

Files: `src/predicate_data.py`, `src/dataset_ori_onehot.py`, `tests/test_paper_data.py`.

- [x] 写原始行情→缓存描述符→batch 的测试，验证乱序、跨股票窗口、完整历史、尾批。
- [x] `_create_ts_slices` 起点不得越过当前股票起点。
- [x] 从 Qlib `$close` 独立读取历史，窗口包含t且只含<=t；同日cycle保持一致。
- [x] 统一样本可用性，拒绝缺失/非正价格及不完整历史；不以零补造原始价格/收益。
- [x] split边界剔除目标日期落入下一split的样本；eval保留全部尾批。

## Task 3 — 实际训练、评估与三种对比形态

Files: `src/model_backbone_phitp.py`, `src/models/*.py`, `tests/test_training_alignment.py`.

- [x] 测试9个发布backbone均可实例化及训练一步，先排除当前引用未发布模块引起的导入故障。
- [x] TSIL使用固定 `sigmoid(penalty_alpha)`，移除backbone中的损失alpha；不改网络forward架构。
- [x] Original=MSE；Feature将同一个原始描述符沿时间步复制后拼接；TSIL仅训练时取描述符。
- [x] 验证按日期聚合所有预测后计算IC/RankIC；MSE按样本计；提前停止20epoch，默认不用额外参数平均。
- [x] 推理计时采用同步的forward耗时，输出ms/sample；支持单样本尾批。

## Task 4 — 配置、种子和选择协议

Files: `configs/f158_csi{300,800}/*.yaml`, `src/run_utils.py`, `train.py`, `search.py`, `tests/test_protocol.py`.

- [x] 18份配置均为 `Ref($close,-1)/$close-1`；去除CSRankNorm；输入改CSZScoreNorm+Fillna；固定penalty_alpha。
- [x] 每个seed真正写入模型配置和独立输出目录，避免目前仅报告名变更且互相覆盖。
- [x] 先用2025与Original搜索网络参数并冻结，再针对每个seed/predicate搜索5个固定beta；搜索阶段不prepare/evaluate test。
- [x] 保存候选验证IC、选择manifest、最佳checkpoint与配置；跨seed平均验证IC选主表predicate，ones不参与（论文未明确跨seed聚合，作为公开约定记录）。

## Task 5 — 回测与执行边界

Files: `train.py`, `src/run_utils.py`, `configs/**/*.yaml`, `tests/test_protocol.py`.

- [x] 正式报告只用Qlib执行器收益，不用预测标签均值替代交易收益。
- [x] buy/sell成交价明确0.01%滑点，费用保持0.05%/0.15%，trade_unit=100；保留停牌/涨跌停过滤。
- [x] 报告区分策略绝对收益与超额收益，记录实际执行时点；不伪造历史表格已重现。

## Task 6 — 复核与交付

- [x] 完整pytest、编译检查、9backbone集成验证、独立审查。
- [x] 对原始论文逐条复查，记录每条修复的公式/文件位置、修改前后行为、证据和限制。
- [x] 输出中文报告、完整patch、基线/修改哈希与可重跑命令。
- [x] 有真实市场数据时执行受影响实验；数据不可用时明确区分数学/合成集成通过与真实实验未重跑。

验证命令：`.venv/Scripts/python.exe -m pytest tests -q`，`.venv/Scripts/python.exe -m compileall -q src train.py search.py`，`git diff --check`。

## 完成时补充

- 用户确认关系模型按2/4/8完整日搜索；其他模型保留256/512/1024样本。
- 独立审查追加修复未来标签决定关系输入、完整连续T20历史、MASTER Feature、跨模板骨干配置。
- 验证结果见 docs/verification/RESULTS.md；真实数据未提供，因此全量市场实验未执行，已在报告明确。
