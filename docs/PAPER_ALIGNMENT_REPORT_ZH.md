# TSIL 原始论文方法一致性修复报告

日期：2026-10-26（Asia/Shanghai）。本报告以用户指定的原始 LaTeX 方法定义为准，修复重新克隆的公开代码。**数学与软件链路验证通过，不代表原论文真实市场实验表格已复现。**

## 1. 来源、版本与交付边界

- 仓库：[Supporon/TSIL](https://github.com/Supporon/TSIL)，基线提交 `5b5ab68fb44741bae72381f015f3b2eb5b4eb3ff`。
- 本地目录：`C:\Users\郑增荣\Documents\TSIL代码\TSIL_paper_aligned_20261025`，分支 `codex/paper-method-alignment`。完整补丁与源码包随交付提供；本次没有更新 GitHub main。
- 权威论文：`1015_TIST/ori/Elsevier_NN (1)/methodology.tex`、`experiments.tex`。原始论文文件没有修改。
- 指定会话：**TIST投稿要求与转投建议**，ID `6acf7bea-5804-83ea-b6c1-35578698d6a1`。其结论与本地 `CODE_EVIDENCE_AND_ISSUES_ZH.md` 仅作为核查线索，逐条回到原始 LaTeX 和新克隆代码验证，未直接把旧审计文字视为方法规范。
- 原方法 SHA256：`DEED8C636A52866CAF251AFEE00126894D258E6D03B59BF326FCDDD91536D794`；实验章节 SHA256：`935D44C9C580AC09BEE8379DBFA98B45CACC4B13A27A31B58D33AA4F6EE22EE2`。

判断顺序是“论文明确公式 → 可重复反例 → 实际入口修复 → 回归测试 → 独立复核”。窗口和关系模型批量的未定细节已向用户确认，见第 4 节。旧 checkpoint、旧标签和旧表格数值不能直接视为修复版的结果。

## 2. 修改前后逐项对照

| 项目及论文依据 | 修改前的实际行为 | 修改后的行为与文件 | 验证证据 |
|---|---|---|---|
| 原始次日收益；Methodology Eq. return_ratio | 标签为 t+1→t+2，学习标签再经过 CSRankNorm | 18 份 YAML 均为 `Ref($close,-1)/$close-1`；移除标签排名变换 | 配置测试；真实 Qlib 标签与独立原始价格比值逐项对照 |
| 每日横截面 Z-score；Experiments Datasets | RobustZScoreNorm 时间统计 | 全部配置改为 CSZScoreNorm + Fillna | 18 配置核查；实际 Alpha158 初始化 |
| 原始行情描述符；Methodology §3.3 | 把已标准化的 Alpha158 因子张量当作价格/收益来源 | `predicate_data.py` 独立读取原始 `$close` 与已实现收益；批次仅取缓存 | 独立 NumPy 数学 oracle；未来追加不变测试 |
| Timestamp Eq. timestamp_predicate | 最后一步整数日期字段，非 OneHot | weekday 7 类 + month 12 类，固定 19 维拼接 | 固定维数、类别边界、批次重排测试 |
| Cycle Eq. cycle_predicate | 沿随机 mini-batch 行轴 FFT；按行号 ceil 分组 | 逐日市场已实现收益时间 FFT，排除 DC；`q(t) mod tau` 的 sin/cos | 已知正弦周期、同日同值、重排等变测试 |
| Phase Eq. phase_predicate | 因子序列 FFT，局部位置 ceil 分组 | 个股原始收益时间 FFT，过去 T 个全局交易日编号 modulo 相位 | 两个已知周期的逐元素 oracle |
| Momentum Eq. momentum_predicate | 因子相对变化再跨特征求均值；T=20 静默丢弃 k=20 | `log(p_t/p_(t-k))`；独立读取足够历史；保留 5/10/20 | 指数增长价格的手算 [0.05,0.10,0.20] |
| Volatility / Distribution 公式 | 因子时间标准差和跨特征均值的分布；40 日窗口消失 | 原始收益的总体 std（ddof=0）、偏度、超额峰度；10/20/40 均生效 | NumPy 总体矩 oracle；缺失/非正价格拒绝 |
| 加性目标；Eq. tsil_neural_objective / Proposition 1 | `eᵀ[tauhat I+(1-tauhat)P]e/B`，基础 MSE 被缩放 | `MSE + Σ beta_s ||Phi_tilde_sᵀ e/B||²`；多谓词独立可加 | 手算数值和 autograd 梯度对照；beta=0 回到 MSE |
| 固定 beta；Experiments Implementation | `alpha` 是各骨干的可训练参数，Adam 会更新 | 移除损失 alpha 参数；训练器从固定 alpha 一次性计算 sigmoid | 九骨干无 alpha；搜索 alpha 网格；固定权重测试 |
| 不构造 B×B Gram；Optimization Implementation | 训练显式构造 I/P | `tsil_loss.py` 直接计算 d×B 与 B 向量的残差矩 | 数学等价测试；生产调用链不使用 diagnostic Gram 接口 |
| 真实历史窗口与因果性 | 新股票的最初序列可读到上一股票末尾；行数窗口可跨日历缺口 | 起点限制在当前股票；要求连续 20 个交易日特征、完整原始历史 | 跨股票反例；成员间断和暖启动测试 |
| 标签可用性不能决定关系输入 | 使用 `_learn`，DropnaLabel 提前删掉缺失次日标签的股票 | 使用 `_infer` 全特征时间线；完整前向之后才 mask 有限标签；保留无标签 score | 真实 GAT：只改未来标签/未来 close，t 日预测保持不变 |
| 按日期构造股票关系 | GAT/MASTER/LSR-IGRU 可在混合日期 batch 内相互注意 | 每个完整日截面独立 forward，再合并多日残差计算损失 | 改动另一日期输入，不改变当前日期预测 |
| Original / Feature / TSIL 三形态；Experiments Compared Forms | Feature 路径未完整提供 | 同一 raw descriptor 在每个时间步拼接；TSIL 只进入损失；MASTER 关闭 gate 时保留全部输入列 | 九骨干 Feature 前向检查；拼接张量逐元素对照 |
| 验证逐日 IC / patience=20 | 验证按 batch 相关平均；部分配置 patience=10；默认参数平均 | 按完整日期 Pearson/Spearman 后取均值；patience=20；默认关闭参数平均 | 每日 IC=1、混合 IC<0 的反例；尾批和 checkpoint 测试 |
| 实际种子与输出隔离 | `main(seed)` 的 seed 主要影响文件名，模型仍用 YAML seed | Python/NumPy/Torch 实际 seed；输出目录含配置摘要和 seed | seed 与路径测试；搜索保存完整配置 |
| 分阶段选择、测试集隔离 | 联合搜索/不同口径难以还原选择关系 | seed2025 Original 先选网络；冻结后逐 seed/predicate 选 alpha；Feature 可复用同一 network manifest | 可控独立分数表验证顺序；真实缩小搜索不 evaluate test |
| 真实交易收益 | 辅助路径直接对处理后的标签构造组合收益；与 Qlib 结果混用 | `train.py` 仅用 Qlib 实际成交后的 return/cost/bench；另存真实持仓 | 合成提供方的实际 Top-50/Drop-5 成交、成本与整手检查 |
| 滑点和交易约束 | YAML 没有显式滑点，整手依赖平台默认 | 买价 close×1.0001，卖价 close×0.9999；显式 trade_unit=100 | 实际 Exchange 报价、停牌/阈值限价过滤测试；不是仅查 YAML |
| 每样本推理毫秒 | 秒/批次，缺 CUDA 同步 | warm-up、perf_counter、CUDA 同步后总耗时×1000/预测样本数 | 单样本尾批与单位检查；CPU 实际计时 |
| 可运行性 | 引用了未发布骨干模块；多模型模板缺专用字段 | 仅导入九个已发布骨干；按目标 YAML 提供架构字段；固定 Joblib 兼容版本 | 实际导入、一轮训练、跨模板九骨干测试、pip check |

## 3. 核心公式及可重复的差异证据

设有监督的有效样本数为 B，`e = prediction - raw_label`。每个描述符先 detach，在当前监督子批上做整体 Frobenius 归一化：

```python
phi_tilde = phi.detach() / (phi.detach().norm() + 1e-8)
g = phi_tilde.T @ e / B
loss = e.square().mean() + sum(beta_s * g_s.square().sum())
```

beta 不参与优化。多个谓词各自归一化、各自加权，不将它们拼成一个共同归一化矩阵；这保留论文对独立谓词的定义。关系模型的上下文可以含缺失未来标签的股票，但损失的 e、Phi 和 B 只包含有有限标签的样本。

以下数值由 `python -m scripts.compare_baseline` 从 Git 中的**真实基线源码**提取执行，详见 [before_after.json](verification/before_after.json)。

| 反例 | 修改前 | 修改后 | 含义 |
|---|---:|---:|---|
| e=[1,2]，ones，beta=0.5 | 2.375 | 3.0625 | 论文手算 MSE=2.5，附加项=0.5625 |
| Cycle：同时重排数据与元数据 | 原 Gram 与重排后的预期 Gram 最大误差 0.21666665 | 描述符重排误差 0 | 修复随机 batch 行序被当成时间 |
| T=20 的 Momentum / Volatility 维度 | 2 / 2 | 3 / 3 | 不再静默丢弃 20 / 40 日窗口 |
| 第二只股票第一个样本的历史股票名 | A,A,B | B（底层切片） | 切片不跨股票；生产端进一步排除不足 T 的端点 |
| 日内完美排序、两日整体偏移相反 | pooled IC 为负 | daily IC=1 | 验证目标不再由任意批次划分决定 |

Cycle 的两个误差量分别检查旧 Gram 和新 raw descriptor 的等变性，不把两者解释成同一量纲的性能指标。以上均是确定性软件反例，不是金融收益提升证据。

## 4. 已确认配置与必须披露的实现约定

**用户明确确认的两项：**

1. Momentum `[5,10,20]`；Volatility `[10,20,40]`；Cycle、Phase、Distribution 默认 20 日。原始历史至少覆盖 40 个已实现收益，即 41 个 close；Alpha158 lookback 仍为 T=20。
2. GAT、MASTER、LSR-IGRU 的批量候选为 **2/4/8 个完整交易日**，日内独立前向再合并损失；其他模型仍搜索 **256/512/1024 条 stock-date 样本**。manifest 明确记录批量单位，不把失效的 stock batch_size 宣称为关系模型已搜索的参数。

**论文未完全给定，本次明确采用：**

- Timestamp 选 weekday/month 两组 OneHot；没有伪造节假日字段。
- 市场收益采用当日历史成分股中有有效价格对的等权平均；它不是指定指数的收益。全局 q 从提供方完整交易日历起点计数；截断提供方历史可能改变相位，必须保存数据版本。
- FFT 排除 DC；并列最大值取第一个频率。零信号时也按此确定性规则返回有限值，不宣称从常量信号识别出了真实周期。
- 三形态使用统一的完整历史门槛；因子窗口必须是连续 20 个提供方交易日。暖启动、停牌缺口、成分股加入后历史不足的端点会被排除；跨 split 的历史输入可以保留，跨 split 的目标收益被剔除。
- 主表统一谓词按各 seed 最佳 beta 对应的验证集日 IC 均值选择，平局取第一个；ones 不参与。论文没有写明跨 seed 聚合方式，这一规则不能被称作已找回历史选择逻辑。
- ICIR/RankICIR 使用日相关序列总体 std（ddof=0）；组合 IR 使用日收益样本 std（ddof=1）。AR 采用 252×日均收益，IR 采用 sqrt(252)×日均收益/日 std，回撤为加法累计口径。论文未给出所有这些口径，故报告同时保留日收益供另算 CAGR 等指标。
- 推理计时包含模型 forward、必要的输入拼接与按日拆分，排除数据加载、缓存构造和指标计算；CPU 证据不能证明 GPU 延迟。TSIL 的部署 forward 不读取描述符；实验数据集初始化仍承担统一样本历史检查及缓存准备，不能把此耗时说成零。

## 5. 选择、训练与回测的完整路径

`search.py` 先在 Original、seed=2025 上搜索网络参数；选定后，对所有 predicate/seed 冻结网络设置。TSIL 的 alpha 网格为 [-4,-2,0,2,4]，每个 seed 独立按验证日 IC 选 beta；Original 与 Feature 不做无意义的 beta 搜索。`--frozen_network_manifest` 让另一种形态直接复用已选网络，避免独立重搜造成对照条件不同。

每个候选保存 config.yaml、model.bin、info.json；候选验证 IC 和最终映射写入 selection_manifest.json。搜索阶段不 prepare/evaluate test。最终用 `train.py --config_file <selected/config.yaml> --checkpoint <selected/model.bin>` 读取选好的权重，只评估测试集；不会重新训练或用测试结果换模型。完整命令见 [README](../README.md)。

正式回测使用 Qlib TopkDropoutStrategy/SimulatorExecutor，K=50、n_drop=5，买/卖手续费分别 0.0005/0.0015、最小费用 5、100 股交易单位，侧别成交价体现 0.0001 滑点。保存 `backtest_daily.csv` 和 `positions.pkl`，报告分别标明 strategy / excess、with / without cost。

**预测目标与成交时点有区别：** t 收盘信息预测 t→t+1 收益；Qlib 在下一交易步使用该信号成交。不能假设看完 t 收盘价后还能按同一已知收盘价无延迟成交，更不能把预测标签平均值直接当作成交组合收益。

**交易过滤的可确认范围：** 本次保留公开配置的 `limit_threshold=0.095`，验证了 Qlib 的停牌和该阈值过滤。它是统一近似，不是按板块、ST 状态和具体日期核实过的真实涨跌停价格规则。100 股换算需要提供方 `$factor` 完整；在真实行情上重新实验时必须审计它的可用性。没有将 Qlib 原先隐含的默认整手规则误报为原先一定不存在。

## 6. 验证证据与独立复核

最终命令、测试总数和运行环境见 [verification/RESULTS.md](verification/RESULTS.md)。主要覆盖：

- 六个描述符的独立数值 oracle、标签来源、归一化、损失与解析梯度、多谓词、常量谱和非法历史。
- 跨股票/跨日期窗口、未来追加不变、完整历史、split 目标边界、不同模式相同样本、保留尾批。
- 九骨干实例化与一步反向传播；Feature 拼接；关系模型日期隔离；真实 GAT 的未来标签缺失不变性。
- 实际 seed、独立输出路径、阶段式选择、跨形态复用网络、无 test 参与选择、checkpoint 只评估。
- 一个真实 Qlib 0.9.6 提供方的端到端合成运行：56 股、120 个业务日，LSTM 三形态各保留 504 个测试预测，10 个回测日；使用真正的 Alpha158、训练、固定 checkpoint、Top-50/Drop-5 和 Exchange。验证原始 label、非零成交与成本、100 股整手、买卖滑点、停牌/阈值过滤、真实缩小版网络及 beta 搜索。

独立审查发现并复核修复了：关系模型 stock batch_size 无效、多模型模板缺专用配置、MASTER Feature 输入被截断、未来标签筛选影响当日关系输入、短/不连续特征窗口。后两项涉及因果性及论文输入定义，已加入针对性反例。集成测试另发现 Qlib 0.9.6 与 Joblib 1.6.0 后端接口不兼容，固定 Joblib 1.4.2 后通过；最终评估 JSON 中 NumPy 标量的序列化问题也已通过实际入口验证。

## 7. 与历史审计未关闭事项的对应

| 旧审计问题 | 本次状态 |
|---|---|
| 12 处方法/实现差异 | 公式、数据源、标签、验证、计时与主回测路径已逐项修复；交易板块真实限价仍按第 5 节限定 |
| Feature 路径缺失 | 已实现并验证，含 MASTER 维数修复 |
| 九模型 alpha | 已移除损失参数，固定 beta 与网络参数分离 |
| seed/predicate/beta 选择关系 | 新实现保存 manifest；旧实验选择记录未恢复 |
| 验证 batch IC / state averaging | 改为日 IC；默认关闭参数平均，patience=20 |
| 用标签代替可执行收益 | 正式入口已去除；遗留 basktesting.py 不参与新报告 |
| 旧推理表单位及数字 | 新实现按 ms/sample 测量；没有重新命名或换算旧表值来冒充新结果 |
| 历史实验来源、整篇优势结论 | 未通过本次代码测试证明，仍需真实市场重跑与日志溯源 |

## 8. 限制与重新实验要求

配置所指 `qlib_data/cn_data` 没有提供真实 CSI300/CSI800 行情。因此本次没有运行 2018–2025 全量九骨干×三形态×谓词×五种子实验，没有更新论文表格、提升比例、误差条或推理延迟图。合成 IC/AR 数值仅用于软件验通，不能用于论文实证结论。

环境实际为 Python 3.12.14 / PyTorch 2.3.1 CPU / Qlib 0.9.6，与原文 PyTorch 1.13.1 / Qlib 0.8.0 不同。固定环境、数据发布版本、交易日历、复权因子、历史成分股、已确认窗口及选择 manifest 后，才能在真实数据上重跑并判断修复后的结论。

改动也意味着旧模型权重和旧 alpha 状态不兼容；必须重新训练。旧 `custom_phi_constructors.py` 等未使用扩展保留为历史文件，修复入口不再回退到它们；不是所有历史工具都被重新认证。未新增预测模型，也没有为了维持旧实验数值而调整论文公式中的 B 缩放或 beta 含义。

代码修复与原始方法的一致性可通过本次证据复查；统计显著性、真实收益优势、跨市场泛化和论文历史表格真实性仍由新的真实实验决定。
