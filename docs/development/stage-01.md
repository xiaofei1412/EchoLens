# 阶段 1：数据与基线

状态：用户已审核，同意进入阶段 2。开始及完成日期：2026-10-08（Asia/Shanghai）。

## 已确认范围与实施选择

用户已审核阶段 0，表示没有明显问题并同意推进阶段 1。这是阶段交付审核，不将其扩大解释为逐条人工金标准标注。

本阶段实现可重复 CSV/JSONL 导入、DuckDB/Parquet 本地存储、质量与重复报告、官方/去污染划分、字符 TF-IDF + LinearSVC 与中文编码器基线。无页面、主题聚类或大模型抽取训练。

中文编码器具体采用 hfl/chinese-roberta-wwm-ext，固定 revision，18 个四分类头。字符 TF-IDF 只在训练集拟合。超参或最佳 epoch 按开发集的端到端方面-情感对 Micro-F1 选择；测试集仅在配置冻结后评测。新建完全独立的 `.venv-stage1`，不修改阶段 0 环境。

## 验收项

- [X] CSV/JSONL 导入幂等，原文、来源、版本、缺失字段可追溯。
- [X] 先验证 1,000 条，再全量生成质量报告与 Parquet。
- [X] 官方划分保留；去污染划分无已知重复组交叉；报告近重复方法局限。
- [X] TF-IDF 与中文编码器完成实际训练、保存重载和逐标签评测。
- [X] 评测口径测试通过；配置、依赖、命令、实际结果与失败记录齐全。
- [X] 用户审核本阶段。

## 评测口径（运行前固定）

- 每个方面原始四类顺序为 -2 未提及、-1 负面、0 中性、1 正面。
- 方面识别：把非 -2 视为提及，报告 18 标签的 Macro/Micro-F1 及逐标签 P/R/F1/support。
- 端到端方面-情感对：18×3 个正向标签位置；预测未提及不计正例，错误情感同时产生 FP/FN。报告 Macro/Micro-F1 和逐对指标。
- 真实提及位置上的情感：仅按真实标签 != -2 选取位置，预测 -2 仍计为错误；报告三个情感类的 Macro-F1、准确率、覆盖率与逐类指标，不筛掉漏检样本。
- 不以四分类整体准确率作为主成绩，不将其与旧项目 F1=0.82 比较。

## 执行、失败与修复

- 开始前确认 D 盘剩余约 23.24 GiB；独立环境与编码器仅保存必要权重和最佳 checkpoint。
- 模型来源与架构依据：[HFL 模型卡](https://huggingface.co/hfl/chinese-roberta-wwm-ext)、[作者仓库](https://github.com/ymcui/Chinese-BERT-wwm)。本阶段是 BERT 类编码器基线，不是阶段 3 的生成式 QLoRA。
- 新建隔离环境安装成功，`pip check` 返回 No broken requirements found。固定直接依赖及完整已安装依赖快照，不继承阶段 0 的全局包。
- 1,000 条 smoke 与全量导入通过；二次导入新增 0。全量 46,730 条，规范化变化 532 条，原文保留。
- 固定字符 5-gram、128 维 MinHash、候选阈值 0.7、真实 Jaccard ≥0.9 后得到 203 个重复组，其中跨划分 79 组、标签不一致 38 组。清理后 train/dev/test=36630/4934/4937，已知组交叉为 0。
- 数据/评测首轮 17 项测试通过。新增 CSV/JSONL 同记录跨格式导入测试：文件包装哈希变化不算内容冲突，保留首次来源哈希；实际文本或标签变更仍拒绝覆盖。
- 编码器加载时上游预训练 MLM/NSP 头被标记为 UNEXPECTED，符合取编码器加新分类头的任务切换；训练损失和梯度保持有限。
- 合成 CSV/JSONL 导入示例实际运行：3 行中空文本排除 1 行，首轮新增 2 条，再跑新增 0 条。测试累计 18 项通过；编译和隔离环境依赖检查通过。
- 两套 TF-IDF 训练完成，均在开发集选择 C=1.0；clean 配对 Micro-F1=0.623364，official=0.625217。训练及开发集候选选择总耗时分别约 143.27 / 149.11 秒，词表均为 120,000。保存重载预测一致。
- 去污染编码器第 1 / 2 轮开发集配对 Micro-F1 分别 0.706600 / 0.729896，按固定三轮计划继续；尚未读取测试集标签进行模型选择。
- 去污染编码器三轮完成，选择第 3 轮，开发集配对 Micro-F1=0.736831。每轮训练约 247.55–251.10 秒，训练峰值 allocated 3.735 GiB；保存重载后全部开发集预测一致。
- 本轮长度固定为 384 tokens，36,630 条训练文本中 7,395 条发生截断（约 20.2%）。该限制在测试前记录，不根据测试结果改长度；后续若研究长文本处理，应作为新的开发集实验。
- 新增全量源文件对照测试，官方视图每条 text_raw 与标签顺序均匹配固定 CSV；测试累计 19 项通过（5.49 秒）。
- 开发集错误面：去污染编码器的“菜品卖相”方面 F1 仅 0.5945；真实提及位置中，中性 F1=0.6420，低于正面 0.8464。一些具体方面-情感对仍为 0，例如优惠折扣/负面（开发集支持 104）。因此不能用 Micro-F1 掩盖长尾未学会的情况；本轮保持预设训练方案并如实报告，不临时追加调参。
- 全量数据准备重复运行完成：首次及重跑生成的 records.jsonl SHA256 都为 `1bf34cd89d25c3b1280f92dad459288c43f0555b8f486364f0df46acebb68a63`；重跑首次写入和紧随的二次导入均新增 0 条。公开 data 报告记录最近一次运行，因此 first_pass_inserted 当前为 0，不代表最初没有导入数据。
- 官方编码器三轮完成，同样选择第 3 轮，开发集配对 Micro-F1=0.734777；三轮训练合计 754.43 秒，峰值 allocated 3.735 GiB，保存重载预测一致。去污染版三轮训练合计 748.78 秒。以上训练时长不含模型加载、分词、开发集评测和保存。
- 四个模型选定后生成 test-freeze.json，固定配置、数据快照、模型权重及选择报告哈希，才进入测试。两个测试视图只做固定评测，没有根据测试成绩重新选模型或修改参数。
- 最终 19 项测试通过（2.83 秒）；编译通过；独立环境 pip check 无依赖冲突。编码器训练无 OOM 或非有限损失/梯度。按 epoch 续训功能已实现并保留 last.pt，但本轮未人为中断来验证恢复后的逐位一致性。
- 从四个实验各两个测试视图的本地 NPZ 重新核对：记录 ID 与实际数据顺序一致、真实标签一致，重算全部指标与公开 JSON 完全相同，冻结文件哈希一致。摘要保存于 artifacts/stage-01/result-verification.txt。
- 去污染编码器对 4,937 条测试评论的批量前向累计约 7.52 秒，batch=16 的 P50/P95 为 24.30/25.29 毫秒，推理峰值 allocated 0.700 GiB；不含模型加载和分词，不是单条请求延迟。968 条测试评论超过 384 tokens，被按冻结配置截断。

## 最终结果

以下统一采用去污染训练与去污染测试（4,937 条）；均为 ASAP 官方标签的分类任务，不包含意图、证据跨度或新领域评测。

| 指标                      | TF-IDF + LinearSVC | 中文 RoBERTa 编码器 |
| ------------------------- | -----------------: | ------------------: |
| 方面识别 Macro-F1         |             0.7201 |              0.8242 |
| 方面识别 Micro-F1         |             0.7824 |              0.8577 |
| 方面-情感对 Macro-F1      |             0.4052 |              0.5600 |
| 方面-情感对 Micro-F1      |             0.6215 |              0.7333 |
| 真实提及位置情感 Macro-F1 |             0.5748 |              0.7327 |

编码器端到端配对 Micro-F1 比字符基线高约 0.1118（11.18 个百分点），但配对 Macro-F1 仅 0.5600，长尾组合仍明显不足。原项目的 F1=0.82 口径不明，不能与这里任何一项直接比较。

官方训练 + 官方测试的配对 Micro-F1 为 TF-IDF 0.6244、编码器 0.7361；这些结果有已知重复污染，只作为公开划分对照。更完整的同测试集对照与 18 个方面逐标签 P/R/F1 见 [结果报告](../reports/stage1-baselines.md)，54 个方面-情感对及支持数见相应 JSON。

## 限制与待审核项

- 单种子、固定小范围超参，没有统计显著性或跨领域泛化结论。词法近重复处理不能保证消除语义、用户或预训练污染。
- 编码器固定截断为 384 tokens，长评论可能遗漏后半段；没有为了提高测试成绩临时修改长度。
- 标签冲突保留原始值，按重复组优先级选择代表行；未把 AI 笔记或用户阶段审核转换成新增金标准。
- 阶段 1 已交付数据和分类基线。工作台页面、主题发现、意见抽取和生成式微调仍在后续阶段，当前未搭建。
- 请审核去污染策略、指标口径与基线结果。后续提出优化/删改时，在下表追加修改和验证，不抹掉本轮结果。

## 命令与产物

完整环境准备与执行顺序见根 README 的阶段 1 命令块。实际后台运行日志位于 `artifacts/stage-01/`，命令中的 `*>` 为 PowerShell 重定向 stdout/stderr：

```powershell
$env:PYTHONUTF8='1'
$env:HF_HUB_DISABLE_PROGRESS_BARS='1'
$env:TOKENIZERS_PARALLELISM='false'
.\.venv-stage1\Scripts\python.exe -u scripts/prepare_stage1.py *> artifacts/stage-01/prepare.log
.\.venv-stage1\Scripts\python.exe -u scripts/train_tfidf.py --version clean *> artifacts/stage-01/tfidf-clean.log
.\.venv-stage1\Scripts\python.exe -u scripts/train_tfidf.py --version official *> artifacts/stage-01/tfidf-official.log
.\.venv-stage1\Scripts\python.exe -u scripts/train_encoder.py --version clean *> artifacts/stage-01/encoder-clean.log
.\.venv-stage1\Scripts\python.exe -u scripts/train_encoder.py --version official *> artifacts/stage-01/encoder-official.log
.\.venv-stage1\Scripts\python.exe scripts/evaluate_stage1.py *> artifacts/stage-01/evaluate.log
.\.venv-stage1\Scripts\python.exe scripts/report_stage1.py
```

- 数据：`data/processed/stage1/`，公开聚合 `docs/reports/stage1-data.json`，契约 `docs/data/stage1-contract.md`。
- 模型：`artifacts/stage-01/{tfidf,encoder}-{official,clean}/`，选择结果 `selection.json`；编码器保留 `best.pt` 和按 epoch 续训的 `last.pt`。
- 结果：`docs/reports/stage1-*-dev.json`、`stage1-*-test.json`；各含完整配置/来源或冻结文件引用、逐标签 P/R/F1/support。
- 比较：`docs/reports/stage1-baselines.md`，由实际完成的 JSON 自动生成。
- 所有真实评论、词表、逐条预测、权重和原始日志保留本地，不进入 Git。

## 审核与变更记录

| 日期       | 来源/请求                            | 实际修改                                                              | 验证结果                                              | 受影响文件                                   |
| ---------- | ------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| 2026-10-08 | 用户认可阶段 0 并要求推进阶段 1      | 完成独立环境、导入、去重、官方/去污染快照、两类基线训练评测及复现文档 | 19 项测试、依赖检查、模型重载通过；固定测试结果见上表 | 本阶段代码、配置、报告、README 与阶段索引    |
| 2026-10-08 | 开发核对发现跨格式导入的文件哈希不同 | 相同实际记录跨 CSV/JSONL 保持幂等；变更内容仍要求新 revision          | 新增回归测试通过                                      | src/echolens/ingest.py、tests/test_stage1.py |

审核更新：用户表示“审核下来没有明显的问题”，同意进入阶段 2。原实验结果和限制保留，本次更新审核状态，不重新调参。
