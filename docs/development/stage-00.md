# 阶段 0：定义与探针

状态：用户已审核，同意进入阶段 1。开始日期：2026-10-07；收尾及审核日期：2026-10-08（Asia/Shanghai）。

## 用户确认与范围

- 2026-10-07：用户确认按 proposal 的定位、功能、数据、模型选型执行。
- 本阶段交付数据卡、字段和许可记录、200 条训练集探索笔记、标签草案、本机 GPU/量化/训练保存重载探针。
- 用户授权在后台运行必要代码、核对并记录结果。
- 每阶段维护开发记录；人工审核的修改在本文末尾追加，并更新实际文档/代码与验证结果。

## 验收清单

- [X] 官方数据固定 revision 下载，记录哈希和许可。
- [X] 字段、样本量、标签、缺失与重复审计完成。
- [X] 200 条训练集探索笔记完成，区分 AI 初读与人工金标准。
- [X] 数据卡、标签指南、输入输出契约完成。
- [X] GPU 前后向与量化算子验证。
- [X] 4B 模型文本推理与有限步数 QLoRA、保存重载验证。
- [X] 复现命令、依赖与检查结果齐全，提交人工审核。
- [X] 用户完成阶段 0 人工审核；有修改时追加记录后重新验证。

## 执行与发现

- 仓库起始仅有用户已审阅的 PROJECT_PROPOSAL.md；保留原文不作重排。
- GPU：NVIDIA GeForce RTX 5080 **Laptop GPU**，16303 MiB，计算能力 12.0，驱动 592.01。物理内存约 31.4 GiB。
- 未检测到可用 WSL；优先验证原生 Windows，不修改操作系统配置。
- 原 Python 3.14.6 已有 torch 2.10.0+cu128、transformers 5.2.0、peft 0.19.1。CUDA 矩阵乘法前向/反向和有限梯度已通过。
- 创建项目 `.venv --system-site-packages` 复用大型已有依赖；新增/覆盖包仅装入项目环境。后续导出精确依赖和重建说明，不把此环境误称为完全隔离环境。
- D 盘初始可用约 33.5 GiB，模型下载采用单份本地目录，避免重复权重。
- GitHub API 查询 commit 返回匿名速率限制 403；改用 `git ls-remote`，固定 ASAP revision `975122a60065240124df62cb4d5dbfd19ed9ef2c`。
- Windows 控制台默认 GBK 打印上游 README 遇到 UnicodeEncodeError；后续运行显式设置 PYTHONUTF8=1。
- 数据实际出现 `star=5.0` 字符串；初次整数解析失败，改为先确认浮点值为 1–5 整数后转换，没有填补或修改评级。
- 全量审计通过，发现 6 组跨划分空白归一化重复；没有修改原始划分。200 条训练探索样本排除了这些泄漏记录。
- 新增项目环境依赖 bitsandbytes 0.50.2、pytest 9.0.2，8 项契约检查通过。
- NF4 CUDA 前后向通过。首次 4B 加载因将多模态 Qwen3_5Config 传给文本类而失败（vocab_size 缺失）；改为显式 get_text_config，并增加 missing/mismatched 权重检查，避免未加载预训练权重却运行探针。
- 项目内 Transformers 更新为 5.19.0；上游用户目录的 5.2.0 保持原样。首次失败日志为 artifacts/stage-00/probe.log，重跑日志为 probe-v2.log。
- 原生 Windows 未安装 causal-conv1d / flash-linear-attention，Qwen3.5 使用 PyTorch 参考实现，性能需实测。
- `pip check` 发现共享已有环境中的 inscriptis/lxml 与 numba/numpy 版本冲突，均不是当前数据/模型探针直接依赖。暂不修改用户全局依赖；正式基线阶段需处理其对聚类环境的影响。不能声称当前共享环境已通过全环境依赖检查。
- 50 步有限范围 QLoRA 已完成：序列 297–334 tokens、micro-batch 1、累积 1、rank 8，仅 full-attention 的 q/k/v/o 投影，1,572,864 个可训练参数。耗时 24.42 秒，训练峰值 allocated 显存 5.174 GiB。
- adapter 确实改变；重新加载新底座及 adapter，参数完全一致，同输入 logits 最大绝对差 0。3 条合成推理样例结构和原文包含校验均通过，不能当作真实任务准确率。
- 新增探索集隔离检查，测试累计 9 项通过。200 条真实训练样本的 AI 初读正在运行，其引用错误保留，不自动当作标注数据。
- 200 条 AI 初读已完成：200/200 可解析 JSON，81/200 通过结构与原文连续匹配联合检查（40.5%）。总生成耗时 441.14 秒，不含模型加载；逐条笔记和原文回查已渲染到 artifacts/stage-00/exploration-200.md。
- 阅读首批 25 条完整原文，另定向回查 54/62/65/88/96 号原文；已浏览全部 200 条 AI 概述。发现实体混淆、指控/因果升级、重复文本及任务外内容。所有检查均为 AI 检查，人工金标准为 0。
- 1024 token 合成 padding 压力探针，all-linear、rank 8、16,232,448 个可训练参数，3 步前后向与优化通过：训练 4.07 秒、峰值 allocated 显存 7.518 GiB；保存重载参数相同，logits 最大差 0。包含 padding，不能声称真实长评论吞吐或正式训练成绩。
- 下载器重复运行验证哈希不变；新增下载完整性测试，测试累计 10 项通过。代码编译检查通过。
- proposal 中示例标记由“人工编写”校正为“合成示例”；examples 为 AI 编写的基础设施样例，不冒充人工标注。
- 嵌入模型 Qwen3-Embedding-0.6B 探针完成：7×1024 向量全部有限，3 个合成查询的 top-1 均符合预期，推理约 0.37 秒，峰值 allocated 1.140 GiB；这不是检索基准成绩。
- 2026-10-08 收尾：核对 reading-v2 的 20 条输出，与 v1 相同训练记录比较，结构/连续引用通过数由 12 增至 16。新增可复算的配对聚合报告；4 个剩余错误保留。v2 仅提取一个意见，覆盖范围不同，且在探索集修正，不能作为独立质量提升结论。
- 最终检查：10 项测试通过（0.32 秒），src/scripts/tests 编译通过。依赖文件固定项目直接依赖；并非全传递依赖锁文件，未验证全新隔离环境复现。
- 收尾一致性检查通过：两份公开训练报告与本地运行 JSON 一致；配置/报告 JSON 均可解析；数据、模型、阅读原文与环境均被 Git 忽略；三个入口的待审核状态一致；Python 源文件无行尾空白。检查摘要保存于 `artifacts/stage-00/final-checks.txt`。

## 结果与产物

| 项目                  | 实际结果                                                                  | 可审核产物                                                                                 |
| --------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 数据审计              | 46,730 条；18 个方面；6 组跨划分空白归一化重复                            | `docs/data/asap.md`、`docs/reports/asap-audit.json`、`asap-manifest.json`            |
| 探索阅读              | 200 条训练集 AI 笔记；81 条通过结构/原文联合检查；人工金标准 0 条         | 本地`artifacts/stage-00/exploration-200.md`；公开 `docs/reports/exploration-review.md` |
| 提示词修正探针        | 同样 20 条训练样本，12/20 → 16/20；仍有 4 条失败                         | `docs/reports/evidence-prompt-comparison.json`                                           |
| 4B 量化与训练         | NF4 前后向、50 步优化、adapter 变化及新底座重载一致性通过；峰值 5.174 GiB | `docs/reports/generation-probe.json`                                                     |
| 1024 padding 压力探针 | all-linear、3 步、重载通过；峰值 7.518 GiB                                | `docs/reports/generation-stress.json`                                                    |
| 嵌入探针              | 有限向量、3 个合成 top-1 检查通过                                         | `docs/reports/embedding-probe.json`                                                      |
| 契约与标签            | 字段、枚举、连续原文证据校验；领域边界草案                                | `src/echolens/contracts.py`、`docs/annotation-guidelines.md`                           |

所有显存均为 PyTorch peak allocated，并非整卡总占用。原始评论、模型权重、adapter、日志在忽略目录；公开推理探针报告中的文本仅为 AI 合成样例。

## 运行命令与日志

在仓库根目录使用 PowerShell，环境准备见 README。以下为对应实际实验的命令；GPU 实验串行运行。下载脚本校验已下载文件，阅读脚本会跳过已有同版本记录。模型训练探针重跑会覆盖指定输出目录，保留历史实验时需指定新的 `--output`。

```powershell
$env:PYTHONUTF8='1'
$env:HF_HUB_DISABLE_PROGRESS_BARS='1'
$env:TRANSFORMERS_VERBOSITY='warning'
.\.venv\Scripts\python.exe scripts/fetch_asap.py
.\.venv\Scripts\python.exe scripts/audit_asap.py
.\.venv\Scripts\python.exe scripts/fetch_models.py
.\.venv\Scripts\python.exe -u scripts/probe_model.py --steps 50 --output artifacts/stage-00/probe-v2
.\.venv\Scripts\python.exe -u scripts/probe_model.py --steps 3 --pad-to-length 1024 --all-linear --output artifacts/stage-00/stress
.\.venv\Scripts\python.exe -u scripts/probe_embedding.py
.\.venv\Scripts\python.exe -u scripts/read_exploration.py
.\.venv\Scripts\python.exe -u scripts/read_exploration.py --limit 20 --strict-evidence
.\.venv\Scripts\python.exe scripts/summarize_reading.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

日志保存在 `artifacts/stage-00/`：`probe.log` 为首次失败，`probe-v2.log` 为修复后的 50 步探针，`stress.log`、`embedding.log`、`evidence-prompt-v2.log` 为对应后续实验。训练 JSON 原件位于 `probe-v2/report.json` 和 `stress/report.json`；其副本分别为公开的 generation-probe/stress 报告。AI 阅读逐条原始输出在 `data/exploration/` 的两个 notes JSONL 文件。

## 待审核项与下一阶段交接

- 审核标签草案、证据规则和 200 条 AI 阅读笔记；本阶段完成的是探索，尚未获得人工金标准。
- 真实评论的连续引用可靠性不足，必须保留校验与复核流程；40.5% 和 80% 均不是任务准确率。
- 6 组跨划分重复已定位；阶段 1 保留官方基准并构建去污染训练版本，不用测试集调参。语义近重复尚未检查。
- ASAP 无日期/产品元数据，不能支持真实趋势；本阶段领域限定餐饮反馈，其他领域尚未验证。
- 原生 Windows 探针可运行；共享环境仍有既有 inscriptis/lxml、numba/numpy 冲突。阶段 1 基线/聚类前需要独立环境验证，不据此宣称正式训练环境完全就绪。
- 已建立本地 Git 仓库；未发布 GitHub。当前仅完成阶段 0，没有提前实现数据管道、基线或页面。

## 审核与变更记录

| 日期       | 来源/请求或发现                    | 修改                                                                               | 验证结果                                              | 受影响文件                                            |
| ---------- | ---------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| 2026-10-07 | 用户要求阶段记录和审核更新留痕     | 创建本记录和项目开发约定                                                           | 2026-10-08 已核查范围、命令、结果、失败与待审核项齐全 | `AGENTS.md`、本文件、开发记录索引                   |
| 2026-10-07 | 开发中发现配置不兼容与数据格式差异 | 修复文本配置加载和星级转换；保留失败记录                                           | 量化/训练/重载及数据审计通过                          | `scripts/probe_model.py`、`scripts/audit_asap.py` |
| 2026-10-08 | 用户要求继续完成任务               | 汇总配对提示词探针、补齐结果与复现命令、修正指南中“餐后等上菜”为“点餐后等上菜” | 200/20 条汇总核对；10 项测试及编译通过                | 汇总脚本、报告、README、标签指南、本文件与索引        |

2026-10-08 审核更新：用户表示“审核下来没有明显的问题”，同意进入阶段 1；本文件的审核勾选由用户完成。原模型笔记及自动校验失败结果保留，不将阶段交付审核当作逐条人工金标准标注。本次仅更新状态与阶段索引，无需重跑阶段 0 模型。
