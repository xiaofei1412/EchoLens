# 阶段 1 数据契约与质量口径

## 导入与版本

入口 `scripts/import_feedback.py` 接受 UTF-8 CSV/JSONL 和显式字段映射。映射至少指定 `source_record_id` 与 `text_raw`，可指定 rating、product_id、created_at、collected_at、parent_id、source_url。当前评分契约仅支持 1–5 整数星级，其他评分尺度先不转换。错误 JSON、冲突 ID、非法评分/标签会终止导入；空 ID/空文本按行号记录排除原因。

`record_id` 是 source、source_revision、source_record_id 的确定性 SHA256。重复导入同一版本同一内容不增加记录；同 ID 内容变更须用新 revision，不能静默覆盖。CSV/JSONL 包装方式或文件哈希改变而实际字段不变时也不增加记录；数据库保留该记录首次导入的文件哈希。

源数据没有的时间、产品、父记录或 URL 字段保持 NULL。rights_status 默认 unknown；本阶段训练脚本只加载固定 ASAP 快照且校验其上游许可状态。通用导入数据不会自动混入训练。

本地表保留 text_raw、text_normalized、原始标签数组、来源版本、文件哈希和处理版本。规范化仅统一真实 CR/CRLF 换行为 LF 并去掉首尾空白；不删除否定词、标点、表情、内部重复或字面 `\\n`。证据仍应引用 text_raw；本阶段分类不产生证据偏移。

语言统计是“包含汉字则 zh_heuristic，否则 und”的规则结果，并非语言识别模型。没有从评论顺序生成日期，也没有自动删除疑似广告或低星反馈。

## ASAP 快照

`scripts/prepare_stage1.py --limit 1000` 先写独立 smoke 目录；不带 limit 才生成全量正式快照。

本地 `data/processed/stage1/` 包含 DuckDB、规范化 JSONL、官方/去污染各三份 Parquet 和 manifest。行顺序按 record_id 固定，原始标签沿用固定配置中 18 个方面顺序；官方 CSV 原件不变。JSONL 的 SHA256 用于实验绑定，实际数据与聚合统计由测试核对。

清洗规则为 `normalize-v1`；重复/拆分配置为 configs/stage1.json 中 `asap-stage1-v1`。更改规则时必须提升处理版本并重建快照，不能把旧模型结果归到新数据版本。

## 去重与划分规则

1. 去除空白后完全相同的文本直接连为重复组。
2. 长度至少 50 字的文本取字符 5-gram 集合，使用固定种子的 128 维 MinHash、LSH 阈值 0.7 找候选。
3. 候选逐对计算真实集合 Jaccard，达到 0.9 才建立近重复连接。没有使用情感标签来选择阈值。
4. 对已核验连接求连通分量；同组优先保留 test，其次 dev，再次 train，同级按源 ID 字典序保留一条。被排除的记录不删除，保留组 ID 和原因。组内原标签不一致也不投票改标签。

测试优先的目的，是从训练/开发中隔离已知相似内容。清理后每组仅保留一条，因此无已知组跨划分；官方视图完整保留，明确标记有污染。两种训练结果都在官方和去污染两个测试视图上报告，比较时使用相同测试视图。

MinHash 候选召回不是穷举，可能漏掉近重复；字符相似不等于语义等价。连通分量的端点未必直接达到 0.9。这是保守的监督划分隔离，不代表已排除预训练污染、作者泄漏或全部语义泄漏。

## 实测结果

| 视图 | train | dev | test |
| --- | ---: | ---: | ---: |
| 官方 | 36,850 | 4,940 | 4,940 |
| 去污染 | 36,630 | 4,934 | 4,937 |

共发现 203 个重复组，其中 79 个跨划分、38 个组内原始标签不完全一致。排除训练 220 条、开发 6 条、测试 3 条；排除清单和核验相似度见 `docs/reports/stage1-data.json`。532 条文本的首尾空白或真实换行发生规范化变化，原文全部保留。

缺失 product_id、created_at、collected_at、parent_id、source_url 均为 46,730 条，缺失 rating 为 0。所有文本包含汉字。上述缺失由源数据决定，不进行事实补全。

## 可运行的合成导入示例

```powershell
.\.venv-stage1\Scripts\python.exe scripts/import_feedback.py examples/feedback-import.jsonl --mapping configs/import-example.json --source synthetic-example --revision v1 --rights synthetic
```

示例由 AI 编写，不是人工金标准。首轮应导入 2 条并排除 1 条空文本；再次运行应新增 0 条。数据库、Parquet 与最近一次导入报告位于被 Git 忽略的 `data/imports/`。
