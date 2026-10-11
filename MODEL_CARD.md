---
language:
  - zh
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
  - restaurant-reviews
  - aspect-based-sentiment-analysis
  - lora
  - text-generation
---

# EchoLens Restaurant v0.1 模型卡

版本：v0.1.0。日期：2026-10-12。[模型下载与校验文件](https://github.com/xiaofei1412/EchoLens/releases/tag/v0.1.0)。

本版用于中文餐饮评价中的意见抽取与证据核对。每条意见包含对象、方面、情感、意图、当前/过去/其他对象归属及连续原文证据。28个方面ID含unknown，完整契约见`configs/labels/restaurant-v0.1.json`及`configs/prompts/restaurant-v0.1.txt`。模型不输出经营事实认证，不能把少量评论解读成用户总体比例。

## 分发内容与加载

- **抽取模型**：Qwen/Qwen3.5-4B，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，第102步LoRA adapter。只包含`adapter_config.json`和`adapter_model.safetensors`，不合并底座、不含优化器。
- **配套分类器**：`hfl/chinese-roberta-wwm-ext`，revision `5c58d0b8ec1d9014354d691c538661bf00bfdb44`，ASAP去污染训练版本；18方面×4分类（未提及/负面/中性/正面）。分发完整state_dict，保留原`best.pt`内容，加载使用`weights_only=True`。它是辅助分类器，不是28方面抽取模型的结果。
- **主题向量模型**：Qwen/Qwen3-Embedding-0.6B，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，单独从上游下载。
- 字节大小、SHA256、包内文件和安装目标见`configs/releases/v0.1-model-files.json`。基础模型不在自有模型包内，安装见[发布安装指南](docs/releases/v0.1-install.md)。

保留原冻结adapter配置中的历史`base_model_name_or_path`字符串，以保持阶段3哈希不变。工作台显式从当前仓库`models/Qwen3.5-4B`加载底座，再加载adapter，不使用该历史绝对路径。不要使用依赖该字符串自动定位底座的加载方式；不要修改原配置后混用旧缓存。

## 数据与训练

源语料：美团点评公开ASAP中文餐饮评价。LoRA数据为810训练/90内部开发，均为AI辅助意见标注，经项目修订；不是810条人工金标准。独立50条来自未用于本项目训练和调参的测试分区，但同属ASAP；不能排除基础模型预训练见过公开语料。数据、逐条教师输出、原文预测及审核库不随包发布。

两轮训练，102个优化步。NF4双重量化、BF16计算；rank8、alpha16、dropout0.05，文本线性层；micro-batch1、梯度累积16，AdamW学习率1e-4、weight decay0.01、3步warmup后恒定。依据开发loss选择checkpoint102（0.180556），不是根据独立50条挑选。详见[训练报告](docs/reports/stage3-sft-full-comparison.md)。缺少未分发的AI标签时，公开脚本不保证精确再现相同训练权重。

## 已有评测

| 90条内部开发集，对AI参考的一致性 | 底座 | 微调后 |
| --- | ---: | ---: |
| 严格证据＋方面＋情感联合F1 | 0.1139 | 0.2527 |
| 宽松联合F1（跨度IoU≥0.5） | 0.2425 | 0.5095 |
| 全记录字段合法率 | 75.56% | 97.78% |
| 平均生成秒数/条 | 36.93 | 51.25 |

这些是教师一致性，开发集参与checkpoint选择，不是独立测试准确率；F1不要求对象名称、意图、归属同时正确。生成更慢，证据边界和召回仍有提升空间。

独立50条：JSON合法47→49，字段合法46→48，触及输出上限2→1，平均生成30.20→48.09秒。用户在对话中整体认可这一批；实际逐条保存的成对审核仅3条，不把其余47条代填为“微调胜出”。48/50字段合法率未达到原99%目标，也不是语义正确率。

辅助分类器在4,937条去污染ASAP测试记录上的方面Micro-F1为0.8577、方面×情感配对Micro-F1为0.7333；这是另一任务，不能与上述意见抽取F1横向比较。完整定义见[阶段1结果](docs/reports/stage1-baselines.md)。

## 推理、限制和失败处理

冻结推理：输入最多4,096 tokens、输出最多4,096新tokens；greedy、关闭thinking、NF4/BF16、SDPA。当前目标为本地NVIDIA CUDA；无CPU或其他操作系统的推理承诺。阶段3实测设备为RTX 5080 Laptop GPU 16GB，Windows/Python3.14；缺少可选优化内核，使用PyTorch回退。当前发布安装验收另见[阶段5记录](docs/development/stage-05.md)，不拿工程样例冒充新质量评测。

超长输入不静默截断；输出截断、JSON/标签/证据错误、不确定归属及完全重复意见转人工复核，保留原输出但不进入可靠证据。字段通过不意味着语义正确；有效空结果也可能漏提。含蓄、讽刺、混合对象、长评论和跨句指代仍有风险。

只面向餐饮；没有跨行业、品牌专属或真实经营效果验证。主题需至少10条不同反馈，小样本可能没有主题。星级不能替代意见情感，无日期不生成趋势，v0.1本身不提供时间趋势。阶段2已有1,000条是基线演示，不是此模型成绩。

许可、上游版权和数据边界见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目与美团点评、Qwen或任一餐饮品牌无官方合作或背书关系。
