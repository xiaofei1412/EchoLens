# 来源、许可与归因

核对日期：2026-10-12。本文件随v0.1.0发布材料提供；维护者已确认Apache-2.0与独立模型附件分发方案。

本项目原创代码、文档和自有模型修改采用Apache-2.0，完整文本见[LICENSE](LICENSE)。第三方模型、数据、库各自保留上游许可；项目许可证不替代这些许可，也不授予评论中涉及的个人信息或品牌权利。

| 来源 | 固定版本及许可证据 | 本次如何使用/分发 |
| --- | --- | --- |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；[上游LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE)，Apache-2.0 | 使用文本部分训练LoRA；仅分发adapter，不合并基础权重；保留Alibaba Cloud版权声明于`licenses/Qwen3.5-4B-LICENSE.txt` |
| Qwen/Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；[固定模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md)声明Apache-2.0 | 未修改，不打包；由用户从上游下载 |
| hfl/chinese-roberta-wwm-ext | `5c58d0b8ec1d9014354d691c538661bf00bfdb44`；[模型卡](https://huggingface.co/hfl/chinese-roberta-wwm-ext/blob/5c58d0b8ec1d9014354d691c538661bf00bfdb44/README.md)声明Apache-2.0；[原项目LICENSE](https://github.com/ymcui/Chinese-BERT-wwm/blob/master/LICENSE) | 发布经ASAP训练的18方面分类器state_dict，保留`licenses/Chinese-BERT-wwm-LICENSE.txt`；tokenizer/config从上游获取 |
| Meituan-Dianping/ASAP | `975122a60065240124df62cb4d5dbfd19ed9ef2c`；[上游LICENSE](https://github.com/Meituan-Dianping/asap/blob/975122a60065240124df62cb4d5dbfd19ed9ef2c/LICENSE)，Apache-2.0 | 用于训练/评测；保留`licenses/ASAP-LICENSE.txt`，不重新打包评论原文或逐条AI标注 |

许可原件下载来源、日期与SHA256记录见`licenses/sources.json`。编码器原项目LICENSE来自核对时的master，保留实际内容SHA256；模型权重及模型卡使用上表固定revision。Embedding固定仓库模型卡是此次许可声明依据，不虚构不存在的LICENSE文件。

第三方依赖通过pip单独安装，本项目不把其wheel或源码重新打入分发包。UI、推理直接依赖及精确版本见`requirements-ui.txt`、`requirements-inference.txt`；本机验收解析出的完整版本见`requirements-release-win-py314.lock.txt`。PyTorch、Transformers、PEFT、bitsandbytes、Streamlit等仍遵循其各自随包许可。

ASAP论文：*ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction*，NAACL 2021，来源见[ASAP官方仓库](https://github.com/Meituan-Dianping/asap)。编码器工作归因于Chinese-BERT-wwm项目及其模型卡所列作者；生成与向量模型归因于Qwen团队。

本版冻结标签与提示由项目迭代整理；810/90意见训练划分使用AI辅助标注，经过窄范围修订，不宣称人工完整金标准。开发过程中使用AI工具不构成这些工具提供商对本项目的背书。原始数据上游许可、代码许可和模型修改许可分别说明，不把“公开可获取”当成任意数据均可再发布的依据。
