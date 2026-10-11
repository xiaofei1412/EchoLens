# 运行餐饮工作台（v0.1准备版）

启动与UI已整理，模型包和Windows干净环境安装已在本地验收，维护者已授权发布v0.1.0。全新目录先按[安装指南](releases/v0.1-install.md)安装依赖和模型；发布附件见[GitHub Release](https://github.com/xiaofei1412/EchoLens/releases/tag/v0.1.0)。

## 启动

```powershell
python scripts/start_workbench.py --check
python scripts/start_workbench.py
```

`--check`只显示解释器选择，不启动服务、不下载、不复制文件，也不代表依赖或GPU验证通过。默认127.0.0.1:8501，可用`--port 8502`更改。

优先使用本地`.venv-stage1`运行工作台、`.venv`运行候选推理，按系统选择`Scripts/python.exe`或`bin/python`；不存在时使用当前Python，适合已安装完整依赖的单环境。可通过`ECHOLENS_APP_PYTHON`和`ECHOLENS_INFERENCE_PYTHON`指定解释器完整路径；显式指定的路径不存在会报错，不静默回退。本轮只实测Windows现有环境，不自动安装依赖，不宣称Linux GPU已验证。

公开提示和标签与阶段3冻结版本字节一致，启动时只向缺失的本地兼容路径恢复这两个文件；不含训练原文，已有不同内容拒绝覆盖。模型仍按`configs/stage3-candidate.json`读取，自有模型包通过`scripts/model_bundle.py`安装并校验。

## 五个日常入口

1. **数据导入**：CSV/JSONL，选择原文、稳定ID和可选评分，选择餐饮微调或原始基线。导入后进入概览，不自动启动推理。重复导入沿用原去重及版本约束。
2. **分析概览**：确认模型、处理/失败数量和统计口径。缺资源时关闭启动按钮并列出原因；已有结果仍可浏览。后台执行后可刷新进度。
3. **主题与证据**：主题浏览、原文检索两个视图；主题代表和边界不是已验证结论。原文检索可按方面、情感筛选，导出当前筛选范围。
4. **人工复核**：检查失败、分歧或随机样本，保留机器结果并单独保存人工修订；兼容原审核格式。
5. **报告导出**：整个数据集的证据报告及审核/主题历史。机器统计与人工修订分开，失败不伪装成可靠证据。

资源提示只检查必要文件是否存在，不代表完整权重、依赖、CUDA或显存已验证；实际运行仍会进行版本与哈希校验。此次已在同一台Windows机器的新隔离环境验收2条合成评论；其他系统、驱动和显卡仍未验证。

侧栏“研究工具”保留人工试标和微调前后对照。默认隐藏工程探针和被新版本替代的历史演示；勾选“显示历史与工程数据”即可查看，文件未删除。列表只有这些项目时默认显示导入入口，不把探针包装成正式演示。

已有普通微调版本时优先打开它；否则打开ASAP v3基线演示或其他可用数据。原1,000条明确标为基线，不改称微调结果。创建微调分析版本保留原版；切换不会自动重跑。v0.1不提供时间趋势。

## 代码入口

| 类别 | 入口 | 责任 |
| --- | --- | --- |
| 启动 | `scripts/start_workbench.py` | 解释器选择、恢复公开提示/标签、启动页面 |
| UI | `app/streamlit_app.py` | 五个用户入口 |
| 运行调度 | `src/echolens/runtime.py`、`scripts/run_workbench_job.py` | 后台进程、资源提示、串行GPU任务 |
| 数据展示 | `src/echolens/workbench_catalog.py` | 可见性及展示名，不改历史元数据 |
| 存储与证据 | `src/echolens/workbench.py` | 快照、统计、定位、审核历史 |
| 冻结推理 | `scripts/analyze_stage3.py`、`src/echolens/candidate.py` | 保持阶段3身份和结果校验规则 |
| 训练评测 | `scripts/train_stage3_full.py`、`scripts/run_stage3_sft.py` | 独立研究流程，不是使用UI的前置步骤 |
| 历史准备/实验 | `scripts/prepare_stage*.py`、`probe_*.py`、`report_*.py`等 | 保留原路径；查对应阶段开发记录 |

没有移动或删除历史实验文件，以免破坏源码哈希、旧命令和审计材料；保留文件不表示用户需要依次执行它们。
