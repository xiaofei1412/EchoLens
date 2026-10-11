# 阶段5：开源v0.1整理

2026-10-11开始。2026-10-12更新：用户已批准许可证、模型分发及公开发布；**v0.1.0已发布**，公开仓库、GitHub Pages展示页及四个Release附件均已核验。企业定制暂不实施。下方保留各批历史结论，正式发布结果追加在文末。

## 审核与变更记录

| 日期 | 用户请求 | 修改 | 验证 |
| --- | --- | --- | --- |
| 2026-10-11 | 专注餐饮；先整理开源v0.1范围、代码与UI，记录结果；企业版以后另议 | 冻结范围；统一启动与调度入口；五个日常页面；研究/历史入口独立；更新README和指南 | 71项测试通过；浏览器导入、报告下载和导航通过；12,969项历史文件哈希未变化；用户审核待进行 |
| 2026-10-12 | 继续模型分发、许可证、模型卡、干净环境安装验证，正式上传GitHub前先审核 | 增加独立模型包与安装/下载脚本、Apache-2.0及归因、模型卡、分层依赖、源码包和审核清单；版本0.1.0rc1 | 干净环境79项测试通过；2条合成GPU分析贯通；34个上游文件哈希核对；12,974个保护文件0变化；未上传/提交/tag |
| 2026-10-12 | 批准既定许可证/分发方案，指定xiaofei1412/EchoLens公开发布v0.1.0，包含展示页、附件、描述 | 确认本机账号；沿用现有空仓库；推送main和标签；发布Pages及四附件Release；补充公开下载与校验说明 | 79项回归通过；分发测试补验8项通过；公开clone的246文件与附件清单一致；四附件服务端SHA256一致、匿名下载均200；Pages已built并核对页面 |

## 验收与执行记录

- 范围：[v0.1发布范围](../releases/v0.1-scope.md)。不新增训练、不重跑1,000条、不公开发布。
- 保留历史脚本和冻结模型身份；通过运行入口整理、页面导航与配置恢复来改善可用性，不以大规模重构替代发布准备。
- 验收：五个用户入口、研究工具独立、工程数据默认隐藏、空数据导入、缺少运行资源提示、解释器选择、原文证据和审核/导出回归、历史文件哈希保护。

## 本批实际修改

- `PROJECT_PROPOSAL.md`增加当前路线说明，保留历史方案；阶段4跨行业扩展暂缓，阶段5处理开源准备。冻结文档为`docs/releases/v0.1-scope.md`。
- 新增`scripts/start_workbench.py`和`src/echolens/runtime.py`，收拢启动、解释器选择、资源存在性提示和后台进程。工作台与推理可分别指定Python，未指定时沿用当前本机环境；启动器向子进程传递本仓库`src`路径。没有下载依赖、模型或启动GPU任务。
- 将不含原文的固定提示与标签复制到`configs/prompts/restaurant-v0.1.txt`和`configs/labels/restaurant-v0.1.json`；内容与原冻结版本逐字节一致。只在原本地兼容路径缺失时恢复，发现不同内容拒绝覆盖。没有修改候选配置或被冻结校验的推理源码。
- 导航调整为数据导入、分析概览、主题与证据、人工复核、报告导出；研究工具保留人工标注和独立样本对照，并可返回主工作台。隐藏自动生成的重复侧栏导航。
- 新导入可以选择餐饮微调或原始基线，默认微调；导入完成后进入对应版本概览，不自动推理。原分析、机器结果及人工历史独立保存。
- `workbench_catalog.py`控制显示名称和可见性：工程数据、旧演示默认隐藏，可手动展开；普通微调版本优先展示。历史1,000条明确标注原始模型基线。分析任务与缺资源提示从质量详情中移出，报告导出从复核页中独立。
- README和`docs/running-workbench.md`更新当前入口；历史实验指南原样保留，当前页面操作以新指南为准。没有移动历史训练脚本，没有增加企业代码。

## 验证命令与实际结果

以下日志位于忽略Git的`artifacts/stage-05/`。无训练、无新的模型语义评测、无全量GPU重跑。

```powershell
.\.venv-stage1\Scripts\python.exe scripts/start_workbench.py --check
.\.venv-stage1\Scripts\python.exe scripts/start_workbench.py
.\.venv-stage1\Scripts\python.exe -m pytest tests/test_release_workbench.py tests/test_candidate.py tests/test_workbench.py tests/test_independent_review.py tests/test_stage2_revision.py tests/test_stage2_v3.py tests/test_stage3_sft.py -q
```

| 项目 | 实际结果 | 证据 |
| --- | --- | --- |
| 最终定向回归 | **71 passed in 11.31s**；新增9项发布整理测试，既有62项回归 | `tests-final.log` |
| 初轮回归 | 31通过、1失败，研究页返回链接与独立页面测试上下文不兼容；已修复 | `tests-initial.log` |
| 中间回归 | 68通过；随后补充调度失败清理、启动器源码路径与返回链接修复，最终71通过 | `tests.log`、`tests-release-final.log`、`tests-navigation-final.log` |
| 启动器 | `--check`正确选择`.venv-stage1`及`.venv`；通过启动器后台启动8501服务 | `ui.stdout.log`、`ui.stderr.log` |
| 空目录UI | 将app/src/configs/.streamlit复制到临时目录，未带data/models，AppTest正常展示数据导入；测试复用本机已安装依赖 | `test_clean_checkout_ui_has_no_private_data_dependency` |
| 缺资源与运行解释器 | 缺模型文件时按钮禁用；非法显式Python路径报错；公开提示/标签拒绝覆盖异版 | 新增发布测试 |
| 后台调度 | 模拟子进程确认分类→候选→主题顺序，推理采用指定解释器；推理失败标记failed并释放锁，不继续主题步骤 | 新增发布测试；本项未执行GPU |
| 浏览器导航 | 五入口可访问；研究页进入和同页返回；历史开关可显示旧基线和工程样例 | 本地8501浏览器；`workbench-preview.jpg` |
| 浏览器合成导入 | 3条输入中2条有效、1条空文本排除；自动进入微调概览，已处理0，未产生job.json | `import-ui-check.json`；使用`examples/feedback-import.jsonl` |
| 浏览器报告下载 | Markdown实际下载，1,523,508字节；文件含1,000条范围、数据快照、证据及非准确率说明 | 本地Downloads/echolens-report.md；含原文，不提交Git |
| 历史文件保护 | **12,969个文件，0变化** | `protected-before.json`、`protected-after.json` |

保护清单覆盖当时已有工作台JSON/JSONL、独立50条核验目录中的文件、冻结配置/推理源码及adapter文件，不把它描述成整个磁盘的所有文件审计。此次未提交人工审核表、未变更真实审核决定。

浏览器验收新建的`upload-e9dd940969a31087`及其`-qlora-v1`仅含仓库合成例句，验收后标为`synthetic_fixture`，默认隐藏且保留可追溯，没有删除用户数据。新样例不计入历史文件哈希保护清单。原文下载和测试产物仅本地保存；`git check-ignore`确认data、models、artifacts及虚拟环境均被忽略。

## 失败与修复

1. 研究页起初使用`st.page_link`返回主入口；独立页面AppTest未注册主入口，因此渲染报错。改为本地根路径链接，保持独立页可渲染。
2. 浏览器验收发现普通Markdown链接默认`target="_blank"`，返回行为不符合预期；改为固定HTML链接`target="_self"`。在浏览器中用键盘激活确认URL回到`/`并显示主导航；该链接不拼接用户输入。
3. 启动器父进程的`sys.path`不会自动传给子进程；补充子进程`PYTHONPATH`并保留已有值，防止依赖旧目录的editable安装。加入调度参数测试，临时目录UI检查通过。

## 限制与下一批

- 本批完成的是发布整理首批，尚未创建tag、推送或上传权重。权重分发、标准安装、模型卡、许可证/归因、轻量演示及完整干净环境GPU验证仍待完成。
- 资源提示只检查必要文件存在，不能替代依赖、CUDA、显存、完整权重和运行时哈希检查。没有宣称本轮验证Linux或全新机器推理。
- 保留阶段3实验限制：AI标注并非人工金标准；独立50条用户整体认可但逐条审核表仍为实际3条；字段合法率48/50，未达到99%目标。本轮不改善或重测模型语义指标。
- 用户待审核：新导航是否易用、研究和工程数据是否有必要继续收起、导入/模型选择/任务启动是否清楚。用户新反馈在本阶段追加，不覆盖本轮结论。

## 第二批：模型与发布材料（2026-10-12）

本节更新首批“限制与下一批”的待办状态，保留首批原始结论。当前本地候选材料见`docs/releases/v0.1-review.md`，最终上传仍需用户审核；没有创建远程、commit、tag或push。

### 实际交付

1. `scripts/model_bundle.py`提供build/install/verify。自有模型包仅含冻结adapter两文件、阶段1去污染分类器best.pt、模型卡和许可，共11个文件。先检查全包成员、大小和SHA256，再写入目标；同版幂等，异版拒绝覆盖，拒绝缺失/重复/额外成员与软链接。未把底座、tokenizer重复副本、优化器或训练数据混入包。
2. `configs/releases/v0.1-model-files.json`冻结上述文件清单；adapter内容与阶段3配置一致，原配置中的历史底座路径未改。实际worker显式从当前仓库加载底座，在无原路径依赖的独立目录已验证。
3. `scripts/fetch_release_models.py`按固定repo/revision和文件列表下载生成/向量/编码器模型并验证SHA256，支持`--verify-only`。34个本地缓存文件已逐个与Hugging Face固定revision的Git blob/LFS哈希核对；没有把固定commit仅当作目录标签。
4. 根目录新增Apache-2.0 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md`、`MODEL_CARD.md`、`CHANGELOG.md`。许可原件和来源哈希保存在licenses中。上游核对来源见归因文档；Qwen固定LICENSE的web页面工具访问失败后，使用官方raw端点成功获取。编码器许可保留master原件内容哈希，模型卡和权重固定revision；embedding许可依据固定模型卡的Apache-2.0声明。
5. 分层依赖为`requirements-ui.txt`和`requirements-inference.txt`，完整GPU验收环境另冻结`requirements-release-win-py314.lock.txt`。SciPy显式固定1.17.0，保持既有聚类数值依赖；未升级已有虚拟环境。新版本号为0.1.0rc1，未宣称正式v0.1已发布。
6. `scripts/prepare_source_release.py`只从公开目录/文件名单打包，排除数据、模型、artifacts、虚拟环境、缓存和环境凭据文件。源码包有逐文件及整包哈希sidecar；新增测试检查本地数据/凭据不混入。权重和运行产物保留在Git忽略目录。

### 干净环境验收方法

本地根目录：`artifacts/stage-05/release-check/`。建立两个新venv（完整GPU、轻量UI），均`include-system-site-packages = false`；没有复制旧site-packages。使用本机现有CPython3.14.6/Windows11和驱动，安装从官方PyPI/PyTorch cu128源进行，允许pip下载缓存。

将候选源码ZIP解压到checkout和ui-checkout，不带真实原文、训练标注、历史结果或审核库。GPU目录的基础模型通过34个已核对缓存文件的硬链接提供，推理只读取；自有模型由新安装器实际从ZIP安装。**此验收覆盖隔离Python环境和隔离源码，不等于另一台全新机器从公网下载全部基础模型。**真实下载命令和上游内容身份已分别核对，其他系统未验收。

轻量目录不带data/models，不安装torch或transformers。两个环境都使用当前源码目录的src，不依赖旧editable安装；GPU执行时设HF_HUB_OFFLINE以确认本地模型足够完成分析。UI服务以独立8503端口后台启动，健康检查通过后结束该验收服务；原8501工作台未停。

```powershell
# 在新环境中安装；这里用完整路径调用，不激活/改动原环境
.\.venv-stage1\Scripts\python.exe -m venv artifacts/stage-05/release-check/venv
artifacts/stage-05/release-check/venv/Scripts/python.exe -m pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
artifacts/stage-05/release-check/venv/Scripts/python.exe -m pip install --index-url https://pypi.org/simple -r requirements-inference.txt pytest==9.0.2
artifacts/stage-05/release-check/venv/Scripts/python.exe -m pip check

# 在独立checkout执行安装与实际worker；数据来自仓库合成例句
python scripts/model_bundle.py install --archive <模型包绝对路径>
python scripts/model_bundle.py verify
python scripts/run_workbench_job.py --dataset <合成样例的候选ID>

# 最终回归在独立checkout/新GPU环境内执行
python -m pytest tests/test_model_distribution.py tests/test_release_workbench.py tests/test_candidate.py tests/test_workbench.py tests/test_independent_review.py tests/test_stage2_revision.py tests/test_stage2_v3.py tests/test_stage3_sft.py -q
```

### 实际结果

| 检查 | 结果 | 本地日志/产物（相对release-check） |
| --- | --- | --- |
| 干净完整环境 | Python3.14.6，torch2.10.0+cu128、transformers5.19.0、peft0.19.1、bnb0.50.2、accelerate1.13.0、streamlit1.64.0；pip check通过 | `install-torch.log`、`install-dependencies.log`、`pip-check.log` |
| 干净轻量环境 | 无torch/transformers、无私有数据/权重；主页面及2个研究入口均正常；pip check通过 | `install-ui-only.log`、`ui-only.json`、`ui-pip-check.log` |
| 真实服务启动 | 独立启动器启动8503，`/_stcore/health`返回ok；检查后关闭验收服务 | `clean-server.out.log`、`clean-server.err.log` |
| 上游文件 | 3个模型，14＋12＋8＝34个文件全部匹配固定revision的上游内容哈希 | `upstream-verification.log`、`upstream-verified.json` |
| 自有模型安装 | 11个文件校验通过，安装在独立checkout；包474,430,258字节 | `model-build.json`、`model-install.json` |
| 合成GPU完整链路 | 3条输入，2有效/1空文本排除；分类2条、抽取2条均机械有效、5条意见、向量生成、0主题、报告2,016字节 | `gpu-smoke-retry.log`、`gpu-smoke-final.log`、`gpu-smoke.json` |
| 模型生成 | 2条生成约10.9/6.7秒，共17.55秒；是生成时间，不是端到端安装/加载时间 | 以上GPU日志及单条输出 |
| 最终回归 | **79 passed in 12.96s**；原71项＋8项分发测试 | `clean-tests-final.log` |
| 历史保护 | **12,974个文件，0变化**；包含既有工作台JSON/JSONL、独立50条材料、冻结候选代码、adapter及分类器 | `protected-before.json`、`protected-after.json` |

模型包SHA256为`6a7a1b7d5b12c673f2496fe5f06adb66fd6eda5ad32673973612d0651cd5201b`。完整源码ZIP在所有公开文档更新后生成，精确哈希记录在其同目录manifest，而非试图将包自身哈希写进包中。公开聚合结果见`docs/reports/v0.1-release-preparation.json`。

### 问题与修复

- 第一次完整环境AppTest冷启动触发默认3秒超时，尚未进入模型分析。验收脚本采用30秒冷启动超时，串开资源密集初始化后通过；没有修改UI逻辑或放宽模型验收。
- GPU分类、抽取、向量与主题已成功后，验收脚本调用`report_markdown`漏传records/classifications/extractions，后处理失败。按现有接口补齐参数，从已保存输出完成报告；**未重复生成2条意见**。初次失败日志原样保留，不把后处理错误说成模型失败。
- 全新依赖解析最初选中了SciPy1.18.1，显式约束为既有1.17.0后重新pip check，完整版本写入release lock。UI和GPU环境各自的其余传递依赖可能不同；完整lock是GPU环境快照，不承诺所有平台通用。
- 本轮没有重新训练、改提示/标签、调试独立50条或重跑1,000条。新增工程结果不是新F1，也不是品牌专属或跨领域性能证明。

### 提交用户审核

本轮准备项已完成：安装方案、模型分发包、许可与归因、模型卡、源码候选、干净环境验证、失败记录。下一步是用户审核`docs/releases/v0.1-review.md`，确认公开定位、Apache-2.0、模型独立附件分发及最终仓库信息，再执行上传。正式发布前仍没有可供外部下载的远程URL；不把本地准备完成描述为发布成功。

## 正式发布批次（2026-10-12）

用户已确认许可证和模型分发，指定公开仓库`xiaofei1412/EchoLens`，授权发布v0.1.0、展示页、Release附件和仓库描述。此前候选结论与候选包哈希保留；正式模型包仅因模型卡/NOTICE等发布文案变化重新打包，adapter和分类器字节未改变。

- 通过本机Git Credential Manager读取凭据并调用GitHub账户API，确认账号xiaofei1412（ID139749021）。凭据仅在进程内使用，不写入文件、命令参数或日志。仓库已存在且公开，账号有管理和push权限，沿用现有空仓库，没有重复创建。
- 设置仓库描述、主题标签、主页链接。新增`docs/index.html`静态展示页、`.nojekyll`和原生SVG README横幅；页面不接收用户评论、不调用推理服务。展示的例句是合成示意；内部90条AI参考F1和独立50条字段合法率分开表述。
- 版本从0.1.0rc1整理为0.1.0；README以使用、下载和模型限制为主，保留旧实验报告及候选审核记录。`.gitattributes`禁止自动换行转换，确保固定提示与分发清单的字节哈希跨checkout保留。
- 页面先在本地8504端口检查DOM及桌面截图；功能、安装和发布链接均指向指定仓库。完成页面后，79项定向回归再次通过（11.56秒），本轮未重新训练或重跑GPU推理。
- 正式附件为`echolens-v0.1.0-models.zip`、`echolens-v0.1.0-source.zip`、源码manifest、`SHA256SUMS.txt`。模型包11文件、474,430,375字节；SHA256为`9a35e0cc2612052cce6412d680e510de71fef4986e3994f6d3147ef2c7df3480`。上传前按公开文件清单暂存，拒绝data/models/artifacts/虚拟环境、权重、原始日志和凭据进入Git；所有附件先在本地生成校验。

发布前本节记录授权和实际完成的本地检查；远程commit/tag、附件及Pages部署结果在核验后追加，不能把准备中的链接视为已发布成功。原始API回执和上传日志仅保存在`artifacts/stage-05/publish/`。

上传前首次按源码清单暂存时发现`data/`忽略规则同时匹配`docs/data/`，导致两份公开数据说明文档未进入暂存区。仅为`docs/data/*.md`增加例外，保留真实数据目录的忽略规则；旧预检包保留为本地产物，修正后重新生成正式源码附件。

### 正式发布结果与外部验证

- 公开仓库：[xiaofei1412/EchoLens](https://github.com/xiaofei1412/EchoLens)。仓库描述、8个主题标签及主页URL已设置；Git remote为该仓库。本次使用Git Credential Manager已有登录，无需用户重新授权；提交使用GitHub noreply邮箱。
- 发布提交：`aef554c7657f0fc9460ad99f407d9e2651be7aa3`；附注标签`v0.1.0`指向该提交。后续发布核验文档作为main上的独立补充提交，不移动标签、不替换已发布附件。
- 正式Release：[EchoLens v0.1.0](https://github.com/xiaofei1412/EchoLens/releases/tag/v0.1.0)，Release ID `409791901`。实际公开时间为2026-10-12 00:52:45（Asia/Shanghai），API时间为2026-10-11T16:52:45Z；draft=false、prerelease=false。
- 展示页：[https://xiaofei1412.github.io/EchoLens/](https://xiaofei1412.github.io/EchoLens/)。Pages使用main分支/docs，构建状态built；匿名HTTP200，返回HTML与本地文件逐字节一致，浏览器桌面渲染与链接检查通过。该页面是静态介绍，推理仍在本地工作台运行。
- 公开源码重新clone到隔离目录后，246个文件SHA256全部匹配源码manifest。首次暂存检查及clone均使用实际文件字节校验，未把Git自动换行转换引入冻结文件。
- 四个附件均为uploaded状态，远程大小与SHA256匹配本地；匿名下载全部HTTP200。三个小附件完整下载后再验哈希；474MB模型包使用GitHub服务端digest校验，并匿名打开下载流验证ZIP头，未重复下载整个模型包。
- 模型包上传约8分钟，最终HTTP201成功；未拆包、改模型或降低校验。匿名GitHub API首次验收因共享IP频率限制返回403，切换为已认证API读取元数据；附件下载仍用无认证请求，不将限流误判为发布失败。
- 全量79项定向回归通过（11.56秒）；公开文件清单新增`.gitattributes`后，8项分发测试补验通过（0.13秒）。本轮没有重复训练、独立50条推理或完整GPU验收。
- 本地8504页面预览服务已关闭；原工作台不受影响。详细API回执、公开clone校验、下载检查及页面截图保存在被忽略的`artifacts/stage-05/publish/`。

| 正式附件 | 字节数 | SHA256 |
| --- | ---: | --- |
| echolens-v0.1.0-models.zip | 474430375 | `9a35e0cc2612052cce6412d680e510de71fef4986e3994f6d3147ef2c7df3480` |
| echolens-v0.1.0-source.zip | 744817 | `a0c5c193dcb5c134cae55965e714cc59953f163164b1a2a9d9d49e9e1a89849d` |
| echolens-v0.1.0-source.manifest.json | 42286 | `5d4dddb7f2ef932281f96bbc5613dbce0e480b55f62de55ea618dcbd2fc0b26e` |
| SHA256SUMS.txt | 292 | `5ecf801ddc251d247287b63fcbbec75451dd0ab7eaebccc28cc284bfa81fa33b` |
