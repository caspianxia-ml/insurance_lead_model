# insurance_lead_model

## 当前状态（2026-10-31）

已有 V1/V1.1、V2 四组 EDA 和新沪苏 P＋U1 四模型结果，入口见 [项目状态与汇总结果](docs/PROJECT_STATUS_20261031.md)。原始数据已登记为 [v2_raw_001](docs/data_versions/v2_raw_001.json)，原件和归档副本 SHA-256 一致；实际导出时间和严格历史快照仍待核验。

公开仓库仅含模型代码、说明、Notebook 模板和聚合结果。取数 SQL 与原文说明按用户选择仅保留在本地，不随公开仓库发布。企业原始数据、逐企业评分、候选名单及本机完整登记留在 Git 忽略目录；克隆仓库可运行合成测试，业务结果重跑需要自有获授权数据。

## V2 数据质量与 EDA（2026-10-22）

四份取数 SQL 已归档于本地 `sql/v2/20261020_1200/`，保留原文、样本对应关系及时间条件说明；该目录的六份文件不公开。

V2 首轮仅审计 2026-10-20 12:00 声明快照的全国 P、U1、U2、U3，不训练模型，也不改动 V1/V1.1。方法、汇总结果和局限见 [V2 EDA 报告](docs/V2_EDA_20261020_1200.md)；可复跑的聚合图表见 [V2 EDA Notebook](notebooks/02_v2_data_quality_eda.ipynb)。

在项目根目录执行下列命令。每次使用新的运行编号；原始 CSV 只读，企业级数据不提交 Git。

    .venv/bin/python -m experiments.v2_eda --run-id <未使用的新编号>

本次本地运行编号为 20261022_v2_eda_1200_04，详细汇总、输入及分析代码哈希、已执行 Notebook 位于被 Git 忽略的 runs/20261022_v2_eda_1200_04/。U3 用宽松引号解析且逐条验证 20 列；异常文本保留并计入质量报告。Gold 候选来源口径已纳入原力和量化Q3榜单，原始 p_quality 不改写。SQL 血缘和严格历史快照仍待补证。

## V1.1 进度（2026-10-21）

用户新导出的 `V2_P_National_20261020_1200.csv` 与 `V2_U1_20261020_1200.csv` 已通过严格 CSV 结构检查。V1.1 的正式定义、A/B/C/D 对照见 [V1.1 定义与数据登记](docs/V1_1_DEFINITION_20261020.md)。C 已在本地版本化运行；D 以新 P＋同一新 U1 独立运行。实验结果和局限见 [V1.1 本轮结果](docs/V1_1_RESULTS_1200.md)。原始企业数据与评分只在本机 `runs/`，Git 不跟踪。

V1.1 的历史数据对照 B 已完成：P train 五折交叉验证、固定 P holdout、全部历史 U 的 LR 折外评分。见 [审计与运行约定](docs/V1_1_AUDIT_AND_RUNBOOK.md) 和 [B 实验结果](docs/V1_1_RESULTS_B.md)。本地明细位于被 Git 忽略的 `runs/20261019_B_archived_v1_1/`。

新 U1 的 CSV 存在列错位，已改用 2026-10-19 的 JSON 导出做本地清洗审计：18,325 条输入，按 V1 规则排除异常经营 15 条、与历史 P 重合 4,446 条，暂得 13,864 家 U。此结果在 `runs/20261019_C_u1_staging/`，**尚未排除新 P 重合，未运行 C 的模型评分**。V1 原有命令和输出保持历史用途；不要用下方 `main.py --step all` 覆盖历史产物来声称 V1.1 已完成。

本轮新 P 在 C 中仅用于排除已知正例；D 才用于训练。精简 U1 CSV 不含旧 JSON 的 `sample_type` 和 `has_welfare_raw`，入口改为核验沪苏标记并在内存中设置清洗器内部的 `sample_type=U`；福利线 SQL 条件不能从精简 CSV 独立证明，须保留 SQL/取数记录。原始文件不变。

```bash
python3 -m experiments.v1_1_stage_u1 --run-id <unused-staging-run-id> --new-u1 '/absolute/path/to/new_u1.json'
python3 -m experiments.v1_1_new_u1_c --run-id <unused-C-run-id> --new-u1 '/absolute/path/to/new_u1.csv' --positive-exclusion '/absolute/path/to/new_p.csv' --reference-date 2026-10-20
python3 -m experiments.v1_1_new_p_d --run-id <unused-D-run-id> --new-p '/absolute/path/to/new_p.csv' --new-u1 '/absolute/path/to/new_u1.csv' --c-run '/absolute/path/to/completed-C-run'
```

2026-10-08 续建：输入采用当前有效保险福利标签 P 和新版 Mid Clean U。
优先按下方命令传入实际文件，不要使用旧 Leads U。原始 CSV 保持只读。

```bash
python main.py --step check --positive '/Users/mima0000/Downloads/P样本v1.csv' --unlabeled '/Users/mima0000/Downloads/u1.csv'
python main.py --step eda --positive '/Users/mima0000/Downloads/P样本v1.csv' --unlabeled '/Users/mima0000/Downloads/u1.csv'
python main.py --step features --positive '/Users/mima0000/Downloads/P样本v1.csv' --unlabeled '/Users/mima0000/Downloads/u1.csv'
python main.py --step baseline --sensitivity --positive '/Users/mima0000/Downloads/P样本v1.csv' --unlabeled '/Users/mima0000/Downloads/u1.csv'
```

所有入口先做实体清洗。P 按实体留出 20%；U 按画像分层取 40% 训练池，
余下 60% 用于候选排序。预处理和字段缺失率准入仅看训练部分。
训练池 U 暂不输出候选分数，全部 U 的 OOF cross-fitting 尚未实现。
`--sensitivity` 对每个来源分别排除其 P/U 训练记录，在固定候选集比较排名相关和 Top10 Jaccard。
这属于敏感性实验，不修改原始业务样本口径。正式完整实验由用户在 PyCharm 运行。

新增输出：`reports/sample_funnel.csv`、`data_quality_report.csv`、
`excluded_records_detail.csv`、`entity_overlap_conflicts.csv`、
`u_registered_region_corrections.csv`、`source_distribution.csv`、
`raw_source_audit/source_distribution.csv`、`entity_split.csv`、
`feature_eligibility.csv`、`source_score_distribution.csv`、`source_topk.csv`、
`source_sensitivity.csv`（需参数）。清洗结果位于 `data/processed/*_modeling.csv`。
报告和派生结果再次运行会刷新，需保留版本时先复制 reports 和派生结果。

评分中的 `is_in_leads` 留空，不能据 leads_company_id 推断新增入库候选。
信用代码校验失败仅标记待核验；缺失信用代码、终止经营、重复实体和 P/U 冲突有逐条排除清单。
外币金额不擅自换算，记为缺失并报告；未写币种的境内金额暂按人民币假设。
平台行业描述、联系信息有无、来源、地域、标签、ID、公司名、采集量均不入模。

以下为原项目说明；与上述新版流程冲突的旧样本口径和“全部 U 评分”描述以新版为准。

这是一个用于保险福利企业线索排序的 Positive–Unlabeled（PU）机器学习 POC。已确认有保险福利的企业是 P（Positive）；未确认企业是 U（Unlabeled），**U 不是可靠 Negative**。第一阶段产出的是可解释的相对排序分数，不是企业拥有保险福利的校准概率。

## 目录结构

```text
insurance_lead_model/
├── data/
│   ├── raw/          # 只读原始 CSV
│   ├── processed/    # 特征与中间结果
│   └── output/       # 最终评分文件
├── notebooks/        # 交互式研究
├── src/              # 正式 pipeline 代码
├── reports/          # 验证、泄漏、评估和 EDA 报告
├── tests/            # 自动测试
├── main.py
└── requirements.txt
```

原始 CSV 不会被程序覆盖。代码没有数据库、Web API 或云服务依赖。

## 在 PyCharm 中配置 Python 3.13

1. 用 PyCharm 打开本目录 `insurance_lead_model`。
2. 打开 **Settings/Preferences → Project → Python Interpreter**。
3. 选择 **Add Interpreter → Add Local Interpreter → Virtualenv**。
4. Environment 选 **New**，Location 选项目下的 `.venv`。
5. Base interpreter 选择本机的标准 CPython 3.13，然后确认。

`.venv` 是只属于本项目的隔离 Python 环境。它避免该项目的依赖版本影响其他项目，也不等同于 Conda 环境。

也可以在项目 Terminal 中创建和安装：

```bash
python3.13 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

不要自行改用 Python 3.11。如果 `xgboost` 或 `lightgbm` 等包安装失败，先记录 `python --version`、包名、包版本和 pip 的完整报错，再检查是否已有支持 Python 3.13 的发行版。

## 准备数据

把文件放到：

- `data/raw/positive.csv`：历史在上海招聘、且按严格规则确认具有保险福利的 P 样本；
- `data/raw/unlabeled.csv`：历史在上海招聘、当前没有保险福利标签的 U 样本。

支持 `utf-8-sig`、`utf-8` 和 `gb18030` 编码。字段并非必须全部存在：候选字段缺失时会记录 warning 并跳过。若两个文件缺失，命令会给出简短放置提示，不打印长 traceback。

## 运行

请在项目根目录的 PyCharm Terminal 中执行：

```bash
python main.py --step check
python main.py --step eda
python main.py --step features
python main.py --step baseline
python main.py --step score
python main.py --step all
```

- `check`：读取数据，检查 schema、重复值、缺失率、类型、日期、异常值、类别基数和标签泄漏；
- `eda`：比较 P/U 分布并将 matplotlib 图片写入 `reports/`；
- `features`：生成可重复特征，验证预处理，并写入 `data/processed/`；
- `baseline`：运行首版 baseline，将探索性分数写入 `data/processed/baseline_unlabeled_scores.csv`；
- `score`：对全部 U 排序，写入 `data/output/unlabeled_scored.csv`；
- `all`：依次完成以上完整流程（评分只生成正式输出一次）。

正式评分包含 Positive centroid 相似度、重复抽样 U 的 Logistic Regression、One-Class SVM 和 Isolation Forest。预处理器只在 P 训练集（或 LR 每轮的 P 训练集加临时抽样 U）上拟合；固定的 20% P holdout 不参与 imputer、scaler、encoder 或模型拟合。

## 查看结果

关键文件：

- `reports/data_validation_report.csv`
- `reports/leakage_report.csv`
- `reports/baseline_evaluation.csv`
- `reports/eda_*.png`
- `data/output/unlabeled_scored.csv`

评分文件包含四类 score、各模型 rank、`model_agreement_count` 和 `priority_level`。A 表示至少两个模型均进入 Top 10%；B 表示至少一个模型进入 Top 10%，或至少两个模型进入 Top 20%；D 表示所有模型均在 Top 20% 之外；其余为 C。规则统一放在 `src/config.py`。

## 正确解释结果

重复抽样 LR 会临时将一部分 U 编为 0，但那只是构造 ranking baseline 的工程手段，不能推断 U 是真实负类。`lr_mean_score`、相似度分数和 one-class 分数都不是 calibrated insurance probability；它们用于比较相对顺序和不同模型的一致性。

标签泄漏尤其重要：`key_word`、`welfare`、保险标签、规则版本等字段本身参与了 P 的定义。把它们放进 X 会让离线结果看起来异常优秀，却不能发现尚未被规则标注的真实保险企业。因此这些字段仅保留用于审计，不进入候选特征。

没有可靠负类时，项目不把 Accuracy、Precision 或 Specificity 当作核心结论。评估重点是未参与拟合的 P holdout 能否被排到较高位置、U 的分数分布、Top 1/5/10/20% 和模型 Top 10% 重合度。

## 排错

先看 Terminal 最后一条 `ERROR` 或 `WARNING`，再检查：

1. `python --version` 是否为 3.13；
2. 当前目录是否为项目根目录；
3. 两个 CSV 的文件名、编码和权限是否正确；
4. `reports/data_validation_report.csv` 是否显示字段差异或极端缺失；
5. `reports/leakage_report.csv` 是否发现只在 P 中出现的字段、类别或数据完整度差异。

运行测试无需额外测试框架：`python -m unittest discover -s tests -v`。

## 公开发布范围

2026-10-31 通过网页发布 50 个文件。四份取数 SQL、取数目录 README 与 source_paste.txt 仅保留本地；企业原始数据、逐企业评分和候选明细也不公开。公开登记中的 SQL 路径指本地证据位置。

本地归档分支历史包含未公开 SQL；后续公开同步应基于远程 main，不应将该归档分支的完整历史直接推送到公开仓库。网页上传不代表本机 Git 已完成登录。
