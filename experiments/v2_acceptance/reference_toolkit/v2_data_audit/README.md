# 保险福利识别模型 V2 | 数据清洗审计程序

本程序执行 **数据质量审计 + 保守实体去重 + P/U 严格交叉排除 + 清洗日志**，不训练模型。

## 先准备输入

将四组导出的原始 CSV 放在本目录的 `data/raw/` 下：

- `P_national.csv`
- `U1.csv`
- `U2.csv`
- `U3.csv`

每份必须至少包含列 `credit_code`、`company_name`，其余字段保持原样。无需手动在 Excel 中修改信用代码。

## 运行（Mac / PyCharm 终端）

在本目录的终端执行：

```bash
python -m pip install -r requirements.txt
python audit_v2.py \
  --p data/raw/P_national.csv \
  --u1 data/raw/U1.csv \
  --u2 data/raw/U2.csv \
  --u3 data/raw/U3.csv \
  --out outputs/v2_first_audit
```

回车后会读取全部数据、清洗并生成报告。请不要提交 `data/` 与 `outputs/` 到公开 Git 仓库。

## 输出文件

| 文件 | 解释 |
|---|---|
| `data_quality_report.csv` | 各 CSV 每列缺失率、唯一值数量、读入类型 |
| `entity_cleaning_log.csv` | 各组原始量、信用代码异常、去重、已知 P 排除、保留量；数量可对账 |
| `record_decisions.csv` | 每条源记录的具体处理决定；`kept`、`removed_duplicate_credit_code`、`removed_known_P_credit_code` |
| `P_U_exact_overlap.csv` | 确认的 P/U 信用代码重合详情；已从 U 中排除 |
| `P_U_name_review.csv` | 仅企业名匹配的疑似 P/U 冲突；**不自动删除** |
| `within_dataset_name_review.csv` | 同一数据集同名、无代码或异码的疑似实体冲突；不自动合并 |
| `U_subset_diagnostics.csv` | U1/U2/U3 三对之间的信用代码交集和集合差异 |
| `cleaned/*.csv` | 每组清洗后数据 + 新的规模标准化列；保留其他原始字段 |
| `run_manifest.json` | 运行时间、输入文件 SHA256、读入编码、使用规则，便于复现 |
| `audit_summary.md` | 汇总报告及方法限制 |

## 核心原则

- **信用代码按字符串读入**；已被 Excel 改写为科学计数法的数据不能靠此程序还原。
- **不因信用代码缺失删行**；仅严格的 18 位格式一致信用代码被用于自动去重及 P/U 排除。
- 业务状态不在第一轮自动删除；先报告真实异常类型，再确定规则。
- 来源出现“原力”只标 Gold 候选，不能替代可靠业务确认。
- `U1` 是历史快照，不能直接宣称严格包含于新时点 `U2`。
- 不根据 `source`、`keyword`、福利标签等变量训练模型，这一步也不训练。

## 后续工作

收到真实四份原始 CSV 后，先审计报告中的异常与重合，再冻结 P/U 训练样本；然后实施 Gold Holdout、EDA、实验 E01-E07。此代码为首次可执行版本，需要结合真实字段核对后再冻结。
