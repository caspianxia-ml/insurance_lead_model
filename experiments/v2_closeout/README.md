# V2 开发版本归档与统一入口（2026-11-07）

本版本保留 E02–E06 已完成结果，补充行业统一与头部诊断。E01独立复现按用户决定暂缓，E07等待业务核验。本次新增模型拟合0次，最终Gold621家未评分。

## 运行环境与入口

在 insurance_lead_model 项目根目录使用既有 `.venv`。开发时Python3.13.6，准确依赖见 `experiments/v2_closeout/requirements.lock.txt`；完整运行环境清单在本地归档的 `runtime_versions.json`。不自动联网安装或更新依赖。

```bash
.venv/bin/python experiments/v2.py status
.venv/bin/python experiments/v2.py verify --source-only
```

status列出各实验正式运行、正式报告、数据角色与限制；公开版verify --source-only只校验代码和聚合文档，不需要企业数据。完整verify需要另外恢复本地冻结文件和正式运行归档，才会核对四份输入及逐文件结果哈希。主登记为 `experiments/v2_closeout/registry.json`，锁定登记为 `snapshot_lock.json`。

| 实验 | 比较问题 | 正式训练 | 正式报告 | 已有新增拟合 |
|---|---|---|---|---:|
| E02 | Gold-only／Gold+Silver | main_001 | main_001 | 300 |
| E03 | 正例地域 | main_001 | main_001 | 600 |
| E04 | U2／U3 | main_001 | main_001 | 600 |
| E05 | 特征消融 | main_001 | main_002 | 2,700 |
| E06 | 线性nnPU／先验情景 | main_002 | main_001 | 3,300 |

E05报告main_002修复路径可迁移性，未重训；E06首次main_001停机保留，正式结果来自main_002。E06中U2 LR精确复用E05 F0，不计入新增拟合。

## 显式复跑

所有输出使用尚不存在的新目录。下面的 `train` 确实会重新训练；常规查看结果只需status／verify与现有报告。

```bash
# 先查看训练命令，不执行
.venv/bin/python experiments/v2.py train --experiment E06 --out runs/E06_new_run --dry-run

# 只重新生成报告，沿用归档模型与评分；独立报告核验会重放评分，不拟合模型
.venv/bin/python experiments/v2.py report --experiment E06 --out reports/v2/E06_new_report

# 重新执行本次行业审计、按首个标准分类合并、头部评分诊断；不训练
.venv/bin/python experiments/v2.py closeout --out runs/V2_closeout_new

# 从冻结数据和已归档身份准入结果准备一份新E02数据
.venv/bin/python experiments/v2.py prepare --experiment E02 --out runs/E02_new_prepared

# 显式指定新准备数据后训练／报告
.venv/bin/python experiments/v2.py train --experiment E02 --prepared runs/E02_new_prepared --out runs/E02_new_train
.venv/bin/python experiments/v2.py report --experiment E02 --prepared runs/E02_new_prepared --run runs/E02_new_train --out reports/v2/E02_new_report
```

prepare／train／report支持E02–E06；父实验评分、逐次U抽样与共同划分默认指向登记的既有归档，便于精确复现该版对照。需要替换父运行的扩展实验，应先建立新版本登记；各旧脚本原入口继续保留。

首次从原冻结文件重做身份验收，可显式调用：

```bash
.venv/bin/python experiments/v2.py audit-data --out runs/V2_acceptance_new
```

该命令读原冻结数据，先执行acceptance再执行用户确定规则的eligibility，不重新导出／冻结。本次没有重复执行这项已完成的大审计；原验收的代码、工具包和正式产物已补入本地归档。

## 本轮核验

```bash
.venv/bin/python experiments/v2_closeout/test_closeout.py --taxonomy runs/20261107_V2_closeout_01/reference/taxonomy_complete.json --registry experiments/v2_closeout/registry.json
.venv/bin/python experiments/v2_closeout/verify_closeout.py --root runs/20261107_V2_closeout_01 --e06 runs/20261107_E06_nnpu_prior_01 --out runs/V2_closeout_verify_new
.venv/bin/python experiments/v2_closeout/verify_source_relationships.py --project . --audit runs/20261107_V2_closeout_01/analysis_002 --out runs/V2_source_relationships_new.json
```

8项针对性测试通过。1,141项独立核验通过，涵盖全部1,080条逐折／种子／方法／K的Gold命中数、既有LR头部评分与名单一致性、行业原列与企业数保留及输出哈希。U1/U2/U3另行核验重叠实体三个行业源值一致。

行业参照抽取脚本需要openpyxl，已在本机捆绑的表格运行环境只读执行；模型环境不要求额外安装openpyxl。参照工作簿、已抽取taxonomy和原schema副本均保存在本地reference，按SHA-256固定。

## 本地保存与恢复

代码、聚合文档、版本登记提交到本地Git；企业表、评分、原冻结CSV、字典工作簿、系数与逐企业记录保持在忽略目录，未加入新提交。没有远程push。

工作目录外的本地备份包含Git bundle、四个冻结文件、E02–E06完整运行、原验收与本次正式产物，并附文件哈希清单。bundle可使用 `git clone <bundle路径> <新代码目录>` 恢复；业务归档tar按原相对路径恢复后运行verify。备份位置及实际提交号记录在本地 `delivery_receipt.json`。

备份与工作目录在同一台机器上，属于独立目录副本；异地备份尚未配置。
