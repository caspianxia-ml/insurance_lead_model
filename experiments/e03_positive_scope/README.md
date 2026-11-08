# E03 沪苏/全国正例范围四组对照

已完成归档：`runs/20261107_E03_P_scope_01/`。最终留出 Gold 未评分。使用脚本入口，输出目录必须是新目录：

```bash
.venv/bin/python experiments/e03_positive_scope/test_e03.py
.venv/bin/python experiments/e03_positive_scope/e03.py train --prepared runs/20261107_E03_P_scope_01/prepared_001 --out runs/<新E03运行编号>
.venv/bin/python experiments/e03_positive_scope/report_e03.py --prepared runs/20261107_E03_P_scope_01/prepared_001 --run runs/<新E03运行编号> --out reports/v2/<新E03报告编号>
```

不以当前来源内 Gold 评价选定目标企业总体业务模型；详见项目结果报告。
