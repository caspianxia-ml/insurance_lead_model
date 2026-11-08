# E02 Gold-only / Gold+Silver

已完成的运行归档在 `runs/20261107_E02_Gold_vs_Silver_01/`，含固定划分、拟合记录、开发预测及报告。最终 Gold 留出未评分。

从项目根目录执行脚本入口（新输出目录必须不存在）：

```bash
.venv/bin/python experiments/e02_gold_silver/test_e02.py
.venv/bin/python experiments/e02_gold_silver/e02.py train --prepared runs/20261107_E02_Gold_vs_Silver_01/prepared_001 --out runs/<新E02运行编号>
.venv/bin/python experiments/e02_gold_silver/report_e02.py --prepared runs/20261107_E02_Gold_vs_Silver_01/prepared_001 --run runs/<新E02运行编号> --out reports/v2/<新E02报告编号>
```

本代码是带共用 U、共同字段和总权重控制的 E02 适配；旧 `src/` 和 V1.1 实验均未修改。
