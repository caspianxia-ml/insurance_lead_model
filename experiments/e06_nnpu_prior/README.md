# E06 线性 nnPU / LR 与类别先验情景

完整本地归档：`runs/20261107_E06_nnpu_prior_01/`；正式训练 `runs/main_002/`，报告 `reports/main_001/`。3,000 次新线性 nnPU＋300 次 U3 LR，U2 LR 直接复用 E05 F0；最终 621 家 Gold 留出未使用。

在项目根目录使用已有 Python 环境，新输出目录必须不存在：

```bash
.venv/bin/python experiments/e06_nnpu_prior/test_e06.py
.venv/bin/python experiments/e06_nnpu_prior/e06.py train --prepared runs/20261107_E06_nnpu_prior_01/prepared_001 --parent-e04 runs/20261107_E04_U_scope_01/runs/main_001 --parent-e05 runs/20261107_E05_feature_ablation_01/runs/main_001 --out runs/<新E06运行编号>
.venv/bin/python experiments/e06_nnpu_prior/report_e06.py --prepared runs/20261107_E06_nnpu_prior_01/prepared_001 --run runs/<新E06运行编号> --parent-e04 runs/20261107_E04_U_scope_01/runs/main_001 --parent-e05 runs/20261107_E05_feature_ablation_01/runs/main_001 --out reports/v2/<新E06报告编号>
```

重建准备版本使用 `e06.py prepare --project <原项目目录> --e05-prepared <E05的prepared_001> --e04-prepared <E04的prepared_001> --out <新准备目录>`。无需重新导出或冻结。报告从所给父运行的相邻准备目录按原哈希定位，支持归档路径变化。

本轮固定 F0 五类画像、相同五折和逐次 U 抽样；两方法均为线性，nnPU 为 logistic-loss 非负估计器的凸数值优化，不是神经网络或原随机 Algorithm 1 的复制。先验 1%/5%/10%/20%/40% 是压力测试情景，不是发生率估计或区间；不能按最高开发分数确定真实先验。两条正例路线保留，不把分数解释为真实福利概率。

每次预处理与系数均保存并独立重新评分；所有风险、驻点、数值原始对偶残差和企业隔离均核验。失败 main_001 及旧代码保留；main_002 只对浮点精度状态2且梯度≤1e−7的退出按严格驻点标准接纳，逐项计数，没有放宽最终检查。

详见 `docs/E06_NNPU_PRIOR_RESULTS_20261107.md`。逐企业数据仅在本地保存；没有开展自动外呼、发消息或生产部署。
