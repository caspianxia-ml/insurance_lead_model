# E05 全国 P/U2 特征消融

已完成 2,700 次新拟合，原 B0 直接复用 E02；4 项针对性测试和 4,656 项独立核验通过。完整本地归档：`runs/20261107_E05_feature_ablation_01/`，正式报告版本 `reports/main_002/`。621 家最终 Gold 留出未构造新增特征、未训练或评分。

在项目根目录使用已有环境，每个新输出目录必须不存在：

```bash
.venv/bin/python experiments/e05_feature_ablation/test_e05.py
.venv/bin/python experiments/e05_feature_ablation/e05.py train --prepared runs/20261107_E05_feature_ablation_01/prepared_001 --parent-run runs/20261107_E02_Gold_vs_Silver_01/runs/main_001 --out runs/<新E05运行编号>
.venv/bin/python experiments/e05_feature_ablation/report_e05.py --prepared runs/20261107_E05_feature_ablation_01/prepared_001 --run runs/<新E05运行编号> --parent-run runs/20261107_E02_Gold_vs_Silver_01/runs/main_001 --out reports/v2/<新E05报告编号>
```

从冻结快照重建准备版本：`e05.py prepare --project <项目目录> --parent <E02的prepared_001> --out <新准备目录>`。无需重新导出或冻结。报告按提供的 E02 运行归档定位其同级准备目录，并核对原有哈希，不依赖旧工作区路径。

固定全部企业和逐次 U 抽样，保留 Gold-only/Gold+Silver 两条路线。行业层级和中类/年份覆盖不一致、简介有无等仅作明确诊断；即使开发得分很高也不批准加入主模型。地区只使用同一注册地址解析规则。详见 `docs/E05_FEATURE_ABLATION_RESULTS_20261107.md`。

未发布逐企业数据或生产评分，未启动 E06。旧原始文件和旧运行保持原样。
