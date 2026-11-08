# E04：固定全国 P，比较 U2/U3 训练范围

已完成 600 次开发拟合、4 项针对性测试和 1,771 项独立核验。完整本地归档在 `runs/20261107_E04_U_scope_01/`；最终 621 家 Gold 留出未评分。

在项目根目录使用现有环境运行，新输出目录必须不存在：

```bash
.venv/bin/python experiments/e04_unlabeled_scope/test_e04.py
.venv/bin/python experiments/e04_unlabeled_scope/e04.py train --prepared runs/20261107_E04_U_scope_01/prepared_001 --out runs/<新E04运行编号>
.venv/bin/python experiments/e04_unlabeled_scope/report_e04.py --prepared runs/20261107_E04_U_scope_01/prepared_001 --run runs/<新E04运行编号> --e02-run runs/20261107_E02_Gold_vs_Silver_01/runs/main_001 --out reports/v2/<新E04报告编号>
```

重建准备数据使用 `e04.py prepare --project <项目目录> --e02-prepared <E02准备目录> --upstream <eda_002目录> --out <新的准备目录>`；输入可在归档 manifest 中按哈希定位。仅需原有冻结快照，不重新导出或冻结。

固定 U2 主面板内，U3 训练相对 U2 的 Top 10%/20% 差为 Gold-only −10.10/−7.53、混合 −7.96/−5.62 个百分点；固定扩展面板内分别 +3.56/+10.17、+9.31/+12.59。只在同一面板内比较，不能据此确定总体业务优劣。Gold/Silver 两条路线与两个 U 范围的结果均保留。

逐企业数据与评分仅在本地归档，不发布。详情见 `docs/E04_U_SCOPE_RESULTS_20261107.md`。来源旧代码按哈希原样保存，未覆盖旧实验。下一项 E05 尚未执行。
