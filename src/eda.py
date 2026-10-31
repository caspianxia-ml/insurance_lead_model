"""Small, repeatable matplotlib EDA report set."""
import logging
import os
from pathlib import Path
import pandas as pd

from .config import REPORTS_DIR
from .feature_engineering import build_features

LOGGER = logging.getLogger(__name__)
def source_diagnostics(p, u, output_dir=REPORTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, frame in [('P', p), ('U', u)]:
        if 'source' in frame:
            for source, count in frame.source.fillna('UNKNOWN').value_counts().items():
                rows.append({'sample':label,'source':source,'count':count,'share':count/len(frame)})
    pd.DataFrame(rows).to_csv(output_dir/'source_distribution.csv', index=False, encoding='utf-8-sig')

FIELDS = ['industry_major_category_clean', 'company_type_clean', 'company_size_clean',
          'insured_count_log1p', 'registered_capital_log1p', 'company_age']


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)


def run_eda(positive: pd.DataFrame, unlabeled: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reports are files: avoid initializing a native GUI from a CLI/PyCharm run.
    cache = output_dir / '.matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']:
        if name in fonts:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False
    p, u = build_features(positive), build_features(unlabeled)
    missing = pd.DataFrame({"P": p.isna().mean(), "U": u.isna().mean()}).sort_index()
    missing.to_csv(output_dir / "eda_missingness.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["P", "U"], [len(p), len(u)])
    ax.set_title("Sample counts")
    fig.tight_layout(); fig.savefig(output_dir / "eda_sample_counts.png", dpi=150); plt.close(fig)
    for field in FIELDS:
        if field not in p or field not in u:
            LOGGER.warning("EDA 跳过不存在字段：%s", field)
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        combined = pd.concat([p[field].rename("value").to_frame().assign(sample="P"),
                              u[field].rename("value").to_frame().assign(sample="U")])
        if pd.api.types.is_numeric_dtype(combined["value"]):
            for label, group in combined.groupby("sample"):
                ax.hist(group["value"].dropna(), bins=30, density=True, alpha=.5, label=label)
            ax.legend()
        else:
            table = combined.dropna().groupby(["value", "sample"]).size().unstack(fill_value=0)
            if table.empty:
                plt.close(fig)
                continue
            table = table.div(table.sum()).fillna(0).sort_values("P" if "P" in table else table.columns[0]).tail(15)
            table.plot(kind="bar", ax=ax)
        ax.set_title(f"P vs U: {field}")
        fig.tight_layout(); fig.savefig(output_dir / f"eda_{_safe_name(field)}.png", dpi=150); plt.close(fig)
