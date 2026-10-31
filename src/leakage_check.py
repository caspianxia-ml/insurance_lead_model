"""Detect explicit and proxy label leakage."""
import logging
from pathlib import Path
import pandas as pd

from .config import LEAKAGE_COLUMNS, REPORTS_DIR

LOGGER = logging.getLogger(__name__)


def check_leakage(positive: pd.DataFrame, unlabeled: pd.DataFrame,
                  output_path: Path = REPORTS_DIR / "leakage_report.csv") -> pd.DataFrame:
    rows: list[dict] = []
    all_columns = sorted(set(positive) | set(unlabeled))
    for column in all_columns:
        if column in LEAKAGE_COLUMNS or any(token in column.lower() for token in ("insurance_tag", "positive_rule")):
            rows.append({"field": column, "risk": "explicit_label_source", "metric": 1.0, "detail": "禁止进入正式模型"})
        if column not in positive or column not in unlabeled:
            rows.append({"field": column, "risk": "sample_exclusive_field", "metric": 1.0,
                         "detail": "仅存在于 P" if column in positive else "仅存在于 U"})
            continue
        p_nonnull, u_nonnull = positive[column].notna().mean(), unlabeled[column].notna().mean()
        if p_nonnull >= .8 and u_nonnull <= .1:
            rows.append({"field": column, "risk": "mostly_nonnull_only_in_P", "metric": p_nonnull - u_nonnull,
                         "detail": f"P non-null={p_nonnull:.3f}, U non-null={u_nonnull:.3f}"})
        if not pd.api.types.is_numeric_dtype(positive[column]) and not pd.api.types.is_numeric_dtype(unlabeled[column]):
            p_values = set(positive[column].dropna().astype(str).unique())
            u_values = set(unlabeled[column].dropna().astype(str).unique())
            exclusive = p_values - u_values
            if exclusive:
                share = positive[column].astype(str).isin(exclusive).mean()
                if share >= .05:
                    rows.append({"field": column, "risk": "P_only_categories", "metric": share,
                                 "detail": f"{len(exclusive)} categories; P share={share:.3f}"})
        p_missing, u_missing = positive[column].isna().mean(), unlabeled[column].isna().mean()
        if abs(p_missing - u_missing) >= .3:
            rows.append({"field": column, "risk": "collection_completeness_gap",
                         "metric": abs(p_missing - u_missing),
                         "detail": f"P missing={p_missing:.3f}, U missing={u_missing:.3f}"})
    report = pd.DataFrame(rows, columns=["field", "risk", "metric", "detail"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("泄漏检查报告：%s", output_path)
    return report
