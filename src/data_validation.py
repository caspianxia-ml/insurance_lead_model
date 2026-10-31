"""Schema and data-quality validation before modeling."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATE_COLUMNS, REPORTS_DIR
from .data_cleaning import _parse_date_dayfirst

LOGGER = logging.getLogger(__name__)


def _row(check: str, sample: str, field: str, value: object, severity: str = "info") -> dict:
    return {"check": check, "sample": sample, "field": field, "value": value, "severity": severity}


def validate_data(positive: pd.DataFrame, unlabeled: pd.DataFrame,
                  output_path: Path = REPORTS_DIR / "data_validation_report.csv") -> pd.DataFrame:
    rows: list[dict] = []
    rows += [_row("sample_size", "P", "*", len(positive)), _row("sample_size", "U", "*", len(unlabeled))]
    p_only = sorted(set(positive) - set(unlabeled))
    u_only = sorted(set(unlabeled) - set(positive))
    rows += [_row("schema_only_in_sample", "P", c, True, "warning") for c in p_only]
    rows += [_row("schema_only_in_sample", "U", c, True, "warning") for c in u_only]
    rows.append(_row("schema_equal", "P/U", "*", set(positive) == set(unlabeled),
                     "info" if set(positive) == set(unlabeled) else "warning"))

    for label, frame in (("P", positive), ("U", unlabeled)):
        for key in ("leads_company_id", "credit_code", "company_name"):
            if key in frame:
                duplicate_count = int(frame[key].dropna().duplicated(keep=False).sum())
                rows.append(_row("duplicate", label, key, duplicate_count,
                                 "warning" if duplicate_count else "info"))
        for column in frame:
            missing = float(frame[column].isna().mean())
            rows.append(_row("missing_ratio", label, column, round(missing, 6),
                             "warning" if missing >= .8 else "info"))
            rows.append(_row("dtype", label, column, str(frame[column].dtype)))
            if pd.api.types.is_numeric_dtype(frame[column]):
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                if len(values):
                    q1, q3 = values.quantile([.25, .75])
                    iqr = q3 - q1
                    outliers = int(((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()) if iqr else 0
                    rows.append(_row("numeric_extreme_outliers", label, column, outliers))
            else:
                rows.append(_row("cardinality", label, column, int(frame[column].nunique(dropna=True))))
        for column in DATE_COLUMNS:
            if column in frame:
                source_non_null = int(frame[column].notna().sum())
                parsed = _parse_date_dayfirst(frame[column])
                success = float(parsed.notna().sum() / source_non_null) if source_non_null else np.nan
                rows.append(_row("date_parse_success_ratio", label, column, success,
                                 "warning" if pd.notna(success) and success < .9 else "info"))

    for column in sorted(set(positive) & set(unlabeled)):
        gap = abs(float(positive[column].isna().mean()) - float(unlabeled[column].isna().mean()))
        if gap >= .2:
            rows.append(_row("extreme_missingness_gap", "P/U", column, round(gap, 6), "warning"))

    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("数据验证报告：%s", output_path)
    return report
