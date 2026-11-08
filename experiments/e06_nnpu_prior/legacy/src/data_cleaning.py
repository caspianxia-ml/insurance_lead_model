"""Conservative, auditable cleaning for the P/U experiment.

Raw CSVs are never changed. This module produces a modelling population and
audit artefacts, while keeping any uncertain fields (notably Leads membership)
out of both the entity-resolution and feature paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import pandas as pd

from .config import (ABNORMAL_STATUS_TOKENS, DATE_COLUMNS, INSURED_COUNT_MAX,
                     PROCESSED_DIR, REPORTS_DIR)


USCC_CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"
USCC_WEIGHTS = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)
MISSING_TEXT = {"", "nan", "none", "null", "nat", "未知", "-", "--"}
PROFILE_COMPLETENESS_COLUMNS = (
    "company_type_clean", "company_size_clean", "business_status_clean",
    "registered_capital_yuan", "establishment_date", "industry_category_clean",
    "insured_count_clean",
)


@dataclass(frozen=True)
class PreparedData:
    """Cleaned P/U populations for the following pipeline stages."""

    positive: pd.DataFrame
    unlabeled: pd.DataFrame


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _normalise_text(series: pd.Series) -> pd.Series:
    value = series.map(_text)
    return value.mask(value.str.lower().isin(MISSING_TEXT), pd.NA)


def _normalise_credit_code(series: pd.Series) -> pd.Series:
    value = _normalise_text(series).astype("string")
    value = value.str.replace(r"\s+", "", regex=True).str.upper()
    return value.mask(value.isin(MISSING_TEXT), pd.NA)


def _valid_uscc(value: object) -> bool:
    if pd.isna(value):
        return False
    code = str(value)
    if len(code) != 18 or any(char not in USCC_CHARS for char in code):
        return False
    total = sum(USCC_CHARS.index(char) * weight for char, weight in zip(code[:17], USCC_WEIGHTS))
    return code[-1] == USCC_CHARS[(31 - total % 31) % 31]


def _parse_date_dayfirst(series: pd.Series) -> pd.Series:
    """Parse ISO first, then ambiguous dates as day/month/year."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.copy()
    raw = _normalise_text(series)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso_mask = raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", na=False)
    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(raw.loc[iso_mask], errors="coerce")
    other = raw.notna() & ~iso_mask
    if other.any():
        try:
            result.loc[other] = pd.to_datetime(raw.loc[other], format="mixed", dayfirst=True,
                                               errors="coerce")
        except (TypeError, ValueError):  # pandas < 2.0
            result.loc[other] = pd.to_datetime(raw.loc[other], dayfirst=True, errors="coerce")
    return result


def _parse_number(value: object) -> float:
    text = _text(value).replace(",", "")
    if text.lower() in MISSING_TEXT:
        return float("nan")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else float("nan")


def _parse_capital(value: object) -> tuple[float, str, str]:
    """Return CNY amount in yuan, currency label, and auditable parse status.

    No foreign-exchange conversion is invented. Chinese registry values with a
    Chinese unit but no explicit currency are marked ``CNY_assumed``.
    """
    text = _text(value).replace(",", "")
    if text.lower() in MISSING_TEXT:
        return float("nan"), "MISSING", "missing"
    amount = _parse_number(text)
    if np.isnan(amount):
        return float("nan"), "UNKNOWN", "unparseable"
    multiplier = 100_000_000.0 if "亿" in text else 10_000.0 if "万" in text else 1.0
    lower = text.lower()
    if amount < 0:
        return float('nan'), 'UNKNOWN', 'negative_amount'
    for token, currency in [('日元','JPY'),('英镑','GBP'),('新加坡','SGD'),('加元','CAD'),('澳元','AUD'),('瑞士','CHF'),('台币','TWD'),('韩元','KRW')]:
        if token in text:
            return float('nan'), currency, 'non_cny'
    if any(token in lower for token in ("usd", "美元", "$")):
        return float("nan"), "USD", "non_cny"
    if any(token in lower for token in ("eur", "欧元", "€")):
        return float("nan"), "EUR", "non_cny"
    if any(token in lower for token in ("hkd", "港币", "港元")):
        return float("nan"), "HKD", "non_cny"
    currency = "CNY" if any(token in text for token in ("人民币", "CNY", "元")) else "CNY_assumed"
    return amount * multiplier, currency, "ok"


def _normalise_company_type(value: object) -> object:
    text = _text(value)
    if not text or text.lower() in MISSING_TEXT:
        return pd.NA
    compact = re.sub(r"\s+", "", text)
    if "股份有限公司" in compact:
        return "股份有限公司"
    if "有限责任公司" in compact or compact.startswith("有限公司"):
        return "有限责任公司"
    if "合伙" in compact:
        return "合伙企业"
    if "个人独资" in compact:
        return "个人独资企业"
    if "个体工商户" in compact:
        return "个体工商户"
    if "事业单位" in compact:
        return "事业单位"
    return compact


def _normalise_size(value: object) -> object:
    text = _text(value)
    if not text or text.lower() in MISSING_TEXT:
        return pd.NA
    text = text.replace(" ", "").replace("－", "-").replace("~", "-").replace("至", "-")
    numbers = re.findall(r'\d+', text.replace(',', ''))
    if not numbers:
        return pd.NA
    lower = int(numbers[0])
    if '以下' in text or '少于' in text:
        lower = 0
    return '0-99' if lower < 100 else '100-999' if lower < 1000 else '1000+'


def _normalise_status(value: object) -> object:
    text = _text(value)
    if not text or text.lower() in MISSING_TEXT:
        return pd.NA
    compact = re.sub(r"\s+", "", text)
    if any(token in compact for token in ABNORMAL_STATUS_TOKENS):
        return "异常/终止"
    if any(token in compact for token in ("存续", "在业", "开业", "正常", "已上市")):
        return "正常经营"
    return compact


def _normalise_industry(value: object) -> object:
    text = _text(value)
    if not text or text.lower() in MISSING_TEXT:
        return pd.NA
    text = re.sub(r"\s+", "", text)
    return text.replace("｜", "丨").replace("|", "丨")


def _registered_region(row: pd.Series) -> tuple[object, str]:
    address = _text(row.get("registered_address"))
    province = _text(row.get("province"))
    city = _text(row.get("city"))
    if address:
        if '上海' in address:
            return '上海', 'registered_address'
        if '苏州' in address:
            return '苏州', 'registered_address'
    address_or_admin = " ".join((province, city))
    if "上海" in address_or_admin:
        return "上海", "registered_address" if "上海" in address else "province_or_city"
    if "苏州" in address_or_admin:
        return "苏州", "registered_address" if "苏州" in address else "province_or_city"
    original = _text(row.get("target_region"))
    if original:
        return original, "original_target_region_fallback"
    return pd.NA, "unresolved"


def _write_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _add_standardised_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_input_order"] = np.arange(len(result))
    credit = result["credit_code"] if "credit_code" in result else pd.Series(pd.NA, index=result.index)
    result["credit_code_normalized"] = _normalise_credit_code(credit)
    result["credit_code_format_valid"] = result["credit_code_normalized"].map(_valid_uscc)
    for column in DATE_COLUMNS:
        if column in result:
            result[column] = _parse_date_dayfirst(result[column])
    for column in ("registered_capital", "paid_capital"):
        if column in result:
            parsed = result[column].map(_parse_capital)
            result[f"{column}_yuan"] = parsed.map(lambda item: item[0])
            result[f"{column}_currency"] = parsed.map(lambda item: item[1])
            result[f"{column}_parse_status"] = parsed.map(lambda item: item[2])
    for column in ("company_type", "company_size", "business_status", "industry_category",
                   "industry_major_category", "industry_medium_category"):
        if column not in result:
            result[column] = pd.NA
    result["company_type_clean"] = result["company_type"].map(_normalise_company_type)
    result["company_size_clean"] = result["company_size"].map(_normalise_size)
    result["business_status_clean"] = result["business_status"].map(_normalise_status)
    for column in ("industry_category", "industry_major_category", "industry_medium_category"):
        result[f"{column}_clean"] = result[column].map(_normalise_industry)
    insured = result["insured_count"] if "insured_count" in result else pd.Series(np.nan, index=result.index)
    result["insured_count_clean"] = insured.map(_parse_number)
    result["insured_count_anomaly"] = np.select(
        [result["insured_count_clean"].lt(0), result["insured_count_clean"].gt(INSURED_COUNT_MAX)],
        ["negative", f"over_{INSURED_COUNT_MAX}"], default="ok_or_missing",
    )
    result.loc[result["insured_count_anomaly"] != "ok_or_missing", "insured_count_clean"] = np.nan
    result["insured_count_log1p"] = np.log1p(result["insured_count_clean"].clip(lower=0))
    for column in ("registered_capital_yuan", "paid_capital_yuan"):
        if column in result:
            result[f"{column.replace('_yuan', '')}_log1p"] = np.log1p(result[column].clip(lower=0))
    if "establishment_date" in result:
        result["company_age"] = ((pd.Timestamp.now().normalize() - result["establishment_date"]).dt.days / 365.25)
        result["company_age"] = result["company_age"].mask(result["company_age"] < 0)
    region = result.apply(_registered_region, axis=1, result_type="expand")
    result["target_region_corrected"] = region[0]
    result["region_correction_source"] = region[1]
    result["profile_completeness"] = result.reindex(columns=PROFILE_COMPLETENESS_COLUMNS).notna().sum(axis=1)
    for column in result.select_dtypes(include=['object','string']):
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def _audit_exclusions(frame: pd.DataFrame, mask: pd.Series, reason: str, sample: str) -> pd.DataFrame:
    detail = frame.loc[mask].copy()
    if detail.empty:
        return detail
    detail.insert(0, "sample", sample)
    detail.insert(1, "exclusion_reason", reason)
    return detail


def _deduplicate_by_entity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["profile_completeness", "_input_order"], ascending=[False, True], kind="stable")
    duplicate = ordered.duplicated("credit_code_normalized", keep="first")
    removed = ordered.loc[duplicate].copy()
    return ordered.loc[~duplicate].sort_values("_input_order", kind="stable").copy(), removed


def _quality_row(metric: str, sample: str, value: object, detail: str = "") -> dict:
    return {"metric": metric, "sample": sample, "value": value, "detail": detail}


def clean_and_prepare(positive: pd.DataFrame, unlabeled: pd.DataFrame,
                      report_dir=REPORTS_DIR, processed_dir=PROCESSED_DIR) -> PreparedData:
    """Build model populations and audit the exact exclusions at every stage."""
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_frames = {"P": _add_standardised_fields(positive), "U": _add_standardised_fields(unlabeled)}
    funnel: list[dict] = []
    quality: list[dict] = []
    exclusions: list[pd.DataFrame] = []
    retained: dict[str, pd.DataFrame] = {}

    for sample, frame in source_frames.items():
        details = frame.loc[(frame.credit_code_normalized.notna() & ~frame.credit_code_format_valid) |
                            frame.insured_count_anomaly.ne('ok_or_missing')]
        _write_csv(details, report_dir / f'{sample}_value_anomalies.csv')
        for field in ('registered_capital', 'paid_capital'):
            if f'{field}_parse_status' in frame:
                for status, count in frame[f'{field}_parse_status'].value_counts().items():
                    quality.append(_quality_row(f'{field}_{status}',sample,int(count)))
                _write_csv(frame.loc[~frame[f'{field}_parse_status'].isin(['ok','missing'])],
                           report_dir / f'{sample}_{field}_unconverted.csv')
        expected_label = 1 if sample == "P" else pd.NA
        expected_type = sample
        type_value = _normalise_text(frame.get("sample_type", pd.Series(pd.NA, index=frame.index)))
        labels = pd.to_numeric(frame.get("observed_positive", pd.Series(pd.NA, index=frame.index)), errors="coerce")
        bad_type = type_value.notna() & (type_value.str.upper() != expected_type)
        bad_label = (labels.isna() | (labels != 1)) if sample == "P" else labels.notna()
        if bad_type.any() or bad_label.any():
            raise ValueError(f'{sample}: sample_type 或 observed_positive 不符合 P=1/U=空 的约定。')
        quality += [
            _quality_row("sample_type_mismatch", sample, int(bad_type.sum())),
            _quality_row("observed_positive_mismatch", sample, int(bad_label.sum())),
            _quality_row("credit_code_missing", sample, int(frame["credit_code_normalized"].isna().sum())),
            _quality_row("credit_code_bad_format_or_checksum", sample,
                         int((frame["credit_code_normalized"].notna() & ~frame["credit_code_format_valid"]).sum())),
            _quality_row("abnormal_business_status", sample,
                         int((frame["business_status_clean"] == "异常/终止").sum())),
            _quality_row("insured_count_anomaly", sample,
                         int((frame["insured_count_anomaly"] != "ok_or_missing").sum())),
        ]
        funnel.append({"sample": sample, "stage": "input", "row_count": len(frame)})
        missing_credit = frame["credit_code_normalized"].isna()
        exclusions.append(_audit_exclusions(frame, missing_credit, "missing_credit_code", sample))
        current = frame.loc[~missing_credit].copy()
        funnel.append({"sample": sample, "stage": "after_missing_credit_code_filter", "row_count": len(current)})
        abnormal = current["business_status_clean"].eq("异常/终止")
        exclusions.append(_audit_exclusions(current, abnormal, "abnormal_business_status", sample))
        current = current.loc[~abnormal].copy()
        funnel.append({"sample": sample, "stage": "after_abnormal_status_filter", "row_count": len(current)})
        current, duplicate = _deduplicate_by_entity(current)
        exclusions.append(_audit_exclusions(duplicate, pd.Series(True, index=duplicate.index), "duplicate_credit_code", sample))
        funnel.append({"sample": sample, "stage": "after_entity_deduplication", "row_count": len(current)})
        current["sample_type"] = sample
        current["observed_positive"] = expected_label
        retained[sample] = current

    p_codes = set(source_frames["P"]["credit_code_normalized"].dropna())
    u_conflict = retained["U"]["credit_code_normalized"].isin(p_codes)
    overlap_codes = set(retained["U"].loc[u_conflict, "credit_code_normalized"])
    conflict_detail = pd.concat([
        retained["P"].loc[retained["P"]["credit_code_normalized"].isin(overlap_codes)].assign(sample="P"),
        retained["U"].loc[u_conflict].assign(sample="U"),
    ], ignore_index=True)
    _write_csv(conflict_detail, report_dir / "entity_overlap_conflicts.csv")
    quality.append(_quality_row("p_u_credit_code_overlap", "P/U", int(u_conflict.sum()),
                                "P takes precedence; matching U rows are removed from modelling U."))
    exclusions.append(_audit_exclusions(retained["U"], u_conflict, "credit_code_already_in_P", "U"))
    retained["U"] = retained["U"].loc[~u_conflict].copy()
    funnel.append({"sample": "U", "stage": "after_P_U_conflict_filter", "row_count": len(retained["U"])})
    funnel.append({"sample": "P", "stage": "final_modelling_population", "row_count": len(retained["P"])})
    funnel.append({"sample": "U", "stage": "final_modelling_population", "row_count": len(retained["U"])})

    parts = [part for part in exclusions if not part.empty]
    exclusion_detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["sample", "exclusion_reason"])
    _write_csv(exclusion_detail, report_dir / "excluded_records_detail.csv")
    _write_csv(pd.DataFrame(funnel), report_dir / "sample_funnel.csv")
    for sample, frame in retained.items():
        original_region = frame.get("target_region", pd.Series(pd.NA, index=frame.index))
        corrected = frame["target_region_corrected"].fillna("") != original_region.fillna("")
        quality += [
            _quality_row("final_modelling_rows", sample, len(frame)),
            _quality_row("unique_credit_codes", sample, int(frame["credit_code_normalized"].nunique())),
            _quality_row("registered_region_corrected", sample, int(corrected.sum()),
                         "Audit only; geography is never a model feature."),
        ]
    _write_csv(pd.DataFrame(quality), report_dir / "data_quality_report.csv")
    original_u_region = retained["U"].get("target_region", pd.Series(pd.NA, index=retained["U"].index))
    region_changed = retained["U"]["target_region_corrected"].fillna("") != original_u_region.fillna("")
    _write_csv(retained["U"].loc[region_changed], report_dir / "u_registered_region_corrections.csv")
    _write_csv(retained["P"], processed_dir / "positive_modeling.csv")
    _write_csv(retained["U"], processed_dir / "unlabeled_modeling.csv")
    return PreparedData(positive=retained["P"].reset_index(drop=True),
                        unlabeled=retained["U"].reset_index(drop=True))


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible one-frame cleaner for notebooks and feature creation."""
    return _add_standardised_fields(frame)
