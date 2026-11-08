#!/usr/bin/env python3
"""Enterprise insurance-benefit PU Learning V2 data audit (no model training).

Input CSVs: P_national, U1, U2, U3. Raw fields stay as strings to avoid
accidental damage to Chinese Unified Social Credit Codes.

Produces auditable cleaning decisions and summary reports. Company-name matches
are review-only, not automatic positive/unlabeled exclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

DATASETS = ("P_national", "U1", "U2", "U3")
NA_STRINGS = {"", "-", "--", "—", "–", "null", "none", "nan", "n/a", "na", "unknown", "未知", "暂无", "未公开"}
SIZE_COLUMNS = ("company_size", "employee_scale")
DATE_COLUMNS = ("updated_time", "updated_at", "modified_time", "modified_at", "created_time", "created_at")
CANDIDATE_COLUMNS = (
    "company_name", "source", "company_size", "employee_scale", "industry", "business_status",
    "registered_address", "company_introduction", "insured_count", "registered_capital", "score_details",
)


def is_missing(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in NA_STRINGS


def clean_code(value: object) -> tuple[str, str]:
    if is_missing(value):
        return "", "missing"
    val = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).upper())
    if val.startswith("'"):
        val = val[1:]
    if re.fullmatch(r"[0-9A-Z]{18}", val):
        return val, "format_18"  # Format check only; NOT a verified USCC checksum.
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[Ee][+-]?[0-9]+", val) or re.fullmatch(r"[0-9]+\.[0-9]+", val):
        return val, "suspected_excel_scientific_or_decimal"
    return val, "other_format"


def clean_name(value: object) -> str:
    if is_missing(value):
        return ""
    # Normalize display variants, not legal names or corporate-group affiliations.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).casefold()


def clean_size(value: object) -> str:
    if is_missing(value):
        return "Unknown"
    s = unicodedata.normalize("NFKC", str(value)).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "-").replace("～", "-").replace("~", "-").replace("至", "-")
    s = s.replace("人以上", "+").replace("人以下", "及以下")
    return s or "Unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    failure = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            # Important: never infer credit_code as float; keep source columns unchanged.
            frame = pd.read_csv(path, dtype="string", keep_default_na=False, encoding=encoding)
            frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
            return frame, encoding
        except UnicodeError as e:
            failure = e
    raise ValueError(f"Cannot decode {path}: {failure}")


def prepare(label: str, path: Path) -> tuple[pd.DataFrame, dict]:
    raw, encoding = read_csv(path)
    missing = {"credit_code", "company_name"} - set(raw.columns)
    if missing:
        raise ValueError(f"{label}: required columns missing: {sorted(missing)}; actual: {list(raw.columns)}")
    df = raw.copy()
    # Original line number assumes a single CSV header and no embedded line breaks.
    df["__record_id"] = [f"{label}#{i + 1}" for i in range(len(df))]
    df["__input_order"] = range(len(df))
    codes = df["credit_code"].map(clean_code)
    df["__credit_code_normalized"] = codes.map(lambda p: p[0])
    df["__code_format"] = codes.map(lambda p: p[1])
    df["__entity_code"] = df["__credit_code_normalized"].where(df["__code_format"].eq("format_18"), "")
    df["__name_normalized"] = df["company_name"].map(clean_name)
    df["__name_missing"] = df["__name_normalized"].eq("")
    nonempty_names = df["__name_normalized"].ne("")
    name_counts = df.loc[nonempty_names, "__name_normalized"].value_counts()
    df["__same_name_row_count"] = df["__name_normalized"].map(name_counts).fillna(0).astype(int)
    name_code_counts = df.loc[nonempty_names & df["__entity_code"].ne("")].groupby("__name_normalized")["__entity_code"].nunique()
    df["__same_name_distinct_valid_code_count"] = df["__name_normalized"].map(name_code_counts).fillna(0).astype(int)
    fields = [x for x in CANDIDATE_COLUMNS if x in df.columns]
    df["__completeness"] = df[fields].apply(lambda r: sum(not is_missing(v) for v in r), axis=1)

    # Prefer most complete row, then newest date, then earliest original row.
    # This is a deterministic within-file choice, NOT a merger of attribute values.
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for field in DATE_COLUMNS:
        if field in df.columns:
            parsed = pd.to_datetime(df[field].replace("", pd.NA), errors="coerce", utc=True)
            # Make timezone-naive for uniform sorting; used for tie-breaking only.
            parsed = parsed.dt.tz_localize(None)
            ts = ts.fillna(parsed)
    df["__preferred_date"] = ts
    df["__size_raw_missing"] = False
    for col in SIZE_COLUMNS:
        if col in df.columns:
            df[f"__{col}_normalized"] = df[col].map(clean_size)
            df["__size_raw_missing"] |= df[col].map(is_missing)
    if "source" in df.columns and label == "P_national":
        df["__gold_source_candidate_only"] = df["source"].str.contains("原力", na=False, regex=False)
    elif label == "P_national":
        df["__gold_source_candidate_only"] = False

    addr = df["registered_address"] if "registered_address" in df.columns else pd.Series("", index=df.index)
    raw_name = df["company_name"]
    df["__region_shanghai"] = addr.str.contains("上海", na=False, regex=False) | raw_name.str.contains("上海", na=False, regex=False)
    df["__region_suzhou"] = addr.str.contains("苏州", na=False, regex=False) | raw_name.str.contains("苏州", na=False, regex=False)

    df["__decision"] = "kept"
    df["__review_flags"] = ""
    ambiguous_names = df["__same_name_row_count"].gt(1) & (df["__same_name_distinct_valid_code_count"].gt(1) | df["__entity_code"].eq(""))
    df.loc[ambiguous_names, "__review_flags"] = "within_dataset_name_collision_review"
    df["__duplicate_group_size"] = 1
    valid = df["__entity_code"].ne("")
    if valid.any():
        counts = df.loc[valid, "__entity_code"].value_counts()
        df.loc[valid, "__duplicate_group_size"] = df.loc[valid, "__entity_code"].map(counts).astype(int)
        ordered = df.loc[valid].sort_values(
            by=["__completeness", "__preferred_date", "__input_order"],
            ascending=[False, False, True], kind="stable", na_position="last",
        )
        remove_ids = ordered.loc[ordered.duplicated("__entity_code", keep="first"), "__record_id"]
        df.loc[df["__record_id"].isin(set(remove_ids)), "__decision"] = "removed_duplicate_credit_code"

    meta = {"path": str(path), "sha256": sha256(path), "encoding": encoding, "columns": list(raw.columns), "raw_rows": len(df)}
    return df, meta


def data_quality(label: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in [c for c in df.columns if not c.startswith("__")]:
        vals = df[col].astype("string")
        miss = vals.map(is_missing)
        row = {
            "dataset": label, "field": col, "loaded_dtype": "string",
            "row_count": len(df), "missing_count": int(miss.sum()),
            "missing_pct": round(float(miss.mean() * 100), 4) if len(df) else 0.0,
            "nonempty_unique_count": int(vals[~miss].nunique()),
            "numeric_parse_note": "not automatically converted",
        }
        if col in ("registered_capital", "insured_count"):
            nonempty = vals[~miss]
            rate = pd.to_numeric(nonempty, errors="coerce").notna().mean() if len(nonempty) else 0.0
            row["numeric_parse_note"] = f"{rate * 100:.1f}% directly numeric; units/currency not standardized"
        rows.append(row)
    return rows


def summary(label: str, df: pd.DataFrame) -> dict:
    counts = df["__decision"].value_counts()
    code = df["__code_format"].value_counts()
    return {
        "dataset": label,
        "raw_rows": len(df),
        "credit_code_missing": int(code.get("missing", 0)),
        "credit_code_format_18": int(code.get("format_18", 0)),
        "credit_code_other_format": int(code.get("other_format", 0)),
        "credit_code_excel_risk": int(code.get("suspected_excel_scientific_or_decimal", 0)),
        "company_name_missing": int(df["__name_missing"].sum()),
        "same_name_multiple_row_groups": int(df.loc[df["__same_name_row_count"].gt(1), "__name_normalized"].nunique()),
        "same_name_multiple_valid_code_groups": int(df.loc[df["__same_name_distinct_valid_code_count"].gt(1), "__name_normalized"].nunique()),
        "duplicate_credit_code_groups": int(df.loc[df["__duplicate_group_size"].gt(1) & df["__entity_code"].ne(""), "__entity_code"].nunique()),
        "removed_duplicate_credit_code": int(counts.get("removed_duplicate_credit_code", 0)),
        "removed_known_P_credit_code": int(counts.get("removed_known_P_credit_code", 0)),
        "kept_rows": int(counts.get("kept", 0)),
        "suspected_P_name_match_review": int(df["__review_flags"].str.contains("P_name_match", regex=False).sum()),
        "shanghai_flag_rows": int(df["__region_shanghai"].sum()),
        "suzhou_flag_rows": int(df["__region_suzhou"].sum()),
        "gold_source_candidate_only": int(df["__gold_source_candidate_only"].sum()) if "__gold_source_candidate_only" in df else "",
        "reconciliation_ok": len(df) == int(sum(counts.values)),
    }


def run(paths: dict[str, Path], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaned").mkdir(exist_ok=True)
    data = {}
    metas = {}
    quality = []
    for label in DATASETS:
        data[label], metas[label] = prepare(label, paths[label])
        quality += data_quality(label, data[label])
    p = data["P_national"]
    p_keep = p.loc[p["__decision"].eq("kept")]
    p_codes = set(p_keep.loc[p_keep["__entity_code"].ne(""), "__entity_code"])
    p_names = set(p_keep.loc[p_keep["__name_normalized"].ne(""), "__name_normalized"])
    p_name_codes = p_keep.loc[p_keep["__name_normalized"].ne(""), ["__name_normalized", "__entity_code"]]
    p_name_code_map = p_name_codes.groupby("__name_normalized")["__entity_code"].apply(set).to_dict()

    conflicts = []
    name_review = []
    within_name_review = []
    for label in DATASETS[1:]:
        df = data[label]
        keep = df["__decision"].eq("kept")
        strict_overlap = keep & df["__entity_code"].ne("") & df["__entity_code"].isin(p_codes)
        # Never put confirmed P entities into a U training pool.
        df.loc[strict_overlap, "__decision"] = "removed_known_P_credit_code"
        for row in df.loc[strict_overlap, ["__record_id", "company_name", "__entity_code"]].itertuples(index=False, name=None):
            conflicts.append({"dataset": label, "record_id": row[0], "company_name": row[1], "credit_code_normalized": row[2], "match_type": "exact_credit_code", "action": "excluded_from_U"})
        possible = df["__decision"].eq("kept") & df["__name_normalized"].ne("") & df["__name_normalized"].isin(p_names)
        df.loc[possible, "__review_flags"] = df.loc[possible, "__review_flags"].map(lambda s: (s + "|" if s else "") + "P_name_match_review_only")
        for row in df.loc[possible, ["__record_id", "company_name", "credit_code", "__name_normalized", "__entity_code"]].itertuples(index=False, name=None):
            name_review.append({
                "dataset": label, "record_id": row[0], "company_name": row[1], "raw_credit_code": row[2],
                "name_normalized": row[3], "normalized_credit_code": row[4],
                "P_credit_codes_for_same_name": ";".join(sorted(p_name_code_map.get(row[3], set()))),
                "action": "manual_review_no_automatic_exclusion",
            })

    sum_rows = []
    decisions = []
    for label in DATASETS:
        df = data[label]
        stat = summary(label, df)
        if label == "U1":
            stat["out_of_shanghai_suzhou_audit_only"] = int((~df["__region_shanghai"] & ~df["__region_suzhou"]).sum())
        else:
            stat["out_of_shanghai_suzhou_audit_only"] = ""
        sum_rows.append(stat)
        collision = df["__review_flags"].str.contains("within_dataset_name_collision_review", regex=False)
        for row in df.loc[collision, ["__record_id", "company_name", "credit_code", "__name_normalized", "__same_name_row_count", "__same_name_distinct_valid_code_count"]].itertuples(index=False, name=None):
            within_name_review.append({"dataset": label, "record_id": row[0], "company_name": row[1], "credit_code": row[2], "name_normalized": row[3], "same_name_rows": row[4], "different_valid_codes": row[5], "action": "manual_review_no_automatic_entity_merge"})
        report_cols = ["__record_id", "credit_code", "__credit_code_normalized", "__code_format", "company_name", "__name_normalized", "__same_name_row_count", "__same_name_distinct_valid_code_count", "__duplicate_group_size", "__decision", "__review_flags", "__completeness", "__region_shanghai", "__region_suzhou"]
        audit = df[report_cols].copy()
        audit.insert(0, "dataset", label)
        decisions.append(audit)
        keep = df.loc[df["__decision"].eq("kept")].copy()
        # Output original source fields plus normalized scale columns, with no private audit columns.
        source_cols = [c for c in df.columns if not c.startswith("__")]
        normalized_cols = [c for c in ("__company_size_normalized", "__employee_scale_normalized") if c in df.columns]
        kept_out = keep[source_cols + normalized_cols].copy()
        kept_out.columns = [c.removeprefix("__") for c in kept_out.columns]
        kept_out.to_csv(out / "cleaned" / f"{label}_cleaned.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    for a, b in combinations(DATASETS[1:], 2):
        ca = set(data[a].loc[data[a]["__decision"].eq("kept") & data[a]["__entity_code"].ne(""), "__entity_code"])
        cb = set(data[b].loc[data[b]["__decision"].eq("kept") & data[b]["__entity_code"].ne(""), "__entity_code"])
        comparisons.append({"set_a": a, "set_b": b, "unique_valid_codes_a": len(ca), "unique_valid_codes_b": len(cb), "overlap_valid_codes": len(ca & cb), "codes_in_a_not_b": len(ca - cb), "codes_in_b_not_a": len(cb - ca), "note": "Different snapshots may violate intended subset; no automatic reassignment"})

    pd.DataFrame(quality).to_csv(out / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sum_rows).to_csv(out / "entity_cleaning_log.csv", index=False, encoding="utf-8-sig")
    pd.concat(decisions, ignore_index=True).to_csv(out / "record_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conflicts, columns=["dataset", "record_id", "company_name", "credit_code_normalized", "match_type", "action"]).to_csv(out / "P_U_exact_overlap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(name_review, columns=["dataset", "record_id", "company_name", "raw_credit_code", "name_normalized", "normalized_credit_code", "P_credit_codes_for_same_name", "action"]).to_csv(out / "P_U_name_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(within_name_review, columns=["dataset", "record_id", "company_name", "credit_code", "name_normalized", "same_name_rows", "different_valid_codes", "action"]).to_csv(out / "within_dataset_name_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(out / "U_subset_diagnostics.csv", index=False, encoding="utf-8-sig")
    run_info = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "input_files": metas, "pipeline": "v2-audit-0.1.0", "rules": {"dedup": "valid 18-char format credit_code within each dataset; completeness then newest date then stable row", "invalid_credit_code": "flag and retain; never silently treated as verified", "P_U": "exclude exact formatted credit-code overlaps from U; name-only matches review", "business_status": "audit later; no automatic removal", "Gold": "source contains 原力 is candidate only, never confirmed Gold"}}
    (out / "run_manifest.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    narrative = ["# V2 数据验收报告", "", f"生成时间 (UTC)：{run_info['generated_at_utc']}", "", "## 清洗前后数量", "", "| 数据集 | 原始行数 | 重复信用代码剔除 | 已知 P 信用代码重合剔除 | 保留行数 |", "|---|---:|---:|---:|---:|"]
    for a in sum_rows:
        narrative.append(f"| {a['dataset']} | {a['raw_rows']:,} | {a['removed_duplicate_credit_code']:,} | {a['removed_known_P_credit_code']:,} | {a['kept_rows']:,} |")
    narrative += ["", "## 规则与限制", "", "- 信用代码按字符串载入，仅 18 位字母数字格式作为严格实体键；这不是信用代码校验位验证。", "- 信用代码为空、长度异常或疑似 Excel 科学计数法时只标记，不自动删行。", "- 同名企业（尤其信用代码缺失或冲突）仅输出人工复核名单，绝不自动并入 P 或排除 U。", "- 企业规模标准化输出新列，不覆盖原始字段；业务状态异常不自动删除。", "- `source` 包含原力只是 Gold 候选，不等于有独立确认证据。", "- U1/U2/U3 各自独立去重与排除 P；跨集合只做诊断，不能把三个 U 相加。", "- 本次没有训练模型，没有划分 Holdout，也没有把地域/标签证据直接加入预测特征。", ""]
    (out / "audit_summary.md").write_text("\n".join(narrative), encoding="utf-8")
    return {"summary": sum_rows, "out_dir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 P/U data audit; CSVs only, no model training")
    parser.add_argument("--p", required=True, type=Path, help="P_national CSV")
    parser.add_argument("--u1", required=True, type=Path, help="U1 CSV")
    parser.add_argument("--u2", required=True, type=Path, help="U2 CSV")
    parser.add_argument("--u3", required=True, type=Path, help="U3 CSV")
    parser.add_argument("--out", default=Path("outputs/v2_data_audit"), type=Path)
    args = parser.parse_args()
    paths = dict(zip(DATASETS, (args.p, args.u1, args.u2, args.u3)))
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        parser.error("Missing input CSV files: " + ", ".join(missing))
    result = run(paths, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
