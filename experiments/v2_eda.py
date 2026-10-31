"""Read-only V2 data quality and EDA; only aggregate results are written."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import numpy as np
import pandas as pd

from src.config import ABNORMAL_STATUS_TOKENS, INSURED_COUNT_MAX
from src.data_cleaning import _parse_capital, _valid_uscc

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = pd.Timestamp("2026-10-20")
INPUTS = {key: ROOT / name for key, name in {
    "P": "V2_P_National_20261020_1200.csv",
    "U1": "V2_U1_20261020_1200.csv",
    "U2": "V2_U2_20261020_1200.csv",
    "U3": "V2_U3_20261020_1200.csv",
}.items()}
MISSING = {"", "null", "none", "nan", "nat", "未知", "-", "--"}
COMMON = ("credit_code", "company_name", "registered_address",
          "region_province", "region_city", "is_shanghai", "is_suzhou",
          "business_status", "establishment_date", "registered_capital",
          "paid_capital", "insured_count", "insured_count_report_year",
          "industry_category", "industry_major_category",
          "industry_medium_category", "company_type", "company_size",
          "has_company_introduction")
CATEGORIES = ("business_status", "industry_category", "industry_major_category",
              "company_type", "company_size", "region_province", "region_city")


def gold_candidate(source: pd.Series) -> pd.Series:
    """EDA policy: 原力 or the observed 量化Q3榜单 source token."""
    text = source.fillna("").astype(str)
    return (text.str.contains("原力", regex=False) |
            text.str.contains("量化Q3榜单", regex=False))


def is_missing(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(MISSING)


def normalize_code(value: object) -> str:
    code = unicodedata.normalize("NFKC", str(value or ""))
    code = re.sub(r"\s+", "", code).upper()
    return "" if code.lower() in MISSING else code


def read_csv(path: Path, weak: bool) -> tuple[pd.DataFrame, dict]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    strict_error = None
    if weak:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                for _ in reader:
                    pass
            except csv.Error as exc:
                strict_error = {"physical_line": reader.line_num, "error": str(exc)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=not weak)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"{path.name}: duplicate headers")
        rows = []
        for record_no, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f"{path.name}: record {record_no} has {len(row)} columns")
            rows.append(row)
    return pd.DataFrame.from_records(rows, columns=header), {
        "file": path.name, "rows": len(rows), "columns": len(header),
        "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
        "file_mtime_local": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
        "parser": "csv.reader(strict=False)" if weak else "csv.reader(strict=True)",
        "strict_error": strict_error, "wrong_width_rows": 0,
    }


def prepare(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    frame["_code"] = frame["credit_code"].map(normalize_code)
    frame["_uscc_ok"] = frame["_code"].map(lambda x: bool(x) and _valid_uscc(x))
    identifier = "leads_company_id" if cohort == "P" else "mid_company_clean_id"
    frame["_id"] = frame[identifier].fillna("").astype(str)
    frame["_filled"] = sum((~is_missing(frame[c])).astype(int) for c in COMMON)
    dates = pd.to_datetime(frame["establishment_date"].where(
        ~is_missing(frame["establishment_date"])), errors="coerce", format="mixed")
    frame["_date"] = dates
    frame["age_years"] = (SNAPSHOT - dates).dt.days / 365.25
    frame.loc[frame["age_years"] < 0, "age_years"] = np.nan
    for field in ("registered_capital", "paid_capital"):
        parsed = frame[field].map(_parse_capital)
        frame[field + "_yuan"] = parsed.map(lambda x: x[0])
        frame[field + "_parse_status"] = parsed.map(lambda x: x[2])
    count = frame["insured_count"].fillna("").astype(str).str.replace(
        ",", "", regex=False).str.strip()
    frame["insured_count_value"] = pd.to_numeric(count, errors="coerce")
    frame["_insured_year"] = pd.to_numeric(
        frame["insured_count_report_year"], errors="coerce")
    return frame


def representative(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.loc[frame["_code"].ne("")]
            .sort_values(["_code", "_filled", "_id"],
                         ascending=[True, False, False], kind="stable")
            .drop_duplicates("_code").reset_index(drop=True))


def inspect(frame: pd.DataFrame, cohort: str) -> tuple[dict, list[dict], list[dict]]:
    code = frame["_code"]
    present = code.ne("")
    duplicated = code[present].duplicated(keep=False)
    entity = {"cohort": cohort, "raw_rows": len(frame),
              "unique_codes": int(code[present].nunique()),
              "missing_code_rows": int((~present).sum()),
              "invalid_uscc_rows": int((present & ~frame["_uscc_ok"]).sum()),
              "duplicate_codes": int(code[present][duplicated].nunique()),
              "rows_in_duplicate_codes": int(duplicated.sum())}
    missing = []
    for field in frame.columns:
        if field.startswith("_") or field.endswith(("_yuan", "_parse_status", "_value")) or field == "age_years":
            continue
        n = int(is_missing(frame[field]).sum())
        missing.append({"cohort": cohort, "field": field, "rows": len(frame),
                        "missing_count": n, "missing_rate": n / len(frame)})
    checks = {}
    for field in ("registered_capital", "paid_capital"):
        for status in ("unparseable", "negative_amount", "non_cny"):
            checks[field + "_" + status] = int(frame[field + "_parse_status"].eq(status).sum())
    checks["establishment_unparseable_nonmissing"] = int((
        ~is_missing(frame["establishment_date"]) & frame["_date"].isna()).sum())
    checks["establishment_future"] = int((frame["_date"] > SNAPSHOT).sum())
    checks["establishment_before_1800"] = int((frame["_date"] < pd.Timestamp("1800-01-01")).sum())
    insured = frame["insured_count_value"]
    checks["insured_unparseable_nonmissing"] = int((
        ~is_missing(frame["insured_count"]) & insured.isna()).sum())
    checks["insured_negative"] = int((insured < 0).sum())
    checks["insured_over_limit"] = int((insured > INSURED_COUNT_MAX).sum())
    checks["insured_fractional"] = int((insured.notna() & (insured % 1 != 0)).sum())
    year = frame["_insured_year"]
    checks["insured_year_unparseable_nonmissing"] = int((
        ~is_missing(frame["insured_count_report_year"]) & year.isna()).sum())
    checks["insured_year_out_of_range"] = int(((year < 1900) | (year > 2026)).sum())
    for field in ("is_shanghai", "is_suzhou", "has_company_introduction"):
        checks[field + "_not_01"] = int((~frame[field].isin(["0", "1"])).sum())
    status = frame["business_status"].fillna("").astype(str)
    checks["abnormal_or_terminal_status"] = int(status.map(
        lambda x: any(token in x for token in ABNORMAL_STATUS_TOKENS)).sum())
    province = frame["region_province"].fillna("").astype(str)
    city = frame["region_city"].fillna("").astype(str)
    checks["shanghai_flag_region_mismatch"] = int((
        frame["is_shanghai"].eq("1") &
        ~(province.str.contains("上海", regex=False) |
          city.str.contains("上海", regex=False))).sum())
    checks["suzhou_flag_region_mismatch"] = int((
        frame["is_suzhou"].eq("1") & ~city.str.contains("苏州", regex=False)).sum())
    checks["html_style_in_address_or_city"] = int((
        frame["registered_address"].str.contains("style", case=False, na=False) |
        city.str.contains("style", case=False, na=False)).sum())
    if cohort == "U1":
        checks["outside_SH_SZ_flags"] = int((
            ~frame["is_shanghai"].eq("1") & ~frame["is_suzhou"].eq("1")).sum())
    if cohort == "P":
        quality = frame["p_quality"].fillna("")
        source = frame["source"].fillna("")
        derived_gold = gold_candidate(source)
        checks["gold_source_inconsistent"] = int((
            quality.eq("Gold") & ~source.str.contains("原力", regex=False)).sum())
        checks["silver_source_inconsistent"] = int((
            quality.eq("Silver") & source.str.contains("原力", regex=False)).sum())
        checks["export_quality_vs_gold_candidate_disagreement"] = int((
            quality.eq("Gold") != derived_gold).sum())
        checks["quality_other"] = int((~quality.isin(["Gold", "Silver"])).sum())
    anomalies = [{"cohort": cohort, "check": name, "count": n,
                  "rate": n / len(frame)} for name, n in sorted(checks.items())]
    return entity, missing, anomalies


def compare(groups: dict[str, pd.DataFrame]) -> tuple[list[dict], list[dict], list[dict]]:
    missing, numeric, categories = [], [], []
    for group, frame in groups.items():
        for field in COMMON:
            n = int(is_missing(frame[field]).sum())
            missing.append({"group": group, "field": field, "entities": len(frame),
                            "missing_count": n,
                            "missing_rate": n / len(frame) if len(frame) else np.nan})
        for field in ("age_years", "registered_capital_yuan",
                      "paid_capital_yuan", "insured_count_value"):
            values = pd.to_numeric(frame[field], errors="coerce")
            if field.endswith("_yuan"):
                values = values.where(frame[field.removesuffix("_yuan") +
                                            "_parse_status"].eq("ok"))
            if field == "insured_count_value":
                values = values.where((values >= 0) & (values <= INSURED_COUNT_MAX))
            values = values.dropna()
            numeric.append({
                "group": group, "field": field, "entities": len(frame),
                "valid_n": len(values), "median": values.median(),
                "q10": values.quantile(.1), "q25": values.quantile(.25),
                "q75": values.quantile(.75), "q90": values.quantile(.9)})
        for field in CATEGORIES:
            values = frame[field].fillna("").astype(str).str.strip()
            values = values.mask(values.str.lower().isin(MISSING), "<缺失>")
            counts = values.value_counts()
            for category, count in counts.head(15).items():
                categories.append({"group": group, "field": field,
                                   "category": category, "count": int(count),
                                   "share": count / len(frame)})
            if len(counts) > 15:
                n = int(counts.iloc[15:].sum())
                categories.append({"group": group, "field": field,
                                   "category": "<其他>", "count": n,
                                   "share": n / len(frame)})
    return missing, numeric, categories


def run(paths: dict[str, Path], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"run already exists: {out}")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True)
    frames, file_info, entities, missing_raw, anomalies = {}, {}, [], [], []
    for cohort, path in paths.items():
        frame, info = read_csv(path, weak=(cohort == "U3"))
        frames[cohort] = prepare(frame, cohort)
        file_info[cohort] = info
        entity, missing, checks = inspect(frames[cohort], cohort)
        entities.append(entity)
        missing_raw.extend(missing)
        anomalies.extend(checks)
    reps = {name: representative(frame) for name, frame in frames.items()}
    codes = {name: set(frame["_code"]) for name, frame in reps.items()}
    derived_gold = gold_candidate(frames["P"]["source"])
    gold = set(frames["P"].loc[derived_gold, "_code"]) - {""}
    silver = set(frames["P"].loc[~derived_gold, "_code"]) - {""}
    overlap = [{"comparison": "P_intersect_" + u, "entities": len(codes["P"] & codes[u])}
               for u in ("U1", "U2", "U3")]
    overlap += [{"comparison": "U1_not_in_U2", "entities": len(codes["U1"] - codes["U2"])},
                {"comparison": "U2_not_in_U3", "entities": len(codes["U2"] - codes["U3"])},
                {"comparison": "U1_not_in_U3", "entities": len(codes["U1"] - codes["U3"])},
        {"comparison": "Gold_candidate_intersect_Silver_candidate",
         "entities": len(gold & silver)}]
    u = {name: reps[name].loc[~reps[name]["_code"].isin(codes["P"])]
         for name in ("U1", "U2", "U3")}
    groups = {
        "P_national": reps["P"],
        "P_SH_SZ": reps["P"].loc[reps["P"]["is_shanghai"].eq("1") |
                                 reps["P"]["is_suzhou"].eq("1")],
        "P_Gold_candidate": reps["P"].loc[reps["P"]["_code"].isin(gold)],
        "P_Silver_candidate": reps["P"].loc[~reps["P"]["_code"].isin(gold)],
        "U1_nonP": u["U1"], "U2_nonP": u["U2"], "U3_nonP": u["U3"],
        "U2_minus_U1_nonP": u["U2"].loc[~u["U2"]["_code"].isin(codes["U1"])],
        "U3_minus_U2_nonP": u["U3"].loc[~u["U3"]["_code"].isin(codes["U2"])],
    }
    group_sizes = [{"group": name, "entities": len(frame)}
                   for name, frame in groups.items()]
    missing_group, numeric, categorical = compare(groups)
    flag_rates = []
    for group, frame in groups.items():
        for field in ("has_company_introduction", "has_score_details",
                      "is_shanghai", "is_suzhou"):
            if field not in frame:
                continue
            n = int(frame[field].eq("1").sum())
            flag_rates.append({"group": group, "flag": field,
                               "entities": len(frame), "value_1_count": n,
                               "value_1_rate": n / len(frame) if len(frame) else np.nan})
    p_missing_code_by_quality = []
    for quality, frame in frames["P"].groupby("p_quality", dropna=False):
        n = int(frame["_code"].eq("").sum())
        p_missing_code_by_quality.append({
            "p_quality": quality, "rows": len(frame),
            "missing_code_rows": n, "missing_code_rate": n / len(frame)})
    p_source = []
    candidate_quality = []
    for label, frame in (("Gold_candidate", frames["P"].loc[derived_gold]),
                         ("Silver_candidate", frames["P"].loc[~derived_gold])):
        candidate_quality.append({
            "group": label, "raw_rows": len(frame),
            "entities_with_code": frame.loc[frame["_code"].ne(""), "_code"].nunique(),
            "missing_code_rows": int(frame["_code"].eq("").sum())})
    for field in ("source_group", "p_quality", "source"):
        counts = frames["P"][field].fillna("").astype(str).value_counts()
        for value, n in counts.head(20).items():
            p_source.append({"field": field, "value": value, "rows": int(n),
                             "share": n / len(frames["P"])})
        if len(counts) > 20:
            n = int(counts.iloc[20:].sum())
            p_source.append({"field": field, "value": "<其他>", "rows": n,
                             "share": n / len(frames["P"])})
    outputs = {
        "entity_quality.csv": entities, "field_missingness_raw.csv": missing_raw,
        "field_missingness_by_group.csv": missing_group, "anomalies.csv": anomalies,
        "entity_overlaps.csv": overlap, "group_sizes.csv": group_sizes,
        "flag_rates_by_group.csv": flag_rates,
        "p_missing_code_by_quality.csv": p_missing_code_by_quality,
        "p_quality_candidate.csv": candidate_quality,
        "p_source_distribution.csv": p_source,
        "numeric_distribution.csv": numeric,
        "categorical_distribution.csv": categorical,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout.strip()
    notebook_path = ROOT / "notebooks" / "02_v2_data_quality_eda.ipynb"
    manifest = {
        "declared_snapshot": "2026-10-20 12:00 Asia/Shanghai",
        "snapshot_limit": "user-declared; strict historical as-of not proven",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_branch": branch, "git_commit": commit,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notebook_template_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "pandas": pd.__version__,
        "numpy": np.__version__, "inputs": file_info,
        "sql_lineage": "not supplied; welfare-line selection cannot be verified from compact U fields",
        "entity_rule": "normalized credit code; most complete row then largest source ID; missing codes excluded from entity comparisons",
        "comparison_rule": "all national P codes excluded from U; U remains unlabeled",
        "gold_candidate_rule": "source contains 原力 or 量化Q3榜单; original p_quality preserved, disagreement reported",
        "age_reference": "2026-10-20",
        "u3_parser_note": "relaxed quote parsing; wrong-width rows rejected, source text contamination audited",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(out), "input_rows": {k: v["rows"] for k, v in file_info.items()},
            "group_sizes": group_sizes, "overlaps": overlap}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    for cohort, path in INPUTS.items():
        parser.add_argument("--" + cohort.lower(), type=Path, default=path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,80}", args.run_id):
        parser.error("run-id: 3-81 letters, digits, underscores or hyphens")
    paths = {cohort: getattr(args, cohort.lower()) for cohort in INPUTS}
    print(json.dumps(run(paths, ROOT / "runs" / args.run_id),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
