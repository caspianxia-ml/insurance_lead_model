"""Read a Workbench JSON result set without changing the source export.

Some Workbench exports contain literal control characters inside JSON strings.
Python's non-strict decoder accepts those characters while still checking the
object/array structure. Never use this reader for an arbitrary network input.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def read_workbench_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as source:
        rows = json.load(source, strict=False)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Workbench JSON must be a non-empty array of objects")
    columns = set(rows[0])
    if not columns or any(set(row) != columns for row in rows):
        raise ValueError("Workbench JSON records have inconsistent fields")
    return pd.DataFrame.from_records(rows)


def inspect_workbench_json(path: Path) -> dict:
    frame = read_workbench_json(path)
    result = {"rows": len(frame), "columns": len(frame.columns),
              "duplicate_columns": int(frame.columns.duplicated().sum())}
    for field in ("sample_type", "has_welfare_raw", "target_region"):
        if field in frame:
            result[field] = {str(k): int(v) for k, v in
                             Counter(frame[field].fillna("MISSING")).items()}
    if "credit_code" in frame:
        result["missing_credit_code"] = int(frame.credit_code.isna().sum())
    return result
