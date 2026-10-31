"""Stage new U1 cleaning, before new-P exclusions and experiment C scoring."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn

from experiments.v1_1_historical_b import _check_entities, _codes, _read, _sha256
from experiments.v1_1_json_export import inspect_workbench_json, read_workbench_json
from src.config import PROJECT_ROOT
from src.data_cleaning import clean_and_prepare


def run(run_id: str, new_u_path: Path, historical_positive: Path) -> Path:
    for path in (new_u_path, historical_positive):
        if not path.is_file():
            raise FileNotFoundError(path)
    diagnostic = inspect_workbench_json(new_u_path)
    if diagnostic.get("sample_type") != {"U1": diagnostic["rows"]}:
        raise ValueError("U1 导出的 sample_type 不全为 U1")
    if diagnostic.get("has_welfare_raw") != {"1": diagnostic["rows"]}:
        raise ValueError("U1 导出的福利线标记不全为 1")
    if set(diagnostic.get("target_region", {})) - {"上海", "苏州"}:
        raise ValueError("U1 导出含上海、苏州以外的目标地区")
    raw_u = read_workbench_json(new_u_path)
    raw_u["sample_type"] = "U"  # V1 cleaner marker; source file remains unchanged.
    old_p = _read(historical_positive)
    _check_entities(old_p, "历史 P")

    output = PROJECT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    prepared = clean_and_prepare(old_p, raw_u,
                                 report_dir=output / "quality",
                                 processed_dir=output / "processed")
    if set(_codes(prepared.positive)) != set(_codes(old_p)):
        raise ValueError("历史 P 重新清洗后实体集合改变，停止审计")
    _check_entities(prepared.unlabeled, "新 U1 已清洗")
    manifest = {
        "stage": "new_U1_cleaning_only",
        "status": "provisional_pending_new_P_exclusion_and_C_scoring",
        "run_id": run_id,
        "new_U1_export": {"path": str(new_u_path.resolve()),
                          "sha256": _sha256(new_u_path),
                          "bytes": new_u_path.stat().st_size,
                          "mtime": new_u_path.stat().st_mtime,
                          "json_decode": "strict_false_for_Workbench_literal_control_characters",
                          **diagnostic},
        "historical_P": {"path": str(historical_positive.resolve()),
                         "sha256": _sha256(historical_positive),
                         "rows": len(old_p)},
        "cleaned_counts": {"historical_P": len(prepared.positive),
                           "new_U1_before_new_P_exclusion": len(prepared.unlabeled)},
        "versions": {"python": platform.python_version(),
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--new-u1", required=True, type=Path)
    parser.add_argument("--historical-positive", type=Path,
                        default=PROJECT_ROOT / "data/processed/positive_modeling.csv")
    args = parser.parse_args()
    print(run(args.run_id, args.new_u1, args.historical_positive))


if __name__ == "__main__":
    main()
