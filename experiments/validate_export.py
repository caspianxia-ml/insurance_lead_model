"""Validate a CSV export's structure without printing enterprise records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def scan_csv(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"path": str(path.resolve()), "bytes": path.stat().st_size,
              "sha256": digest.hexdigest(), "expected_columns": 0,
              "logical_records": 0, "invalid_width_records": 0,
              "first_invalid_record_numbers": [], "parse_error": None}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader)
            result["expected_columns"] = len(header)
            result["duplicate_header_columns"] = len(header) - len(set(header))
            for record_number, row in enumerate(reader, 2):
                result["logical_records"] += 1
                if len(row) != len(header):
                    result["invalid_width_records"] += 1
                    if len(result["first_invalid_record_numbers"]) < 10:
                        result["first_invalid_record_numbers"].append(record_number)
    except (csv.Error, UnicodeError, StopIteration) as exc:
        result["parse_error"] = type(exc).__name__
    result["valid"] = bool(result["expected_columns"] and not result["parse_error"]
                           and not result["duplicate_header_columns"]
                           and not result["invalid_width_records"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    results = [scan_csv(path) for path in args.paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(item["valid"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
