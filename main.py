"""Command-line entry point for the PU-ranking POC."""
import argparse
import logging
import sys

from pathlib import Path
from src.config import OUTPUT_DIR, PROCESSED_DIR, REPORTS_DIR, POSITIVE_PATH, UNLABELED_PATH
from src.data_cleaning import clean_and_prepare
from src.data_loader import DataFileMissingError, load_positive, load_unlabeled
from src.data_validation import validate_data
from src.feature_engineering import build_features
from src.leakage_check import check_leakage
from src.preprocessing import infer_feature_spec, make_preprocessor
from src.score_unlabeled import score_unlabeled

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("insurance_lead_model")


def _check(p, u) -> None:
    validate_data(p, u)
    check_leakage(p, u)


def _features(p, u) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p_features, u_features = build_features(p), build_features(u)
    spec = infer_feature_spec(p_features, u_features)
    if not spec.all:
        raise ValueError("未找到可用候选特征。")
    # Fit only against P here as a preprocessing smoke test; no combined P+U fit.
    # Feature export does not fit any learned preprocessing.
    p_features.to_csv(PROCESSED_DIR / "positive_features.csv", index=False, encoding="utf-8-sig")
    u_features.to_csv(PROCESSED_DIR / "unlabeled_features.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("特征工程完成；实际特征 numeric=%s, categorical=%s", spec.numeric, spec.categorical)


def run(step: str, positive_path=POSITIVE_PATH, unlabeled_path=UNLABELED_PATH, sensitivity=False) -> None:
    p, u = load_positive(positive_path), load_unlabeled(unlabeled_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.eda import source_diagnostics
    source_diagnostics(p, u, REPORTS_DIR / 'raw_source_audit')
    if step in ("check", "all"):
        _check(p, u)
    prepared = clean_and_prepare(p, u)
    p, u = prepared.positive, prepared.unlabeled
    from src.eda import source_diagnostics
    source_diagnostics(p, u)
    if step in ("eda", "all"):
        from src.eda import run_eda
        run_eda(p, u)
    if step in ("features", "all"):
        _features(p, u)
    if step == "baseline":
        score_unlabeled(p, u, PROCESSED_DIR / "baseline_unlabeled_scores.csv", sensitivity=sensitivity)
    if step in ("score", "all"):
        score_unlabeled(p, u, sensitivity=sensitivity)


def main() -> int:
    parser = argparse.ArgumentParser(description="保险福利线索 PU ranking POC")
    parser.add_argument("--step", choices=["check", "eda", "features", "baseline", "score", "all"],
                        default="all")
    parser.add_argument('--positive', type=Path, default=POSITIVE_PATH)
    parser.add_argument('--unlabeled', type=Path, default=UNLABELED_PATH)
    parser.add_argument('--sensitivity', action='store_true', help='额外进行分源排除训练敏感性分析')
    args = parser.parse_args()
    try:
        run(args.step, args.positive, args.unlabeled, args.sensitivity)
        return 0
    except DataFileMissingError:
        print("数据文件不存在。\n请将 P 样本放到：data/raw/positive.csv\n"
              "请将 U 样本放到：data/raw/unlabeled.csv", file=sys.stderr)
        return 2
    except (ValueError, UnicodeError) as exc:
        LOGGER.error("执行失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
