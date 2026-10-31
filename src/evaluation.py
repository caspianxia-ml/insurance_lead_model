"""Ranking-oriented evaluation without pretending U is ground-truth negative."""
from pathlib import Path
import numpy as np
import pandas as pd

from .config import REPORTS_DIR


SCORE_COLUMNS = ["similarity_score", "lr_mean_score", "oneclass_svm_score", "isolation_forest_score"]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def write_evaluation(u_scores: pd.DataFrame, holdout_scores: pd.DataFrame,
                     output_path: Path = REPORTS_DIR / "baseline_evaluation.csv") -> pd.DataFrame:
    rows: list[dict] = []
    for score in SCORE_COLUMNS:
        for sample, frame in (("P_holdout", holdout_scores), ("U", u_scores)):
            values = frame[score].dropna()
            for statistic, value in values.describe(percentiles=[.01, .05, .1, .2, .5, .8, .9, .95, .99]).items():
                rows.append({"model": score, "sample": sample, "metric": statistic, "value": value})
        combined = pd.concat([
            holdout_scores[[score]].assign(sample="P_holdout"),
            u_scores[[score]].assign(sample="U"),
        ], ignore_index=True)
        combined["pct"] = percentile_rank(combined[score].to_numpy())
        p_pct = combined.loc[combined["sample"] == "P_holdout", "pct"]
        for top in (.01, .05, .10, .20):
            recall = float((p_pct >= 1 - top).mean()) if len(p_pct) else float("nan")
            rows.append({"model": score, "sample": "P_holdout", "metric": f"share_in_combined_top_{int(top*100)}pct", "value": recall})
    top_sets = {c: set(u_scores.index[u_scores[c].rank(pct=True) >= .9]) for c in SCORE_COLUMNS}
    intersection = set.intersection(*top_sets.values()) if top_sets else set()
    rows.append({"model": "all_models", "sample": "U", "metric": "top10_intersection_count", "value": len(intersection)})
    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return report
