"""New SH/SZ P + new U1, V1 four-model baseline with CV and U OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from experiments.v1_1_historical_b import _check_entities, _codes, _read, _recall_rows, _scores, _sha256
from experiments.v1_1_new_u1_c import _validate_new_u1
from experiments.validate_export import scan_csv
from src.config import PROJECT_ROOT, RANDOM_STATE, LR_REPEATS
from src.data_cleaning import _normalise_credit_code, clean_and_prepare
from src.evaluation import SCORE_COLUMNS
from src.feature_engineering import build_features
from src.model_logistic_baseline import repeated_sampled_u_scores, stratified_subset
from src.model_oneclass import oneclass_scores
from src.model_similarity import positive_similarity
from src.preprocessing import infer_feature_spec, make_preprocessor


def run(run_id: str, p_path: Path, u_path: Path, repeats: int = LR_REPEATS) -> Path:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for path in (p_path, u_path):
        if not path.is_file() or not scan_csv(path)["valid"]:
            raise ValueError(f"Missing or invalid CSV: {path}")
    raw_all_p, raw_u = _read(p_path), _read(u_path)
    if not {"is_shanghai", "is_suzhou", "credit_code", "p_quality"}.issubset(raw_all_p):
        raise ValueError("P lacks required columns")
    _validate_new_u1(raw_u)
    local = raw_all_p.is_shanghai.eq(1) | raw_all_p.is_suzhou.eq(1)
    raw_p = raw_all_p.loc[local].copy()
    raw_p["sample_type"], raw_p["observed_positive"] = "P", 1
    raw_u = raw_u.copy()
    raw_u["sample_type"] = "U"
    all_p_codes = set(_normalise_credit_code(raw_all_p.credit_code).dropna().astype(str))
    out = PROJECT_ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    prepared = clean_and_prepare(raw_p, raw_u, report_dir=out / "quality",
                                 processed_dir=out / "processed")
    p = build_features(prepared.positive, reference_date=pd.Timestamp("2026-10-20")).reset_index(drop=True)
    u = build_features(prepared.unlabeled, reference_date=pd.Timestamp("2026-10-20")).reset_index(drop=True)
    u = u.loc[~_codes(u).isin(all_p_codes)].reset_index(drop=True)
    _check_entities(p, "new local P")
    _check_entities(u, "new U1 after all-known-P exclusion")
    if len(p) < 50 or len(u) < 100:
        raise ValueError("Insufficient cleaned entities")
    idx_train, idx_hold = train_test_split(
        np.arange(len(p)), test_size=.2, random_state=RANDOM_STATE,
        stratify=p.p_quality if p.p_quality.value_counts().min() >= 2 else None)
    train, holdout = p.iloc[idx_train].reset_index(drop=True), p.iloc[idx_hold].reset_index(drop=True)
    pool = stratified_subset(u, max(1, int(len(u) * .4)), RANDOM_STATE)
    unseen = u.loc[~_codes(u).isin(set(_codes(pool)))].reset_index(drop=True)
    if set(_codes(pool)) & set(_codes(unseen)):
        raise RuntimeError("evaluation U overlaps the LR fit pool")
    metrics = []
    fixed = _scores(train, pool, pd.concat([unseen, holdout], ignore_index=True), repeats)
    metrics += _recall_rows(fixed.iloc[len(unseen):], fixed.iloc[:len(unseen)], "new_local_P_holdout")
    cv_rows = []
    for fold, (fit, val) in enumerate(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(train), 1):
        p_fit, p_val = train.iloc[fit], train.iloc[val]
        score = _scores(p_fit, pool, pd.concat([unseen, p_val], ignore_index=True), repeats)
        metrics += _recall_rows(score.iloc[len(unseen):], score.iloc[:len(unseen)], "new_local_P_5fold", fold)
        cv_rows += [{"fold": fold, "credit_code_normalized": code, **values}
                    for code, values in zip(_codes(p_val), score.iloc[len(unseen):].to_dict("records"))]
    pd.DataFrame(metrics).to_csv(out / "known_P_ranking_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "P_cv_oof_scores.csv", index=False)
    spec = infer_feature_spec(train, pool)
    scaled, unscaled = make_preprocessor(spec), make_preprocessor(spec, False)
    scores = pd.DataFrame(index=u.index)
    scores["similarity_score"], _ = positive_similarity(scaled, train, u)
    scores["oneclass_svm_score"], scores["isolation_forest_score"] = oneclass_scores(
        scaled, unscaled, train, u)
    lr = np.full(len(u), np.nan)
    fold_ids = np.zeros(len(u), dtype=int)
    for fold, (fit, val) in enumerate(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(u), 1):
        fit_u = u.iloc[fit]
        fit_pool = stratified_subset(fit_u, min(len(pool), len(fit_u)), RANDOM_STATE + fold)
        if set(_codes(fit_pool)) & set(_codes(u.iloc[val])):
            raise RuntimeError("U OOF leakage")
        fold_scaled = make_preprocessor(infer_feature_spec(train, fit_pool))
        lr[val], _ = repeated_sampled_u_scores(
            fold_scaled, train, fit_pool, score_targets=u.iloc[val], repeats=repeats)
        fold_ids[val] = fold
    if not np.isfinite(lr).all() or not (fold_ids > 0).all():
        raise RuntimeError("U OOF incomplete")
    scores["lr_mean_score"] = lr
    result = u[["credit_code_normalized", "company_name"]].copy()
    for column in SCORE_COLUMNS:
        result[column] = scores[column].to_numpy()
        result[column.replace("score", "rank")] = result[column].rank(
            ascending=False, method="min").astype(int)
    result["lr_oof_fold"] = fold_ids
    result.sort_values("lr_mean_rank").to_csv(out / "all_U1_oof_scores.csv", index=False)
    result.nsmallest(100, "lr_mean_rank").to_csv(out / "top100_candidates.csv", index=False)
    manifest = {
        "definition": "new SH/SZ P from national export + new U1; V1 features/four models",
        "reference_date": "2026-10-20", "seed": RANDOM_STATE, "folds": 5,
        "lr_repeats": repeats, "holdout_fraction": .2,
        "P_raw_local": len(raw_p), "P_clean": len(p), "P_train": len(train),
        "P_holdout": len(holdout), "U_raw": len(raw_u), "U_final": len(u),
        "U_fit_pool": len(pool), "U_evaluation_unseen": len(unseen),
        "U_exclusion": "all known national P codes, including nonlocal",
        "P_quality": "export Gold/Silver both trained; source quality not independently verified",
        "inputs": {str(x): _sha256(x) for x in (p_path, u_path)},
        "runner_sha256": _sha256(Path(__file__)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--p", type=Path, default=PROJECT_ROOT / "V2_P_National_20261020_1200.csv")
    parser.add_argument("--u1", type=Path, default=PROJECT_ROOT / "V2_U1_20261020_1200.csv")
    parser.add_argument("--repeats", type=int, default=LR_REPEATS)
    args = parser.parse_args()
    print(run(args.run_id, args.p, args.u1, args.repeats))
