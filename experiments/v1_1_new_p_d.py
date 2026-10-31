"""V1.1 D: new P and new U1, compared with C on frozen entities."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold

from experiments.v1_1_historical_b import (
    _check_entities, _codes, _read, _read_archived_split, _recall_rows, _scores, _sha256,
)
from experiments.v1_1_new_u1_c import _validate_new_u1
from experiments.validate_export import scan_csv
from src.config import PROJECT_ROOT, RANDOM_STATE, LR_REPEATS
from src.data_cleaning import clean_and_prepare
from src.evaluation import SCORE_COLUMNS
from src.feature_engineering import build_features
from src.model_logistic_baseline import repeated_sampled_u_scores, stratified_subset
from src.model_oneclass import oneclass_scores
from src.model_similarity import positive_similarity
from src.preprocessing import infer_feature_spec, make_preprocessor


def run(run_id: str, new_p_path: Path, new_u_path: Path, c_run: Path,
        historical_positive: Path, historical_unlabeled: Path, historical_split: Path,
        seeds: list[int], repeats: int, reference_date: pd.Timestamp) -> Path:
    if not seeds or len(set(seeds)) != len(seeds) or repeats < 1:
        raise ValueError("种子必须非空、不重复，LR 重复次数必须为正")
    paths = [new_p_path, new_u_path, historical_positive, historical_unlabeled, historical_split]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    c_eval_path = c_run / "fixed_common_evaluation_scores.csv"
    c_u_path = c_run / "all_new_U1_oof_scores.csv"
    for path in (c_eval_path, c_u_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    diagnostics = {path.name: scan_csv(path) for path in (new_p_path, new_u_path)}
    if not all(item["valid"] for item in diagnostics.values()):
        raise ValueError("新 P/U1 CSV 的严格结构检查未通过")

    raw_p, raw_u = _read(new_p_path), _read(new_u_path)
    needed_p = {"credit_code", "p_quality", "source"}
    if not needed_p.issubset(raw_p.columns):
        raise ValueError(f"新 P 缺少字段：{sorted(needed_p - set(raw_p.columns))}")
    if set(raw_p.p_quality.dropna()) != {"Gold", "Silver"} or raw_p.p_quality.isna().any():
        raise ValueError("新 P 的 Gold/Silver 标签必须完整且口径明确")
    export_checks = _validate_new_u1(raw_u)
    if "sample_type" in raw_p and not raw_p.sample_type.astype(str).str.upper().eq("P").all():
        raise ValueError("新 P 原有 sample_type 与 P 冲突")
    raw_p = raw_p.copy()
    raw_u = raw_u.copy()
    raw_p["sample_type"], raw_p["observed_positive"] = "P", 1
    raw_u["sample_type"] = "U"

    old_p, old_u = _read(historical_positive), _read(historical_unlabeled)
    _check_entities(old_p, "历史 P")
    _check_entities(old_u, "历史 U")
    old_p_features = build_features(old_p, reference_date=reference_date).reset_index(drop=True)
    old_u_features = build_features(old_u, reference_date=reference_date).reset_index(drop=True)
    roles = _read_archived_split(historical_split, old_p_features, old_u_features)
    fixed_holdout = roles["P_holdout"]
    holdout_codes = set(_codes(fixed_holdout))

    output = PROJECT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    prepared = clean_and_prepare(raw_p, raw_u, report_dir=output / "quality",
                                 processed_dir=output / "processed")
    p = build_features(prepared.positive, reference_date=reference_date).reset_index(drop=True)
    u = build_features(prepared.unlabeled, reference_date=reference_date).reset_index(drop=True)
    # New P and all historical known P are exclusion evidence; neither can remain in U.
    old_p_overlap = _codes(u).isin(set(_codes(old_p)))
    extra_excluded = int(old_p_overlap.sum())
    u.loc[old_p_overlap, ["credit_code_normalized"]].assign(
        exclusion_reason="historical_P_not_in_new_P"
    ).to_csv(output / "quality/historical_P_U_extra_conflicts.csv", index=False,
             encoding="utf-8-sig")
    u = u.loc[~old_p_overlap].reset_index(drop=True)
    _check_entities(p, "新 P")
    _check_entities(u, "新 U1")
    c_u = _read(c_u_path)
    if set(_codes(u)) != set(_codes(c_u)):
        raise ValueError("D 与 C 的最终 U1 实体集合不同，停止单变量 P 对照")

    train = p.loc[~_codes(p).isin(holdout_codes)].reset_index(drop=True)
    if set(_codes(train)) & holdout_codes:
        raise RuntimeError("固定 P holdout 泄漏到 D 训练集")
    common_codes = set(_codes(roles["U_unseen"])) & set(_codes(u))
    common_u = u.loc[_codes(u).isin(common_codes)].reset_index(drop=True)
    pool_candidates = u.loc[~_codes(u).isin(common_codes)]
    pool_size = max(1, int(len(u) * .4))
    if len(pool_candidates) < pool_size:
        raise ValueError("非冻结 U 不足以形成 40% 训练池")
    u_pool = stratified_subset(pool_candidates, pool_size, RANDOM_STATE)
    c_eval = _read(c_eval_path)
    if set(c_eval.loc[c_eval.role == "U_common", "credit_code_normalized"]) != common_codes:
        raise ValueError("C/D 共用 U 评价实体不一致")
    if set(c_eval.loc[c_eval.role == "P_holdout", "credit_code_normalized"]) != holdout_codes:
        raise ValueError("C/D 固定 P 留出实体不一致")

    manifest = {
        "experiment": "D", "run_id": run_id, "snapshot_claim": "2026-10-20 noon; not a strict historical DB snapshot",
        "runner_sha256": _sha256(Path(__file__)),
        "C_reference": {"run_dir": str(c_run.resolve()),
                        "fixed_evaluation_sha256": _sha256(c_eval_path),
                        "final_U_scores_sha256": _sha256(c_u_path)},
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "seeds": seeds, "cv_folds": 5, "lr_repeats": repeats,
        "P_policy": "all_observed_Gold_and_Silver; quality_is_candidate_source_not_independent_verification",
        "holdout_policy": "historical_P_holdout_excluded_from_new_P_training",
        "U_policy": "same_final_entities_as_C; new_and_historical_known_P_excluded",
        "new_U1_export_checks": export_checks,
        "csv_checks": diagnostics,
        "files": {str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size,
                              "modified_time": path.stat().st_mtime} for path in paths},
        "counts": {"new_P_raw": len(raw_p), "new_P_clean": len(p),
                   "new_P_train": len(train), "new_P_quality_raw": raw_p.p_quality.value_counts().to_dict(),
                   "new_P_quality_clean": p.p_quality.value_counts().to_dict(),
                   "new_U_raw": len(raw_u), "new_U_after_new_P_clean": len(prepared.unlabeled),
                   "historical_P_extra_U_excluded": extra_excluded, "new_U_final": len(u),
                   "common_frozen_U": len(common_u), "fixed_historical_P_holdout": len(fixed_holdout),
                   "U_fit_pool": len(u_pool)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    pd.DataFrame({"credit_code_normalized": _codes(train),
                  "p_quality": train.p_quality}).to_csv(output / "P_train_entities.csv", index=False)
    pd.DataFrame({"credit_code_normalized": _codes(fixed_holdout)}).to_csv(
        output / "fixed_P_holdout_entities.csv", index=False)

    targets = pd.concat([common_u, fixed_holdout], ignore_index=True)
    fixed_scores = _scores(train, u_pool, targets, repeats)
    fixed_output = pd.concat([
        pd.DataFrame({"credit_code_normalized": _codes(targets).reset_index(drop=True),
                      "role": ["U_common"] * len(common_u) + ["P_holdout"] * len(fixed_holdout)}),
        fixed_scores.reset_index(drop=True)], axis=1)
    fixed_output.to_csv(output / "fixed_common_evaluation_scores.csv", index=False,
                        encoding="utf-8-sig")
    aligned = c_eval[["credit_code_normalized", "role", *SCORE_COLUMNS]].merge(
        fixed_output[["credit_code_normalized", "role", *SCORE_COLUMNS]],
        on=["credit_code_normalized", "role"], suffixes=("_C", "_D"), validate="one_to_one")
    if len(aligned) != len(fixed_output):
        raise ValueError("C/D 固定评价集合行数不一致")
    common_aligned = aligned.loc[aligned.role == "U_common"]
    comparisons = []
    for column in SCORE_COLUMNS:
        top_c = set(common_aligned.nlargest(100, f"{column}_C").credit_code_normalized)
        top_d = set(common_aligned.nlargest(100, f"{column}_D").credit_code_normalized)
        comparisons.append({"model": column, "common_U_count": len(common_aligned),
                            "spearman_C_vs_D": float(common_aligned[f"{column}_C"].corr(
                                common_aligned[f"{column}_D"], method="spearman")),
                            "top100_jaccard": len(top_c & top_d) / len(top_c | top_d)})
    pd.DataFrame(comparisons).to_csv(output / "C_D_common_U_comparison.csv", index=False)
    eda = []
    for sample, frame in (("P_clean", p), ("U1_final", u)):
        for column in ("company_age", "registered_capital_log1p", "paid_capital_log1p",
                       "insured_count_log1p", "company_type_clean", "company_size_clean",
                       "industry_major_category_clean", "industry_medium_category_clean",
                       "has_company_introduction"):
            if column in frame:
                eda.append({"sample": sample, "field": column, "row_count": len(frame),
                            "missing_count": int(frame[column].isna().sum())})
    pd.DataFrame(eda).to_csv(output / "quality/incremental_eda_missingness.csv", index=False)
    metrics = _recall_rows(fixed_scores.iloc[len(common_u):].reset_index(drop=True),
                           fixed_scores.iloc[:len(common_u)].reset_index(drop=True),
                           "D_fixed_common_U")
    qualities = train.p_quality.astype(str)
    if qualities.value_counts().min() < 5:
        raise ValueError("Gold/Silver 样本不足以做 5 折分层验证")
    cv_rows = []
    for fold, (fit_idx, val_idx) in enumerate(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(train, qualities), 1):
        p_fit, p_val = train.iloc[fit_idx], train.iloc[val_idx]
        fold_scores = _scores(p_fit, u_pool, pd.concat([common_u, p_val], ignore_index=True), repeats)
        metrics += _recall_rows(fold_scores.iloc[len(common_u):].reset_index(drop=True),
                               fold_scores.iloc[:len(common_u)].reset_index(drop=True),
                               "D_new_P_5fold_CV", fold)
        cv_rows += [{"fold": fold, "credit_code_normalized": code,
                     "p_quality": quality, **values}
                    for code, quality, values in zip(_codes(p_val), p_val.p_quality,
                                                     fold_scores.iloc[len(common_u):].to_dict("records"))]
    pd.DataFrame(metrics).to_csv(output / "known_P_ranking_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output / "new_P_cv_oof_scores.csv", index=False)

    spec = infer_feature_spec(train, u_pool)
    scaled, unscaled = make_preprocessor(spec), make_preprocessor(spec, False)
    p_only = pd.DataFrame(index=u.index)
    p_only["similarity_score"], _ = positive_similarity(scaled, train, u)
    p_only["oneclass_svm_score"], p_only["isolation_forest_score"] = oneclass_scores(
        scaled, unscaled, train, u)
    seed_scores = {}
    fold_rows = []
    for seed in seeds:
        values = np.full(len(u), np.nan)
        for fold, (fit_idx, score_idx) in enumerate(KFold(
                n_splits=5, shuffle=True, random_state=seed).split(u), 1):
            fit_u = u.iloc[fit_idx]
            pool = stratified_subset(fit_u, min(len(u_pool), len(fit_u)), seed + fold)
            if set(_codes(pool)) & set(_codes(u.iloc[score_idx])):
                raise RuntimeError("D 的 LR U 折外评分出现自身训练")
            fold_scaled = make_preprocessor(infer_feature_spec(train, pool))
            fold_values, _ = repeated_sampled_u_scores(
                fold_scaled, train, pool, score_targets=u.iloc[score_idx], repeats=repeats)
            values[score_idx] = fold_values
            fold_rows.append({"seed": seed, "fold": fold, "count": len(score_idx),
                              "mean": float(np.mean(fold_values)), "std": float(np.std(fold_values))})
        if not np.isfinite(values).all():
            raise RuntimeError("D 的 LR 折外评分未覆盖全部 U")
        seed_scores[seed] = values
    pd.DataFrame(fold_rows).to_csv(output / "U_lr_fold_scale.csv", index=False)
    scored = u[["credit_code_normalized", "company_name"]].copy()
    for column in SCORE_COLUMNS:
        scored[column] = seed_scores[seeds[0]] if column == "lr_mean_score" else p_only[column].to_numpy()
        scored[column.replace("score", "rank")] = scored[column].rank(
            ascending=False, method="min").astype(int)
    scored["score_population"] = "new_U1_all_LR_out_of_fold"
    scored.sort_values("lr_mean_rank").to_csv(output / "all_new_U1_oof_scores.csv",
                                                index=False, encoding="utf-8-sig")
    scored.nsmallest(100, "lr_mean_rank").to_csv(output / "top100_candidates.csv",
                                                  index=False, encoding="utf-8-sig")
    stability = []
    for seed in seeds[1:]:
        base, other = seed_scores[seeds[0]], seed_scores[seed]
        for k in (100, 500, 1000):
            a, b = set(np.argsort(-base)[:k]), set(np.argsort(-other)[:k])
            stability.append({"seed_a": seeds[0], "seed_b": seed, "top_k": k,
                              "jaccard": len(a & b) / len(a | b),
                              "spearman": float(pd.Series(base).corr(pd.Series(other), method="spearman"))})
    pd.DataFrame(stability).to_csv(output / "U_lr_seed_stability.csv", index=False)
    (output / "REPORT_zh.md").write_text(
        "# V1.1 D：新 P＋新 U1\n\n"
        f"新 P 清洗后 {len(p):,} 家，固定历史 P 留出 {len(fixed_holdout):,} 家不参与拟合。"
        f"新 U1 最终 {len(u):,} 家，C/D 共用评价 U {len(common_u):,} 家。"
        "Gold/Silver 均作为观察正例，但 Gold 只是来源候选，不是独立核验。"
        "本实验不将 U 当真负类；分数不是概率。详见 manifest 与各项 CSV。\n",
        encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--new-p", type=Path, required=True)
    parser.add_argument("--new-u1", type=Path, required=True)
    parser.add_argument("--c-run", type=Path, default=PROJECT_ROOT / "runs/20261021_C_1200_v1_1")
    parser.add_argument("--historical-positive", type=Path,
                        default=PROJECT_ROOT / "data/processed/positive_modeling.csv")
    parser.add_argument("--historical-unlabeled", type=Path,
                        default=PROJECT_ROOT / "data/processed/unlabeled_modeling.csv")
    parser.add_argument("--historical-split", type=Path,
                        default=PROJECT_ROOT / "reports/entity_split.csv")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--repeats", type=int, default=LR_REPEATS)
    parser.add_argument("--reference-date", default="2026-10-20")
    args = parser.parse_args()
    print(run(args.run_id, args.new_p, args.new_u1, args.c_run, args.historical_positive,
              args.historical_unlabeled, args.historical_split,
              [int(value) for value in args.seeds.split(",")], args.repeats,
              pd.Timestamp(args.reference_date)))


if __name__ == "__main__":
    main()
