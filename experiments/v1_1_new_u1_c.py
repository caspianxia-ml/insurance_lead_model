"""V1.1 C: fixed historical P, freshly cleaned U1, and exclusion-only new P."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold

from experiments.v1_1_historical_b import (
    _check_entities, _codes, _read, _read_archived_split, _recall_rows,
    _scores, _sha256,
)
from experiments.validate_export import scan_csv
from experiments.v1_1_json_export import read_workbench_json, inspect_workbench_json
from src.config import PROJECT_ROOT, RANDOM_STATE, LR_REPEATS
from src.data_cleaning import _normalise_credit_code, clean_and_prepare
from src.evaluation import SCORE_COLUMNS
from src.feature_engineering import build_features
from src.model_logistic_baseline import repeated_sampled_u_scores, stratified_subset
from src.model_oneclass import oneclass_scores
from src.model_similarity import positive_similarity
from src.preprocessing import infer_feature_spec, make_preprocessor


def _quality_summary(old_u: pd.DataFrame, new_u: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in (("old_U1_clean", old_u), ("new_U1_clean", new_u)):
        rows.append({"snapshot": name, "dimension": "total", "value": "all", "count": len(frame)})
        for column in ("target_region_corrected", "source", "company_size_clean"):
            if column in frame:
                for value, count in frame[column].fillna("MISSING").value_counts().items():
                    rows.append({"snapshot": name, "dimension": column,
                                 "value": str(value), "count": int(count)})
        for column in ("company_age", "registered_capital_log1p", "paid_capital_log1p",
                       "insured_count_log1p"):
            if column in frame:
                rows.append({"snapshot": name, "dimension": f"missing:{column}",
                             "value": "missing_count", "count": int(frame[column].isna().sum())})
    return pd.DataFrame(rows)


def _new_p_codes(paths: list[Path]) -> set[str]:
    codes: set[str] = set()
    for path in paths:
        frame = (read_workbench_json(path) if path.suffix.lower() == ".json"
                 else pd.read_csv(path, encoding="utf-8-sig", usecols=["credit_code"],
                                  dtype={"credit_code": "string"}, low_memory=False))
        if "credit_code" not in frame:
            raise ValueError(f"新 P 缺少信用代码字段：{path}")
        normalised = _normalise_credit_code(frame["credit_code"])
        codes.update(normalised.dropna().astype(str))
    return codes


def _validate_new_u1(frame: pd.DataFrame) -> dict[str, object]:
    """Check explicit export markers without requiring legacy JSON-only columns."""
    if "sample_type" in frame and not frame.sample_type.astype(str).str.upper().eq("U1").all():
        raise ValueError("新 U1 的 sample_type 必须全部是 U1")
    if "has_welfare_raw" in frame and not pd.to_numeric(
            frame.has_welfare_raw, errors="coerce").eq(1).all():
        raise ValueError("新 U1 的福利线标记必须全部为 1")
    if "sample_type" not in frame:
        if not {"is_shanghai", "is_suzhou"}.issubset(frame.columns):
            raise ValueError("精简 U1 导出缺少样本类型和沪苏范围核验字段")
        region = frame[["is_shanghai", "is_suzhou"]].apply(pd.to_numeric, errors="coerce")
        if not region.isin([0, 1]).all().all() or not region.eq(1).any(axis=1).all():
            raise ValueError("精简 U1 导出含非沪苏企业或无效地区标记")
    return {"sample_type_check": "explicit_U1" if "sample_type" in frame else "inferred_from_export_and_region_flags",
            "welfare_line_check": "explicit_all_1" if "has_welfare_raw" in frame else "not_independently_verifiable_from_compact_CSV"}


def run(run_id: str, new_u_path: Path, new_p_paths: list[Path],
        seeds: list[int], repeats: int, reference_date: pd.Timestamp,
        historical_positive: Path, historical_unlabeled: Path,
        historical_split: Path, b_run: Path) -> Path:
    if not new_p_paths:
        raise ValueError("必须提供至少一份新 P 信用代码导出，供实体冲突审计")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("随机种子不能为空或重复")
    paths = [new_u_path, *new_p_paths, historical_positive, historical_unlabeled, historical_split]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (b_run / "fixed_holdout_scores.csv").is_file():
        raise FileNotFoundError(b_run / "fixed_holdout_scores.csv")
    for path in (new_u_path, *new_p_paths):
        diagnostic = (inspect_workbench_json(path) if path.suffix.lower() == ".json"
                      else scan_csv(path))
        if path.suffix.lower() != ".json" and not diagnostic["valid"]:
            raise ValueError(f"CSV 导出结构无效：{path}；诊断={json.dumps(diagnostic, ensure_ascii=False)}")

    old_p = _read(historical_positive)
    old_u = _read(historical_unlabeled)
    _check_entities(old_p, "历史 P")
    _check_entities(old_u, "历史 U")
    new_raw = (read_workbench_json(new_u_path) if new_u_path.suffix.lower() == ".json"
               else _read(new_u_path))
    export_checks = _validate_new_u1(new_raw)
    new_raw = new_raw.copy()
    new_raw["sample_type"] = "U"  # V1 cleaner's binary sample marker; raw export stays untouched.
    exclusion_codes = _new_p_codes(new_p_paths)
    if not exclusion_codes:
        raise ValueError("新 P 导出没有可用信用代码")

    output = PROJECT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    prepared = clean_and_prepare(old_p, new_raw,
                                 report_dir=output / "quality",
                                 processed_dir=output / "processed")
    if set(_codes(prepared.positive)) != set(_codes(old_p)):
        raise ValueError("重新清洗后历史 P 实体集合改变；停止 C 以保持 P 固定")
    u_before_new_p = prepared.unlabeled.copy()
    overlap_mask = _codes(u_before_new_p).isin(exclusion_codes)
    overlap_count = int(overlap_mask.sum())
    u_before_new_p.loc[overlap_mask, ["credit_code_normalized"]].assign(
        exclusion_reason="credit_code_in_new_P_exclusion_only"
    ).to_csv(output / "quality/new_P_U_conflicts.csv", index=False, encoding="utf-8-sig")
    new_u = u_before_new_p.loc[~overlap_mask].reset_index(drop=True)
    _check_entities(new_u, "新 U1")
    if set(_codes(old_p)) & set(_codes(new_u)):
        raise ValueError("C 仍有历史 P/U 实体冲突")

    p = build_features(old_p, reference_date=reference_date).reset_index(drop=True)
    u = build_features(new_u, reference_date=reference_date).reset_index(drop=True)
    old_u = build_features(old_u, reference_date=reference_date).reset_index(drop=True)
    roles = _read_archived_split(historical_split, p, old_u)
    train, holdout = roles["P_train"], roles["P_holdout"]
    if len(u) < 5:
        raise ValueError("新 U1 清洗后不足 5 家")
    common_codes = set(_codes(roles["U_unseen"])) & set(_codes(u))
    common_u = u.loc[_codes(u).isin(common_codes)].reset_index(drop=True)
    if len(common_u) < 100:
        raise ValueError("与 B 共用的冻结 U 评价集合不足 100 家；需检查新 U1 导出口径")
    # Fixed B/C comparison entities must never enter the sampled-U LR fit pool.
    pool_candidates = u.loc[~_codes(u).isin(common_codes)]
    pool_size = max(1, int(len(u) * .4))
    if len(pool_candidates) < pool_size:
        raise ValueError("新 U1 中非冻结评价实体不足 40% 训练池；无法保持 B/C 评价隔离")
    u_pool = stratified_subset(pool_candidates, pool_size, RANDOM_STATE)

    manifest = {"experiment": "C", "run_id": run_id,
                "snapshot_claim": "2026-10-20 noon; not a strict historical DB snapshot",
                "runner_sha256": _sha256(Path(__file__)),
                "reference_date": reference_date.strftime("%Y-%m-%d"),
                "seeds": seeds, "cv_folds": 5, "lr_repeats": repeats,
                "new_P_role": "exclusion_only_never_training",
                "new_U1_export_checks": export_checks,
                "files": {str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size,
                                      "modified_time": path.stat().st_mtime} for path in paths},
                "counts": {"historical_P": len(p), "historical_U": len(old_u),
                           "new_U_raw": len(new_raw),
                           "new_U_after_V1_cleaning_before_new_P": len(u_before_new_p),
                           "new_P_overlap_excluded": overlap_count, "new_U_final": len(u),
                           "common_frozen_U_evaluation": len(common_u),
                           "C_U_fit_pool": len(u_pool),
                           "P_train": len(train), "P_holdout": len(holdout)},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _quality_summary(old_u, u).to_csv(output / "quality/incremental_eda.csv",
                                      index=False, encoding="utf-8-sig")

    # The frozen evaluation set is P_holdout plus U entities present in both B and C.
    targets = pd.concat([common_u, holdout], ignore_index=True)
    scores = _scores(train, u_pool, targets, repeats)
    c_u = scores.iloc[:len(common_u)].reset_index(drop=True)
    c_p = scores.iloc[len(common_u):].reset_index(drop=True)
    c_eval = pd.concat([
        pd.concat([common_u[["credit_code_normalized"]], holdout[["credit_code_normalized"]]],
                  ignore_index=True).assign(role=["U_common"] * len(common_u) + ["P_holdout"] * len(holdout)),
        scores.reset_index(drop=True)], axis=1)
    c_eval.to_csv(output / "fixed_common_evaluation_scores.csv", index=False, encoding="utf-8-sig")
    b_eval = _read(b_run / "fixed_holdout_scores.csv")
    b_u = b_eval.loc[(b_eval.role == "U_unseen") &
                     (b_eval.credit_code_normalized.astype("string").isin(common_codes))]
    b_p = b_eval.loc[b_eval.role == "P_holdout"]
    if len(b_u) != len(common_u) or len(b_p) != len(holdout):
        raise ValueError("B/C 冻结评价集合不一致")
    metrics = _recall_rows(b_p, b_u, "B_common_U") + _recall_rows(c_p, c_u, "C_common_U")

    cv_rows = []
    for fold, (fit_index, eval_index) in enumerate(
            KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(train), 1):
        p_fit, p_eval = train.iloc[fit_index], train.iloc[eval_index]
        fold_targets = pd.concat([common_u, p_eval], ignore_index=True)
        fold_scores = _scores(p_fit, u_pool, fold_targets, repeats)
        metrics += _recall_rows(fold_scores.iloc[len(common_u):].reset_index(drop=True),
                               fold_scores.iloc[:len(common_u)].reset_index(drop=True),
                               "C_P_train_5fold_CV", fold)
        cv_rows += [{"fold": fold, "credit_code_normalized": code, **values}
                    for code, values in zip(_codes(p_eval),
                                            fold_scores.iloc[len(common_u):].to_dict("records"))]
    pd.DataFrame(metrics).to_csv(output / "known_P_ranking_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_rows).to_csv(output / "P_train_cv_oof_scores.csv", index=False, encoding="utf-8-sig")

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
        for fold, (fit_index, score_index) in enumerate(
                KFold(n_splits=5, shuffle=True, random_state=seed).split(u), 1):
            fit_u = u.iloc[fit_index]
            pool = stratified_subset(fit_u, min(len(u_pool), len(fit_u)), seed + fold)
            if set(_codes(pool)) & set(_codes(u.iloc[score_index])):
                raise RuntimeError("U 折外评分训练池包含本折评分实体")
            fold_scaled = make_preprocessor(infer_feature_spec(train, pool))
            fold_values, _ = repeated_sampled_u_scores(
                fold_scaled, train, pool, score_targets=u.iloc[score_index], repeats=repeats)
            values[score_index] = fold_values
            fold_rows.append({"seed": seed, "fold": fold, "count": len(score_index),
                              "mean": float(np.mean(fold_values)),
                              "std": float(np.std(fold_values))})
        if not np.isfinite(values).all():
            raise RuntimeError("新 U1 的 OOF 评分未覆盖全体企业")
        seed_scores[seed] = values
    pd.DataFrame(fold_rows).to_csv(output / "U_lr_fold_scale.csv", index=False, encoding="utf-8-sig")

    scored = u[["credit_code_normalized", "company_name"]].copy()
    for column in SCORE_COLUMNS:
        scored[column] = seed_scores[seeds[0]] if column == "lr_mean_score" else p_only[column].to_numpy()
        scored[column.replace("score", "rank")] = scored[column].rank(ascending=False, method="min").astype(int)
    scored["score_population"] = "new_U1_all_LR_out_of_fold"
    scored.sort_values("lr_mean_rank").to_csv(output / "all_new_U1_oof_scores.csv",
                                               index=False, encoding="utf-8-sig")
    scored.nsmallest(100, "lr_mean_rank").to_csv(output / "top100_candidates.csv",
                                                 index=False, encoding="utf-8-sig")
    stability = []
    for seed in seeds[1:]:
        baseline, other = seed_scores[seeds[0]], seed_scores[seed]
        for k in (100, 500, 1000):
            x, y = set(np.argsort(-baseline)[:k]), set(np.argsort(-other)[:k])
            stability.append({"seed_a": seeds[0], "seed_b": seed, "top_k": k,
                              "jaccard": len(x & y) / len(x | y),
                              "spearman": float(pd.Series(baseline).corr(
                                  pd.Series(other), method="spearman"))})
    pd.DataFrame(stability).to_csv(output / "U_lr_seed_stability.csv",
                                   index=False, encoding="utf-8-sig")

    merged = b_u[["credit_code_normalized"] + SCORE_COLUMNS].merge(
        pd.concat([common_u[["credit_code_normalized"]], c_u], axis=1),
        on="credit_code_normalized", suffixes=("_B", "_C"), validate="one_to_one")
    comparisons = []
    for column in SCORE_COLUMNS:
        row = {"model": column, "common_U_count": len(merged),
               "spearman_B_vs_C": float(merged[f"{column}_B"].corr(
                   merged[f"{column}_C"], method="spearman"))}
        k = min(100, len(merged))
        top_b = set(merged.nlargest(k, f"{column}_B").credit_code_normalized)
        top_c = set(merged.nlargest(k, f"{column}_C").credit_code_normalized)
        row["top100_jaccard"] = len(top_b & top_c) / len(top_b | top_c)
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output / "B_C_common_U_comparison.csv",
                                      index=False, encoding="utf-8-sig")
    report = ("# V1.1 C 新 U1 实验\n\n"
              f"历史 P 固定 {len(p):,} 家；新 U1 原始 {len(new_raw):,} 条，"
              f"V1 清洗后 {len(u_before_new_p):,} 家，再排除与新 P 重叠 {overlap_count:,} 家，"
              f"最终评分 {len(u):,} 家。\n\n"
              f"B/C 共用 U 评价集合 {len(common_u):,} 家，固定 P holdout {len(holdout):,} 家。"
              "详见 CSV 指标、增量 EDA、清洗漏斗和 manifest。U 不是真负例；分数不是概率。\n")
    (output / "REPORT_zh.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--new-u1", type=Path, required=True)
    parser.add_argument("--positive-exclusion", type=Path, action="append", required=True)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--repeats", type=int, default=LR_REPEATS)
    parser.add_argument("--reference-date", default="2026-10-08")
    parser.add_argument("--historical-positive", type=Path,
                        default=PROJECT_ROOT / "data/processed/positive_modeling.csv")
    parser.add_argument("--historical-unlabeled", type=Path,
                        default=PROJECT_ROOT / "data/processed/unlabeled_modeling.csv")
    parser.add_argument("--historical-split", type=Path,
                        default=PROJECT_ROOT / "reports/entity_split.csv")
    parser.add_argument("--b-run", type=Path,
                        default=PROJECT_ROOT / "runs/20261019_B_archived_v1_1")
    args = parser.parse_args()
    result = run(args.run_id, args.new_u1, args.positive_exclusion,
                 [int(value) for value in args.seeds.split(",")], args.repeats,
                 pd.Timestamp(args.reference_date), args.historical_positive,
                 args.historical_unlabeled, args.historical_split, args.b_run)
    print(result)


if __name__ == "__main__":
    main()
