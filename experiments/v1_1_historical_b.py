"""V1.1 B: frozen V1 entities with P cross-validation and all-U OOF scores.

This runner reads the archived cleaned snapshots. It does not re-run raw-data
cleaning and therefore cannot replace the raw-input audit or experiment C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold

from src.config import PROJECT_ROOT, RANDOM_STATE, LR_REPEATS
from src.evaluation import SCORE_COLUMNS
from src.feature_engineering import build_features
from src.model_logistic_baseline import repeated_sampled_u_scores, stratified_subset
from src.model_oneclass import oneclass_scores
from src.model_similarity import positive_similarity
from src.preprocessing import infer_feature_spec, make_preprocessor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                       dtype={"credit_code": "string", "credit_code_normalized": "string"})


def _codes(frame: pd.DataFrame) -> pd.Series:
    return frame["credit_code_normalized"].astype("string").str.strip().str.upper()


def _check_entities(frame: pd.DataFrame, sample: str) -> None:
    code = _codes(frame)
    if code.isna().any() or code.duplicated().any():
        raise ValueError(f"{sample}: 信用代码非空且唯一的前提不成立")


def _scores(p_train: pd.DataFrame, u_pool: pd.DataFrame,
            targets: pd.DataFrame, repeats: int) -> pd.DataFrame:
    spec = infer_feature_spec(p_train, u_pool)
    if not spec.all:
        raise ValueError("没有通过训练折特征准入的字段")
    scaled, unscaled = make_preprocessor(spec), make_preprocessor(spec, False)
    out = pd.DataFrame(index=targets.index)
    out["similarity_score"], _ = positive_similarity(scaled, p_train, targets)
    out["lr_mean_score"], out["lr_score_std"] = repeated_sampled_u_scores(
        scaled, p_train, u_pool, score_targets=targets, repeats=repeats)
    out["oneclass_svm_score"], out["isolation_forest_score"] = oneclass_scores(
        scaled, unscaled, p_train, targets)
    return out


def _recall_rows(scores_p: pd.DataFrame, scores_u: pd.DataFrame,
                 label: str, fold: int | None = None) -> list[dict]:
    """Known-P capture within a frozen P+U ranking, never U-as-negative AUC."""
    rows = []
    for column in SCORE_COLUMNS:
        merged = pd.concat([scores_p[column], scores_u[column]], ignore_index=True)
        ranks = merged.rank(method="average", pct=True).iloc[:len(scores_p)]
        for fraction in (0.01, 0.05, 0.10, 0.20):
            rows.append({"evaluation": label, "fold": fold, "model": column,
                         "metric": f"known_P_share_in_top_{int(fraction * 100)}pct",
                         "value": float((ranks >= 1 - fraction).mean()),
                         "P_count": len(scores_p), "U_count": len(scores_u)})
    return rows


def _read_archived_split(path: Path, p: pd.DataFrame, u: pd.DataFrame) -> dict[str, pd.DataFrame]:
    manifest = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if set(manifest.columns) != {"credit_code_normalized", "role"}:
        raise ValueError("历史实体划分文件的 schema 不符合预期")
    if manifest.credit_code_normalized.isna().any() or manifest.credit_code_normalized.duplicated().any():
        raise ValueError("历史实体划分文件包含空键或重复实体")
    roles = {}
    for role, frame in (("P_train", p), ("P_holdout", p),
                        ("U_fit_pool", u), ("U_unseen", u)):
        selected = set(manifest.loc[manifest.role == role, "credit_code_normalized"])
        roles[role] = frame.loc[_codes(frame).isin(selected)].reset_index(drop=True)
        if len(roles[role]) != len(selected):
            raise ValueError(f"{role}: 历史实体划分与数据快照不一致")
    if len(roles["P_train"]) + len(roles["P_holdout"]) != len(p):
        raise ValueError("P 历史划分未覆盖全体 P")
    if len(roles["U_fit_pool"]) + len(roles["U_unseen"]) != len(u):
        raise ValueError("U 历史划分未覆盖全体 U")
    return roles


def _fold_summary(values: np.ndarray, fold: int, seed: int, count_train: int) -> dict:
    return {"seed": seed, "fold": fold, "score_count": len(values),
            "U_fit_candidate_count": count_train,
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "p05": float(np.quantile(values, .05)),
            "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95))}


def run(run_id: str, seeds: list[int], repeats: int,
        reference_date: pd.Timestamp,
        positive_path: Path, unlabeled_path: Path,
        split_path: Path, archived_scores_path: Path) -> Path:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("至少提供一个不重复的随机种子")
    paths = (positive_path, unlabeled_path, split_path, archived_scores_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    p = _read(positive_path)
    u = _read(unlabeled_path)
    _check_entities(p, "P")
    _check_entities(u, "U")
    if set(_codes(p)) & set(_codes(u)):
        raise ValueError("P/U 历史清洗快照存在实体重叠")
    if set(p.sample_type.dropna()) != {"P"} or set(u.sample_type.dropna()) != {"U"}:
        raise ValueError("历史清洗快照的 P/U 标识不符合预期")
    if not p.observed_positive.eq(1).all() or u.observed_positive.notna().any():
        raise ValueError("历史清洗快照的 P/U 标签不符合预期")

    # build_features uses only V1's explicit allow-list. Its age is frozen here.
    p = build_features(p, reference_date=reference_date).reset_index(drop=True)
    u = build_features(u, reference_date=reference_date).reset_index(drop=True)
    roles = _read_archived_split(split_path, p, u)
    train, holdout = roles["P_train"], roles["P_holdout"]
    old_pool, frozen_u = roles["U_fit_pool"], roles["U_unseen"]

    output = PROJECT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {"experiment": "B", "run_id": run_id,
                "reference_date": reference_date.strftime("%Y-%m-%d"),
                "seeds": seeds, "cv_folds": 5, "lr_repeats": repeats,
                "baseline_git_commit": "407684f", "raw_inputs_available": False,
                "files": {str(path): {"sha256": _sha256(path),
                                      "bytes": path.stat().st_size,
                                      "modified_time": path.stat().st_mtime} for path in paths},
                "counts": {"P": len(p), "U": len(u),
                           **{role: len(frame) for role, frame in roles.items()}},
                "versions": {"python": platform.python_version(),
                             "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__}}
    (output / "manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Fixed historical holdout. Same P and untouched-U entities as archived A.
    fixed_targets = pd.concat([frozen_u, holdout], ignore_index=True)
    fixed_scores = _scores(train, old_pool, fixed_targets, repeats)
    fixed_u = fixed_scores.iloc[:len(frozen_u)].reset_index(drop=True)
    fixed_p = fixed_scores.iloc[len(frozen_u):].reset_index(drop=True)
    fixed_output = pd.concat([
        pd.concat([frozen_u[["credit_code_normalized"]], holdout[["credit_code_normalized"]]],
                  ignore_index=True).assign(role=["U_unseen"] * len(frozen_u) + ["P_holdout"] * len(holdout)),
        fixed_scores.reset_index(drop=True)], axis=1)
    fixed_output.to_csv(output / "fixed_holdout_scores.csv", index=False, encoding="utf-8-sig")
    metric_rows = _recall_rows(fixed_p, fixed_u, "fixed_holdout")

    # Entity-level CV is confined to historical P_train, preserving P_holdout.
    cv_rows = []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_index, eval_index) in enumerate(cv.split(train), 1):
        p_fit, p_eval = train.iloc[fit_index], train.iloc[eval_index]
        targets = pd.concat([frozen_u, p_eval], ignore_index=True)
        score = _scores(p_fit, old_pool, targets, repeats)
        fold_u = score.iloc[:len(frozen_u)].reset_index(drop=True)
        fold_p = score.iloc[len(frozen_u):].reset_index(drop=True)
        metric_rows += _recall_rows(fold_p, fold_u, "P_train_5fold_CV", fold)
        cv_rows += [{"fold": fold, "credit_code_normalized": code,
                     **values} for code, values in zip(_codes(p_eval), fold_p.to_dict("records"))]
    pd.DataFrame(cv_rows).to_csv(output / "P_train_cv_oof_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(output / "known_P_ranking_metrics.csv", index=False, encoding="utf-8-sig")

    # Each U gets one score from LR models fitted with that U's fold excluded.
    # P-only models fit the frozen P_train once and can score all U directly.
    spec = infer_feature_spec(train, old_pool)
    scaled, unscaled = make_preprocessor(spec), make_preprocessor(spec, False)
    p_only = pd.DataFrame(index=u.index)
    p_only["similarity_score"], _ = positive_similarity(scaled, train, u)
    p_only["oneclass_svm_score"], p_only["isolation_forest_score"] = oneclass_scores(
        scaled, unscaled, train, u)

    seed_scores = {}
    fold_rows = []
    for seed in seeds:
        out = np.full(len(u), np.nan)
        fold_id = np.full(len(u), -1, dtype=int)
        for fold, (fit_index, score_index) in enumerate(
                KFold(n_splits=5, shuffle=True, random_state=seed).split(u), 1):
            fit_u = u.iloc[fit_index]
            pool = stratified_subset(fit_u, min(len(old_pool), len(fit_u)), seed + fold)
            if set(_codes(pool)) & set(_codes(u.iloc[score_index])):
                raise RuntimeError("U 折外评分的训练池包含本折评分实体")
            fold_spec = infer_feature_spec(train, pool)
            fold_scaled = make_preprocessor(fold_spec)
            mean, _ = repeated_sampled_u_scores(
                fold_scaled, train, pool, score_targets=u.iloc[score_index], repeats=repeats)
            out[score_index] = mean
            fold_id[score_index] = fold
            fold_rows.append(_fold_summary(mean, fold, seed, len(pool)))
        if not np.isfinite(out).all() or (fold_id < 1).any():
            raise RuntimeError("U 的 OOF 覆盖不完整")
        seed_scores[seed] = out
        if seed == seeds[0]:
            p_only["lr_mean_score"] = out
            p_only["lr_oof_fold"] = fold_id
    pd.DataFrame(fold_rows).to_csv(output / "U_lr_fold_scale.csv", index=False, encoding="utf-8-sig")

    all_u = u[["credit_code_normalized", "company_name"]].copy()
    for column in SCORE_COLUMNS:
        all_u[column] = p_only[column].to_numpy()
        all_u[column.replace("score", "rank")] = all_u[column].rank(ascending=False, method="min").astype(int)
    all_u["lr_oof_fold"] = p_only["lr_oof_fold"].to_numpy()
    all_u["score_population"] = "all_U_LR_out_of_fold"
    all_u.sort_values("lr_mean_score", ascending=False).to_csv(
        output / "all_U1_oof_scores.csv", index=False, encoding="utf-8-sig")
    all_u.nsmallest(100, "lr_mean_rank").to_csv(
        output / "top100_candidates.csv", index=False, encoding="utf-8-sig")

    stability = []
    first = seed_scores[seeds[0]]
    for seed in seeds[1:]:
        other = seed_scores[seed]
        for k in (100, 500, 1000):
            top_a = set(np.argsort(-first)[:k])
            top_b = set(np.argsort(-other)[:k])
            stability.append({"seed_a": seeds[0], "seed_b": seed, "top_k": k,
                              "jaccard": len(top_a & top_b) / len(top_a | top_b),
                              "spearman": float(pd.Series(first).corr(pd.Series(other), method="spearman"))})
    pd.DataFrame(stability).to_csv(output / "U_lr_seed_stability.csv", index=False, encoding="utf-8-sig")

    archived = _read(archived_scores_path)
    if "credit_code" not in archived or not set(SCORE_COLUMNS).issubset(archived):
        raise ValueError("历史 A 评分文件缺少实体键或评分字段")
    archived["credit_code_normalized"] = archived.credit_code.astype("string").str.strip().str.upper()
    joined = archived[["credit_code_normalized"] + SCORE_COLUMNS].merge(
        pd.concat([frozen_u[["credit_code_normalized"]].reset_index(drop=True), fixed_u], axis=1),
        on="credit_code_normalized", how="inner", suffixes=("_A", "_B"), validate="one_to_one")
    if len(joined) != len(frozen_u):
        raise ValueError("A/B 没有匹配到完整、冻结的 U 评价集合")
    comparisons = [{"model": column, "frozen_U_count": len(joined),
                    "spearman_A_vs_B": float(joined[f"{column}_A"].corr(
                        joined[f"{column}_B"], method="spearman"))} for column in SCORE_COLUMNS]
    pd.DataFrame(comparisons).to_csv(output / "A_B_fixed_U_comparison.csv", index=False, encoding="utf-8-sig")
    report = ("# V1.1 B 历史数据验证报告\n\n"
              f"运行 ID：`{run_id}`；参考日期：`{reference_date.date()}`。\n\n"
              f"沿用历史清洗快照：P {len(p):,} 家、U {len(u):,} 家。"
              f"固定 P holdout {len(holdout):,} 家，P_train 做 5 折交叉验证。"
              f"全部 {len(u):,} 家 U 已获得 LR 的折外评分；其余三模型仅拟合 P_train 后评分。\n\n"
              "输入是 2026-10-08 的清洗后产物，原始旧 P/U 文件尚未找到；"
              "因此 B 不是原始数据重新清洗的复现。"
              "P 中 Gold/Silver 分层信息在该快照中不可证实，结果仅按已知 P 解释。"
              "U 不是负例，不计算 AUC、F1、Precision。\n\n"
              "明细 CSV 和企业名称只留在本地 `runs/`，不会进入 Git。\n")
    (output / "REPORT_zh.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--repeats", type=int, default=LR_REPEATS)
    parser.add_argument("--reference-date", default="2026-10-08")
    parser.add_argument("--positive", type=Path, default=PROJECT_ROOT / "data/processed/positive_modeling.csv")
    parser.add_argument("--unlabeled", type=Path, default=PROJECT_ROOT / "data/processed/unlabeled_modeling.csv")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "reports/entity_split.csv")
    parser.add_argument("--archived-scores", type=Path, default=PROJECT_ROOT / "data/output/unlabeled_scored.csv")
    args = parser.parse_args()
    output = run(args.run_id, [int(item) for item in args.seeds.split(",")], args.repeats,
                 pd.Timestamp(args.reference_date), args.positive, args.unlabeled,
                 args.split, args.archived_scores)
    print(output)


if __name__ == "__main__":
    main()
