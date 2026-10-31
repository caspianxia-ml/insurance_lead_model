"""Entity-separated PU ranking; only untouched U receives candidate scores."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import (HOLDOUT_SIZE, ID_COLUMNS, LR_REPEATS, OUTPUT_DIR,
                     RANDOM_STATE, REPORTS_DIR, U_FIT_FRACTION)
from .evaluation import SCORE_COLUMNS, write_evaluation
from .feature_engineering import build_features
from .model_logistic_baseline import repeated_sampled_u_scores, stratified_subset
from .model_oneclass import oneclass_scores
from .model_similarity import positive_similarity
from .preprocessing import infer_feature_spec, make_preprocessor, feature_eligibility


def _score_models(p, targets, pool, scaled, unscaled, repeats):
    out = pd.DataFrame(index=targets.index)
    out['similarity_score'], _ = positive_similarity(scaled, p, targets)
    out['lr_mean_score'], out['lr_score_std'] = repeated_sampled_u_scores(
        scaled, p, pool, score_targets=targets, repeats=repeats)
    out['oneclass_svm_score'], out['isolation_forest_score'] = oneclass_scores(scaled, unscaled, p, targets)
    return out


def _priority(top10, top20):
    n10, n20 = top10.sum(axis=1), top20.sum(axis=1)
    return pd.Series(np.select([n10 >= 2, (n10 >= 1) | (n20 >= 2), n20 == 0],
                               ['A', 'B', 'D'], default='C'), index=top10.index)


def score_unlabeled(positive, unlabeled, output_path=OUTPUT_DIR / 'unlabeled_scored.csv',
                    repeats=LR_REPEATS, evaluation_path=REPORTS_DIR / 'baseline_evaluation.csv',
                    sensitivity=False):
    if len(positive) < 5 or len(unlabeled) < 4:
        raise ValueError('建模至少需要 5 条 P 和 4 条 U。')
    p, u = build_features(positive).reset_index(drop=True), build_features(unlabeled).reset_index(drop=True)
    for frame in (p, u):
        codes = frame['credit_code_normalized']
        if codes.isna().any() or codes.duplicated().any():
            raise ValueError('请先清洗：信用代码必须非空且实体唯一。')
    if set(p.credit_code_normalized) & set(u.credit_code_normalized):
        raise ValueError('P/U 实体交集尚未清除。')
    train = stratified_subset(p, max(2, int(len(p) * (1-HOLDOUT_SIZE))), RANDOM_STATE)
    holdout = p.drop(train.index)
    pool = stratified_subset(u, max(1, int(len(u)*U_FIT_FRACTION)), RANDOM_STATE)
    candidates = u.drop(pool.index)
    # Eligibility selection also sees training rows only, never holdout/candidates.
    spec = infer_feature_spec(train, pool)
    if not spec.all:
        raise ValueError('没有通过共有字段及缺失率审查的特征。')
    report_dir = Path(evaluation_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_eligibility(train, pool).to_csv(report_dir/'feature_eligibility.csv', index=False)
    scaled, unscaled = make_preprocessor(spec, True), make_preprocessor(spec, False)
    targets = pd.concat([candidates, holdout], ignore_index=True)
    scores = _score_models(train, targets, pool, scaled, unscaled, repeats)
    us = scores.iloc[:len(candidates)].reset_index(drop=True)
    hs = scores.iloc[len(candidates):].reset_index(drop=True)
    write_evaluation(us, hs, evaluation_path)
    manifests = []
    for frame, role in ((train,'P_train'), (holdout,'P_holdout'), (pool,'U_fit_pool'), (candidates,'U_unseen')):
        manifests.append(frame[['credit_code_normalized']].assign(role=role))
    pd.concat(manifests).to_csv(report_dir/'entity_split.csv', index=False)
    selected = [c for c in ID_COLUMNS + ['source','target_region_corrected'] if c in candidates]
    output = candidates[selected].reset_index(drop=True).copy()
    output = pd.concat([output, us], axis=1)
    for score, rank in zip(SCORE_COLUMNS, ['similarity_rank','lr_rank','oneclass_rank','isolation_rank']):
        output[rank] = output[score].rank(method='min', ascending=False).astype(int)
    top10 = us[SCORE_COLUMNS].rank(pct=True) > .9
    top20 = us[SCORE_COLUMNS].rank(pct=True) > .8
    output['model_agreement_count'] = top10.sum(axis=1)
    output['priority_level'] = _priority(top10, top20)
    output['is_in_leads'] = pd.NA
    output['leads_membership_status'] = 'unknown_not_inferred_from_id'
    output['score_population'] = 'U_never_used_in_fit'
    if 'source' in output:
        output.groupby('source', dropna=False)[SCORE_COLUMNS].describe().to_csv(report_dir/'source_score_distribution.csv')
        output.assign(top10=top10.sum(axis=1)>0).groupby('source', dropna=False).agg(
            count=('top10','size'), any_model_top10_share=('top10','mean')).to_csv(report_dir/'source_topk.csv')
    if sensitivity and 'source' in train and 'source' in pool:
        rows = []
        for source in sorted(set(train.source.dropna()) | set(pool.source.dropna())):
            pt, ut = train.loc[train.source != source], pool.loc[pool.source != source]
            if len(pt)<5 or len(ut)<2:
                rows.append({'excluded_source':source, 'status':'insufficient_training_rows'})
                continue
            changed = _score_models(pt, targets, ut, scaled, unscaled, repeats).iloc[:len(candidates)]
            for c in SCORE_COLUMNS:
                k = max(1, int(np.ceil(.1*len(candidates))))
                base_top, new_top = set(us[c].nlargest(k).index), set(changed[c].nlargest(k).index)
                rows.append({'excluded_source':source,'model':c,'status':'ok',
                             'rank_correlation':us[c].rank().corr(changed[c].rank()),
                             'top10_jaccard':len(base_top & new_top)/len(base_top | new_top),
                             'P_train_count':len(pt),'U_pool_count':len(ut)})
        pd.DataFrame(rows).to_csv(report_dir/'source_sensitivity.csv', index=False)
    output = output.sort_values(['priority_level','lr_rank'])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.getLogger(__name__).info('仅对未参与 fit 的 %d 条 U 评分；%d 条 U fit pool 留待 cross-fitting。',len(output),len(pool))
    return output
