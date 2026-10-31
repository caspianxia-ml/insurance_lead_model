"""Repeated sampled-U logistic ranking baseline (not calibrated probability)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TEMP_NEGATIVE_RATIO


STRATIFY_COLUMNS = ("industry_category_clean", "company_size_clean")


def _strata(frame: pd.DataFrame) -> pd.Series:
    available = [column for column in STRATIFY_COLUMNS if column in frame]
    if not available:
        return pd.Series("__all__", index=frame.index)
    values = frame[available].fillna("__missing__").astype(str)
    return values.agg(" | ".join, axis=1)


def stratified_subset(frame: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Proportionally sample a U subset using permitted profile fields only."""
    if n >= len(frame):
        return frame.copy()
    if n <= 0:
        return frame.iloc[0:0].copy()
    groups = _strata(frame)
    counts = groups.value_counts(sort=False)
    ideal = counts * n / len(frame)
    allocation = ideal.astype(int).clip(upper=counts)
    remaining = n - int(allocation.sum())
    fractions = (ideal - allocation).sort_values(ascending=False)
    while remaining:
        changed = False
        for stratum in fractions.index:
            if allocation[stratum] < counts[stratum]:
                allocation[stratum] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:  # defensive: cannot happen when n <= len(frame)
            break
    parts: list[pd.DataFrame] = []
    for offset, (stratum, take) in enumerate(allocation.items()):
        if take:
            parts.append(frame.loc[groups == stratum].sample(n=int(take), replace=False,
                                                              random_state=random_state + offset))
    return pd.concat(parts).sort_index(kind="stable")


def repeated_sampled_u_scores(preprocessor, p_train: pd.DataFrame, unlabeled_pool: pd.DataFrame,
                              score_targets: pd.DataFrame | None = None,
                              repeats: int = 10) -> tuple[np.ndarray, np.ndarray]:
    all_scores: list[np.ndarray] = []
    targets = unlabeled_pool if score_targets is None else score_targets
    sample_size = min(len(unlabeled_pool), max(1, int(len(p_train) * TEMP_NEGATIVE_RATIO)))
    for offset in range(repeats):
        seed = RANDOM_STATE + offset
        sampled_u = stratified_subset(unlabeled_pool, n=sample_size, random_state=seed)
        train = pd.concat([p_train, sampled_u], ignore_index=True)
        labels = np.r_[np.ones(len(p_train)), np.zeros(len(sampled_u))]
        model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ])
        model.fit(train, labels)
        all_scores.append(model.predict_proba(targets)[:, 1])
    matrix = np.vstack(all_scores)
    return matrix.mean(axis=0), matrix.std(axis=0)
