"""Positive-centroid similarity model."""
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import normalize


def positive_similarity(preprocessor, p_train, targets) -> tuple[np.ndarray, object]:
    fitted = clone(preprocessor).fit(p_train)
    p_matrix = fitted.transform(p_train)
    target_matrix = fitted.transform(targets)
    centroid = np.asarray(p_matrix.mean(axis=0)).ravel().reshape(1, -1)
    centroid = normalize(centroid)
    scores = normalize(target_matrix).dot(centroid.T)
    return np.asarray(scores).ravel(), fitted
