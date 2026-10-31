"""One-class models fitted only on known positives."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .config import RANDOM_STATE


def oneclass_scores(scaled_preprocessor, unscaled_preprocessor, p_train, targets) -> tuple[np.ndarray, np.ndarray]:
    svm_prep = clone(scaled_preprocessor).fit(p_train)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(svm_prep.transform(p_train))
    svm_score = svm.decision_function(svm_prep.transform(targets)).ravel()

    iso_prep = clone(unscaled_preprocessor).fit(p_train)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(iso_prep.transform(p_train))
    iso_score = iso.decision_function(iso_prep.transform(targets)).ravel()
    return np.asarray(svm_score), np.asarray(iso_score)
