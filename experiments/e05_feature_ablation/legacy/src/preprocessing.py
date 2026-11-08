"""Build unfitted sklearn preprocessors; callers fit on training data only."""
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def feature_eligibility(train_frame: pd.DataFrame, score_frame: pd.DataFrame | None = None,
                        max_missingness_gap: float = 0.20) -> pd.DataFrame:
    """Audit the allow-listed fields before any preprocessor is fitted.

    A shared column with a large P/U completeness gap is not used: that gap can
    encode a different collection pipeline rather than a company attribute.
    """
    rows: list[dict] = []
    for feature_type, columns in (("numeric", NUMERIC_FEATURES), ("categorical", CATEGORICAL_FEATURES)):
        for column in columns:
            in_train = column in train_frame
            in_score = score_frame is None or column in score_frame
            train_non_null = float(train_frame[column].notna().mean()) if in_train else 0.0
            score_non_null = (float(score_frame[column].notna().mean())
                              if score_frame is not None and in_score else train_non_null)
            gap = abs(train_non_null - score_non_null)
            eligible = in_train and in_score and train_non_null > 0 and score_non_null > 0 and gap <= max_missingness_gap
            reason = "eligible" if eligible else (
                "missing_column" if not (in_train and in_score) else
                "all_missing_in_P_train_or_U" if min(train_non_null, score_non_null) == 0 else
                "P_U_missingness_gap"
            )
            rows.append({"feature": column, "type": feature_type, "eligible": eligible,
                         "P_train_non_null_ratio": train_non_null,
                         "U_non_null_ratio": score_non_null,
                         "missingness_gap": gap, "reason": reason})
    return pd.DataFrame(rows)


def infer_feature_spec(frame: pd.DataFrame, score_frame: pd.DataFrame | None = None) -> FeatureSpec:
    audit = feature_eligibility(frame, score_frame)
    return FeatureSpec(
        numeric=audit.loc[(audit["type"] == "numeric") & audit["eligible"], "feature"].tolist(),
        categorical=audit.loc[(audit["type"] == "categorical") & audit["eligible"], "feature"].tolist(),
    )


def make_preprocessor(spec: FeatureSpec, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), spec.numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), spec.categorical),
        ],
        remainder="drop",
    )
