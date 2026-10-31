"""Feature creation from the cleaned, allowed company-profile fields only."""
import logging
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .data_cleaning import clean_data

LOGGER = logging.getLogger(__name__)


def build_features(frame: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return cleaned fields plus a deliberately small, leakage-safe X allow-list.

    ``source`` is retained in the returned audit frame but is never included in
    the feature specification. The same applies to geography, IDs, labels and
    all collection-history fields.
    """
    result = clean_data(frame)
    reference = reference_date or pd.Timestamp.now().normalize()
    if "establishment_date" in result:
        age = (reference - result["establishment_date"]).dt.days / 365.25
        result["company_age"] = age.where(age >= 0)
    allowed = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [name for name in allowed if name not in result]
    if missing:
        LOGGER.warning("缺少允许的基础画像字段：%s", missing)
    return result
