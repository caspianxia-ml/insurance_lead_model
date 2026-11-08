"""Central project configuration for the leakage-safe PU ranking experiment."""
from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
POSITIVE_PATH = RAW_DIR / "positive.csv"
UNLABELED_PATH = RAW_DIR / "unlabeled.csv"

# These columns may appear in reports and the final business file, but never enter X.
ID_COLUMNS = ["leads_company_id", "mid_company_clean_id", "company_name", "credit_code"]
LEAKAGE_COLUMNS = {
    "source", "target_region", "region_match_source", "welfare", "welfare_by_credit",
    "welfare_by_raw_id", "key_word", "raw_keywords", "insurance_tag_name",
    "insurance_tag_algorithm", "insurance_tag_source", "insurance_tag_confidence",
    "tag_version", "tagged_at", "positive_rule", "observed_positive", "sample_type",
}

# Feature allow-list: only basic company-profile attributes that are available in
# both P and Mid Clean U.  Geography, source, tags, identifiers, free-text and
# all Raw/Mid/Job collection fields are intentionally absent.
NUMERIC_FEATURES = [
    "company_age", "registered_capital_log1p", "paid_capital_log1p",
    "insured_count_log1p",
]
CATEGORICAL_FEATURES = [
    "company_type_clean", "company_size_clean", "business_status_clean",
    "industry_major_category_clean",
    "industry_medium_category_clean",
]

HOLDOUT_SIZE = 0.20
LR_REPEATS = 10
TEMP_NEGATIVE_RATIO = 1.0
# U is not a negative class.  Only this deterministic, stratified subset may
# serve as temporary sampled-U rows for LR.  The full U remains the score target.
U_FIT_FRACTION = 0.40
MIN_STRATUM_SIZE = 8
TOP10_FRACTION = 0.10
TOP20_FRACTION = 0.20

# A: >=2 models Top10%; B: >=1 Top10% or >=2 models Top20%; D: all models
# outside Top20%; remaining records are C.
PRIORITY_RULES = {
    "A_MIN_TOP10_MODELS": 2,
    "B_MIN_TOP10_MODELS": 1,
    "B_MIN_TOP20_MODELS": 2,
}

DATE_COLUMNS = ["establishment_date", "approval_date"]

# Company records with these statuses are retained in raw data and listed in the
# audit, but excluded from the modelling population.
ABNORMAL_STATUS_TOKENS = ("注销", "吊销", "撤销", "清算", "终止", "关闭", "歇业", "迁出")
INSURED_COUNT_MAX = 10_000_000
