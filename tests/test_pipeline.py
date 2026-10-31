import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path

from src.feature_engineering import build_features
from src.leakage_check import check_leakage
from src.score_unlabeled import score_unlabeled
from src.data_cleaning import clean_and_prepare, _parse_date_dayfirst, _parse_capital, _registered_region
from src.preprocessing import infer_feature_spec


def _sample(n: int, positive: bool) -> pd.DataFrame:
    rng = np.random.default_rng(4 if positive else 9)
    return pd.DataFrame({
        "leads_company_id": np.arange(n) + (0 if positive else 10_000),
        "company_name": [f"公司{i}" for i in range(n)],
        "credit_code": [f"CODE{i + (0 if positive else 10_000)}" for i in range(n)],
        "company_type": rng.choice(["有限责任", "股份"], n),
        "employee_scale": rng.choice(["20-99", "100-499"], n),
        "industry_category": rng.choice(["科技", "制造", "服务"], n),
        "insured_count": rng.integers(2, 200, n),
        "registered_capital": rng.integers(10, 1000, n).astype(str) + np.repeat("万", n),
        "establishment_date": pd.date_range("2010-01-01", periods=n, freq="30D").astype(str),
        "raw_record_count": rng.integers(1, 30, n),
        "raw_source_count": rng.integers(1, 4, n),
        "first_raw_time": "2023-01-01",
        "latest_raw_time": "2024-01-01",
        "key_word": "六险一金" if positive else np.nan,
    })


class PipelineTest(unittest.TestCase):
    def test_cleaning_conflicts_dates_and_feature_exclusions(self):
        self.assertEqual(_parse_date_dayfirst(pd.Series(['4/8/2026'])).iloc[0], pd.Timestamp('2026-08-04'))
        self.assertEqual(_parse_capital('2万人民币')[0], 20000)
        self.assertTrue(np.isnan(_parse_capital('2万美元')[0]))
        self.assertEqual(_registered_region(pd.Series({'registered_address':'苏州市工业园区','province':'上海市'}))[0], '苏州')
        p, u = _sample(10, True), _sample(12, False)
        p['sample_type'], p['observed_positive'] = 'P', 1
        u['sample_type'], u['observed_positive'] = 'U', np.nan
        u.loc[0,'credit_code'] = ' code0 '
        u.loc[1,'credit_code'] = np.nan
        u.loc[2,'business_status'] = '注销'
        with tempfile.TemporaryDirectory() as d:
            result = clean_and_prepare(p,u,Path(d)/'reports',Path(d)/'processed')
        self.assertEqual(len(result.unlabeled),9)
        self.assertFalse(set(result.positive.credit_code_normalized) & set(result.unlabeled.credit_code_normalized))
        features = infer_feature_spec(build_features(p)).all
        self.assertFalse(any(c in features for c in ['source','raw_record_count','key_word','company_name','province']))

    def test_feature_engineering_is_repeatable(self):
        once = build_features(_sample(10, True), reference_date=pd.Timestamp("2025-01-01"))
        twice = build_features(once, reference_date=pd.Timestamp("2025-01-01"))
        pd.testing.assert_series_equal(once["company_age"], twice["company_age"])

    def test_leakage_and_scoring(self):
        p, u = _sample(30, True), _sample(40, False)
        # A U-only field must be audited and ignored by modeling.
        u["u_only_field"] = "audit-only"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = check_leakage(p, u, root / "leakage.csv")
            self.assertTrue(((report.field == "key_word") &
                             (report.risk == "explicit_label_source")).any())
            scored = score_unlabeled(p, u, root / "scored.csv", repeats=2,
                                      evaluation_path=root / "evaluation.csv", sensitivity=True)
            splits = pd.read_csv(root / 'entity_split.csv')
            fit = set(splits.loc[splits.role == 'U_fit_pool','credit_code_normalized'])
            self.assertFalse(fit & set(scored.credit_code))
            self.assertEqual(splits.credit_code_normalized.nunique(), len(p)+len(u))
        self.assertEqual(len(scored), len(u)-int(.4*len(u)))
        self.assertTrue({"lr_mean_score", "model_agreement_count", "priority_level"}.issubset(scored))
        self.assertTrue(scored["model_agreement_count"].between(0, 4).all())


if __name__ == "__main__":
    unittest.main()
