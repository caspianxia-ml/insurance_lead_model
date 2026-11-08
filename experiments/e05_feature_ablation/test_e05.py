import unittest
import numpy as np
import pandas as pd
from e05 import address_region,parse_year,parse_intro,SPECS,field_audit

class E05Tests(unittest.TestCase):
    def test_address_parser_uses_same_conservative_rule(self):
        for address,expected in [('上海市浦东新区世纪大道1号',('上海','上海')),
          ('中国（上海）自由贸易试验区郭守敬路1号',('上海','上海')),
          ('江苏省苏州市工业园区1号',('江苏','苏州')),
          ('新疆维吾尔自治区乌鲁木齐市新市区',('新疆','乌鲁木齐'))]:
            self.assertEqual(address_region(address),expected)
        self.assertTrue(all(pd.isna(x) for x in address_region('NULL')))
        self.assertTrue(all(pd.isna(x) for x in address_region('<html>上海</html>')))
        self.assertTrue(all(pd.isna(x) for x in address_region('美国纽约')))

    def test_year_and_coverage_preserve_missing_and_zero(self):
        values=parse_year(pd.Series(['2023','2022.0','2027','NULL','2023年','0']))
        np.testing.assert_array_equal(values[:2],[2023,2022]);self.assertTrue(values[2:].isna().all())
        values=parse_intro(pd.Series(['0','1','NULL','unknown','2']))
        np.testing.assert_array_equal(values[:2],[0,1]);self.assertTrue(values[2:].isna().all())

    def test_single_factor_diagnostics_are_isolated(self):
        for high,low,extra in [('F1_size','F0',{'company_size_clean'}),
          ('F2_safe','F1_size',{'paid_capital_log1p'}),('F2_year','F2_safe',{'insured_report_year_numeric'}),
          ('F3','F2_safe',{'address_province','address_city'}),('F4','F2_safe',{'introduction_flag'})]:
            self.assertEqual(set(SPECS[high].all)-set(SPECS[low].all),extra)
            self.assertFalse(set(SPECS[low].all)-set(SPECS[high].all))
        forbidden={'source','quality_effective','p_quality','credit_code_normalized','company_name','region_province','region_city','has_score_details'}
        for spec in SPECS.values():self.assertFalse(forbidden&set(spec.all))

    def test_large_gap_is_exposed_as_risk_not_silently_passed(self):
        fields=set(c for spec in SPECS.values() for c in spec.all)
        p=pd.DataFrame({c:[1.]*10 for c in fields});u=p.copy()
        p['industry_medium_category_clean']=[1.]+[np.nan]*9
        audit=field_audit(p,p,u,1)
        rows=audit[audit.feature.eq('industry_medium_category_clean')]
        self.assertFalse(rows.legacy_gap_guard_pass.any())
        self.assertTrue(rows.risk.str.contains('over_20pp').all())

if __name__=='__main__':unittest.main()
