"""Small synthetic smoke test (not actual enterprise data)."""
import tempfile
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_v2 import run, clean_code, clean_size


def test_smoke():
    a = '913100001234567890'
    b = '913100001234567891'
    c = '913100001234567892'
    d = '913100001234567893'
    assert clean_code(' '+a+' ') == (a,'format_18')
    assert clean_code('9.131E+17')[1] == 'suspected_excel_scientific_or_decimal'
    assert clean_code('91310000E234567890')[1] == 'format_18'
    assert clean_size('0_19') == '0-19'
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = {
            'P_national': [
                {'credit_code':a, 'company_name':'甲公司', 'company_size':'0_19', 'created_time':'2026-10-01', 'source':'原力'},
                {'credit_code':a, 'company_name':'甲公司', 'company_size':'-', 'created_time':'2026-10-02', 'source':'爬虫'},
                {'credit_code':b, 'company_name':'乙公司', 'company_size':'20-99', 'created_time':'2026-10-01', 'source':'平台'},
            ],
            'U1': [
                {'credit_code':a, 'company_name':'甲公司', 'registered_address':'上海市'},
                {'credit_code':c, 'company_name':'乙公司', 'registered_address':'苏州市'},
                {'credit_code':'', 'company_name':'甲公司', 'registered_address':'上海市'},
                {'credit_code':d, 'company_name':'丙公司', 'registered_address':'苏州市'},
                {'credit_code':d, 'company_name':'丙公司', 'registered_address':'苏州市'},
            ],
            'U2': [{'credit_code':d, 'company_name':'丙公司'}],
            'U3': [{'credit_code':d, 'company_name':'丙公司'}, {'credit_code':'9.131E+17', 'company_name':'戊公司'}],
        }
        paths={}
        for label, vals in rows.items():
            path = root / f'{label}.csv'; pd.DataFrame(vals).to_csv(path,index=False)
            paths[label] = path
        res = run(paths, root / 'out')
        stats = {x['dataset']:x for x in res['summary']}
        assert stats['P_national']['removed_duplicate_credit_code'] == 1
        assert stats['U1']['removed_duplicate_credit_code'] == 1
        assert stats['U1']['removed_known_P_credit_code'] == 1
        assert stats['U1']['kept_rows'] == 3
        assert stats['U1']['suspected_P_name_match_review'] == 2
        assert stats['U3']['credit_code_excel_risk'] == 1
        assert stats['U1']['reconciliation_ok']
        assert (root/'out'/'run_manifest.json').exists()
        assert len(pd.read_csv(root/'out'/'P_U_exact_overlap.csv')) == 1
        assert len(pd.read_csv(root/'out'/'P_U_name_review.csv')) == 2
        print('PASS: synthetic test — duplicate=1, U1 P conflict=1, remaining U1=3, 2 manual review')


if __name__ == '__main__':
    test_smoke()
