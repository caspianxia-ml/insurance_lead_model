import csv
from pathlib import Path
import tempfile
import unittest
import acceptance as a

COMMON='credit_code company_name registered_address region_province region_city is_shanghai is_suzhou business_status establishment_date registered_capital paid_capital insured_count insured_count_report_year industry_category industry_major_category industry_medium_category company_type company_size has_company_introduction'.split()

class AcceptanceTests(unittest.TestCase):
    def test_normalization_boundaries(self):
        self.assertEqual(a.clean_code(' ９１３１００００１２３４５６７８９０ '),('913100001234567890','format_18'))
        self.assertEqual(a.clean_code('9.13E+17')[1],'suspected_excel_scientific_or_decimal')
        self.assertEqual(a.clean_code("'913100001234567890")[1],'other_format')
        self.assertEqual(a.integer('0',0,10000000),('0','ok'))
        self.assertEqual(a.integer('NULL',0,10000000),('','missing'))
        self.assertEqual(a.integer('1.5',0,10000000)[1],'invalid_integer')
        self.assertEqual(a.integer('-1',0,10000000)[1],'out_of_range')
        self.assertEqual(a.capital('1.5亿人民币'),('150000000.0','CNY','ok'))
        self.assertEqual(a.capital('100万')[2],'currency_assumed_review')
        self.assertEqual(a.capital('100万美元')[2],'non_cny')
        self.assertEqual(a.date('2026-10-21')[1],'outside_declared_snapshot_range')
        self.assertEqual(a.date('01/02/2020')[1],'unparseable_or_ambiguous')
        self.assertEqual(a.date('2020-02-30')[1],'unparseable')

    def test_missing_name_overlap_and_duplicate_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); paths={}
            codes=['913100001234567890','913100001234567891','913100001234567892']
            examples={
              'P_national':[{'credit_code':codes[0],'company_name':'甲','p_quality':'Silver','source':'量化Q3榜单'},
                            {'credit_code':'','company_name':'乙','p_quality':'Gold','source':'原力'}],
              'U1':[{'credit_code':codes[0],'company_name':'甲'},
                    {'credit_code':codes[1],'company_name':'乙'},
                    {'credit_code':'','company_name':'甲'},
                    {'credit_code':codes[2],'company_name':'丙','company_size':'0_19'},
                    {'credit_code':codes[2],'company_name':'丙'}],
              'U2':[{'credit_code':codes[1],'company_name':'乙'}],
              'U3':[{'credit_code':'9.13E+17','company_name':'甲'}]}
            for label,vals in examples.items():
                path=root/f'{label}.csv'; paths[label]=path
                columns=(['leads_company_id','source','p_quality','created_time'] if label=='P_national' else ['mid_company_clean_id'])+COMMON
                with path.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
                    for row in vals:w.writerow(row)
            a.parse_info.clear();a.frames.clear();a.changes.clear();a.schema.clear()
            a.audit.read_csv=a.read_csv;a.audit.clean_code=a.clean_code;a.audit.prepare=a.prepare;a.audit.data_quality=a.raw_quality
            result=a.audit.run(paths,root/'out'); stats={r['dataset']:r for r in result['summary']}
            self.assertEqual(stats['U1']['removed_duplicate_credit_code'],1)
            self.assertEqual(stats['U1']['removed_known_P_credit_code'],1)
            self.assertEqual(stats['U1']['kept_rows'],3)
            self.assertEqual(stats['U1']['suspected_P_name_match_review'],2)
            self.assertEqual(stats['U3']['credit_code_excel_risk'],1)
            self.assertTrue(a.frames['P_national']['p_quality_candidate'].eq('Gold_candidate').all())
            self.assertEqual(a.frames['P_national']['p_quality'].tolist(),['Silver','Gold'])
            with (root/'out/cleaned/U1_cleaned.csv').open(encoding='utf-8-sig',newline='') as f:
                output=list(csv.DictReader(f,strict=True))
            self.assertIn('',[r['credit_code'] for r in output])
            self.assertEqual(next(r for r in output if r['credit_code']==codes[2])['company_size_normalized'],'0-19')

    def test_parser_refuses_wrong_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'example.csv';p.write_text('a,b\n1,2,3\n')
            with self.assertRaises(ValueError): a.read_csv(p)

if __name__=='__main__':unittest.main()
