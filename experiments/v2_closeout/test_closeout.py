"""Targeted behavioral checks for the new policy and safe unified entry."""
import json, tempfile, unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_closeout import Taxonomy, order
from standardize_industry import resolve
from v2 import command, new_path

class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ap=__import__('argparse').ArgumentParser();ap.add_argument('--taxonomy',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True);a,unused=ap.parse_known_args()
        cls.tax=Taxonomy(json.loads(a.taxonomy.read_text()));cls.registry=json.loads(a.registry.read_text())
    def test_shared_row_and_names_keep_all_levels(self):
        self.assertEqual(set(self.tax.names['其他农业']),{'019','0190'})
        result,_=resolve(['房地产业','',''],self.tax);self.assertEqual(result['standard_section_code'],'K');self.assertEqual(result['standard_division_code'],'70')
    def test_conflict_uses_first_valid_fields(self):
        r,e=resolve(['制造业','汽车制造业','橡胶制品业'],self.tax)
        self.assertEqual(r['standard_section_code'],'C');self.assertEqual(r['standard_division_code'],'36');self.assertEqual(r['standard_group_code'],'');self.assertTrue(r['industry_conflict_resolved_first'])
        r,_=resolve(['其他资本市场服务','商务服务业',''],self.tax);self.assertEqual(r['standard_division_code'],'67')
    def test_unknown_retained_and_hierarchy(self):
        r,_=resolve(['电子/半导体/集成电路','',''],self.tax);self.assertEqual(r['industry_standard_status'],'unknown_retained')
        r,_=resolve(['','科学研究和技术服务业>科技推广和应用服务业>技术推广服务>其他技术推广服务',''],self.tax);self.assertEqual(r['standard_division_code'],'75');self.assertEqual(r['standard_class_code'],'7519')
    def test_later_conflicting_tokens_never_override_first(self):
        r,e=resolve(['制造业|批发和零售业','汽车制造业',''],self.tax);self.assertEqual(r['standard_division_code'],'36');self.assertEqual(e[1]['decision'],'conflict_ignored_later')
    def test_coarse_only_cannot_invent_fine_code(self):
        r,_=resolve(['制造业','',''],self.tax);self.assertEqual(r['standard_section_code'],'C');self.assertEqual(r['standard_division_code'],'')
    def test_deterministic_tie_order(self):self.assertEqual(order([.5,.7,.5],['B','C','A']).tolist(),[1,2,0])
    def test_output_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'exists').mkdir()
            with self.assertRaises(FileExistsError):new_path(p/'exists',p)
            for name in ['data','src','experiments','docs','.git']:
                with self.assertRaises(ValueError):new_path(p/name/'new',p)
    def test_registry_commands_use_formal_archives(self):
        p=Path('/project')
        for experiment in ['E02','E03','E04','E05','E06']:
            for action in ['prepare','train','report']:
                args=command(p,self.registry,action,experiment,p/'new')
                self.assertTrue(args[0].endswith('.py'));self.assertIn('--out',args)
        args=command(p,self.registry,'report','E06',p/'new')
        self.assertTrue(args[args.index('--run')+1].endswith('runs/main_002'))

if __name__=='__main__':unittest.main(argv=[__file__],verbosity=2)
