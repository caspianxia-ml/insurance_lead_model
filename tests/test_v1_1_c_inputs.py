import tempfile
import unittest
import json
from pathlib import Path
import pandas as pd

from experiments.v1_1_new_u1_c import _new_p_codes, _validate_new_u1
from experiments.v1_1_json_export import inspect_workbench_json, read_workbench_json


class NewPositiveExclusionTest(unittest.TestCase):
    def test_compact_u1_uses_region_flags_without_inventing_welfare_evidence(self):
        frame = pd.DataFrame({"is_shanghai": [1, 0], "is_suzhou": [0, 1]})
        checks = _validate_new_u1(frame)
        self.assertEqual(checks["welfare_line_check"],
                         "not_independently_verifiable_from_compact_CSV")
        with self.assertRaises(ValueError):
            _validate_new_u1(pd.DataFrame({"is_shanghai": [0], "is_suzhou": [0]}))

    def test_credit_codes_are_normalized_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_p.csv"
            path.write_text("credit_code\n abc 123 \nABC123\n\n", encoding="utf-8")
            self.assertEqual(_new_p_codes([path]), {"ABC123"})

    def test_workbench_json_with_literal_newline_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_u1.json"
            path.write_text('[{"credit_code":"ABC123","sample_type":"U1",'
                            '"has_welfare_raw":1,"company_introduction":"line 1\nline 2"}]',
                            encoding="utf-8")
            frame = read_workbench_json(path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.loc[0, "company_introduction"], "line 1\nline 2")
            self.assertEqual(inspect_workbench_json(path)["columns"], 4)
            with self.assertRaises(json.JSONDecodeError):
                json.loads(path.read_text(encoding="utf-8"))

    def test_new_p_json_codes_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_p.json"
            path.write_text('[{"credit_code":" abc 123 "},{"credit_code":"ABC123"}]',
                            encoding="utf-8")
            self.assertEqual(_new_p_codes([path]), {"ABC123"})
