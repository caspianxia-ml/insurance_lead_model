import tempfile
import unittest
from pathlib import Path

from experiments.validate_export import scan_csv


class ExportValidationTest(unittest.TestCase):
    def test_quoted_multiline_text_is_valid_but_extra_column_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.csv"
            invalid = root / "invalid.csv"
            valid.write_text('id,description\n1,"two lines\nwith a comma, too"\n', encoding="utf-8")
            invalid.write_text("id,description\n1,unquoted,comma\n", encoding="utf-8")
            self.assertTrue(scan_csv(valid)["valid"])
            report = scan_csv(invalid)
            self.assertFalse(report["valid"])
            self.assertEqual(report["invalid_width_records"], 1)
            self.assertEqual(report["first_invalid_record_numbers"], [2])
