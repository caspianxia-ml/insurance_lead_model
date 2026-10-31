"""Synthetic checks for V2 EDA parsing and entity definitions."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.v2_eda import gold_candidate, normalize_code, read_csv, representative


class V2EdaTests(unittest.TestCase):
    def test_credit_code_normalization(self):
        self.assertEqual(normalize_code(" ａ ｂ１２ "), "AB12")
        self.assertEqual(normalize_code("NULL"), "")

    def test_gold_candidate_keeps_original_labels_separate(self):
        source = pd.Series(["原力", "猎聘,量化Q3榜单", "智联招聘", "猎聘,原力"])
        self.assertEqual(gold_candidate(source).tolist(), [True, True, False, True])

    def test_representative_prefers_completeness_then_id(self):
        frame = pd.DataFrame([
            {"_code": "A", "_filled": 2, "_id": "10"},
            {"_code": "A", "_filled": 3, "_id": "09"},
            {"_code": "B", "_filled": 1, "_id": "01"},
            {"_code": "", "_filled": 9, "_id": "99"},
        ])
        selected = representative(frame)
        self.assertEqual(selected["_code"].tolist(), ["A", "B"])
        self.assertEqual(selected["_id"].tolist(), ["09", "01"])

    def test_relaxed_csv_does_not_accept_wrong_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            path.write_text('id,city\n1,"上海"style="x"\n', encoding="utf-8")
            data, info = read_csv(path, weak=True)
            self.assertEqual(data.shape, (1, 2))
            self.assertIsNotNone(info["strict_error"])
            path.write_text("id,city\n1,上海,extra\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_csv(path, weak=True)


if __name__ == "__main__":
    unittest.main()
