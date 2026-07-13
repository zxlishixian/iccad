from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import anytime_inference as ai


SUCCESS_BACKEND = r'''
import csv, sys
args = sys.argv[1:]
inp = args[args.index("--input") + 1]
out = args[args.index("--output") + 1]
with open(inp, newline="", encoding="utf-8") as src:
    rows = list(csv.DictReader(src))
with open(out, "w", newline="", encoding="utf-8") as dst:
    writer = csv.writer(dst)
    writer.writerow(["Case", "bucket"])
    for idx, row in enumerate(rows):
        writer.writerow([row["Case"], f"bucket_{idx % 2:03d}"])
'''


class AnytimeInferenceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.input_csv = self.root / "input.csv"
        with self.input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
            writer.writerow(["1", "a/regr.log", "a/sim.log.gz", "a/trace.log.gz"])
            writer.writerow(["2", "b/regr.log", "b/sim.log.gz", "b/trace.log.gz"])
        self.cases = ["1", "2"]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_singleton_is_immediately_valid(self):
        output = self.root / "output.csv"
        ai.write_singleton_output(output, self.cases)
        self.assertTrue(ai.validate_output(output, self.cases))

    def test_valid_backend_candidate_replaces_singleton(self):
        output = self.root / "candidate.csv"
        result = ai.run_backend(
            "success",
            [sys.executable, "-c", SUCCESS_BACKEND],
            self.input_csv,
            output,
            2,
            2.0,
            self.cases,
        )
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.output_valid)

    def test_timeout_preserves_preexisting_singleton(self):
        output = self.root / "output.csv"
        candidate = self.root / "candidate.csv"
        ai.write_singleton_output(output, self.cases)
        result = ai.run_backend(
            "timeout",
            [sys.executable, "-c", "import time; time.sleep(10)"],
            self.input_csv,
            candidate,
            2,
            0.05,
            self.cases,
        )
        self.assertEqual(result.status, "timeout")
        self.assertFalse(result.output_valid)
        self.assertTrue(ai.validate_output(output, self.cases))


if __name__ == "__main__":
    unittest.main()
