from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import parallel_anytime_inference as pai


def backend_source(bucket: str, sleep_sec: float, valid: bool = True) -> str:
    return f'''#!/usr/bin/env python3
import csv, sys, time
time.sleep({sleep_sec!r})
args = sys.argv[1:]
inp = args[args.index("--input") + 1]
out = args[args.index("--output") + 1]
with open(inp, newline="", encoding="utf-8") as src:
    rows = list(csv.DictReader(src))
with open(out, "w", newline="", encoding="utf-8") as dst:
    writer = csv.writer(dst)
    writer.writerow({["Case", "bucket"] if valid else ["bad", "header"]!r})
    for row in rows:
        writer.writerow([row["Case"], {bucket!r}])
'''


class ParallelAnytimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.input_csv = self.root / "input.csv"
        with self.input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Case", "Regr Log", "Sim Log"])
            writer.writerow(["1", "a/regr.log", "a/sim.log"])
            writer.writerow(["2", "b/regr.log", "b/sim.log"])
        self.output = self.root / "output.csv"
        self.diagnostics = self.root / "diagnostics.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, baseline: str, expert: str, baseline_timeout=1.0, expert_timeout=1.0):
        baseline_path = self.root / "baseline.py"
        expert_path = self.root / "expert.py"
        baseline_path.write_text(baseline, encoding="utf-8")
        expert_path.write_text(expert, encoding="utf-8")
        baseline_path.chmod(0o755)
        expert_path.chmod(0o755)
        started = time.monotonic()
        status = pai.main([
            "--input", str(self.input_csv),
            "--output", str(self.output),
            "--k", "2",
            "--baseline-backend", str(baseline_path),
            "--expert-backend", str(expert_path),
            "--baseline-extra-json", "[]",
            "--expert-extra-json", "[]",
            "--baseline-timeout", str(baseline_timeout),
            "--expert-timeout", str(expert_timeout),
            "--total-timeout", "2",
            "--diagnostics", str(self.diagnostics),
        ])
        return status, time.monotonic() - started

    def _buckets(self) -> list[str]:
        with self.output.open(newline="", encoding="utf-8") as handle:
            return [row["bucket"] for row in csv.DictReader(handle)]

    def test_baseline_publishes_then_expert_upgrades(self):
        status, runtime = self._run(
            backend_source("baseline", 0.05),
            backend_source("expert", 0.20),
        )
        self.assertEqual(status, 0)
        self.assertLess(runtime, 0.8)
        self.assertEqual(self._buckets(), ["expert", "expert"])
        diagnostics = json.loads(self.diagnostics.read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["selected"], "expert")
        self.assertEqual({row["name"] for row in diagnostics["outcomes"]}, {"baseline", "expert"})

    def test_expert_timeout_preserves_baseline(self):
        status, runtime = self._run(
            backend_source("baseline", 0.05),
            backend_source("expert", 5.0),
            expert_timeout=0.20,
        )
        self.assertEqual(status, 0)
        self.assertLess(runtime, 0.8)
        self.assertEqual(self._buckets(), ["baseline", "baseline"])

    def test_expert_can_finish_before_slow_baseline(self):
        status, runtime = self._run(
            backend_source("baseline", 5.0),
            backend_source("expert", 0.05),
            baseline_timeout=1.0,
        )
        self.assertEqual(status, 0)
        self.assertLess(runtime, 0.8)
        self.assertEqual(self._buckets(), ["expert", "expert"])

    def test_invalid_outputs_preserve_singletons(self):
        status, _runtime = self._run(
            backend_source("bad", 0.01, valid=False),
            backend_source("bad", 0.01, valid=False),
        )
        self.assertEqual(status, 0)
        self.assertTrue(all(bucket.startswith("bucket_emergency_") for bucket in self._buckets()))

    def test_missing_backend_preserves_singletons(self):
        status = pai.main([
            "--input", str(self.input_csv),
            "--output", str(self.output),
            "--k", "2",
            "--baseline-backend", str(self.root / "missing_backend"),
            "--baseline-extra-json", "[]",
            "--baseline-timeout", "0.2",
            "--total-timeout", "0.5",
            "--diagnostics", str(self.diagnostics),
        ])
        self.assertEqual(status, 0)
        self.assertTrue(all(bucket.startswith("bucket_emergency_") for bucket in self._buckets()))
        diagnostics = json.loads(self.diagnostics.read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["selected"], "singleton")
        self.assertEqual(len(diagnostics["outcomes"]), 1)
        self.assertTrue(
            diagnostics["outcomes"][0]["status"].startswith("startup_error:")
        )


if __name__ == "__main__":
    unittest.main()
