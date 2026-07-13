from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "beta_v2_router.sh"


def backend_source(bucket: str, sleep_sec: float = 0.0, valid: bool = True) -> str:
    return f'''#!/usr/bin/env python3
import csv, sys, time
time.sleep({sleep_sec!r})
args = sys.argv[1:]
def last(flag):
    indices = [i for i, value in enumerate(args) if value == flag]
    return args[indices[-1] + 1]
inp = last("--input")
out = last("--output")
with open(inp, newline="", encoding="utf-8") as src:
    rows = list(csv.DictReader(src))
with open(out, "w", newline="", encoding="utf-8") as dst:
    writer = csv.writer(dst)
    writer.writerow({["wrong", "header"] if not valid else ["Case", "bucket"]!r})
    for row in rows:
        writer.writerow([row["Case"], {bucket!r}])
'''


class BetaV2RouterTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.package = self.root / "package"
        self.fast = self.package / "fast/regr_fail_bucketing_fast/regr_fail_bucketing_fast"
        self.multiview = self.package / "multiview/regr_fail_bucketing_multiview"
        self.full = self.package / "regr_fail_bucketing_full"
        self.input_csv = self.root / "input.csv"
        with self.input_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
            writer.writerow(["1", "a/regr.log", "a/sim.log", "a/trace.log"])
            writer.writerow(["2", "b/regr.log", "b/sim.log", "b/trace.log"])
        self.output = self.root / "output.csv"
        self._write_backend(self.fast, backend_source("bucket_baseline"))
        self._write_backend(self.multiview, backend_source("bucket_expert"))
        self._write_backend(self.full, backend_source("bucket_dual"))

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_backend(path: Path, source: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def _run(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({
            "BETA_V2_PACKAGE_ROOT": str(self.package),
            "LLM_MODEL_CONFIG": "embedding: {}",
            "BETA_V2_BASELINE_LIMIT": "2",
            "BETA_V2_EXPERT_LIMIT": "2",
            "BETA_V2_EXIT_RESERVE": "1",
        })
        env.update(overrides)
        return subprocess.run(
            [
                str(ROUTER), "--input", str(self.input_csv),
                "--output", str(self.output), "--k", "2",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=False,
        )

    def _buckets(self) -> list[str]:
        with self.output.open(newline="", encoding="utf-8") as handle:
            return [row["bucket"] for row in csv.DictReader(handle)]

    def test_valid_expert_atomically_upgrades_baseline(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._buckets(), ["bucket_expert", "bucket_expert"])
        self.assertIn("published baseline output", result.stderr)
        self.assertIn("published multiview output", result.stderr)

    def test_expert_timeout_preserves_baseline(self):
        self._write_backend(self.multiview, backend_source("bucket_expert", sleep_sec=5.0))
        result = self._run(BETA_V2_EXPERT_LIMIT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._buckets(), ["bucket_baseline", "bucket_baseline"])
        self.assertIn("keeping baseline", result.stderr)

    def test_missing_embedding_config_keeps_baseline(self):
        result = self._run(LLM_MODEL_CONFIG="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._buckets(), ["bucket_baseline", "bucket_baseline"])

    def test_invalid_backends_leave_valid_singletons(self):
        self._write_backend(self.fast, backend_source("bad", valid=False))
        self._write_backend(self.multiview, backend_source("bad", valid=False))
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        buckets = self._buckets()
        self.assertEqual(len(buckets), 2)
        self.assertTrue(all(bucket.startswith("bucket_emergency_") for bucket in buckets))


if __name__ == "__main__":
    unittest.main()
