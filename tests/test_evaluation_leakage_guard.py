import tempfile
import unittest
from pathlib import Path

from evaluation_leakage_guard import (
    TrainingEvaluationOverlapError,
    assert_held_out,
    dataset_identity,
    find_training_overlap,
    training_identities,
)


def make_dataset(root: Path, name: str, contents: str) -> Path:
    dataset = root / name
    dataset.mkdir()
    (dataset / "input.csv").write_text(contents, encoding="utf-8")
    return dataset


def make_log_dataset(root: Path, name: str, sim: str, regr: str) -> Path:
    dataset = root / name
    dataset.mkdir()
    (dataset / "sim.log").write_text(sim, encoding="utf-8")
    (dataset / "regr.log").write_text(regr, encoding="utf-8")
    (dataset / "input.csv").write_text(
        "Case,Sim Log,Regr Log\n1,sim.log,regr.log\n", encoding="utf-8"
    )
    return dataset


class LeakageGuardTests(unittest.TestCase):
    def test_rejects_same_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset = make_dataset(Path(temp), "set1", "Case,Sim Log\n1,a.log\n")
            with self.assertRaises(TrainingEvaluationOverlapError):
                assert_held_out({"training_dataset_names": ["set1"]}, [dataset])

    def test_rejects_renamed_copy_by_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = make_dataset(root, "train", "Case,Sim Log\n1,a.log\n")
            renamed = make_dataset(root, "renamed", "Case,Sim Log\n1,a.log\n")
            overlap = find_training_overlap(
                {"training_datasets": [dataset_identity(train)]}, [renamed]
            )
            self.assertEqual(overlap[0]["matched_by"], "input_sha256")

    def test_rejects_different_csv_with_identical_case_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = make_log_dataset(root, "train", "fatal A", "mismatch B")
            test = make_log_dataset(root, "test", "fatal A", "mismatch B")
            manifest = {
                "training_datasets": [dataset_identity(train, include_case_logs=True)]
            }
            overlap = find_training_overlap(manifest, [test])
            self.assertEqual(overlap[0]["matched_by"], "case_log_sha256")
            self.assertEqual(overlap[0]["shared_case_hashes"], 1)

    def test_accepts_disjoint_case_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = make_log_dataset(root, "train", "fatal A", "mismatch B")
            test = make_log_dataset(root, "test", "fatal C", "mismatch D")
            manifest = {
                "training_datasets": [dataset_identity(train, include_case_logs=True)]
            }
            assert_held_out(manifest, [test])

    def test_missing_training_identity_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            training_identities({})


if __name__ == "__main__":
    unittest.main()
