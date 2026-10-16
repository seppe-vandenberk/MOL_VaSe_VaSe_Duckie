"""Offline checks of per-run CSV export and evaluator process handling."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts/evaluate_models.ipynb"


class EvaluationNotebookTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {}
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, str(NOTEBOOK), "exec")
            if "import zipfile" in source or "def evaluate_models(" in source:
                exec(source, self.namespace)
        self.report = json.loads((ROOT / "models/baseline.json").read_text())["evaluation"]

    def model(self, folder, name):
        path = folder / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({
                "training_config": {"LEARNING_RATE": 1e-5, "MAX_SPEED": 0.42},
                "learning_rate": 1e-5, "num_timesteps": 5000, "warm_start_steps": 23177,
            }))
        return path

    def evaluate(self, paths, output):
        with redirect_stdout(io.StringIO()):
            return self.namespace["evaluate_models"](
                paths, output, runs=5, duration=60, port=7512,
                entity="map_0/vehicle_0", device="cpu", same_start=False,
            )

    def evaluator(self, command, **kwargs):
        self.assertEqual(Path(command[2]), ROOT / "scripts/evaluate.py")
        self.assertEqual(command[command.index("--runs") + 1], "5")
        self.assertEqual(command[command.index("--port") + 1], "7512")
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(self.report))
        # A failed screening is a valid report, despite exit code 1.
        return subprocess.CompletedProcess(command, 1)

    def test_all_models_export_five_distinct_runs_with_numeric_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = [self.model(folder, name) for name in ("baseline.zip", "checkpoint.zip")]
            output = folder / "results"
            with patch.object(subprocess, "run", side_effect=self.evaluator) as run:
                rows, issues = self.evaluate(paths, output)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(issues, [])
            self.assertEqual(len(rows), 10)
            with (output / "evaluations.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames[:4], ["learning_rate", "speed", "training_steps", "steps"])
                saved = list(reader)
            for name in ("baseline.zip", "checkpoint.zip"):
                model_rows = [row for row in saved if row["model"] == name]
                self.assertEqual([int(row["run"]) for row in model_rows], [1, 2, 3, 4, 5])
                self.assertEqual([float(row["reward_total"]) for row in model_rows],
                                 [run["reward_total"] for run in self.report["runs"]])
            for row in saved:
                self.assertEqual(int(row["training_steps"]), 5000)
                self.assertEqual(float(row["speed"]), 0.42)
                self.assertEqual(int(row["warm_start_steps"]), 23177)
                self.assertEqual(sum(int(row[f"action_count_{name}"])
                                     for name in self.namespace["ACTION_NAMES"]), int(row["steps"]))
                self.assertEqual(int(row["termination_reason_count"]), int(row["failures"]))
            reason = "termination_reason_count::lane not recovered for 40 steps"
            self.assertEqual([int(row[reason]) for row in saved[:5]], [0, 1, 0, 0, 1])

    def test_short_report_keeps_real_rows_and_marks_batch_incomplete(self):
        self.report["runs"].pop()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = self.model(folder, "checkpoint.zip")
            with patch.object(subprocess, "run", side_effect=self.evaluator):
                rows, issues = self.evaluate([path], folder / "results")
            self.assertEqual(len(rows), 4)
            self.assertIn("expected 5 runs, received 4", issues[0]["error"])
            persisted = json.loads((folder / "results/issues.json").read_text())
            self.assertEqual(issues, persisted)

    def test_process_error_does_not_reuse_stale_report_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = [self.model(folder, name) for name in ("broken.zip", "working.zip")]
            output = folder / "results"
            (output / "reports").mkdir(parents=True)
            stale = output / "reports/broken.json"
            stale.write_text(json.dumps(self.report))

            def evaluate(command, **kwargs):
                if command[command.index("--model") + 1].endswith("broken.zip"):
                    return subprocess.CompletedProcess(command, 2)
                return self.evaluator(command, **kwargs)

            with patch.object(subprocess, "run", side_effect=evaluate):
                rows, issues = self.evaluate(paths, output)
            self.assertFalse(stale.exists())
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["model"] == "working.zip" for row in rows))
            self.assertEqual(issues[0]["model"], "broken.zip")
            self.assertIn("without a JSON report", issues[0]["error"])

    def test_renamed_baseline_metadata_and_minimum_run_count(self):
        metadata = self.namespace["checkpoint_metadata"](ROOT / "models/baseline.zip")
        self.assertEqual(metadata["training_steps"], 23177)
        self.assertEqual(metadata["learning_rate"], 0.0001)
        self.assertEqual(metadata["speed"], 0.35)
        with self.assertRaisesRegex(ValueError, "five runs"):
            self.namespace["evaluate_models"](
                [], Path("unused"), runs=4, duration=60, port=7512,
                entity="map_0/vehicle_0", device="cpu", same_start=False,
            )


if __name__ == "__main__":
    unittest.main()
