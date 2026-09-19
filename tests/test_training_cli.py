import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import train
from model_persistence import load_prediction_pipeline
from model_training import select_best_model, train_bagging_decision_tree
from modelling_helper_functions import FeaturePreprocessor, clean_data


class TrainingCliTests(unittest.TestCase):
    def test_cli_forwards_configuration(self):
        with patch.object(train, "get_the_best_model") as run, patch.object(
            train.logging, "basicConfig"
        ):
            train.main(
                [
                    "--dataset",
                    "input.csv",
                    "--sample-size",
                    "all",
                    "--seed",
                    "7",
                    "--output",
                    "artifacts/model.pkl",
                    "--report",
                    "artifacts/run.json",
                    "--cv",
                    "2",
                    "--n-jobs",
                    "1",
                ]
            )
        run.assert_called_once_with(
            dataset_path=Path("input.csv"),
            sample_size=None,
            seed=7,
            output_path=Path("artifacts/model.pkl"),
            report_path=Path("artifacts/run.json"),
            cv=2,
            n_jobs=1,
        )

    def test_imports_do_not_configure_root_logging(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging; from unittest.mock import patch; "
                "p=patch('logging.basicConfig'); m=p.start(); "
                "import model_training, train; m.assert_not_called()",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bagging_seed_is_reproducible_without_mutating_parameters(self):
        X = pd.DataFrame({"x": np.arange(30)})
        y = pd.Series([0, 1] * 15)
        params = {"n_estimators": 3}
        first, _ = train_bagging_decision_tree(
            X, y, bagging_params=params, random_state=7
        )
        second, _ = train_bagging_decision_tree(
            X, y, bagging_params=params, random_state=7
        )
        self.assertEqual(params, {"n_estimators": 3})
        self.assertEqual(first.named_steps["model"].random_state, 7)
        np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))

    def test_run_writes_loadable_pipeline_and_json_report(self):
        times = pd.date_range("2020-01-01", periods=40, freq="h")
        rows = pd.DataFrame(
            {
                "Start_Time": times,
                "State": "CA",
                "Weather_Timestamp": times,
                "Temperature(F)": [10.0, np.nan, 20.0, 30.0] * 10,
                "Wind_Direction": ["N", "S", None, "N"] * 10,
                "Severity": [1, 2] * 20,
            }
        )
        candidates = {
            "Small Tree": (
                Pipeline(
                    [
                        ("preprocessor", FeaturePreprocessor()),
                        ("model", DecisionTreeClassifier(random_state=7)),
                    ]
                ),
                {"max_depth": [1, 2]},
            )
        }
        selected = []

        def small_search(X, y, **kwargs):
            result = select_best_model(X, y, candidates=candidates, **kwargs)
            selected.append(result[1])
            return result

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            output = Path(directory) / "artifacts" / "model.pkl"
            rows.to_csv(source, index=False)
            with patch.object(train, "select_best_model", side_effect=small_search):
                report = train.get_the_best_model(
                    source, sample_size=32, seed=7, output_path=output, cv=2, n_jobs=1
                )
            saved_report = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(saved_report, report)
            self.assertEqual(report["selected_model"], "Small Tree")
            self.assertEqual(report["training_config"]["split"]["random_state"], 7)
            self.assertEqual(len(report["cv_results"][0]["fold_scores"]), 2)
            self.assertEqual(len(report["cv_results"][0]["parameter_results"]), 2)
            self.assertGreaterEqual(
                report["cv_results"][0]["training_duration_seconds"], 0
            )
            self.assertIn("classification_report", report["final_evaluation"])
            restored = load_prediction_pipeline(output)
            self.assertEqual(restored.metadata_["metrics"], report["final_evaluation"])
            features = clean_data(rows.drop(columns="Severity"))
            features["Wind_Direction"] = "UNSEEN"
            np.testing.assert_allclose(
                restored.predict_proba(features), selected[0].predict_proba(features)
            )

    def test_output_cannot_overwrite_dataset(self):
        with self.assertRaisesRegex(ValueError, "different paths"):
            train.get_the_best_model("data.csv", output_path="data.csv")


if __name__ == "__main__":
    unittest.main()
