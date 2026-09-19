import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from model_training import (
    select_best_model,
    evaluate_model,
    train_random_forest_with_grid_search,
)
import train as workflow


class SelectionTests(unittest.TestCase):
    def test_random_forest_uses_rows_beyond_old_subset_limit(self):
        X = pd.DataFrame({"x": np.arange(20010)})
        y = pd.Series([0, 1] * 10005)
        model, _ = train_random_forest_with_grid_search(
            X,
            y,
            param_grid={"max_depth": [1]},
            cv=2,
            n_jobs=1,
            n_estimators=1,
            bootstrap=False,
            random_state=42,
        )
        self.assertEqual(
            model.named_steps["model"].estimators_[0].tree_.n_node_samples[0], len(X)
        )

    def test_report_exposes_minority_failure_and_training_baseline(self):
        model = Mock()
        model.predict.return_value = np.array([1, 1, 1, 1])
        metrics = evaluate_model(
            model, pd.DataFrame({"x": range(4)}), [1, 1, 1, 2], y_train=[2, 2, 2, 1, 3]
        )
        self.assertEqual(metrics["labels"], [1, 2, 3])
        self.assertEqual(metrics["confusion_matrix"], [[3, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(metrics["classification_report"]["2"]["recall"], 0.0)
        self.assertEqual(metrics["class_counts"]["3"], 0)
        self.assertAlmostEqual(metrics["macro_f1"], (6 / 7) / 3)
        self.assertEqual(metrics["majority_baseline"]["predicted_class"], 2)
        self.assertEqual(metrics["majority_baseline"]["accuracy"], 0.25)
        model.predict.assert_called_once()

    def test_winner_uses_shared_training_cv_and_is_refitted(self):
        X = pd.DataFrame({"signal": [0.0, 1.0] * 12})
        y = pd.Series([0, 1] * 12)
        candidates = {
            "baseline": (
                Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
                {},
            ),
            "tree": (
                Pipeline([("model", DecisionTreeClassifier(random_state=42))]),
                {"max_depth": [1, 2]},
            ),
        }
        name, model, results = select_best_model(
            X, y, cv=3, n_jobs=1, candidates=candidates
        )
        self.assertEqual(name, "tree")
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertEqual(model.named_steps["model"].tree_.n_node_samples[0], len(X))
        folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
        for result in results:
            pipeline = candidates[result["model_name"]][0]
            pipeline.set_params(**result["best_params"])
            expected = cross_val_score(
                pipeline, X, y, cv=folds, scoring="balanced_accuracy"
            )
            self.assertAlmostEqual(result["mean_cv_balanced_accuracy"], expected.mean())

    def test_evaluation_predicts_once(self):
        model = Mock()
        model.predict.return_value = np.array([0, 1, 1, 0])
        X = pd.DataFrame({"x": range(4)})
        metrics = evaluate_model(model, X, [0, 1, 1, 0])
        model.predict.assert_called_once_with(X)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_workflow_keeps_holdout_out_of_selection(self):
        X_train = pd.DataFrame({"x": range(8)})
        X_test = pd.DataFrame({"x": [100, 101]})
        y_train, y_test = pd.Series([0, 1] * 4), pd.Series([0, 1])
        winner = Mock()
        cv_results = [{"model_name": "winner", "mean_cv_balanced_accuracy": 0.8}]
        events = []

        def select(*args, **kwargs):
            self.assertEqual(len(args), 2)
            self.assertIs(args[0], X_train)
            self.assertIs(args[1], y_train)
            self.assertNotIn("X_test", kwargs)
            self.assertNotIn("y_test", kwargs)
            events.append("select")
            return "winner", winner, cv_results

        def evaluate(model, X, y, **kwargs):
            self.assertEqual(events, ["select"])
            self.assertIs(model, winner)
            self.assertIs(X, X_test)
            self.assertIs(y, y_test)
            self.assertIs(kwargs["y_train"], y_train)
            events.append("evaluate")
            return {"balanced_accuracy": 0.5}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            workflow.pd, "read_csv"
        ), patch.object(workflow, "clean_data"), patch.object(
            workflow, "split_data", return_value=(X_train, X_test, y_train, y_test)
        ), patch.object(
            workflow, "select_best_model", side_effect=select
        ) as selection, patch.object(
            workflow, "evaluate_model", side_effect=evaluate
        ) as evaluation, patch.object(
            workflow, "save_prediction_pipeline"
        ) as save:
            workflow.get_the_best_model(output_path=Path(directory) / "model.pkl")
        selection.assert_called_once()
        evaluation.assert_called_once()
        save.assert_called_once()
        self.assertIs(save.call_args.args[0], winner)
        self.assertEqual(save.call_args.kwargs["metrics"], {"balanced_accuracy": 0.5})
        self.assertEqual(
            save.call_args.kwargs["training_config"]["cv_results"], cv_results
        )


if __name__ == "__main__":
    unittest.main()
