import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from modelling_helper_functions import (
    FeaturePreprocessor, GroupImputer, clean_data, feature_engineer_train_test,
)
from model_training import (
    train_logistic_regression,
    train_decision_tree_with_grid_search,
    train_random_forest_with_grid_search,
    train_bagging_decision_tree,
)
from model_persistence import save_prediction_pipeline, load_prediction_pipeline


def accident_rows():
    return pd.DataFrame({
        "State": ["CA"] * 12,
        "Start_Time": ["2020-01-01 12:00:00"] * 12,
        "End_Time": ["2020-01-01 13:00:00"] * 12,
        "Weather_Timestamp": ["2020-01-01 11:00:00"] * 12,
        "Temperature(F)": [10., np.nan, 20., 30., 40., 50., 60., 70., 80., 90., 100., 110.],
        "Wind_Direction": ["N", None, "S", "N"] * 3,
        "Severity": [1, 2] * 6,
    })


class PreprocessingTests(unittest.TestCase):
    def test_unused_missing_columns_do_not_drop_rows_and_retention_is_logged(self):
        rows = accident_rows()
        expected = clean_data(rows)
        for col in ["Street", "City", "County", "Zipcode", "Airport_Code", "Weather_Condition"]:
            rows[col] = None
        with self.assertLogs("modelling_helper_functions", level="INFO") as logs:
            result = clean_data(rows)
        pd.testing.assert_frame_equal(result, expected)
        self.assertTrue(any("Severity 1 retention: 6/6" in line for line in logs.output))
        self.assertTrue(any("Severity 2 retention: 6/6" in line for line in logs.output))
        rows.loc[0, "State"] = None
        result = clean_data(rows)
        self.assertEqual(len(result), len(expected) - 1)
        self.assertNotIn(0, result.index)

    def test_outcome_fields_are_neither_required_nor_used(self):
        rows = accident_rows()
        rows["End_Lat"] = np.nan
        rows["End_Lng"] = np.nan
        rows["Distance(mi)"] = np.nan
        rows["Time_Duration(min)"] = 99999
        rows["Description"] = None
        rows["Future_Outcome"] = 123
        rows["End_Time"] = "not a timestamp"
        excluded = ["End_Time", "End_Lat", "End_Lng", "Distance(mi)",
                    "Time_Duration(min)", "Description", "Future_Outcome"]
        cleaned = clean_data(rows)
        pd.testing.assert_frame_equal(cleaned, clean_data(rows.drop(columns=excluded)))
        self.assertEqual(len(cleaned), 12)
        X, y = cleaned.drop(columns="Severity"), cleaned["Severity"]
        model, _ = train_logistic_regression(
            X, y, preprocessor=FeaturePreprocessor(), max_iter=1000
        )
        contaminated = X.assign(**{col: 99999 for col in excluded})
        np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(contaminated))
        preprocessor = FeaturePreprocessor().fit(contaminated)
        self.assertTrue(set(excluded).isdisjoint(preprocessor.get_feature_names_out()))
        self.assertTrue(set(excluded).isdisjoint(preprocessor.feature_names_in_))

    def test_weather_must_be_observed_by_prediction_time(self):
        rows = accident_rows().iloc[:5].copy()
        rows["Start_Lat"] = range(5)  # Keep rows distinct after unavailable data is masked.
        rows["Temperature(F)"] = [10., 20., 999., 888., 777.]
        rows["Wind_Speed(mph)"] = 50.
        rows["Wind_Direction"] = "Calm"
        rows["Weather_Timestamp"] = [
            "2020-01-01 11:00:00", "2020-01-01 12:00:00",
            "2020-01-01 12:01:00", None, "invalid",
        ]
        cleaned = clean_data(rows)
        self.assertEqual(cleaned["Temperature(F)"].iloc[:2].tolist(), [10., 20.])
        self.assertTrue(cleaned["Temperature(F)"].iloc[2:].isna().all())
        self.assertTrue(cleaned["Wind_Direction"].iloc[2:].isna().all())
        self.assertTrue(cleaned["Wind_Speed(mph)"].iloc[2:].isna().all())
        preprocessor = FeaturePreprocessor().fit(cleaned.drop(columns="Severity"))
        self.assertEqual(preprocessor.imputer_.fallbacks_["Temperature(F)"], 15.)
        no_timestamp = clean_data(rows.drop(columns="Weather_Timestamp"))
        self.assertTrue(no_timestamp["Temperature(F)"].isna().all())

    def test_encoder_reuses_training_categories_across_batches(self):
        X = clean_data(accident_rows()).drop(columns="Severity")
        X["Side"] = ["L", "R"] * 6
        X["Civil_Twilight"] = ["Day", "Night"] * 6
        preprocessor = FeaturePreprocessor().fit(X)
        original_categories = [c.copy() for c in preprocessor.encoder_.categories_]
        held_out = X.iloc[[2, 0]].copy()
        held_out["Wind_Direction"] = ["S", "W"]
        held_out["Side"] = ["R", "NEW"]
        held_out["Civil_Twilight"] = ["Night", "NEW"]
        result = preprocessor.transform(held_out)
        self.assertEqual(list(result), list(preprocessor.transform(X)))
        self.assertEqual(list(result), list(preprocessor.get_feature_names_out()))
        self.assertEqual(result.index.tolist(), held_out.index.tolist())
        self.assertEqual(result.loc[2, "Wind_Direction_S"], 1)
        for prefix in ["Wind_Direction_", "Side_", "Civil_Twilight_"]:
            self.assertEqual(result.filter(like=prefix).loc[0].sum(), 0)
        pd.testing.assert_frame_equal(
            preprocessor.transform(held_out.iloc[[0]]), result.iloc[[0]]
        )
        pd.testing.assert_frame_equal(
            preprocessor.transform(held_out[held_out.columns[::-1]]), result
        )
        for before, after in zip(original_categories, preprocessor.encoder_.categories_):
            np.testing.assert_array_equal(before, after)
        train_encoded, test_encoded = feature_engineer_train_test(X, held_out)
        self.assertEqual(list(train_encoded), list(test_encoded))
        pd.testing.assert_frame_equal(test_encoded, result)

    def test_single_category_and_no_categorical_features(self):
        X = clean_data(accident_rows()).drop(columns="Severity")
        X["Wind_Direction"] = "N"
        preprocessor = FeaturePreprocessor().fit(X)
        held_out = X.iloc[[0]].copy()
        held_out["Wind_Direction"] = "S"
        self.assertEqual(preprocessor.transform(X)["Wind_Direction_N"].sum(), len(X))
        self.assertEqual(preprocessor.transform(held_out)["Wind_Direction_N"].iloc[0], 0)
        numeric = X.drop(columns="Wind_Direction")
        numeric_preprocessor = FeaturePreprocessor().fit(numeric)
        self.assertIsNone(numeric_preprocessor.encoder_)
        self.assertEqual(len(numeric_preprocessor.transform(numeric)), len(numeric))

    def test_cleaning_preserves_missing_values_for_training(self):
        cleaned = clean_data(accident_rows())
        self.assertEqual(len(cleaned), 12)
        self.assertTrue(pd.isna(cleaned.loc[1, "Temperature(F)"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "Wind_Direction"]))

    def test_transform_uses_training_statistics_and_fallbacks(self):
        train = pd.DataFrame({
            "State": ["CA", "CA", "TX"], "Start_Month": [1, 1, 1],
            "Temperature(F)": [10., 30., np.nan],
            "Wind_Direction": ["N", "N", None],
            "Humidity(%)": [np.nan] * 3,
            "Civil_Twilight": [None] * 3,
        })
        imputer = GroupImputer().fit(train)
        held_out = train.copy()
        held_out.index = [9, 7, 5]
        held_out["State"] = ["CA", "CA", "UNSEEN"]
        held_out["Temperature(F)"] = [10000., np.nan, np.nan]
        held_out["Wind_Direction"] = ["S", None, None]
        original = held_out.copy(deep=True)
        result = imputer.transform(held_out)
        self.assertEqual(result["Temperature(F)"].tolist(), [10000., 20., 20.])
        self.assertEqual(result["Wind_Direction"].tolist(), ["S", "N", "N"])
        self.assertEqual(result["Humidity(%)"].tolist(), [0.] * 3)
        self.assertEqual(result["Civil_Twilight"].tolist(), ["Unknown"] * 3)
        pd.testing.assert_frame_equal(held_out, original)
        pd.testing.assert_frame_equal(
            imputer.transform(held_out.iloc[[1]]), result.iloc[[1]]
        )

    def test_cross_validation_fits_statistics_per_fold(self):
        data = clean_data(accident_rows())
        X, y = data.drop(columns="Severity"), data["Severity"]
        X["Wind_Direction"] = ["N"] * 4 + ["S"] * 4 + ["W"] * 4
        folds = list(KFold(3).split(X))
        pipeline = Pipeline([
            ("preprocessor", FeaturePreprocessor()),
            ("model", DecisionTreeClassifier(random_state=42)),
        ])
        results = cross_validate(pipeline, X, y, cv=folds, return_estimator=True)
        for (train, _), estimator in zip(folds, results["estimator"]):
            statistics = estimator.named_steps["preprocessor"].imputer_.statistics_
            self.assertAlmostEqual(
                statistics["Temperature(F)"].loc[("CA", 1)],
                X.iloc[train]["Temperature(F)"].mean(),
            )
            encoder = estimator.named_steps["preprocessor"].encoder_
            self.assertEqual(
                set(encoder.categories_[0]), set(X.iloc[train]["Wind_Direction"])
            )

    def test_all_trainers_predict_and_serialize_with_preprocessing(self):
        data = clean_data(accident_rows())
        X, y = data.drop(columns="Severity"), data["Severity"]
        test = X.iloc[-2:].copy()
        test["Wind_Direction"] = "UNSEEN"
        test["Temperature(F)"] = np.nan
        trainers = [
            (train_logistic_regression, {"max_iter": 1000}),
            (train_decision_tree_with_grid_search, {"param_grid": {"max_depth": [2]}, "cv": 2, "n_jobs": 1}),
            (train_random_forest_with_grid_search, {"param_grid": {"max_depth": [2]}, "cv": 2, "n_jobs": 1, "n_estimators": 2}),
            (train_bagging_decision_tree, {"bagging_params": {"n_estimators": 2, "random_state": 42}}),
        ]
        for trainer, kwargs in trainers:
            with self.subTest(trainer=trainer.__name__):
                model, metrics = trainer(
                    X.iloc[:-2], y.iloc[:-2], test, y.iloc[-2:],
                    preprocessor=FeaturePreprocessor(), **kwargs,
                )
                self.assertIn("accuracy", metrics)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pipeline.pkl"
                    config = {"random_state": 42, "train_rows": len(X) - 2}
                    save_prediction_pipeline(
                        model, path, model_name=trainer.__name__,
                        metrics=metrics, training_config=config,
                    )
                    restored = load_prediction_pipeline(path)
                    # The file remains a directly usable sklearn Pipeline.
                    with path.open("rb") as file:
                        self.assertIsInstance(pickle.load(file), Pipeline)
                np.testing.assert_array_equal(model.predict(test), restored.predict(test))
                np.testing.assert_allclose(model.predict_proba(test), restored.predict_proba(test))
                self.assertEqual(restored.metadata_["input_columns"], list(X.columns))
                self.assertEqual(restored.metadata_["training_config"], config)
                self.assertEqual(restored.metadata_["metrics"], metrics)
                self.assertEqual(
                    restored.metadata_["encoded_feature_names"],
                    list(model.named_steps["preprocessor"].get_feature_names_out()),
                )
                self.assertIn("scikit-learn", restored.metadata_["versions"])
                if trainer is train_logistic_regression:
                    np.testing.assert_array_equal(
                        model.named_steps["scaler"].scale_,
                        restored.named_steps["scaler"].scale_,
                    )
                self.assertAlmostEqual(
                    model.named_steps["preprocessor"].imputer_.fallbacks_["Temperature(F)"],
                    X.iloc[:-2]["Temperature(F)"].mean(),
                )

    def test_save_rejects_estimator_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pkl"
            with self.assertRaisesRegex(ValueError, "complete fitted prediction Pipeline"):
                save_prediction_pipeline(
                    DecisionTreeClassifier(), path, model_name="incomplete",
                    metrics={}, training_config={},
                )
            self.assertFalse(path.exists())

    def test_load_rejects_legacy_estimator_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            with path.open("wb") as file:
                pickle.dump(DecisionTreeClassifier(), file)
            with self.assertRaisesRegex(ValueError, "complete fitted prediction Pipeline"):
                load_prediction_pipeline(path)


if __name__ == "__main__":
    unittest.main()
