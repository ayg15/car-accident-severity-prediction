"""Persist fitted prediction pipelines and the metadata needed to reuse them."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import pickle
import platform

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from modelling_helper_functions import FeaturePreprocessor, EXCLUDED_OUTCOME_COLUMNS


def _validate_pipeline(pipeline):
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Expected a complete fitted prediction Pipeline.")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if not isinstance(preprocessor, FeaturePreprocessor):
        raise ValueError("Pipeline must contain a fitted FeaturePreprocessor.")
    check_is_fitted(preprocessor, ["feature_names_out_", "input_dtypes_", "prediction_time_"])
    check_is_fitted(pipeline.named_steps["model"])
    if "scaler" in pipeline.named_steps:
        check_is_fitted(pipeline.named_steps["scaler"])


def save_prediction_pipeline(
    pipeline, path, *, model_name, metrics, training_config
):
    """Save the entire fitted pipeline with metadata attached as metadata_.

    Inputs to the saved pipeline are cleaned, unencoded feature DataFrames.
    Imputation, encoding, optional scaling, and prediction require no refitting.
    """
    _validate_pipeline(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    pipeline.metadata_ = {
        "artifact_version": 2,
        "prediction_time": preprocessor.prediction_time_,
        "excluded_outcome_columns": EXCLUDED_OUTCOME_COLUMNS.copy(),
        "weather_policy": "clean_data masks observations with missing/invalid timestamps or timestamps after Start_Time",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "input_format": "cleaned, unencoded pandas DataFrame; target excluded",
        "input_columns": preprocessor.feature_names_in_.tolist(),
        "input_dtypes": preprocessor.input_dtypes_.copy(),
        "encoded_feature_names": preprocessor.get_feature_names_out().tolist(),
        "classes": pipeline.classes_.tolist(),
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        },
        "metrics": dict(metrics),
        "training_config": dict(training_config),
        "estimator_params": pipeline.named_steps["model"].get_params(deep=True),
        "scaler_params": (
            pipeline.named_steps["scaler"].get_params()
            if "scaler" in pipeline.named_steps else None
        ),
    }
    with Path(path).open("wb") as file:
        pickle.dump(pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_prediction_pipeline(path):
    """Load a trusted pipeline artifact produced by save_prediction_pipeline.

    Pickle files must come from a trusted source. Use the recorded dependency
    versions and this project's modules when loading a saved pipeline.
    """
    with Path(path).open("rb") as file:
        pipeline = pickle.load(file)
    _validate_pipeline(pipeline)
    if getattr(pipeline, "metadata_", {}).get("artifact_version") != 2:
        raise ValueError("Unsupported or missing artifact metadata; retrain and save again.")
    return pipeline
