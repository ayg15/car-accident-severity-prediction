import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from model_persistence import save_prediction_pipeline
from modelling_helper_functions import (
    clean_data,
    split_data,
    FeaturePreprocessor,
)
from model_training import select_best_model, evaluate_model
import logging

logger = logging.getLogger(__name__)


def get_the_best_model(
    dataset_path="US_Accidents_Dec20_updated.csv",
    sample_size=200000,
    seed=42,
    output_path="best_model.pkl",
    report_path=None,
    cv=5,
    n_jobs=-1,
):
    """Train, select, evaluate once, and write the pipeline plus a JSON report."""
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive or None (all rows).")
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    dataset_path, output_path = Path(dataset_path), Path(output_path)
    report_path = (
        Path(report_path) if report_path else output_path.with_suffix(".report.json")
    )
    if len({p.resolve() for p in [dataset_path, output_path, report_path]}) != 3:
        raise ValueError("Dataset, model output, and report must have different paths.")
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    split_config = {
        "target_column": "Severity",
        "sample_size": sample_size,
        "test_size": 0.2,
        "random_state": seed,
        "stratify": True,
    }
    df = pd.read_csv(dataset_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df, **split_config)
    # Selection has no access to the held-out test set.
    selection_started = perf_counter()
    best_name, best_model, cv_results = select_best_model(
        X_train,
        y_train,
        preprocessor=FeaturePreprocessor(),
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
    )
    training_duration = perf_counter() - selection_started
    logger.info("Selected model from training CV: %s", best_name)

    # Evaluate exactly once, after the winner and its parameters are fixed.
    test_metrics = evaluate_model(best_model, X_test, y_test, y_train=y_train)
    logger.info("Final held-out test metrics: %s", test_metrics)

    training_config = {
        "dataset_path": str(dataset_path),
        "split": split_config,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_fit_rows": len(X_train),
        "selection_metric": "mean CV balanced_accuracy",
        "cv": cv,
        "cv_random_state": seed,
        "n_jobs": n_jobs,
        "cv_scoring": "balanced_accuracy",
        "cv_results": cv_results,
        "evaluation_split": "held-out test (winner only)",
        "training_duration_seconds": training_duration,
    }
    report = {
        "report_version": 1,
        "started_at_utc": started_at,
        "selected_model": best_name,
        "model_path": str(output_path),
        "training_config": training_config,
        "cv_results": cv_results,
        "final_evaluation": test_metrics,
        "duration_seconds": {
            "training_and_selection": training_duration,
            "through_evaluation": perf_counter() - started,
        },
    }
    # Serialize before writing either artifact, so invalid report data fails early.
    report_json = json.dumps(report, indent=2, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_pipeline(
        best_model,
        output_path,
        model_name=best_name,
        metrics=test_metrics,
        training_config=training_config,
    )
    report_path.write_text(report_json + "\n", encoding="utf-8")

    logger.info(
        "Prediction pipeline saved to %s; report saved to %s", output_path, report_path
    )
    return report


def _sample_size(value):
    if value.lower() == "all":
        return None
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive integer or 'all'.") from error
    if size <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer or 'all'.")
    return size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate accident severity pipelines."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("US_Accidents_Dec20_updated.csv")
    )
    parser.add_argument(
        "--sample-size", type=_sample_size, default=200000, help="Row count or 'all'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling, splitting, CV and estimators.",
    )
    parser.add_argument("--output", type=Path, default=Path("best_model.pkl"))
    parser.add_argument(
        "--report", type=Path, help="Defaults to OUTPUT with suffix .report.json."
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.cv < 2:
        parser.error("--cv must be at least 2")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return get_the_best_model(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        output_path=args.output,
        report_path=args.report,
        cv=args.cv,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
