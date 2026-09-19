import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import logging
from time import perf_counter

logger = logging.getLogger(__name__)


def _training_pipeline(estimator, preprocessor=None, scaler=None):
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", clone(preprocessor)))
    if scaler is not None:
        steps.append(("scaler", clone(scaler)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _pipeline_grid(param_grid):
    """Keep accepting estimator parameter names used by existing callers."""

    def qualify(grid):
        return {
            key if "__" in key else f"model__{key}": value
            for key, value in grid.items()
        }

    if isinstance(param_grid, list):
        return [qualify(grid) for grid in param_grid]
    return qualify(param_grid)


def select_best_model(
    X_train,
    y_train,
    preprocessor=None,
    cv=5,
    n_jobs=-1,
    random_state=42,
    candidates=None,
):
    """Select by mean training-CV balanced accuracy, without access to test data.

    Candidates map names to (unfitted pipeline, parameter grid) pairs. All
    candidates use the same stratified folds and all training rows. GridSearchCV
    refits each winner on training data; the best pipeline is ready to predict.
    CV scores are selection scores, not an unbiased performance estimate.
    """
    if candidates is None:
        candidates = {
            "Logistic Regression": (
                _training_pipeline(
                    LogisticRegression(max_iter=500, random_state=random_state),
                    preprocessor,
                    MinMaxScaler(),
                ),
                {},
            ),
            "Decision Tree": (
                _training_pipeline(
                    DecisionTreeClassifier(random_state=random_state), preprocessor
                ),
                {
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "min_samples_split": [10, 15, 30, 45, 60],
                    "min_samples_leaf": [10, 15, 30, 45, 60],
                },
            ),
            "Random Forest": (
                _training_pipeline(
                    RandomForestClassifier(random_state=random_state), preprocessor
                ),
                {
                    "max_depth": [11, 13],
                    "min_samples_split": [15, 30],
                    "min_samples_leaf": [15, 30],
                    "bootstrap": [False],
                },
            ),
            "Bagging Decision Tree": (
                _training_pipeline(
                    BaggingClassifier(
                        DecisionTreeClassifier(
                            max_depth=13,
                            min_samples_split=45,
                            min_samples_leaf=10,
                            random_state=random_state,
                        ),
                        random_state=random_state,
                    ),
                    preprocessor,
                ),
                {},
            ),
        }
    if not candidates:
        raise ValueError("At least one model candidate is required.")
    folds = list(
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state).split(
            X_train, y_train
        )
    )
    results = []
    best_name, best_model, best_score = None, None, -np.inf
    for name, (pipeline, param_grid) in candidates.items():
        search = GridSearchCV(
            pipeline,
            _pipeline_grid(param_grid),
            cv=folds,
            scoring="balanced_accuracy",
            n_jobs=n_jobs,
            error_score="raise",
        )
        started = perf_counter()
        search.fit(X_train, y_train)
        duration = perf_counter() - started
        score = float(search.best_score_)
        if not np.isfinite(score):
            raise ValueError(f"Non-finite cross-validation score for {name}.")
        results.append(
            {
                "model_name": name,
                "mean_cv_balanced_accuracy": score,
                "std_cv_balanced_accuracy": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "best_params": search.best_params_,
                "training_duration_seconds": duration,
                "fold_scores": [
                    float(
                        search.cv_results_[f"split{i}_test_score"][search.best_index_]
                    )
                    for i in range(cv)
                ],
                "parameter_results": [
                    {
                        "params": params,
                        "mean_cv_balanced_accuracy": float(
                            search.cv_results_["mean_test_score"][i]
                        ),
                        "std_cv_balanced_accuracy": float(
                            search.cv_results_["std_test_score"][i]
                        ),
                    }
                    for i, params in enumerate(search.cv_results_["params"])
                ],
            }
        )
        logger.info("%s CV balanced accuracy: %.4f", name, score)
        if score > best_score:
            best_name, best_model, best_score = name, search.best_estimator_, score
    return best_name, best_model, results


def _prediction_metrics(y_true, predictions, labels):
    """Use a fixed label order, including classes absent from this holdout."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predictions, labels=labels, average="macro", zero_division=0
        ),
        "labels": labels.tolist(),
        "class_counts": {
            str(label): int(np.sum(np.asarray(y_true) == label)) for label in labels
        },
        "classification_report": classification_report(
            y_true, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=labels
        ).tolist(),
    }


def evaluate_model(model, X_test, y_test, y_train=None):
    """Evaluate once; optionally compare a majority baseline fitted on training labels.

    Confusion-matrix rows are true classes, columns are predictions. Macro F1
    includes all training/test/predicted classes, with zero for undefined scores.
    """
    predictions = model.predict(X_test)
    labels = np.unique(
        np.concatenate(
            [
                np.asarray(y_test),
                np.asarray(predictions),
                np.asarray(y_train) if y_train is not None else np.asarray(y_test),
            ]
        )
    )
    metrics = _prediction_metrics(y_test, predictions, labels)
    if y_train is not None:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(y_train), 1)), y_train)
        baseline_predictions = baseline.predict(np.zeros((len(y_test), 1)))
        metrics["train_class_counts"] = {
            str(label): int(np.sum(np.asarray(y_train) == label)) for label in labels
        }
        metrics["majority_baseline"] = _prediction_metrics(
            y_test, baseline_predictions, labels
        )
        metrics["majority_baseline"]["predicted_class"] = np.asarray(
            baseline_predictions[0]
        ).item()
    return metrics


def train_logistic_regression(
    X_train, y_train, X_test=None, y_test=None, scaler=None, preprocessor=None, **kwargs
):
    """
    Train a pipeline, fitting preprocessing and scaling only on training rows.
    A supplied scaler is cloned and refitted. Returns pipeline and metrics.
    """
    logger.info("Training Logistic Regression...")
    model = _training_pipeline(
        LogisticRegression(**kwargs),
        preprocessor,
        scaler if scaler is not None else MinMaxScaler(),
    )
    model.fit(X_train, y_train)

    metrics = (
        evaluate_model(model, X_test, y_test, y_train=y_train)
        if X_test is not None and y_test is not None
        else {}
    )

    logger.info(f"Logistic Regression Metrics: {metrics}\n")

    return model, metrics


def train_decision_tree_with_grid_search(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    param_grid={},
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    preprocessor=None,
    **kwargs,
):
    """
    Train a Decision Tree classifier with hyperparameter tuning using GridSearchCV.
    Returns best estimator and metrics.
    """
    logger.info("Training Decision Tree...")
    grid = GridSearchCV(
        _training_pipeline(DecisionTreeClassifier(**kwargs), preprocessor),
        _pipeline_grid(param_grid),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    grid.fit(X_train, y_train)
    metrics = (
        evaluate_model(grid, X_test, y_test, y_train=y_train)
        if X_test is not None and y_test is not None
        else {}
    )

    logger.info(f"Decision Tree Metrics: {metrics}\n")

    return grid.best_estimator_, metrics


def train_random_forest_with_grid_search(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    param_grid={},
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    preprocessor=None,
    **kwargs,
):
    """
    Tune and refit a Random Forest pipeline on all training rows.
    Returns model and metrics.
    """
    logger.info("Training Random Forest...")
    grid = GridSearchCV(
        _training_pipeline(RandomForestClassifier(**kwargs), preprocessor),
        _pipeline_grid(param_grid),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    grid.fit(X_train, y_train)
    metrics = (
        evaluate_model(grid, X_test, y_test, y_train=y_train)
        if X_test is not None and y_test is not None
        else {}
    )

    logger.info(f"Random Forest Metrics: {metrics}\n")

    return grid.best_estimator_, metrics


def train_bagging_decision_tree(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    dt_params=None,
    bagging_params=None,
    preprocessor=None,
    random_state=42,
):
    """
    Train a BaggingClassifier with DecisionTreeClassifier as base estimator.
    dt_params: dict for DecisionTreeClassifier
    bagging_params: dict for BaggingClassifier
    Returns model and predictions.
    """
    logger.info("Training Bagging Classifier with Decision Tree...")
    dt_params = dict(dt_params or {})
    bagging_params = dict(bagging_params or {})
    dt_params.setdefault("random_state", random_state)
    bagging_params.setdefault("random_state", random_state)

    base_estimator = DecisionTreeClassifier(**dt_params)
    model = _training_pipeline(
        BaggingClassifier(base_estimator, **bagging_params), preprocessor
    )
    model.fit(X_train, y_train)

    metrics = (
        evaluate_model(model, X_test, y_test, y_train=y_train)
        if X_test is not None and y_test is not None
        else {}
    )

    logger.info(f"Bagging Decision Tree Metrics: {metrics}\n")

    return model, metrics
