import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train, y_train, X_test=None, y_test=None, scaler=None, **kwargs
):
    """
    Train a logistic regression model. Optionally scale features.
    Returns model, train and test predictions, and scaler used.
    """
    logger.info("Training Logistic Regression...")
    if scaler is None:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    model = LogisticRegression(**kwargs)
    model.fit(X_train_scaled, y_train)

    y_test_pred = model.predict(X_test_scaled) if X_test is not None else None

    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["precision"] = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_test, y_test_pred, average="weighted")

    logger.info(f"Logistic Regression Metrics: {metrics}\n")

    return model, metrics

def train_decision_tree(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """
    Train a Decision Tree classifier. Returns model and predictions.
    kwargs are passed to DecisionTreeClassifier.
    """
    logger.info("Training Decision Tree...")
    model = DecisionTreeClassifier(**kwargs)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test) if X_test is not None else None

    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["precision"] = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_test, y_test_pred, average="weighted")

    logger.info(f"Decision Tree Metrics: {metrics}\n")

    return model, metrics


def tune_decision_tree(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    param_grid={},
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    **kwargs
):
    """
    Perform hyperparameter tuning for DecisionTreeClassifier using GridSearchCV.
    Returns best estimator and grid search results.
    """
    logger.info("Tuning Decision Tree...")
    grid = GridSearchCV(
        DecisionTreeClassifier(**kwargs), param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs
    )
    grid.fit(X_train, y_train)
    y_test_pred = grid.predict(X_test) if X_test is not None else None

    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["precision"] = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_test, y_test_pred, average="weighted")

    logger.info(f"Tuned Decision Tree Metrics: {metrics}\n")

    return grid.best_estimator_, metrics


def train_random_forest(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """
    Train a Random Forest classifier. Returns model and predictions.
    kwargs are passed to RandomForestClassifier.
    """
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test) if X_test is not None else None

    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["precision"] = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_test, y_test_pred, average="weighted")

    logger.info(f"Random Forest Metrics: {metrics}\n")

    return model, metrics


def train_bagging_decision_tree(
    X_train, y_train, X_test=None, y_test=None, dt_params=None, bagging_params=None
):
    """
    Train a BaggingClassifier with DecisionTreeClassifier as base estimator.
    dt_params: dict for DecisionTreeClassifier
    bagging_params: dict for BaggingClassifier
    Returns model and predictions.
    """
    logger.info("Training Bagging Classifier with Decision Tree...")
    if dt_params is None:
        dt_params = {}
    if bagging_params is None:
        bagging_params = {}

    base_estimator = DecisionTreeClassifier(**dt_params)
    model = BaggingClassifier(base_estimator, **bagging_params)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test) if X_test is not None else None

    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics["accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["precision"] = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_test, y_test_pred, average="weighted")

    logger.info(f"Bagging Decision Tree Metrics: {metrics}\n")

    return model, metrics
