import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score
from modelling_helper_functions import (
    clean_data,
    split_data,
    feature_engineer_train_test,
)
from classification_modelling import (
    train_logistic_regression,
    train_decision_tree_with_grid_search,
    train_random_forest_with_grid_search,
    train_bagging_decision_tree,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_the_best_model():

    df = pd.read_csv("US_Accidents_Dec20_updated.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = feature_engineer_train_test(X_train, X_test)

    # Logistic Regression
    logreg_model, lrmetrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, max_iter=500, random_state=42
    )

    # Decision Tree
    param_grid = {
        "max_depth": [3, 5, 7, 9, 11, 13],
        "min_samples_split": [10, 15, 30, 45, 60],
        "min_samples_leaf": [10, 15, 30, 45, 60],
    }
    best_dt, bestdt_metrics = train_decision_tree_with_grid_search(
        X_train, 
        y_train, 
        X_test=X_test, 
        y_test=y_test, 
        param_grid=param_grid, 
        cv=5, 
        scoring='balanced_accuracy',
        random_state=42
    )

    # Random Forest
    param_grid = {
        'max_depth': [11, 13],
        'min_samples_split': [15, 30],
        'min_samples_leaf': [15, 30],
        'bootstrap': [False]
    }
    rf_model, rfmetrics = train_random_forest_with_grid_search(
        X_train,
        y_train,
        X_test,
        y_test,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy',
        random_state=42
    )

    # Bagging Classifier with Decision Tree
    bag_model, bagmetrics = train_bagging_decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
        dt_params={
            "max_depth": 13,
            "min_samples_split": 45,
            "min_samples_leaf": 10,
        },
    )

    # Choose best model (example: by test accuracy)
    results = [
        (
            "Logistic Regression",
            logreg_model,
            lrmetrics.get("precision", 0),
        ),
        (
            "Decision Tree",
            best_dt,
            bestdt_metrics.get("precision", 0),
        ),
        (
            "Random Forest",
            rf_model,
            rfmetrics.get("precision", 0),
        ),
        (
            "Bagging Decision Tree",
            bag_model,
            bagmetrics.get("precision", 0),
        ),
    ]
    best_name, best_model, best_precision = max(results, key=lambda x: x[2])
    logger.info(f"\nBest Model: {best_name} (Precision: {best_precision:.4f})")

    # Save best model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    logger.info("Best model saved to best_model.pkl")


if __name__ == "__main__":
    get_the_best_model()
