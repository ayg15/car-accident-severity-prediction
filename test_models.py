import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelling_functions import (
    clean_data, split_data, feature_engineer_train_test,
    train_logistic_regression, train_decision_tree, tune_decision_tree,
    train_random_forest, train_bagging_decision_tree
)

# def print_metrics(y_true, y_pred, model_name):
#     print(f"\n{model_name} Metrics:")
#     print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
#     print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
#     print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

def main():
    # Load your data here
    df = pd.read_csv('US_Accidents_Dec20_updated.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = feature_engineer_train_test(X_train, X_test)

    # Logistic Regression
    logreg_model, y_train_pred_lr, y_test_pred_lr, scaler, logmetrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    # print_metrics(y_test, y_test_pred_lr, 'Logistic Regression')

    # Decision Tree
    dt_model, y_train_pred_dt, y_test_pred_dt, dtmetrics = train_decision_tree(X_train, y_train, X_test, y_test, max_depth=13, min_samples_split=15, min_samples_leaf=15)
    # print_metrics(y_test, y_test_pred_dt, 'Decision Tree')

    # Hyperparameter tuning for Decision Tree
    param_grid = {'max_depth': [10, 13, 16], 'min_samples_split': [10, 15, 20], 'min_samples_leaf': [10, 15, 20]}
    best_dt, cv_results, best_params = tune_decision_tree(X_train, y_train, param_grid)
    y_test_pred_bestdt = best_dt.predict(X_test)
    # print_metrics(y_test, y_test_pred_bestdt, 'Tuned Decision Tree')

    # Random Forest
    rf_model, y_train_pred_rf, y_test_pred_rf, rfmetrics = train_random_forest(X_train, y_train, X_test, y_test, max_depth=13, min_samples_split=15, min_samples_leaf=15, n_jobs=-1, bootstrap=False)
    # print_metrics(y_test, y_test_pred_rf, 'Random Forest')

    # Bagging Classifier with Decision Tree
    bag_model, y_train_pred_bag, y_test_pred_bag, bagmetrics = train_bagging_decision_tree(
        X_train, y_train, X_test, y_test,
        dt_params={'max_depth': 13, 'min_samples_split': 45, 'min_samples_leaf': 10}
    )
    # print_metrics(y_test, y_test_pred_bag, 'Bagging Decision Tree')

    # Choose best model (example: by test accuracy)
    results = [
        ('Logistic Regression', logreg_model, precision_score(y_test, y_test_pred_lr, average='weighted', zero_division=0)),
        ('Decision Tree', dt_model, precision_score(y_test, y_test_pred_dt, average='weighted', zero_division=0)),
        ('Tuned Decision Tree', best_dt, precision_score(y_test, y_test_pred_bestdt, average='weighted', zero_division=0)),
        ('Random Forest', rf_model, precision_score(y_test, y_test_pred_rf, average='weighted', zero_division=0)),
        ('Bagging Decision Tree', bag_model, precision_score(y_test, y_test_pred_bag, average='weighted', zero_division=0)),
    ]
    best_name, best_model, best_precision = max(results, key=lambda x: x[2])
    print(f"\nBest Model: {best_name} (Precision: {best_precision:.4f})")
    # Save best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved to best_model.pkl")

if __name__ == '__main__':
    main()
