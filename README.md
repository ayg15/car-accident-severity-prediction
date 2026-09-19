# Car Accidents Analysis & Severity Prediction

Accidents are a major cause of traffic congestion and identifying where and why the number of accidents is higher and why there are more severe accidents helps in taking precautionary steps in better traffic management and this eventually helps in reducing the number of accidents. 

The aim of this project is to predict the severity class of the car accident based on various environmental factors, location and traffic conditions. The severity prediction can help improve the traffic safety and also can help to understand the reason for the accidents.


The dataset used here is a [US Accident Dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents) from Kaggle. This dataset covers car accidents records from 49 US states with the data collected between February 2016 to December 2020.

## Project Workflow

### 1. Data Cleaning & Preprocessing (`modelling_helper_functions.py`)
- **normalize_wind_direction**: Standardizes wind direction values.
- **clean_data**: Removes unused columns before checking missing values, performs deterministic cleaning, converts time columns, and removes duplicates. Missing values in imputed columns are preserved until model fitting. Logs overall retention, missing-value/duplicate removals, and retained counts/percentages by severity.
- **GroupImputer**: Learns numeric means and categorical modes by state/month from training rows only. Unseen or empty groups use training-wide statistics; entirely missing training columns use `0` or `Unknown`.
- **FeaturePreprocessor**: Fits imputation and one `OneHotEncoder(handle_unknown="ignore")` on training rows. Validation, test, and prediction data reuse that encoder with identical output columns and order. Each GridSearchCV fold fits its own preprocessor. All known categories retain a column; unseen categories produce zeros for that feature's encoded columns.
- **split_data**: Shuffles, samples, and splits the dataset into train and test sets (default: 200,000 samples, 80/20 split).
- **feature_engineering**: Extracts time-based features, removes unused columns, and converts boolean types. Categorical columns remain unencoded until `FeaturePreprocessor` applies its fitted encoder.
- **feature_engineer_train_test**: Convenience helper that fits preprocessing on training rows and transforms both splits. For cross-validation, use `FeaturePreprocessor` inside the estimator pipeline instead.

### 2. Model Training (`model_training.py`)
- **select_best_model**: Compares all four model families using the same shuffled, stratified training-only CV folds and balanced accuracy. Tunes tree/forest parameters, evaluates fixed logistic/bagging configurations, and returns the best fitted pipeline plus CV results. All candidates use the full training split.
- **evaluate_model**: Computes final holdout metrics for the selected winner with one prediction call, including per-class results and a majority-class baseline learned from training labels.
- The individual `train_*` helpers below remain available for standalone experiments; the main selection workflow uses `select_best_model`.
- **train_logistic_regression**: Trains a logistic regression model (optionally with feature scaling) and evaluates metrics.
- **train_decision_tree_with_grid_search**: Trains a Decision Tree with hyperparameter tuning using GridSearchCV.
- **train_random_forest_with_grid_search**: Tunes a Random Forest using GridSearchCV on all training rows and refits the best complete pipeline on all training rows.
- **train_bagging_decision_tree**: Trains a Bagging Classifier with Decision Tree as the base estimator.

### 3. Model Selection & Saving (`train.py`)
- **get_the_best_model**: Orchestrates the workflow:
	- Loads and cleans the data
	- Splits and engineers features
	- Trains all models above
	- Selects the winner using mean training-CV balanced accuracy
	- Evaluates only the selected winner once on the untouched test set
	- Saves the complete fitted prediction pipeline and metadata as `best_model.pkl`

### 4. Testing & Evaluation
- Final test performance includes accuracy, balanced accuracy, weighted precision/recall/F1, macro F1, training/test class counts, per-class precision/recall/F1/support, and a confusion matrix.
- Confusion-matrix rows represent true classes and columns represent predicted classes, in the saved `labels` order. Macro F1 includes all classes seen in training, testing, or predictions; undefined class scores are zero. Balanced accuracy averages recall over classes present in the evaluated labels.
- `majority_baseline` reports the same metrics for a classifier that always predicts the most frequent **training** class. The test distribution never determines the baseline class. This comparison is diagnostic and does not affect model selection.
- CV scores and best parameters are saved separately in `metadata_["training_config"]["cv_results"]`; `metadata_["metrics"]` contains only the winner's final test metrics. CV selection scores are not unbiased performance estimates.
- Test results do not change the winner or its parameters. Repeated tuning based on that final report would require a fresh holdout.
- Both the main workflow and standalone random-forest helper tune/refit on all training rows. This can take longer than the former 20,000-row subset.
- Feature importance can be extracted for tree-based models.

## How to Run

Supported Python version: **3.14** (tested with Python 3.14.5). Use the same
Python and dependency versions when reloading saved models.

Create an environment and install runtime dependencies on Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional dependency groups:

- `python -m pip install -r requirements-dev.txt` adds Black. Tests use standard-library `unittest`.
- `python -m pip install -r requirements-notebook.txt` adds IPython, the notebook kernel, Matplotlib, and Seaborn. Both files include runtime dependencies.

1. Install the dependencies as above.
2. Place the dataset file (`US_Accidents_Dec20_updated.csv`) in the project directory.
3. Run the main workflow:
   ```bash
   python train.py
   ```
   This logs CV scores for all candidates, evaluates the winner once, and writes
   `best_model.pkl` and `best_model.report.json`.

Customize a run:

```powershell
python train.py --dataset "data/US_Accidents_Dec20_updated.csv" --sample-size 200000 --seed 42 --output "artifacts/severity.pkl" --report "artifacts/run.json" --cv 5 --n-jobs -1
```

Use `--sample-size all` for every cleaned row. `--seed` controls sampling, the
train/test split, shuffled CV folds, and all stochastic estimators, including
bagging. The default seed is now 42 throughout (the old split seed was 124).
If `--report` is omitted, the report uses the model path with `.report.json` as
its suffix. Output directories are created automatically. `python train.py --help`
lists all options. A repeated output path replaces that run's artifacts.

The JSON report includes each model's mean/std CV score, winning fold scores,
best parameters, all searched parameter combinations and their mean/std scores,
per-model training duration, overall training/selection duration, run settings,
and the winner's final evaluation (including baseline and per-class results).
Durations use seconds; per-model training includes CV search and the final refit.
`through_evaluation` also includes input loading and cleaning, but excludes file
writing. Reports are strict JSON and can be read without loading a pickle.

## Prediction-Time Feature Policy

The model assumes prediction at the first accident report, using `Start_Time`
as the cutoff. This is a modeling assumption: the dataset does not prove when
each field was actually delivered to a live prediction system.

| Inputs | Policy |
| --- | --- |
| `End_Time`, `Time_Duration(min)` | Excluded; the accident has not ended at the prediction cutoff. Neither field is required. |
| `End_Lat`, `End_Lng`, `Distance(mi)`, `Description` | Excluded conservatively because the recorded extent or narrative may reflect later updates; initial-report availability has not been established. |
| Start coordinates, `State`, `Side` | Assumed available from the initial report/location lookup. State supports imputation and is not a model feature. |
| Year, month, day, hour, weekday | Derived only from `Start_Time`. |
| Road/POI flags and `Civil_Twilight` | Assumed available from existing map data or time/location calculations. Production inputs must use map information available at the cutoff. |
| Temperature, humidity, pressure, visibility, wind speed/direction | Retained only when `Weather_Timestamp <= Start_Time`. Later, missing, or invalid timestamps cause weather values to be masked before training-fitted imputation. |
| Other or newly added columns | Excluded by an explicit feature allowlist until reviewed. |

Run `clean_data` before training or prediction to apply the timestamp check.
Weather and start timestamps must use the same time basis. Observation time is
only a necessary condition: production weather feeds must also ensure the
observation was received by the prediction cutoff. This repository cannot verify
delivery latency or historical map availability from the CSV alone.

Excluded fields are removed before missing-value filtering, so absent end data
does not discard otherwise usable examples. Saved artifacts record this policy
and use artifact version 2; retrain older models to apply the new feature set.

## File Descriptions
- `modelling_helper_functions.py`: Data cleaning, preprocessing, and feature engineering functions.
- `model_training.py`: Model training functions for various classifiers.
- `model_persistence.py`: Save/load helpers for complete fitted pipelines and training metadata.
- `train.py`: Main script to run the full pipeline and select the best model.
- `US_Accidents_Dec20_updated.csv`: Input dataset.
- `requirements.txt`: Runtime dependencies.
- `requirements-dev.txt`: Development tools plus runtime dependencies.
- `requirements-notebook.txt`: Notebook/plotting tools plus runtime dependencies.

## Notes
- The pipeline is designed for reproducibility and modularity.
- Model hyperparameter grids live in `select_best_model` in `model_training.py`; run settings are CLI options in `train.py`.
- Library modules retain module loggers and do not configure root logging on import. Logging is configured once in `train.main`.
- The best model is saved as a pickle file for later use.
- Training functions return fitted pipelines. The main workflow supplies `FeaturePreprocessor`; predictions then accept the same cleaned, unencoded feature columns returned by `split_data`. Logistic regression also retains its fitted scaler. A supplied scaler is cloned and fitted on training data.
- The existing `best_model.pkl` is not updated until training is rerun.

## Loading a Saved Pipeline

After rerunning training, load the pipeline and predict on cleaned, unencoded
features. The artifact contains the fitted imputer, categorical encoder, optional
scaler, and classifier; do not refit or separately encode/scale prediction data.

```python
import pandas as pd
from model_persistence import load_prediction_pipeline
from modelling_helper_functions import clean_data

pipeline = load_prediction_pipeline("best_model.pkl")
rows = pd.read_csv("new_accidents.csv")
features = clean_data(rows.drop(columns="Severity", errors="ignore"))
predictions = pd.Series(pipeline.predict(features), index=features.index)
print(predictions)
print(pipeline.metadata_)
```

`clean_data` remains a separate deterministic preparation step and may remove
incomplete or duplicate rows. Predictions correspond to the retained rows.
`metadata_` records input columns/dtypes, encoded feature names, class labels,
dependency versions, fitted estimator/scaler parameters, split configuration,
training row counts, evaluation metrics, prediction-time policy, and creation time.

Keep the project's Python modules available and use the recorded dependency
versions when loading. Load only trusted pickle files. Older estimator-only
artifacts must be regenerated by rerunning training.

## Regression Tests

Run `python -m unittest discover -s tests -v` to check training-only imputation, fold isolation, consistent categorical encoding, unseen categories, fallback handling, prediction/serialization, holdout isolation, random-forest fitting beyond 20,000 rows, minority-class metrics, the training-derived baseline, and retention with missing unused columns.

Tests also cover CLI configuration, bagging reproducibility, logging import
behavior, and writing/loading a pipeline alongside its JSON report. Test
artifacts use temporary directories and do not replace `best_model.pkl`.
