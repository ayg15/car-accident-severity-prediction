# Car Accidents Analysis & Severity Prediction

Accidents are a major cause of traffic congestion and identifying where and why the number of accidents is higher and why there are more severe accidents helps in taking precautionary steps in better traffic management and this eventually helps in reducing the number of accidents. 

The aim of this project is to predict the severity class of the car accident based on various environmental factors, location and traffic conditions. The severity prediction can help improve the traffic safety and also can help to understand the reason for the accidents.


The dataset used here is a [US Accident Dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents) from Kaggle. This dataset covers car accidents records from 49 US states with the data collected between February 2016 to December 2020.

## Project Workflow

### 1. Data Cleaning & Preprocessing (`modelling_helper_functions.py`)
- **normalize_wind_direction**: Standardizes wind direction values.
- **clean_data**: Drops irrelevant columns, handles missing values, converts time columns, and removes duplicates.
- **split_data**: Shuffles, samples, and splits the dataset into train and test sets (default: 200,000 samples, 80/20 split).
- **feature_engineering**: Extracts time-based features, encodes categorical variables, and prepares data for modeling.
- **feature_engineer_train_test**: Applies feature engineering to both train and test sets.

### 2. Model Training (`model_training.py`)
- **train_logistic_regression**: Trains a logistic regression model (optionally with feature scaling) and evaluates metrics.
- **train_decision_tree_with_grid_search**: Trains a Decision Tree with hyperparameter tuning using GridSearchCV.
- **train_random_forest_with_grid_search**: Trains a Random Forest with hyperparameter tuning (GridSearchCV, subsamples for speed).
- **train_bagging_decision_tree**: Trains a Bagging Classifier with Decision Tree as the base estimator.

### 3. Model Selection & Saving (`test_models.py`)
- **get_the_best_model**: Orchestrates the workflow:
	- Loads and cleans the data
	- Splits and engineers features
	- Trains all models above
	- Compares models by precision
	- Saves the best model as `best_model.pkl`

### 4. Testing & Evaluation
- Model performance is evaluated using accuracy, precision, recall, and F1-score (weighted).
- Feature importance can be extracted for tree-based models.

## How to Run

1. Ensure all dependencies in `requirements.txt` are installed (preferably in a virtual environment).
2. Place the dataset file (`US_Accidents_Dec20_updated.csv`) in the project directory.
3. Run the main workflow:
   ```bash
   python test_models.py
   ```
   This will output metrics for all models and save the best one.

## File Descriptions
- `modelling_helper_functions.py`: Data cleaning, preprocessing, and feature engineering functions.
- `model_training.py`: Model training functions for various classifiers.
- `test_models.py`: Main script to run the full pipeline and select the best model.
- `US_Accidents_Dec20_updated.csv`: Input dataset.
- `requirements.txt`: Python dependencies.

## Notes
- The pipeline is designed for reproducibility and modularity.
- Hyperparameters for models can be adjusted in `test_models.py`.
- The best model is saved as a pickle file for later use.
