import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def normalize_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the Wind_Direction column to unify similar values.
    """
    wind_dir_map = {
        'Calm': 'CALM',
        'South': 'S',
        'West': 'W',
        'North': 'N',
        'East': 'E',
        'Variable': 'VAR',
        'WSW': 'SW',
        'WNW': 'NW',
        'NNW': 'NW',
        'SSE': 'SE',
        'SSW': 'SW',
        'ESE': 'SE',
        'NNE': 'NE',
        'ENE': 'NE',
    }
    df['Wind_Direction'] = df['Wind_Direction'].replace(wind_dir_map)
    return df

def clean_data(df1: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by:
    1. Normalizing wind direction
    2. Dropping columns with high missing values or low variance
    3. Converting time columns
    4. Imputing missing values by group
    5. Dropping remaining nulls and duplicates
    """
    df = df1.copy()
    df = normalize_wind_direction(df)

    # Drop columns with high missing values or not useful
    drop_cols = [
        'Number', 'Wind_Chill(F)', 'Precipitation(in)', 'Weather_Timestamp',
        'Country', 'Turning_Loop', 'ID'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Convert time columns
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='ISO8601')

    # Extract time features
    df['Start_Year'] = df['Start_Time'].dt.year
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Start_Day'] = df['Start_Time'].dt.day
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_Weekday'] = df['Start_Time'].dt.weekday

    # Fill wind speed for CALM
    if 'Wind_Direction' in df.columns and 'Wind_Speed(mph)' in df.columns:
        df.loc[(df['Wind_Direction']=='CALM'), 'Wind_Speed(mph)'] = df.loc[(df['Wind_Direction'] == 'CALM'), 'Wind_Speed(mph)'].fillna(0.0)
    
    # Impute missing values by group
    for col in ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']:
        if col in df.columns:
            df[col] = df.groupby(['State','Start_Month'], sort=False)[col].transform(lambda x: x.fillna(x.mean()))
    for col in ['Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']:
        if col in df.columns:
            df[col] = df.groupby(['State','Start_Month'], sort=False)[col].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
    
    # Drop remaining nulls and duplicates
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def split_data(
    df: pd.DataFrame,
    target_column: str = 'Severity',
    sample_size: int = 200000,
    test_size: float = 0.2,
    random_state: int = 124,
    stratify: bool = True
) -> tuple:
    """
    Shuffles, samples, and splits the DataFrame into train and test sets.
    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.
        sample_size: Number of rows to sample for modeling.
        test_size: Proportion of test set.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify by target.
    Returns:
        X_train, X_test, y_train, y_test
    """
    df2 = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if sample_size is not None and sample_size < len(df2):
        acc_df = df2.iloc[:sample_size, :].copy()
    else:
        acc_df = df2.copy()
    
    feature_cols = list(acc_df.columns)
    feature_cols.remove(target_column)

    X = acc_df[feature_cols]
    y = acc_df[target_column]

    stratify_y = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    
    return X_train, X_test, y_train, y_test

def feature_engineering(df):
    """
    Perform feature engineering on the input DataFrame.
    - Calculates time duration
    - Drops unused columns
    - Converts types and encodes categoricals
    - One-hot encodes categorical columns
    """
    df = df.copy()
    
    # Time duration in minutes
    df['Time_Duration(min)'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    df = df.drop(columns=['End_Time', 'Start_Time'], errors='ignore')
    
    drop_cols = [
        'Description', 'Street', 'County', 'State', 'End_Lat', 'End_Lng',
        'Zipcode', 'Airport_Code', 'Weather_Condition', 'Sunrise_Sunset',
        'City', 'Nautical_Twilight', 'Timezone', 'Astronomical_Twilight'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Convert types
    if 'Side' in df.columns:
        df['Side'] = df['Side'].astype('category')
    
    bool_cols = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # One-hot encoding
    cat_cols = [col for col in ['Side', 'Civil_Twilight', 'Wind_Direction'] if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, prefix=cat_cols)
    
    return df

def feature_engineer_train_test(X_train, X_test):
    """
    Apply feature engineering to both train and test sets.
    Returns transformed X_train, X_test.
    """
    X_train_fe = feature_engineering(X_train)
    X_test_fe = feature_engineering(X_test)
    return X_train_fe, X_test_fe

def train_logistic_regression(X_train, y_train, X_test=None, y_test=None, scaler=None, max_iter=200):
    """
    Train a logistic regression model. Optionally scale features.
    Returns model, train and test predictions, and scaler used.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled) if X_test is not None else None
    
    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted')
    
    return model, y_train_pred, y_test_pred, scaler, metrics

def train_decision_tree(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """
    Train a Decision Tree classifier. Returns model and predictions.
    kwargs are passed to DecisionTreeClassifier.
    """
    model = DecisionTreeClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test) if X_test is not None else None
    
    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted')
    return model, y_train_pred, y_test_pred, metrics

def tune_decision_tree(X_train, y_train, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
    """
    Perform hyperparameter tuning for DecisionTreeClassifier using GridSearchCV.
    Returns best estimator and grid search results.
    """
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.cv_results_, grid.best_params_

def get_feature_importance(model, feature_names):
    """
    Get feature importances from a fitted tree-based model.
    Returns a sorted list of (feature, importance).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    else:
        raise ValueError('Model does not have feature_importances_ attribute.')

def train_random_forest(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """
    Train a Random Forest classifier. Returns model and predictions.
    kwargs are passed to RandomForestClassifier.
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test) if X_test is not None else None
    
    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted')
    
    return model, y_train_pred, y_test_pred, metrics

def train_bagging_decision_tree(X_train, y_train, X_test=None, y_test=None, dt_params=None, bagging_params=None):
    """
    Train a BaggingClassifier with DecisionTreeClassifier as base estimator.
    dt_params: dict for DecisionTreeClassifier
    bagging_params: dict for BaggingClassifier
    Returns model and predictions.
    """
    if dt_params is None:
        dt_params = {}
    if bagging_params is None:
        bagging_params = {}
    
    base_estimator = DecisionTreeClassifier(**dt_params)
    model = BaggingClassifier(base_estimator, **bagging_params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test) if X_test is not None else None
    
    metrics = {}
    if y_test is not None and y_test_pred is not None:
        metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted')
    
    return model, y_train_pred, y_test_pred, metrics
