import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


NUMERIC_IMPUTE_COLUMNS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)",
]
CATEGORICAL_IMPUTE_COLUMNS = [
    "Wind_Direction", "Weather_Condition", "Sunrise_Sunset", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight",
]

PREDICTION_TIME = "first report (Start_Time)"
EXCLUDED_OUTCOME_COLUMNS = [
    "End_Time", "End_Lat", "End_Lng", "Time_Duration(min)", "Distance(mi)",
    "Description",
]
ROAD_FEATURE_COLUMNS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]
MODEL_FEATURE_COLUMNS = [
    "Start_Lat", "Start_Lng", "Start_Year", "Start_Month", "Start_Day",
    "Start_Hour", "Start_Weekday", "Side", "Civil_Twilight", "Wind_Direction",
] + NUMERIC_IMPUTE_COLUMNS + ROAD_FEATURE_COLUMNS
PREPROCESSOR_INPUT_COLUMNS = ["State", "Start_Time"] + MODEL_FEATURE_COLUMNS


def _prediction_inputs(df):
    """Allow only reviewed inputs; unknown/new columns never become features."""
    return df.loc[:, [col for col in df if col in PREPROCESSOR_INPUT_COLUMNS]].copy()


def _mode_or_missing(values):
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else np.nan


class GroupImputer(TransformerMixin, BaseEstimator):
    """Learn state/month means and modes exclusively from the fit data.

    Unknown or empty groups use the training-wide mean/mode. Entirely missing
    training columns fall back to zero (numeric) or 'Unknown' (categorical).
    Transform preserves rows and never learns from the prediction batch.
    """

    def fit(self, X, y=None):
        self.statistics_ = {}
        self.fallbacks_ = {}
        for col in NUMERIC_IMPUTE_COLUMNS + CATEGORICAL_IMPUTE_COLUMNS:
            if col not in X.columns:
                continue
            numeric = col in NUMERIC_IMPUTE_COLUMNS
            aggregate = "mean" if numeric else _mode_or_missing
            self.statistics_[col] = X.groupby(
                ["State", "Start_Month"], observed=True
            )[col].agg(aggregate)
            fallback = X[col].mean() if numeric else _mode_or_missing(X[col])
            self.fallbacks_[col] = (
                (0.0 if numeric else "Unknown") if pd.isna(fallback) else fallback
            )
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_")
        result = X.copy()
        keys = pd.MultiIndex.from_frame(result[["State", "Start_Month"]])
        for col, statistics in self.statistics_.items():
            fill_values = pd.Series(statistics.reindex(keys).to_numpy(), index=X.index)
            result[col] = result[col].fillna(fill_values).fillna(self.fallbacks_[col])
        return result


class FeaturePreprocessor(TransformerMixin, BaseEstimator):
    """Fit imputation and one categorical encoder per training fold.

    Transform reuses the fitted encoder and preserves training feature order.
    Unseen categories map to zeros; all known categories retain a column.
    """

    def fit(self, X, y=None):
        X = _prediction_inputs(X)
        self.prediction_time_ = PREDICTION_TIME
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.input_dtypes_ = {col: str(dtype) for col, dtype in X.dtypes.items()}
        self.imputer_ = GroupImputer().fit(X)
        features = feature_engineering(self.imputer_.transform(X))
        self.categorical_columns_ = [
            col
            for col in ["Side", "Civil_Twilight", "Wind_Direction"]
            if col in features.columns
        ]
        self.numeric_columns_ = [
            col for col in features if col not in self.categorical_columns_
        ]
        self.encoder_ = None
        encoded_names = []
        if self.categorical_columns_:
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop=None
            )
            self.encoder_.fit(features[self.categorical_columns_])
            encoded_names = self.encoder_.get_feature_names_out().tolist()
        self.feature_names_out_ = np.asarray(
            self.numeric_columns_ + encoded_names, dtype=object
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        X = _prediction_inputs(X)
        features = feature_engineering(self.imputer_.transform(X))
        numeric = features[self.numeric_columns_]
        if self.encoder_ is None:
            return numeric.copy()
        encoded = pd.DataFrame(
            self.encoder_.transform(features[self.categorical_columns_]),
            columns=self.encoder_.get_feature_names_out(),
            index=features.index,
        )
        return pd.concat([numeric, encoded], axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def normalize_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the Wind_Direction column to unify similar values.
    """
    wind_dir_map = {
        "Calm": "CALM",
        "South": "S",
        "West": "W",
        "North": "N",
        "East": "E",
        "Variable": "VAR",
        "WSW": "SW",
        "WNW": "NW",
        "NNW": "NW",
        "SSE": "SE",
        "SSW": "SW",
        "ESE": "SE",
        "NNE": "NE",
        "ENE": "NE",
    }
    df["Wind_Direction"] = df["Wind_Direction"].replace(wind_dir_map)
    return df


def clean_data(df1: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by:
    1. Normalizing wind direction
    2. Removing unused columns before missing-value filtering
    3. Converting time columns
    4. Preserving missing values handled by the fitted GroupImputer
    5. Dropping other nulls and duplicates

    Only deterministic, row-level cleaning happens before splitting the data.
    """
    allowed = PREPROCESSOR_INPUT_COLUMNS + ["Severity", "Weather_Timestamp"]
    df = df1.loc[:, [col for col in df1 if col in allowed]].copy()
    input_counts = df["Severity"].value_counts(dropna=False) if "Severity" in df else None

    # Convert time columns
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="ISO8601")

    # A weather observation after the report cannot be used at prediction time.
    # Without a usable timestamp, treat the observation as unavailable too.
    if "Weather_Timestamp" in df:
        weather_time = pd.to_datetime(
            df["Weather_Timestamp"], format="ISO8601", errors="coerce"
        )
        available = weather_time.notna() & weather_time.le(df["Start_Time"])
    else:
        available = pd.Series(False, index=df.index)
    weather_columns = [
        col for col in NUMERIC_IMPUTE_COLUMNS + ["Wind_Direction"] if col in df
    ]
    for col in weather_columns:
        df[col] = df[col].where(available)
    df = df.drop(columns="Weather_Timestamp", errors="ignore")
    if "Wind_Direction" in df:
        df = normalize_wind_direction(df)

    # Extract time features
    df["Start_Year"] = df["Start_Time"].dt.year
    df["Start_Month"] = df["Start_Time"].dt.month
    df["Start_Day"] = df["Start_Time"].dt.day
    df["Start_Hour"] = df["Start_Time"].dt.hour
    df["Start_Weekday"] = df["Start_Time"].dt.weekday

    # Fill wind speed for CALM
    if "Wind_Direction" in df.columns and "Wind_Speed(mph)" in df.columns:
        df.loc[(df["Wind_Direction"] == "CALM"), "Wind_Speed(mph)"] = df.loc[
            (df["Wind_Direction"] == "CALM"), "Wind_Speed(mph)"
        ].fillna(0.0)

    # Learned imputation belongs after splitting, inside the estimator pipeline.
    imputed_columns = NUMERIC_IMPUTE_COLUMNS + CATEGORICAL_IMPUTE_COLUMNS
    required_columns = [
        col for col in ["Start_Time", "State", "Severity"] + MODEL_FEATURE_COLUMNS
        if col in df and col not in imputed_columns
    ]
    df = df.dropna(subset=required_columns)
    after_missing = len(df)
    df = df.drop_duplicates()
    logger.info(
        "Cleaning retention: %d/%d rows; dropped %d with missing required values, %d duplicates",
        len(df), len(df1), len(df1) - after_missing, after_missing - len(df),
    )
    if input_counts is not None:
        retained_counts = df["Severity"].value_counts(dropna=False)
        for label, count in input_counts.items():
            retained = int(retained_counts.get(label, 0))
            logger.info("Severity %s retention: %d/%d (%.1f%%)",
                        label, retained, count, 100 * retained / count)
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = "Severity",
    sample_size: int = 200000,
    test_size: float = 0.2,
    random_state: int = 124,
    stratify: bool = True,
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
    - Keeps only features reviewed for availability at the first report
    - Converts boolean types while leaving categorical columns unencoded

    Use FeaturePreprocessor for fitted categorical encoding shared by splits.
    """
    df = df.loc[:, [col for col in df if col in MODEL_FEATURE_COLUMNS]].copy()

    # Convert types
    if "Side" in df.columns:
        df["Side"] = df["Side"].astype("category")

    for col in ROAD_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def feature_engineer_train_test(X_train, X_test):
    """
    Fit preprocessing on training rows, then transform both sets.
    For cross-validation, use FeaturePreprocessor inside the model pipeline
    instead, so each fold learns its own imputation statistics.
    Returns transformed X_train, X_test.
    """
    preprocessor = FeaturePreprocessor()
    X_train_fe = preprocessor.fit_transform(X_train)
    X_test_fe = preprocessor.transform(X_test)
    return X_train_fe, X_test_fe


def get_feature_importance(model, feature_names):
    """
    Get feature importances from a fitted tree-based model.
    Returns a sorted list of (feature, importance).
    """
    if hasattr(model, "named_steps"):
        if "preprocessor" in model.named_steps:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        model = model.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")
