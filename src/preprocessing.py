from typing import List, Tuple, Dict

import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor


def select_important_columns(
        feature_importance_sk: List[Tuple[str, float]], threshold: float
) -> List[Tuple[str, float]]:
    """
    Select the most important features based on cumulative importance
    until a given threshold is reached.

    The function assumes the input list of feature importances is
    pre-sorted in descending order by importance.

    Parameters
    ----------
    feature_importance_sk : List[Tuple[str, float]]
        A list of (feature_name, importance_score) tuples, sorted by
        importance in descending order.
    threshold : float
        The cumulative importance threshold (between 0 and 1).
        Features are selected until their cumulative sum exceeds this value.

    Returns
    -------
    List[Tuple[str, float]]
        A list of (feature_name, importance_score) tuples representing
        the selected features whose cumulative importance does not
        exceed the threshold.
    """
    selected_sum = 0.0
    selected_items: List[Tuple[str, float]] = []

    # Iterate through the sorted items
    for key, value in feature_importance_sk:
        if selected_sum + value <= threshold:
            selected_sum += value
            selected_items.append((key, value))
        else:
            break

    return selected_items


def custom_map_categorical_data(df: pd.DataFrame):
    """
    Encode Yes/No style columns into numeric values.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with encoded columns
    """

    # custom map the categorical data
    object_cols: List = list(df.select_dtypes(include="object"))

    for column in object_cols:
        unique_values = df[f"{column}"].unique()
        column_dict: Dict = {key: value for value, key in enumerate(unique_values)}
        df[f"{column}"] = df[f"{column}"].map(column_dict)

    return df

def fill_in_regression_nan(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Fill missing values (NaNs) in a numeric target column using a
    Random Forest regression model trained on the available data.

    The function:
    - Splits the dataframe into rows with and without missing values
      in the target column.
    - Uses all other numeric columns (excluding the target) as features.
    - Trains a RandomForestRegressor on rows where the target is not NaN.
    - Predicts missing target values for rows where the target is NaN.
    - Returns the dataframe with NaNs in the target column filled.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the target column with missing values.
    target_col : str
        The name of the numeric column to impute using regression.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN values in `target_col` replaced with
        predictions from the Random Forest regressor.
    """

    # Separate rows with and without NaN
    train = df[df[target_col].notna()]
    test = df[df[target_col].isna()]

    # Features (all other numeric columns except the target)
    features: List[str] = [
        col for col in df.select_dtypes(include="number").columns if col != target_col
    ]
    X_train = train[features]
    y_train = train[target_col]
    X_test = test[features]

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict missing values
    predicted_values = model.predict(X_test)

    # Fill NaNs
    df.loc[df[target_col].isna(), target_col] = predicted_values

    return df


def load_and_preprocess_dataset(
        path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load and preprocess a customer churn dataset from a CSV file.

    Steps performed:
    1. Load the dataset from the given file path.
    2. Set "customerID" as the DataFrame index.
    3. Apply categorical/boolean conversions using
       `custom_map_categorical_data` (assumed user-defined).
    4. Fill missing numeric values in the "TotalCharges" column
       using regression-based imputation (`fill_in_regression_nan`).
    5. Separate the target column ("Churn") from the features.

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        - dataset : pd.DataFrame
            The preprocessed dataset including all features and target.
        - dataset_features : pd.DataFrame
            A copy of the dataset with only features (no "Churn" column).
        - dataset_labels : pd.Series
            The target labels extracted from the "Churn" column.
    """
    dataset = pd.read_csv(path, delimiter=",", header=0)

    # Use customerID as index
    dataset = dataset.set_index("customerID")

    dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

    # Convert categorical/boolean values
    dataset = custom_map_categorical_data(dataset)

    # Fill missing values in "TotalCharges"
    dataset = fill_in_regression_nan(dataset, "TotalCharges")

    # Separate labels and features
    dataset_labels: Series = dataset.pop("Churn")
    dataset_features: DataFrame = dataset.copy()

    return dataset, dataset_features, dataset_labels