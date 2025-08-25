import os
import pickle
from typing import List

import pandas as pd
from imblearn.over_sampling import SMOTE
from numpy import mean
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from xgboost import XGBClassifier

from src.preprocessing import select_important_columns, load_and_preprocess_dataset


def determine_important_columns(
    dataset_features: pd.DataFrame, dataset_lables: pd.Series
) -> List[str]:
    """
    Determine the most important feature columns using an XGBoost classifier.

    Steps performed:
    1. Split the dataset into training and testing sets (80%/20%).
    2. Train an XGBClassifier on the training set.
    3. Retrieve feature importances from the trained model.
    4. Round importances to 3 decimal places and sort in descending order.
    5. Select features whose cumulative importance does not exceed 0.9
       using `select_important_columns`.

    Parameters
    ----------
    dataset_features : pd.DataFrame
        DataFrame containing feature columns.
    dataset_labels : pd.Series
        Series containing target labels.

    Returns
    -------
    List[str]
        List of feature column names selected as important.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_features, dataset_lables, test_size=0.20, random_state=0
    )
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # get importance
    importances_sk = model.feature_importances_
    feature_importance_sk = {}
    columns = X_train.columns.tolist()
    for i, column in enumerate(columns):
        feature_importance_sk[column] = round(importances_sk[i], 3)
    feature_importance_sk = sorted(
        feature_importance_sk.items(), key=lambda x: x[1], reverse=True
    )
    feature_importance_sk = select_important_columns(feature_importance_sk, 0.9)
    important_columns = [set[0] for set in feature_importance_sk]
    return important_columns

def fit_rf_model(
        grid_search_cv: bool,
        random_forest: RandomForestClassifier,
        X_train: DataFrame,
        y_train: Series
) -> RandomForestClassifier:
    """
    Fit a Random Forest model, optionally using GridSearchCV for hyperparameter tuning.

    Parameters
    ----------
    grid_search_cv : bool
        If True, perform GridSearchCV to find the best hyperparameters.
    random_forest : RandomForestClassifier
        An instance of RandomForestClassifier to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Target values for training.

    Returns
    -------
    RandomForestClassifier
        The trained Random Forest model. If `grid_search_cv` is True,
        this will be the best estimator from the grid search; otherwise,
        it will be the original `random_forest` fitted on the training data.
    """
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    if grid_search_cv:
        # GridSearchCV with F1 scoring
        grid_search = GridSearchCV(random_forest, param_grid, scoring='f1', cv=5, n_jobs=1)
        grid_search.fit(X_train, y_train)

        # Best model
        best_rf = grid_search.best_estimator_

        return best_rf
    else:
        random_forest.fit(X_train, y_train)

    return random_forest



def main():
    grid_search_cv = False
    (
        dataset,
        dataset_features,
        dataset_lables,
    ) = load_and_preprocess_dataset("data/Vodafone_Customer_Churn_Sample_Dataset.csv")
    important_columns = determine_important_columns(dataset_features, dataset_lables)
    filtered_dataset = dataset[important_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        filtered_dataset, dataset_lables, test_size=0.20, random_state=4
    )
    random_forest = RandomForestClassifier(
        class_weight="balanced", n_estimators=500, random_state=4
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    random_forest = fit_rf_model(grid_search_cv=grid_search_cv, random_forest=random_forest, X_train=X_train, y_train=y_train)

    if grid_search_cv:
        # Predict probabilities on test set
        y_prob = random_forest.predict_proba(X_test)[:, 1]

        # Adjust threshold to improve minority class recall
        threshold = 0.4
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = random_forest.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))

    # cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(
        random_forest, filtered_dataset, dataset_lables, cv=kf, scoring="f1"
    )
    for fold, score in enumerate(scores):
        print(f"Fold {fold}: F1 = {score:.2f}")

    # Calculate and print the mean and standard deviation of the scores
    print(f"Mean F1: {mean(scores):.2f}")

    pkl_filename = "models/random_forest.pkl"

    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)

    with open(pkl_filename, "wb") as file:
        pickle.dump(random_forest, file)
    probabilities = random_forest.predict_proba(filtered_dataset)
    filtered_dataset.loc[:, "pd"] = probabilities[:, 1]


if __name__ == "__main__":
    main()
