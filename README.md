# Customer Churn Prediction with Random Forest

This project provides a machine learning pipeline to predict customer churn using a Random Forest classifier. It includes feature selection, model training with optional hyperparameter tuning, handling class imbalance with SMOTE, and model evaluation.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Model Saving](#model-saving)
- [Evaluation](#evaluation)

---

## Requirements

- Python 3.11+
- Libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - imbalanced-learn
    - pickle
    - os

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses a customer churn dataset (Vodafone_Customer_Churn_Sample_Dataset.csv). The dataset should contain feature columns and a target label indicating whether a customer has churned.

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

## Usage

Run the main script to train and evaluate the Random Forest model:

```bash
python train_model.py
```
This will:

1. Load and preprocess the dataset.

2. Select the most important features using XGBoost.

3. Train a Random Forest classifier.

4. Apply SMOTE to balance the classes.

5. Evaluate the model using F1 score with 10-fold cross-validation.

6. Save the trained model as models/random_forest.pkl.

7. Output predicted probabilities in the dataset.

## Pipeline Overview

### Feature Selection
- Uses `XGBClassifier` to calculate feature importances.
- Selects top features covering 90% cumulative importance.

### Model Training
- Trains `RandomForestClassifier`.
- Optional hyperparameter tuning with `GridSearchCV`.

### Handling Imbalance
- `SMOTE` is applied to oversample the minority class in training data.

### Evaluation
- Standard classification metrics are printed.
- 10-fold cross-validation F1 scores are calculated.

### Model Export
- Trained Random Forest model is saved as a pickle file: `models/random_forest.pkl`.

## Model Saving
- The trained `RandomForestClassifier` model is saved to:
- It can be loaded later for inference:
```python
import pickle

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)
```
## Evaluation

- Classification report (precision, recall, F1) is printed.

- 10-fold cross-validation F1 scores are displayed for model robustness.

- Mean F1 score across folds is calculated.

