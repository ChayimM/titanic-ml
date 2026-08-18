# Experiment Log

## Experiment 1 — Logistic Regression baseline

### Features

- Pclass
- Sex
- Age
- Fare
- Title
- FamilySize
- IsAlone
- IsMother

### Preprocessing

- Median imputation for numerical missing values
- One-hot encoding for categorical variables

### Model

Logistic Regression

### Validation

5-fold stratified cross-validation

### Results

- Mean CV accuracy: ~0.82
- Kaggle accuracy: ~0.76

### Observation

The Kaggle score was noticeably lower than the cross-validation estimate.
This suggests that our local validation performance may not perfectly represent
performance on Kaggle's hidden test set.

### Next question

Can feature engineering, model selection, or hyperparameter tuning improve
generalization to unseen data?