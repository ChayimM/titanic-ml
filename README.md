# Titanic — Machine Learning & Statistical Analysis

A machine learning and data analysis project based on the Kaggle Titanic dataset.

The project began as an introduction to Python and machine learning, but developed into an exploration of feature engineering, model comparison, cross-validation, hyperparameter optimization, and statistical analysis.

## Overview

The central question was simple:

> Can we predict whether a Titanic passenger survived?

I built several classification models and compared their performance using stratified cross-validation. Alongside the modelling process, I investigated relationships in the data through hypothesis-driven analysis.

The project was deliberately treated as an experiment rather than a race for the highest Kaggle score.

### Main technologies

- Python
- pandas
- scikit-learn
- statsmodels
- matplotlib

- Git / GitHub
- Kaggle

## Project Structure
.
├── README.md
├── experiments.md
├── titanic.py
├── final_model_comparison.csv
└── gradient_gridsearch.csv

final_model_comparison.csv: Summary of the final model comparison using repeated stratified cross-validation.
gradient_gridsearch.csv:    Results from the Gradient Boosting hyperparameter search.
experiments.md:             Detailed research log containing the hypotheses, experiments, results, interpretations, and limitations encountered throughout the project.

## What I learned
What I enjoyed most was hypothesizing about relationships in the data and then testing whether the evidence actually supported them.

Through this process, I learned a great deal about what makes a good analysis: questioning my assumptions, testing them systematically, and avoiding conclusions that the data does not support.

The project also taught me that model evaluation is not simply about finding the algorithm with the highest score. A model that performs well on one validation split may perform considerably worse when evaluated repeatedly.

## Key Lessons
1. Validation matters

A single validation split produced an accuracy of approximately 0.894 for the tuned Gradient Boosting model.

Repeated cross-validation estimated its performance at approximately 0.834 ± 0.020.

The difference demonstrated why a single split can give an overly optimistic impression of model performance.

2. Feature engineering matters

Features such as Title, FamilySize, IsAlone, and IsMother provided useful information that was not directly available in the original variables.

3. More complex models are not automatically better

Random Forest did not outperform Logistic Regression on this dataset.

Gradient Boosting performed somewhat better, but the improvement remained relatively modest.

4. Relationships need context

The apparent relationship between group size and survival changed substantially when passenger class and sex were taken into account.

This was one of the most useful analytical lessons from the project.

## Notes
This project is primarily an educational and analytical exercise.

The Titanic dataset is small and historical, so the results should not be interpreted as evidence of general principles applicable to other datasets.

The full experimental process, including unsuccessful hypotheses and intermediate results, is documented in experiments.md