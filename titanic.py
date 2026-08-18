import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.proportion import proportion_confint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def create_features(df, include_ticket_group=False):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r", ([A-Za-z]+)\.")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["IsMother"] = (
        (df["Sex"] == "female") &
        (df["Parch"] > 0) &
        (df["Age"] > 18) &
        (df["Title"] != "Miss")
    ).astype(int)

    if include_ticket_group:
        df["TicketGroupSize"] =  (
            df.groupby("Ticket")["Ticket"].transform("count")
        )
        df["SmallTicketGroup"] = (
            (df["TicketGroupSize"] >= 2) &
            (df["TicketGroupSize"] <= 4)
        ).astype(int)

    return df

raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

train = create_features(raw_train)
train_with_tgs = create_features(raw_train, include_ticket_group=True)

train["FamilyGroup"] = pd.cut(
    train["FamilySize"],
    bins=[0, 1, 4, float("inf")],
    labels=["Alone", "Small", "Large"]
)
train_with_tgs["TicketGroup"] = pd.cut(
    train_with_tgs["TicketGroupSize"],
    bins=[0, 1, 4, float("inf")],
    labels=["Alone", "Small", "Large"]
)

print(
    train_with_tgs.groupby("TicketGroup", observed=True)["Survived"]
    .agg(["mean", "count"])
)

print(
    train.groupby("FamilyGroup", observed=True)["Survived"]
    .agg(["mean", "count"])
)

sex_group_analysis = (
    train
    .groupby(["FamilyGroup", "Sex"])["Survived"]
    .agg(["mean", "count"])
)
class_group_analysis = (
    train
    .groupby(["FamilyGroup", "Pclass"])["Survived"]
    .agg(["mean", "count"])
)

print(class_group_analysis)

print(sex_group_analysis)

test = create_features(raw_test)
test_with_tgs = create_features(raw_test, include_ticket_group=True)

# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

train["Title"] = train["Name"].str.extract(r", ([A-Za-z]+)\.")
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
print(train[[
    "SibSp",
    "Parch",
    "FamilySize"
]].head())
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
print(
    train.groupby("IsAlone")["Survived"].mean()
)
print(
    train.groupby("FamilySize")["Survived"].mean()
)
train["IsMother"] = (
    (train["Sex"] == "female") &
    (train["Parch"] > 0) &
    (train["Age"] > 18) &
    (train["Title"] != "Miss")
).astype(int)
print(train.groupby("IsMother")["Survived"].mean())
feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Title",
    "FamilySize",
    "IsAlone",
    "IsMother"
]
feature_columns_tgs = feature_columns + [
    "TicketGroupSize",
    "SmallTicketGroup"
]
y = train["Survived"]
x = train[feature_columns]

test_x = test[feature_columns]
x_train, x_valid, y_train, y_valid = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. DEFINE PREPROCESSING
# ============================================================

numeric_features = [
    "Pclass",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "IsMother"
]
numeric_features_with_tgs = numeric_features + [
    "TicketGroupSize",
    "SmallTicketGroup"
]
categorical_features = [
    "Sex",
    "Title"
]
def create_preprocessor(numeric_features):
    numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])
preprocessor = create_preprocessor(numeric_features)
preprocessor_tgs = create_preprocessor(numeric_features_with_tgs)


# ============================================================
# 5. CREATE GRADIENT BOOSTING PIPELINE
# ============================================================

def create_logistic_model(numeric_features):
    return Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
    ])
def create_forest_model(numeric_features):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            random_state=42
        ))
    ])

model = create_logistic_model(numeric_features)

model_tgs = create_logistic_model(numeric_features_with_tgs)

forest_model = create_forest_model(numeric_features)

forest_model_tgs = create_forest_model(numeric_features_with_tgs)

model.fit(x_train, y_train)

predictions = model.predict(x_valid)
accuracy = accuracy_score(y_valid, predictions)

forest_model.fit(x_train, y_train)
forest_predictions = forest_model.predict(x_valid)
forest_accuracy = accuracy_score(
    y_valid,
    forest_predictions
)

print(f"Logistic Regression: {accuracy:.3f}")
print(f"Random Forest:       {forest_accuracy:.3f}")

print("Logistic Regression:")
print(confusion_matrix(y_valid, predictions))

print("Random Forest:")
print(confusion_matrix(y_valid, forest_predictions))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# scores = cross_val_score(
#     model,
#     x,
#     y,
#     cv=cv,
#     scoring="accuracy"
# )
# print("Cross-validation scores:", scores)
# print(f"Mean accuracy: {scores.mean():.3f}")
# print(f"Standard deviation: {scores.std():.3f}")



# forest_scores = cross_val_score(
#     forest_model,
#     x,
#     y,
#     cv=cv,
#     scoring="accuracy"
# )

# print("Random Forest scores:", forest_scores)
# print(f"Mean accuracy: {forest_scores.mean():.3f}")
# print(f"Standard deviation: {forest_scores.std():.3f}")

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]
importance = pd.Series(
    coefficients,
    index=feature_names
).sort_values()
print(
    importance.sort_values(
        key=abs,
        ascending=False
    ).head(10)
)

# final predictions
model.fit(x, y)
test_predictions = model.predict(test_x)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(submission.shape)
print(submission["Survived"].value_counts())

#region GRADIENT BOOSTING

print("\n\n\n\nGradient Boosting")
gradient_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        random_state=42
    ))
])
param_grid = {
    "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
    "classifier__max_iter": [50, 100, 150, 200]
}
grid_search = GridSearchCV(
    gradient_model,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)
grid_search.fit(x, y)
print("Best parameters:")
print(grid_search.best_params_)

print(f"Best CV score: {grid_search.best_score_:.3f}")
best_gradient_model = grid_search.best_estimator_



results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_classifier__learning_rate",
        "param_classifier__max_iter",
        "mean_test_score",
        "std_test_score"
    ]
].sort_values("mean_test_score", ascending=False)

print(results.to_string(index=False))



gradient_model.fit(x_train, y_train)

gradient_predictions = gradient_model.predict(x_valid)

gradient_accuracy = accuracy_score(
    y_valid,
    gradient_predictions
)

print(f"Gradient Boosting:   {gradient_accuracy:.3f}")

tuned_predictions = grid_search.predict(x_valid)

tuned_accuracy = accuracy_score(
    y_valid,
    tuned_predictions
)

print(f"Tuned Gradient Boosting: {tuned_accuracy:.3f}")

print("Best parameters:")
print(grid_search.best_params_)




print("\nTuned Gradient Boosting - Repeated Cross-Validation")

# gradient_scores = cross_val_score(
#     gradient_model,
#     x,
#     y,
#     cv=cv_repeated,
#     scoring="accuracy"
# )

# print(
#     f"Gradient Boosting: "
#     f"{gradient_scores.mean():.3f} "
#     f"+/- {gradient_scores.std():.3f}"
# )


#endregion
#region FINAL MODEL COMPARISON

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

models = {
    "Logistic Regression": model,
    "Random Forest": forest_model,
    "Gradient Boosting": gradient_model,
    "Tuned Gradient Boosting": grid_search.best_estimator_
}

final_results = []

# for name, current_model in models.items():

#     scores = cross_val_score(
#         current_model,
#         x,
#         y,
#         cv=cv_repeated,
#         scoring="accuracy",
#         n_jobs=-1
#     )

#     final_results.append({
#         "Model": name,
#         "Mean": scores.mean(),
#         "Std": scores.std(),
#         "Min": scores.min(),
#         "Max": scores.max()
#     })

results_df = pd.DataFrame(final_results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Mean": "{:.3f}".format,
            "Std": "{:.3f}".format,
            "Min": "{:.3f}".format,
            "Max": "{:.3f}".format
        }
    )
)
final_model = grid_search.best_estimator_

final_model.fit(x, y)

# ============================================================
# 8. GENERATE TEST PREDICTIONS
# ============================================================

final_predictions = final_model.predict(test_x)

# ============================================================
# 9. CREATE KAGGLE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

# ============================================================
# 10. SUMMARY
# ============================================================

print("\nFinal model:")
print("Tuned Gradient Boosting")

print("\nTest predictions:")
print(submission.head())

print(
    f"\nNumber of predictions: "
    f"{len(submission)}"
)

print(
    f"Predicted survival rate: "
    f"{submission['Survived'].mean():.3f}"
)

print("\nSaved submission.csv")
#endregion
print("\n" + "=" * 50)
print("RANDOM SEED EXPERIMENT")
print("=" * 50)

seeds = [1, 7, 42, 123, 999]

seed_results = []

for seed in seeds:

    cv_seed = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    gradient_model_seed = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            random_state=seed
        ))
    ])

    grid_seed = GridSearchCV(
        gradient_model_seed,
        param_grid,
        cv=cv_seed,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_seed.fit(x, y)

    print(f"\nSeed: {seed}")
    print(f"Best parameters: {grid_seed.best_params_}")
    print(f"Best CV score: {grid_seed.best_score_:.3f}")

    final_model_seed = grid_seed.best_estimator_
    final_model_seed.fit(x, y)

    seed_predictions = final_model_seed.predict(test_x)

    seed_submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": seed_predictions
    })

    filename = f"submission_seed_{seed}.csv"
    seed_submission.to_csv(filename, index=False)

    print(f"Predicted survival rate: {seed_predictions.mean():.3f}")
    print(f"Saved: {filename}")

    seed_results.append({
        "Seed": seed,
        "CV Score": grid_seed.best_score_,
        "Learning Rate": grid_seed.best_params_[
            "classifier__learning_rate"
        ],
        "Max Iter": grid_seed.best_params_[
            "classifier__max_iter"
        ],
        "Predicted Survival": seed_predictions.mean()
    })

seed_results_df = pd.DataFrame(seed_results)

print("\n" + "=" * 50)
print("TRAIN / TEST SANITY CHECK")
print("=" * 50)

print("\nTraining shape:")
print(x.shape)

print("\nTest shape:")
print(test_x.shape)

print("\nTraining survival rate:")
print(y.mean())

print("\nTest predictions:")
print(final_predictions.mean())

print("\nTraining Sex:")
print(train["Sex"].value_counts(normalize=True))

print("\nTest Sex:")
print(test["Sex"].value_counts(normalize=True))

print("\nTraining Pclass:")
print(train["Pclass"].value_counts(normalize=True))

print("\nTest Pclass:")
print(test["Pclass"].value_counts(normalize=True))

print("\nPredictions by Sex:")
print(
    test.assign(Prediction=final_predictions)
    .groupby("Sex")["Prediction"]
    .agg(["mean", "count"])
)

print("\nPredictions by Sex and Class:")
print(
    test.assign(Prediction=final_predictions)
    .groupby(["Sex", "Pclass"])["Prediction"]
    .agg(["mean", "count"])
)

print("\nSeed comparison:")
print(seed_results_df.to_string(index=False))