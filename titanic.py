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
# region Loading Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# endregion
# region Deciding useful variable types
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
y = train["Survived"]
x = train[feature_columns]
# endregion
# region Splitting training and validation
# neither x_train nor x_valid has been modified
x_train, x_valid, y_train, y_valid = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
numeric_features = [
    "Pclass",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "IsMother"
]
categorical_features = [
    "Sex",
    "Title"
]
# endregion
# region Creating preprocessor
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])
# endregion
# region Creating Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
# region Creating Forest Model
forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        random_state=42
    ))
])
# endregion
# endregion
# region Loading and Output
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

scores = cross_val_score(
    model,
    x,
    y,
    cv=15,
    scoring="accuracy"
)
print("Cross-validation scores:", scores)
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")

forest_scores = cross_val_score(
    forest_model,
    x,
    y,
    cv=15,
    scoring="accuracy"
)

print("Random Forest scores:", forest_scores)
print(f"Mean accuracy: {forest_scores.mean():.3f}")
print(f"Standard deviation: {forest_scores.std():.3f}")
#endregion