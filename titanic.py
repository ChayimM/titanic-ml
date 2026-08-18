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
# region Loading Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
def create_features(df):
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

    return df
train = create_features(train)
test = create_features(test)
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

test_x = test[feature_columns]
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
#endregion
#region Experimentation


print("TRAIN")
print(train[["Pclass", "Sex", "Age", "Fare"]].describe())

print("\nTEST")
print(test[["Pclass", "Sex", "Age", "Fare"]].describe())


print("\nTRAIN SEX")
print(train["Sex"].value_counts(normalize=True))

print("\nTEST SEX")
print(test["Sex"].value_counts(normalize=True))

print("\nTRAIN CLASS")
print(train["Pclass"].value_counts(normalize=True))

print("\nTEST CLASS")
print(test["Pclass"].value_counts(normalize=True))

print("\nTRAIN MISSING")
print(train.isnull().sum())

print("\nTEST MISSING")
print(test.isnull().sum())

#endregion