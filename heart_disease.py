import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

df = pd.read_csv(url, header=None)

print(df.head())
df.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]
print(df.head())
df = df.replace("?",pd.NA)
print(df.isnull().sum())
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])

df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())
print(df.isnull().sum())
df["target"] = df["target"].astype(int)
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
X = df.drop("target", axis=1)
y = df["target"]
print("Features (X):")
print(X.head())

print("Target (y):")
print(y.head())
print(y.value_counts())
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Predictions
train_pred = dt.predict(X_train)
test_pred = dt.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("Decision Tree Training Accuracy:", train_accuracy)
print("Decision Tree Test Accuracy:", test_accuracy)
# Step 7: Shallow Decision Tree

dt_shallow = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_shallow.fit(X_train, y_train)

shallow_train_pred = dt_shallow.predict(X_train)
shallow_test_pred = dt_shallow.predict(X_test)

shallow_train_accuracy = accuracy_score(y_train, shallow_train_pred)
shallow_test_accuracy = accuracy_score(y_test, shallow_test_pred)

print("Shallow Decision Tree Training Accuracy:", shallow_train_accuracy)
print("Shallow Decision Tree Test Accuracy:", shallow_test_accuracy)
from sklearn.ensemble import RandomForestClassifier

# Step 8: Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Test Accuracy:", rf_accuracy)
import matplotlib.pyplot as plt

# Step 9: Feature Importance

importance = rf.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
# Step 10: XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("XGBoost Test Accuracy:", xgb_accuracy)
# Step 11: Random Forest experiments

for n in [50, 100, 200]:

    rf_test = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    rf_test.fit(X_train, y_train)

    pred = rf_test.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print("Random Forest - Trees:", n,
          "Test Accuracy:", accuracy)
    # Step 12: XGBoost experiments

xgb_settings = [
    (50, 3),
    (100, 3),
    (200, 4)
]

for n, depth in xgb_settings:

    xgb_test = XGBClassifier(
        n_estimators=n,
        max_depth=depth,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_test.fit(X_train, y_train)

    pred = xgb_test.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print("XGBoost - Trees:", n,
          "Depth:", depth,
          "Test Accuracy:", accuracy)
    # Step 13: 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

dt_cv = DecisionTreeClassifier(random_state=42)

rf_cv = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

xgb_cv = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

dt_scores = cross_val_score(dt_cv, X, y, cv=5, scoring="accuracy")
rf_scores = cross_val_score(rf_cv, X, y, cv=5, scoring="accuracy")
xgb_scores = cross_val_score(xgb_cv, X, y, cv=5, scoring="accuracy")

print("Decision Tree CV Average:", dt_scores.mean())
print("Random Forest CV Average:", rf_scores.mean())
print("XGBoost CV Average:", xgb_scores.mean())
# Step 14: Final evaluation of best model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

best_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

best_rf.fit(X_train, y_train)

best_pred = best_rf.predict(X_test)

print("Final Random Forest Accuracy:", accuracy_score(y_test, best_pred))
print("Precision:", precision_score(y_test, best_pred))
print("Recall:", recall_score(y_test, best_pred))
print("F1 Score:", f1_score(y_test, best_pred))

cm = confusion_matrix(y_test, best_pred)

print("Confusion Matrix:")
print(cm)