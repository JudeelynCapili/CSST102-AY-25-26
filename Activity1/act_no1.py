# ==============================================================
# Machine Problem No. 1 - Fundamentals of Machine Learning
# Course: CSST102 - Basic Machine Learning
# Academic Year: 2025–2026
# ==============================================================
# Author: [Your Name]
# Section: [Your Section]
# ==============================================================

# Import libraries
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    mean_squared_error,
)
import numpy as np

# ==============================================================
# ========== PART 1: IRIS DATASET (CLASSIFICATION) ==========
# ==============================================================

print("==============================================================")
print("PART 1: IRIS DATASET (Classification Problem)")
print("==============================================================")

# Load dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target, name="species")

# Dataset exploration
print("\n===== DATASET EXPLORATION =====")
print("Shape of dataset:", X_iris.shape)
print("\nFirst 5 rows of features:\n", X_iris.head())
print("\nTarget labels:", iris.target_names)
print("\nFeature names:", iris.feature_names)

print("\nMini-task Answers:")
print("Input (features): Sepal length, Sepal width, Petal length, Petal width")
print("Output (label): Species of Iris (Setosa, Versicolor, Virginica)")
print("Type of Learning: Supervised Learning (Classification)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Model training
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
conf = confusion_matrix(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("\nConfusion Matrix:\n", conf)

# Reflection
print("\n===== REFLECTION (Classification) =====")
print("1. ML Type Used: Supervised Learning - Classification")
print("2. Possible Challenge: Overfitting if dataset is small or unbalanced.")
print("3. Missing/wrong values can cause inaccurate predictions.")
print("4. Data quality and preprocessing are critical for real-world ML.\n")

# ==============================================================
# ========== PART 2: CALIFORNIA HOUSING (REGRESSION) ==========
# ==============================================================

print("==============================================================")
print("PART 2: CALIFORNIA HOUSING DATASET (Regression Problem)")
print("==============================================================")

# Load dataset
housing = fetch_california_housing()
X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
y_house = pd.Series(housing.target, name="MedianHouseValue")

# Dataset exploration
print("\n===== DATASET EXPLORATION =====")
print("Shape of dataset:", X_house.shape)
print("\nFirst 5 rows of features:\n", X_house.head())
print("\nFeature names:", housing.feature_names)
print("\nMini-task Answers:")
print("Input (features):", ", ".join(housing.feature_names))
print("Output (label): Median House Value")
print("Type of Learning: Supervised Learning (Regression)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n===== MODEL EVALUATION =====")
print("Root Mean Squared Error (RMSE):", round(rmse, 3))

# Reflection
print("\n===== REFLECTION (Regression) =====")
print("1. ML Type Used: Supervised Learning - Regression")
print("2. Possible Challenge: Underfitting if model is too simple or data not normalized.")
print("3. Missing/outlier values can greatly affect predictions.")
print("4. In real-world ML, preprocessing and feature scaling improve regression performance.")
print("==============================================================")
