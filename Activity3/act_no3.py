# ==============================================================
# Machine Problem No. 2 - Evaluating ML Model Performance
# Topic: Logistic Regression (Classification)
# Dataset: California Housing (converted to binary classification)
# Course: CSST102 - Basic Machine Learning
# Academic Year: 2025–2026
# ==============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

# --------------------------------------------------------------
# STEP 1: DATA LOADING AND PREPARATION
# --------------------------------------------------------------
print("=== Loading and preparing the California Housing dataset ===")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Create binary target: 1 = Above median, 0 = Below or equal to median
median_value = df['MedHouseVal'].median()
df['HighValue'] = (df['MedHouseVal'] > median_value).astype(int)

X = df.drop(['MedHouseVal', 'HighValue'], axis=1)
y = df['HighValue']

# Handle missing values (not needed here, but good practice)
X = X.fillna(X.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------------
# STEP 2: TRAIN-TEST SPLIT
# --------------------------------------------------------------
print("\n=== Performing Train-Test Split (80/20) ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# --------------------------------------------------------------
# STEP 3: MODEL BUILDING - LOGISTIC REGRESSION
# --------------------------------------------------------------
print("\n=== Training Logistic Regression model ===")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Testing Accuracy: {test_acc:.3f}")

# --------------------------------------------------------------
# STEP 4: CROSS-VALIDATION (5-Fold)
# --------------------------------------------------------------
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

with open("cross_validation.txt", "w") as f:
    f.write("5-Fold Cross Validation Results\n")
    f.write("---------------------------------\n")
    f.write(f"Scores: {cv_scores}\n")
    f.write(f"Mean Accuracy: {cv_mean:.3f}\n")
    f.write(f"Standard Deviation: {cv_std:.3f}\n")

# --------------------------------------------------------------
# STEP 5: CONFUSION MATRIX & METRICS
# --------------------------------------------------------------
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Low", "High"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (California Housing)")
plt.savefig("confusion_matrix.png")
plt.close()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# --------------------------------------------------------------
# STEP 6: LEARNING CURVE VISUALIZATION
# --------------------------------------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    model, X_scaled, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")
plt.title("Learning Curve - Logistic Regression (California Housing)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("learning_curve.png")
plt.close()

# --------------------------------------------------------------
# STEP 7: PERFORMANCE COMPARISON VISUALIZATION
# --------------------------------------------------------------
metrics = ["Training Accuracy", "Testing Accuracy", "CV Mean Accuracy"]
values = [train_acc, test_acc, cv_mean]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FF9800"])
plt.ylim(0, 1)
plt.title("Model Performance Comparison - Logistic Regression")
plt.ylabel("Accuracy Score")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
             f"{bar.get_height():.3f}", ha="center", color="white", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("comparison.png")
plt.close()

# --------------------------------------------------------------
# STEP 8: FINAL SUMMARY
# --------------------------------------------------------------
print("\n===== SUMMARY REPORT =====")
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Testing Accuracy: {test_acc:.3f}")
print(f"Cross-Validation Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")
print("==============================================================")
