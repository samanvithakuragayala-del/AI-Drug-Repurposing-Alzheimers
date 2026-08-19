import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("ml_dataset.csv")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Identify feature columns
# --------------------------------------------------

fingerprint_columns = [
    col for col in df.columns
    if col.startswith("FP_")
]

target_columns = [
    col for col in df.columns
    if col.startswith("TARGET_")
]

feature_columns = fingerprint_columns + target_columns

X = df[feature_columns]
y = df["interaction_label"]

groups = df["molecule_chembl_id"]

print("Fingerprint features:", len(fingerprint_columns))
print("Target features:", len(target_columns))
print("Total ML features:", len(feature_columns))

print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Grouped train/test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")

# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1-score :  {f1:.4f}")
print(f"ROC-AUC  :  {auc:.4f}")

# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Weak/Inactive", "Active"]
).plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 8. ROC curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 9. Save model predictions
# --------------------------------------------------

results = df.iloc[test_idx].copy()

results["predicted_probability"] = y_probability
results["predicted_class"] = y_pred

results.to_csv(
    "model_test_predictions.csv",
    index=False
)

print("\nSaved:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- model_test_predictions.csv")