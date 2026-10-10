
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. LOAD DATA
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "sheets" / "gamma.csv")

FEATURES = ["FP1-FZ", "FP2-FZ", "T5-O1", "T6-O2"]
TARGET = "group"

required = ["subject", TARGET] + FEATURES
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

# Keep only ASD and TD rows.
df = df[df[TARGET].isin(["ASD", "TD"])].copy()

# Convert features to numeric and flag invalid/missing values.
df[FEATURES] = df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)

# Remove rows with no usable feature values at all.
df = df.dropna(subset=FEATURES, how="all")

X = df[FEATURES]
y = df[TARGET]

print("Dataset size:", len(df))
print("\nClass counts:")
print(y.value_counts())

# ============================================================
# 2. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

print("\nTraining subjects:", len(X_train))
print("Testing subjects:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ============================================================
# 3. BUILD AND FIT RANDOM FOREST
# ============================================================

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(X_train, y_train)

# ============================================================
# 4. EVALUATE ON HELD-OUT SUBJECTS
# ============================================================

y_pred = model.predict(X_test)

print("\n===== HELD-OUT TEST RESULTS =====")
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred):.3f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["ASD", "TD"],
        zero_division=0,
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(
    pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=["ASD", "TD"]),
        index=["True ASD", "True TD"],
        columns=["Predicted ASD", "Predicted TD"],
    )
)

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

rf = model.named_steps["randomforestclassifier"]

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

importance.to_csv(
    ROOT / "sheets" / "gamma_feature_importance.csv",
    index=False,
)

# ============================================================
# 6. SAVE HELD-OUT PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "subject": df.loc[X_test.index, "subject"].values,
    "actual_group": y_test.values,
    "predicted_group": y_pred,
})

predictions.to_csv(
    ROOT / "sheets" / "gamma_test_predictions.csv",
    index=False,
)

print("\nSaved feature importance and test predictions.")