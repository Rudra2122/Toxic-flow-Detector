import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score
)

os.makedirs("results", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("results/dataset.csv")

print(f"Loaded {len(df):,} rows")

# -------------------------
# FEATURES
# -------------------------

features = [
    "spread",
    "relative_spread",
    "bid_qty",
    "ask_qty",
    "top5_bid",
    "top5_ask",
    "depth_ratio",
    "imbalance",
    "microprice_dev",
    "trade_qty",
    "log_trade_qty",
    "trade_side",

    "imbalance_lag1",
    "imbalance_lag3",
    "spread_lag1",
    "microprice_lag1",
    "trade_qty_lag1",

    "rolling_imbalance_5",
    "rolling_microprice_5",
    "rolling_spread_5",
    "rolling_trade_qty_5",

    "mid_return_1",
    "mid_return_3",
    "microprice_change_1",
]

target = "toxic"

X = df[features]
y = df[target]

# -------------------------
# Chronological split
# -------------------------

split_idx = int(len(df) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("\nTrain/Test Split")
print("------------------")
print(f"Train rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

# -------------------------
# Class imbalance
# -------------------------

toxic_rate = y_train.mean()

scale_pos_weight = (
    (1 - toxic_rate) / toxic_rate
)

print("\nClass Balance")
print("------------------")
print(f"Toxic Rate: {toxic_rate:.4f}")
print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# -------------------------
# Model
# -------------------------

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.02,
    min_child_weight=5,
    gamma=1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete")

# -------------------------
# Predictions
# -------------------------

probs = model.predict_proba(X_test)[:, 1]

# Balanced threshold
THRESHOLD = 0.50

preds = (probs > THRESHOLD).astype(int)

# -------------------------
# Metrics
# -------------------------

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

print("\nModel Metrics")
print("------------------")
print(f"Threshold: {THRESHOLD}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# -------------------------
# Save predictions
# -------------------------

predictions = df.iloc[split_idx:].copy()

predictions["predicted"] = preds
predictions["probability"] = probs

predictions.to_csv(
    "results/predictions.csv",
    index=False
)

# -------------------------
# Feature importance
# -------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nTop Features")
print("------------------")
print(importance.head(15))

plt.figure(figsize=(10, 7))
plt.barh(
    importance.head(15)["feature"],
    importance.head(15)["importance"]
)
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("results/feature_importance.png")

# -------------------------
# ROC Curve
# -------------------------

fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("results/roc_curve.png")

# -------------------------
# Save model
# -------------------------

joblib.dump(
    model,
    "results/model.pkl"
)

print("\nSaved:")
print("results/model.pkl")
print("results/predictions.csv")
print("results/roc_curve.png")
print("results/feature_importance.png")
print("results/feature_importance.csv")