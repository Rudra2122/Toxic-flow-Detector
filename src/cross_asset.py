import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

print("Loading dataset...")

df = pd.read_csv("results/dataset.csv")

print(f"Loaded {len(df):,} rows")

# -------------------------
# TRAIN / TEST ASSETS
# -------------------------

train_assets = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
]

test_assets = [
    "avaxusdt",
    "linkusdt",
    "ltcusdt",
]

print("\nTrain Assets")
print(train_assets)

print("\nTest Assets")
print(test_assets)

# -------------------------
# FEATURE SET
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

# -------------------------
# SPLIT BY ASSET
# -------------------------

train_df = df[
    df["symbol"].isin(train_assets)
].copy()

test_df = df[
    df["symbol"].isin(test_assets)
].copy()

print("\nDataset Split")
print("---------------------")
print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# -------------------------
# CLASS BALANCE
# -------------------------

toxic_rate = y_train.mean()

scale_pos_weight = (
    (1 - toxic_rate)
    / toxic_rate
)

print("\nClass Balance")
print("---------------------")
print(f"Toxic Rate: {toxic_rate:.4f}")
print(
    f"Scale Pos Weight: "
    f"{scale_pos_weight:.2f}"
)

# -------------------------
# MODEL
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
# PREDICTIONS
# -------------------------

probs = model.predict_proba(
    X_test
)[:, 1]

THRESHOLD = 0.50

preds = (
    probs > THRESHOLD
).astype(int)

# -------------------------
# OVERALL METRICS
# -------------------------

auc = roc_auc_score(
    y_test,
    probs
)

precision = precision_score(
    y_test,
    preds
)

recall = recall_score(
    y_test,
    preds
)

f1 = f1_score(
    y_test,
    preds
)

print("\nOverall Metrics")
print("---------------------")
print(f"ROC-AUC: {auc:.4f}")
print(
    f"Precision: "
    f"{precision:.4f}"
)
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

# -------------------------
# PER-ASSET RESULTS
# -------------------------

print("\nPer Asset Results")
print("---------------------")

results = []

for asset in test_assets:

    asset_df = test_df[
        test_df["symbol"] == asset
    ]

    X_asset = asset_df[features]
    y_asset = asset_df[target]

    probs_asset = model.predict_proba(
        X_asset
    )[:, 1]

    preds_asset = (
        probs_asset > THRESHOLD
    ).astype(int)

    auc_asset = roc_auc_score(
        y_asset,
        probs_asset
    )

    precision_asset = precision_score(
        y_asset,
        preds_asset,
        zero_division=0
    )

    recall_asset = recall_score(
        y_asset,
        preds_asset,
        zero_division=0
    )

    f1_asset = f1_score(
        y_asset,
        preds_asset,
        zero_division=0
    )

    results.append({
        "asset": asset,
        "auc": auc_asset,
        "precision": precision_asset,
        "recall": recall_asset,
        "f1": f1_asset
    })

    print(f"\n{asset.upper()}")
    print(
        f"AUC: {auc_asset:.4f}"
    )
    print(
        f"Precision: "
        f"{precision_asset:.4f}"
    )
    print(
        f"Recall: "
        f"{recall_asset:.4f}"
    )
    print(
        f"F1: "
        f"{f1_asset:.4f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/cross_asset_results.csv",
    index=False
)

print("\nSaved:")
print(
    "results/"
    "cross_asset_results.csv"
)
