import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

print("Loading dataset...")

df = pd.read_csv("results/dataset.csv")

print(f"Loaded {len(df):,} rows")

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

train_parts = []
test_parts = []

print("\nPer Asset Chronological Split")
print("-----------------------------")

for symbol in df["symbol"].unique():

    asset_df = df[
        df["symbol"] == symbol
    ].copy()

    asset_df = asset_df.sort_index()

    split_idx = int(len(asset_df) * 0.7)

    early_asset = asset_df.iloc[:split_idx]
    late_asset = asset_df.iloc[split_idx:]

    if symbol in train_assets:
        train_parts.append(early_asset)

    if symbol in test_assets:
        test_parts.append(late_asset)

    print(
        f"{symbol:<10} | "
        f"rows={len(asset_df):,} | "
        f"train={len(early_asset):,} | "
        f"test={len(late_asset):,}"
    )

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print("\nResearch Grade Split")
print("--------------------")
print(f"Train assets: {train_assets}")
print(f"Test assets: {test_assets}")
print("Split type: per-asset chronological")
print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")

print("\nFuture test asset row counts")
print(test_df["symbol"].value_counts())

X_train = train_df[features]
y_train = train_df["toxic"]

X_test = test_df[features]
y_test = test_df["toxic"]

toxic_rate = y_train.mean()

scale_pos_weight = (
    (1 - toxic_rate)
    / toxic_rate
)

print("\nClass Balance")
print("--------------------")
print(f"Toxic rate: {toxic_rate:.4f}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

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
    n_jobs=-1,
)

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training complete")

probs = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.50

preds = (
    probs > THRESHOLD
).astype(int)

auc = roc_auc_score(
    y_test,
    probs
)

precision = precision_score(
    y_test,
    preds,
    zero_division=0
)

recall = recall_score(
    y_test,
    preds,
    zero_division=0
)

f1 = f1_score(
    y_test,
    preds,
    zero_division=0
)

print("\nOverall Research Grade Metrics")
print("------------------------------")
print(f"Threshold: {THRESHOLD}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

results = []

print("\nPer Asset Future Holdout Results")
print("--------------------------------")

for asset in test_assets:

    asset_df = test_df[
        test_df["symbol"] == asset
    ].copy()

    if len(asset_df) == 0:
        print(f"\n{asset.upper()}")
        print("No rows found. Skipping.")
        continue

    y_asset = asset_df["toxic"]

    if y_asset.nunique() < 2:
        print(f"\n{asset.upper()}")
        print("Only one class present.")
        continue

    X_asset = asset_df[features]

    probs_asset = (
        model.predict_proba(X_asset)[:, 1]
    )

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
        "f1": f1_asset,
        "rows": len(asset_df),
    })

    print(f"\n{asset.upper()}")
    print(f"Rows: {len(asset_df):,}")
    print(f"AUC: {auc_asset:.4f}")
    print(f"Precision: {precision_asset:.4f}")
    print(f"Recall: {recall_asset:.4f}")
    print(f"F1: {f1_asset:.4f}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/cross_asset_chrono_results.csv",
    index=False
)

print("\nSaved:")
print(
    "results/cross_asset_chrono_results.csv"
)