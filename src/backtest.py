import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print("Loading dataset...")

df = pd.read_csv("results/dataset.csv")

print(f"Loaded {len(df):,} rows")

FEATURES = [
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

TRAIN_ASSETS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt"]
TEST_ASSETS = ["avaxusdt", "linkusdt", "ltcusdt"]

THRESHOLD = 0.60

BASE_SPREAD_CAPTURE = 0.50
TOXIC_SPREAD_BOOST = 1.50

train_parts = []
test_parts = []

print("\nPer Asset Chronological Split")
print("--------------------------------")

for symbol in df["symbol"].unique():
    asset_df = df[df["symbol"] == symbol].reset_index(drop=True)

    split_idx = int(len(asset_df) * 0.7)

    train_asset = asset_df.iloc[:split_idx]
    test_asset = asset_df.iloc[split_idx:]

    if symbol in TRAIN_ASSETS:
        train_parts.append(train_asset)

    if symbol in TEST_ASSETS:
        test_parts.append(test_asset)

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print(f"\nTrain rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")

X_train = train_df[FEATURES]
y_train = train_df["toxic"]

X_test = test_df[FEATURES]
y_test = test_df["toxic"]

toxic_rate = y_train.mean()
scale_pos_weight = (1 - toxic_rate) / toxic_rate

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

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete")

probs = model.predict_proba(X_test)[:, 1]

test_df = test_df.copy()
test_df["pred_prob"] = probs
test_df["pred_toxic"] = (test_df["pred_prob"] > THRESHOLD).astype(int)

# ------------------------------------------------
# Baseline market-making PnL
# ------------------------------------------------

buy_aggressor = test_df["trade_side"] == 1
sell_aggressor = test_df["trade_side"] == -1

test_df["baseline_pnl"] = test_df["spread"] * BASE_SPREAD_CAPTURE

test_df.loc[buy_aggressor, "baseline_pnl"] -= test_df.loc[
    buy_aggressor,
    "future_return",
]

test_df.loc[sell_aggressor, "baseline_pnl"] += test_df.loc[
    sell_aggressor,
    "future_return",
]

# ------------------------------------------------
# Strategy: toxicity-aware spread widening
# ------------------------------------------------

test_df["strategy_spread_capture"] = np.where(
    test_df["pred_prob"] > THRESHOLD,
    BASE_SPREAD_CAPTURE * TOXIC_SPREAD_BOOST,
    BASE_SPREAD_CAPTURE,
)

test_df["strategy_pnl"] = (
    test_df["spread"] * test_df["strategy_spread_capture"]
)

test_df.loc[buy_aggressor, "strategy_pnl"] -= test_df.loc[
    buy_aggressor,
    "future_return",
]

test_df.loc[sell_aggressor, "strategy_pnl"] += test_df.loc[
    sell_aggressor,
    "future_return",
]

# ------------------------------------------------
# Metrics
# ------------------------------------------------

baseline_pnl = test_df["baseline_pnl"].sum()
strategy_pnl = test_df["strategy_pnl"].sum()

baseline_returns = test_df["baseline_pnl"].fillna(0)
strategy_returns = test_df["strategy_pnl"].fillna(0)

baseline_sharpe = (
    baseline_returns.mean()
    / (baseline_returns.std() + 1e-9)
) * np.sqrt(252)

strategy_sharpe = (
    strategy_returns.mean()
    / (strategy_returns.std() + 1e-9)
) * np.sqrt(252)

pnl_improvement = (
    (strategy_pnl - baseline_pnl)
    / (abs(baseline_pnl) + 1e-9)
) * 100

sharpe_improvement = (
    (strategy_sharpe - baseline_sharpe)
    / (abs(baseline_sharpe) + 1e-9)
) * 100

flagged_trades = test_df["pred_toxic"].mean() * 100

auc = roc_auc_score(y_test, probs)

print("\nBacktest Results")
print("--------------------------------")
print(f"ROC-AUC: {auc:.4f}")
print(f"Threshold: {THRESHOLD}")
print(f"Toxic spread boost: {TOXIC_SPREAD_BOOST}")

print("\nPnL")
print(f"Baseline: {baseline_pnl:.4f}")
print(f"Strategy: {strategy_pnl:.4f}")
print(f"Improvement: {pnl_improvement:.2f}%")

print("\nSharpe Ratio")
print(f"Baseline: {baseline_sharpe:.4f}")
print(f"Strategy: {strategy_sharpe:.4f}")
print(f"Improvement: {sharpe_improvement:.2f}%")

print("\nRisk Control")
print(f"Predicted toxic trades: {flagged_trades:.2f}%")

test_df.to_csv(
    "results/backtest_results.csv",
    index=False,
)

print("\nSaved:")
print("results/backtest_results.csv")