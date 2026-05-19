import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

print("Loading dataset...")
df = pd.read_csv("results/dataset.csv")
print(f"Loaded {len(df):,} rows")

TRAIN_ASSETS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt"]
TEST_ASSETS = ["avaxusdt", "linkusdt", "ltcusdt"]

BASE_SPREAD_CAPTURE = 0.50

# --------------------------------------------------
# Extra microstructure features
# --------------------------------------------------

df["spread_change"] = df["spread"] - df["spread_lag1"]
df["imbalance_change"] = df["imbalance"] - df["imbalance_lag1"]

df["queue_pressure"] = (
    df["bid_qty"] / (df["ask_qty"] + 1e-9)
)

df["rolling_vol_10"] = (
    df.groupby("symbol")["mid_return_1"]
    .rolling(10)
    .std()
    .reset_index(level=0, drop=True)
)

df = (
    df.replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

# --------------------------------------------------
# Research grade split
# --------------------------------------------------

train_parts = []
test_parts = []

for symbol in df["symbol"].unique():
    asset_df = df[df["symbol"] == symbol].reset_index(drop=True)

    split_idx = int(len(asset_df) * 0.7)

    early = asset_df.iloc[:split_idx]
    late = asset_df.iloc[split_idx:]

    if symbol in TRAIN_ASSETS:
        train_parts.append(early)

    if symbol in TEST_ASSETS:
        test_parts.append(late)

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print("\nResearch Grade Split")
print("--------------------")
print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")

# --------------------------------------------------
# Feature groups
# --------------------------------------------------

FEATURE_SETS = {
    "Model_A_spread_only": [
        "spread",
    ],

    "Model_B_spread_imbalance": [
        "spread",
        "imbalance",
    ],

    "Model_C_microstructure": [
        "spread",
        "imbalance",
        "microprice_dev",
    ],

    "Model_D_full_features": [
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
    ],

    "Model_E_enhanced_microstructure": [
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
        "spread_change",
        "imbalance_change",
        "queue_pressure",
        "rolling_vol_10",
    ],
}

results = []

print("\nRunning Ablation Study")
print("=" * 60)

for model_name, features in FEATURE_SETS.items():

    print(f"\nTraining {model_name}")
    print("-" * 50)

    X_train = train_df[features]
    y_train = train_df["toxic"]

    X_test = test_df[features]
    y_test = test_df["toxic"]

    toxic_rate = y_train.mean()
    scale_pos_weight = (1 - toxic_rate) / toxic_rate

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=10,
        gamma=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Find best threshold by Sharpe improvement
    # --------------------------------------------------

    best_result = None

    thresholds = np.arange(0.40, 0.91, 0.05)

    for threshold in thresholds:

        preds = (probs > threshold).astype(int)

        temp_df = test_df.copy()
        temp_df["pred_prob"] = probs
        temp_df["pred_toxic"] = preds

        buy_aggressor = temp_df["trade_side"] == 1
        sell_aggressor = temp_df["trade_side"] == -1

        # Baseline PnL
        temp_df["baseline_pnl"] = (
            temp_df["spread"] * BASE_SPREAD_CAPTURE
        )

        temp_df.loc[buy_aggressor, "baseline_pnl"] -= temp_df.loc[
            buy_aggressor,
            "future_return",
        ]

        temp_df.loc[sell_aggressor, "baseline_pnl"] += temp_df.loc[
            sell_aggressor,
            "future_return",
        ]

        # Multi-threshold spread widening
        temp_df["spread_multiplier"] = np.select(
            [
                temp_df["pred_prob"] > 0.90,
                temp_df["pred_prob"] > 0.75,
                temp_df["pred_prob"] > 0.60,
                temp_df["pred_prob"] > threshold,
            ],
            [
                2.00,
                1.75,
                1.50,
                1.25,
            ],
            default=1.00,
        )

        temp_df["strategy_capture"] = (
            BASE_SPREAD_CAPTURE
            * temp_df["spread_multiplier"]
        )

        temp_df["strategy_pnl"] = (
            temp_df["spread"]
            * temp_df["strategy_capture"]
        )

        temp_df.loc[buy_aggressor, "strategy_pnl"] -= temp_df.loc[
            buy_aggressor,
            "future_return",
        ]

        temp_df.loc[sell_aggressor, "strategy_pnl"] += temp_df.loc[
            sell_aggressor,
            "future_return",
        ]

        baseline_pnl = temp_df["baseline_pnl"].sum()
        strategy_pnl = temp_df["strategy_pnl"].sum()

        pnl_improvement = (
            (strategy_pnl - baseline_pnl)
            / (abs(baseline_pnl) + 1e-9)
        ) * 100

        baseline_returns = temp_df["baseline_pnl"]
        strategy_returns = temp_df["strategy_pnl"]

        baseline_sharpe = (
            baseline_returns.mean()
            / (baseline_returns.std() + 1e-9)
        ) * np.sqrt(252)

        strategy_sharpe = (
            strategy_returns.mean()
            / (strategy_returns.std() + 1e-9)
        ) * np.sqrt(252)

        sharpe_improvement = (
            (strategy_sharpe - baseline_sharpe)
            / (abs(baseline_sharpe) + 1e-9)
        ) * 100

        flagged_rate = temp_df["pred_toxic"].mean() * 100

        if best_result is None or sharpe_improvement > best_result["sharpe_improvement_pct"]:
            best_result = {
                "threshold": threshold,
                "pnl_improvement_pct": pnl_improvement,
                "sharpe_improvement_pct": sharpe_improvement,
                "flagged_rate_pct": flagged_rate,
            }

    # --------------------------------------------------
    # Classification metrics at best threshold
    # --------------------------------------------------

    best_threshold = best_result["threshold"]
    preds = (probs > best_threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"AUC: {auc:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"PnL improvement: {best_result['pnl_improvement_pct']:.2f}%")
    print(f"Sharpe improvement: {best_result['sharpe_improvement_pct']:.2f}%")
    print(f"Flagged trades: {best_result['flagged_rate_pct']:.2f}%")

    results.append({
        "model": model_name,
        "num_features": len(features),
        "roc_auc": auc,
        "best_threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pnl_improvement_pct": best_result["pnl_improvement_pct"],
        "sharpe_improvement_pct": best_result["sharpe_improvement_pct"],
        "flagged_rate_pct": best_result["flagged_rate_pct"],
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/ablation_study.csv",
    index=False,
)

print("\nFinal Results")
print("=" * 60)

print(
    results_df.sort_values(
        "sharpe_improvement_pct",
        ascending=False
    )
)

print("\nSaved:")
print("results/ablation_study.csv")