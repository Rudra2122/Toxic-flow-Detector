import pandas as pd
import numpy as np
import os

os.makedirs("results", exist_ok=True)

print("Loading data...")

df = pd.read_csv("data/orderbook.csv")

print(f"Loaded {len(df):,} rows")

# ---------------------------
# Clean data
# ---------------------------

df = df.dropna()

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

df = df.sort_values(
    ["symbol", "timestamp"]
).reset_index(drop=True)

print("Data cleaned")

# ---------------------------
# Horizon labeling
# ---------------------------

# 5 snapshots = roughly 500ms
HORIZON = 3

df["future_mid"] = (
    df.groupby("symbol")["mid"]
    .shift(-HORIZON)
)

df["future_return"] = (
    df["future_mid"] - df["mid"]
)

# ---------------------------
# Directional toxicity label
# ---------------------------

# Binance buyer_is_maker convention:
# 1 means buyer is maker, so sell aggressor
# 0 means buyer is taker, so buy aggressor
#
# Convert:
#  1 = buy aggressor
# -1 = sell aggressor

df["trade_side"] = np.where(
    df["trade_side"] == 1,
    -1,
    1
)

df["signed_return"] = (
    df["future_return"] * df["trade_side"]
)

threshold = df["spread"] * 0.5

df["toxic"] = np.where(
    df["signed_return"] > threshold,
    1,
    0
)

print("Labels created")

# ---------------------------
# Feature engineering
# ---------------------------

df["depth_ratio"] = (
    df["top5_bid"] / (df["top5_ask"] + 1e-9)
)

df["log_trade_qty"] = np.log1p(
    df["trade_qty"]
)

df["relative_spread"] = (
    df["spread"] / (df["mid"] + 1e-9)
)

# Lag features

df["imbalance_lag1"] = (
    df.groupby("symbol")["imbalance"]
    .shift(1)
)

df["imbalance_lag3"] = (
    df.groupby("symbol")["imbalance"]
    .shift(3)
)

df["spread_lag1"] = (
    df.groupby("symbol")["spread"]
    .shift(1)
)

df["microprice_lag1"] = (
    df.groupby("symbol")["microprice_dev"]
    .shift(1)
)

df["trade_qty_lag1"] = (
    df.groupby("symbol")["trade_qty"]
    .shift(1)
)

# Rolling pressure features

df["rolling_imbalance_5"] = (
    df.groupby("symbol")["imbalance"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

df["rolling_microprice_5"] = (
    df.groupby("symbol")["microprice_dev"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

df["rolling_spread_5"] = (
    df.groupby("symbol")["spread"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

df["rolling_trade_qty_5"] = (
    df.groupby("symbol")["trade_qty"]
    .rolling(5)
    .mean()
    .reset_index(level=0, drop=True)
)

# Momentum style features

df["mid_return_1"] = (
    df.groupby("symbol")["mid"]
    .diff(1)
)

df["mid_return_3"] = (
    df.groupby("symbol")["mid"]
    .diff(3)
)

df["microprice_change_1"] = (
    df.groupby("symbol")["microprice_dev"]
    .diff(1)
)

print("Features created")

# ---------------------------
# Final dataset
# ---------------------------

features = [
    "symbol",

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

    "future_return",
    "toxic",
]

dataset = (
    df[features]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

print("\nDataset Stats")
print("-------------------")
print(f"Rows: {len(dataset):,}")
print(f"Toxic rate: {dataset['toxic'].mean():.2%}")
print(f"Horizon: {HORIZON} snapshots")

dataset.to_csv(
    "results/dataset.csv",
    index=False
)

print("\nSaved to:")
print("results/dataset.csv")