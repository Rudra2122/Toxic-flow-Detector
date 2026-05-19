import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

os.makedirs("results", exist_ok=True)

LABEL_MAP = {
    "Model_A_spread_only": "Spread Only",
    "Model_B_spread_imbalance": "Spread + Imbalance",
    "Model_C_microstructure": "Microstructure Features",
    "Model_D_full_features": "Full Feature Set",
    "Model_E_enhanced_microstructure": "Enhanced Microstructure",
}


def add_bar_labels(bars, suffix="", decimals=3):
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# -------------------------
# 1. ROC Curve
# -------------------------

backtest = pd.read_csv("results/backtest_results.csv")

y_true = backtest["toxic"]
y_prob = backtest["pred_prob"]

auc = roc_auc_score(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"ROC AUC = {auc:.3f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Toxic Flow Classifier ROC Curve")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# -------------------------
# 2. PnL Comparison
# -------------------------

baseline_pnl = backtest["baseline_pnl"].sum()
strategy_pnl = backtest["strategy_pnl"].sum()

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Baseline MM", "Toxic Flow Strategy"],
    [baseline_pnl, strategy_pnl],
)

add_bar_labels(
    bars,
    decimals=1,
)

plt.ylabel("Total Simulated PnL")
plt.title("PnL Comparison")

plt.tight_layout()

plt.savefig(
    "results/pnl_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# -------------------------
# 3. Cumulative PnL Curve
# -------------------------

backtest["baseline_cum_pnl"] = backtest["baseline_pnl"].cumsum()
backtest["strategy_cum_pnl"] = backtest["strategy_pnl"].cumsum()

plt.figure(figsize=(9, 5))

plt.plot(
    backtest["baseline_cum_pnl"],
    label="Baseline MM",
    linewidth=2.5,
    alpha=0.8,
)

plt.plot(
    backtest["strategy_cum_pnl"],
    label="Toxic Flow Strategy",
    linewidth=2.5,
)

plt.xlabel("Trade Index")
plt.ylabel("Cumulative PnL")

plt.title("Cumulative Strategy Performance")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/pnl_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# -------------------------
# 4. Feature Importance
# -------------------------

fi_path = "results/feature_importance.csv"

if os.path.exists(fi_path):
    fi = pd.read_csv(fi_path)

    fi = (
        fi.sort_values(
            "importance",
            ascending=False,
        )
        .head(15)
    )

    plt.figure(figsize=(9, 6))

    bars = plt.barh(
        fi["feature"],
        fi["importance"],
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.grid(axis="x", alpha=0.3)

    for bar in bars:
        value = bar.get_width()

        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

else:
    print(
        "Skipping feature importance: "
        "results/feature_importance.csv not found"
    )


# -------------------------
# 5. Ablation AUC Chart
# -------------------------

ablation = pd.read_csv("results/ablation_study.csv")

ablation["label"] = (
    ablation["model"]
    .map(LABEL_MAP)
    .fillna(ablation["model"])
)

plt.figure(figsize=(10, 5))

bars = plt.bar(
    ablation["label"],
    ablation["roc_auc"],
)

add_bar_labels(
    bars,
    decimals=3,
)

plt.xticks(rotation=20, ha="right")
plt.ylabel("ROC AUC")
plt.title("Feature Ablation Study")
plt.ylim(0, max(ablation["roc_auc"]) + 0.08)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/ablation_auc.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# -------------------------
# 6. Ablation PnL Chart
# -------------------------

# Exclude Model A from PnL chart because it flagged 100%
# of trades and behaves like a trivial/non-actionable baseline.
ablation_pnl = ablation[
    ablation["model"] != "Model_A_spread_only"
].copy()

plt.figure(figsize=(10, 5))

bars = plt.bar(
    ablation_pnl["label"],
    ablation_pnl["pnl_improvement_pct"],
)

add_bar_labels(
    bars,
    suffix="%",
    decimals=1,
)

plt.xticks(rotation=20, ha="right")
plt.ylabel("PnL Improvement (%)")
plt.title("Trading Impact by Feature Set")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/ablation_pnl.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("Graphs saved:")
print("results/roc_curve.png")
print("results/pnl_comparison.png")
print("results/pnl_curve.png")
print("results/feature_importance.png")
print("results/ablation_auc.png")
print("results/ablation_pnl.png")