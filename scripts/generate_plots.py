# master_thesis/scripts/generate_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(x=y_true, y=y_pred)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot"):

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))

    sns.scatterplot(x=y_pred, y=residuals)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(results_df):

    metrics = ["MAE", "MSE", "R2"]

    for metric in metrics:

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=results_df,
            x="Model",
            y=metric,
        )

        plt.xticks(rotation=45)
        plt.title(f"{metric} Comparison")
        plt.tight_layout()
        plt.show()


def plot_feature_importance(model, feature_names, top_n=10):

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model does not provide feature_importances_."
        )

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices],
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.show()
