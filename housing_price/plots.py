import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _money(x, _pos=None):
    c = config.CURRENCY
    if abs(x) >= 1e6:
        return f"{c}{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{c}{x / 1e3:.0f}K"
    return f"{c}{x:.0f}"


def price_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[config.TARGET], kde=True, color="#4C72B0", ax=ax)
    ax.set_title("Distribution of Sale Prices")
    ax.set_xlabel("Sale price")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_money))
    return _save(fig, "housing_price_distribution.png")


def correlation_heatmap(df, top_n=12):
    # 79 features won't fit, so show the numeric columns most correlated with price
    numeric = df.select_dtypes("number").drop(columns=config.DROP_COLS, errors="ignore")
    top = numeric.corr()[config.TARGET].abs().sort_values(ascending=False).head(top_n).index
    corr = numeric[top].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Top {top_n} Features Correlated with Sale Price")
    return _save(fig, "correlation_heatmap.png")


def model_comparison(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    order = results.sort_values("R2", ascending=True)
    ax1.barh(order["Model"], order["R2"], color="#55A868")
    ax1.set_title("Cross-Validated R²  (higher is better)")
    ax1.set_xlabel("R² score")

    order_rmse = results.sort_values("RMSE", ascending=False)
    ax2.barh(order_rmse["Model"], order_rmse["RMSE"], color="#C44E52")
    ax2.set_title("Cross-Validated RMSE  (lower is better)")
    ax2.set_xlabel("RMSE")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(_money))

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "model_evaluation_plot.png")


def predicted_vs_actual(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolor="white")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="grey", label="Perfect prediction")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title(f"Predicted vs Actual - {model_name}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_money))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_money))
    ax.legend()
    return _save(fig, "predicted_vs_actual.png")


def residual_plot(y_test, y_pred, model_name):
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, color="#8172B3", edgecolor="white")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(f"Residual Plot - {model_name}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_money))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_money))
    return _save(fig, "residual_plot.png")


def feature_importance(model, model_name, top_n=15):
    # handles both tree importances and linear coefficients. The model is a
    # TransformedTargetRegressor, so the fitted pipeline is under .regressor_
    pipe = getattr(model, "regressor_", model)
    estimator = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return None

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        xlabel = "Importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
        xlabel = "Absolute coefficient"
    else:
        return None

    if len(values) != len(names):
        return None

    imp = (
        pd.Series(values, index=names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_title(f"Top Feature Importances - {model_name}")
    ax.set_xlabel(xlabel)
    return _save(fig, "feature_importance.png")
