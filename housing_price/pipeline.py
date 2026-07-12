"""End-to-end pipeline: load → explore → cross-validate → evaluate → report.

Run it with ``python main.py`` (or ``python -m housing_price.pipeline``).
"""

from __future__ import annotations

import pandas as pd

from . import config, plots
from .data import load_data, split_features_target, train_test_data
from .evaluate import cross_validate_models, evaluate_on_test
from .models import get_models


def _fmt_money(x: float) -> str:
    return f"₹{x:,.0f}"


def _write_results(cv_results: pd.DataFrame, best_name: str, test_metrics: dict) -> None:
    """Persist a machine-readable CSV and a human-readable markdown report."""
    config.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(config.RESULTS_CSV, index=False)

    lines = [
        "# Housing Price Prediction — Results",
        "",
        "All models share a leak-free scikit-learn pipeline: preprocessing "
        "(scaling + one-hot encoding) is fitted inside each cross-validation "
        "fold, and the target `price` is kept in rupees so every error metric "
        "is interpretable.",
        "",
        f"- **Dataset:** {config.RAW_DATA.name} (545 homes, 12 features)",
        f"- **Validation:** {config.CV_FOLDS}-fold cross-validation + a "
        f"{int(config.TEST_SIZE * 100)}% held-out test set",
        "",
        "## Cross-Validated Model Comparison",
        "",
        "| Model | R² | RMSE (₹) | MAE (₹) |",
        "|-------|----|----------|---------|",
    ]
    for _, r in cv_results.iterrows():
        lines.append(
            f"| {r['Model']} | {r['R2']:.3f} ± {r['R2_std']:.3f} "
            f"| {r['RMSE']:,.0f} | {r['MAE']:,.0f} |"
        )

    lines += [
        "",
        f"**Best model:** {best_name}",
        "",
        "## Held-Out Test Performance",
        "",
        f"Evaluated on the unseen {int(config.TEST_SIZE * 100)}% test split:",
        "",
        f"- **R²:** {test_metrics['r2']:.3f}",
        f"- **RMSE:** {_fmt_money(test_metrics['rmse'])}",
        f"- **MAE:** {_fmt_money(test_metrics['mae'])}",
        "",
        "## Figures",
        "",
        "![Model comparison](figures/model_evaluation_plot.png)",
        "![Predicted vs actual](figures/predicted_vs_actual.png)",
        "![Residual plot](figures/residual_plot.png)",
        "![Feature importance](figures/feature_importance.png)",
        "![Price distribution](figures/housing_price_distribution.png)",
        "![Correlation heatmap](figures/correlation_heatmap.png)",
        "",
        "## Takeaways",
        "",
        "- Keeping the target in rupees and fitting preprocessing inside each "
        "fold gives honest, directly readable error metrics.",
        "- Unregularised polynomial regression is numerically unstable on binary "
        "dummy features; pairing it with L2 regularisation (Ridge) makes it "
        "well-behaved.",
        "- On this dataset the relationship is largely linear: regularised "
        "linear models (Ridge/Lasso) match or slightly beat the tree ensembles, "
        "and `area`, `bathrooms` and `airconditioning` are among the most "
        "influential features.",
        "",
    ]
    config.RESULTS_MD.write_text("\n".join(lines))


def run() -> None:
    print("Housing Price Prediction — pipeline\n" + "=" * 40)

    # 1. Load ---------------------------------------------------------------
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {config.RAW_DATA.name}")

    # 2. Exploratory figures ------------------------------------------------
    plots.price_distribution(df)
    plots.correlation_heatmap(df)
    print("Saved exploratory figures.")

    # 3. Cross-validate every candidate model -------------------------------
    X, y = split_features_target(df)
    models = get_models()
    print(f"\nCross-validating {len(models)} models ({config.CV_FOLDS}-fold)...")
    cv_results = cross_validate_models(models, X, y)
    print(cv_results[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    best_name = cv_results.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    # 4. Held-out test evaluation for the best model ------------------------
    X_train, X_test, y_train, y_test = train_test_data(df)
    test_metrics = evaluate_on_test(best_model, X_train, y_train, X_test, y_test)
    print(
        f"Held-out test — R²: {test_metrics['r2']:.3f}, "
        f"RMSE: {_fmt_money(test_metrics['rmse'])}, "
        f"MAE: {_fmt_money(test_metrics['mae'])}"
    )

    # 5. Result figures -----------------------------------------------------
    plots.model_comparison(cv_results)
    plots.predicted_vs_actual(y_test, test_metrics["y_pred"], best_name)
    plots.residual_plot(y_test, test_metrics["y_pred"], best_name)
    plots.feature_importance(best_model, best_name)

    # 6. Reports ------------------------------------------------------------
    _write_results(cv_results, best_name, test_metrics)
    print(f"\nWrote {config.RESULTS_CSV.name} and {config.RESULTS_MD.name}. Done.")


if __name__ == "__main__":
    run()
