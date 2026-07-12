import joblib

from . import config, plots
from .data import load_data, split_features_target, train_test_data
from .evaluate import cross_validate_models, evaluate_on_test
from .models import get_models


def _fmt_money(x):
    return f"₹{x:,.0f}"


def _write_results(cv_results, best_name, test_metrics):
    config.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(config.RESULTS_CSV, index=False)

    lines = [
        "# Housing Price Prediction - Results",
        "",
        "Preprocessing (scaling + one-hot encoding) runs inside each CV fold, and "
        "`price` is left in rupees so RMSE and MAE are readable.",
        "",
        f"- Dataset: {config.RAW_DATA.name} (545 homes, 12 features)",
        f"- Validation: {config.CV_FOLDS}-fold cross-validation + a "
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
        "## Notes",
        "",
        "- The relationship here is mostly linear: Ridge/Lasso do about as well "
        "as the tree ensembles.",
        "- `area`, `bathrooms` and `airconditioning` are the strongest features.",
        "- Plain polynomial regression is unstable on the binary dummies, so the "
        "degree-2 model uses Ridge.",
        "",
    ]
    config.RESULTS_MD.write_text("\n".join(lines))


def run():
    print("Housing Price Prediction\n" + "=" * 40)

    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {config.RAW_DATA.name}")

    plots.price_distribution(df)
    plots.correlation_heatmap(df)

    X, y = split_features_target(df)
    models = get_models()
    print(f"\nCross-validating {len(models)} models ({config.CV_FOLDS}-fold)...")
    cv_results = cross_validate_models(models, X, y)
    print(cv_results[["Model", "R2", "RMSE", "MAE"]].to_string(index=False))

    best_name = cv_results.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    X_train, X_test, y_train, y_test = train_test_data(df)
    test_metrics = evaluate_on_test(best_model, X_train, y_train, X_test, y_test)
    print(
        f"Held-out test - R2: {test_metrics['r2']:.3f}, "
        f"RMSE: {_fmt_money(test_metrics['rmse'])}, "
        f"MAE: {_fmt_money(test_metrics['mae'])}"
    )

    plots.model_comparison(cv_results)
    plots.predicted_vs_actual(y_test, test_metrics["y_pred"], best_name)
    plots.residual_plot(y_test, test_metrics["y_pred"], best_name)
    plots.feature_importance(best_model, best_name)

    _write_results(cv_results, best_name, test_metrics)

    # refit the winner on all the data and save it for the Streamlit app
    final_model = get_models()[best_name]
    final_model.fit(X, y)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "name": best_name}, config.MODEL_PATH)

    print(f"\nWrote {config.RESULTS_CSV.name}, {config.RESULTS_MD.name} and {config.MODEL_PATH.name}.")


if __name__ == "__main__":
    run()
