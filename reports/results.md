# Housing Price Prediction - Results

Preprocessing (scaling + one-hot encoding) runs inside each CV fold, and `price` is left in rupees so RMSE and MAE are readable.

- Dataset: Housing.csv (545 homes, 12 features)
- Validation: 5-fold cross-validation + a 20% held-out test set

## Cross-Validated Model Comparison

| Model | R² | RMSE (₹) | MAE (₹) |
|-------|----|----------|---------|
| Ridge Regression | 0.633 ± 0.072 | 1,083,790 | 805,794 |
| Lasso Regression | 0.633 ± 0.073 | 1,084,287 | 806,463 |
| Linear Regression | 0.632 ± 0.074 | 1,084,538 | 807,180 |
| Random Forest | 0.621 ± 0.064 | 1,108,356 | 804,469 |
| Gradient Boosting | 0.612 ± 0.070 | 1,117,763 | 804,724 |
| Polynomial (deg 2) + Ridge | 0.612 ± 0.101 | 1,109,692 | 814,444 |

**Best model:** Ridge Regression

## Held-Out Test Performance

Evaluated on the unseen 20% test split:

- **R²:** 0.652
- **RMSE:** ₹1,326,231
- **MAE:** ₹971,714

## Figures

![Model comparison](figures/model_evaluation_plot.png)
![Predicted vs actual](figures/predicted_vs_actual.png)
![Residual plot](figures/residual_plot.png)
![Feature importance](figures/feature_importance.png)
![Price distribution](figures/housing_price_distribution.png)
![Correlation heatmap](figures/correlation_heatmap.png)

## Notes

- The relationship here is mostly linear: Ridge/Lasso do about as well as the tree ensembles.
- `area`, `bathrooms` and `airconditioning` are the strongest features.
- Plain polynomial regression is unstable on the binary dummies, so the degree-2 model uses Ridge.
