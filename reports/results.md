# Housing Price Prediction - Results

Missing values are imputed and features scaled/one-hot encoded inside each CV fold. `SalePrice` is skewed, so models train on its log and predict back in dollars, keeping RMSE and MAE readable.

- Dataset: AmesHousing.csv (1460 homes, 79 features)
- Validation: 5-fold cross-validation + a 20% held-out test set

## Cross-Validated Model Comparison

| Model | R² | RMSE ($) | MAE ($) |
|-------|----|----------|---------|
| Random Forest | 0.840 ± 0.082 | 30,667 | 17,697 |
| Lasso Regression | 0.835 ± 0.156 | 29,214 | 14,819 |
| Gradient Boosting | 0.832 ± 0.131 | 30,284 | 16,341 |
| Ridge Regression | 0.829 ± 0.168 | 29,529 | 15,089 |

### How the model was chosen

The top 4 models sit within one standard error of the best R² (0.840 ± 0.082 across 5 folds), so their R² ranking is inside the fold-to-fold noise and does not identify a winner. They are treated as tied and separated on MAE instead -- it is denominated in dollars, and it is the error the app's estimate is actually judged on.

**Selected model:** Lasso Regression (lowest MAE among the tied models, $14,819)

## Held-Out Test Performance

Evaluated on the unseen 20% test split:

- **R²:** 0.914
- **RMSE:** $25,638
- **MAE:** $15,407

## Figures

![Model comparison](figures/model_evaluation_plot.png)
![Predicted vs actual](figures/predicted_vs_actual.png)
![Residual plot](figures/residual_plot.png)
![Feature importance](figures/feature_importance.png)
![Price distribution](figures/housing_price_distribution.png)
![Correlation heatmap](figures/correlation_heatmap.png)

## Notes

- No model separates from the others on R². The spread between the best and worst is smaller than the spread across folds for any one of them, so the ranking is not meaningful at this sample size.
- The regularised linear models give the lowest dollar error despite not topping the R² table -- on this data the preprocessing matters more than the choice of estimator.
- Log-transforming the skewed target and imputing missing values are the two changes that matter most for accuracy.
- `OverallQual` and `GrLivArea` are consistently the strongest predictors.
