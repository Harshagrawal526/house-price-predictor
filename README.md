# 🏠 PriceWise

Predicting home sale prices from 79 property features, using a clean,
reproducible, **leak-free** machine-learning pipeline.

<p>
  <img alt="CI" src="https://github.com/Harshagrawal526/PriceWise/actions/workflows/ci.yml/badge.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

### ▶️ [Try the live demo](https://house-price-predictor-ha.streamlit.app/)

An interactive Streamlit app takes a house's details and predicts its price.

---

## Overview

The [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
describes 1,460 home sales in Ames, Iowa with **79 features** — lot size, living
area, quality ratings, year built, neighborhood, garage, basement, and dozens
more, including real missing values. The goal is to predict the **sale price**.

The project compares four regression models under identical, honest conditions,
reports every error metric **in dollars** so the results are interpretable, and
refuses to pick a winner on a difference smaller than its own error bars.

> **Dataset:** [House Prices: Advanced Regression Techniques (Kaggle / Ames, Iowa)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Quick start

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Run the full pipeline: train, evaluate, generate figures + the saved model
python main.py

# 3. Launch the interactive demo
streamlit run app.py

# 4. (optional) run the tests
pytest
```

Running `main.py` prints the model comparison to the console and regenerates
everything under `data/`, `reports/`, and the saved model in `models/`.

## Results

Five-fold cross-validation across all candidate models:

| Model | R² | RMSE ($) | MAE ($) |
|-------|-----|----------|---------|
| Random Forest | 0.840 ± 0.082 | 30,667 | 17,697 |
| **Lasso Regression** ⭐ | 0.835 ± 0.156 | 29,214 | **14,819** |
| Gradient Boosting | 0.832 ± 0.131 | 30,284 | 16,341 |
| Ridge Regression | 0.829 ± 0.168 | 29,529 | 15,089 |

On the held-out 20% test set, the selected model (**Lasso Regression**) scores
**R² = 0.914**, **RMSE = $25,638**, **MAE = $15,407**.

### Why Lasso, when Random Forest has the higher R²

Because that R² gap is not real. Random Forest leads by **0.004**, while R²
swings by **±0.082** between folds — the lead is about 5% of the noise it is
measured against. Sorting the table and taking the top row would be ranking
random variation, so all four models are treated as tied on R² and separated on
**MAE** instead: it is denominated in dollars, and it is the error a price
estimate is actually judged on. On that metric Lasso is **~$2,900 per house**
better than Random Forest, a gap roughly twice its own fold-to-fold spread.

The held-out set backs this up. Random Forest scored R² 0.890 / MAE ~$17K there;
Lasso scores **0.914 / $15,407** — better on every metric, on data neither model
had seen. Picking on the R² column alone would have shipped the worse model.

**Key finding:** no model separates from the others on R² at this sample size,
and the regularised linear models give the lowest dollar error despite never
topping that column — the preprocessing matters more than the estimator.
`OverallQual` and `GrLivArea` are consistently the most influential features.

<p>
  <img src="reports/figures/model_evaluation_plot.png" width="49%">
  <img src="reports/figures/predicted_vs_actual.png" width="49%">
</p>

See [`reports/results.md`](reports/results.md) for the full write-up and every
figure.

## Demo

A [Streamlit](https://streamlit.io) app wraps the trained model: fill in a
house's attributes and get an instant price estimate.

**[▶️ Live app](https://house-price-predictor-ha.streamlit.app/)** — or run it locally:

```bash
streamlit run app.py
```

The app loads the saved model from `models/`, training it on first launch if it
isn't there yet. It's deployed for free on Streamlit Community Cloud.

## Design decisions

- **No data leakage** — imputation, scaling, and one-hot encoding are all fit
  inside each cross-validation fold via a scikit-learn `Pipeline`, never on the
  full dataset.
- **Handles messy data** — missing values are imputed, rare categories are
  grouped, and outliers are clipped so the linear models stay stable across the
  ~260 encoded columns.
- **Skew-aware target** — `SalePrice` is right-skewed, so models train on its log
  and predict back in dollars, keeping RMSE and MAE interpretable.
- **Selection that survives its own error bars** — models within one standard
  error of the best R² are treated as tied and separated on MAE, so a lead
  smaller than the fold-to-fold noise never decides which model ships.
- **Reproducible** — a fixed seed and a single `python main.py` regenerate every
  result, figure, and the saved model.

## Project structure

```
PriceWise/
├── housing_price/            # the package
│   ├── config.py             # paths and constants
│   ├── data.py               # loading & train/test splitting
│   ├── features.py           # leak-free ColumnTransformer (impute/scale/encode)
│   ├── models.py             # four model pipelines
│   ├── evaluate.py           # cross-validation & hold-out metrics
│   ├── plots.py              # all visualisations
│   ├── train.py              # fit the best model and save it
│   └── pipeline.py           # end-to-end orchestration
├── data/
│   ├── AmesHousing.csv       # raw dataset
│   └── Model_Evaluation_Results.csv   # generated
├── reports/
│   ├── figures/              # generated plots
│   └── results.md            # generated report
├── tests/                    # pytest suite
├── .github/workflows/ci.yml  # runs the tests on every push
├── app.py                    # Streamlit demo
├── main.py                   # run the pipeline
└── requirements.txt
```

## License

Released under the [MIT License](LICENSE).
