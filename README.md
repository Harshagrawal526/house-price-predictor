# 🏠 Housing Price Prediction

Predicting house prices from their physical attributes and amenities, using a
clean, reproducible, **leak-free** machine-learning pipeline.

<p>
  <img alt="CI" src="https://github.com/Harshagrawal526/house-price-predictor/actions/workflows/ci.yml/badge.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

### ▶️ [Try the live demo](https://house-price-predictor-ha.streamlit.app/)

An interactive Streamlit app takes a house's details and predicts its price.

---

## Overview

Given 545 houses described by 12 features (area, bedrooms, bathrooms, stories,
parking, and amenities such as air-conditioning, a basement, or a preferred
location), the goal is to predict the sale **price**.

The project compares six regression models under identical, honest conditions
and reports every error metric **in real rupees** so the results are actually
interpretable.

> **Dataset:** [Kaggle Housing dataset](https://www.kaggle.com/datasets/ashydv/housing-dataset)

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

| Model | R² | RMSE (₹) | MAE (₹) |
|-------|-----|----------|---------|
| **Ridge Regression** ⭐ | **0.633** | 1,083,790 | 805,794 |
| Lasso Regression | 0.633 | 1,084,287 | 806,463 |
| Linear Regression | 0.632 | 1,084,538 | 807,180 |
| Random Forest | 0.621 | 1,108,356 | 804,469 |
| Gradient Boosting | 0.612 | 1,117,763 | 804,724 |
| Polynomial (deg 2) + Ridge | 0.612 | 1,109,692 | 814,444 |

On the held-out 20% test set, the best model (**Ridge**) scores **R² = 0.65**,
**RMSE ≈ ₹1.33M**, **MAE ≈ ₹0.97M**.

**Key finding:** the price relationship on this dataset is largely *linear* —
regularised linear models match or slightly beat the tree ensembles, and
`area`, `bathrooms`, and `airconditioning` are the most influential features.

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

- **No data leakage** — preprocessing is fit inside each cross-validation fold
  via a scikit-learn `Pipeline`, never on the full dataset.
- **Interpretable metrics** — `price` is kept in rupees, so RMSE and MAE are
  real amounts rather than scaled numbers.
- **Reproducible** — a fixed seed and a single `python main.py` regenerate every
  result, figure, and the saved model.

## Project structure

```
house-price-predictor/
├── housing_price/            # the package
│   ├── config.py             # paths, column groups, constants
│   ├── data.py               # loading & train/test splitting
│   ├── features.py           # leak-free ColumnTransformer (scale + one-hot)
│   ├── models.py             # six model pipelines
│   ├── evaluate.py           # cross-validation & hold-out metrics
│   ├── plots.py              # all visualisations
│   ├── train.py              # fit the best model and save it
│   └── pipeline.py           # end-to-end orchestration
├── data/
│   ├── Housing.csv           # raw dataset
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
