import pandas as pd
import streamlit as st

from housing_price.data import load_data, split_features_target
from housing_price.train import load_model

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def get_model():
    bundle = load_model()
    return bundle["model"], bundle["name"]


@st.cache_data
def get_baseline():
    # a full feature row filled with typical values (median for numbers, most
    # common for categories); the sliders below override the important ones
    X, _ = split_features_target(load_data())
    baseline = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            baseline[col] = X[col].median()
        else:
            baseline[col] = X[col].mode()[0]
    return X, baseline


model, model_name = get_model()
X, baseline = get_baseline()

st.title("🏠 House Price Predictor")
st.caption(
    f"A {model_name} model trained on the Ames Housing dataset (79 features). "
    "Set the main details below — the rest use typical values."
)

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall quality (1–10)", 1, 10, 7)
    gr_liv_area = st.number_input("Above-ground living area (sq ft)", 400, 6000, 1500, 50)
    total_bsmt = st.number_input("Basement area (sq ft)", 0, 3000, 900, 50)
    lot_area = st.number_input("Lot area (sq ft)", 1000, 60000, 9500, 500)

with col2:
    year_built = st.slider("Year built", 1900, 2010, 1975)
    garage_cars = st.slider("Garage capacity (cars)", 0, 4, 2)
    full_bath = st.slider("Full bathrooms", 0, 4, 2)
    neighborhood = st.selectbox(
        "Neighborhood", sorted(X["Neighborhood"].unique()),
        index=sorted(X["Neighborhood"].unique()).index(baseline["Neighborhood"]),
    )

if st.button("Predict price", type="primary"):
    row = dict(baseline)
    row.update({
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt,
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "Neighborhood": neighborhood,
    })
    X_row = pd.DataFrame([row])[X.columns]
    price = model.predict(X_row)[0]
    st.success(f"Estimated price: ${price:,.0f}")
    st.caption("Trained on 1,460 Ames, Iowa home sales. Estimate only.")
