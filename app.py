import pandas as pd
import streamlit as st

from housing_price import config
from housing_price.data import load_data
from housing_price.train import load_model

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠")


@st.cache_resource
def get_model():
    bundle = load_model()
    return bundle["model"], bundle["name"]


@st.cache_data
def get_ranges():
    df = load_data()
    return df


model, model_name = get_model()
df = get_ranges()

st.title("🏠 Housing Price Predictor")
st.caption(f"Predicts a house price with a {model_name} model. Enter the details below.")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000,
                           value=int(df["area"].median()), step=100)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 1)
    stories = st.slider("Stories", 1, 4, 2)
    parking = st.slider("Parking spaces", 0, 3, 1)

with col2:
    mainroad = st.selectbox("On main road?", ["yes", "no"])
    guestroom = st.selectbox("Guest room?", ["no", "yes"])
    basement = st.selectbox("Basement?", ["no", "yes"])
    hotwaterheating = st.selectbox("Hot water heating?", ["no", "yes"])
    airconditioning = st.selectbox("Air conditioning?", ["yes", "no"])
    prefarea = st.selectbox("Preferred area?", ["no", "yes"])
    furnishingstatus = st.selectbox(
        "Furnishing", ["furnished", "semi-furnished", "unfurnished"]
    )

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }])[config.FEATURES]

    price = model.predict(row)[0]
    st.success(f"Estimated price: ₹{price:,.0f}")
    st.caption("Trained on the Kaggle Housing dataset (545 homes). Estimate only.")
