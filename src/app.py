import streamlit as st
import joblib
import numpy as np

# Load the saved model
model_data = joblib.load("src/house_model.pkl")
w = model_data["w"]
b = model_data["b"]
x_mean = model_data["mean"]
x_std = model_data["std"]

# Prediction function
def predict_price(size, w, b, mean, std):
    x_norm = (size - mean) / std
    return w * x_norm + b

# Streamlit App UI
st.title("🏡 House Price Predictor (Linear Regression)")

st.write("This app predicts the price of a house based on its size.")

size_input = st.number_input("Enter house size (in sq.ft):", min_value=500, max_value=5000, value=1200, step=50)

if st.button("Predict Price"):
    price = predict_price(size_input, w, b, x_mean, x_std)
    st.success(f"Estimated Price: ${price:.2f}k")
