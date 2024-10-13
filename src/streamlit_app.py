import streamlit as st
import requests
import pandas as pd

# Define the FastAPI endpoint
api_url = "http://127.0.0.1:8000/predict"

# Create a form to accept user input
st.title("California Housing Price Prediction")

# Define the input fields for each feature
longitude = st.number_input("Longitude", value=-40.6401)
latitude = st.number_input("Latitude", value=22.9444)
housing_median_age = st.number_input("Housing Median Age", value=10.0, step=1.0)
total_rooms = st.number_input("Total Rooms", value=5.0, step=1.0)
total_bedrooms = st.number_input("Total Bedrooms", value=2.0, step=1.0)
population = st.number_input("Population", value=815000.0, step=1000.0)
households = st.number_input("Households", value=4.0, step=1.0)
median_income = st.number_input("Median Income", value=27000.0, step=500.0)

# Hardcode the unique values for ocean_proximity
ocean_proximity_choices = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_choices)

# Create a button to make the prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    # Make the prediction request
    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Median House Value: ${prediction:.2f}")
    else:
        st.error("Failed to make prediction")