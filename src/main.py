from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Load .env file
load_dotenv()

# Get environment variables
model_path = os.path.expanduser(os.getenv('MODEL_PATH'))
scaler_path = os.path.expanduser(os.getenv('SCALER_PATH'))
encoder_path = os.path.expanduser(os.getenv('ENCODER_PATH'))

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Define the input data model statically based on the feature columns
class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# Define the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData) -> dict:
    
    """FastApi route to make predictions

    Args:
        data (InputData): Data to make predictions

    Returns:
        dict: Prediction
    """ 
    
       
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    
    # ocean_proximity_choices = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    # Apply one-hot encoding to ocean_proximity
    encoded_ocean_proximity = encoder.transform(input_df[['ocean_proximity']]).toarray()
    encoded_ocean_proximity_df = pd.DataFrame(encoded_ocean_proximity, columns=encoder.get_feature_names_out(['ocean_proximity']))

    # Drop the original ocean_proximity column and concatenate the encoded columns
    input_df = input_df.drop('ocean_proximity', axis=1)
    input_df = pd.concat([input_df.reset_index(drop=True), encoded_ocean_proximity_df.reset_index(drop=True)], axis=1)

    # Scale numerical features
    numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])


    # Ensure the columns are in the correct order
    expected_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
        'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
        'ocean_proximity_NEAR OCEAN'
    ]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Convert input data to numpy array
    input_data = input_df.values

    # Make prediction
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}