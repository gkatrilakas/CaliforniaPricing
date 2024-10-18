import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_valid_input():
    """
    Function to test the predict endpoint with a valid input
    """    

    # Define a valid input payload
    payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }

    # Send a POST request to the prediction endpoint
    response = client.post("/predict", json=payload)

    # Assert the response status code and content
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_missing_field():
    # Define an input payload with a missing field
    payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
        # Missing "ocean_proximity"
    }

    # Send a POST request to the prediction endpoint
    response = client.post("/predict", json=payload)

    # Assert the response status code and content
    assert response.status_code == 422  # Unprocessable Entity

if __name__ == '__main__':
    pytest.main()