
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import pickle


# Load .env file
load_dotenv()

# Get environment variables
data_path = os.path.expanduser(os.getenv('DATA_PATH'))
model_path = os.path.expanduser(os.getenv('MODEL_PATH'))
scaler_path = os.path.expanduser(os.getenv('SCALER_PATH'))
encoder_path = os.path.expanduser(os.getenv('ENCODER_PATH'))
results_path = os.path.expanduser(os.getenv('RESULTS_PATH'))

model = None

def run_ml_pipeline():
    # Load the dataset
    # data_path = '~/Library/CloudStorage/OneDrive-Pfizer/Desktop/Projects/California_Pricing/data/housing.csv'
    df = pd.read_csv(data_path)


    # Separate features and target variable
    X = df.drop('median_house_value', axis=1)  # Features
    y = df['median_house_value']  # Target variable

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')
    X[numerical_cols] = numerical_transformer.fit_transform(X[numerical_cols])

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    X_encoded = categorical_transformer.fit_transform(X[categorical_cols]).toarray()
    X_encoded_df = pd.DataFrame(X_encoded, columns=categorical_transformer.get_feature_names_out(categorical_cols))

    # Combine numerical and encoded categorical data
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # MinMaxScaler
    # scaler = MinMaxScaler()
    # X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # RobustScaler
    # scaler = RobustScaler()
    # X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Regressor': SVR()
    }

    # Initialize a list to store the results
    results = []
    best_model = None
    best_r2 = float('-inf')

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Append the results to the list
        results.append({
            'Model': name,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae,
            'R-squared': r2,
            'Run DateTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Comment': 'Added Scaler and encoder'
        })
        
        # Check if this model is the best so far
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results table
    print(results_df)

    # Save the results to a CSV file
    # csv_file = os.path.expanduser('~/Library/CloudStorage/OneDrive-Pfizer/Desktop/Projects/California_Pricing/data/resultsdb/model_results.csv')
    if os.path.isfile(results_path):
        results_df.to_csv(results_path, mode='a', header=False, index=False)

    # Save the best model's parameters to a pickle file
    # file_path = os.path.expanduser('~/Library/CloudStorage/OneDrive-Pfizer/Desktop/Projects/California_Pricing/data/model_params/best_model.pkl')
    with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
    
    # Save the encoder
    # file_path = os.path.expanduser('~/Library/CloudStorage/OneDrive-Pfizer/Desktop/Projects/California_Pricing/data/model_params/encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(categorical_transformer, f)

    # Save the scaler
    # file_path = os.path.expanduser('~/Library/CloudStorage/OneDrive-Pfizer/Desktop/Projects/California_Pricing/data/model_params/scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

# Run the pipeline
if __name__ == "__main__":
    run_ml_pipeline()