import os
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Load .env file
load_dotenv()

# Get environment variables
data_path = os.path.expanduser(os.getenv('DATA_PATH'))
model_path = os.path.expanduser(os.getenv('MODEL_PATH'))
scaler_path = os.path.expanduser(os.getenv('SCALER_PATH'))
encoder_path = os.path.expanduser(os.getenv('ENCODER_PATH'))
results_path = os.path.expanduser(os.getenv('RESULTS_PATH'))

def run_ml_pipeline():
    # Load the dataset
    df = pd.read_csv(data_path)

    # Separate features and target variable
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

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

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }

    # Initialize a list to store the results
    results = []
    best_model = None
    best_r2 = float('-inf')

    # Train and evaluate models
    for name, model in models.items():
        if name == 'Linear Regression':
            # Scale numerical features for Linear Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            # No scaling for Decision Tree
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Append the results to the list
        results.append({
            'Model': name,
            "Mean Squared Error": np.nan,
            "Root Mean Squared Error": np.nan,
            'Mean Absolute Error': mae,
            'R-squared': r2,
            'Run DateTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Comment': 'Added Random Forest model'
        })
        
        # Check if this model is the best so far
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_scaler = scaler if name == 'Linear Regression' else None

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results table
    print(results_df)

    # Save the results to a CSV file
    if os.path.isfile(results_path):
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_path, mode='w', header=True, index=False)

    # Save the best model's parameters to a pickle file
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
    except Exception as e:
        print(f"Error saving the model: {e}")

    # Save the encoder
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(categorical_transformer, f)
    except Exception as e:
        print(f"Error saving the encoder: {e}")

    # Save the scaler if the best model is Linear Regression
    if best_scaler:
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(best_scaler, f)
        except Exception as e:
            print(f"Error saving the scaler: {e}")

# Run the pipeline
if __name__ == "__main__":
    run_ml_pipeline()