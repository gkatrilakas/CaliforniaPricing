# California Housing Price Prediction

## Project Overview

This project aims to predict the median house value for households within a block in California using various machine learning models. The project involves data preprocessing, exploratory data analysis (EDA), model training, and deployment using FastAPI and Streamlit.

## Project Structure

```
California_Pricing/
│
├── data/
│   ├── housing.csv
│   ├── model_params/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── encoder.pkl
│   └── resultsdb/
│       └── model_results.csv
│
├── notebooks/
│   └── eda_analysis.py
│
├── src/
│   ├── main.py
│   ├── mlpipe.py
│   └── streamlit_app.py
│
├── tests/
│   └── test_main.py
│
├── .vscode/
│   └── settings.json
│   └── extesions.json
│
├── .env
├── pyproject.toml
└── README.md
```

## How to Install Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run:

```sh
poetry install
```
To enable the virtual env run:
```sh
poetry shell
```
## How to Run the App

### 1. Run EDA

To perform exploratory data analysis, run:

```sh
poetry run python src/eda_analysis.py
```
or 

run as jupyter notebook

### 2. Train the Model

To train the model and save the best model, scaler, and encoder, run:

```sh
poetry run python src/mlpipe.py
```

### 3. Start the FastAPI Server

To start the FastAPI server, run:

```sh
poetry run uvicorn src.main:app --reload
```

### 4. Start the Streamlit App

To start the Streamlit application, run:

```sh
poetry run streamlit run src/streamlit_app.py
```

## Settings

### .env File

The `.env` file contains environment variables used in the project.

### VS Code Settings

The `.vscode/settings.json` file contains settings for Visual Studio Code. Here is an example:

```json
{
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true
    },
    "python.formatting.provider": "black"
}
```

## CI/CD

### Continuous Integration [Proposal - To do]

Set up a CI/CD pipeline using tools like GitHub Actions to automate testing and deployment.

### Linting and Formatting [Proposal - To do]

Integrate tools like `flake8` for linting and `black` for code formatting into the CI pipeline. Need to add these steps to the GitHub Actions workflow.

## Good Practices

### Dependency Management

- **Poetry**: Use Poetry to manage dependencies and virtual environments. This ensures that the project is reproducible and that dependencies are isolated.
- **Lock File**: Commit the `poetry.lock` file to ensure that the same versions of dependencies are installed across different environments.

### How to Run the Tests
To run the tests, you can use pytest. Make sure you have pytest installed:
``` sh
poetry add pytest
```

Then, run the tests using the following command:
``` sh
poetry run pytest tests/
```