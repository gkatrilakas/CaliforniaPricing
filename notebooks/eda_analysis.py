# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# load .env file
load_dotenv()

data_path = os.path.expanduser(os.getenv('DATA_PATH'))

# %%
def eda_analysis(data:str, categorical_columns: list, target_variable:str) -> None:
    """EDA Analysis Function

    Args:
        data (str): Path to the dataset
        categorical_columns (list): List of categorical columns
        target_variable (str): Target variable
    """    

    # Load the dataset
    df = pd.read_csv(data)

    # Basic Information
    print("## Shape of the DataFrame ##")
    print(df.shape)
    print("\n## Info of the DataFrame ##")
    print(df.info())
    print("\n## Description of the DataFrame ##")
    print(df.describe().T)

    # Missing Values
    print("\n## Missing Values ##")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # Data Types
    print("\n## Data Types ##")
    print(df.dtypes)

    # Distribution of Numerical Variables
    df.hist(bins=30, figsize=(20, 15))
    plt.suptitle('Distribution of Numerical Variables')
    plt.show()

    # Distribution of Categorical Variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.show()

    """
    # Correlation Analysis
    The correlation matrix provides a visual representation of the relationships between numerical features.
    It helps to identify strong correlations between features, which can inform feature selection for modeling.
    """
    correlation_matrix = df.loc[:, ~df.columns.isin(categorical_columns)].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    """
    # Feature Relationships Table
    The table below shows the correlation coefficients between each feature and the target variable.
    This helps to identify the most influential features for modeling.
    """

    correlations = df.loc[:, ~df.columns.isin(categorical_columns)].corr()[target_variable].sort_values(ascending=False)
    correlation_table = pd.DataFrame(correlations).reset_index()
    correlation_table.columns = ['Feature', 'Correlation with Target']
    print("\n## Correlation with Target Variable ##")
    print(correlation_table)

    """
    # Target Variable Analysis
    The histogram shows the distribution of the target variable, which helps to understand its characteristics and distribution.
    """

    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_variable], kde=True)
    plt.title(f'Distribution of {target_variable}')
    plt.show()

# %%
eda_analysis(
    data=data_path,
    categorical_columns=['ocean_proximity'],
    target_variable='median_house_value'
)
# %%
