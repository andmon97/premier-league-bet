import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    """
    Loads and cleans data by dropping specific columns and handling missing values.

    Parameters
    ----------
    filepath : str
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned data.
    """
    data = pd.read_csv(filepath)
    columns_to_drop = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot', 
                       'goal_diff', 'day', 'pk', 'pkatt', 'fk', 
                       'referee', 'dist', 'points', 'season_winner', 'hour', 'result_encoded', 'day_code', 'season']
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    data.dropna(inplace=True)
    data.columns = data.columns.str.strip()
    return data

def preprocess_data(data, target_col, threshold=100):
    """
    Preprocesses data by encoding categorical columns and scaling numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be preprocessed.
    target_col : str
        The name of the target column to exclude from preprocessing.
    threshold : int
        The minimum frequency for categorical values to be kept.

    Returns
    -------
    tuple
        The preprocessed features (pd.DataFrame) and the unaltered target column (pd.Series).
    """
    # Exclude the target column during preprocessing
    features = data.drop(columns=[target_col])

    # Identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle rare categories in categorical columns
    for col in categorical_cols:
        frequencies = features[col].value_counts()
        features[col] = features[col].apply(lambda x: x if frequencies[x] >= threshold else 'Other')

    # One-hot encode categorical data
    data_categorical = pd.get_dummies(features[categorical_cols])
    
    # Scale numerical data
    scaler = StandardScaler()
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    # Combine processed features
    processed_features = pd.concat([data_categorical, features[numerical_cols]], axis=1)
    return processed_features, data[target_col]

def split_data(data, target_col='result', test_size=0.2, val_size=0.25, random_state=42):
    """
    Splits data into training, validation, and testing sets.

    Parameters
    ----------
    data : pd.DataFrame
        The preprocessed data.
    target_col : str
        The name of the target column.
    test_size : float
        The proportion of the data to include in the test set.
    val_size : float
        The proportion of the training data to include in the validation set.
    random_state : int
        The seed used by the random number generator.

    Returns
    -------
    tuple
        The training, validation, and testing data (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = data.drop(columns=[target_col])
    y = LabelEncoder().fit_transform(data[target_col])

    # First split: train+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
