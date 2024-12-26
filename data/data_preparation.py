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
                       'referee', 'dist', 'points', 'season_winner', 'hour', 'result_encoded', 'day_code']
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
        The preprocessed features (pd.DataFrame), unaltered target column (pd.Series), and the season column.
    """
    # Keep the target and season columns separate during preprocessing
    target = data[target_col]
    season = data['season']
    features = data.drop(columns=[target_col, 'season'])

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
    return processed_features, target, season

def split_data(features, target, season, test_season=2024, val_size=0.25, random_state=42):
    """
    Splits data into training, validation, and testing sets based on seasons.

    Parameters
    ----------
    features : pd.DataFrame
        The preprocessed feature data.
    target : pd.Series
        The target column.
    season : pd.Series
        The season column.
    test_season : int
        The season to be used for the test set.
    val_size : float
        The proportion of the training data to include in the validation set.
    random_state : int
        The seed used by the random number generator.

    Returns
    -------
    tuple
        The training, validation, and testing data (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Separate test data
    train_val_mask = (season != test_season)
    test_mask = ~train_val_mask

    X_train_val = features[train_val_mask]
    y_train_val = target[train_val_mask]
    X_test = features[test_mask]
    y_test = target[test_mask]

    # Stratified train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
