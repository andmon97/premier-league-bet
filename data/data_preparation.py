import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    """
    Loads data from a csv file.

    Parameters
    ----------
    filepath : str
        The path to the csv file.

    Returns
    -------
    data : pd.DataFrame
        The loaded data.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, categorical_cols, numerical_cols):
    # One-hot encode categorical data
    """
    Preprocesses data by one-hot encoding categorical data and scaling numerical data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be preprocessed.
    categorical_cols : list
        The names of the categorical columns.
    numerical_cols : list
        The names of the numerical columns.

    Returns
    -------
    data_preprocessed : pd.DataFrame
        The preprocessed data.
    """
    data_categorical = pd.get_dummies(data[categorical_cols])
    # Scale numerical data
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    # Combine data
    data_preprocessed = pd.concat([data_categorical, data[numerical_cols]], axis=1)
    return data_preprocessed

def split_data(data, target_col, test_size=0.2):
    """
    Splits data into training and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be split.
    target_col : str
        The name of the target column.
    test_size : float
        The proportion of the data to include in the test set.

    Returns
    -------
    X_train : pd.DataFrame
        The features of the training set.
    X_test : pd.DataFrame
        The features of the test set.
    y_train : pd.Series
        The target of the training set.
    y_test : pd.Series
        The target of the test set.
    """
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
