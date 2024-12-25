import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import numpy as np
import pandas as pd

from models.neural_network.neural_network import DynamicNet, train_model, test_model
from models.logistic_regression.logistic_regression import LogisticRegressionModel
from models.hyperparameters.load_hyperparameters import load_hyperparameters
from data.data_preparation import load_and_clean_data, preprocess_data, split_data

HYPERPARAMETERS_FILE_PATH = 'models/hyperparameters/hyperparameters_'
DATASET_PATH = 'data/processed/matches_processed.csv'
TARGET_COLUMN = 'result'
MODEL_NAMES = ['neural_network', 'logistic_regression']  # List of algorithms


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_and_clean_data(DATASET_PATH)
    data_features, data_target = preprocess_data(data, TARGET_COLUMN)

    # Combine features and target for splitting
    data_preprocessed = pd.concat([data_features, data_target], axis=1)

    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_preprocessed, TARGET_COLUMN)

    # Convert data to PyTorch tensors for neural network
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.int64))
    X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.astype(np.int64))
    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.astype(np.int64))

    # Create DataLoaders for neural network
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

    for model_name in MODEL_NAMES:
        hyperparams = load_hyperparameters(f'{HYPERPARAMETERS_FILE_PATH}{model_name}.json')

        if model_name == 'neural_network':
            # Grid search for neural network
            for lr, epochs, neurons_1layer, neurons_2layer, activation, batch_size in product(
                hyperparams['learning_rate'],
                hyperparams['epochs'],
                hyperparams['neurons_1layer'],
                hyperparams['neurons_2layer'],
                hyperparams['activation_functions'],
                hyperparams['batch_size']):

                model = DynamicNet(input_dim=X_train.shape[1], neurons_1layer=neurons_1layer, neurons_2layer=neurons_2layer, activation_func=activation)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                print(f'Training Neural Network with lr={lr}, epochs={epochs}, neurons_1layer={neurons_1layer}, neurons_2layer={neurons_2layer}, activation={activation}, batch_size={batch_size}')
                train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
                test_model(model, test_loader, criterion, device)

        elif model_name == 'logistic_regression':
            # Grid search for logistic regression
            for penalty, C, solver, max_iter in product(
                hyperparams['penalty'],
                hyperparams['C'],
                hyperparams['solver'],
                hyperparams['max_iter']):

                model = LogisticRegressionModel(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

                print(f'Training Logistic Regression with penalty={penalty}, C={C}, solver={solver}, max_iter={max_iter}')
                model.train(X_train, y_train)
                val_metrics = model.validate(X_val, y_val)
                print(f'Validation Metrics: {val_metrics}')

                test_metrics = model.test(X_test, y_test)
                print(f'Test Metrics: {test_metrics}')

if __name__ == '__main__':
    main()
