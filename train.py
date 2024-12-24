import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import numpy as np
import pandas as pd

from models.neural_network.neural_network import DynamicNet, train_model, test_model
from models.hyperparameters.load_hyperparameters import load_hyperparameters
from data.data_preparation import load_and_clean_data, preprocess_data, split_data

HYPERPARAMETERS_FILE_PATH = 'models/hyperparameters/hyperparameters_'
DATASET_PATH = 'data/processed/matches_processed.csv'
MODEL_NAME = 'neural_network'
TARGET_COLUMN = 'result'
MODEL_SAVE_PATH = f'models/{MODEL_NAME}_best_model.pth'

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams = load_hyperparameters(f'{HYPERPARAMETERS_FILE_PATH}{MODEL_NAME}.json')
    data = load_and_clean_data(DATASET_PATH)
    data_features, data_target = preprocess_data(data, TARGET_COLUMN)

    # Combine features and target for splitting
    data_preprocessed = pd.concat([data_features, data_target], axis=1)

    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_preprocessed, TARGET_COLUMN)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.int64))
    X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.astype(np.int64))
    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.astype(np.int64))

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=hyperparams['batch_size'][0], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=hyperparams['batch_size'][0], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=hyperparams['batch_size'][0], shuffle=False)

    # Grid search over hyperparameters
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
        
        print(f'Training with lr={lr}, epochs={epochs}, neurons_1layer={neurons_1layer}, neurons_2layer={neurons_2layer}, activation={activation}, batch_size={batch_size}')
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
        test_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
