import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from utils.metrics import compute_metrics

class DynamicNet(nn.Module):
    def __init__(self, input_dim, neurons_1layer, neurons_2layer, activation_func):
        """
        Constructor for DynamicNet class.

        Parameters
        ----------
        input_dim : int
            The number of features in the input data.
        neurons_1layer : int
            The number of neurons in the first hidden layer.
        neurons_2layer : int
            The number of neurons in the second hidden layer.
        activation_func : str
            The activation function to use. Can be "relu", "sigmoid", or "tanh".
        """
        super(DynamicNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, neurons_1layer)
        if activation_func == "relu":
            self.activation1 = nn.ReLU()
        elif activation_func == "sigmoid":
            self.activation1 = nn.Sigmoid()
        elif activation_func == "tanh":
            self.activation1 = nn.Tanh()

        self.layer2 = nn.Linear(neurons_1layer, neurons_2layer)
        self.output_layer = nn.Linear(neurons_2layer, 3) 

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.activation1(self.layer1(x))
        x = self.activation1(self.layer2(x))
        x = self.output_layer(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu'):
    """
    Trains a neural network model with additional features like validation metrics.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    criterion : loss function
        Loss function used for training.
    optimizer : Optimizer
        Optimizer used for updating model weights.
    epochs : int
        Number of training epochs.
    device : str
        Device to run the model on ('cpu' or 'cuda').

    Returns
    -------
    dict
        Dictionary containing training and validation metrics for each epoch.
    """
    model.to(device)
    metrics_history = {'train': [], 'validation': []}

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_training_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(all_labels, all_preds)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = compute_metrics(val_labels, val_preds)

        # Store Metrics
        metrics_history['train'].append({'loss': avg_training_loss, **train_metrics})
        metrics_history['validation'].append({'loss': avg_val_loss, **val_metrics})

        print(f"Epoch {epoch+1}: Train Loss = {avg_training_loss:.4f}, Train Metrics = {train_metrics}")
        print(f"Validation Loss = {avg_val_loss:.4f}, Validation Metrics = {val_metrics}")

    return metrics_history

def test_model(model, test_loader, criterion, device='cpu'):
    """
    Tests the neural network model and calculates metrics.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be tested.
    test_loader : DataLoader
        DataLoader for testing data.
    criterion : loss function
        Loss function used for evaluating the model.
    device : str
        Device to run the model on ('cpu' or 'cuda').

    Returns
    -------
    dict
        Dictionary containing test metrics.
    """
    model.to(device)
    model.eval()
    total_test_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_metrics = compute_metrics(all_labels, all_preds)
    test_metrics['loss'] = avg_test_loss

    print(f'Test Loss: {avg_test_loss:.4f}, Test Metrics: {test_metrics}')
    return test_metrics

