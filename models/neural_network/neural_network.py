import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

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
    Trains a neural network model with additional features like validation and early stopping.

    Parameters:
    - model (nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - criterion (loss function): Loss function used for training.
    - optimizer (Optimizer): Optimizer used for updating model weights.
    - epochs (int): Number of training epochs.
    - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - model (nn.Module): The trained model.
    """

    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_training_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_training_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # Save model if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print("Model saved as validation loss improved.")

        # Optional: Implement early stopping logic here

    return model

def test_model(model, test_loader, criterion, device='cpu'):
    """
    Tests the neural network model and calculates metrics such as accuracy.

    Parameters:
    - model (nn.Module): The neural network model to be tested.
    - test_loader (DataLoader): DataLoader for testing data.
    - criterion (loss function): Loss function used for evaluating the model.
    - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - None
    """
    model.to(device)
    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
