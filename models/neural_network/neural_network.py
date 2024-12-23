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
        self.output_layer = nn.Linear(neurons_2layer, 3)  # Assuming 3 output classes

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

def train_model(model, train_loader, criterion, optimizer, epochs):
    """
    Trains a neural network model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        The DataLoader for loading the training data.
    criterion : loss function
        The loss function used to compute the error between the output and target.
    optimizer : optimizer
        The optimizer used to update the model parameters.
    epochs : int
        The number of epochs to train the model.

    Returns
    -------
    nn.Module
        The trained neural network model.
    """

    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
