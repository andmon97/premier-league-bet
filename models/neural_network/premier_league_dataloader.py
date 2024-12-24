from torch.utils.data import Dataset, DataLoader
import torch

class PremierDataset(Dataset):
    def __init__(self, data, targets):
        """
        Initialization of the dataset.

        Parameters:
        - data (array-like): The input features.
        - targets (array-like): The labels corresponding to the input data.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Parameters:
        - index (int): The index of the item.

        Returns:
        - A tuple containing (data, target) for the specified index.
        """
        return self.data[index], self.targets[index]

