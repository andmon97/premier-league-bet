import json

def load_hyperparameters(filename):
    """
    Loads hyperparameters from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file containing the hyperparameters.

    Returns
    -------
    hyperparameters : dict
        A dictionary with the hyperparameters.
    """
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters
