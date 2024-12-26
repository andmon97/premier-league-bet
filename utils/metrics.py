import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import json
import os

def compute_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """
    Compute classification metrics including precision, recall, F1-score, and AUC-ROC.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (true labels).
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities (for AUC-ROC calculation).
    average : str, default='weighted'
        Averaging method for multi-class metrics ('weighted', 'macro', 'micro').

    Returns:
    -------
    metrics : dict
        Dictionary containing precision, recall, F1-score, and AUC-ROC (if y_prob is provided).
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average),
    }

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)

    return metrics

def save_metrics_to_file(metrics, algorithm_name, config_name, output_dir="metrics_output"):
    """
    Save computed metrics to a JSON file.

    Parameters:
    ----------
    metrics : dict
        Dictionary of computed metrics.
    algorithm_name : str
        Name of the algorithm.
    config_name : str
        Configuration name combining algorithm name and hyperparameter values.
    output_dir : str
        Directory where the metrics file will be saved.

    Returns:
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{algorithm_name}_{config_name}.json")
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")
