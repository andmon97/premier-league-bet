import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
        'acccuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average),
    }

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)

    return metrics

import os

# utils/metrics.py
import os

def save_metrics_to_txt(metrics_dict, config_name, model_name, output_dir="output"):
    """
    Saves train, validation, and test metrics to a .txt file, using a
    descriptive name derived from the model name and configuration name.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing 'train_metrics', 'val_metrics', and 'test_metrics'.
    config_name : str
        The hyperparameter configuration that produced these metrics.
    model_name : str
        A label for the model/algorithm (e.g., 'neural_network', 'logistic_regression', etc.).
    output_dir : str, default='output'
        The directory where the metrics file will be saved.

    Returns
    -------
    metrics_file_path : str
        The path to the .txt file containing the saved metrics.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Derive a consistent filename
    # e.g., output/neural_network_lr=0.001_..._myutilsmetrics.txt
    file_stem = f"{model_name}_{config_name}_myutilsmetrics"
    metrics_file_path = os.path.join(output_dir, f"{file_stem}.txt")

    with open(metrics_file_path, "w") as f:
        f.write(f"=== {model_name.upper()} ===\n")
        f.write(f"Best Configuration: {config_name}\n\n")

        if "train_metrics" in metrics_dict:
            f.write("=== Training Metrics ===\n")
            for k, v in metrics_dict["train_metrics"].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        if "val_metrics" in metrics_dict:
            f.write("=== Validation Metrics ===\n")
            for k, v in metrics_dict["val_metrics"].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        if "test_metrics" in metrics_dict:
            f.write("=== Test Metrics ===\n")
            for k, v in metrics_dict["test_metrics"].items():
                f.write(f"{k}: {v}\n")

    print(f"Additional metrics saved to: {metrics_file_path}")
    return metrics_file_path
