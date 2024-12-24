import matplotlib.pyplot as plt
from metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix heatmap.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (true labels).
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of label names.
    title : str
        Title for the plot.

    Returns:
    -------
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    """
    Plot the ROC curve.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (true labels).
    y_prob : array-like
        Predicted probabilities.
    title : str
        Title for the plot.

    Returns:
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='AUC = {:.4f}'.format(roc_auc_score(y_true, y_prob)))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
