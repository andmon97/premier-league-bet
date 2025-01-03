import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import numpy as np
import pandas as pd
import os

from models.neural_network.neural_network import DynamicNet, train_model, test_model
from models.logistic_regression.logistic_regression import LogisticRegressionModel
from models.svm.svm import SVMModel
from models.random_forest.random_forest import RandomForestModel
from models.gradient_boosting.gradient_boosting import GradientBoostingModel
from models.knn.knn import KNNModel
from models.naive_bayes.naive_bayes import NaiveBayesModel
from models.decision_tree.decision_tree import DecisionTreeModel
from models.hyperparameters.load_hyperparameters import load_hyperparameters
from data.data_preparation import load_and_clean_data, preprocess_data, split_data
from utils.metrics import save_metrics_to_txt
from sklearn.preprocessing import LabelEncoder

HYPERPARAMETERS_FILE_PATH = 'models/hyperparameters/hyperparameters_'
DATASET_PATH = 'data/processed/matches_processed.csv'
TARGET_COLUMN = 'result'
MODEL_NAMES = ['neural_network', 'logistic_regression', 'svm', 'random_forest', 'gradient_boosting', 'knn', 'naive_bayes', 
               'decision_tree']

def neural_network_workflow(device, X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Dynamic Neural Network model with various hyperparameters.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The file path where the best model is saved.
    """
    model_name = 'neural_network'
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.int64))
    X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.astype(np.int64))
    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.astype(np.int64))

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None

    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all combinations of hyperparameters
    for lr, epochs, neurons_1layer, neurons_2layer, activation, batch_size in product(
            hyperparams['learning_rate'],
            hyperparams['epochs'],
            hyperparams['neurons_1layer'],
            hyperparams['neurons_2layer'],
            hyperparams['activation_functions'],
            hyperparams['batch_size']):

        # Create DataLoaders
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=batch_size,
            shuffle=False
        )

        model = DynamicNet(
            input_dim=X_train.shape[1],
            neurons_1layer=neurons_1layer,
            neurons_2layer=neurons_2layer,
            activation_func=activation
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        config_name = (f"lr={lr}_epochs={epochs}_neurons1={neurons_1layer}_neurons2={neurons_2layer}"
                       f"_activation={activation}_batch={batch_size}")
        print(f"Training Neural Network: {config_name}")

        metrics_history = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs, device
        )
        # Use the last epoch's validation metrics
        val_metrics = metrics_history["validation"][-1]
        val_f1 = val_metrics["f1_score"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = config_name
            test_result = test_model(model, test_loader, criterion, device)
            best_metrics = {
                "train_metrics": metrics_history["train"][-1],
                "val_metrics": val_metrics,
                "test_metrics": test_result
            }
            best_model = model

    best_model_path = None
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"neural_network_{best_config}.pth")
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path}")
        # Also save metrics with the new shared approach
        save_metrics_to_txt(best_metrics, best_config, model_name, output_dir)

    else:
        print("No best model found. Training may have failed.")

    return best_metrics, best_config, best_model_path

def logistic_regression_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Logistic Regression model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk, along with
    a .txt file containing train, validation, and test metrics. Additionally, we call
    the save_metrics_to_txt function to show how we can unify metric logging across all models.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    for penalty, C, solver, max_iter in product(
            hyperparams['penalty'],
            hyperparams['C'],
            hyperparams['solver'],
            hyperparams['max_iter']):
        if solver == 'lbfgs' and penalty not in ['none', 'l2']:
            continue
        if solver == 'liblinear' and penalty in ['none']:
            continue
        penalty_to_use = None if penalty == 'none' else penalty

        config_name = f"penalty={penalty}_C={C}_solver={solver}_max_iter={max_iter}"
        print(f"Training Logistic Regression: {config_name}")

        model = LogisticRegressionModel(penalty=penalty_to_use, C=C, solver=solver, max_iter=max_iter)
        train_metrics = model.train(X_train, y_train)
        val_metrics = model.validate(X_val, y_val)
        test_metrics = model.test(X_test, y_test)

        val_f1 = val_metrics['f1_score']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = config_name
            best_metrics = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics
            }
            best_model = model

    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"logistic_regression_{best_config}.pkl")
        best_model.save_model(best_model_path)
        print(f"Best model saved: {best_model_path}")

        save_metrics_to_txt(best_metrics, best_config, "logistic_regression", output_dir)
    else:
        print("No valid model was found or no improvement over initial baseline.")
    
    return best_metrics, best_config, best_model_path

def svm_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate an SVM model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    # Extract parameters from hyperparams
    for C in hyperparams["C"]:
        for kernel in hyperparams["kernel"]:
            for gamma in hyperparams["gamma"]:
                for degree in hyperparams["degree"]:
                    # If kernel != 'poly', 'degree' won't matter, but let's skip an invalid combo if desired
                    if kernel != 'poly' and degree != 3:
                        # Optionally continue or skip
                        pass

                    config_name = f"C={C}_kernel={kernel}_gamma={gamma}_degree={degree}"
                    print(f"Training SVM: {config_name}")

                    # Initialize and train the model
                    model = SVMModel(C=C, kernel=kernel, gamma=gamma, degree=degree)
                    train_metrics = model.train(X_train, y_train)
                    val_metrics = model.validate(X_val, y_val)
                    test_metrics = model.test(X_test, y_test)

                    # Compare validation F1-score
                    val_f1 = val_metrics['f1_score']
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_config = config_name
                        best_metrics = {
                            "train_metrics": train_metrics,
                            "val_metrics": val_metrics,
                            "test_metrics": test_metrics
                        }
                        best_model = model

    # After all hyperparameter combos, save the best model + metrics
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"svm_{best_config}.pkl")
        best_model.save_model(best_model_path)

        # Write .txt file containing the metrics
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Best model saved to: {best_model_path}")
        print(f"Metrics saved to: {metrics_file_path}")
    else:
        print("No valid SVM model found or no improvement over baseline.")

    return best_metrics, best_config, best_model_path

def random_forest_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Random Forest model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    # Extract parameters from hyperparams
    for n_estimators in hyperparams["n_estimators"]:
        for criterion in hyperparams["criterion"]:
            for max_depth in hyperparams["max_depth"]:
                for min_samples_split in hyperparams["min_samples_split"]:
                    for min_samples_leaf in hyperparams["min_samples_leaf"]:
                        
                        # Convert null from JSON to Python None
                        max_depth_py = None if max_depth is None else max_depth

                        config_name = (f"n_estimators={n_estimators}_criterion={criterion}_max_depth={max_depth_py}"
                                       f"_min_split={min_samples_split}_min_leaf={min_samples_leaf}")
                        print(f"Training Random Forest: {config_name}")

                        # Initialize and train the model
                        model = RandomForestModel(
                            n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth_py,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf
                        )

                        train_metrics = model.train(X_train, y_train)
                        val_metrics = model.validate(X_val, y_val)
                        test_metrics = model.test(X_test, y_test)

                        # Compare validation F1-score
                        val_f1 = val_metrics['f1_score']
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_config = config_name
                            best_metrics = {
                                "train_metrics": train_metrics,
                                "val_metrics": val_metrics,
                                "test_metrics": test_metrics
                            }
                            best_model = model

    # After trying all configurations, save the best model and metrics
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"random_forest_{best_config}.pkl")
        best_model.save_model(best_model_path)
        print(f"Best model saved: {best_model_path}")

        # Also save metrics to a .txt file
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Metrics saved to: {metrics_file_path}")
    else:
        print("No valid Random Forest model found or no improvement over initial baseline.")

    return best_metrics, best_config, best_model_path

def gradient_boosting_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Gradient Boosting model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    # Iterate over hyperparameter combinations
    for loss in hyperparams["loss"]:
        for learning_rate in hyperparams["learning_rate"]:
            for n_estimators in hyperparams["n_estimators"]:
                for subsample in hyperparams["subsample"]:
                    for criterion in hyperparams["criterion"]:
                        for min_samples_split in hyperparams["min_samples_split"]:
                            for min_samples_leaf in hyperparams["min_samples_leaf"]:
                                for max_depth in hyperparams["max_depth"]:

                                    config_name = (f"loss={loss}_lr={learning_rate}_n_est={n_estimators}_sub={subsample}_"
                                                   f"crit={criterion}_min_split={min_samples_split}_"
                                                   f"min_leaf={min_samples_leaf}_depth={max_depth}")
                                    print(f"Training Gradient Boosting: {config_name}")

                                    # Convert null (None) if loaded from JSON
                                    max_depth_py = None if max_depth is None else max_depth

                                    model = GradientBoostingModel(
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        n_estimators=n_estimators,
                                        subsample=subsample,
                                        criterion=criterion,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_depth=max_depth_py
                                    )

                                    # Train, validate, and test
                                    train_metrics = model.train(X_train, y_train)
                                    val_metrics = model.validate(X_val, y_val)
                                    test_metrics = model.test(X_test, y_test)

                                    val_f1 = val_metrics["f1_score"]
                                    if val_f1 > best_val_f1:
                                        best_val_f1 = val_f1
                                        best_config = config_name
                                        best_metrics = {
                                            "train_metrics": train_metrics,
                                            "val_metrics": val_metrics,
                                            "test_metrics": test_metrics
                                        }
                                        best_model = model

    # Save the best model and metrics
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"gradient_boosting_{best_config}.pkl")
        best_model.save_model(best_model_path)

        # Also save metrics to a .txt file
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Best model saved to: {best_model_path}")
        print(f"Metrics saved to: {metrics_file_path}")
    else:
        print("No valid Gradient Boosting model found or no improvement over initial baseline.")

    return best_metrics, best_config, best_model_path

def knn_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a KNN model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    # Extract parameters from hyperparams
    for n_neighbors in hyperparams["n_neighbors"]:
        for weights in hyperparams["weights"]:
            for algorithm in hyperparams["algorithm"]:
                for leaf_size in hyperparams["leaf_size"]:
                    for p in hyperparams["p"]:
                        config_name = (f"n_neighbors={n_neighbors}_weights={weights}_algo={algorithm}"
                                       f"_leaf={leaf_size}_p={p}")
                        print(f"Training KNN: {config_name}")

                        model = KNNModel(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            leaf_size=leaf_size,
                            p=p
                        )

                        # Train, validate, and test
                        train_metrics = model.train(X_train, y_train)
                        val_metrics = model.validate(X_val, y_val)
                        test_metrics = model.test(X_test, y_test)

                        val_f1 = val_metrics["f1_score"]  # Use F1-score to pick best model
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_config = config_name
                            best_metrics = {
                                "train_metrics": train_metrics,
                                "val_metrics": val_metrics,
                                "test_metrics": test_metrics
                            }
                            best_model = model

    # After exploring all hyperparams, save the best model and metrics
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"knn_{best_config}.pkl")
        best_model.save_model(best_model_path)
        print(f"Best model saved: {best_model_path}")

        # Also save metrics to a .txt file
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Metrics saved to: {metrics_file_path}")
    else:
        print("No best KNN model found or no improvement over baseline.")

    return best_metrics, best_config, best_model_path

def decision_tree_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Decision Tree model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    for criterion in hyperparams["criterion"]:
        for splitter in hyperparams["splitter"]:
            for max_depth in hyperparams["max_depth"]:
                # convert null from JSON to None in Python
                max_depth_py = None if max_depth is None else max_depth

                for min_samples_split in hyperparams["min_samples_split"]:
                    for min_samples_leaf in hyperparams["min_samples_leaf"]:
                        for max_features in hyperparams["max_features"]:
                            max_features_py = None if max_features is None else max_features

                            config_name = (
                                f"criterion={criterion}_splitter={splitter}_max_depth={max_depth_py}_"
                                f"min_split={min_samples_split}_min_leaf={min_samples_leaf}_"
                                f"max_features={max_features_py}"
                            )
                            print(f"Training Decision Tree: {config_name}")

                            model = DecisionTreeModel(
                                criterion=criterion,
                                splitter=splitter,
                                max_depth=max_depth_py,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features_py
                            )

                            # Train, validate, test
                            train_metrics = model.train(X_train, y_train)
                            val_metrics = model.validate(X_val, y_val)
                            test_metrics = model.test(X_test, y_test)

                            val_f1 = val_metrics["f1_score"]
                            if val_f1 > best_val_f1:
                                best_val_f1 = val_f1
                                best_config = config_name
                                best_metrics = {
                                    "train_metrics": train_metrics,
                                    "val_metrics": val_metrics,
                                    "test_metrics": test_metrics
                                }
                                best_model = model

    # Save the best model + metrics if found
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"decision_tree_{best_config}.pkl")
        best_model.save_model(best_model_path)
        print(f"Best model saved: {best_model_path}")

        # Also save metrics to a .txt file
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Metrics saved to: {metrics_file_path}")
    else:
        print("No valid Decision Tree model found or no improvement over baseline.")

    return best_metrics, best_config, best_model_path

def naive_bayes_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams):
    """
    Train and evaluate a Naive Bayes model with various hyperparameters.
    Only the best model (based on validation F1-score) is saved to disk,
    along with a .txt file containing train, validation, and test metrics.

    Parameters
    ----------
    X_train : array-like
        The training data.
    X_val : array-like
        The validation data.
    X_test : array-like
        The test data.
    y_train : array-like
        The target values for the training data.
    y_val : array-like
        The target values for the validation data.
    y_test : array-like
        The target values for the test data.
    hyperparams : dict
        A dictionary containing the hyperparameters to grid search over.

    Returns
    -------
    best_metrics : dict
        The metrics of the best configuration (train, validation, and test).
    best_config : str
        The hyperparameter configuration that achieved the best validation F1-score.
    best_model_path : str
        The path where the best model is saved.
    """
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1
    best_config = None
    best_metrics = None
    best_model = None
    best_model_path = None

    # Extract parameters from hyperparams
    for nb_type in hyperparams["nb_type"]:
        for alpha in hyperparams["alpha"]:
            for var_smoothing in hyperparams["var_smoothing"]:

                config_name = f"nb_type={nb_type}_alpha={alpha}_var_smoothing={var_smoothing}"
                print(f"Training Naive Bayes: {config_name}")

                # If nb_type == 'gaussian', alpha is ignored, else var_smoothing is ignored
                # We'll let the NaiveBayesModel handle ignoring inessential parameters,
                # but we can skip them logically if desired:
                
                # Create and train the model
                model = NaiveBayesModel(nb_type=nb_type, alpha=alpha, var_smoothing=var_smoothing)
                
                train_metrics = model.train(X_train, y_train)
                val_metrics = model.validate(X_val, y_val)
                test_metrics = model.test(X_test, y_test)

                # Compare by F1-score
                val_f1 = val_metrics["f1_score"]
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_config = config_name
                    best_metrics = {
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics
                    }
                    best_model = model

    # After the loop, save the best model and metrics
    if best_model is not None:
        best_model_path = os.path.join(output_dir, f"naive_bayes_{best_config}.pkl")
        best_model.save_model(best_model_path)
        print(f"Best model saved: {best_model_path}")

        # Save metrics to a .txt file
        metrics_file_path = os.path.splitext(best_model_path)[0] + "_metrics.txt"
        with open(metrics_file_path, "w") as f:
            f.write(f"Best Configuration: {best_config}\n\n")

            f.write("=== Training Metrics ===\n")
            for k, v in best_metrics["train_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Validation Metrics ===\n")
            for k, v in best_metrics["val_metrics"].items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== Test Metrics ===\n")
            for k, v in best_metrics["test_metrics"].items():
                f.write(f"{k}: {v}\n")

        print(f"Metrics saved: {metrics_file_path}")
    else:
        print("No valid Naive Bayes model found or no improvement over baseline.")

    return best_metrics, best_config, best_model_path

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_and_clean_data(DATASET_PATH)

    # Preprocess data
    data_features, data_target, data_season = preprocess_data(data, TARGET_COLUMN)

    # Encode target using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(data_target)

    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data_features,  # Features only
        pd.Series(encoded_target, name=TARGET_COLUMN),  
        data_season,  # Season column
        val_size=0.25
    )

    for model_name in MODEL_NAMES:
        hyperparams = load_hyperparameters(f'{HYPERPARAMETERS_FILE_PATH}{model_name}.json')

        if model_name == 'neural_network':
            neural_network_workflow(device, X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'logistic_regression':
            logistic_regression_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'svm':
            svm_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'random_forest':
            random_forest_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'gradient_boosting':
            gradient_boosting_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'knn':
            knn_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'naive_bayes':
            naive_bayes_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

        elif model_name == 'decision_tree':
            decision_tree_workflow(X_train, X_val, X_test, y_train, y_val, y_test, hyperparams)

if __name__ == '__main__':
    main()
