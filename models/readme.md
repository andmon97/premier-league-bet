# Models Overview

This document provides an overview of the implemented and available models for classification in the project. Each model is designed for specific scenarios, offering different strengths and trade-offs. The models include:

## 1. Neural Network
- **Description**: A customizable neural network implemented using PyTorch, allowing dynamic configuration of hyperparameters such as the number of layers, neurons per layer, activation functions, and more.
- **Key Features**:
  - Supports grid search for hyperparameter optimization.
  - Suitable for complex, non-linear relationships in data.
  - Requires more computational resources compared to traditional algorithms.

## 2. Logistic Regression
- **Description**: A linear model for binary or multi-class classification.
- **Key Features**:
  - Efficient and interpretable.
  - Serves as a good baseline model.
  - Assumes a linear relationship between features and the target variable.

## 3. Support Vector Machines (SVM)
- **Description**: A robust classifier for high-dimensional spaces.
- **Key Features**:
  - Effective for both linear and non-linear classification tasks.
  - Kernels like RBF and polynomial allow handling of non-linear data.
  - Computationally intensive for large datasets.

## 4. Random Forest
- **Description**: An ensemble model that combines multiple decision trees.
- **Key Features**:
  - Robust to overfitting.
  - Handles both numerical and categorical data well.
  - Suitable for large datasets.

## 5. Gradient Boosting Machines
- **Description**: A sequential ensemble model that builds trees incrementally to minimize errors.
- **Key Features**:
  - Often more accurate than Random Forest but slower to train.
  - Works well with numerical data.

## 6. K-Nearest Neighbors (KNN)
- **Description**: A non-parametric model that classifies based on the majority label of the nearest neighbors.
- **Key Features**:
  - Simple and interpretable.
  - Sensitive to the choice of `k` and scaling of data.
  - Computationally expensive for large datasets.

## 7. Naive Bayes
- **Description**: A probabilistic classifier based on Bayes' theorem.
- **Key Features**:
  - Fast and efficient for text classification and categorical data.
  - Assumes feature independence (may not hold in real-world data).

## 8. Decision Trees
- **Description**: A tree-based model for classification and regression.
- **Key Features**:
  - Simple and interpretable.
  - Prone to overfitting; often used as a base model in ensembles.

## 9. AdaBoost
- **Description**: An ensemble model that combines weak learners iteratively to improve accuracy.
- **Key Features**:
  - Works well with binary and multi-class classification.
  - Sensitive to noisy data and outliers.

## 10. XGBoost
- **Description**: An optimized implementation of gradient boosting.
- **Key Features**:
  - Extremely fast and powerful.
  - Requires external library (XGBoost).

---

### Adding a New Model
To add a new model:
1. Implement the model under the `models` folder.
2. Create a corresponding JSON file for hyperparameters in the `models/hyperparameters/` folder with the naming convention `hyperparameters_{MODEL_NAME}.json`.
3. Update the training script to include the new model.

### Note
Each model can be evaluated using the metrics and visualization tools provided in the `utils` module. For details on metrics and visualizations, refer to the `utils/metrics.py` and `utils/visualization.py` files.

