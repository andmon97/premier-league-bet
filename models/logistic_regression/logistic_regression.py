from sklearn.linear_model import LogisticRegression
from utils.metrics import compute_metrics
import joblib

class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=100):
        """
        Constructor for LogisticRegressionModel class.

        Parameters
        ----------
        penalty : str
            The norm used in the penalization. Default is 'l2'.
        C : float
            Inverse of regularization strength; smaller values specify stronger regularization.
        solver : str
            Algorithm to use in the optimization problem. Default is 'lbfgs'.
        max_iter : int
            Maximum number of iterations for the solver. Default is 100.
        """
        self.model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

    def train(self, X_train, y_train):
        """
        Train the model on the given data.

        Parameters
        ----------
        X_train : array-like
            The training data.
        y_train : array-like
            The target values.

        Returns
        -------
        train_metrics : dict
            A dictionary containing the precision, recall, f1 score, and accuracy of the model on the training data.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the model on the given data.

        Parameters
        ----------
        X_val : array-like
            The validation data.
        y_val : array-like
            The target values.

        Returns
        -------
        validation_metrics : dict
            A dictionary containing the precision, recall, f1 score, and accuracy of the model on the validation data.
        """
        predictions = self.model.predict(X_val)
        validation_metrics = compute_metrics(y_val, predictions)
        return validation_metrics

    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
