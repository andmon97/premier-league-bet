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
        Train the logistic regression model.

        Parameters
        ----------
        X_train : array-like
            Training feature set.
        y_train : array-like
            Training target labels.

        Returns
        -------
        None
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the logistic regression model and calculate metrics.

        Parameters
        ----------
        X_val : array-like
            Validation feature set.
        y_val : array-like
            Validation target labels.

        Returns
        -------
        dict
            A dictionary containing accuracy and loss.
        """
        predictions = self.model.predict(X_val)
        validation_metrics = compute_metrics(y_val, predictions)
        return validation_metrics

    def test(self, X_test, y_test):
        """
        Test the logistic regression model and calculate accuracy.

        Parameters
        ----------
        X_test : array-like
            Test feature set.
        y_test : array-like
            Test target labels.

        Returns
        -------
        dict
            A dictionary containing accuracy.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return 

    def save_model(self, path):
        """
        Save the trained model to a file.

        Parameters
        ----------
        path : str
            Path to save the model file.

        Returns
        -------
        None
        """
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a saved logistic regression model from a file.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns
        -------
        None
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
