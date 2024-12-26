from sklearn import svm
from utils.metrics import compute_metrics  

class SVMModel:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3):
        """
        Constructor for the SVMModel class.

        Parameters
        ----------
        C : float
            Regularization parameter. The strength of the regularization is inversely
            proportional to C.
        kernel : str
            Specifies the kernel type to be used in the algorithm (‘linear’, ‘poly’,
            ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable).
        gamma : str or float
            Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’. Default is 'scale'.
        degree : int
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        """
        self.model = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True)

    def train(self, X_train, y_train):
        """
        Train the SVM model on the given data.

        Returns
        -------
        train_metrics : dict
            Metrics calculated on the training set after fitting the model.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the SVM model on the given data.

        Returns
        -------
        val_metrics : dict
            Metrics calculated on the validation set.
        """
        predictions = self.model.predict(X_val)
        val_metrics = compute_metrics(y_val, predictions)
        return val_metrics

    def test(self, X_test, y_test):
        """
        Test the SVM model on the given data.

        Returns
        -------
        test_metrics : dict
            Metrics calculated on the test set.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        """
        Save the trained SVM model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"SVM Model saved to {path}")

    def load_model(self, path):
        """
        Load a saved SVM model from a file.
        """
        import joblib
        self.model = joblib.load(path)
        print(f"SVM Model loaded from {path}")
