from sklearn.neighbors import KNeighborsClassifier
from utils.metrics import compute_metrics
  
class KNNModel:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2):
        """
        Initializes the KNN model.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use.
        weights : str
            Weight function used in prediction ('uniform' or 'distance').
        algorithm : str
            Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size : int
            Leaf size passed to the underlying BallTree or KDTree.
        p : int
            Power parameter for the Minkowski metric. (p=1 for Manhattan, p=2 for Euclidean)
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )

    def train(self, X_train, y_train):
        """
        Train the KNN model on the given data.

        Returns
        -------
        dict
            A dictionary of computed metrics on the training set.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the KNN model on the given data.

        Returns
        -------
        dict
            A dictionary of computed metrics on the validation set.
        """
        predictions = self.model.predict(X_val)
        val_metrics = compute_metrics(y_val, predictions)
        return val_metrics

    def test(self, X_test, y_test):
        """
        Test the KNN model on the given data.

        Returns
        -------
        dict
            A dictionary of computed metrics on the test set.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        """
        Save the trained KNN model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"KNN model saved to {path}")

    def load_model(self, path):
        """
        Load a saved KNN model from a file.
        """
        import joblib
        self.model = joblib.load(path)
        print(f"KNN model loaded from {path}")
