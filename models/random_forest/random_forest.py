from sklearn.ensemble import RandomForestClassifier
from utils.metrics import compute_metrics 

class RandomForestModel:
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initializes the Random Forest model.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        criterion : str
            The function to measure the quality of a split ('gini' or 'entropy').
        max_depth : int or None
            The maximum depth of the tree (None for unlimited).
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest model on the given data.

        Returns
        -------
        dict
            Metrics calculated on the training set.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the Random Forest model on the given data.

        Returns
        -------
        dict
            Metrics calculated on the validation set.
        """
        predictions = self.model.predict(X_val)
        val_metrics = compute_metrics(y_val, predictions)
        return val_metrics

    def test(self, X_test, y_test):
        """
        Test the Random Forest model on the given data.

        Returns
        -------
        dict
            Metrics calculated on the test set.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        """
        Save the trained Random Forest model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"Random Forest model saved to {path}")

    def load_model(self, path):
        """
        Load a saved Random Forest model from a file.
        """
        import joblib
        self.model = joblib.load(path)
        print(f"Random Forest model loaded from {path}")
