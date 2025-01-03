from sklearn.tree import DecisionTreeClassifier
from utils.metrics import compute_metrics 

class DecisionTreeModel:
    def __init__(
        self,
        criterion='gini',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None
    ):
        """
        Initializes the Decision Tree model.

        Parameters
        ----------
        criterion : str, optional
            The function to measure the quality of a split ('gini' or 'entropy').
        splitter : str, optional
            The strategy used to choose the split at each node ('best' or 'random').
        max_depth : int or None, optional
            The maximum depth of the tree (None means unlimited).
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node.
        max_features : int, float, str or None, optional
            The number of features to consider when looking for the best split.
        """
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )

    def train(self, X_train, y_train):
        """
        Train the Decision Tree model on the given data.

        Returns
        -------
        dict
            Metrics on the training set.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the Decision Tree model on the given data.

        Returns
        -------
        dict
            Metrics on the validation set.
        """
        predictions = self.model.predict(X_val)
        val_metrics = compute_metrics(y_val, predictions)
        return val_metrics

    def test(self, X_test, y_test):
        """
        Test the Decision Tree model on the given data.

        Returns
        -------
        dict
            Metrics on the test set.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        """
        Save the trained Decision Tree model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"Decision Tree model saved to {path}")

    def load_model(self, path):
        """
        Load a saved Decision Tree model from a file.
        """
        import joblib
        self.model = joblib.load(path)
        print(f"Decision Tree model loaded from {path}")
