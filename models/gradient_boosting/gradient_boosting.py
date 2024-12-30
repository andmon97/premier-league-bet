from sklearn.ensemble import GradientBoostingClassifier
from utils.metrics import compute_metrics  

class GradientBoostingModel:
    def __init__(
        self,
        loss='deviance',
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=3
    ):
        """
        Initializes the Gradient Boosting model.

        Parameters
        ----------
        loss : str
            The loss function to be optimized. ('deviance', 'exponential')
        learning_rate : float
            Shrinks the contribution of each tree by learning_rate.
        n_estimators : int
            The number of boosting stages to perform.
        subsample : float
            The fraction of samples to be used for fitting the individual base learners.
        criterion : str
            The function to measure the quality of a split ('friedman_mse', 'mse', 'mae').
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        max_depth : int
            Maximum depth of each tree.
        """
        self.model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )

    def train(self, X_train, y_train):
        """
        Train the Gradient Boosting model on the given data.

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
        Validate the Gradient Boosting model on the given data.

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
        Test the Gradient Boosting model on the given data.

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
        Save the trained Gradient Boosting model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"Gradient Boosting model saved to {path}")

    def load_model(self, path):
        """
        Load a saved Gradient Boosting model from a file.
        """
        import joblib
        self.model = joblib.load(path)
        print(f"Gradient Boosting model loaded from {path}")
