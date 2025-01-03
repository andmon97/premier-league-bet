from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from utils.metrics import compute_metrics  

class NaiveBayesModel:
    def __init__(self, nb_type='gaussian', alpha=1.0, var_smoothing=1e-9):
        """
        Initializes the Naive Bayes model.

        Parameters
        ----------
        nb_type : str, optional
            Which Naive Bayes variant to use. 
            Options: 'gaussian', 'multinomial', 'bernoulli'.
        alpha : float, optional
            Additive smoothing parameter used by 'multinomial' or 'bernoulli'. 
            Ignored for 'gaussian'.
        var_smoothing : float, optional
            Portion of the largest variance of all features added to variances 
            for calculation stability (GaussianNB only). Ignored for other types.
        """
        if nb_type == 'gaussian':
            self.model = GaussianNB(var_smoothing=var_smoothing)
        elif nb_type == 'multinomial':
            self.model = MultinomialNB(alpha=alpha)
        elif nb_type == 'bernoulli':
            self.model = BernoulliNB(alpha=alpha)
        else:
            raise ValueError(f"Unknown nb_type: {nb_type}")

        self.nb_type = nb_type
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes model on the given data.

        Returns
        -------
        dict
            A dictionary containing metrics on the training set.
        """
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, predictions)
        return train_metrics

    def validate(self, X_val, y_val):
        """
        Validate the Naive Bayes model on the given data.

        Returns
        -------
        dict
            A dictionary containing metrics on the validation set.
        """
        predictions = self.model.predict(X_val)
        val_metrics = compute_metrics(y_val, predictions)
        return val_metrics

    def test(self, X_test, y_test):
        """
        Test the Naive Bayes model on the given data.

        Returns
        -------
        dict
            A dictionary containing metrics on the test set.
        """
        predictions = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, predictions)
        return test_metrics

    def save_model(self, path):
        """
        Save the trained Naive Bayes model to a file.
        """
        import joblib
        joblib.dump(self.model, path)
        print(f"Naive Bayes model saved to {path}")

    def load_model(self, path):
        """
        Load a saved Naive Bayes model from a file.
        """
        import joblib
        self.model = joblib.lo
