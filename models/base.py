import numpy as np


class BaseMulticlassClassifier:
    """
    Model base for multiclass classification
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return np.array([np.flip(self.model.classes_[np.argsort(proba)]) for proba in self.model.predict_proba(X)])
