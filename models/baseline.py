import numpy as np
from collections import Counter


class Baseline:
    def __init__(self):
        self.ordered_by_quantity = np.array([])

    def fit(self, X, y):
        self.ordered_by_quantity = np.array([x[0] for x in Counter(y).most_common()])

    def predict(self, X):
        return np.array([self.ordered_by_quantity for _ in X])
