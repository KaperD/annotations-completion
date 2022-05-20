import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BaseMulticlassBinaryClassifier:
    """
    Model base for binary multiclass classification
    """

    def __init__(self, model, name, classes=None):
        self.model = model
        self.name = name
        self.classes = classes
        self.vectorizer = CountVectorizer()

    def fit(self, X, y):
        new_X = []
        names = []
        new_y = []
        classes = list(set(y))
        if self.classes is None:
            self.classes = classes
        self.vectorizer.fit(classes)
        for i in range(len(X)):
            for c in classes:
                new_X.append(X[i])
                names.append(c)
                new_y.append(1 if c == y[i] else 0)
        names_column = self.vectorizer.transform(np.array(names)).toarray()
        new_X = np.array(new_X)
        new_y = np.array(new_y)
        new_X = np.concatenate((new_X, names_column), axis=1)
        self.model.fit(new_X, new_y)

    def predict(self, X, classes=None):
        if classes is None:
            classes = self.classes
        classes = np.array(classes)
        result = []
        i, = np.where(self.model.classes_ == 1)
        i = i[0]
        names_column = self.vectorizer.transform(classes).toarray()
        for x in X:
            xs = np.array([x for _ in names_column])
            xs = np.concatenate((xs, names_column), axis=1)
            predictions = [a[i] for a in self.model.predict_proba(xs)]
            result.append(classes[np.flip(np.argsort(predictions))])
        return np.array(result)
