from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from models.base import BaseMulticlassClassifier


class SVM(BaseMulticlassClassifier):
    """
    SVM for multiclass classification
    """

    def __init__(self, c=1, max_iter=1000, need_scaler=False):
        model = SVC(C=c, max_iter=max_iter, probability=True, kernel='linear')
        if need_scaler:
            model = make_pipeline(StandardScaler(), model)
        super().__init__(model, 'SVM')


class LinearSVM(BaseMulticlassClassifier):
    """
    Another version of SVM for multiclass classification based on LinearSVC
    """

    def __init__(self, c=1, max_iter=1000, need_scaler=False):
        model = LinearSVC(C=c, max_iter=max_iter)
        if need_scaler:
            model = make_pipeline(StandardScaler(), model)
        model = CalibratedClassifierCV(model)
        super().__init__(model, 'Linear SVM')


class LogisticReg(BaseMulticlassClassifier):
    """
    Logistic regression for multiclass classification
    """

    def __init__(self, c=1, max_iter=1000, solver='lbfgs', need_scaler=False):
        model = LogisticRegression(C=c, max_iter=max_iter, solver=solver, n_jobs=-1)
        if need_scaler:
            model = make_pipeline(StandardScaler(), model)
        super().__init__(model, 'Logistic reg')

