from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from models.binary.base import BaseMulticlassBinaryClassifier


class CatBoostBinary(BaseMulticlassBinaryClassifier):
    """
    Boosting from CatBoost for multiclass classification
    """

    def __init__(self, task_type='CPU', classes=None):
        model = CatBoostClassifier(verbose=False,
                                   thread_count=-1,
                                   task_type=task_type)
        super().__init__(model, 'CatBoost binary', classes=classes)


class RandomForestBinary(BaseMulticlassBinaryClassifier):
    """
    Random forest for multiclass classification
    """

    def __init__(self, n_estimators=1000, classes=None):
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        super().__init__(model, 'Random forest binary', classes=classes)


class GradientBoostingBinary(BaseMulticlassBinaryClassifier):
    """
    Gradient boosting for multiclass classification
    """

    def __init__(self, n_estimators=100, classes=None):
        model = GradientBoostingClassifier(n_estimators=n_estimators)
        super().__init__(model, 'Gradient boosting binary', classes=classes)
