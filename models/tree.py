from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from base import BaseMulticlassClassifier


class CatBoost(BaseMulticlassClassifier):
    """
    Boosting from CatBoost for multiclass classification
    """

    def __init__(self, task_type='CPU'):
        model = CatBoostClassifier(verbose=False,
                                   thread_count=-1,
                                   task_type=task_type)
        super().__init__(model)


class RandomForest(BaseMulticlassClassifier):
    """
    Random forest for multiclass classification
    """

    def __init__(self, n_estimators=1000):
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        super().__init__(model)


class GradientBoosting(BaseMulticlassClassifier):
    """
    Gradient boosting for multiclass classification
    """

    def __init__(self, n_estimators=1000):
        model = GradientBoostingClassifier(n_estimators=n_estimators)
        super().__init__(model)
