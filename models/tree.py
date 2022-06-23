from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier, Pool
from models.base import BaseMulticlassClassifier
from sklearn.model_selection import train_test_split


class CatBoost(BaseMulticlassClassifier):
    """
    Boosting from CatBoost for multiclass classification
    """

    def __init__(self, task_type='CPU', early_stopping_rounds=20, verbose=False, iterations=1000, learning_rate=None, depth=6):
        model = CatBoostClassifier(verbose=verbose,
                                   thread_count=-1,
                                   task_type=task_type,
                                   early_stopping_rounds=early_stopping_rounds,
                                   iterations=iterations,
                                   learning_rate=learning_rate,
                                   depth=depth)

        super().__init__(model, 'CatBoost')

    def fit(self, X, y):
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.7, shuffle=False)
        known_classes = set(y_train)
        mask = [c in known_classes for c in y_validate]
        X_validate = X_validate[mask]
        y_validate = y_validate[mask]
        train_pool = Pool(X_train, y_train)
        validate_pool = Pool(X_validate, y_validate)
        self.model.fit(train_pool, eval_set=validate_pool)


class RandomForest(BaseMulticlassClassifier):
    """
    Random forest for multiclass classification
    """

    def __init__(self, n_estimators=1000):
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        super().__init__(model, 'Random forest')


class GradientBoosting(BaseMulticlassClassifier):
    """
    Gradient boosting for multiclass classification
    """

    def __init__(self, n_estimators=100):
        model = GradientBoostingClassifier(n_estimators=n_estimators)
        super().__init__(model, 'Gradient boosting')
