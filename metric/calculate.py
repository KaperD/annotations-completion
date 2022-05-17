from metric import Metric


def calculate_and_print(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    metric = Metric(predicted, y_test)
    print(f'Count: {len(y_test)}')
    for i in range(1, 6):
        print(f'Top {i}: {metric.top_i(i)}')
    for i in range(1, 2):
        print(f'Top1 {i}: {metric.top1_i(i)}')
    print(f'Mean: {metric.mean}')


def caclulate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return Metric(predicted, y_test)
