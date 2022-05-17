import statistics


class Metric:
    def __init__(self, predicted_y, expected_y):
        self.orders = [(list(predicted_y[i]) + [expected_y[i]]).index(expected_y[i]) + 1 for i in
                       range(len(expected_y))]
        self.orders1 = [
            (list(filter(lambda x: x.startswith(expected_y[i][0]), predicted_y[i])) + [expected_y[i]]).index(
                expected_y[i]) + 1 for i in range(len(expected_y))]
        self.mean = statistics.mean(self.orders)

    def top_i(self, i):
        """
        Top i accuracy
        """
        return sum(map(lambda x: x <= i, self.orders)) / len(self.orders)

    def top1_i(self, i):
        """
        Top i accuracy with 1 letter typed
        """
        return sum(map(lambda x: x <= i, self.orders1)) / len(self.orders1)
