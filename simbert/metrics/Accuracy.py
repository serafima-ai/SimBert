from simbert.metrics.metric import Metric
from sklearn.metrics import accuracy_score


class Accuracy(Metric):

    def evaluate(self, y_true, y_pred) -> accuracy_score:
        return accuracy_score(y_true, y_pred)
