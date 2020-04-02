from simbert.metrics.metric import Metric
from sklearn.metrics import precision_score


class Precision(Metric):

    def evaluate(self, y_true, y_pred) -> precision_score:
        average = self.average if self.average is not None else 'binary'
        return precision_score(y_true, y_pred, average=average)
