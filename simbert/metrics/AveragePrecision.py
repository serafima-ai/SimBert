from dotmap import DotMap
from simbert.metrics.metric import Metric
from sklearn.metrics import recall_score


class AveragePrecision(Metric):
    avg_supported = [None, 'micro', 'macro', 'samples', 'weighted']

    def evaluate(self, y_true, y_pred) -> recall_score:
        average = self.average if self.average is not None else 'macro'
        return recall_score(y_true, y_pred, average=average)
