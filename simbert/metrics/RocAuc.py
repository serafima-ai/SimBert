from dotmap import DotMap
from simbert.metrics.metric import Metric
from sklearn.metrics import roc_auc_score


class RocAuc(Metric):
    avg_supported = [None, 'micro', 'macro', 'samples', 'weighted']

    def evaluate(self, y_true, y_pred) -> roc_auc_score:
        average = self.average if self.average is not None else 'macro'
        return roc_auc_score(y_true, y_pred, average=average)
