import numpy as np
from simbert.metrics.metric import Metric
from sklearn.metrics import f1_score


class F1(Metric):

    def evaluate(self, y_true, y_pred) -> f1_score:
        average = self.average if self.average is not None else 'binary'

        return f1_score(y_true, y_pred, average=average, labels=np.unique(y_pred))
