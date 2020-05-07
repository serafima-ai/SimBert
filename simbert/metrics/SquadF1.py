import collections
from simbert.metrics.metric import Metric
from simbert.metrics.squad_metrics import apply_no_ans_threshold


def compute_f1(a_gold, a_pred):

    gold_toks = a_gold.split()
    pred_toks = a_pred.split()

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1





class SquadF1(Metric):

    def evaluate(self, predictions: list, answers_data: list):

        answers = answers_data[0]
        qas_id_to_has_answer = answers_data[1]
        has_answer_qids = answers_data[2]
        no_answer_qids = answers_data[3]
        no_answer_probs = answers_data[4]
        no_answer_probability_threshold = answers_data[5]

        f1_scores = {}

        for ind, prediction in enumerate(predictions):
            gold_answers, qas_id = answers[ind]

            f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

        f1_threshold = apply_no_ans_threshold(f1_scores, no_answer_probs, qas_id_to_has_answer,
                                              no_answer_probability_threshold)

        return sum(f1_threshold.values()) / len(predictions)


