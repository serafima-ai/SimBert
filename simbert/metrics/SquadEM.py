from simbert.metrics.metric import Metric
from simbert.metrics.squad_metrics import apply_no_ans_threshold


def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)


class SquadEM(Metric):

    def evaluate(self, predictions: list, answers_data: list):
        answers = answers_data[0]
        qas_id_to_has_answer = answers_data[1]
        has_answer_qids = answers_data[2]
        no_answer_qids = answers_data[3]
        no_answer_probs = answers_data[4]
        no_answer_probability_threshold = answers_data[5]

        exact_scores = {}

        for ind, prediction in enumerate(predictions):
            gold_answers, qas_id = answers[ind]

            exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)

        exact_threshold = apply_no_ans_threshold(exact_scores, no_answer_probs, qas_id_to_has_answer,
                                              no_answer_probability_threshold)

        return sum(exact_threshold.values()) / len(predictions)
