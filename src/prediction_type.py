import logging
from functools import partial

import utils
import visualizations
import dataset
import preprocessing
from prediction import Classifier, PredictorComparison

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TypeClassifier(Classifier):

    def __init__(self, binary=True, *args, **kwargs):
        self.binary = binary
        labels = ['res', 'non-res'] if binary else dataset.BUILDING_TYPES

        super().__init__(*args, **kwargs, target_attribute=dataset.TYPE_ATTRIBUTE, labels=labels, initialize_only=True)

        self.preprocessing_stages.append(partial(preprocessing.categorical_to_int, var=self.target_attribute))

        self._e2e_training()


    def evaluate(self):
        self.print_classification_report()
        _, axis = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        visualizations.plot_classification_error(self.model, multiclass=self.multiclass, ax=axis[0, 0])
        visualizations.plot_log_loss(self.model, multiclass=self.multiclass, ax=axis[0, 1])
        visualizations.plot_histogram(self.y_test, self.y_predict[[self.target_attribute]], bins=[
                                      0, 0.5, 1], bin_labels=self.labels, ax=axis[1, 0])
        visualizations.plot_confusion_matrix(
            self.y_test, self.y_predict[[self.target_attribute]], class_labels=self.labels, ax=axis[1, 1])
        plt.show()


    def evaluate_classification(self):
        self.evaluate()  # for backwards compatibility


class TypeClassifierComparison(PredictorComparison):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, predictor=TypeClassifier)


    def evaluate(self):
        evals_results = {}
        comparison_metrics = []
        for name, predictor in self.predictors.items():
            test, pred = predictor.y_test, predictor.y_predict[[predictor.target_attribute]]

            eval_metrics = {}
            eval_metrics['name'] = name
            eval_metrics['MCC'] = metrics.matthews_corrcoef(test, pred)
            eval_metrics['F1'] = metrics.f1_score(test, pred, average='macro')
            for idx, label in enumerate(predictor.labels):
                eval_metrics[f'Recall_{label}'] = metrics.recall_score(
                    test, pred, pos_label=idx, labels=[idx], average='macro')

            comparison_metrics.append(eval_metrics)

            eval_metric = 'merror' if predictor.multiclass else 'error'
            evals_results[f'{name}_train'] = predictor.evals_result['validation_0'][eval_metric]
            evals_results[f'{name}_test'] = predictor.evals_result['validation_1'][eval_metric]

        _, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        visualizations.plot_models_classification_error(evals_results, ax=axis)
        plt.show()

        return pd.DataFrame(comparison_metrics).sort_values(by=['MCC'])


    def evaluate_comparison(self):
        return self.evaluate()  # for backwards compatibility
