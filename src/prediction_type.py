import logging
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt

import visualizations
import dataset
import preprocessing
from prediction import Classifier, PredictorComparison

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TypeClassifier(Classifier):

    def __init__(self, labels=None, *args, **kwargs):
        self.labels = labels or ['res', 'non-res']

        super().__init__(*args, **kwargs, target_attribute=dataset.TYPE_ATTRIBUTE, labels=self.labels, initialize_only=True)

        self.preprocessing_stages.insert(0, preprocessing.remove_buildings_with_unknown_type)
        self.preprocessing_stages.append(partial(preprocessing.categorical_to_int, var=self.target_attribute, labels=self.labels))

        self._e2e_training()


    def evaluate(self):
        self.print_model_error()
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
        super().__init__(*args, **kwargs, predictor_type=TypeClassifier)


    def _evaluate_experiment(self, name):
        predictors = self.predictors[name]
        eval_metrics = {}
        eval_metrics['name'] = name
        eval_metrics['MCC'] = self._mean(predictors, 'mcc')
        eval_metrics['MCC_std'] = self._std(predictors, 'mcc')
        eval_metrics['F1'] = self._mean(predictors, 'f1')
        eval_metrics['F1_std'] = self._std(predictors, 'f1')

        for idx, label in enumerate(predictors[0].labels):
            eval_metrics[f'Recall_{label}'] = self._mean(predictors, 'recall', idx)

        for seed, predictor in enumerate(predictors):
            eval_metrics[f'MCC_seed_{seed}'] = predictor.mcc()

        return eval_metrics


    def evaluate(self, include_plot=False):
        if include_plot:
            evals_results = {}
            for name, predictors in self.predictors.items():
                eval_metric = 'merror' if predictors[0].multiclass else 'error'
                evals_results[f'{name}_train'] = predictors[0].evals_result['validation_0'][eval_metric]
                evals_results[f'{name}_test'] = predictors[0].evals_result['validation_1'][eval_metric]

            _, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
            visualizations.plot_models_classification_error(evals_results, ax=axis)
            plt.show()

        return pd.DataFrame(self.comparison_metrics).sort_values(by=['MCC'])


    def evaluate_comparison(self):
        return self.evaluate()  # for backwards compatibility
