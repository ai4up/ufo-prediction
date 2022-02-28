import logging
from functools import partial

import utils
import visualizations
import dataset
import preprocessing
from prediction import Classifier, Regressor, PredictorComparison

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgePredictor(Regressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_attribute=dataset.AGE_ATTRIBUTE, initialize_only=True)

        logger.info(f"Dataset standard deviation: {self.df[dataset.AGE_ATTRIBUTE].std()}")
        logger.info(f"Dataset mean age: {self.df[dataset.AGE_ATTRIBUTE].mean()}")

        self._preprocess()

        # The standard deviation in the test set gives us an indication of a baseline. We want to be able to be substantially below that value.
        logger.info(f"Test dataset standard deviation after preprocessing: {self.y_test[dataset.AGE_ATTRIBUTE].std()}")
        logger.info(f"Test dataset mean age after preprocessing: {self.y_test[dataset.AGE_ATTRIBUTE].mean()}")

        self._train()
        self._predict()


    def evaluate(self):
        self.print_model_error()
        _, axis = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        visualizations.plot_regression_error(self.model, ax=axis[0, 0])
        visualizations.plot_histogram(
            self.y_test, self.y_predict, bins=utils.age_bins(self.y_predict), ax=axis[1, 0])
        visualizations.plot_relative_grid(self.y_test, self.y_predict, ax=axis[1, 1])
        plt.show()
        visualizations.plot_grid(self.y_test, self.y_predict)


    def evaluate_regression(self):
        self.evaluate() # for backwards compatibility


class AgeClassifier(Classifier):

    def __init__(self, bins=[], bin_config=None, *args, **kwargs):

        if not bins and bin_config is None or bins and bin_config:
            logger.exception('Please either specify the bins or define a bin config to have them generated automatically.')

        self.bins = bins or utils.generate_bins(bin_config)

        super().__init__(*args, **kwargs, target_attribute=dataset.AGE_ATTRIBUTE, labels=utils.generate_labels(self.bins), initialize_only=True)


        logger.info(f'Generated bins: {self.bins}')
        logger.info(f'Generated bins with the following labels: {self.labels}')

        self.preprocessing_stages.append(partial(preprocessing.categorize_age, bins=self.bins))

        self._preprocess()
        self._train()
        self._predict()


    def evaluate(self):
        self.print_classification_report()
        _, axis = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        visualizations.plot_classification_error(self.model, multiclass=self.multiclass, ax=axis[0, 0])
        visualizations.plot_log_loss(self.model, multiclass=self.multiclass, ax=axis[0, 1])
        visualizations.plot_histogram(self.y_test, self.y_predict[[dataset.AGE_ATTRIBUTE]], bins=list(range(0, len(self.bins))), bin_labels=self.labels, ax=axis[1, 0])
        visualizations.plot_confusion_matrix(self.y_test, self.y_predict[[dataset.AGE_ATTRIBUTE]], class_labels=self.labels, ax=axis[1, 1])
        plt.show()


    def evaluate_classification(self):
        self.evaluate() # for backwards compatibility


class AgePredictorComparison(PredictorComparison):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, predictor=AgePredictor)


    def evaluate(self):
        age_distributions = {}
        comparison_metrics = []
        for name, predictor in self.predictors.items():
            eval_metrics = {}
            eval_metrics['name'] =  name
            eval_metrics['R2'] =  metrics.r2_score(predictor.y_test, predictor.y_predict)
            eval_metrics['MAE'] =  metrics.mean_absolute_error(predictor.y_test, predictor.y_predict)
            eval_metrics['RMSE'] =  np.sqrt(metrics.mean_squared_error(predictor.y_test, predictor.y_predict))
            comparison_metrics.append(eval_metrics)

            age_distributions[f'{name}_predict'] = predictor.y_predict[dataset.AGE_ATTRIBUTE]
            age_distributions[f'{name}_test'] = predictor.y_test[dataset.AGE_ATTRIBUTE]

        visualizations.plot_distribution(age_distributions)
        return pd.DataFrame(comparison_metrics).sort_values(by=['R2'])


    def evaluate_comparison(self):
        self.evaluate() # for backwards compatibility


    def determine_predictor_identifier(self, param_name, param_value, baseline_value):
        if param_name == 'preprocessing_stages':
            additional_stages = list(set(param_value) - set(baseline_value))
            stage_names = [getattr(stage, '__name__', stage) for stage in additional_stages]
            return f'add_preprocessing:{stage_names}'

        return f'{param_name}_{param_value}'
