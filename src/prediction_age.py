import logging
from functools import partial

import utils
import visualizations
import dataset
import preprocessing
from prediction import Classifier, Regressor, PredictorComparison

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgePredictor(Regressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_attribute=dataset.AGE_ATTRIBUTE)


    def _pre_preprocess_analysis_hook(self):
        logger.info(f"Dataset standard deviation: {self.df[self.target_attribute].std()}")
        logger.info(f"Dataset mean age: {self.df[self.target_attribute].mean()}")
        logger.info(f'Training dataset length: {len(self.df_train)}')
        logger.info(f'Test dataset length: {len(self.df_test)}')


    def _post_preprocess_analysis_hook(self):
        # The standard deviation in the test set gives us an indication of a
        # baseline. We want to be able to be substantially below that value.
        logger.info(f"Test dataset standard deviation after preprocessing: {self.df_test[self.target_attribute].std()}")
        logger.info(f"Test dataset mean age after preprocessing: {self.df_test[self.target_attribute].mean()}")
        logger.info(f'Training dataset length after preprocessing: {len(self.df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(self.df_test)}')


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
        self.evaluate()  # for backwards compatibility


class AgeClassifier(Classifier):

    def __init__(self, bins=[], bin_config=None, *args, **kwargs):

        if not bins and bin_config is None or bins and bin_config:
            logger.exception('Please either specify the bins or define a bin config to have them generated automatically.')

        self.bins = bins or utils.generate_bins(bin_config)

        super().__init__(*args, **kwargs, target_attribute=dataset.AGE_ATTRIBUTE,
                         labels=utils.generate_labels(self.bins), initialize_only=True)

        logger.info(f'Generated bins: {self.bins}')
        logger.info(f'Generated bins with the following labels: {self.labels}')

        self.metric_target_attribute = 'age_metric'
        self.preprocessing_stages.append(partial(preprocessing.categorize_age, bins=self.bins, metric_col=self.metric_target_attribute))

        self._e2e_training()


    def evaluate(self):
        self.print_classification_report()
        _, axis = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        visualizations.plot_classification_error(self.model, multiclass=self.multiclass, ax=axis[0, 0])
        visualizations.plot_log_loss(self.model, multiclass=self.multiclass, ax=axis[0, 1])
        visualizations.plot_histogram(self.y_test, self.y_predict[[self.target_attribute]], bins=list(
            range(0, len(self.bins))), bin_labels=self.labels, ax=axis[1, 0])
        visualizations.plot_confusion_matrix(
            self.y_test, self.y_predict[[self.target_attribute]], class_labels=self.labels, ax=axis[1, 1])
        plt.show()


    def evaluate_classification(self):
        self.evaluate()  # for backwards compatibility


class AgePredictorComparison(PredictorComparison):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, predictor=AgePredictor)


    def evaluate(self, include_plot=False, include_error_distribution=False, include_spatial_autocorrelation=False):
        age_distributions = {}
        comparison_metrics = []
        for name, predictor in self.predictors.items():
            eval_metrics = {}
            eval_metrics['name'] = name
            eval_metrics['R2'] = predictor.r2()
            eval_metrics['MAE'] = predictor.mae()
            eval_metrics['RMSE'] = predictor.rmse()

            if include_error_distribution:
                eval_metrics['skew'] = scipy.stats.skew(predictor.y_test - predictor.y_predict)[0]
                eval_metrics['kurtosis'] = scipy.stats.kurtosis(predictor.y_test - predictor.y_predict)[0]

            if include_spatial_autocorrelation:
                eval_metrics['residuals_moranI_KNN'] = predictor.spatial_autocorrelation_moran('error', 'knn').I
                eval_metrics['residuals_moranI_block'] = predictor.spatial_autocorrelation_moran('error', 'block').I
                eval_metrics['residuals_moranI_distance'] = predictor.spatial_autocorrelation_moran('error', 'distance').I
                eval_metrics['prediction_moranI_KNN'] = predictor.spatial_autocorrelation_moran(predictor.target_attribute, 'knn').I
                eval_metrics['prediction_moranI_block'] = predictor.spatial_autocorrelation_moran(predictor.target_attribute, 'block').I
                eval_metrics['prediction_moranI_distance'] = predictor.spatial_autocorrelation_moran(predictor.target_attribute, 'distance').I

            if include_plot:
                age_distributions[f'{name}_predict'] = predictor.y_predict[predictor.target_attribute]
                age_distributions[f'{name}_test'] = predictor.y_test[predictor.target_attribute]

            comparison_metrics.append(eval_metrics)

        if include_plot:
            visualizations.plot_distribution(age_distributions)

        return pd.DataFrame(comparison_metrics).sort_values(by=['R2'])


    def evaluate_comparison(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)  # for backwards compatibility


    def determine_predictor_identifier(self, param_name, param_value, baseline_value):
        if param_name == 'preprocessing_stages':
            additional_stages = list(set(param_value) - set(baseline_value))
            stage_names = [getattr(stage, '__name__', stage) for stage in additional_stages]
            return f'add_preprocessing:{stage_names}'

        return f'{param_name}_{param_value}'


class AgeClassifierComparison(PredictorComparison):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, predictor=AgeClassifier)


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
