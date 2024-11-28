import pickle
import time
import logging
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
import visualizations
import dataset
import preprocessing
import energy_modeling
from prediction import Predictor, Classifier, Regressor, PredictorComparison

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


    def eval_metrics(self):
        eval_df = super().eval_metrics()

        r2, mape = self.energy_error()
        eval_df.at['total', 'energy_r2'] = r2
        eval_df.at['total', 'energy_mape'] = mape

        bins = [0, 5, 10, 20]
        hist = self.error_cum_hist(bins)

        for idx, bin in enumerate(bins[1:]):
            eval_df.at['total', f'within_{bin}_years'] = hist.flat[idx]

        if self.cross_validation_split:
            fold_histograms = self.error_cum_hist(bins, across_folds=True)
            fold_energy_errors = self.energy_error(across_folds=True)
            logger.info(fold_energy_errors)

            for fold, hist in enumerate(fold_histograms):
                eval_df.at[f'fold_{fold}', 'energy_r2'] = fold_energy_errors[fold][0]
                eval_df.at[f'fold_{fold}', 'energy_mape'] = fold_energy_errors[fold][1]

                for idx, bin in enumerate(bins[1:]):
                    eval_df.at[f'fold_{fold}', f'within_{bin}_years'] = hist.flat[idx]

        return eval_df


    @Predictor.cv_aware
    def print_model_error(self):
        super().print_model_error()
        r2, mape = self.energy_error()
        print(f'R2: {r2:.4f}')
        print(f'MAPE: {mape:.4f}')


    def evaluate(self):
        ticks = [1920, 1940, 1960, 1980, 2000]
        bins = utils.age_bins(self.y_predict)
        self.print_model_error()
        _, axis = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        visualizations.plot_regression_error(self.model, ax=axis[0])
        visualizations.plot_histogram(
            self.y_test, self.y_predict, bins=bins, ax=axis[1])
        visualizations.plot_relative_grid(self.y_test, self.y_predict, dataset.AGE_ATTRIBUTE, bins, ticks)
        plt.show()


    def evaluate_regression(self):
        self.evaluate()  # for backwards compatibility


    @Predictor.cv_aware
    def energy_error(self):
        try:
            y_true = pd.concat([self.y_test, self.aux_vars_test], axis=1, join="inner")
            y_pred = pd.concat([self.y_predict, self.aux_vars_test], axis=1, join="inner")
            return energy_modeling.calculate_energy_error(y_true, y_pred)
        except Exception as e:
            logger.error(f'Failed to calculate energy error: {e}')
            return np.nan, np.nan


    @staticmethod
    def load(path):
        predictor = pickle.load(open(path, 'rb'))

        if isinstance(predictor, AgePredictor):
            return predictor

        logger.error(f'The object loaded from {path} is not an AgePredictor instance.')


class AgeClassifier(Classifier):

    def __init__(self, bins=[], bin_config=None, resampling=None, *args, **kwargs):

        if not bins and bin_config is None or bins and bin_config:
            logger.exception('Please either specify the bins or define a bin config to have them generated automatically.')

        self.bins = bins or utils.generate_bins(bin_config)
        self.resampling = resampling

        super().__init__(*args, **kwargs, target_attribute=dataset.AGE_ATTRIBUTE,
                         labels=utils.generate_labels(self.bins), initialize_only=True, validate_labels=False)

        logger.info(f'Generated bins: {self.bins}')
        logger.info(f'Generated bins with the following labels: {self.labels}')

        self.metric_target_attribute = 'age_metric'
        self.preprocessing_stages.append(partial(preprocessing.categorize_age, bins=self.bins, metric_col=self.metric_target_attribute))
        if self.resampling:
            self.preprocessing_stages.append(self.resampling)

        self._e2e_training()


    def eval_metrics(self):
        eval_df = super().eval_metrics()

        if self.bins in dataset.TABULA_AGE_BINS.values():
            r2, mape = self.energy_error()
            eval_df.at['total', 'energy_r2'] = r2
            eval_df.at['total', 'energy_mape'] = mape

        if self.cross_validation_split:
            fold_energy_errors = self.energy_error(across_folds=True)
            for fold, energy_error in enumerate(fold_energy_errors):
                eval_df.at[f'fold_{fold}', 'energy_r2'] = energy_error[0]
                eval_df.at[f'fold_{fold}', 'energy_mape'] = energy_error[1]

        return eval_df

    @Predictor.cv_aware
    def print_model_error(self):
        super().print_model_error()
        if self.bins in dataset.TABULA_AGE_BINS.values():
            r2, mape = self.energy_error()
            print(f'R2: {r2:.4f}')
            print(f'MAPE: {mape:.4f}')


    def evaluate(self):
        self.print_model_error()
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


    @Predictor.cv_aware
    def energy_error(self):
        try:
            y_true = pd.concat([self.y_test, self.aux_vars_test], axis=1, join='inner')
            y_pred = pd.concat([self.y_predict, self.aux_vars_test], axis=1, join='inner')
            return energy_modeling.calculate_energy_error(y_true, y_pred, labels=self.labels)
        except Exception as e:
            logger.error(f'Failed to calculate energy error: {e}')
            return np.nan, np.nan


    @staticmethod
    def load(path):
        predictor = pickle.load(open(path, 'rb'))

        if isinstance(predictor, AgeClassifier):
            return predictor

        logger.error(f'The object loaded from {path} is not an AgeClassifier instance.')


class AgePredictorComparison(PredictorComparison):

    def __init__(self, compare_error_distribution=False, compare_fit_time=False, compare_energy_error=False, compare_spatial_autocorrelation=False, compare_classification_error=False, *args, **kwargs) -> None:
        self.compare_fit_time = compare_fit_time
        self.compare_error_distribution = compare_error_distribution
        self.compare_energy_error = compare_energy_error
        self.compare_spatial_autocorrelation = compare_spatial_autocorrelation
        self.compare_classification_error = compare_classification_error
        super().__init__(*args, **kwargs, predictor_type=AgePredictor)


    def _evaluate_experiment(self, name):
        predictors = self.predictors[name]
        eval_metrics = {}
        eval_metrics['name'] = name
        # average eval metrics across seeds
        eval_metrics['R2'] = self._mean(predictors, 'r2')
        eval_metrics['R2_std'] = self._std(predictors, 'r2')
        eval_metrics['MAE'] = self._mean(predictors, 'mae')
        eval_metrics['MAE_std'] = self._std(predictors, 'mae')
        eval_metrics['RMSE'] = self._mean(predictors, 'rmse')
        eval_metrics['RMSE_std'] = self._std(predictors, 'rmse')

        bins = [0, 5, 10, 20]
        # bins = [0, 0.5, 1, 2.5, 5, 10]
        hist = self._mean(predictors, 'error_cum_hist', bins)

        for idx, bin in enumerate(bins[1:]):
            eval_metrics[f'within_{bin}_years'] = hist.flat[idx]

        for seed, predictor in enumerate(predictors):
            eval_metrics[f'R2_seed_{seed}'] = predictor.r2()

        if self.compare_fit_time:
            eval_metrics['e2e_time'] = (time.time() - self.time_start) / len(predictors)

        if self.compare_error_distribution:
            eval_metrics['skew'] = self._mean(predictors, 'skew')
            eval_metrics['kurtosis'] = self._mean(predictors, 'kurtosis')

        if self.compare_energy_error:
            r2, mape = self._mean(predictors, 'energy_error')
            eval_metrics['energy_r2'] = r2
            eval_metrics['energy_mape'] = mape

        if self.compare_spatial_autocorrelation:
            eval_metrics['residuals_moranI_KNN'] = predictors[0].spatial_autocorrelation_moran('error', 'knn').I
            eval_metrics['residuals_moranI_block'] = predictors[0].spatial_autocorrelation_moran('error', 'block').I
            eval_metrics['residuals_moranI_distance'] = predictors[0].spatial_autocorrelation_moran('error', 'distance').I
            eval_metrics['prediction_moranI_KNN'] = predictors[0].spatial_autocorrelation_moran(predictors[0].target_attribute, 'knn').I
            eval_metrics['prediction_moranI_block'] = predictors[0].spatial_autocorrelation_moran(predictors[0].target_attribute, 'block').I
            eval_metrics['prediction_moranI_distance'] = predictors[0].spatial_autocorrelation_moran(predictors[0].target_attribute, 'distance').I

        if self.compare_classification_error:
            for bin_size in [5, 10, 20]:
                bins = utils.generate_bins((1900, 2020, bin_size))
                eval_metrics[f'MCC_{bin_size}'] = self._mean(predictors, 'mcc', bins)
                eval_metrics[f'MCC_std_{bin_size}'] = self._std(predictors, 'mcc', bins)

        return eval_metrics


    def evaluate(self, include_plot=False):
        if include_plot:
            age_distributions = {}
            for name, predictors in self.predictors.items():
                age_distributions[f'{name}_predict'] = predictors[0].y_predict[predictors[0].target_attribute]
                age_distributions[f'{name}_test'] = predictors[0].y_test[predictors[0].target_attribute]
            visualizations.plot_distribution(age_distributions)

        return pd.DataFrame(self.comparison_metrics).sort_values(by=['R2'])


    def evaluate_comparison(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)  # for backwards compatibility


    def determine_predictor_identifier(self, param_name, param_value, baseline_value):
        if param_name == 'preprocessing_stages':
            additional_stages = list(set(param_value) - set(baseline_value))
            stage_names = [getattr(stage, '__name__', stage) for stage in additional_stages]
            return f'add_preprocessing:{stage_names}'

        return f'{param_name}_{param_value}'


class AgeClassifierComparison(PredictorComparison):

    def __init__(self, compare_fit_time=False, compare_energy_error=False, *args, **kwargs) -> None:
        self.compare_energy_error = compare_energy_error
        self.compare_fit_time = compare_fit_time
        super().__init__(*args, **kwargs, predictor_type=AgeClassifier)


    def _evaluate_experiment(self, name):
        predictors = self.predictors[name]
        eval_metrics = {}
        eval_metrics['name'] = name
        # average eval metrics across seeds
        eval_metrics['MCC'] = self._mean(predictors, 'mcc')
        eval_metrics['MCC_std'] = self._std(predictors, 'mcc')
        eval_metrics['F1'] = self._mean(predictors, 'f1')
        eval_metrics['F1_std'] = self._std(predictors, 'f1')

        for idx, label in enumerate(predictors[0].labels):
            eval_metrics[f'Recall_{label}'] = self._mean(predictors, 'recall', idx)

        for seed, predictor in enumerate(predictors):
            eval_metrics[f'MCC_seed_{seed}'] = predictor.mcc()

        if self.compare_fit_time:
            eval_metrics['e2e_time'] = (time.time() - self.time_start) / len(predictors)

        if self.compare_energy_error:
            r2, mape = self._mean(predictors, 'energy_error')
            eval_metrics['energy_r2'] = r2
            eval_metrics['energy_mape'] = mape

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
