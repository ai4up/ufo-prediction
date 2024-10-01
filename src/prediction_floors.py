import logging
from functools import partial

import matplotlib.pyplot as plt

import utils
import visualizations
import dataset
import preprocessing
from prediction import Regressor, Classifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FloorsPredictor(Regressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_attribute=dataset.FLOORS_ATTRIBUTE)


    def _pre_preprocess_analysis_hook(self):
        logger.info(f"Dataset standard deviation: {self.df[self.target_attribute].std()}")
        logger.info(f"Dataset average # of floors: {self.df[self.target_attribute].mean()}")
        logger.info(f'Training dataset length: {len(self.df_train)}')
        logger.info(f'Test dataset length: {len(self.df_test)}')


    def _post_preprocess_analysis_hook(self):
        # The standard deviation in the test set gives us an indication of a
        # baseline. We want to be able to be substantially below that value.
        logger.info(f"Test dataset standard deviation after preprocessing: {self.df_test[self.target_attribute].std()}")
        logger.info(f"Test average # of floors after preprocessing: {self.df_test[self.target_attribute].mean()}")
        logger.info(f'Training dataset length after preprocessing: {len(self.df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(self.df_test)}')


    def evaluate(self):
        self.print_model_error()
        _, axis = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        visualizations.plot_regression_error(self.model, ax=axis[0])
        visualizations.plot_histogram(
            self.y_test, self.y_predict, bins=list(range(8)), ax=axis[1])
        visualizations.plot_relative_grid_floors(self.y_test, self.y_predict)
        plt.show()


    def evaluate_regression(self):
        self.evaluate()  # for backwards compatibility



class FloorsClassifier(Classifier):

    def __init__(self, bins=[], bin_config=None, resampling=None, *args, **kwargs):

        if not bins and bin_config is None or bins and bin_config:
            logger.exception('Please either specify the bins or define a bin config to have them generated automatically.')

        self.bins = bins or utils.generate_bins(bin_config)
        self.resampling = resampling

        super().__init__(*args, **kwargs, target_attribute=dataset.FLOORS_ATTRIBUTE,
                         labels=utils.generate_labels(self.bins), initialize_only=True, validate_labels=False)

        logger.info(f'Generated bins: {self.bins}')
        logger.info(f'Generated bins with the following labels: {self.labels}')

        self.preprocessing_stages.append(partial(preprocessing.categorize_floors, bins=self.bins))
        if self.resampling:
            self.preprocessing_stages.append(self.resampling)

        self._e2e_training()


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
