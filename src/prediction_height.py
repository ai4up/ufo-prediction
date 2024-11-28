import logging

import matplotlib.pyplot as plt

import utils
import visualizations
import dataset
from prediction import Regressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HeightPredictor(Regressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_attribute=dataset.HEIGHT_ATTRIBUTE)


    def _pre_preprocess_analysis_hook(self):
        logger.info(f"Dataset standard deviation: {self.df[self.target_attribute].std()}")
        logger.info(f"Dataset average height: {self.df[self.target_attribute].mean()}")
        logger.info(f'Training dataset length: {len(self.df_train)}')
        logger.info(f'Test dataset length: {len(self.df_test)}')


    def _post_preprocess_analysis_hook(self):
        # The standard deviation in the test set gives us an indication of a
        # baseline. We want to be able to be substantially below that value.
        logger.info(f"Test dataset standard deviation after preprocessing: {self.df_test[self.target_attribute].std()}")
        logger.info(f"Test average height after preprocessing: {self.df_test[self.target_attribute].mean()}")
        logger.info(f'Training dataset length after preprocessing: {len(self.df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(self.df_test)}')


    def evaluate(self):
        bins = utils.generate_bins((0, 50, 2))
        self.print_model_error()
        _, axis = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        visualizations.plot_regression_error(self.model, ax=axis[0])
        visualizations.plot_histogram(
            self.y_test, self.y_predict, bins=bins, ax=axis[1])
        visualizations.plot_relative_grid(self.y_test, self.y_predict, dataset.HEIGHT_ATTRIBUTE, bins)
        plt.show()


    def evaluate_regression(self):
        self.evaluate()  # for backwards compatibility
