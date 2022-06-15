#!/usr/bin/env python

import os
import sys
import time
import pickle
import logging
import datetime

PROJECT_ROOT = os.path.realpath(os.path.join(__file__, '..', '..'))
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'src')
SUBMODULE = os.path.join(PROJECT_ROOT, 'cluster-utils')

sys.path.append(PROJECT_SRC)
sys.path.append(SUBMODULE)

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import dataset
import preprocessing as pp
from prediction_age import AgeClassifier, AgePredictor
import cluster_utils.dataset as cluster_dataset
import cluster_utils.slack_notifications as slack

MODEL_TYPE = 'classification'
COUNTRIES = ['netherlands', 'france', 'spain']
CITIES = []
N_CITIES = None
DATA_DIR = '/p/projects/eubucco/data/2-database-city-level-v0_1'
RESULT_DIR = '/p/tmp/floriann/ml-training'

# with open(os.path.join(PROJECT_ROOT, 'metadata', 'feature-selection-cities.pkl'), 'rb') as f:
#     CITIES = pickle.load(f)

start_time = time.time()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
slack_channel = os.environ.get('SLACK_CHANNEL')
slack_token = os.environ.get('SLACK_TOKEN')


logger.info('Extracting features...')
# df = cluster_dataset.load(
#     countries=COUNTRIES,
#     path=DATA_DIR,
#     cities=CITIES,
#     n_cities=N_CITIES,
#     dropna_for_col=dataset.AGE_ATTRIBUTE,
#     seed=dataset.GLOBAL_REPRODUCIBILITY_SEED
#     )

# df.to_pickle(os.path.join(RESULT_DIR, 'df-nl-fr-es.pkl'))
df = pd.read_pickle(os.path.join(RESULT_DIR, 'df-feature-selection.pkl'))

import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
logger.info(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')


if MODEL_TYPE == 'classification':
    logger.info('Training classification model...')

    tabula_nl_bins = [1900, 1965, 1975, 1992, 2006, 2015, 2022]
    equally_sized_bins = (1900, 2020, 5)

    predictor = AgeClassifier(
        model=XGBClassifier(tree_method='gpu_hist'),
        df=df,
        test_training_split=pp.split_80_20,
        # cross_validation_split=pp.city_cross_validation,
        preprocessing_stages=[pp.remove_outliers, pp.remove_other_attributes, pp.add_noise_feature],
        mitigate_class_imbalance=True,
        bin_config=equally_sized_bins
    )

else:
    logger.info('Training regression model...')
    predictor = AgePredictor(
        model=XGBRegressor(tree_method='gpu_hist'),
        df=df,
        test_training_split=pp.split_80_20,
        # cross_validation_split=pp.city_cross_validation,
        preprocessing_stages=[pp.remove_outliers, pp.remove_other_attributes, pp.add_noise_feature]
    )


logger.info('Calculating model error...')
predictor.print_model_error()


logger.info('Calculating feature importance...')
fts_importance = predictor.normalized_feature_importance()
timestr = time.strftime('%Y%m%d-%H-%M-%S')
file_path = f'{RESULT_DIR}/feature-importance-{MODEL_TYPE}-{len(COUNTRIES)}-{N_CITIES or len(CITIES)}-{timestr}.csv'
fts_importance.to_csv(file_path, index=False)


logger.info('Saving model...')
model_path = f'{RESULT_DIR}/model-{MODEL_TYPE}-{len(COUNTRIES)}-{N_CITIES or len(CITIES)}-{timestr}.pkl'
predictor.save(model_path, results_only=True)


# logger.info('Sending slack notification...')
# try:
#     duration = str(datetime.timedelta(seconds=time.time() - start_time)).split('.')[0]
#     slack.send_message(f'Model training for {' ,'.join(COUNTRIES)} finished after {duration}. 🚀', slack_channel, slack_token)
# except Exception as e:
#     logger.error(f'Failed to send Slack message: {e}')
