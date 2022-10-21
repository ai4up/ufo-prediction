import os
import time
import logging
import json

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import lib_importer
import preprocessing as pp
from prediction_age import AgeClassifier, AgePredictor


# PIK cluster
DATA_DIR = '/p/projects/eubucco/data/3-ml-inputs'
RESULT_DIR = '/p/tmp/floriann/ml-training'

# local test
# DATA_DIR = os.path.join(os.path.abspath(''), '..', 'data', 'exp')
# RESULT_DIR = os.path.join(os.path.abspath(''), '..', 'data', 'exp', 'results')


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# regression
reg_params = {
    'max_depth': range(5, 15, 1),
    'learning_rate': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    'n_estimators': range(500, 1500, 250),
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
    'colsample_bylevel': [0.3, 0.5, 0.7, 0.9],
    'max_bin': [128, 256, 512]
}
# classification
class_params = {
    'max_depth': range(10, 15, 1),
    'learning_rate': [0.025, 0.05, 0.1],
    'n_estimators': range(750, 1750, 250),
    'colsample_bytree': [0.5, 0.7, 0.9],
    'colsample_bylevel': [0.3, 0.5, 0.7, 0.9],
}

# model complexity assessment
max_depth_params = {
    'max_depth': range(1, 20, 1),
    'learning_rate': [0.1],
    'n_estimators': [1000],
}


def tune(method):
    logger.info('Extracting features...')
    df = pd.read_pickle(os.path.join(DATA_DIR, 'df-NLD-preliminary.pkl'))

    if method == 'classification':
        logger.info('Training classification model...')

        tabula_nl_bins = [1900, 1965, 1975, 1992, 2006, 2015, 2022]
        equally_sized_bins = (1900, 2020, 10)

        predictor = AgeClassifier(
            model=XGBClassifier(tree_method='gpu_hist'),
            df=df,
            cross_validation_split=pp.neighborhood_cross_validation,
            preprocessing_stages=[pp.remove_buildings_pre_1900],
            hyperparameter_tuning_only=True,
            hyperparameter_tuning_space=class_params,
            early_stopping=True,
            mitigate_class_imbalance=True,
            bin_config=equally_sized_bins,
        )

    else:
        logger.info('Training regression model...')
        print('Training regression model...')
        predictor = AgePredictor(
            model=XGBRegressor(tree_method='gpu_hist'),
            df=df,
            cross_validation_split=pp.neighborhood_cross_validation,
            preprocessing_stages=[pp.remove_buildings_pre_1900],
            hyperparameter_tuning_only=True,
            hyperparameter_tuning_space=reg_params,
            early_stopping=True,
        )

    logger.info('Saving hyperparameter optimized model...')
    job_id = os.environ.get('SLURM_JOBID')
    timestr = time.strftime('%Y%m%d-%H-%M-%S')
    path = f'{RESULT_DIR}/model-optimized-{method}-{job_id}-{timestr}.pkl'
    predictor.save(path, results_only=True)


def tune_for_regional_model(method):
    df = pd.read_pickle(os.path.join(DATA_DIR, 'df-NLD-preliminary.pkl'))

    with open(os.path.join(DATA_DIR, 'regions.json'), 'r', encoding='utf-8') as f:
        regions = json.load(f)

    for i, (region, cities) in enumerate(regions['NLD'].items()):
        if i > 4:
            return

        if method == 'regression':

            df_exp = df[df['city'].isin(cities)]
            logger.info(f'Training regression model for {region}...')
            predictor = AgePredictor(
                model=XGBRegressor(tree_method='gpu_hist'),
                df=df_exp,
                cross_validation_split=pp.cross_validation,
                preprocessing_stages=[pp.remove_buildings_pre_1900],
                hyperparameter_tuning_only=True,
                hyperparameter_tuning_space=reg_params,
                early_stopping=True,
            )

            logger.info(f'Saving hyperparameter optimized model for {region}...')
            job_id = os.environ.get('SLURM_JOBID')
            timestr = time.strftime('%Y%m%d-%H-%M-%S')
            path = f'{RESULT_DIR}/model-optimized-{method}-{job_id}-{timestr}.pkl'
            predictor.save(path, results_only=True)
