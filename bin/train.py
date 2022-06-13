#!/usr/bin/env python

import os
import sys
import time
import logging

PROJECT_ROOT = os.path.realpath(os.path.join(__file__, '..', '..'))
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'src')
SUBMODULE = os.path.join(PROJECT_ROOT, 'cluster-utils')

sys.path.append(PROJECT_SRC)
sys.path.append(SUBMODULE)

from xgboost import XGBRegressor

import preprocessing as pp
from prediction_age import AgePredictor
import cluster_utils.dataset as cluster_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

COUNTRY = 'spain'
N_CITIES = 4000
CITIES = []
DATA_DIR = '/p/projects/eubucco/data/2-database-city-level-v0_1'
RESULT_DIR = '/p/tmp/floriann/ml-training'

logger.info('Extracting features...')
df = cluster_dataset.load(country_name=COUNTRY, path=DATA_DIR, cities=CITIES, n_cities=N_CITIES)

logger.info('Training model...')
predictor = AgePredictor(
    model=XGBRegressor(),
    df=df,
    cross_validation_split=pp.city_cross_validation,
    preprocessing_stages=[pp.remove_outliers, pp.remove_other_attributes]
)

logger.info('Saving model...')
timestr = time.strftime('%Y%m%d-%H-%M-%S')
model_path = f'{RESULT_DIR}/model-{COUNTRY}-{N_CITIES or len(CITIES)}-{timestr}.pkl'
predictor.save(model_path)
