import os
import time
import logging

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
import pandas as pd

import lib_importer
from run_experiments import *
import run_experiments
import geometry
import spatial_autocorrelation
import preprocessing as pp
from prediction_age import AgePredictorComparison, AgeClassifierComparison

# PIK cluster
RESULT_DIR = '/p/tmp/floriann/ml-exp'
DATA_DIR = '/p/projects/eubucco/data/3-ml-inputs-v0_1-alpha'
FRA_DATA_PATH = os.path.join(DATA_DIR, 'df-FRA-preliminary.pkl')
ESP_DATA_PATH = os.path.join(DATA_DIR, 'df-ESP-preliminary.pkl')
NLD_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-preliminary.pkl')
ALL_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-FRA-ESP-preliminary.pkl')

# local test
# DATA_DIR = os.path.join(os.path.abspath(''), 'data', 'exp')
# RESULT_DIR = os.path.join(os.path.abspath(''), 'data', 'exp', 'results')
# FRA_DATA_PATH = os.path.join(DATA_DIR, 'df-FRA-exp.pkl')
# ESP_DATA_PATH = os.path.join(DATA_DIR, 'df-ESP-exp.pkl')
# NLD_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-exp.pkl')
# ALL_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-FRA-ESP-exp.pkl')

XGBOOST_PARAMS = {'tree_method': 'gpu_hist'}
RF_PARAMS = {'n_jobs': -1}
ADA_PARAMS = {'n_jobs': -1}

HYPERPARAMETERS = {}
CLASSIFICATION_HYPERPARAMETERS = {}

ada_param_search_space = {
    'learning_rate': [0.01, 0.025, 0.05, 0.1],
    'n_estimators': [250, 500, 1000, 2500, 5000]
}

xgboost_param_search_space = {
    'max_depth': [10, 12, 14],
    'learning_rate': [0.01, 0.025, 0.05, 0.1],
    'n_estimators': [750, 1000, 1250],
    'colsample_bytree': [0.5, 0.9],
    'colsample_bylevel': [0.5, 0.9],
}

rf_param_search_space = {
    'max_depth': [None],
    'max_features': [0.3, 'sqrt', None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5],
    'n_estimators': [250, 500, 1000]
}

HYPERPARAMETER_N_ITER = 10
FRAC = 0.1
PREPROC_STAGES = [pp.remove_buildings_pre_1900]
BIN_CONFIG = (1900, 2020, 10)
MITIGATE_CLASS_IMBALANCE = True
EARLY_STOPPING = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

job_id = os.environ.get('SLURM_JOBID', 'local')
timestr = time.strftime('%Y%m%d-%H-%M-%S')


def compare_resampling_strategies():
    exp_name = f'resampling-{job_id}-{timestr}'

    grid_comparison_config = {
        'Netherlands': {'df': NLD_DATA_PATH},
        'France': {'df': FRA_DATA_PATH},
        'Spain': {'df': ESP_DATA_PATH},
        'All': {'df': ALL_DATA_PATH},
    }

    comparison_config = {
        'oversampling': {'resampling': pp.oversample},
        'undersampling': {'resampling': pp.undersample},
        'sample weights': {'mitigate_class_imbalance': False},
        'ADASYN': {'resampling': pp.imblearn_adasyn},
        'SMOTE': {'resampling': pp.imblearn_smote},
    }

    logger.info('Comparing resampling strategies...')
    comparison = AgeClassifierComparison(
        exp_name=exp_name,
        model= XGBClassifier(**XGBOOST_PARAMS),
        df=None,
        frac=FRAC,
        cross_validation_split=pp.cross_validation,
        preprocessing_stages=PREPROC_STAGES,
        bin_config=BIN_CONFIG,
        predict_probabilities=False,
        mitigate_class_imbalance=False,
        compare_fit_time=True,
        compare_feature_importance=False,
        include_baseline=False,
        garbage_collect_after_training=True,
        comparison_config=comparison_config,
        grid_comparison_config=grid_comparison_config,
    )

    run_experiments._save_comparison_result(comparison, 'resampling')


def model_selection(method):

    logger.info('PRE EXP - Comparing prediction performance of different models')

    exp_name = f'model-section-{job_id}-{timestr}'

    comparison_config = {
        'France': {'df': FRA_DATA_PATH},
        'Netherlands': {'df': NLD_DATA_PATH},
        'Spain': {'df': ESP_DATA_PATH},
        'All': {'df': ALL_DATA_PATH},
    }

    if method == 'regression':

        grid_comparison_config = {
            'RandomForest': {'model': RandomForestRegressor(**RF_PARAMS), 'hyperparameter_tuning_space': rf_param_search_space},
            'AdaBoost': {'model': AdaBoostRegressor(**ADA_PARAMS), 'hyperparameter_tuning_space': ada_param_search_space},
            'XGboost': {'model': XGBRegressor(**XGBOOST_PARAMS), 'hyperparameter_tuning_space': xgboost_param_search_space},
        }

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=None,
            df=None,
            frac=FRAC,
            # test_training_split=pp.split_80_20,
            cross_validation_split=pp.cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameter_tuning_space=None,
            hyperparameter_n_iter=HYPERPARAMETER_N_ITER,
            compare_fit_time=True,
            compare_feature_importance=False,
            compare_classification_error=True,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    elif method == 'classification':

        grid_comparison_config = {
            'RandomForest': {'model': RandomForestClassifier(**RF_PARAMS), 'hyperparameter_tuning_space': rf_param_search_space},
            'AdaBoost': {'model': AdaBoostClassifier(**ADA_PARAMS), 'hyperparameter_tuning_space': ada_param_search_space},
            'XGboost': {'model': XGBClassifier(**XGBOOST_PARAMS), 'hyperparameter_tuning_space': xgboost_param_search_space},
        }

        logger.info('Comparing classification models...')
        comparison = AgeClassifierComparison(
            exp_name=exp_name,
            model=None,
            df=None,
            frac=FRAC,
            cross_validation_split=pp.cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameter_tuning_space=None,
            hyperparameter_n_iter=HYPERPARAMETER_N_ITER,
            bin_config=BIN_CONFIG,
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            compare_fit_time=True,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    run_experiments._save_comparison_result(comparison, 'model-selection')


def analyze_spatial_autocorrelation():
    # for country, df_path in {'NLD': NLD_DATA_PATH, 'FRA': FRA_DATA_PATH, 'ESP': ESP_DATA_PATH}.items():
    for country, df_path in {'FRA': FRA_DATA_PATH, 'ESP': ESP_DATA_PATH}.items():
        path = os.path.join(RESULT_DIR, f'spatial-autocorrelation-band-{country}-{job_id}-{timestr}.csv')
        df = pd.read_pickle(df_path)
        
        cities = df['city'].value_counts().between(5000, 20000)
        cities = list(cities[cities].index)[:10]
        df = df[df['city'].isin(cities)]

        # attributes = ['age', 'height', 'FootprintArea', 'Perimeter']
        attributes = ['age']
        aux_attributes = ['id', 'geometry', 'neighborhood', 'sbb', 'city', 'country']

        aux_vars_geo = geometry.to_gdf(df[aux_attributes + attributes])
        aux_vars_geo.dropna(subset=['age', 'id'], inplace=True)

        results = aux_vars_geo.groupby('city').apply(lambda x: spatial_autocorrelation.plot_correlogram_over_distance(x, attributes=attributes, distances=[10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]))
        results.to_csv(path, index=True)
