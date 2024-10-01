import os
import sys
import time
import logging
import json
import shutil
import functools
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import lib_importer
import utils
import dataset
import preparation
import preprocessing as pp
from prediction_age import AgePredictor, AgeClassifier, AgePredictorComparison, AgeClassifierComparison
from prediction_type import TypeClassifierComparison
import cluster_utils.dataset as cluster_dataset

# DATA
RESULT_DIR = '/p/tmp/floriann/ml-exp' # PIK cluster
DATA_DIR = '/p/projects/eubucco/data/3-ml-inputs-v0_1-alpha' # PIK cluster
# DATA_DIR = os.path.join(os.path.abspath(''), 'data', 'exp') # local test
# RESULT_DIR = os.path.join(os.path.abspath(''), 'data', 'exp', 'results') # local test

FRA_DATA_PATH = os.path.join(DATA_DIR, 'df-FRA-exp.pkl')
ESP_DATA_PATH = os.path.join(DATA_DIR, 'df-ESP-exp.pkl')
NLD_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-exp.pkl')
ALL_DATA_PATH = os.path.join(DATA_DIR, 'df-NLD-FRA-ESP-exp.pkl')


# DEFAULT MODEL PARAMS
XGBOOST_PARAMS = {'tree_method': 'hist'}
FRAC = 0.2
PREPROC_STAGES = [pp.remove_buildings_pre_1900]
BIN_CONFIG = (1900, 2020, 5)
MITIGATE_CLASS_IMBALANCE = True
EARLY_STOPPING = True
HYPERPARAMETERS = {
    'max_depth': 13,
    'learning_rate': 0.025,
    'n_estimators': 1000,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.5,
}
CLASSIFICATION_HYPERPARAMETERS = {
    'max_depth': 12,
    'learning_rate': 0.05,
    'n_estimators': 1250,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.3,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

job_id = os.environ.get('SLURM_JOBID', 'local')
timestr = time.strftime('%Y%m%d-%H-%M-%S')

"""
Research Question 1

Comparing a regression and classification model for an energy modelling use-case.
The heating energy demand is estimated for the buildings based on the building age.
The difference in energy estimates is compared for the predicted and the actual building age (MAPE & R2).
The energy estimates for certain building types and construction ranges used in this experiment are taken from TABULA.
"""
def compare_energy_error():
    logger.info('RQ1 - Comparing heating energy estimates derived from predictions...')

    regressor = AgePredictor(
        model=XGBRegressor(**XGBOOST_PARAMS),
        df=FRA_DATA_PATH,
        frac=FRAC,
        cross_validation_split=pp.cross_validation,
        preprocessing_stages=PREPROC_STAGES + [preparation.add_residential_type_column],
        hyperparameters=HYPERPARAMETERS,
        early_stopping=EARLY_STOPPING,
    )
    _save_eval_metrics(regressor, 'reg-energy-error')
    _save_model(regressor, 'reg-energy-error')
    del regressor

    classifier = AgeClassifier(
        model=XGBClassifier(**XGBOOST_PARAMS),
        df=FRA_DATA_PATH,
        frac=FRAC,
        cross_validation_split=pp.cross_validation,
        preprocessing_stages=PREPROC_STAGES + [preparation.add_residential_type_column],
        hyperparameters=HYPERPARAMETERS,
        early_stopping=EARLY_STOPPING,
        mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
        bins=dataset.TABULA_AGE_BINS['france'],
    )

    _save_eval_metrics(classifier, 'class-energy-error')
    _save_model(classifier, 'class-energy-error')
    del classifier


def compare_class_vs_reg():
    logger.info('RQ1 - Comparing regression and classification predictions...')

    regressor = AgePredictor(
        model=XGBRegressor(**XGBOOST_PARAMS),
        df=NLD_DATA_PATH,
        frac=FRAC,
        cross_validation_split=pp.neighborhood_cross_validation,
        preprocessing_stages=PREPROC_STAGES,
        hyperparameters=HYPERPARAMETERS,
        early_stopping=EARLY_STOPPING,
    )
    _save_eval_metrics(regressor, 'reg-nl-neighborhood')
    _save_model(regressor, 'reg-nl-neighborhood')
    del regressor

    classifier = AgeClassifier(
        model=XGBClassifier(**XGBOOST_PARAMS),
        df=NLD_DATA_PATH,
        frac=FRAC,
        cross_validation_split=pp.neighborhood_cross_validation,
        preprocessing_stages=PREPROC_STAGES,
        hyperparameters=HYPERPARAMETERS,
        early_stopping=EARLY_STOPPING,
        mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
        bin_config=BIN_CONFIG,
    )

    _save_eval_metrics(classifier, 'class-nl-neighborhood')
    _save_model(classifier, 'class-nl-neighborhood')
    del classifier


"""
Research Question 2

Inference of building age in cities with partial age coverage.
Assessment of data and prediction quality across countries.
"""
def compare_countries(method):

    logger.info('RQ2 - Evaluating prediction performance differences for FR, ES & NL...')

    exp_name = f'country-comparison-{job_id}-{timestr}'

    comparison_config = {
        # 'Spain': {'df': ESP_DATA_PATH},
        # 'France': {'df': FRA_DATA_PATH},
        'Netherlands': {'df': NLD_DATA_PATH},
        # 'All': {'df': ALL_DATA_PATH},
    }

    grid_comparison_config = {
        # 'random-cv': {'cross_validation_split': pp.cross_validation},
        'neighborhood-cv': {'cross_validation_split': pp.neighborhood_cross_validation},
        # 'city-cv': {'cross_validation_split': pp.city_cross_validation},
        # 'block-cv': {'cross_validation_split': pp.sbb_cross_validation},
    }

    if method == 'regression':

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            compare_feature_importance=False,
            compare_classification_error=True,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    elif method == 'classification':

        logger.info('Comparing classification models...')
        comparison = AgeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=CLASSIFICATION_HYPERPARAMETERS,
            # bin_config=BIN_CONFIG,
            bins=dataset.TABULA_AGE_BINS['harmonized'],
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    _save_comparison_result(comparison, 'country-comparison')
    # _save_model(comparison, 'country-comparison')


"""
Research Question 3

Inference of building age across countries with no local information available.
Assessment of data and prediction quality across countries.
"""
def generalize_across_countries(method):
    n_cv_splits = pp.N_CV_SPLITS
    pp.N_CV_SPLITS = 3

    if method == 'classification':
        import numpy as np
        bins = [1900, 1945, np.inf]
        logger.info('RQ3 - Training classification model for cross-country generalization...')
        predictor = AgeClassifier(
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=ALL_DATA_PATH,
            frac=FRAC,
            cross_validation_split=pp.country_cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=CLASSIFICATION_HYPERPARAMETERS,
            early_stopping=EARLY_STOPPING,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            # bin_config=BIN_CONFIG,
            bins=bins,
        )

    elif method == 'regression':

        logger.info('RQ3 - Training regression model for cross-country generalization...')
        predictor = AgePredictor(
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=ALL_DATA_PATH, # do not reuse same df as it has been grouped
            frac=FRAC,
            cross_validation_split=pp.country_cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            early_stopping=EARLY_STOPPING,
        )


    logger.info('Calculating model error...')

    # predictor.print_model_error()
    # predictor.print_model_error(across_folds=True)

    _save_eval_metrics(predictor, 'generalization')
    _save_predictions(predictor, 'generalization')
    _save_model(predictor, 'generalization')

    pp.N_CV_SPLITS = n_cv_splits


def evaluate_impact_of_spatial_distance_on_generalization(method, country):
    logger.info('RQ3b - Evaluating impact of spatial distance on generalization performance...')

    exp_name = f'spatial-distance-{job_id}-{timestr}'

    df = pd.read_pickle(os.path.join(DATA_DIR, f'df-{country}-exp.pkl'))

    tmp_path = Path(os.path.join(DATA_DIR, 'tmp'))
    tmp_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(DATA_DIR, f'experiment-spatial-distance-cities-{country}.json'), 'r', encoding='utf-8') as f:
        cities = json.load(f)

    comparison_config = {}
    for region, distances in cities.items():
        for dis, (train_cities, test_cities) in enumerate(distances):
            train_path = os.path.join(DATA_DIR, 'tmp', utils.truncated_uuid4() + '.pkl')
            test_path = os.path.join(DATA_DIR, 'tmp', utils.truncated_uuid4() + '.pkl')

            df_train = df[df['city'].isin(train_cities)]
            df_test = df[df['city'].isin(test_cities)]
            df_train.to_pickle(train_path)
            df_test.to_pickle(test_path)

            logger.info(f'Length of train set for {(dis + 1) * 50}km in {region}: {len(df_train)}')
            logger.info(f'Length of test set {(dis + 1) * 50}km in {region}: {len(df_test)}')

            comparison_config[f'{(dis + 1) * 50}-{region}'] = {'df': train_path, 'test_set': test_path}

    if method == 'regression':

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            test_set=None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            compare_feature_importance=False,
            compare_error_distribution=True,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
        )

    _save_comparison_result(comparison, f'spatial-distance-{country}')
    # _save_model(comparison, f'spatial-distance-{country}')
    shutil.rmtree(os.path.join(DATA_DIR, 'tmp'))



"""
Research Question 4

Evaluate the impact of adding additional training data to the prediction problem,
specifically for inner-city inference, cross-city generalization and cross-country generalization.
"""
def evaluate_impact_of_additional_data(method, exploit_spatial_autocorrelation=True, across_cities=False, across_countries=False, include_data_from_other_countries=False):

    logger.info('RQ4 - Evaluating impact of additional data on prediction performance...')

    name = 'additional-data'
    if across_cities:
        name += '-across-cities'
    elif across_countries:
        name += '-across-countries'
    elif exploit_spatial_autocorrelation:
        name += '-inner-cities'
    else:
        name += '-neighborhood-cv'

    exp_name = f'{name}-{job_id}-{timestr}'

    tmp_path = Path(os.path.join(DATA_DIR, 'tmp'))
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(FRA_DATA_PATH)

    if include_data_from_other_countries:
        df_nl = pd.read_pickle(NLD_DATA_PATH)
        df_nl = df_nl[~df_nl['city'].isin(['Maastricht', 'Almelo', 'Ede'])]

    if across_cities:
        df_test_set = df[df['city'].isin(['Aix-en-Provence', 'Rennes', 'Limoges'])]
        df = df[~df['city'].isin(['Aix-en-Provence', 'Rennes', 'Limoges'])]

    generalize = across_cities or across_countries
    comparison_config = {}
    for i in range(13):
        n = 2 ** i
        n_fr_max = len(df['city'].unique())
        n_fr = min(n_fr_max, n)

        if include_data_from_other_countries:
            n_additional_cities = max(0, n - n_fr_max)
            n_nl_max = len(df_nl['city'].unique())
            n_nl = min(n_nl_max, n_additional_cities)

            logger.info(f'Experiment for {n} cities: Adding {n_nl} Dutch cities to {n_fr}/{n_fr_max} French cities...')
            path = os.path.join(DATA_DIR, 'tmp', utils.truncated_uuid4() + '.pkl')
            pd.concat([
                utils.sample_cities(df, n=n_fr),
                utils.sample_cities(df_nl, n=n_nl)]
                ).to_pickle(path)

            comparison_config[f'{n}-cities'] = {'df': path}

            if n_nl == n_nl_max:
                logger.info('All cities from both countries were added. Last iteration is for {n} cities with actually {n_nl_max} Dutch and {n_fr_max} French cities.')
                break

        else:
            comparison_config[f'{n_fr}-cities'] = {'n_cities': n_fr}

            if n_fr == n_fr_max:
                logger.info('All cities from France were added. Last iteration is for {n} cities with actually {n_fr_max} French cities.')
                break

    if across_countries:
        df_test_set = cluster_dataset.load(
            countries=['netherlands'],
            path='/p/projects/eubucco/data/2-database-city-level-v0_1',
            cities=['Maastricht', 'Almelo', 'Ede'],
            dropna_for_col=dataset.AGE_ATTRIBUTE,
            seed=dataset.GLOBAL_REPRODUCIBILITY_SEED
        )

    if exploit_spatial_autocorrelation:
        cv = pp.cross_validation
    else:
        cv = pp.neighborhood_cross_validation

    if method == 'regression':

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=df,
            frac=FRAC,
            cross_validation_split=None if generalize else cv,
            test_set=df_test_set if generalize else None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            compare_feature_importance=False,
            compare_error_distribution=True,
            n_seeds=5, # TODO: increase back to 10
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
        )

    elif method == 'classification':

        logger.info('Comparing classification models...')
        comparison = AgeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=df,
            frac=FRAC,
            cross_validation_split=None if generalize else cv,
            test_set=df_test_set if generalize else None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            compare_feature_importance=False,
            compare_error_distribution=True,
            n_seeds=5, # TODO: increase back to 10
            bin_config=BIN_CONFIG,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
        )


    _save_comparison_result(comparison, name)
    # _save_model(comparison, name)
    shutil.rmtree(os.path.join(DATA_DIR, 'tmp'))


def compare_countries_height_prediction():

    logger.info('SI - Evaluating height prediction performance differences for FR, ES & NL...')

    exp_name = f'height-country-comparison-{job_id}-{timestr}'

    comparison_config = {
        'France': {'df': FRA_DATA_PATH},
        'Spain': {'df': ESP_DATA_PATH},
        'Netherlands': {'df': NLD_DATA_PATH},
        'All': {'df': ALL_DATA_PATH},
    }

    grid_comparison_config = {
        'random-cv': {'cross_validation_split': pp.cross_validation},
        'neighborhood-cv': {'cross_validation_split': pp.neighborhood_cross_validation},
        'city-cv': {'cross_validation_split': pp.city_cross_validation},
        'block-cv': {'cross_validation_split': pp.sbb_cross_validation},
    }

    comparison = AgePredictorComparison(
        exp_name=exp_name,
        model=XGBRegressor(**XGBOOST_PARAMS),
        df=None,
        frac=FRAC,
        cross_validation_split=None,
        preprocessing_stages=[],
        hyperparameters=HYPERPARAMETERS,
        compare_feature_importance=False,
        include_baseline=False,
        garbage_collect_after_training=True,
        comparison_config=comparison_config,
        grid_comparison_config=grid_comparison_config,
    )

    _save_comparison_result(comparison, 'height-country-comparison')
    # _save_model(comparison, 'height-country-comparison')


def compare_countries_type_prediction(binary=True):

    logger.info('SI - Evaluating type prediction performance differences for FR, ES & NL...')

    exp_name = f'type-country-comparison-{job_id}-{timestr}'

    grid_comparison_config = {
        'random-cv': {'cross_validation_split': pp.cross_validation},
        'neighborhood-cv': {'cross_validation_split': pp.neighborhood_cross_validation},
        'city-cv': {'cross_validation_split': pp.city_cross_validation},
        'block-cv': {'cross_validation_split': pp.sbb_cross_validation},
    }
    if binary:
        comparison_config = {
            'Spain': {'df': ESP_DATA_PATH},
            'France': {'df': FRA_DATA_PATH},
        }
        comparison = TypeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=[],
            hyperparameters=CLASSIFICATION_HYPERPARAMETERS,
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )
    else:
        comparison_config = {
            'Spain': {'df': ESP_DATA_PATH, 'labels': ['residential', 'commercial', 'agricultural', 'industrial']},
            'France': {'df': FRA_DATA_PATH, 'labels': ['residential', 'commercial', 'agricultural', 'industrial', 'others']},
        }

        comparison = TypeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=[pp.harmonize_group_source_types],
            hyperparameters=CLASSIFICATION_HYPERPARAMETERS,
            labels=None,
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    _save_comparison_result(comparison, 'type-country-comparison')
    # _save_model(comparison, 'type-country-comparison')


def evaluate_impact_of_additional_features(method):

    logger.info('SI - Evaluating impact of additional features and other preprocessing variations on prediction performance for FR, ES & NL...')

    exp_name = f'additional-features-{job_id}-{timestr}'

    grid_comparison_config = {
        'France': {'df': FRA_DATA_PATH},
        'Spain': {'df': ESP_DATA_PATH},
        'Netherlands': {'df': NLD_DATA_PATH},
    }

    comparison_config = {
        'type': {'preprocessing_stages': [pp.remove_buildings_pre_1900, pp.use_type_as_feature]},
        'height': {'preprocessing_stages': [pp.remove_buildings_pre_1900, pp.use_height_as_feature]},
        'all': {'preprocessing_stages': [pp.remove_buildings_pre_1900, pp.use_type_as_feature, pp.use_height_as_feature]},
        'residential': {'preprocessing_stages': [pp.remove_buildings_pre_1900, pp.remove_non_residential_buildings]},
        'spatially-explicit-features': {'preprocessing_stages': [pp.remove_buildings_pre_1900, functools.partial(pp.filter_features, selection=dataset.SPATIALLY_EXPLICIT_FEATURES)]},
        'building-features': {'preprocessing_stages': [pp.remove_buildings_pre_1900, functools.partial(pp.filter_features, selection=dataset.BUILDING_FEATURES)]},
        'neighborhood-features': {'preprocessing_stages': [pp.remove_buildings_pre_1900, functools.partial(pp.filter_features, selection=dataset.NEIGHBORHOOD_FEATURES)]},
    }

    if method == 'regression':

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=pp.neighborhood_cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=HYPERPARAMETERS,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    elif method == 'classification':

        logger.info('Comparing classification models...')
        comparison = AgeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=pp.neighborhood_cross_validation,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=CLASSIFICATION_HYPERPARAMETERS,
            bin_config=BIN_CONFIG,
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    _save_comparison_result(comparison, 'additional-features')
    # _save_model(comparison, 'country-comparison')


def evaluate_specialized_regional_models(method):

    logger.info('SI - Training specialized models for provinces / departments in FR, ES & NL...')

    exp_name = f'specialized-regional-models-{job_id}-{timestr}'
    tmp_path = Path(os.path.join(DATA_DIR, 'tmp'))
    tmp_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'regions.json'), 'r', encoding='utf-8') as f:
        regions = json.load(f)

    grid_comparison_config = {}
    for country, df_path in {'NLD': NLD_DATA_PATH}.items():
        df = pd.read_pickle(df_path)
        for region, cities in regions[country].items():
            logger.info(f'Creating dataset for region {region} in {country}...')
            path = os.path.join(DATA_DIR, 'tmp', utils.truncated_uuid4() + '.pkl')
            df_region = df[df['city'].isin(cities)]
            if (n := len(df_region)) > 1000:
                df_region.to_pickle(path)
                grid_comparison_config[f'{country}_{region}_{n}'] = {'df': path}

    comparison_config = {
        '80': {'test_training_split': pp.split_80_20},
        '50': {'test_training_split': pp.split_50_50},
        '20': {'test_training_split': pp.split_20_80},
        '10': {'test_training_split': pp.split_10_90},
    }

    hyperparameters = {
        'max_depth': 10,
        'learning_rate': 0.025,
        'n_estimators': 900,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.5,
    }
    if method == 'regression':

        logger.info('Comparing regression models...')
        comparison = AgePredictorComparison(
            exp_name=exp_name,
            model=XGBRegressor(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=hyperparameters,
            n_seeds=5,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    elif method == 'classification':

        logger.info('Comparing classification models...')
        comparison = AgeClassifierComparison(
            exp_name=exp_name,
            model=XGBClassifier(**XGBOOST_PARAMS),
            df=None,
            frac=FRAC,
            cross_validation_split=None,
            preprocessing_stages=PREPROC_STAGES,
            hyperparameters=hyperparameters,
            bin_config=BIN_CONFIG,
            predict_probabilities=False,
            mitigate_class_imbalance=MITIGATE_CLASS_IMBALANCE,
            n_seeds=5,
            compare_feature_importance=False,
            include_baseline=False,
            garbage_collect_after_training=True,
            comparison_config=comparison_config,
            grid_comparison_config=grid_comparison_config,
        )

    _save_comparison_result(comparison, 'specialized-regional-models')
    # _save_model(comparison, 'country-comparison')


def _exit_if_already_in_progress(exp_name):
    path = os.path.join(RESULT_DIR, exp_name + '.marker')
    if os.path.exists(path):
        logger.info(f'Experiment {exp_name} already in progress. Exiting...')
        sys.exit(0)
    else:
        open(path, 'a').close()


def _save_comparison_result(comparison, exp_name):
    logger.info('Saving comparison metrics...')
    path = os.path.join(RESULT_DIR, f'results-{exp_name}-{job_id}-{timestr}.csv')
    results = comparison.evaluate()
    results.to_csv(path, index=False)
    logger.info(results)


def _save_eval_metrics(predictor, exp_name):
    logger.info('Saving evaluation metrics for prediction...')
    path = os.path.join(RESULT_DIR, f'error-{exp_name}-{job_id}-{timestr}.csv')
    predictor.eval_metrics().to_csv(path, index=True)


def _save_predictions(predictor, exp_name):
    logger.info('Saving predictions...')
    path = os.path.join(RESULT_DIR, f'predictions-{exp_name}-{job_id}-{timestr}.csv')
    pred = predictor.y_predict.rename(columns={predictor.target_attribute: 'predict'})
    test = predictor.y_test.rename(columns={predictor.target_attribute: 'test'})
    predictions = pd.concat([pred, test], axis=1)
    predictions.to_pickle(path)


def _save_model(predictor, exp_name):
    logger.info('Saving model(s)...')
    path = os.path.join(RESULT_DIR, f'model-{exp_name}-{job_id}-{timestr}-exp.pkl')
    predictor.save(path, results_only=True)
