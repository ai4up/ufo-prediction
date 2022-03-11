import os
from functools import partial

import dataset
from prediction_age import AgePredictor, AgeClassifier, AgePredictorComparison, AgeClassifierComparison
from prediction_type import TypeClassifier
from preprocessing import *

import pytest
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

path_test_data = os.path.join('test', 'e2e', 'test-data.csv')
test_data = pd.read_csv(path_test_data)


@pytest.fixture(autouse=True)
def mock_plotting(mocker):
    mocker.patch("visualizations.plt.show")


@pytest.fixture
def mock_sbb(mocker):
    def mock_add_street_block_feature(df):
        np.random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
        df['sbb'] = np.random.randint(1, 50, df.shape[0])
        return df

    mocker.patch("preprocessing.add_street_block_feature", side_effect=mock_add_street_block_feature)


@pytest.mark.parametrize("kwargs, r2", [
    ({}, 0.045654793721693565),
    ({'mitigate_class_imbalance': True}, 0.16138557969070444),
    ({'early_stopping': False}, 0.03767797982924315),
])
def test_age_regression(kwargs, r2):
    predictor = AgePredictor(
        model=XGBRegressor(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_other_attributes],
        **kwargs,
    )
    assert predictor.r2() == r2


@pytest.mark.parametrize("stage, r2", [
    (remove_outliers, 0.06812998958856764),
    (keep_other_attributes, 0.07966057676899851),
    (remove_non_residential_buildings, 0.1573732054114514),
    (remove_buildings_with_unknown_type, 0.07966057676899851),
    (partial(filter_features, selection=dataset.STREET_FEATURES_CENTRALITY), 0.16069264681349593),
    (partial(normalize_features_citywise, selection=dataset.CITY_FEATURES), 0.045654793721693565), # TODO double check
])
def test_age_regression_preprocessing_stages(stage, r2):
    predictor = AgePredictor(
        model=XGBRegressor(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[stage, remove_other_attributes],
    )
    assert predictor.r2() == r2


@pytest.mark.parametrize("cv, r2", [
    (cross_validation, 0.23502689287489764),
    (city_cross_validation, 0.3036658508998582),
    (sbb_cross_validation, 0.3202048287778162),
    (block_cross_validation, 0.3004533353681359),
])
def test_cross_validation(cv, r2, mock_sbb):
    predictor = AgePredictor(
        model=XGBRegressor(),
        df=test_data,
        cross_validation_split=cv,
        preprocessing_stages=[remove_other_attributes],
    )
    assert predictor.r2() == r2


@pytest.mark.parametrize("kwargs, mcc", [
    ({}, 0.5342540437527113),
    ({'predict_probabilities': True}, 0.4639429246324437),
    ({'mitigate_class_imbalance': True}, 0.5637492018857397),
    ({'early_stopping': False}, 0.50202154565679),
])
def test_age_classification(kwargs, mcc):
    classifier = AgeClassifier(
        model=XGBClassifier(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_other_attributes],
        bins=dataset.EHS_AGE_BINS,
        **kwargs,
    )
    assert classifier.mcc() == mcc


@pytest.mark.parametrize("bin_config, mcc", [
    ((1900, 2000, 50), 0.5113718907544225),
    ((1900, 2020, 20), 0.45082210214224455),
])
def test_age_classification_bin_config(bin_config, mcc):
    classifier = AgeClassifier(
        model=XGBClassifier(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_other_attributes],
        bin_config=bin_config,
    )
    assert classifier.mcc() == mcc


@pytest.mark.parametrize("kwargs, mcc", [
    ({}, 0.7868353326785262),
    ({'predict_probabilities': True}, 0.7649463099740119),
    ({'mitigate_class_imbalance': True}, 0.7298364424042092),
    ({'early_stopping': False}, 0.7211212609650841),
    ({'binary': False}, 0.6082564354063136),
])
def test_type_classification(kwargs, mcc):
    classifier = TypeClassifier(
        model=XGBClassifier(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_non_type_attributes, remove_buildings_with_unknown_type],
        **kwargs,
    )
    assert classifier.mcc() == mcc


def test_regression_comparison():
    grid_comparison_config = {
        '(80/20)': {'test_training_split': split_80_20},
        '(50/50)': {'test_training_split': split_50_50},
    }
    comparison_config = {
        'early stopping': {'early_stopping': False},
        'imbalanced data': {'mitigate_class_imbalance': True},
    }

    comparison_regression = AgePredictorComparison(
        model=XGBRegressor(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_other_attributes],
        comparison_config=comparison_config,
        grid_comparison_config=grid_comparison_config,
    )
    assert list(comparison_regression.evaluate()['R2'].values) == [-0.026387288084199323, 0.03767797982924315, 0.045654793721693565, 0.1263538449657533, 0.16138557969070444]


def test_classification_comparison():
    grid_comparison_config = {
        '(80/20)': {'test_training_split': split_80_20},
        '(50/50)': {'test_training_split': split_50_50},
    }
    comparison_config = {
        'early stopping': {'early_stopping': False},
        'imbalanced data': {'mitigate_class_imbalance': True},
    }

    comparison_classification = AgeClassifierComparison(
        model=XGBClassifier(),
        df=test_data,
        test_training_split=split_80_20,
        preprocessing_stages=[remove_other_attributes],
        bins=dataset.EHS_AGE_BINS,
        comparison_config=comparison_config,
        grid_comparison_config=grid_comparison_config,
    )
    assert list(comparison_classification.evaluate()['MCC'].values) == [0.50202154565679, 0.5342540437527113, 0.5637492018857397, 0.5683885528165616, 0.5832512879458551]
