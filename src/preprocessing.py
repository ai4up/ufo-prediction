import logging
from datetime import date
from collections import Counter

import utils
import dataset
import visualizations

import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


def remove_non_type_attributes(df):
    return remove_other_attributes(df, target=dataset.TYPE_ATTRIBUTE)


def remove_other_attributes(df, target=dataset.AGE_ATTRIBUTE):
    other_attributes = dataset.TARGET_ATTRIBUTES.copy()
    other_attributes.remove(target)
    return df.drop(columns=other_attributes)


def keep_other_attributes(df):
    # Remove all buildings that do not have one of our four variables (age/type/floor/height).
    df = df.dropna(subset=dataset.TARGET_ATTRIBUTES)
    df = remove_buildings_with_unknown_type(df)

    # Encode categorical variable building type
    # df = utils.dummy_encoding(df, dataset.TYPE_ATTRIBUTE) # one-hot encoding
    df = categorical_to_int(df, dataset.TYPE_ATTRIBUTE) # label encoding
    return df


def normalize_features(df_train, df_test):
    scaler = preprocessing.MinMaxScaler()
    feature_columns = list(set(df_train.columns) - set(dataset.AUX_VARS) - set(dataset.TARGET_ATTRIBUTES))
    df_train[feature_columns] = scaler.fit_transform(df_train[feature_columns])
    df_test[feature_columns] = scaler.transform(df_test[feature_columns])
    return df_train, df_test


# TODO: fit only on training data to avoid information leakage into test set
def normalize_centrality_features_citywise(df):
    centrality_features = df.filter(regex='_buffer').columns
    df[centrality_features] = df.groupby('city')[centrality_features].apply(normalize_columns)
    return df


def normalize_features_citywise(df, selection=None, regex=None):
    features = selection or df.filter(regex=regex).columns
    df[features] = df.groupby('city')[features].apply(normalize_columns)
    return df


def normalize_columns(df, columns=None):
    columns = columns or df.columns
    scaler = preprocessing.MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def filter_features(df, selection=[], regex=None):
    non_feature_columns = set(df.columns) - set(dataset.FEATURES)
    filtered_features = set(selection) or set(df.filter(regex=regex)).intersection(dataset.FEATURES)
    return df[filtered_features.union(non_feature_columns)]


def drop_features(df, selection=None, regex=None):
    dropped_features = selection or set(df.filter(regex=regex)).intersection(dataset.FEATURES)
    return df[df.columns.drop(dropped_features)]


def drop_unimportant_features(df):
    return df[dataset.SELECTED_FEATURES + [dataset.AGE_ATTRIBUTE] + dataset.AUX_VARS]


def remove_buildings_pre_1850(df):
    return df[df[dataset.AGE_ATTRIBUTE] >= 1850]


def remove_buildings_pre_1900(df):
    return df[df[dataset.AGE_ATTRIBUTE] >= 1900]


def remove_buildings_pre_1950(df):
    return df[df[dataset.AGE_ATTRIBUTE] >= 1950]


def remove_buildings_pre_2000(df):
    return df[df[dataset.AGE_ATTRIBUTE] >= 2000]


def remove_buildings_post_2009(df):
    return df[df[dataset.AGE_ATTRIBUTE] < 2010]


def remove_buildings_post_1980(df):
    return df[df[dataset.AGE_ATTRIBUTE] <= 1980]


def remove_buildings_between_1930_1990(df):
    return df[~df[dataset.AGE_ATTRIBUTE].between(1930, 1990)]


def remove_outliers(df):
    df = df[df[dataset.AGE_ATTRIBUTE] > 1900]
    df = df[df[dataset.AGE_ATTRIBUTE] < 2020]
    return df


def remove_non_residential_buildings(df):
    return df[df[dataset.TYPE_ATTRIBUTE] == 'Résidentiel']


def group_non_residential_buildings(df):
    df[dataset.TYPE_ATTRIBUTE].loc[df[dataset.TYPE_ATTRIBUTE] != 'Résidentiel'] = 'non-residential'
    return df


def remove_buildings_with_unknown_type(df):
    df = df[df[dataset.TYPE_ATTRIBUTE] != 'Indifférencié']
    return df


def undersample_skewed_distribution(df):
    rus = RandomUnderSampler(random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
    X, y = utils.split_target_var(df)
    undersampled_X, undersampled_y = rus.fit_resample(X, y)

    visualizations.plot_histogram(undersampled_y, y, bins=utils.age_bins(undersampled_y))
    logger.info(f'Downsampling distribution results in: {sorted(Counter(undersampled_y[dataset.AGE_ATTRIBUTE]).items())}')

    undersampled_df = pd.concat([undersampled_X, undersampled_y], axis=1, join="inner")
    return undersampled_df


def categorical_to_int(df, var):
    df[var] = df[var].astype('category').cat.codes
    return df


def categorize_age(df, bins):
    df[dataset.AGE_ATTRIBUTE] = pd.cut(df[dataset.AGE_ATTRIBUTE], bins, right=False).cat.codes
    df = df[df[dataset.AGE_ATTRIBUTE] >= 0]
    logger.info(f'{dataset.AGE_ATTRIBUTE} attribute has been categorized (lowest age included: {bins[0]}; highest age included: {bins[-1]-1}; other buildings have been removed).')
    return df


def categorize_age_custom_bands(df):
    return categorize_age(df, dataset.CUSTOM_AGE_BINS)


def categorize_age_EHS(df):
    return categorize_age(df, dataset.EHS_AGE_BINS)


def categorize_age_5y_bins(df):
    bins = utils.age_bins(df, bin_size=5)
    return categorize_age(df, bins)


def categorize_age_10y_bins(df):
    bins = utils.age_bins(df, bin_size=10)
    return categorize_age(df, bins)


def round_age(df):
    df[dataset.AGE_ATTRIBUTE] = utils.custom_round(df[dataset.AGE_ATTRIBUTE])
    return df


def add_noise_feature(df):
    df["feature_noise"] = np.random.normal(size=len(df))
    return df


def split_80_20(df):
    return model_selection.train_test_split(df, test_size=0.2, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_50_50(df):
    return model_selection.train_test_split(df, test_size=0.5, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_by_region(df):
    # We aim to cross-validate our results using five French sub-regions 'departement' listed below.
    # one geographic region for validation, rest for testing
    region_names = ['Haute-Vienne', 'Hauts-de-Seine',
                    'Aisne', 'Orne', 'Pyrénées-Orientales']
    df_test = df[df.departement == region_names[dataset.GLOBAL_REPRODUCIBILITY_SEED % len(region_names)]]
    df_train = df[~df.index.isin(df_test.index)]
    return df_train, df_test


def filter_french_medium_sized_cities_with_old_center(df):
    city_names = ['Valence', 'Aurillac', 'Oyonnax', 'Aubenas', 'Vichy', 'Montluçon', 'Montélimar', 'Bourg-en-Bresse']
    # city_names = ['Valence', 'Oyonnax', 'Bourg-en-Bresse'] # very similar in terms of building age structure
    return df[df['city'].isin(city_names)]


def split_and_filter_by_french_medium_sized_cities_with_old_center(df):
    city_names = ['Valence', 'Aurillac', 'Oyonnax', 'Aubenas', 'Vichy', 'Montluçon', 'Montélimar', 'Bourg-en-Bresse']
    test_city = city_names[dataset.GLOBAL_REPRODUCIBILITY_SEED % len(city_names)]
    city_names.remove(test_city)
    df_test = df[df['city'] == test_city]
    df_train = df[df['city'].isin(city_names)]
    return df_train, df_test


def split_by_city(df):
    cities = sorted(df['city'].unique())
    df_test = df[df['city'] == cities[dataset.GLOBAL_REPRODUCIBILITY_SEED % len(cities)]]
    df_train = df[~df.index.isin(df_test.index)]
    return df_train, df_test