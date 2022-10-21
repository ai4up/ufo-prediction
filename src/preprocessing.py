import logging
import random

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import utils
import dataset
import preparation
import geometry

N_CV_SPLITS = 5

logger = logging.getLogger(__name__)


def use_other_attributes_as_features(df, target=dataset.AGE_ATTRIBUTE):
    other_attrib = dataset.TARGET_ATTRIBUTES.copy()
    other_attrib.remove(target)
    dataset.FEATURES.extend(other_attrib)

    # Remove all buildings that do not have one of our four variables (age/type/floor/height).
    # df = df.dropna(subset=other_attrib)
    # df = remove_buildings_with_unknown_type(df)

    # Encode categorical variable building type
    if target is not dataset.TYPE_ATTRIBUTE:
        # df = utils.dummy_encoding(df, dataset.TYPE_ATTRIBUTE) # one-hot encoding
        df = categorical_to_int(df, var=dataset.TYPE_ATTRIBUTE)  # label encoding

    return df


def use_height_as_feature(df):
    dataset.FEATURES.append(dataset.HEIGHT_ATTRIBUTE)
    return df


def use_type_as_feature(df):
    dataset.FEATURES.append(dataset.TYPE_ATTRIBUTE)
    df = preparation.add_residential_type_column(df)
    df = harmonize_group_source_types(df)
    df[dataset.TYPE_ATTRIBUTE] = np.where((df[dataset.TYPE_ATTRIBUTE] == 'residential') | (df[dataset.TYPE_ATTRIBUTE].isnull()), df['residential_type'], df[dataset.TYPE_ATTRIBUTE])

    df = categorical_to_int(df, var=dataset.TYPE_ATTRIBUTE)  # label encoding

    return df


def shuffle_feature_values(df):
    return df.apply(lambda col: np.random.permutation(col.values))


def split_80_20(df):
    return model_selection.train_test_split(df, test_size=0.2, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_50_50(df):
    return model_selection.train_test_split(df, test_size=0.5, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_20_80(df):
    return model_selection.train_test_split(df, test_size=0.8, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_10_90(df):
    return model_selection.train_test_split(df, test_size=0.9, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)


def split_by_city(df, frac=0.8):
    return split(df, 'city', frac)


def split_by_block(df, frac=0.8):
    return split(df, 'block', frac)


def split_by_neighborhood(df, frac=0.8):
    return split(df, 'neighborhood', frac)


def split(df, attribute, frac):
    values = sorted(df[attribute].unique())
    n = round(frac * len(values))

    if len(values) < 2:
        raise Exception(f'At least two distinct values of "{attribute}" are required to split the dataset. Found only {values}.')

    if n == 0:
        logger.warning(f'Provided fraction ({frac}) is too small. Increasing fraction to {1 / len(values)} to include at least one distinct "{attribute}" value per split.')
        n = 1

    if n == len(values):
        logger.warning(f'Provided fraction ({frac}) is too big. Decreasing fraction to {(len(values) - 1) / len(values)} to include at least one distinct "{attribute}" value per split.')
        n = len(values) - 1

    random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
    sampled_values = random.sample(values, n)

    filter_mask = df[attribute].isin(sampled_values)
    return df[filter_mask], df[~filter_mask]


def cross_validation(df, balanced_attribute=None):
    if balanced_attribute:
        kfold = model_selection.StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
        iterator = kfold.split(df, df[balanced_attribute])
    else:
        kfold = model_selection.KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
        iterator = kfold.split(df)

    for train_idx, test_idx in iterator:
        yield df.iloc[train_idx], df.iloc[test_idx]


def country_cross_validation(df, balanced_attribute=None, spatial_buffer_size=None):
    return _group_cross_validation(df, 'country', balanced_attribute, spatial_buffer_size)


def city_cross_validation(df, balanced_attribute=None, spatial_buffer_size=None):
    return _group_cross_validation(df, 'city', balanced_attribute, spatial_buffer_size)


def sbb_cross_validation(df, balanced_attribute=None, spatial_buffer_size=None):
    if 'sbb' in df.columns:
        logger.info('Reusing street-based block (sbb) column existing in data.')
    else:
        raise Exception(f'Street-based block (sbb) column not found in dataset. Run add_street_block_column() to prepare the dataset.')


    return _group_cross_validation(df, 'sbb', balanced_attribute, spatial_buffer_size)


def block_cross_validation(df, balanced_attribute=None, spatial_buffer_size=None):
    if 'block' in df.columns:
        logger.info('Reusing urban block (based on TouchesIndexes) column existing in data.')
    else:
        df = preparation.add_block_column(df)

    return _group_cross_validation(df, 'block', balanced_attribute, spatial_buffer_size)


def neighborhood_cross_validation(df, balanced_attribute=None, spatial_buffer_size=None):
    if 'neighborhood' in df.columns:
        logger.info('Reusing neighborhood column existing in data.')
    else:
        df = preparation.add_neighborhood_column(df)

    return _group_cross_validation(df, 'neighborhood', balanced_attribute, spatial_buffer_size)


def _group_cross_validation(df, attribute, balanced_attribute=None, spatial_buffer_size=None):
    n_splits = min(df[attribute].nunique(), N_CV_SPLITS)

    if n_splits < N_CV_SPLITS:
        logger.warning(f'Fewer unique {attribute} attributes than cross-validation folds. Reducing the number of folds from {N_CV_SPLITS} to {n_splits}.')

    if balanced_attribute:
        group_kfold = model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
        iterator = group_kfold.split(df, df[balanced_attribute], groups=df[attribute].values)
    else:
        group_kfold = model_selection.GroupKFold(n_splits=n_splits)
        iterator = group_kfold.split(df, groups=df[attribute].values)

    for train_idx, test_idx in iterator:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if spatial_buffer_size:
            buffer_ids = geometry.spatial_buffer_around_block(df, block_type=attribute, buffer_size_meters=spatial_buffer_size, block_ids=test_df[attribute].unique())
            train_df = train_df[train_df['id'].isin(buffer_ids)]

        yield train_df, test_df


def balanced_cross_validation(df):
    return cross_validation(df, balanced=True)


def balanced_city_cross_validation(df):
    return city_cross_validation(df, balanced=True)


def balanced_sbb_cross_validation(df):
    return sbb_cross_validation(df, balanced=True)


def balanced_block_cross_validation(df):
    return block_cross_validation(df, balanced=True)


def balanced_neighborhood_cross_validation(df):
    return neighborhood_cross_validation(df, balanced=True)


def normalize_features(df_train, df_test):
    scaler = preprocessing.MinMaxScaler()
    df_train[dataset.FEATURES] = scaler.fit_transform(df_train[dataset.FEATURES])
    df_test[dataset.FEATURES] = scaler.transform(df_test[dataset.FEATURES])
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
    return df[sorted(filtered_features.union(non_feature_columns))]


def drop_features(df, selection=None, regex=None):
    dropped_features = selection or set(df.filter(regex=regex)).intersection(dataset.FEATURES)
    return df.drop(columns=dropped_features)


def drop_unimportant_features(df):
    return df.drop(columns=set(dataset.FEATURES) - dataset.SELECTED_FEATURES)


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
    return df[df[dataset.TYPE_ATTRIBUTE] == 'residential']


def group_non_residential_buildings(df):
    df[dataset.TYPE_ATTRIBUTE].loc[df[dataset.TYPE_ATTRIBUTE] != 'residential'] = 'non-residential'
    return df


def harmonize_group_source_types(df):
    mask_residential = df['type_source'].isin(['Résidentiel', '1_residential'])
    mask_commercial = df['type_source'].isin(['Commercial et services', '4_3_publicServices', '4_2_retail', '4_1_office'])
    mask_agricultural = df['type_source'].isin(['Agricole', '2_agriculture'])
    mask_industrial = df['type_source'].isin(['Industriel', '3_industrial'])
    mask_others = df['type_source'].isin(['Annexe', 'Sportif', 'Religieux'])
    mask_unknown = (df['type_source'] == 'Indifférencié') | (df[dataset.TYPE_ATTRIBUTE] == 'unknown')

    df[dataset.TYPE_ATTRIBUTE].loc[mask_residential] = 'residential'
    df[dataset.TYPE_ATTRIBUTE].loc[mask_commercial] = 'commercial'
    df[dataset.TYPE_ATTRIBUTE].loc[mask_agricultural] = 'agricultural'
    df[dataset.TYPE_ATTRIBUTE].loc[mask_industrial] = 'industrial'
    df[dataset.TYPE_ATTRIBUTE].loc[mask_others] = 'others'
    df[dataset.TYPE_ATTRIBUTE].loc[mask_unknown] = None

    return df


def remove_buildings_with_unknown_type(df):
    df = df[df['type_source'] != 'Indifférencié']
    df = df[df[dataset.TYPE_ATTRIBUTE] != 'unknown']
    return df


def keep_only_one_building_per_block(df):
    return df.drop_duplicates(subset=['block'], keep='first')


def keep_only_one_building_per_sbb(df):
    return df.drop_duplicates(subset=['sbb'], keep='first')


def imblearn_smote(df_train, df_test, target_var=dataset.AGE_ATTRIBUTE):
    smote = SMOTE()

    smote_cols = list(df_train.columns.intersection(dataset.FEATURES)) + [target_var]
    df_features = df_train[smote_cols]
    df_features_resampled = resample_skewed_distribution(df_features, smote)
    smote_cols.remove('FootprintArea')
    # known issue: small chance that synthetic samples will get aux vars from the original data (depending on which FootprintArea were generated for them. In the original data, FootprintArea is almost unique.)
    df_train_resampled = df_features_resampled.merge(df_train.drop(columns=smote_cols), on='FootprintArea', how='left')
    return df_train_resampled, df_test


def imblearn_adasyn(df_train, df_test, target_var=dataset.AGE_ATTRIBUTE):
    adasyn = ADASYN(random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)

    adasyn_cols = list(df_train.columns.intersection(dataset.FEATURES)) + [target_var]
    df_features = df_train[adasyn_cols]
    df_features_resampled = resample_skewed_distribution(df_features, adasyn)
    adasyn_cols.remove('FootprintArea')
    # known issue: small chance that synthetic samples will get aux vars from the original data (depending on which FootprintArea were generated for them. In the original data, FootprintArea is almost unique.)
    df_train_resampled = df_features_resampled.merge(df_train.drop(columns=adasyn_cols), on='FootprintArea', how='left')
    return df_train_resampled, df_test


def oversample(df_train, df_test):
    ros = RandomOverSampler(random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
    return resample_skewed_distribution(df_train, ros), df_test


def undersample(df_train, df_test):
    rus = RandomUnderSampler(random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
    return resample_skewed_distribution(df_train, rus), df_test


def resample_skewed_distribution(df, sampler):
    logger.info(f"Original dataset types {list(df.select_dtypes(include=['object']).columns)}")

    X, y = utils.split_target_var(df)
    logger.info(f'Original dataset distribution {y.value_counts()}')
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.DataFrame(y_resampled, columns=y.columns)

    for k, v in X.dtypes.items():
        X_resampled[k] = X_resampled[k].astype(v)

    logger.info(f'Resampled dataset distribution {y_resampled.value_counts()}')
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1, join="inner")
    logger.info(f"Resampled dataset object columns {list(df_resampled.select_dtypes(include=['object']).columns)}")
    return df_resampled


def categorical_to_int(df, var, labels=None):
    df[var + '_label'] = df[var]
    df[var] = df[var].astype('category')

    if labels:
        df[var] = df[var].cat.remove_unused_categories()
        df[var] = df[var].cat.reorder_categories(labels)

    df[var] = df[var].cat.codes
    return df


def categorize_age(df, bins, metric_col=None, remove_outliers=True):
    if metric_col:
        df[metric_col] = df[dataset.AGE_ATTRIBUTE]

    df[dataset.AGE_ATTRIBUTE] = pd.cut(df[dataset.AGE_ATTRIBUTE], bins, right=False).cat.codes
    logger.info(
        f'{dataset.AGE_ATTRIBUTE} attribute has been categorized (lowest age included: {bins[0]}; highest age included: {bins[-1]-1}).')

    if not remove_outliers:
        return df

    df_filtered = df[df[dataset.AGE_ATTRIBUTE] >= 0]

    if n_buildings_removed := len(df) - len(df_filtered):
        logger.warning(
            f'During {dataset.AGE_ATTRIBUTE} categorization {n_buildings_removed} buildings outside the age bins have been removed.')

    return df_filtered


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
    np.random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
    df['feature_noise'] = np.random.normal(size=len(df))
    dataset.FEATURES.append('feature_noise')
    return df


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
