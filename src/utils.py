import os
import ast
import math
import random
import logging
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset
import geometry

logger = logging.getLogger(__name__)

"""
Comments on binning / categorizing numeric variables

np.histogram which is being used by plt.hist and sns.histplot defines bins to be left closed/inclusive,
except the last bin which is closed on both sides, e.g. [0,1,2,3] results in [[0,1), [1,2), [2,3]].

To be consistent, left closed/inclusive intervals are used as well when categorizing numeric variables with pd.cut.
Methods, which determine bin edges/breaks, also assume subsequent left closed/inclusive binning.
"""


def age_bins(y, bin_size=1):
    # lower bound inclusive
    min_age = math.floor(y[dataset.AGE_ATTRIBUTE].min())
    # upper bound inclusive for histogram plotting, exclusive for categorizing variables
    max_age = math.ceil(y[dataset.AGE_ATTRIBUTE].max())
    return list(range(min_age, max_age + 1))[0::bin_size]


def generate_bins(bin_config):
    min_age = bin_config[0]  # inclusive
    bin_max = bin_config[1]  # exclusive
    bin_size = bin_config[2]
    return list(range(min_age, bin_max + 1))[0::bin_size]


def generate_labels(bins):
    # assuming integer data
    labels = [f'{i}-{j-1}' for i, j in zip(bins[:-1], bins[1:])]

    if np.isinf(bins[-1]):
        labels[-1] = f'>={bins[-2]}'

    if bins[0] == 0:
        labels[0] = f'<{bins[1]}'

    return labels


def dummy_encoding(df, var):
    one_hot_encoding = pd.get_dummies(df[var], prefix=var)
    df = df.drop(columns=[var])
    df = df.merge(
        one_hot_encoding, left_index=True, right_index=True)
    return df


def custom_round(column, base=5):
    return column.apply(lambda x: int(base * round(float(x) / base)))


def split_target_var(df):
    X = df.drop(columns=[dataset.AGE_ATTRIBUTE])
    y = df[[dataset.AGE_ATTRIBUTE]]
    return X, y


def duplicates(df):
    print(len(df))
    print(len(df['id'].unique()))
    return df[df.duplicated(keep=False)]


def grid_subplot(n_plots, n_cols=4):
    ncols = n_cols if n_plots > n_cols else n_plots
    nrows = math.ceil(n_plots / n_cols)
    _, axis = plt.subplots(nrows, ncols, figsize=(30, 20), constrained_layout=True)
    return [axis[idx // n_cols, idx % n_cols] if n_plots > n_cols else axis[idx % n_cols] for idx in range(0, n_plots)]


def flatten(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def stratified_sampling(df, group, frac=None, n=None):
    return df.groupby(group, group_keys=False).apply(lambda x: x.sample(frac=frac, n=min(len(x), n)))


def stratified_city_sampling(df, group, frac):
    return df.groupby(group, group_keys=False).apply(lambda x: sample_cities(x, frac))


def sample_cities_with_distributional_constraint(df, frac, attr, min_samples_per_group=100):
    if 1 / frac > min_samples_per_group:
        logger.warning('A bias may be introduced because the sampling fraction is too small for the min_samples_per_group parameter. Consider increasing the min_samples_per_group or the sampling fraction.')

    n_groups = int(len(df[attr]) / min_samples_per_group)  # every quantile group should have at least min_samples_per_group samples

    if n_groups < 2:
        raise Exception(f'The dataframe only has {len(df)} samples. Please specify a lower number of min_samples_per_group than {min_samples_per_group} to yield at least two quantiles to sample from.')

    df['group'] = pd.qcut(df[attr].rank(method='first'), n_groups, labels=False)
    return stratified_city_sampling(df, 'group', frac)


def sample_cities(df, frac=None, n=None):
    if n == 0:
        return df.drop(df.index)

    cities = sorted(df['city'].unique())
    n = n or round(frac * len(cities))

    if n > len(cities):
        logger.warning(f'Sample n={n} is larger than number of cities. Using all {len(cities)} cities instead.')
        n = len(cities)

    if n == 0:
        logger.warning(f'Provided fraction ({frac}) is too small. Increasing fraction to {1 / len(cities)} to include at least one city in the sample.')
        n = 1

    random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
    sampled_cities = random.sample(cities, n)

    return df[df['city'].isin(sampled_cities)]


def sample_cities_until_n_buildings(df, min_n_buildings):
    cities = sorted(df['city'].unique())

    for n in range(len(cities)):
        sampled_df = sample_cities(df, n=n+1)

        if len(sampled_df) > min_n_buildings:
            return sampled_df

    logger.warning(f'Could not sample {min_n_buildings} or more buildings from the {len(cities)} cities. Not sufficient buildings in the dataset. Returning full dataset.')
    return df


def exclude_neighbors_from_own_block(neighbors, df, block_type):
    id_to_block = df[['id', block_type]].set_index('id').to_dict()[block_type]

    neighbors_exc_block = {}
    for building_id, neighbor_ids in neighbors.items():
        own_block = id_to_block[building_id]
        neighbors_exc_block[building_id] = [id for id in neighbor_ids if id_to_block[id] != own_block]

    return neighbors_exc_block


def verbose():
    return logging.root.level <= logging.DEBUG


def truncated_uuid4():
    return str(uuid.uuid4())[:8]


def seq_to_unique_id(series):
    seq_to_unique_mapping = {seq_id: truncated_uuid4() for seq_id in series.unique()}
    return series.map(seq_to_unique_mapping)


def load_data(country, geo=False, eval_columns=[id], crs=3035, **kwargs):
    def parse_int_list(l):
        return [int(i) for i in ast.literal_eval(l)]

    country_files = {
        'france': 'df-FRA.pkl',
        'spain': 'df-ESP.pkl',
        'netherlands': 'df-NLD.pkl',
    }

    converters = {
        'id': str,
        'block_bld_ids': ast.literal_eval,
        'sbb_bld_ids': ast.literal_eval,
        'TouchesIndexes': parse_int_list,
        }

    logger.debug('Loading pickle...')
    path = os.path.realpath(os.path.join(dataset.DATA_DIR, country_files[country]))
    df = pd.read_pickle(path, **kwargs)

    logger.debug('Parsing data...')
    for col, func in converters.items():
        if col in eval_columns:
            df[col] = df[col].apply(func)
            logger.debug(f'Finished converting {col}.')

    if geo:
        logger.debug('Adding geometry column...')
        df = geometry.add_geometry_column(df, crs=crs, countries=[country])

    return df


def load_df(df_path):
    if '.csv' in df_path:
        return pd.read_csv(df_path)

    if '.pkl' in df_path:
        return pd.read_pickle(df_path)

    if '.parquet' in df_path:
        return pd.read_parquet(df_path)

    raise Exception('File type not supported, please use .csv, .pkl, .parquet files.')
