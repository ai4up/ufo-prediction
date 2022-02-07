import math
import glob

import dataset

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt

# Comments on binning / categorizing numeric variables
#
# np.histogram which is being used by plt.hist and sns.histplot defines bins to be left closed/inclusive,
# except the last bin which is closed on both sides, e.g. [0,1,2,3] results in [[0,1), [1,2), [2,3]].
#
# To be consistent, left closed/inclusive intervals are used as well when categorizing numeric variables with pd.cut.
# Methods, which determine bin edges/breaks, also assume subsequent left closed/inclusive binning.

def age_bins(y, bin_size=1):
    min_age = math.floor(y[dataset.AGE_ATTRIBUTE].min()) # inclusive
    max_age = math.ceil(y[dataset.AGE_ATTRIBUTE].max()) # inclusive for histogram plotting, exclusive for categorizing variables
    return list(range(min_age, max_age+1))[0::bin_size]


def generate_bins(bin_config):
    min_age = bin_config[0] # inclusive
    bin_max = bin_config[1] # exclusive
    bin_size = bin_config[2]
    return list(range(min_age, bin_max+1))[0::bin_size]


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
    return column.apply(lambda x: int(base * round(float(x)/base)))


def split_target_var(df):
    X = df.drop(columns=[dataset.AGE_ATTRIBUTE])
    y = df[[dataset.AGE_ATTRIBUTE]]
    return X, y


def duplicates(df):
    print(len(df))
    print(len(df['id'].unique()))
    return df[df.duplicated(keep=False)]


def to_gdf(df, crs=2154):
    geo_wkt = df['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_wkt, crs=crs)
