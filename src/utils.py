import os
import math
import glob

import dataset

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn import model_selection
import matplotlib.pyplot as plt

DATA_DIR = os.path.join('..', 'data')
DATA_GEO_DIR = os.path.join(DATA_DIR, 'geographics')

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


def tune_hyperparameter(model, X, y):
    params = {
        'max_depth': [1, 3, 6, 10],  # try ada trees
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'subsample': [0.7, 1.0],
    }

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    clf = model_selection.GridSearchCV(estimator=model,
                                       param_grid=params,
                                       scoring='neg_root_mean_squared_error',
                                       verbose=1)
    clf.fit(X, y)
    print("Best parameters: ", clf.best_params_)
    print("Lowest RMSE: ", np.sqrt(-clf.best_score_))

    tuning_results = pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
        clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    tuning_results.to_csv('hyperparameter-tuning-results.csv', sep='\t')
    print('All hyperparameter tuning results:\n', tuning_results)

    return clf.best_params_


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


def add_geometry_column(df):
    files_geom = glob.glob(os.path.join(DATA_GEO_DIR, '*', '*_geom.csv')) + glob.glob(os.path.join(DATA_GEO_DIR, '*_geom.csv'))
    data_geom = pd.concat((pd.read_csv(f) for f in files_geom), ignore_index=True)
    data_geom.drop_duplicates(subset=['id', 'geometry'], inplace=True)

    data_geom['id'] = data_geom['id'].astype(str)
    df['id'] = df['id'].astype(str)
    df_w_geometry = df.reset_index().merge(data_geom[['id', 'geometry']], on='id', how="inner").set_index('index')
    return df_w_geometry


def load_street_polygons():
    files_sbb = glob.glob(os.path.join(DATA_GEO_DIR, '*', '*_sbb.csv'))
    data_sbb = pd.concat((pd.read_csv(f) for f in files_sbb), ignore_index=True)
    gdf_sbb = to_gdf(data_sbb)
    gdf_sbb = gdf_sbb.drop_duplicates(subset=['geometry'])
    return gdf_sbb.reset_index(drop=True)


def to_gdf(df, crs=2154):
    geo_wkt = df['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_wkt, crs=crs)
