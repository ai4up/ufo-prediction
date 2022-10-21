import logging

import pandas as pd
import libpysal as lps
import splot
from esda.moran import Moran

import dataset
import preparation
import utils

logger = logging.getLogger(__name__)


def moran_within_block(df, attribute=dataset.AGE_ATTRIBUTE):
    df = df.dropna(subset=[attribute])

    if not 'block_bld_ids' in df.columns:
        df = preparation.add_block_building_ids_column(df)

    weights = _within_block_weights(df)
    return Moran(df[attribute], weights)


def moran_within_sbb(df, attribute=dataset.AGE_ATTRIBUTE):
    df = df.dropna(subset=[attribute])

    if not 'sbb_bld_ids' in df.columns:
        df = preparation.add_sbb_building_ids_column(df)

    weights = _within_sbb_weights(df)
    return Moran(df[attribute], weights)


def moran_between_blocks(df, attribute=dataset.AGE_ATTRIBUTE):
    df = df.dropna(subset=[attribute])

    if not 'block' in df.columns:
        df = preparation.add_block_column(df)

    weights = _between_blocks_weights(df)
    return Moran(df[attribute], weights)


def moran_between_sbbs(df, attribute=dataset.AGE_ATTRIBUTE):
    df = df.dropna(subset=[attribute])

    if not 'sbb' in df.columns:
        raise Exception(f'Street-based block (sbb) column not found in dataset. Run add_street_block_column() to prepare the dataset.')

    weights = _between_sbbs_weights(df)
    return Moran(df[attribute], weights)


def moran_between_groups(df_group_1, df_group_2, attribute=dataset.AGE_ATTRIBUTE):
    df_group_1 = df_group_1.dropna(subset=[attribute])
    df_group_2 = df_group_2.dropna(subset=[attribute])

    ids_train = df_group_1['id'].values
    ids_test = df_group_2['id'].values

    df = pd.concat([df_group_1, df_group_2], axis=0, ignore_index=True)

    neighbors = {**dict.fromkeys(ids_train, ids_test), **dict.fromkeys(ids_test, ids_train)}
    weights = lps.weights.W(neighbors, id_order=df['id'].values, silence_warnings=True)

    return Moran(df[attribute], weights)


def moran_distance(gdf, distance_threshold=15, attribute=dataset.AGE_ATTRIBUTE, exc_block_type=None):
    gdf = gdf.dropna(subset=[attribute])

    if exc_block_type:
        weights = _distance_weights_exc_block(gdf, exc_block_type, distance_threshold)
    else:
        weights = _distance_weights(gdf, distance_threshold)

    return Moran(gdf[attribute], weights)


def moran_knn(gdf, k=4, attribute=dataset.AGE_ATTRIBUTE):
    gdf = gdf.dropna(subset=[attribute])

    weights = _knn_weights(gdf, k)
    return Moran(gdf[attribute], weights)


def moran_I(df, weights, attribute=dataset.AGE_ATTRIBUTE):
    return Moran(df[attribute], weights).I


def features_moran_I(df, weight_func, features=dataset.FEATURES):
    weights = weight_func(df)
    m_features = []
    for feat in set(df.columns).intersection(features):
        m = Moran(df[feat], weights)
        m_features.append({'feature': feat, 'moran_I': m.I, 'moran_p': m.p_sim})

    return pd.DataFrame(m_features).sort_values(by='moran_I', ascending=False)


def plot_correlogram_over_distance(df, attributes, distances=None):
    distances = distances or [2 ** i for i in range(2, 11)]
    coefficients = []
    neighbors = {}
    for dis in distances:
        weights = _distance_band_weights(df, distance_threshold=dis, neighbors_to_exclude=neighbors)
        neighbors = weights.neighbors
        coefficients.append({attr: Moran(df[attr], weights).I for attr in attributes})

    df = pd.DataFrame(coefficients, index=distances)
    ax = df.plot(kind='line', title='Spatial autocorrelation over distance')
    ax.set_ylabel("Moran's I")
    ax.set_xlabel('distance [m]')
    return df


def plot_neighbors_histogram(weights, bin_size=5):
    n_neighbors = pd.Series(len(l) for l in weights.neighbors.values())
    hist = utils.custom_round(n_neighbors, base=bin_size).value_counts(normalize=True).sort_index()
    hist.plot(kind='bar', title='Histogram of spatial weight neighbors')
    return hist


def plot_few_spatial_weights(weights, gdf, every_n=100):
    few_ids = list(weights.neighbors.keys())[dataset.GLOBAL_REPRODUCIBILITY_SEED::every_n]
    few_neighbors = {k: v for k, v in weights.neighbors.items() if k in few_ids}
    few_dis = lps.weights.W(few_neighbors, ids=few_neighbors.keys(), silence_warnings=True)
    splot.libpysal.plot_spatial_weights(few_dis, gdf, indexed_on='id')


def _within_block_weights(df):
    neighbors = df[['id', 'block_bld_ids']].set_index('id').to_dict()['block_bld_ids']
    _remove_neighbors_missing_in_dataset(neighbors)
    neighbors_exc_self = {k: v[1:] for k, v in neighbors.items()}
    return lps.weights.W(neighbors_exc_self, id_order=df['id'].values, silence_warnings=True)


def _within_sbb_weights(df):
    neighbors = df[['id', 'sbb_bld_ids']].set_index('id').to_dict()['sbb_bld_ids']
    neighbors_exc_self = {k: v[1:] for k, v in neighbors.items()}
    return lps.weights.W(neighbors_exc_self, id_order=df['id'].values, silence_warnings=True)


def _between_blocks_weights(df):
    neighbors = _neighboring_blocks_buildings(df, 'block', 50)
    return lps.weights.W(neighbors, id_order=df['id'].values, silence_warnings=True)


def _between_sbbs_weights(df):
    neighbors = _neighboring_blocks_buildings(df, 'sbb', 100)
    return lps.weights.W(neighbors, id_order=df['id'].values, silence_warnings=True)


def _distance_weights(gdf, distance_threshold):
    return lps.weights.DistanceBand.from_dataframe(gdf, threshold=distance_threshold, silence_warnings=True)


def _distance_band_weights(gdf, distance_threshold, neighbors_to_exclude):
    dis = lps.weights.DistanceBand.from_dataframe(gdf, threshold=distance_threshold, ids=gdf['id'].values, silence_warnings=True)
    neighbors = {}
    for building_id, neighbor_ids in dis.neighbors.items():
        neighbors[building_id] = [n for n in neighbor_ids if n not in neighbors_to_exclude.get(building_id, [])]
    return lps.weights.W(neighbors, id_order=gdf['id'].values, silence_warnings=True)


def _distance_weights_exc_block(gdf, block_type, distance_threshold):
    if not block_type in gdf.columns:
        raise Exception(f'block_type {block_type} not found in columns. Consider executing add_block_column() or add_street_block_column() to prepare the dataset.')

    dis = lps.weights.DistanceBand.from_dataframe(gdf, threshold=distance_threshold, ids=gdf['id'].values, silence_warnings=True)
    neighbors_exc_block = utils.exclude_neighbors_from_own_block(dis.neighbors, gdf, block_type)

    return lps.weights.W(neighbors_exc_block, id_order=gdf['id'].values, silence_warnings=True)


def _knn_weights(gdf, k):
    return lps.weights.distance.KNN.from_dataframe(gdf, k=k, silence_warnings=True)


def _remove_neighbors_missing_in_dataset(neighbors_dict):
    for neighbors in neighbors_dict.values():
        for n in neighbors.copy():
            if not n in neighbors_dict.keys():
                neighbors.remove(n)


def _neighboring_blocks_buildings(df, block_type, distance_threshold):
    if not block_type in df.columns:
        raise Exception(f'block_type {block_type} not found in columns. Consider executing add_block_column() or add_street_block_column() to prepare the dataset.')

    dis = lps.weights.DistanceBand.from_dataframe(df, threshold=distance_threshold, ids=df['id'].values, silence_warnings=True)
    id_to_block = df[['id', block_type]].set_index('id').to_dict()[block_type]

    neighbor_blocks = {}
    for building_id, neighbor_ids in dis.neighbors.items():
        own_block = id_to_block[building_id]
        neighbor_blocks[building_id] = [id_to_block[id] for id in neighbor_ids if id_to_block[id] != own_block]

    block_to_ids = df.groupby(block_type)['id'].apply(list).to_dict()

    neighbor_buildings = {}
    for building_id, blocks in neighbor_blocks.items():
        neighbor_buildings[building_id] = utils.flatten([block_to_ids[block] for block in blocks])

    return neighbor_buildings
