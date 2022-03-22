import logging

import dataset
import preparation
import utils

import pandas as pd
import libpysal as lps
from esda.moran import Moran

logger = logging.getLogger(__name__)


def moran_within_block(df, attribute=dataset.AGE_ATTRIBUTE):
    if not 'block_bld_ids' in df.columns:
        df = preparation.add_block_building_ids_column(df)

    weights = _within_block_weights(df)
    return Moran(df[attribute], weights)


def moran_within_sbb(df, attribute=dataset.AGE_ATTRIBUTE):
    if not 'sbb_bld_ids' in df.columns:
        df = preparation.add_sbb_building_ids_column(df)

    weights = _within_sbb_weights(df)
    return Moran(df[attribute], weights)


def moran_between_blocks(df, attribute=dataset.AGE_ATTRIBUTE):
    if not 'block' in df.columns:
        df = preparation.add_block_column(df)

    weights = _between_blocks_weights(df)
    return Moran(df[attribute], weights)


def moran_between_sbbs(df, attribute=dataset.AGE_ATTRIBUTE):
    if not 'sbb' in df.columns:
        df = preparation.add_street_block_column(df)

    weights = _between_sbbs_weights(df)
    return Moran(df[attribute], weights)


def moran_distance(gdf, distance_threshold=15, attribute=dataset.AGE_ATTRIBUTE):
    weights = _distance_weights(gdf, distance_threshold)
    return Moran(gdf[attribute], weights)


def moran_knn(gdf, k=4, attribute=dataset.AGE_ATTRIBUTE):
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


def _within_block_weights(df):
    neighbors = df[['id', 'block_bld_ids']].set_index('id').to_dict()['block_bld_ids']
    _remove_neighbors_missing_in_dataset(neighbors)
    neighbors_exc_self = {k: v[1:] for k, v in neighbors.items()}
    return lps.weights.W(neighbors_exc_self, ids=neighbors_exc_self.keys(), silence_warnings=True)


def _within_sbb_weights(df):
    neighbors = df[['id', 'sbb_bld_ids']].set_index('id').to_dict()['sbb_bld_ids']
    neighbors_exc_self = {k: v[1:] for k, v in neighbors.items()}
    return lps.weights.W(neighbors_exc_self, ids=neighbors_exc_self.keys(), silence_warnings=True)


def _between_blocks_weights(df):
    neighbors = _neighboring_blocks_buildings(df, 'block', 50)
    return lps.weights.W(neighbors, ids=neighbors.keys(), silence_warnings=True)


def _between_sbbs_weights(df):
    neighbors = _neighboring_blocks_buildings(df, 'sbb', 100)
    return lps.weights.W(neighbors, ids=neighbors.keys(), silence_warnings=True)


def _distance_weights(gdf, distance_threshold):
    return lps.weights.DistanceBand.from_dataframe(gdf, threshold=distance_threshold, silence_warnings=True)


def _knn_weights(gdf, k):
    return lps.weights.distance.KNN.from_dataframe(gdf, k=k, silence_warnings=True)


def _remove_neighbors_missing_in_dataset(neighbors_dict):
    for neighbors in neighbors_dict.values():
        for n in neighbors.copy():
            if not n in neighbors_dict.keys():
                neighbors.remove(n)


def _neighboring_blocks_buildings(df, block_type, threshold_distance):
    if not block_type in df.columns:
        raise Exception(f'block_type {block_type} not found in columns. Consider executing add_block_column() or add_street_block_column() to prepare the dataset.')

    dis = lps.weights.DistanceBand.from_dataframe(df, threshold=threshold_distance, ids=df['id'].values, silence_warnings=True)
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
