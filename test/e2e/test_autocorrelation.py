import os

import spatial_autocorrelation
import geometry
import preparation

import pandas as pd

path_test_data = os.path.join('test', 'e2e', 'test-data.csv')
test_data = pd.read_csv(path_test_data)


def test_neighbor_exclusion():
    gdf = geometry.to_gdf(test_data)
    I = spatial_autocorrelation.moran_distance(gdf, distance_threshold=50).I

    gdf = preparation.add_block_column(gdf)
    I_exc_neighbors = spatial_autocorrelation.moran_distance(gdf, distance_threshold=50, exc_block_type='block').I

    gdf['block'] = list(range(gdf.shape[0]))
    I_exc_neighbors_one_building_per_block = spatial_autocorrelation.moran_distance(gdf, distance_threshold=50, exc_block_type='block').I

    assert I != I_exc_neighbors
    assert I == I_exc_neighbors_one_building_per_block

