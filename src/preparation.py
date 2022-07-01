import logging

import geometry
import utils

import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def prepare(df, sbb_gdf):
    gdf = geometry.lat_lon_to_gdf(df)
    gdf = add_block_column(gdf)
    gdf = gdf.groupby('city', as_index=False).apply(lambda city_gdf: add_street_block_column(city_gdf, sbb_gdf[sbb_gdf['city'] == city_gdf.name]))
    gdf = gdf.groupby('city', as_index=False).apply(add_neighborhood_column)
    return gdf


def add_block_column(df):
    df['block'] = utils.seq_to_unique_id(df.groupby(['city', 'TouchesIndexes']).ngroup())
    return df


def add_neighborhood_column(gdf, max_neighborhood_size_m=1000):
    columns = list(gdf.columns)
    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.info('Using lat lon coordinates of building instead of full geometry to determine street block centroids. The result may vary slightly.')
        gdf = geometry.lat_lon_to_gdf(gdf)

    sbb_centroids = gdf.dissolve(by='sbb').to_crs(3035).centroid

    if len(sbb_centroids) < 2:
        logger.info('Not enough street-based blocks to cluster them into neighborhoods.')
        gdf['neighborhood'] = utils.seq_to_unique_id(gdf['sbb'])
        return gdf[columns + ['neighborhood']]

    distance_matrix = geometry.distance_matrix(sbb_centroids)

    ac = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=max_neighborhood_size_m)
    clusters = ac.fit(distance_matrix)

    logger.info(f'On average {int(len(clusters.labels_) / len(set(clusters.labels_)))} street blocks have been assigned per neighborbood cluster.')

    sbb_to_neighborhood = dict(zip(sbb_centroids.index, clusters.labels_))
    gdf['neighborhood'] = utils.seq_to_unique_id(gdf['sbb'].map(sbb_to_neighborhood))

    return gdf[columns + ['neighborhood']]


def add_street_block_column(gdf, sbb_gdf):
    columns = list(gdf.columns)
    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.info('Using lat lon coordinates of building instead of full geometry to determine street block centroids. The result may vary slightly.')
        gdf = geometry.lat_lon_to_gdf(gdf)

    geometry.ensure_same_crs(gdf, sbb_gdf)
    logger.debug(f'Building geom: {gdf.geometry.head(10)}')
    logger.debug(f'Street geom: {sbb_gdf.geometry.head(10)}')

    gdf = gpd.sjoin(gdf, sbb_gdf[['geometry']], how="left", op="within")
    gdf['sbb'] = gdf['index_right']
    gdf.drop(columns=['index_right'], inplace=True)
    gdf.dropna(subset=['sbb'], inplace=True)
    gdf['sbb'] = utils.seq_to_unique_id(gdf['sbb'])

    if any(gdf.duplicated(subset='id')):
        logger.warning('Spatial joining resulted in duplicate buildings in dataset. Most likely street polygons were overlapping and buildings were assigned to more than one during gpd.sjoin().')
        logger.info('Removing duplicated buildings, keeping only first one.')
        gdf.drop_duplicates(subset='id', keep='first', inplace=True)

    return gdf[set(columns + ['sbb'])]


def add_block_building_ids_column(df):
    if not 'block' in df.columns:
        df = add_block_column(df)

    block_building_ids = df.groupby('block')['id'].apply(list).to_dict()
    df['block_bld_ids'] = df['block'].map(block_building_ids)
    return df


def add_sbb_building_ids_column(df):
    if not 'sbb' in df.columns:
        raise Exception(f'Street-based block (sbb) column not found in dataset. Run add_street_block_column() to prepare the dataset.')

    block_building_ids = df.groupby('sbb')['id'].apply(list).to_dict()
    df['sbb_bld_ids'] = df['sbb'].map(block_building_ids)
    return df
