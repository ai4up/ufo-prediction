import logging

import geometry

import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def prepare(df):
    df = geometry.add_geometry_column(df, crs=3035)
    df = add_block_column(df)
    df = add_street_block_column(df)
    df = add_neighborhood_column(df)
    return df


def add_block_column(df):
    df['block'] = df.groupby(['city', 'TouchesIndexes']).ngroup()
    return df


def add_neighborhood_column(gdf, max_neighborhood_size_km=1):
    columns = list(gdf.columns)
    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.info('Using lat lon coordinates of building instead of full geometry to determine street block centroids. The result may vary slightly.')
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']))


    sbb_centroids = gdf.dissolve(by='sbb').to_crs(4326).centroid
    distance_matrix = geometry.distance_matrix(sbb_centroids)

    ac = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=max_neighborhood_size_km)
    clusters = ac.fit(distance_matrix)

    logger.info(f'On average {int(len(clusters.labels_) / len(set(clusters.labels_)))} street blocks have been assigned per neighborbood cluster.')

    sbb_to_neighborhood = dict(zip(sbb_centroids.index, clusters.labels_))
    gdf['neighborhood'] = gdf['sbb'].map(sbb_to_neighborhood)

    return gdf[columns + ['neighborhood']]


def add_street_block_column(df):
    columns = list(df.columns)
    if not isinstance(df, gpd.GeoDataFrame):
        df = geometry.add_geometry_column(df)

    sbb_gdf = geometry.load_street_geometry()
    geometry.ensure_same_crs(df, sbb_gdf)

    joined_gdf = gpd.sjoin(df, sbb_gdf[['geometry']], how="left", op="within")
    joined_gdf.rename(columns={'index_right': 'sbb'}, inplace=True)
    joined_gdf.dropna(subset=['sbb'], inplace=True)
    joined_gdf['sbb'] = joined_gdf['sbb'].astype(int)

    if any(joined_gdf.duplicated(subset='id')):
        logger.warning('Spatial joining resulted in duplicate buildings in dataset. Most likely street polygons were overlapping and buildings were assigned to more than one during gpd.sjoin().')
        logger.info('Removing duplicated buildings, keeping only first one.')
        joined_gdf.drop_duplicates(subset='id', keep='first', inplace=True)

    return joined_gdf[columns + ['sbb']]


def add_block_building_ids_column(df):
    if not 'block' in df.columns:
        df = add_block_column(df)

    block_building_ids = df.groupby('block')['id'].apply(list).to_dict()
    df['block_bld_ids'] = df['block'].map(block_building_ids)
    return df


def add_sbb_building_ids_column(df):
    if not 'sbb' in df.columns:
        df = add_street_block_column(df)

    block_building_ids = df.groupby('sbb')['id'].apply(list).to_dict()
    df['sbb_bld_ids'] = df['sbb'].map(block_building_ids)
    return df
