import os
import glob
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, cdist, squareform
import pyproj
from shapely import wkt
from haversine import haversine

import dataset

logger = logging.getLogger(__name__)


def to_gdf(df, crs=3035):
    if isinstance(df, gpd.GeoDataFrame):
        logger.warning('Dataset is already a GeoDataFrame. Skipping convertion.')
        return df

    if type(df['geometry'].dtype) == gpd.array.GeometryDtype:
        return gpd.GeoDataFrame(df)

    geo_wkt = df['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_wkt, crs=crs)


def lat_lon_to_gdf(df, crs=3035):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=4326).to_crs(crs)


def add_geometry_column(df, crs=3035, countries=[]):
    cities = list(df['city'].values) if 'city' in df.columns else []
    data_geom = load_building_geometry(crs, countries, cities)

    data_geom['id'] = data_geom['id'].astype(str)
    df['id'] = df['id'].astype(str)
    df = df.reset_index(drop=True)
    df_w_geometry = data_geom[['id', 'geometry']].merge(df, on='id', how="inner")
    return df_w_geometry.set_index('id')


def nearest_neighbors(df, distance_threshold, identifier='id'):
    neighbors = {}
    ids = list(df[identifier].values)
    matrix = distance_matrix(df['geometry'].centroid)

    for building_idx, building_neighbors in enumerate(matrix):
        neighbors[ids[building_idx]] = [ids[idx] for idx, dis in enumerate(building_neighbors) if dis < distance_threshold and idx != building_idx]

    return neighbors


def knn(df, k, identifier='id'):
    neighbors = {}
    ids = list(df[identifier].values)
    matrix = distance_matrix(df['geometry'].centroid)

    for building_idx, building_neighbors in enumerate(matrix):
        knn_indices = np.argsort(building_neighbors)[1:k+1]
        neighbors[ids[building_idx]] = [ids[idx] for idx in knn_indices]

    return neighbors


def spatial_buffer_around_block(df, block_type, buffer_size_meters, block_ids=None):
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy().to_crs(4326)
    else:
        logger.info('Using lat lon coordinates of building instead of full geometry to determine street block centroids. The result may vary slightly.')
        gdf = gpd.GeoDataFrame(df[['id', block_type]], geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=4326)

    # improve runtime by considering only buildings from k nearest blocks when determining neighbors instead of all available buildings
    block_centroids = gdf.dissolve(by=block_type).reset_index()
    k_nearest_blocks = knn(block_centroids, k=8, identifier=block_type)

    buildings_in_buffer = []
    for block, nearest_blocks in k_nearest_blocks.items():
        if block_ids is None or block in block_ids:
            nearest_blocks_gdf = gdf[gdf[block_type].isin(nearest_blocks)]
            block_gdf = gdf[gdf[block_type] == block]
            building_ids = list(nearest_blocks_gdf['id'].values)

            nearest_blocks_geom = nearest_blocks_gdf['geometry'].centroid
            block_geom = block_gdf['geometry'].centroid
            matrix = pairwise_distance_matrix(block_geom, nearest_blocks_geom)

            for building_neighbors in matrix:
                buildings_in_buffer.extend([building_ids[idx] for idx, dis in enumerate(building_neighbors) if dis < buffer_size_meters / 1000])

    return buildings_in_buffer


def distance_matrix(geometry):
    coords = list(zip(geometry.x, geometry.y))
    metric = haversine if geometry.crs == pyproj.CRS(4326) else 'euclidean'
    return squareform(pdist(coords, metric))


def pairwise_distance_matrix(geometry_1, geometry_2):
    coords_1 = list(zip(geometry_1.x, geometry_1.y))
    coords_2 = list(zip(geometry_2.x, geometry_2.y))
    metric = haversine if geometry_1.crs == pyproj.CRS(4326) else 'euclidean'
    return cdist(coords_1, coords_2, metric)


def load_building_geometry(crs=3035, countries=[], cities=[]):
    gdf = _load_geometry('geom', crs, countries, cities)
    gdf.drop_duplicates(subset=['id'], inplace=True)
    return gdf


def load_street_geometry(crs=3035, countries=[]):
    try:
        file = os.path.join(dataset.DATA_DIR, f'sbb-{"-".join(countries)}.pkl')
        df = pd.read_pickle(file)
        gdf = to_gdf(df, crs=crs)
    except FileNotFoundError:
        gdf = _load_geometry('sbb', crs, countries)

    return gdf


def ensure_same_crs(gdf_1, gdf_2):
    if not isinstance(gdf_1, gpd.GeoDataFrame) or not isinstance(gdf_2, gpd.GeoDataFrame):
        raise Exception('Passed dataframes must be Geopandas dataframe.')

    gdf_2 = gdf_2.to_crs(gdf_1.crs)


def prepare_street_polygons(crs=3035, countries=[]):
    gdf_sbb = _load_geometry('sbb', crs, countries)
    gdf_sbb = gdf_sbb.drop_duplicates(subset=['geometry'])
    gdf_sbb = _merge_intersecting_geometries(gdf_sbb)
    return gdf_sbb.reset_index(drop=True)


def _merge_intersecting_geometries(gdf, aggfunc='first'):
    overlap_matrix = gdf.geometry.apply(lambda x: gdf.intersects(x)).values.astype(int)
    _, distinct_groups = csgraph.connected_components(overlap_matrix)
    gdf['group'] = distinct_groups
    return gdf.dissolve(by='group', aggfunc=aggfunc)


def _determine_crs(country):
    file_gadm = os.path.join(dataset.METADATA_DIR, 'gadm_table.csv')
    gadm_info = pd.read_csv(file_gadm)

    crs_strings = gadm_info[gadm_info['country_name'] == country]['local_crs'].values
    if not crs_strings:
        return None

    crs_code = crs_strings[0].split("EPSG:", 1)[1]
    return int(crs_code)


def _load_geometry(type, crs=3035, countries=[], cities=[]):
    country_dirs = list(os.walk(dataset.DATA_DIR))[0][1]
    selected_countries = set(countries).intersection(country_dirs) if countries else country_dirs
    gdfs = []

    for country in selected_countries:
        if not (country_crs := _determine_crs(country)):
            logger.warning('CRS for country directory "{country}" not found. Skipping directory.')
            continue

        files_geom = glob.glob(os.path.join(dataset.DATA_DIR, country, '**', f'*_{type}.csv'), recursive=True)

        if cities:
            files_geom = [f for f in files_geom if any(city in f for city in cities)]

        if not files_geom:
            continue

        dfs = []
        for f in files_geom:
            df = pd.read_csv(f)
            df['city'] = f.rsplit('/')[-2]
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['geometry'])
        country_gdf = to_gdf(df, crs=country_crs).to_crs(crs)
        country_gdf = country_gdf[country_gdf['geometry'].is_valid]
        gdfs.append(country_gdf)

    return pd.concat(gdfs, ignore_index=True).reset_index(drop=True)
