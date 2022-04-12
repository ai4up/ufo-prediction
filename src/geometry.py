import os
import glob
import logging

import pandas as pd
import geopandas as gpd
from scipy.sparse import csgraph
from shapely import wkt
from scipy.spatial.distance import pdist, squareform
from haversine import haversine

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join('..', 'data')
DATA_GEO_DIR = os.path.join(DATA_DIR, 'geometry')


def to_gdf(df, crs=3035):
    if isinstance(df, gpd.GeoDataFrame):
        logger.warning('Dataset is already a GeoDataFrame. Skipping convertion.')
        return df

    if type(df['geometry'].dtype) == gpd.array.GeometryDtype:
        return gpd.GeoDataFrame(df)

    geo_wkt = df['geometry'].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geo_wkt, crs=crs)


def add_geometry_column(df, crs=3035, countries=[]):
    cities = list(df['city'].values) if 'city' in df.columns else []
    data_geom = load_building_geometry(crs, countries, cities)

    data_geom['id'] = data_geom['id'].astype(str)
    df['id'] = df['id'].astype(str)
    df = df.reset_index()
    df_w_geometry = data_geom[['id', 'geometry']].merge(df, on='id', how="inner")
    return df_w_geometry.set_index('index')


def distance_matrix(geometry):
    coords = list(zip(geometry.x, geometry.y))
    return squareform(pdist(coords, haversine))


def load_building_geometry(crs=3035, countries=[], cities=[]):
    gdf = _load_geometry('geom', crs, countries, cities)
    gdf.drop_duplicates(subset=['id'], inplace=True)
    return gdf


def load_street_geometry(crs=3035, countries=[]):
    try:
        file = os.path.join(DATA_DIR, 'harmonized', 'sbb.csv')
        df = pd.read_csv(file)
        gdf = to_gdf(df, crs=crs)
    except FileNotFoundError:
        gdf = _load_geometry('sbb', crs, countries)

    return gdf


def prepare_street_polygons_file(crs=3035):
    gdf_sbb = _load_geometry('sbb', crs)
    gdf_sbb = gdf_sbb.drop_duplicates(subset=['geometry'])
    gdf_sbb = merge_intersecting_geometries(gdf_sbb)
    df = pd.DataFrame(gdf_sbb).reset_index(drop=True)
    df.to_csv(os.path.join(DATA_DIR, 'harmonized', 'sbb.csv'), index=False)


def merge_intersecting_geometries(gdf, aggfunc='first'):
    overlap_matrix = gdf.geometry.apply(lambda x: gdf.intersects(x)).values.astype(int)
    _, distinct_groups = csgraph.connected_components(overlap_matrix)
    gdf['group'] = distinct_groups
    return gdf.dissolve(by='group', aggfunc=aggfunc)


def ensure_same_crs(gdf_1, gdf_2):
    if not isinstance(gdf_1, gpd.GeoDataFrame) or not isinstance(gdf_2, gpd.GeoDataFrame):
        raise Exception('Passed dataframes must be Geopandas dataframe.')

    gdf_2 = gdf_2.to_crs(gdf_1.crs)


def _determine_crs(country):
    file_gadm = os.path.join(DATA_DIR, 'gadm_table.csv')
    gadm_info = pd.read_csv(file_gadm)

    crs_strings = gadm_info[gadm_info['country_name'] == country]['local_crs'].values
    if not crs_strings:
        return None

    crs_code = crs_strings[0].split("EPSG:", 1)[1]
    return int(crs_code)


def _load_geometry(type, crs=3035, countries=[], cities=[]):
    country_dirs = list(os.walk(DATA_GEO_DIR))[0][1]
    selected_countries = set(countries).intersection(country_dirs) if countries else country_dirs
    gdfs = []

    for country in selected_countries:
        if not (country_crs := _determine_crs(country)):
            logger.warning('CRS for country directory "{country}" not found. Skipping directory.')
            continue

        files_geom = glob.glob(os.path.join(DATA_GEO_DIR, country, '**', f'*_{type}.csv'), recursive=True)

        if cities:
            files_geom = [f for f in files_geom if any(city in f for city in cities)]

        if not files_geom:
            continue

        df = pd.concat([pd.read_csv(f) for f in files_geom], ignore_index=True)
        df = df.drop_duplicates(subset=['geometry'])
        country_gdf = to_gdf(df, crs=country_crs).to_crs(crs)
        country_gdf = country_gdf[country_gdf['geometry'].is_valid]
        gdfs.append(country_gdf)

    return pd.concat(gdfs, ignore_index=True).reset_index(drop=True)
