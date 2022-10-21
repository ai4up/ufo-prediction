#!/usr/bin/env python

import os
import gc
import shutil
import logging
import json

import pandas as pd

import lib_importer
import utils
import geometry
import preparation
import dataset
import cluster_utils.dataset as cluster_dataset

# PIK cluster
DATA_DIR = '/p/projects/eubucco/data/2-database-city-level-v0_1'
RESULT_DIR = '/p/projects/eubucco/data/3-ml-inputs'
COUNTRIES = {'netherlands': 'NLD', 'france': 'FRA', 'spain': 'ESP'}

# local test
# DATA_DIR = 'data/prepared'
# RESULT_DIR = 'data/results'
# COUNTRIES = {'france': 'FRA', 'netherlands': 'NLD'}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def prepare():
    for country, abbr in COUNTRIES.items():
        df_path = _full_path(f'df-{abbr}.pkl')
        gdf_path = _full_path(f'sbb-{abbr}.pkl')

        logger.info(f'(1) {abbr} - Loading building data...')
        if os.path.isfile(df_path):
            df = pd.read_pickle(df_path)
            # df = df[df['city'].isin(['Bayeux', 'Montélimar'])]
        else:
            df = cluster_dataset.load(
                countries=[country],
                path=DATA_DIR,
                dropna_for_col=dataset.AGE_ATTRIBUTE,
                seed=dataset.GLOBAL_REPRODUCIBILITY_SEED
                )
            df.to_pickle(df_path)

        logger.info(f'(2) {abbr} - Loading street data...')
        if os.path.isfile(gdf_path):
            sbb_gdf = pd.read_pickle(gdf_path)
        else:
            sbb_gdf = geometry.prepare_street_polygons(countries=[country])
            sbb_gdf.to_pickle(gdf_path)

        logger.info(f'(3) {abbr} - Preparing dataset...')
        df['country'] = abbr
        df = preparation.prepare(df, sbb_gdf)

        logger.info(f'(4) {abbr} - Storing prepared dataset...')
        shutil.move(df_path, df_path + '.bak')
        df.to_pickle(df_path)

        cities = _preliminary_exp_cities(country)
        df_pre = df[df['city'].isin(cities)]
        df_exp = df[~df['city'].isin(cities)]

        df_pre.to_pickle(_full_path(f'df-{abbr}-preliminary.pkl'))
        df_exp.to_pickle(_full_path(f'df-{abbr}-exp.pkl'))

        del df
        del df_pre
        del df_exp
        del sbb_gdf
        gc.collect()

    logger.info(f'(5) {"-".join(COUNTRIES.values())} - Storing combined prepared dataset...')
    df_all = pd.concat([pd.read_pickle(_full_path(f'df-{abbr}-exp.pkl')) for abbr in COUNTRIES.values()], ignore_index=True)
    df_all.to_pickle(_full_path(f'df-{"-".join(COUNTRIES.values())}-exp.pkl'))


def _preliminary_exp_cities(country):
    with open(os.path.join(RESULT_DIR, 'preliminary-exp-cities.json'), 'r', encoding='utf-8') as f:
        cities = json.load(f)
        cities = utils.flatten([region_cities for type_cities in cities[country].values() for region_cities in type_cities.values()])
        return cities


def _full_path(filename):
    return os.path.join(RESULT_DIR, filename)


if __name__ == '__main__':
    prepare()