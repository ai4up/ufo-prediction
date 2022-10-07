import os
import logging

import dataset

from sklearn import metrics
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_energy_error(y_true, y_predict, labels=None):
    y_true = assign_heating_energy_demand(y_true, labels)
    y_predict = assign_heating_energy_demand(y_predict, labels)

    r2 = metrics.r2_score(y_true['heating_demand'], y_predict['heating_demand'])
    mape = metrics.mean_absolute_percentage_error(y_true['heating_demand'], y_predict['heating_demand'])

    logger.info(f'R2: {r2:.4f}')
    logger.info(f'MAPE: {mape:.4f}')

    return r2, mape


def assign_heating_energy_demand(df, labels=None):
    tabula_energy_path = os.path.join(dataset.METADATA_DIR, 'TABULA_heating_demand.csv')
    tabula_energy_df = pd.read_csv(tabula_energy_path)

    df = df.dropna(subset=['country', 'residential_type'])

    n_buildings = len(df)
    index = df.index

    if labels:
        # matching on existing age bins (classification)
        class_to_label = dict(enumerate(labels))
        df['age_bin'] = df['age'].map(class_to_label)
        df = pd.merge(df, tabula_energy_df,  how='left', on=['country', 'residential_type', 'age_bin']).set_index('id', drop=False)
    else:
        # binning continuous age (regression)
        df = pd.merge(df, tabula_energy_df,  how='left', on=['country', 'residential_type']).set_index('id', drop=False)
        df = df.query(f'age_min <= {dataset.AGE_ATTRIBUTE} < age_max')

    if n_buildings != len(df):
        logger.error(f'Assigning heating energy demand failed. Number of building changed during merge of TABULA data from {n_buildings} to {len(df)}. Dropped buildings include:\n{list(index.difference(df.index))[:10]}')

    return df
