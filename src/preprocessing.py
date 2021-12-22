import dataset

import pandas as pd
import numpy as np
from sklearn import model_selection


def remove_other_attributes(df_input):
    return df_input.drop(columns=dataset.OTHER_ATTRIBUTES)


def keep_other_attributes(df):
    # Remove all buildings that do not have one of our four variables (age/type/floor/height).
    df = df.dropna(subset=dataset.OTHER_ATTRIBUTES+[dataset.AGE_ATTRIBUTE])
    df = df[df[dataset.TYPE_ATTRIBUTE] != 'Indifférencié']

    # Encode 'usage type', which is a categorical variable, into multiple dummy variables.
    df = dummy_encoding(df, dataset.TYPE_ATTRIBUTE)
    return df


def dummy_encoding(df, var):
    one_hot_encoding = pd.get_dummies(df[var], prefix=var)
    df = df.drop(columns=[var])
    df = df.merge(
        one_hot_encoding, left_index=True, right_index=True)
    return df


def remove_outliers(df):
    return df[df[dataset.AGE_ATTRIBUTE] > 1920]


def categorize_age(df):
    bins = [0, 1915, 1945, 1965, 1980, 2000, np.inf]
    names = ['<1915', '1915-1944', '1945-1964',
             '1965-1979', '1980-2000', '>2000']

    # bins = [0, 1945, 1980, np.inf]
    # names = ['<1944', '1945-1979', '>1980']

    df[dataset.AGE_ATTRIBUTE] = pd.cut(
        df[dataset.AGE_ATTRIBUTE], bins, labels=names).cat.codes

    return df


def add_noise_feature(df):
    df["feature_noise"] = np.random.normal(size=len(df))
    return df


def split_80_20(df):
    return model_selection.train_test_split(df, test_size=0.2)


def split_by_region(df):
    # We aim to cross-validate our results using five French sub-regions 'departement' listed below.
    # one geographic region for validation, rest for testing
    region_names = ['Haute-Vienne', 'Hauts-de-Seine',
                    'Aisne', 'Orne', 'Pyrénées-Orientales']
    df_test = df[df.departement == region_names[0]]
    df_train = df[~df.index.isin(df_test.index)]
    return df_train, df_test
