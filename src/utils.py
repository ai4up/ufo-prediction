import dataset

import pandas as pd


def age_bins(y, bin_size=1):
    min_age = int(y[dataset.AGE_ATTRIBUTE].min())
    max_age = int(y[dataset.AGE_ATTRIBUTE].max())
    return list(range(min_age, max_age+2))[0::bin_size]


def dummy_encoding(df, var):
    one_hot_encoding = pd.get_dummies(df[var], prefix=var)
    df = df.drop(columns=[var])
    df = df.merge(
        one_hot_encoding, left_index=True, right_index=True)
    return df


def custom_round(column, base=5):
    return column.apply(lambda x: int(base * round(float(x)/base)))


def split_target_var(df):
    X = df.drop(columns=[dataset.AGE_ATTRIBUTE])
    y = df[[dataset.AGE_ATTRIBUTE]]
    return X, y


def duplicates(df):
    print(len(df))
    print(len(df['ID'].unique()))
    return df[df.duplicated(keep=False)]
