from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor, XGBClassifier
from scipy import stats

import utils
import dataset
from prediction_type import TypeClassifier
from prediction_floors import FloorsPredictor
from prediction_height import HeightPredictor
import preprocessing as pp

XGBOOST_PARAMS = {
    'tree_method': 'hist',
    'device': 'cpu',
    'enable_categorical': True
}
REG_HYPERPARAMETERS = {
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.5,
}
TYPE_HYPERPARAMETERS = {
    'max_depth': 11,
    'learning_rate': 0.2,
    'n_estimators': 200,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.5,
}
RES_TYPE_HYPERPARAMETERS = {
    'max_depth': 9,
    'learning_rate': 0.2,
    'n_estimators': 100,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.5,
}

TYPE_MODEL_PARAMS = dict(
    model=XGBClassifier(**XGBOOST_PARAMS),
    predictor_class=TypeClassifier,
    hyperparameters=TYPE_HYPERPARAMETERS,
    feature_cols=dataset.FEATURES_TYPE + dataset.FEATURES_TYPE_LOCAL,
    test_training_split=pp.split_by_existing_split,
    preprocessing_stages=[
        pp.drop_type_ground_truth_features,
        pp.fill_block_na_with_bldg_features,
        pp.drop_na_cols,
    ],
    mitigate_class_imbalance=True,
    predict_confidence=True,
    labels=dataset.TYPE_LABELS,
    label_groups=[['residential'], ['commercial', 'industrial', 'agricultural', 'public', 'others']],
)

HEIGHT_MODEL_PARAMS = dict(
    model=XGBRegressor(**XGBOOST_PARAMS),
    predictor_class=HeightPredictor,
    hyperparameters=REG_HYPERPARAMETERS,
    feature_cols=dataset.FEATURES_HEIGHT + dataset.FEATURES_HEIGHT_LOCAL,
    test_training_split=pp.split_by_existing_split,
    preprocessing_stages=[
        pp.drop_height_ground_truth_features,
        pp.remove_height_outliers,
        pp.fill_block_na_with_bldg_features,
        pp.drop_na_cols,
    ],
    n_bootstrapping=10,
)

FLOORS_MODEL_PARAMS = dict(
    model=XGBRegressor(**XGBOOST_PARAMS),
    predictor_class=FloorsPredictor,
    hyperparameters=REG_HYPERPARAMETERS,
    feature_cols=dataset.FEATURES_FLOORS + dataset.FEATURES_HEIGHT_LOCAL,
    test_training_split=pp.split_by_existing_split,
    preprocessing_stages=[
        pp.drop_floors_ground_truth_features,
        pp.fill_block_na_with_bldg_features,
        pp.drop_na_cols,
    ],
    n_bootstrapping=10,
)

RES_TYPE_MODEL_PARAMS = dict(
    model=XGBClassifier(**XGBOOST_PARAMS),
    predictor_class=TypeClassifier,
    hyperparameters=RES_TYPE_HYPERPARAMETERS,
    feature_cols=dataset.FEATURES_RES_TYPE + dataset.FEATURES_RES_TYPE_LOCAL,
    test_training_split=pp.split_by_existing_split,
    preprocessing_stages=[
        pp.drop_res_type_ground_truth_features,
        pp.fill_block_na_with_bldg_features,
        pp.drop_na_cols,
    ],
    mitigate_class_imbalance=True,
    predict_confidence=True,
    labels=dataset.RES_TYPE_LABELS,
)

MODEL_PARAMS = {
    'height': HEIGHT_MODEL_PARAMS,
    'floors': FLOORS_MODEL_PARAMS,
    'type': TYPE_MODEL_PARAMS,
    'residential_type': RES_TYPE_MODEL_PARAMS,
}

ATTR_PRED_ORDER = ['type', 'residential_type', 'height', 'floors']


# def infer_missing_region(region, data_dir, results_dir, model_dir, model_scope_path):
#     data_dir = Path(data_dir)
#     results_dir = Path(results_dir)
#     model_dir = Path(model_dir)

#     results_dir.mkdir(parents=True, exist_ok=True)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     df = load_buildings(data_dir, region)
#     df_val = df[['validation']]
#     df_true = df[['height', 'floors', 'age', 'type', 'residential_type']].add_suffix('_true')

#     print(f"--------- Inferring building type for region {region}... --------- ")
#     type_pred = infer_missing_type(df, region, data_dir, results_dir, model_dir, model_scope_path)

#     print(f"--------- Inferring building residential type for region {region}... --------- ")
#     df_res = df[(df['type'] == 'residential') | df.index.isin(type_pred[type_pred['type_pred'] == 'residential'].index)]
#     res_type_pred = infer_missing_residential_type(df_res, region, data_dir, results_dir, model_dir, model_scope_path)

#     print(f"--------- Inferring building height for region {region}... --------- ")
#     height_pred = infer_missing_height(df, region, data_dir, results_dir, model_dir, model_scope_path)

#     print(f"--------- Inferring building floors for region {region}... --------- ")
#     df["height"] = df["height"].fillna(height_pred["height_pred"])  # Ensure consistent height/floors inference
#     floors_pred = infer_missing_floors(df, region, data_dir, results_dir, model_dir, model_scope_path)

#     predictions = pd.concat([df_val, df_true, height_pred, floors_pred, type_pred, res_type_pred], axis=1)
#     predictions['region'] = region
#     predictions.to_parquet(results_dir / f"{region}.parquet")


def infer_missing(data_dir, results_dir, model_dir, model_scope_path, attrs=None, regions=None, scope=None, override=False):
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    model_dir = Path(model_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    regions = regions or pd.read_csv(model_scope_path, index_col=0).index.tolist()

    for region in regions:
        results_path = results_dir / f"{region}.parquet"
        predictions = pd.read_parquet(results_path) if results_path.exists() else pd.DataFrame()

        if not predictions.empty and not override:
            print(f"Predictions for region {region} already exist, skipping...")
            continue

        for attr in attrs or ATTR_PRED_ORDER:
            scope = scope or retrieve_model_scope(region, attr, model_scope_path)
            df = load_training_data(data_dir, results_dir, region, attr)
            f = inference if attr in ["type", "residential_type"] else bootstrapped_inference

            print(f"--------- Inferring building {attr} for region {region} using {scope}-scoped model... --------- ")
            attr_pred = f(
                df=df,
                attr=attr,
                scope=scope,
                infer_preprocessing_stages=[pp.fill_block_na_with_bldg_features],
                data_dir=data_dir,
                results_dir=results_dir,
                model_dir=model_dir,
                **MODEL_PARAMS[attr]
            )

            if attr == 'type':
                attr_pred = attr_pred.rename(columns={'type_confidence_group': 'binary_type_confidence'})

            predictions = attr_pred.combine_first(predictions)

        predictions['region'] = region
        predictions.to_parquet(results_path)


def train_models(data_dir, results_dir, model_dir, model_scope_path, attr):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    results_dir = Path(results_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_scope = pd.read_csv(model_scope_path, index_col=0)
    model_scope_lvl = model_scope[attr].str[-1]
    model_levels = set(region[:int(lvl)+2] for region, lvl in zip(model_scope_lvl.index, model_scope_lvl.values) if lvl != 'U')

    for scope in sorted(list(model_levels), key=len)[:10]:
        try:
            print(f"--------- Training {attr} model for {scope}... --------- ")
            training(
                attr=attr,
                scope=scope,
                data_dir=data_dir,
                results_dir=results_dir,
                model_dir=model_dir,
                **MODEL_PARAMS[attr]
            )
        except Exception as e:
            print(f"Error training model for scope {scope}: {e}")
            continue


def train_country_models(data_dir, model_dir, model_scope_path):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_scope = pd.read_csv(model_scope_path, index_col=0)

    for attr, params in MODEL_PARAMS.items():
        countries = model_scope[model_scope[attr] == 'NUTS0'].index.str[:2].unique()
        for country in countries:
            print(f"--------- Training {attr} model for country {country}... --------- ")
            training(
                attr=attr,
                scope=country,
                data_dir=data_dir,
                model_dir=model_dir,
                **params
            )


def infer_missing_type(df, region, data_dir, results_dir, model_dir, model_scope_path):
    return inference(
        df=df,
        attr='type',
        region=region,
        model_scope_path=model_scope_path,
        infer_preprocessing_stages=[pp.fill_block_na_with_bldg_features],
        data_dir=data_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        **TYPE_MODEL_PARAMS
    ).rename(columns={'type_confidence_group': 'binary_type_confidence'})


def infer_missing_residential_type(df, region, data_dir, results_dir, model_dir, model_scope_path):
    return inference(
        df=df,
        attr='residential_type',
        region=region,
        model_scope_path=model_scope_path,
        infer_preprocessing_stages=[pp.fill_block_na_with_bldg_features],
        data_dir=data_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        **RES_TYPE_MODEL_PARAMS,
    )


def infer_missing_height(df, region, data_dir, results_dir, model_dir, model_scope_path):
    return bootstrapped_inference(
        df=df,
        attr='height',
        region=region,
        model_scope_path=model_scope_path,
        infer_preprocessing_stages=[pp.fill_block_na_with_bldg_features],
        data_dir=data_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        **HEIGHT_MODEL_PARAMS,
    )


def infer_missing_floors(df, region, data_dir, results_dir, model_dir, model_scope_path):
    return bootstrapped_inference(
        df=df,
        attr='floors',
        region=region,
        model_scope_path=model_scope_path,
        infer_preprocessing_stages=[pp.fill_block_na_with_bldg_features],
        data_dir=data_dir,
        results_dir=results_dir,
        model_dir=model_dir,
        **FLOORS_MODEL_PARAMS,
    )


def training(attr, scope, n_bootstrapping=1, **kwargs):
    for i in range(n_bootstrapping):
        dataset.GLOBAL_REPRODUCIBILITY_SEED = i
        load_or_train_predictor(scope=scope, attr=attr, **kwargs)


# fix
def validation(region, attr, scope, data_dir, results_dir, model_dir):
    df = load_training_data(data_dir, results_dir, region, attr)
    mask_val = df['validation'] & (df[attr].notna())
    df_val = df[mask_val]

    dataset.GLOBAL_REPRODUCIBILITY_SEED = 0
    model_path = Path(model_dir) / f"{attr}_{scope}_{dataset.GLOBAL_REPRODUCIBILITY_SEED}.json"
    
    params = MODEL_PARAMS[attr].copy()
    predictor = params.pop("predictor_class").load_model(path=model_path, target_attribute=attr, **params)
    y_pred = predictor.predict(df_val, preprocessing_stages=[pp.fill_block_na_with_bldg_features])

    predictor.y_predict = y_pred
    predictor.y_test = df_val[[attr]]
    predictor.print_model_error()


def inference(df, attr, scope, infer_preprocessing_stages, **kwargs):
    mask_missing = df[attr].isna()
    mask_val = df['validation'] & (df[attr].notna())
    df_pred = df[mask_missing | mask_val]

    dataset.GLOBAL_REPRODUCIBILITY_SEED = 0

    predictor = load_or_train_predictor(scope=scope, attr=attr, **kwargs)
    y_pred = predictor.predict(df_pred, preprocessing_stages=infer_preprocessing_stages)
    y_pred = y_pred.rename(columns={attr: 'pred'})
    y_pred['true'] = df[mask_val][attr]

    return y_pred.add_prefix(f"{attr}_")


def bootstrapped_inference(df, attr, scope, n_bootstrapping, infer_preprocessing_stages, **kwargs):
    mask_missing = df[attr].isna()
    mask_val = df['validation'] & (df[attr].notna())
    df_pred = df[mask_missing | mask_val]

    y_preds = []
    for i in range(n_bootstrapping):
        dataset.GLOBAL_REPRODUCIBILITY_SEED = i

        predictor = load_or_train_predictor(scope=scope, attr=attr, **kwargs)
        y_pred = predictor.predict(df_pred, preprocessing_stages=infer_preprocessing_stages)
        y_preds.append(y_pred[attr])

    y_pred_bs = bootstrapped_average(pd.concat(y_preds, axis=1))
    y_pred_bs['true'] = df[mask_val][attr]

    return y_pred_bs[['true', 'pred', 'confidence_lower', 'confidence_upper', 'std']].add_prefix(f"{attr}_")


def bootstrapped_average(y_pred):
    mean = y_pred.mean(axis=1)
    std = y_pred.std(axis=1)
    t = stats.t.ppf(0.975, df=y_pred.shape[1] - 1)  # 95% CI
    lower = mean - t * std
    upper = mean + t * std

    bootstrapped_results = pd.DataFrame({
        'pred': mean,
        'confidence_lower': lower,
        'confidence_upper': upper,
        'std': std,
    }, index=mean.index)

    return bootstrapped_results


def load_or_train_predictor(scope, attr, data_dir, model_dir, results_dir, model, predictor_class, **kwargs):
    model_path = model_dir / f"{attr}_{scope}_{dataset.GLOBAL_REPRODUCIBILITY_SEED}.json"

    if model_path.exists():
        print(f"Loading model from {model_path}")
        predictor = predictor_class.load_model(model, model_path, target_attribute=attr, **kwargs)
    else:
        print(f"Training new model and saving to {model_path}")
        df = load_training_data(data_dir, results_dir, scope, attr)
        predictor = predictor_class(model=model, df=df, target_attribute=attr, **kwargs)
        predictor.save_model(model_path)
        predictor.print_model_error()

    return predictor


def load_training_data(data_dir, results_dir, region, attr):
    data_path = data_dir / f"{region}*.parquet"
    results_path = results_dir / f"{region}*.parquet"
    cols = list(set([attr] + MODEL_PARAMS[attr]['feature_cols'] + dataset.AUX_VARS))
    df = utils.load_df(data_path, cols=cols)

    # df = df.sample(n=min(len(df), 5_000_000), random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)  # limit to 5 million for memory reasons

    df = fill_missings_with_merged_osm_attributes(df)

    # workarounds - should be done in attr-cleaning 
    df = discard_unrealistic_floors_n_heights(df)
    df = age_cleaning(df)
    df = floors_cleaning(df)

    # # Optional: Load all prior inferred attributes to improve prediction and ensure consistent inter-attribute relationships (e.g. height/floors)
    # pred = utils.load_df(results_path)
    # if pred.empty:
    #     return df

    # for a in ATTR_PRED_ORDER:
    #     if a == attr:
    #         break

    #     df[a] = df[a].fillna(pred[f"{a}_pred"])

    if attr == 'residential_type':
        pred = utils.load_df(results_path)
        if not pred.empty and 'type_pred' in pred.columns:
            df['type'] = df['type'].fillna(pred['type_pred'])

        df = df[df['type'] == 'residential']

    return df


def retrieve_model_scope(region, attr, scope_path):
    model_scope = pd.read_csv(scope_path, index_col=0)
    nuts_level_str = model_scope.loc[region, attr]
    if nuts_level_str == 'EU':
        return 'EU'

    nuts_level = int(nuts_level_str[-1])

    return region[:nuts_level+2]


def fill_missings_with_merged_osm_attributes(df: gpd.GeoDataFrame) -> None:
    if "osm_height_merged" in df.columns:
        df["height"] = df["height"].fillna(
            df[df["osm_height_confidence"] > 0.2]["osm_height_merged"])
        df["floors"] = df["floors"].fillna(
            df[df["osm_floors_confidence"] > 0.2]["osm_floors_merged"])
        df["age"] = df["age"].fillna(
            df[df["osm_age_confidence"] > 0.2]["osm_age_merged"])
        df["type"] = df["type"].fillna(
            df[df["osm_type_confidence"] > 0.2]["osm_type_merged"])
        df["residential_type"] = df["residential_type"].fillna(
            df[df["osm_residential_type_confidence"] > 0.2]["osm_residential_type_merged"])

    return df


def discard_unrealistic_floors_n_heights(df):
    unreal_heights = (df["height"] <= 0) | (df["height"] > 350)
    unreal_floors = (df["floors"] <= 0) | (df["floors"] > 100)
    unreal_floor_height_ratio = (df['floors'] > df['height'] / 1.5) # assuming min floor height of 1.5m

    df.loc[unreal_floor_height_ratio | unreal_heights, "height"] = np.nan
    df.loc[unreal_floor_height_ratio | unreal_floors, "floors"] = np.nan

    return df


def age_cleaning(df):
    df['age'] = df['age'].dropna().astype(str).apply(_extract_year).astype(float)
    df['age'] = df['age'].clip(lower=0)
    df['age'] = df['age'].replace(0, np.nan)
    return df


def floors_cleaning(df):
    df['floors'] = df['floors'].clip(lower=1)
    return df


def _extract_year(s):
    try:
        s = float(s[:4])  # extract year from YYYY-MM-DD encoded string
        if s < 1000:
            return np.nan

        return s

    except Exception:
        return np.nan
