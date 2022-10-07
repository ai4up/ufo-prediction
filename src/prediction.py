import os
import sys
import gc
import logging
import inspect
import pickle
import copy
import time
import collections
from functools import wraps

import dataset
import utils
import visualizations
import preprocessing
import spatial_autocorrelation
import geometry

import shap
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
import xgboost
import sklearn.ensemble

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor:

    def __init__(
            self,
            model,
            df=None,
            frac=None,
            n_cities=None,
            test_set=None,
            test_training_split=None,
            cross_validation_split=None,
            preprocessing_stages=[],
            target_attribute=None,
            mitigate_class_imbalance=False,
            early_stopping=True,
            hyperparameter_n_iter=20,
            hyperparameter_tuning_space=None,
            hyperparameter_tuning_only=False,
            hyperparameters=None,
            initialize_only=False) -> None:

        self.model = model
        self.df = df
        self.frac = frac
        self.n_cities = n_cities
        self.test_set = test_set
        self.test_training_split = test_training_split
        self.cross_validation_split = cross_validation_split
        self.preprocessing_stages = preprocessing_stages
        self.target_attribute = target_attribute
        self.mitigate_class_imbalance = mitigate_class_imbalance
        self.early_stopping = early_stopping
        self.hyperparameter_n_iter = hyperparameter_n_iter
        self.hyperparameter_tuning_space = hyperparameter_tuning_space
        self.hyperparameter_tuning_only = hyperparameter_tuning_only
        self.hyperparameters = hyperparameters
        self.uuid = utils.truncated_uuid4()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_predict = None
        self.evals_result = None
        self.shap_explainer = None
        self.shap_values = None
        self.sample_weights = None
        self.hyperparameter_tuning_results = None

        if not initialize_only:
            self._e2e_training()


    def _e2e_training(self):
        self._load()
        self._clean()
        # self._preprocess_before_splitting()

        for _ in self._cv_aware_split():
            self._abort_signal()
            self._pre_preprocess_analysis_hook()
            self._preprocess()
            self._post_preprocess_analysis_hook()
            self._train()
            self._predict()


    def _load(self):
        if isinstance(self.df, str):
            self.df = utils.load_df(self.df)

        if isinstance(self.test_set, str):
            self.test_set = utils.load_df(self.test_set)

        if self.frac or self.n_cities:
            self.df = utils.sample_cities(self.df, frac=self.frac, n=self.n_cities)


    def _clean(self):
        self.df.dropna(subset=[self.target_attribute], inplace=True)
        self.df.drop_duplicates(subset=['id'], inplace=True)
        logger.info(f'Dataset length: {len(self.df)}')
        logger.info(f'Dataset allocated memory: {int(self.df.memory_usage(index=True).sum() / 1024 / 1024)} MB')


    def _pre_preprocess_analysis_hook(self):
        logger.info(f'Training dataset length: {len(self.df_train)}')
        logger.info(f'Test dataset length: {len(self.df_test)}')
        logger.info(f'Test cities: {self.df_test["city"].unique()}')


    def _abort_signal(self):
        job = os.environ.get('SLURM_JOBID', 'local')
        path = os.path.join(dataset.METADATA_DIR, f'{job}-{self.uuid}.abort')
        if os.path.exists(path):
            logger.info('Abort signal file found. Exiting...')
            sys.exit(0)
        else:
            logger.info(f'No abort signal received. Continuing... To abort please create {path}.')


    def _post_preprocess_analysis_hook(self):
        logger.info(f'Training dataset length after preprocessing: {len(self.df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(self.df_test)}')


    def _preprocess(self):

        for func in self.preprocessing_stages:
            params = inspect.signature(func).parameters

            if 'df_train' in params and 'df_test' in params:
                self.df_train, self.df_test = func(df_train=self.df_train, df_test=self.df_test)
            else:
                self.df_train = func(self.df_train)
                self.df_test = func(self.df_test)


        self.df_train = sklearn.utils.shuffle(self.df_train, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
        self.df_test = sklearn.utils.shuffle(self.df_test, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)

        self.df_train = self.df_train.set_index('id', drop=False)
        self.df_test = self.df_test.set_index('id', drop=False)

        feature_cols = list(self.df_test.columns.intersection(dataset.FEATURES))

        self.aux_vars_train = self.df_train.drop(columns=feature_cols + [self.target_attribute])
        self.aux_vars_test = self.df_test.drop(columns=feature_cols + [self.target_attribute])

        self.X_train = self.df_train[feature_cols]
        self.y_train = self.df_train[[self.target_attribute]]

        self.X_test = self.df_test[feature_cols]
        self.y_test = self.df_test[[self.target_attribute]]


    def _train(self):
        if self.mitigate_class_imbalance:
            self.sample_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.y_train[self.target_attribute]
            )

        model_params = {
            'random_state': dataset.GLOBAL_REPRODUCIBILITY_SEED,
        }

        fit_params = {
            'X': self.X_train,
            'y': self.y_train,
            'sample_weight': self.sample_weights,
        }

        if self.hyperparameter_tuning_space:
            self._tune_hyperparameters(fit_params)
            return

        if self.hyperparameters:
            model_params = {**model_params, **self.hyperparameters}

        if self._xgboost_model():
            fit_params['verbose'] = utils.verbose()
            fit_params['eval_set'] = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

            if self.early_stopping:
                fit_params['early_stopping_rounds'] = max(50, self.model.n_estimators / 10)

        if self._sklearn_model():
            model_params['verbose'] = 2 if utils.verbose() else 0

        self.model.set_params(**model_params)
        self.model.fit(**fit_params)

        if self._xgboost_model():
            self.evals_result = self.model.evals_result()


    def _predict(self):
        if not self.hyperparameter_tuning_only:
            self.y_predict = pd.DataFrame(
                {self.target_attribute: self.model.predict(self.X_test)}, index=self.X_test.index)


    def _tune_hyperparameters(self, fit_params, grid=False):
        if self.cross_validation_split:
            if not self.df_train.index.equals(self.X_train.index):
                raise Exception('Unexpected index inconsistencies found between df_train and X_train. \
                    The cross_validation_split requires df_train to access auxiliary attributes like the city and assumes the df_train indices to be consistent with X_train.')

            inner_cv = ((train_df.index, test_df.index)
                        for train_df, test_df in self.cross_validation_split(self.df_train.reset_index(drop=True)))
        else:
            inner_cv = preprocessing.N_CV_SPLITS

        if grid:
            clf = model_selection.GridSearchCV(
                estimator=self.model,
                param_grid=self.hyperparameter_tuning_space,
                verbose=2,
                cv=inner_cv,
                return_train_score=True,
                random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED
            )
        else:
            # TODO remove scoring='neg_root_mean_squared_error' to use estimator's score method
            # (however it uses R2 although reg:squarederror is default for xgboost regression)
            clf = model_selection.RandomizedSearchCV(
                estimator=self.model,
                n_iter=self.hyperparameter_n_iter,
                param_distributions=self.hyperparameter_tuning_space,
                scoring='neg_root_mean_squared_error',
                verbose=2,
                cv=inner_cv,
                return_train_score=True,
                random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED
            )

        clf.fit(**fit_params)

        self.model = clf.best_estimator_
        self.hyperparameter_tuning_results = clf.cv_results_
        self.hyperparameters = clf.best_params_

        logger.info(f'Best hyperparameters: {clf.best_params_}')
        logger.info(f'Corresponding score: {clf.best_score_}')
        timestr = time.strftime('%Y%m%d-%H-%M-%S')
        pd.DataFrame(clf.cv_results_).to_csv(f'hyperparameter-tuning-results-{timestr}.csv', sep='\t')


    def _cv_aware_split(self):
        if sum([bool(self.test_training_split), bool(self.cross_validation_split), isinstance(self.test_set, pd.DataFrame)]) > 1:
            raise Exception('Only one of test_training_split, cross_validation_split or test_set can be configured.')

        if self.hyperparameter_tuning_only and self.hyperparameter_tuning_space is None:
            raise Exception('Please specify a hyperparameter_tuning_space to be used for hyperparameter tuning.')


        if self.hyperparameter_tuning_only:
            self.df_train = self.df
            self.df_test = self.df.drop(self.df.index)
            yield

        if self.test_set is not None:
            self.df_train = self.df
            self.df_test = self.test_set
            yield

        if self.cross_validation_split:
            yield from self._cv()

        if self.test_training_split:
            self.df_train, self.df_test = self.test_training_split(self.df)
            yield


    def _cv(self):
        y_predict_all_cv_folds = pd.DataFrame()
        y_test_all_cf_folds = pd.DataFrame()
        aux_vars_test_all_cf_folds = pd.DataFrame()

        for fold_idx, (df_train, df_test) in enumerate(self.cross_validation_split(self.df)):
            self.df_train = df_train
            self.df_test = df_test

            yield

            self.aux_vars_test['cv_fold_idx'] = fold_idx

            y_predict_all_cv_folds = pd.concat([y_predict_all_cv_folds, self.y_predict], axis=0)
            y_test_all_cf_folds = pd.concat([y_test_all_cf_folds, self.y_test], axis=0)
            aux_vars_test_all_cf_folds = pd.concat([aux_vars_test_all_cf_folds, self.aux_vars_test], axis=0)

        self.y_test = y_test_all_cf_folds
        self.y_predict = y_predict_all_cv_folds
        self.aux_vars_test = aux_vars_test_all_cf_folds


    def _do_across_folds(self, func, *args, **kwargs):
        results = []
        y_test = self.y_test
        y_predict = self.y_predict
        aux_vars_test = self.aux_vars_test

        try:
            for _, fold in aux_vars_test.groupby('cv_fold_idx'):
                ids = fold.index.values
                self.y_test = y_test.loc[ids]
                self.y_predict = y_predict.loc[ids]
                self.aux_vars_test = aux_vars_test.loc[ids]

                results.append(func(self, *args, **kwargs))

        finally:
            self.y_test = y_test
            self.y_predict = y_predict
            self.aux_vars_test = aux_vars_test

        return results


    def _garbage_collect(self):
        # prevent compatibility problems between machines with and without GPU support
        if hasattr(self, 'model') and hasattr(self.model, 'tree_method') and self.model.tree_method == 'gpu_hist':
            # workaround for https://github.com/dmlc/xgboost/issues/3045 and https://github.com/dmlc/xgboost/issues/5291
            try:
                self.model.get_booster().set_param({'updater':''})
            except Exception as e:
                logger.error(f'Failed to garbage collect model: {e}')
            delattr(self, 'model')

        for attr in ['df', 'df_test', 'df_train', 'X_train', 'y_train', 'sample_weights', 'aux_vars_train']:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()


    def _xgboost_model(self):
        return getattr(self.model, '__module__', None) == xgboost.__name__


    def _sklearn_model(self):
        return getattr(self.model, '__module__', None) == sklearn.ensemble.__name__


    @staticmethod
    def load(path):
        predictor = pickle.load(open(path, 'rb'))

        if isinstance(predictor, Predictor):
            return predictor

        logger.error(f'The object loaded from {path} is not a Predictor instance.')


    def save(self, path, results_only=False):
        if results_only:
            self._garbage_collect()

        pickle.dump(self, open(path, 'wb'))


    def cv_aware(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            if kwargs.pop('across_folds', None) == True:
                return self._do_across_folds(f, *args, **kwargs)

            return f(self, *args, **kwargs)

        return wrapped


    def evaluate(self):
        raise NotImplementedError('To be implemented.')


    def calculate_SHAP_values(self):
        if self.shap_values is None:
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.shap_explainer.shap_values(self.X_test)
        return self.shap_values


    def SHAP_analysis(self):
        self.calculate_SHAP_values()
        shap.summary_plot(self.shap_values, self.X_test)
        shap.summary_plot(self.shap_values, self.X_test, plot_type='bar')


    def normalized_feature_importance(self):
        # Calculate feature importance based on SHAP values
        self.calculate_SHAP_values()

        avg_shap_value = np.abs(self.shap_values).mean(0)
        normalized_shap_value = avg_shap_value / sum(avg_shap_value)
        feature_names = self.X_test.columns

        feature_importance = pd.DataFrame(
            {'feature': feature_names, 'normalized_importance': normalized_shap_value})
        return feature_importance.sort_values(by=['normalized_importance'], ascending=False)


    def feature_selection(self):
        if 'feature_noise' not in self.X_test.columns:
            raise Exception(
                "feature_noise column missing. Please add 'add_noise_feature' preprocessing step before doing feature selection.")

        df_fi = self.normalized_feature_importance()

        # Dismiss features which have a lower impact than the noise feature
        significance_level = 0.005
        noise_feature_importance = df_fi.query(
            "feature=='feature_noise'").normalized_importance.values[0]
        threshold = df_fi.normalized_importance > noise_feature_importance + significance_level

        selected_features = df_fi[threshold]
        # remove noise feature from this list
        excluded_features = df_fi[~threshold].iloc[1:]

        print(f'{len(excluded_features)} of {len(self.X_test.columns)-1} features have been excluded:')
        print(excluded_features)

        return selected_features, excluded_features


    def feature_dependence_plot(self, feature1, feature2, low_percentile=0, high_percentile=100, transparency=1):
        self.calculate_SHAP_values()
        shap.dependence_plot(
            feature1,
            self.shap_values,
            self.X_test,
            interaction_index=feature2,
            xmin=f"percentile({low_percentile})",
            xmax=f"percentile({high_percentile})",
            alpha=transparency)
        plt.show()


    def neighborhood_feature_importance(self):
        df_fi = self.normalized_feature_importance()
        df_neighborhood_fi = df_fi.loc[df_fi['feature'].str.contains(
            'within_buffer')]
        return round(sum(df_neighborhood_fi.normalized_importance), 2)


    def print_feature_importance(self):
        feature_accuracy_contribution = self.model.get_booster(
        ).get_score(importance_type="gain")
        feature_importance = pd.DataFrame(
            {'importance': feature_accuracy_contribution})
        feature_importance.sort_values(
            by=['importance'], ascending=False, inplace=True)
        print(feature_importance.head(15))


class Regressor(Predictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @Predictor.cv_aware
    def print_model_error(self):
        print('MAE: {:.2f} y'.format(self.mae()))
        print('RMSE: {:.2f} y'.format(self.rmse()))
        print('R2: {:.4f}'.format(self.r2()))


    @Predictor.cv_aware
    def mae(self):
        return metrics.mean_absolute_error(self.y_test, self.y_predict)


    @Predictor.cv_aware
    def rmse(self):
        return np.sqrt(metrics.mean_squared_error(self.y_test, self.y_predict))


    @Predictor.cv_aware
    def r2(self):
        return metrics.r2_score(self.y_test, self.y_predict)


    @Predictor.cv_aware
    def kurtosis(self):
        return stats.kurtosis(self.y_test - self.y_predict)[0]


    @Predictor.cv_aware
    def skew(self):
        return stats.skew(self.y_test - self.y_predict)[0]


    @Predictor.cv_aware
    def individual_prediction_error(self):
        df = self.y_predict - self.y_test
        df = df.rename(columns={self.target_attribute: 'error'})
        return df


    @Predictor.cv_aware
    def error_cum_hist(self, bins):
        residuals = self.individual_prediction_error()['error'].abs()
        hist = np.histogram(residuals, bins=bins)[0] / len(residuals)
        cum_hist = np.cumsum(hist)
        return cum_hist


    @Predictor.cv_aware
    def mcc(self, bins):
        y_test_cat = pd.cut(self.y_test[self.target_attribute], bins, right=False).cat.codes
        y_predict_cat = pd.cut(self.y_predict.loc[y_test_cat.index][self.target_attribute], bins, right=False).cat.codes
        return metrics.matthews_corrcoef(y_test_cat, y_predict_cat)


    def eval_metrics(self):
        eval_df = pd.DataFrame(columns=['R2', 'MAE', 'RMSE', 'Kurtosis', 'Skew'])

        for col in eval_df.columns:
            calc_metric = getattr(self, col.lower())
            eval_df.at['total', col] = calc_metric()

            if self.cross_validation_split:
                for fold, value in enumerate(calc_metric(across_folds=True)):
                        eval_df.at[f'fold_{fold}', col] = value

        return eval_df


    @Predictor.cv_aware
    def spatial_autocorrelation_moran(self, attribute, type):
        if attribute == 'error':
            y = self.individual_prediction_error()
        elif attribute == self.target_attribute:
            y = self.y_predict
        else:
            raise Exception(f'Please specify either "error" or "{self.target_attribute}" as the attribute for calculation spatial autocorrelation.')

        aux_df = pd.concat([y, self.aux_vars_test], axis=1, join="inner").reset_index()

        if type == 'block':
            moran = spatial_autocorrelation.moran_within_block(aux_df, attribute=attribute)
        elif type == 'knn':
            moran = spatial_autocorrelation.moran_knn(geometry.to_gdf(aux_df), attribute=attribute)
        elif type == 'distance':
            moran = spatial_autocorrelation.moran_distance(geometry.to_gdf(aux_df), attribute=attribute)
        else:
            raise Exception('Please specify either "knn", "block" or "distance", as type for calculation spatial autocorrelation.')

        logger.info(f'Moran I for spatial autocorrelation of {attribute}: {moran.I:.4f} ({type} weights with p value of {moran.p_norm:.4f})')
        return moran


    @Predictor.cv_aware
    def prediction_error_distribution(self, bins=[0, 10, 20, np.inf]):
        error_df = self.y_predict - self.y_test
        prediction_error_bins = np.histogram(error_df[self.target_attribute].abs(), bins)[0] / len(error_df)
        logger.info(f'Distribution of prediction error: {error_df.describe()}')
        logger.info(f'Prediction error bins: {list(zip(utils.generate_labels(bins), np.around(prediction_error_bins, 2)))}')
        return prediction_error_bins


class Classifier(Predictor):

    def __init__(self, labels, predict_probabilities=False, initialize_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs, initialize_only=True)

        self.labels = labels
        self._validate_labels()
        self.multiclass = len(self.labels) > 2
        self.predict_probabilities = predict_probabilities

        objective = 'multi:softprob' if self.multiclass else 'binary:logistic'
        eval_metric = ['mlogloss', 'merror'] if self.multiclass else ['logloss', 'error']
        self.model.set_params(objective=objective, eval_metric=eval_metric, use_label_encoder=False)

        if not initialize_only:
            self._e2e_training()


    def _predict(self):
        if not self.predict_probabilities:
            return super()._predict()

        class_probabilities = self.model.predict_proba(self.X_test)
        class_drawn = np.apply_along_axis(self._sample_class_from_probabilities,
                                          axis=1, arr=class_probabilities).ravel()
        self.y_predict = pd.DataFrame({self.target_attribute: class_drawn, 'probabilities': list(class_probabilities)})


    def _validate_labels(self):
        if not isinstance(self.df, pd.DataFrame):
            logger.warning('Labels can not be validated because no DataFrame but a file path is passed to the predictor.')
            return

        labels_dataset = self.df[self.target_attribute].unique()
        if len(self.labels) > len(labels_dataset):
            self.labels = [l for l in self.labels if l in labels_dataset]
            logger.error(f'Some labels are not in the dataset. They will be ignored. New labels are {self.labels}')

        if len(self.labels) < len(labels_dataset):
            raise Exception(f'Length of labels provided ({self.labels}) does not match the labels in the dataset ({labels_dataset}).')


    def _sample_class_from_probabilities(self, prob):
        np.random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
        classes = list(range(0, len(self.labels)))
        sampled_class = np.random.choice(classes, 1, p=prob)
        return sampled_class


    @Predictor.cv_aware
    def classification_report(self):
        report = metrics.classification_report(
            self.y_test, self.y_predict[[self.target_attribute]], target_names=self.labels, output_dict=True)
        return pd.DataFrame(report).transpose().astype({'support': int})


    @Predictor.cv_aware
    def kappa(self):
        return metrics.cohen_kappa_score(self.y_test, self.y_predict[[self.target_attribute]])


    @Predictor.cv_aware
    def mcc(self):
        return metrics.matthews_corrcoef(self.y_test, self.y_predict[[self.target_attribute]])


    @Predictor.cv_aware
    def f1(self):
        return metrics.f1_score(self.y_test, self.y_predict[[self.target_attribute]], average='macro')


    @Predictor.cv_aware
    def recall(self, label_idx):
        return metrics.recall_score(self.y_test, self.y_predict[[self.target_attribute]], pos_label=label_idx, labels=[label_idx], average='macro')


    def eval_metrics(self):
        eval_df = self.classification_report()
        eval_df.at['total', 'kappa'] = self.kappa()
        eval_df.at['total', 'mcc'] = self.mcc()
        eval_df.at['total', 'accuracy'] = eval_df['recall']['accuracy']
        eval_df.drop('accuracy', inplace=True)

        if self.cross_validation_split:
            for fold, mcc in enumerate(self.mcc(across_folds=True)):
                eval_df.at[f'fold_{fold}', 'mcc'] = mcc

        return eval_df


    @Predictor.cv_aware
    def print_model_error(self):
        print(f'Classification report:\n {self.classification_report()}')
        print(f'Cohen’s kappa: {self.kappa():.4f}')
        print(f'Matthews correlation coefficient (MCC): {self.mcc():.4f}')


    def normalized_feature_importance(self):
        # Calculate feature importance based on SHAP values
        self.calculate_SHAP_values()

        # average across classes for multiclass classification
        axis = (0, 1) if np.array(self.shap_values).ndim == 3 else 0

        avg_shap_value = np.abs(self.shap_values).mean(axis=axis)
        normalized_shap_value = avg_shap_value / sum(avg_shap_value)
        feature_names = self.X_train.columns

        feature_importance = pd.DataFrame(
            {'feature': feature_names, 'normalized_importance': normalized_shap_value})
        return feature_importance.sort_values(by=['normalized_importance'], ascending=False)


    def feature_dependence_plot(self, feature1, feature2, low_percentile=0, high_percentile=100, transparency=1):
        self.calculate_SHAP_values()

        # binary classification
        if np.array(self.shap_values).ndim != 3:
            shap.dependence_plot(
                feature1,
                self.shap_values,
                self.X_train,
                interaction_index=feature2,
                xmin=f"percentile({low_percentile})",
                xmax=f"percentile({high_percentile})",
                alpha=transparency)
            return

        # multiclass classification
        axis = utils.grid_subplot(len(self.shap_values))
        for idx, class_shap_values in enumerate(self.shap_values):
            shap.dependence_plot(
                feature1,
                class_shap_values,
                self.X_train,
                interaction_index=feature2,
                xmin=f"percentile({low_percentile})",
                xmax=f"percentile({high_percentile})",
                alpha=transparency,
                ax=axis[idx],
                title=self.labels[idx],
                show=False)
        plt.show()


class PredictorComparison:

    def __init__(
            self,
            predictor,
            comparison_config,
            grid_comparison_config={'': {}},
            compare_feature_importance=False,
            garbage_collect_after_training=False,
            include_baseline=True,
            n_seeds=1,
            **baseline_kwargs) -> None:

        self.comparison_config = comparison_config
        self.grid_comparison_config = grid_comparison_config
        self.compare_feature_importance = compare_feature_importance
        self.garbage_collect_after_training = garbage_collect_after_training
        self.include_baseline = include_baseline
        self.baseline_kwargs = baseline_kwargs
        self.n_seeds = n_seeds
        self.predictors = collections.defaultdict(list)

        if self.include_baseline:
            self.predictors['baseline'] = predictor(**copy.deepcopy(self.baseline_kwargs))

            if self.compare_feature_importance:
                self.predictors['baseline'].calculate_SHAP_values()

        for grid_experiment_name, grid_experiment_kwargs in self.grid_comparison_config.items():
            for experiment_name, experiment_kwargs in self.comparison_config.items():
                name = f'{experiment_name}_{grid_experiment_name}'
                kwargs = {**copy.deepcopy(self.baseline_kwargs), **grid_experiment_kwargs, **experiment_kwargs}
                logger.info(f'Starting experiment {name}...')

                for seed in range(self.n_seeds):
                    logger.debug(f'Training predictor ({name}) (seed {seed}) with following args:\n{kwargs}')
                    dataset.GLOBAL_REPRODUCIBILITY_SEED = seed
                    self.predictors[name].append(predictor(**kwargs))

                    if self.compare_feature_importance:
                        self.predictors[name][seed].calculate_SHAP_values()

                    if self.garbage_collect_after_training:
                        self.predictors[name][seed]._garbage_collect()


    def save(self, path, results_only=False):
        file_name, ext = os.path.splitext(path)
        for name, predictors in self.predictors.items():
            for seed, predictor in enumerate(predictors):
                predictor.save(f'{file_name}_{name}_{seed}_{ext}', results_only)


    def evaluate_feature_importance(self, normalize_by_number_of_features=True):
        if not self.include_baseline:
            raise Exception('Evaluating feature importance changes is only possible if a baseline prediction is defined.')

        baseline_importance_df = self.predictors.get('baseline').normalized_feature_importance().set_index('feature')

        for name, predictors in self.predictors.items():
            importance_df = predictors[0].normalized_feature_importance().set_index('feature')
            normalization_factor = len(importance_df) if normalize_by_number_of_features else 1
            baseline_importance_df['diff_' + name] = (importance_df['normalized_importance'] -
                                                      baseline_importance_df['normalized_importance']) * normalization_factor

        baseline_importance_df['var'] = baseline_importance_df.var(axis=1)

        for name in self.grid_comparison_config.keys():
            columns = [c for c in baseline_importance_df.columns if name in c]
            baseline_importance_df['agg_diff_' + name] = baseline_importance_df[columns].sum(axis=1)

        return baseline_importance_df.sort_values(by='var', ascending=False)


    def plot_feature_importance_changes(self):
        dfs = [p.normalized_feature_importance() for predictors in self.predictors.values() for p in predictors]
        all_top_5_features = set().union(*[df[:5]['feature'].values for df in dfs])
        visualizations.slope_chart(dfs, labels=self.predictors.keys(), feature_selection=all_top_5_features)


    def _mean(self, predictors, func, *args, **kwargs):
        return np.mean([getattr(p, func)(*args, **kwargs) for p in predictors], axis=0)


    def _std(self, predictors, func, *args, **kwargs):
        return np.std([getattr(p, func)(*args, **kwargs) for p in predictors], axis=0)
