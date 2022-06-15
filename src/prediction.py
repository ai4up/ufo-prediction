import logging
import inspect
import pickle
import copy
from functools import wraps

import dataset
import utils
import visualizations
import spatial_autocorrelation
import geometry

import shap
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from memory_profiler import profile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor:

    @profile
    def __init__(
            self,
            model,
            df,
            test_training_split=None,
            cross_validation_split=None,
            preprocessing_stages=[],
            target_attribute=None,
            mitigate_class_imbalance=False,
            early_stopping=True,
            hyperparameter_tuning=False,
            hyperparameters=None,
            initialize_only=False) -> None:

        self.model = model
        self.df = df.copy()
        self.test_training_split = test_training_split
        self.cross_validation_split = cross_validation_split
        self.preprocessing_stages = preprocessing_stages
        self.target_attribute = target_attribute
        self.mitigate_class_imbalance = mitigate_class_imbalance
        self.early_stopping = early_stopping
        self.hyperparameter_tuning = hyperparameter_tuning
        self.hyperparameters = hyperparameters

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_predict = None
        self.evals_result = None
        self.shap_explainer = None
        self.shap_values = None
        self.sample_weights = None

        if not initialize_only:
            self._e2e_training()

    @profile
    def _e2e_training(self):
        self._clean()
        # self._preprocess_before_splitting()

        for _ in self._cv_aware_split():
            self._pre_preprocess_analysis_hook()
            self._preprocess()
            self._post_preprocess_analysis_hook()
            self._train()
            self._predict()


    def _clean(self):
        self.df.dropna(subset=[self.target_attribute], inplace=True)
        self.df.drop_duplicates(subset=['id'], inplace=True)
        logger.info(f'Dataset length: {len(self.df)}')
        logger.info(f'Dataset allocated memory: {self.df.memory_usage(index=True).sum()}')


    def _pre_preprocess_analysis_hook(self):
        logger.info(f'Training dataset length: {len(self.df_train)}')
        logger.info(f'Test dataset length: {len(self.df_test)}')


    def _post_preprocess_analysis_hook(self):
        logger.info(f'Training dataset length after preprocessing: {len(self.df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(self.df_test)}')


    @profile
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


    @profile
    def _train(self):
        if self.hyperparameter_tuning:
            self.hyperparameters = utils.tune_hyperparameter(
                self.model, self.X_train, self.y_train)

        if self.hyperparameters:
            self.model.set_params(**self.hyperparameters)

        if self.mitigate_class_imbalance:
            self.sample_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.y_train[self.target_attribute]
            )

        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        early_stopping_rounds = self.model.n_estimators / 10 if self.early_stopping else None
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights, verbose=False,
                       eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
        self.evals_result = self.model.evals_result()


    def _predict(self):
        self.y_predict = pd.DataFrame(
            {self.target_attribute: self.model.predict(self.X_test)}, index=self.X_test.index)


    def _cv_aware_split(self):
        if not self.test_training_split and not self.cross_validation_split:
            logger.exception('Please specify either a test_training_split or cross_validation_split function.')

        if self.test_training_split and self.cross_validation_split:
            logger.warning('Both, a test_training_split and cross_validation_split function are specified. The cross_validation_split function will take precedence and test_training_split will be ignored.')

        if not self.cross_validation_split:
            self.df_train, self.df_test = self.test_training_split(self.df)
            yield
            return

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
        try:
            for _, fold in self.aux_vars_test.groupby('cv_fold_idx'):
                ids = fold.index.values
                self.y_test = y_test.loc[ids]
                self.y_predict = y_predict.loc[ids]

                results.append(func(self, *args, **kwargs))

        finally:
            self.y_test = y_test
            self.y_predict = y_predict

        return results


    @staticmethod
    def load(path):
        predictor = pickle.load(open(path, 'rb'))

        if isinstance(predictor, Predictor):
            return predictor

        logger.error(f'The object loaded from {path} is not a Predictor instance.')


    def save(self, path):
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
            self.shap_values = self.shap_explainer.shap_values(self.X_train)
        return self.shap_values


    def SHAP_analysis(self):
        self.calculate_SHAP_values()
        shap.summary_plot(self.shap_values, self.X_train)
        shap.summary_plot(self.shap_values, self.X_train, plot_type='bar')


    def normalized_feature_importance(self):
        # Calculate feature importance based on SHAP values
        self.calculate_SHAP_values()

        avg_shap_value = np.abs(self.shap_values).mean(0)
        normalized_shap_value = avg_shap_value / sum(avg_shap_value)
        feature_names = self.X_train.columns

        feature_importance = pd.DataFrame(
            {'feature': feature_names, 'normalized_importance': normalized_shap_value})
        return feature_importance.sort_values(by=['normalized_importance'], ascending=False)


    def feature_selection(self):
        if 'feature_noise' not in self.X_train.columns:
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

        print(f'{len(excluded_features)} of {len(self.X_train.columns)-1} features have been excluded:')
        print(excluded_features)

        return selected_features, excluded_features


    def feature_dependence_plot(self, feature1, feature2, low_percentile=0, high_percentile=100, transparency=1):
        self.calculate_SHAP_values()
        shap.dependence_plot(
            feature1,
            self.shap_values,
            self.X_train,
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


    def individual_prediction_error(self):
        df = self.y_predict - self.y_test
        df = df.rename(columns={'age': 'error'})
        return df


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


    def _sample_class_from_probabilities(self, prob):
        np.random.seed(dataset.GLOBAL_REPRODUCIBILITY_SEED)
        classes = list(range(0, len(self.labels)))
        sampled_class = np.random.choice(classes, 1, p=prob)
        return sampled_class


    def classification_report(self):
        return metrics.classification_report(
            self.y_test, self.y_predict[[self.target_attribute]], target_names=self.labels)


    def kappa(self):
        return metrics.cohen_kappa_score(self.y_test, self.y_predict[[self.target_attribute]])


    def mcc(self):
        return metrics.matthews_corrcoef(self.y_test, self.y_predict[[self.target_attribute]])


    def f1(self):
        return metrics.f1_score(self.y_test, self.y_predict[[self.target_attribute]], average='macro')


    def recall(self, label_idx):
        return metrics.recall_score(self.y_test, self.y_predict[[self.target_attribute]], pos_label=label_idx, labels=[label_idx], average='macro')


    def print_classification_report(self):
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
            **baseline_kwargs) -> None:

        self.comparison_config = comparison_config
        self.grid_comparison_config = grid_comparison_config
        self.compare_feature_importance = compare_feature_importance
        self.baseline_kwargs = baseline_kwargs
        self.predictors = {}
        self.predictors['baseline'] = predictor(**copy.deepcopy(self.baseline_kwargs))

        if self.compare_feature_importance:
            self.predictors['baseline'].calculate_SHAP_values()

        for grid_experiment_name, grid_experiment_kwargs in self.grid_comparison_config.items():
            for experiment_name, experiment_kwargs in self.comparison_config.items():

                name = f'{experiment_name}_{grid_experiment_name}'
                kwargs = {**copy.deepcopy(self.baseline_kwargs), **grid_experiment_kwargs, **experiment_kwargs}
                logger.info(f'Starting experiment {name}...')
                logger.debug(f'Training predictor ({name}) with following args:\n{kwargs}')

                self.predictors[name] = predictor(**kwargs)

                if self.compare_feature_importance:
                    self.predictors[name].calculate_SHAP_values()


    def evaluate_feature_importance(self, normalize_by_number_of_features=True):
        baseline_importance_df = self.predictors.get('baseline').normalized_feature_importance().set_index('feature')

        for name, predictor in self.predictors.items():
            importance_df = predictor.normalized_feature_importance().set_index('feature')
            normalization_factor = len(importance_df) if normalize_by_number_of_features else 1
            baseline_importance_df['diff_' + name] = (importance_df['normalized_importance'] -
                                                      baseline_importance_df['normalized_importance']) * normalization_factor

        baseline_importance_df['var'] = baseline_importance_df.var(axis=1)

        for name in self.grid_comparison_config.keys():
            columns = [c for c in baseline_importance_df.columns if name in c]
            baseline_importance_df['agg_diff_' + name] = baseline_importance_df[columns].sum(axis=1)

        return baseline_importance_df.sort_values(by='var', ascending=False)


    def plot_feature_importance_changes(self):
        dfs = [p.normalized_feature_importance() for p in self.predictors.values()]
        all_top_5_features = set().union(*[df[:5]['feature'].values for df in dfs])
        visualizations.slope_chart(dfs, labels=self.predictors.keys(), feature_selection=all_top_5_features)
