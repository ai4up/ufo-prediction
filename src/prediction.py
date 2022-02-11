import logging
import inspect

import dataset
import utils

import shap
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor:

    def __init__(self, model, df, test_training_split, preprocessing_stages=[], target_attribute=None, mitigate_class_imbalance=False, hyperparameter_tuning=False, hyperparameters=None, initialize_only=False) -> None:
        self.model = model
        self.df = df.copy()
        self.test_training_split = test_training_split
        self.preprocessing_stages = preprocessing_stages
        self.target_attribute = target_attribute
        self.mitigate_class_imbalance = mitigate_class_imbalance
        self.hyperparameter_tuning = hyperparameter_tuning
        self.hyperparameters = hyperparameters

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_predict = None
        self.shap_explainer = None
        self.shap_values = None
        self.sample_weights = None

        if not initialize_only:
            self._preprocess()
            self._train()
            self._predict()


    def _preprocess(self):
        self.df.dropna(subset=[self.target_attribute], inplace=True)
        self.df.drop_duplicates(subset=['id'], inplace=True)
        logger.info(f'Dataset length: {len(self.df)}')

        # Test & Training Split
        df_train, df_test = self.test_training_split(self.df)
        logger.info(f'Training dataset length: {len(df_train)}')
        logger.info(f'Test dataset length: {len(df_test)}')

        # Preprocessing & Cleaning
        for func in self.preprocessing_stages:
            params = inspect.signature(func).parameters

            if 'df_train' in params and 'df_test' in params:
                df_train, df_test = func(df_train=df_train, df_test=df_test)
            else:
                df_train = func(df_train)
                df_test = func(df_test)

        logger.info(f'Training dataset length after preprocessing: {len(df_train)}')
        logger.info(f'Test dataset length after preprocessing: {len(df_test)}')

        df_train = sklearn.utils.shuffle(df_train, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)
        df_test = sklearn.utils.shuffle(df_test, random_state=dataset.GLOBAL_REPRODUCIBILITY_SEED)

        self.aux_vars_train = df_train[dataset.AUX_VARS]
        self.aux_vars_test = df_test[dataset.AUX_VARS]

        self.X_train = df_train.drop(columns=dataset.AUX_VARS+[self.target_attribute])
        self.y_train = df_train[[self.target_attribute]]

        self.X_test = df_test.drop(columns=dataset.AUX_VARS+[self.target_attribute])
        self.y_test = df_test[[self.target_attribute]]

        self.aux_vars_train.reset_index(drop=True, inplace=True)
        self.aux_vars_test.reset_index(drop=True, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)


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
        early_stopping = self.model.n_estimators / 10
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights, verbose=False, eval_set=eval_set, early_stopping_rounds=early_stopping)


    def _predict(self):
        self.y_predict = pd.DataFrame(
            {self.target_attribute: self.model.predict(self.X_test)})


    def evaluate(self):
        raise NotImplementedError("To be implemented.")


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
        shap.dependence_plot(feature1, self.shap_values, self.X_train, interaction_index=feature2,
                             xmin=f"percentile({low_percentile})", xmax=f"percentile({high_percentile})", alpha=transparency)


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


    def print_model_error(self):
        print('MAE: {} y'.format(
            metrics.mean_absolute_error(self.y_test, self.y_predict)))
        print('RMSE: {} y'.format(
            np.sqrt(metrics.mean_squared_error(self.y_test, self.y_predict))))
        print('R2: {}'.format(metrics.r2_score(self.y_test, self.y_predict)))


    def individual_prediction_error(self):
        df = self.aux_vars_test[['id']]
        df['error'] = self.y_predict - self.y_test
        return df


    def prediction_error_distribution(self, bins=[0, 10, 20, np.inf]):
        error_df = self.y_predict - self.y_test
        prediction_error_bins = np.histogram(error_df[dataset.AGE_ATTRIBUTE].abs(), bins)[0] / len(error_df)
        print(f'Distribution of prediction error: {error_df.describe()}')
        print(f'Prediction error bins: {list(zip(utils.generate_labels(bins), np.around(prediction_error_bins, 2)))}')
        return prediction_error_bins


class Classifier(Predictor):

    def __init__(self, labels, predict_probabilities=False, initialize_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs, initialize_only=True)

        self.labels = labels
        self.multiclass = len(self.labels) > 2
        self.predict_probabilities = predict_probabilities

        objective = 'multi:softprob' if self.multiclass else 'binary:logistic'
        eval_metric = ['mlogloss', 'merror'] if self.multiclass else  ['logloss', 'error']
        self.model.set_params(objective=objective, eval_metric=eval_metric, use_label_encoder=False)

        if not initialize_only:
            self._preprocess()
            self._train()
            self._predict()


    def _predict(self):
        if not self.predict_probabilities:
            return super()._predict()

        class_probabilities = self.model.predict_proba(self.X_test)
        class_drawn = np.apply_along_axis(self._sample_class_from_probabilities, axis=1, arr=class_probabilities).ravel()
        self.y_predict = pd.DataFrame({self.target_attribute: class_drawn, 'probabilities': list(class_probabilities)})


    def _sample_class_from_probabilities(self, prob):
        classes = list(range(0, len(self.labels)))
        sampled_class = np.random.choice(classes, 1, p=prob)
        return sampled_class


    def print_classification_report(self):
        print(metrics.classification_report(self.y_test, self.y_predict[[self.target_attribute]]))
        print(f"Cohen’s kappa: {metrics.cohen_kappa_score(self.y_test, self.y_predict[[self.target_attribute]])}")


class PredictorComparison:

    def __init__(self, predictor, comparison_config, grid_comparison_config={'':{}}, compare_feature_importance=False, **baseline_kwargs) -> None:
        self.comparison_config = comparison_config
        self.grid_comparison_config = grid_comparison_config
        self.compare_feature_importance = compare_feature_importance
        self.baseline_kwargs = baseline_kwargs
        self.predictors = {}
        self.predictors['baseline'] = predictor(**self.baseline_kwargs)

        if self.compare_feature_importance:
            self.predictors['baseline'].calculate_SHAP_values()

        for grid_experiment_name, grid_experiment_kwargs in self.grid_comparison_config.items():
            for experiment_name, experiment_kwargs in self.comparison_config.items():

                kwargs = {**self.baseline_kwargs.copy(), **grid_experiment_kwargs, **experiment_kwargs}
                self.predictors[f'{experiment_name}_{grid_experiment_name}'] = predictor(**kwargs)

                if self.compare_feature_importance:
                    self.predictors[f'{experiment_name}_{grid_experiment_name}'].calculate_SHAP_values()


    def evaluate_feature_importance(self, normalize_by_number_of_features=True):
        baseline_importance_df = self.predictors.get('baseline').normalized_feature_importance().set_index('feature')

        for name, predictor in self.predictors.items():
            importance_df = predictor.normalized_feature_importance().set_index('feature')
            normalization_factor = len(importance_df) if normalize_by_number_of_features else 1
            baseline_importance_df['diff_' + name] = (importance_df['normalized_importance'] - baseline_importance_df['normalized_importance']) * normalization_factor

        baseline_importance_df['var'] = baseline_importance_df.var(axis=1)

        for name in self.grid_comparison_config.keys():
            columns = [c for c in baseline_importance_df.columns if name in c]
            baseline_importance_df['agg_diff_' + name] = baseline_importance_df[columns].sum(axis=1)

        return baseline_importance_df.sort_values(by='var', ascending=False)