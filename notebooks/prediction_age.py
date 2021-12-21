import logging

import shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import utils, metrics, model_selection

logger = logging.getLogger(__name__)
age_bins = [1850, 1915, 1945, 1965, 1980, 2000, 2025]

AGE_ATTRIBUTE = 'DATE_APP'
TYPE_ATTRIBUTE = 'USAGE1'
HEIGHT_ATTRIBUTE = 'HAUTEUR'
OTHER_ATTRIBUTES = [TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'NB_ETAGES']
AUX_VARS = ['ID', 'USAGE2', 'PREC_ALTI', 'NB_LOGTS', 'MAT_TOITS', 'MAT_MURS',
            'geometry', 'city', 'departement', 'is_buffer', 'TouchesIndexes']


def remove_other_attributes(df_input):
    return df_input.drop(columns=OTHER_ATTRIBUTES)


def keep_other_attributes(df):
    # Remove all buildings that do not have one of our four variables (age/type/floor/height).
    df = df.dropna(subset=OTHER_ATTRIBUTES+[AGE_ATTRIBUTE])
    df = df[df[TYPE_ATTRIBUTE] != 'Indifférencié']

    # Encode 'usage type', which is a categorical variable, into multiple dummy variables.
    df = dummy_encoding(df, TYPE_ATTRIBUTE)
    return df


def dummy_encoding(df, var):
    one_hot_encoding = pd.get_dummies(df[var], prefix=var)
    df = df.drop(columns=[var])
    df = df.merge(
        one_hot_encoding, left_index=True, right_index=True)
    return df


def remove_outliers(df):
    return df[df[AGE_ATTRIBUTE] > 1920]


def categorize_age(df):
    bins = [0, 1915, 1945, 1965, 1980, 2000, np.inf]
    names = ['<1915', '1915-1944', '1945-1964',
             '1965-1979', '1980-2000', '>2000']

    # bins = [0, 1945, 1980, np.inf]
    # names = ['<1944', '1945-1979', '>1980']

    df[AGE_ATTRIBUTE] = pd.cut(
        df[AGE_ATTRIBUTE], bins, labels=names).cat.codes

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

class AgePredictor:

    def __init__(self, model, df, test_training_split, preprocessing_stages=[], hyperparameter_tuning=False) -> None:
        self.model = model
        self.df = df
        self.test_training_split = test_training_split
        self.preprocessing_stages = preprocessing_stages
        self.hyperparameter_tuning = hyperparameter_tuning

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_predict = None
        self.shap_explainer = None
        self.shap_values = None

        self._preprocess()
        self._train()


    def _preprocess(self):
        logger.info(f'Dataset length: {len(self.df)}')

        # Test & Training Split
        df_train, df_test = self.test_training_split(self.df)
        logger.info(f'Test dataset length: {len(df_test)}')
        logger.info(f'Training dataset length: {len(df_train)}')

        # The standard deviation in the test set gives us an indication of a baseline. We want to be able to be substantially below that value.
        logger.info(
            f"Standard deviation of test set: {df_test[AGE_ATTRIBUTE].std()}")

        # Preprocessing & Cleaning
        for func in self.preprocessing_stages:
            df_train = func(df_train)
            df_test = func(df_test)

        df_train = utils.shuffle(df_train, random_state=0)
        df_test = utils.shuffle(df_test, random_state=0)

        self.X_train = df_train.drop(columns=AUX_VARS+[AGE_ATTRIBUTE])
        self.y_train = df_train[[AGE_ATTRIBUTE]]

        self.X_test = df_test.drop(columns=AUX_VARS+[AGE_ATTRIBUTE])
        self.y_test = df_test[[AGE_ATTRIBUTE]]


    def _train(self):
        if self.hyperparameter_tuning:
            params = tune_hyperparameter(self.model, self.X_train, self.y_train)
            self.model.set_params(**params)

        # Training & Predicting
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model.fit(self.X_train, self.y_train, verbose=False, eval_set=eval_set)
        self.y_predict = self.model.predict(self.X_test)


    def evaluate_classification(self):
        print_classification_report(self.y_test, self.y_predict)
        plot_histogram(self.y_test, self.y_predict)
        plot_confusion_matrix(self.y_test, self.y_predict, classes=[0, 1, 2, 3, 4, 5])


    def evaluate_regression(self):
        print_model_error(self.y_test, self.y_predict)
        plot_histogram(self.y_test, self.y_predict, age_bins=[
                    1850, 1915, 1945, 1965, 1980, 2000, 2025])
        plot_grid(self.y_test, self.y_predict)


    def calculate_SHAP_values(self):
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
        noise_feature_importance = df_fi.query("feature=='feature_noise'").normalized_importance.values[0]
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
        df_neighborhood_fi = df_fi.loc[df_fi['feature'].str.contains('within_buffer')]
        return round(sum(df_neighborhood_fi.normalized_importance), 2)


def tune_hyperparameter(model, X, y):
    params = {
        'max_depth': [1, 3, 6, 10], # try ada trees
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'subsample': [0.7, 1.0],
    }

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    clf = model_selection.GridSearchCV(estimator=model,
                                       param_grid=params,
                                       scoring='neg_root_mean_squared_error',
                                       verbose=1)
    clf.fit(X, y)
    print("Best parameters: ", clf.best_params_)
    print("Lowest RMSE: ", np.sqrt(-clf.best_score_))

    tuning_results = pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
        clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    tuning_results.to_csv('hyperparameter-tuning-results.csv', sep='\t')
    print('All hyperparameter tuning results:\n', tuning_results)

    return clf.best_params_


def print_model_error(y_test, y_predict):
    print('MAE: {} y'.format(metrics.mean_absolute_error(y_test, y_predict)))
    print('RMSE: {} y'.format(np.sqrt(metrics.mean_squared_error(y_test, y_predict))))
    print('R2: {}'.format(metrics.r2_score(y_test, y_predict)))


def print_classification_report(y_test, y_predict):
    print(metrics.classification_report(y_test, y_predict))


def print_feature_importance(model):
    feature_accuracy_contribution = model.get_booster().get_score(importance_type="gain")
    feature_importance = pd.DataFrame({'importance': feature_accuracy_contribution})
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
    print(feature_importance.head(15))


def plot_histogram(y_test, y_predict, age_bins=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.distplot(
        y_predict,
        ax=ax,
        hist=True,
        kde=False,
        hist_kws=dict(edgecolor="k", linewidth=1),
        bins=age_bins,
        label='y_predict'
    )
    sns.distplot(
        y_test,
        ax=ax,
        hist=True,
        kde=False,
        hist_kws=dict(edgecolor="k", linewidth=1),
        bins=age_bins,
        label='y_test'
    )
    ax.legend()
    plt.title('age distributions')
    plt.show()


def plot_grid(y_test, y_predict):
    time_range = [1800, 2020]
    df_test = pd.DataFrame()
    df_test['y_predict'] = y_predict
    df_test['y_test'] = y_test

    fig = plt.figure()

    g = sns.JointGrid(
        df_test['y_predict'], df_test['y_test'], xlim=time_range, ylim=time_range)
    g.plot_joint(plt.hexbin, cmap="Purples", gridsize=30,
                 extent=time_range+time_range)

    g.ax_joint.plot(time_range, time_range, color='grey', linewidth=1.5)

    g.ax_marg_x.hist(df_test['y_predict'], color="tab:purple", alpha=.6)
    g.ax_marg_y.hist(df_test['y_test'], color="tab:purple",
                     alpha=.6, orientation="horizontal")

    g.set_axis_labels('Predicted ages in years', 'Target ages in years')

    plt.show()


def plot_log_loss(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])

    # plot log loss
    fig, ax = plt.subplots(figsize=(12, 12))
    x_axis = range(0, epochs)
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()

    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()


def plot_classification_error(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])

    # plot classification error
    fig, ax = plt.subplots(figsize=(12, 12))
    x_axis = range(0, epochs)
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax.legend()

    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()


def plot_confusion_matrix(y_test, y_predict, classes):
    cm = metrics.confusion_matrix(y_test, y_predict)
    plt.figure(figsize=[7, 6])
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_cm, annot=np.round(norm_cm, 2), fmt='g',
                xticklabels=classes, yticklabels=classes, cmap='bone')
