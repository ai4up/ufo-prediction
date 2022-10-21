import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from scipy.stats import kurtosis, skew, norm
from sklearn import metrics

import dataset
import preprocessing
import utils
import geometry


def plot_histogram(y_test, y_predict, bins=None, bin_labels=[], **kwargs):
    with SubplotManager(**kwargs) as ax:
        if bin_labels:
            ax.set_xticklabels([None] + bin_labels)

        sns.histplot(
            y_predict,
            ax=ax,
            kde=False,
            line_kws=dict(edgecolor="k", linewidth=1),
            palette='Set2',
            bins=bins,
            label='y_predict'
        )
        sns.histplot(
            y_test,
            ax=ax,
            kde=False,
            line_kws=dict(edgecolor="k", linewidth=1),
            palette='husl',
            bins=bins,
            label='y_test'
        )
        ax.legend()
        ax.set_title('age distributions')


def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for label, y in data.items():
            sns.distplot(
                y,
                ax=ax,
                hist=False,
                kde=True,
                norm_hist=True,
                label=label,
                hist_kws=dict(edgecolor="k", linewidth=1)
            )

    ax.legend()
    plt.title('age distributions')
    plt.show()


def plot_relative_grid(y_test, y_predict, bin_size=5, **kwargs):
    # Idea: periods with more buildings will not be brighter than periods with little buildings
    bins = utils.age_bins(y_predict, bin_size=bin_size)
    ticks = [1920, 1940, 1960, 1980, 2000]
    X, Y = np.meshgrid(bins, bins)
    age_test = y_test[dataset.AGE_ATTRIBUTE]
    age_predict = y_predict[dataset.AGE_ATTRIBUTE]
    H = np.histogram2d(age_test, age_predict, bins=bins)[0]
    # H_norm: each row describes relative share of all prediction age bands for buildings of a certain test band
    H_norm = (H.T / H.sum(axis=1)).T

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0,0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white', 'cadetblue'])
    colors = ['thistle', 'lightsteelblue', 'plum']

    ax_scatter.plot([0, 1], [0, 1], transform=ax_scatter.transAxes, color='cadetblue')
    ax_scatter.pcolormesh(X, Y, H_norm, cmap=cmap, rasterized=True)
    ax_scatter.set_yticks(ticks, labels=ticks)
    ax_scatter.set_xticks(ticks, labels=ticks)

    if 'country' in y_predict.columns and 'country' in y_test.columns:
        for i, (_, g) in enumerate(y_predict.groupby('country')):
            ax_hist_x.hist(g[dataset.AGE_ATTRIBUTE], bins=bins, color=colors[i], alpha=0.4)

        for i, (_, g) in enumerate(y_test.groupby('country')):
            ax_hist_y.hist(g[dataset.AGE_ATTRIBUTE], bins=bins, orientation='horizontal', color=colors[i], alpha=0.4)
    else:
        ax_hist_x.hist(y_predict[dataset.AGE_ATTRIBUTE], bins=bins, color=colors[0], alpha=0.4)
        ax_hist_y.hist(y_test[dataset.AGE_ATTRIBUTE], bins=bins, orientation='horizontal', color=colors[0], alpha=0.4)

    ax_hist_x.set_axis_off()
    ax_hist_y.set_axis_off()
    fig.subplots_adjust(hspace=0.05, wspace=0.02)

    ax_scatter.spines['top'].set_linewidth(0.5)
    ax_scatter.spines['right'].set_linewidth(0.5)
    ax_scatter.spines['left'].set_linewidth(0.5)
    ax_scatter.spines['bottom'].set_linewidth(0.5)

    ax_scatter.set_xlabel('Predicted construction year')
    ax_scatter.set_ylabel('True construction year')


def plot_grid(y_test, y_predict):
    min_age = int(y_predict[dataset.AGE_ATTRIBUTE].min())
    max_age = int(y_predict[dataset.AGE_ATTRIBUTE].max())
    time_range = [min_age, max_age]

    df_test = pd.DataFrame()
    df_test['y_predict'] = y_predict
    df_test['y_test'] = y_test

    fig = plt.figure()

    g = sns.JointGrid(
        df_test['y_predict'], df_test['y_test'], xlim=time_range, ylim=time_range)
    g.plot_joint(plt.hexbin, cmap="Purples", gridsize=30,
                 extent=time_range + time_range)

    g.ax_joint.plot(time_range, time_range, color='grey', linewidth=1.5)

    g.ax_marg_x.hist(df_test['y_predict'], color="tab:purple", alpha=.6)
    g.ax_marg_y.hist(df_test['y_test'], color="tab:purple",
                     alpha=.6, orientation="horizontal")

    g.set_axis_labels('Predicted ages in years', 'Target ages in years')

    plt.show()


def plot_log_loss(model, multiclass=False, **kwargs):
    metric = 'mlogloss' if multiclass else 'logloss'
    plot_eval_metric(model, metric=metric, title='Classification Log Loss', **kwargs)


def plot_classification_error(model, multiclass=False, **kwargs):
    metric = 'merror' if multiclass else 'error'
    plot_eval_metric(model, metric=metric, title='Classification Error', **kwargs)


def plot_regression_error(model, **kwargs):
    plot_eval_metric(model, metric='rmse', title='Regression Error', scale_y_axis=True, **kwargs)


def plot_eval_metric(model, metric, title, scale_y_axis=False, **kwargs):
    # retrieve performance metrics
    results = model.evals_result()
    error_train = results['validation_0'][metric]
    error_test = results['validation_1'][metric]
    epochs = len(error_train)

    # plot performance metrics
    with SubplotManager(**kwargs) as ax:
        x_axis = range(0, epochs)
        ax.plot(x_axis, error_train, label='Train')
        ax.plot(x_axis, error_test, label='Test')
        ax.legend()

        if scale_y_axis:
            y_min = min(error_train) / 1.25
            y_max = np.quantile(error_test, 0.90)
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel('epochs / trees')
        ax.set_ylabel(metric)
        ax.set_title('XGBoost ' + title)


def plot_models_classification_error(evals_results, **kwargs):
    with SubplotManager(**kwargs) as ax:
        for name, result in evals_results.items():
            x_axis = range(0, len(result))
            ax.plot(x_axis, result, label=name)

        ax.legend(loc='upper right')
        ax.set_xlabel('epochs / trees')
        ax.set_ylabel('error / merror')
        ax.set_title('XGBoost Classification Error Comparison')


def plot_confusion_matrix(y_test, y_predict, class_labels, **kwargs):
    with SubplotManager(**kwargs) as ax:
        cm = metrics.confusion_matrix(y_test, y_predict)
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=np.round(norm_cm, 2), fmt='g', ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels, cmap='bone', cbar=True)
        ax.set_xlabel('Predicted construction year', fontsize=16, labelpad=15)
        ax.set_ylabel('True construction year', fontsize=16, labelpad=15)


def plot_feature_over_time(df, overlap_coef=None, feature_selection=[], round=1, **kwargs):
    df = df.copy()
    feature_selection = feature_selection or df.columns.drop(['country', dataset.AGE_ATTRIBUTE])
    axis = utils.grid_subplot(len(feature_selection), n_cols=3)
    df = preprocessing.remove_outliers(df)
    if overlap_coef is not None:
        overlap_coef = preprocessing.remove_outliers(overlap_coef)
    df[dataset.AGE_ATTRIBUTE] = utils.custom_round(df[dataset.AGE_ATTRIBUTE], base=round)

    sns.set()
    with sns.plotting_context(font_scale=1):
        colors = ['plum', 'lightseagreen', 'darksalmon', 'steelblue']
        sns.set_theme(palette=colors, style='white')

        for i, (feature, type) in enumerate(df[feature_selection].dtypes.iteritems()):
            if type == 'object' or type == 'bool':
                feature_df = df[[dataset.AGE_ATTRIBUTE, feature]] \
                    .groupby([dataset.AGE_ATTRIBUTE, feature]) \
                    .size() \
                    .unstack(level=0) \
                    .fillna(0)
                sns.heatmap(feature_df, cmap="Blues", cbar_kws={"shrink": .7}, ax=axis[i])
            else:
                df[feature] = df[feature].clip(
                    lower=df[feature].quantile(0.005),
                    upper=df[feature].quantile(0.995))

                sns.lineplot(data=df, x=dataset.AGE_ATTRIBUTE, y=feature, hue='country', ci='sd' if overlap_coef else 99, legend=False, ax=axis[i])
                axis[i].tick_params(axis='both', which='both', length=0)

                if overlap_coef is not None:
                    ax_sec = axis[i].twinx()
                    sns.lineplot(data=overlap_coef, x=dataset.AGE_ATTRIBUTE, y=feature, ci=None, legend=False, color='steelblue', ax=ax_sec, linestyle='--')
                    ax_sec.set(ylim=(0, 1))
                    ax_sec.set_ylabel('OVL')
                    ax_sec.set_yticks(ticks=[0.25, 0.75], labels=['0.25', '0.75'])
                    ax_sec.tick_params(axis='both', which='both', length=0)
                    ax_sec.spines['top'].set_linewidth(0.5)
                    ax_sec.spines['right'].set_linewidth(0.5)
                    ax_sec.spines['left'].set_linewidth(0.5)
                    ax_sec.spines['bottom'].set_linewidth(0.5)
                    ax_sec.spines['top'].set_visible(False)
                    # ax_sec.spines['right'].set_visible(False)
                    ax_sec.spines['left'].set_visible(False)
                    ax_sec.spines['bottom'].set_visible(False)

            axis[i].spines['top'].set_visible(False)
            axis[i].spines['right'].set_visible(False)
            axis[i].spines['left'].set_linewidth(0.5)
            axis[i].spines['bottom'].set_linewidth(0.5)

        labels = ['France', 'Spain', 'Netherlands']
        lines = [Line2D([0], [0], color=c, linewidth=1.5, linestyle='-') for c in colors[:len(labels)]]
        if overlap_coef is not None:
            labels.append('OVL (Overlapping Coefficient)')
            lines.append(Line2D([0], [0], color='steelblue', linewidth=1.5, linestyle='--'))
        axis[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=len(labels), handles=lines, labels=labels)
        plt.show()


def plot_impact_of_spatial_distance_on_error(dfs, countries=[], **kwargs):
    with SubplotManager(**kwargs) as ax1:
        linestyles = ['-', '--', ':', '-.'] * len(countries)
        for i, df in enumerate(dfs):
            df = df[df['name'] != 'baseline']
            df['distance'] = df['name'].str.split('-').str[0].astype(int)
            df = df.groupby('distance').agg(['mean', 'std']).sort_index()

            df[[('R2', 'mean')]].plot(ax=ax1, xlabel='Distance between train and test set (km)', ylabel='$R^2$', xticks=list(df.index)[::2], ylim=(-0.25, 0.35), style=linestyles[i], legend=True)
            ax1.set_xticklabels(list(df.index + 25)[::2])
            ax1.fill_between(
                df.index,
                df[('R2', 'mean')] - df[('R2', 'std')] / 2,
                df[('R2', 'mean')] + df[('R2', 'std')] / 2,
                alpha=0.1,
                label='_nolegend_',
            )
        ax1.legend(countries)
        ax1.xaxis.labelpad = 15
        ax1.set_title('Generalizing over distance')


def plot_impact_of_additional_data_on_error(df, **kwargs):
    with SubplotManager(**kwargs) as ax1:
        df = df[df['name'] != 'baseline']
        df['n_cities'] = df['name'].str.split('-').str[0].astype(int)
        df.sort_values('n_cities', inplace=True)
        df.set_index('n_cities', inplace=True)

        ax1 = df[['RMSE', 'MAE']].plot(ax=ax1, xticks=df.index)#, xlim=(1, None), ylim=(7, 20))
        ax1.set_xscale('log', base=2)
        ax1.set_xticklabels(df.index)
        ax2 = ax1.twinx()
        df[['R2']].plot(ax=ax2, color='red', ylim=(0, 1), logx=True)
        for metric in ['RMSE', 'MAE']:
            ax1.fill_between(
                df.index,
                df[metric] - df[metric + '_std'],
                df[metric] + df[metric + '_std'],
                alpha=0.1,
            )
        ax2.fill_between(
            df.index,
            df['R2'] - df['R2_std'],
            df['R2'] + df['R2_std'],
            alpha=0.1,
        )


def plot_prediction_error_histogram(y_test, y_predict):
    error = y_predict - y_test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sns.distplot(error, fit=norm, kde=False)
    plt.title('Histogram of prediction errors')
    plt.show()
    # A positively skewed distribution has a tail to the right, while a negative one has a tail to the left.
    # If the distribution has positive kurtosis, it has fatter tails than the normal distribution; conversely,
    # the tails would be thinner in negative scenario.
    print('Excess kurtosis of normal distribution (should be 0): {}'.format(kurtosis(error, fisher=True)))
    print('Skewness of normal distribution (should be 0): {}'.format(skew(error)))


def plot_hyperparameter_tuning_results(results, hyperparameter, eval_metrics, include_fit_time=False, **kwargs):
    if isinstance(results, pd.DataFrame):
        results = {k: np.array(v) for k, v in results.to_dict(orient='list').items()}

    with SubplotManager(**kwargs) as ax:
        ax.set_title(f'Hyperparameter tuning for xgboost parameter {hyperparameter}', fontsize=24, pad=15)
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel('RMSE')
        ax.set_xticks([4,8,12,16])

        # Get regular np.array from MaskedArray
        X_axis = np.ma.getdata(results[f'param_{hyperparameter}']).astype(int)

        # Plot all eval metrics defined
        for metric in sorted(eval_metrics):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = np.abs(results['mean_%s_%s' % (sample, metric)])
                sample_score_std = np.abs(results['std_%s_%s' % (sample, metric)])
                ax.plot(
                    X_axis,
                    sample_score_mean,
                    style,
                    alpha=1 if sample == 'test' else 0.7,
                    label='%s (%s)' % ('RMSE', sample),
                )
                ax.set_ylim([0, 23])
                # ax.set_xticks(np.arange(min(X_axis)+1, max(X_axis), 2.0))
                ax.fill_between(
                    X_axis,
                    sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0,
                )

            best_index = np.nonzero(results['rank_test_%s' % metric] == 1)[0][0]
            best_score = np.abs(results['mean_test_%s' % metric][best_index])

            # Plot dotted vertical line at best eval metric score
            ax.plot(
                [X_axis[best_index]] * 2,
                [0, best_score],
                linestyle='-.',
                marker='x',
                markeredgewidth=3,
                ms=8,
            )

            # Annotate best eval metric score during training
            ax.annotate('%0.2f' % best_score, (X_axis[best_index], best_score + 0.5))

            if include_fit_time:
                ax2 = ax.twinx()
                ax2.plot(
                    X_axis,
                    results['mean_fit_time'] / 60,
                    alpha=1,
                    color='gray',
                    label='Mean fit time',
                )
                ax2.set_ylabel('Time (min)')
                ax2.legend(loc=1)

        ax.legend(loc=2)
        ax.grid(False)


def plot_age_on_map(age_df, geometry_df):
    age_df = age_df.dropna(subset=[dataset.AGE_ATTRIBUTE])
    age_df = preprocessing.remove_outliers(age_df)
    age_df = preprocessing.round_age(age_df)
    age_df[dataset.AGE_ATTRIBUTE] = age_df[dataset.AGE_ATTRIBUTE].astype(str)
    plot_attribute_on_map(age_df, geometry_df, dataset.AGE_ATTRIBUTE)


def plot_prediction_error_on_map(prediction_error_df, geometry_df):
    prediction_error_df['error'] = utils.custom_round(prediction_error_df['error'], base=20)
    prediction_error_df['error'] = prediction_error_df['error'].astype(str)
    plot_attribute_on_map(prediction_error_df, geometry_df, 'error')


def plot_prediction_error_on_map_by_city(predictor, city_boundaries, gadm_level=2, dissolve_level=None, **kwargs):
    y = predictor.individual_prediction_error()
    y = pd.concat([y, predictor.aux_vars_test], axis=1, join="inner")
    y['error'] = y['error'].abs()

    error_by_city = y.groupby('city')['error'].mean()
    error_by_city = city_boundaries.merge(error_by_city, left_on=f'NAME_{gadm_level}', right_on='city', how='left')


    with SubplotManager(**kwargs) as ax:
        if dissolve_level:
            error_by_city = error_by_city[['error', f'NAME_{dissolve_level}', 'geometry']].dissolve(by=f'NAME_{dissolve_level}', aggfunc='mean').reset_index()
            # error_by_city.apply(lambda x: ax.annotate(text=x[f'NAME_{dissolve_level}'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        else:
            min = error_by_city.iloc[error_by_city['error'].idxmin()]
            max = error_by_city.iloc[error_by_city['error'].idxmax()]
            min_text = f"min error {min['error']:.2f}\n({min[f'NAME_{gadm_level}']})"
            max_text = f"max error {max['error']:.2f}\n({max[f'NAME_{gadm_level}']})"
            ax.annotate(text=min_text, xy=min.geometry.centroid.coords[0], ha='center')
            ax.annotate(text=max_text, xy=max.geometry.centroid.coords[0], ha='center')

        error_by_city.boundary.plot(ax=ax, color='grey', alpha=0.2)
        error_by_city.plot(ax=ax, column='error', cmap='Reds', legend=True)


def plot_attribute_on_map(
        attribute_df,
        geometry_df,
        attribute_name,
        boundaries_df=None,
        crs=2154,
        vmin=None,
        vmax=None):

    df = pd.concat([geometry_df[['id', 'geometry']].set_index('id'),
                   attribute_df.set_index('id')], axis=1, join="inner").reset_index()

    if not isinstance(df, gpd.GeoDataFrame):
        df = geometry.to_gdf(df, crs=crs)

    _, ax = plt.subplots(1, 1)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if vmin and vmax else None
    df.plot(column=attribute_name, ax=ax, legend=True, norm=norm)

    if boundaries_df is not None:
        boundaries_df.to_crs(crs).exterior.plot(ax=ax)
    plt.show()


def slope_chart(dfs, labels=None, feature_selection=None, **kwargs):
    xticks = list(range(0, len(dfs)))
    df = pd.concat(dfs, axis=0, keys=xticks).reset_index(level=[0])
    df = df[df['feature'].isin(feature_selection)]

    with SubplotManager(**kwargs) as ax:
        for feature, group in df.groupby('feature'):
            fts_importance_left = group['normalized_importance'].values[0]
            fts_importance_right = group['normalized_importance'].values[-1]
            ax.plot(group['level_0'], group['normalized_importance'], marker='o', markersize=5)
            if fts_importance_left > 0.01:
                ax.text(-0.05, fts_importance_left, feature, ha='right', fontsize='xx-small')
            if fts_importance_right > 0.01:
                ax.text(len(dfs) - 0.95, fts_importance_right, feature, ha='left', fontsize='xx-small')

        if labels:
            ax.set_xticks(xticks, labels, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlim(-1.5, len(dfs) + 0.5)
        ax.set_ylim(bottom=0.01, top=0.12)
        ax.set_yscale('log')
        ax.set_yticks(ticks=[0.01, 0.02, 0.04, 0.08], labels=['1%', '2%', '4%', '8%'])
        # ax.set_yticks(ticks=[0.01, 0.02, 0.05, 0.1], labels=['1%', '2%', '5%', '10%'])
        ax.set_title('Feature Importance Differences')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


class SubplotManager:
    def __init__(self, **kwargs):
        ax = kwargs.get('ax')
        self.mainplot = not bool(ax)
        self.ax = ax or plt.subplots(**{'figsize': (12, 12), **kwargs})[1]

        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14
        # SMALL_SIZE = 18
        # MEDIUM_SIZE = 20
        # BIGGER_SIZE = 22

        plt.rc('font', size=BIGGER_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)
        plt.rc('ytick', labelsize=MEDIUM_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)


    def __enter__(self):
        return self.ax


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.mainplot:
            plt.show()
