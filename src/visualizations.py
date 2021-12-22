import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


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
