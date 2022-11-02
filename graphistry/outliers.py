from typing import Any, List, Optional, Union

import matplotlib.font_manager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

# import shap
# import dr_explainer as dre # pip install cluster-shapley

import random

from .util import setup_logger
from .constants import VERBOSE, TRACE


logger = setup_logger(__name__, VERBOSE, TRACE)



# #####################################################################################################################
#
#                        Outlier Modeling
#
# #####################################################################################################################

def get_outliers(embedding, df, outlier_fraction =0.2):
    outlier_clf = neighbors.LocalOutlierFactor(contamination=outlier_fraction)
    outlier_scores = outlier_clf.fit_predict(embedding)
    outlying = df[outlier_scores == -1]
    
    return outlying, outlier_scores, outlier_clf

def get_embedding_extent(embedding):
    mminx, mmaxx = embedding.T[0].min(), embedding.T[0].max()
    mminy, mmaxy = embedding.T[1].min(), embedding.T[1].max()

    deltax = mmaxx-mminx
    deltay = mmaxy-mminy

    logger.info(f'min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}')
    logger.info(f'min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}')
    return mminx, mmaxx, mminy, mmaxy


def plot_outliers(embedding, classifiers, name, xy_extent=((0, 10), (0, 10)), figsize=(7, 4), xy=(1,1), xytext=(2,2), border=1):
    colors = ["m", "r", "b", "g"]
    legend1 = {}
    if xy_extent is None:
        mminx, mmaxx = embedding.T[0].min(), embedding.T[0].max()
        mminy, mmaxy = embedding.T[1].min(), embedding.T[1].max()
    else:
        mminx, mmaxx = xy_extent[0]
        mminy, mmaxy = xy_extent[1]

    deltax = mmaxx-mminx
    deltay = mmaxy-mminy

    logger.info(f'min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}')
    logger.info(f'min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}')

    # Learn a frontier for outlier detection with several classifiers
    xx, yy = np.meshgrid(np.linspace(mminx-border, mmaxx+border, 500), np.linspace(mminy-border, mmaxy+border, 500))
    fig = plt.figure(1, figsize=figsize)
    ax = plt.subplot()
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        plt.figure(1, figsize=figsize)
        Z1 = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z1 = Z1.reshape(xx.shape)
        legend1[clf_name] = plt.contour(
            xx, yy, Z1, levels=[0], linewidths=2, colors=colors[i]
        )
    
    legend1_values_list = list(legend1.values())
    legend1_keys_list = list(legend1.keys())

    # Plot the results (= shape of the data points cloud)
    plt.figure(1, figsize=figsize)  # two clusters
    plt.title(f"Outlier detection: {name}")
    plt.scatter(embedding[:, 0], embedding[:, 1], color="black", s=22, alpha=0.75)
    bbox_args = dict(boxstyle="round", fc="0.8")
    arrow_args = dict(arrowstyle="->")
    plt.annotate(
        "outlying points",
        xy=xy,
        xycoords="data",
        textcoords="data",
        xytext=xytext,
        bbox=bbox_args,
        arrowprops=arrow_args,
    )
    plt.xlim((xx.min(), xx.max()))
    plt.ylim((yy.min(), yy.max()))
    plt.legend(
        (
            legend1_values_list[0].collections[0],
            legend1_values_list[1].collections[0],
            legend1_values_list[2].collections[0],
            legend1_values_list[3].collections[0],
        ),
        (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2], legend1_keys_list[3]),
        loc="upper center",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )
    plt.ylabel("y UMAP")
    plt.xlabel("x UMAP")
    
    plt.show()
    
    return fig, ax


def detect_outliers(embedding, name='data', contamination=0.25, gamma=0.35, xy_extent=None, xy=(8, 3),  xytext=(5, 1), figsize=(17, 10), border=1):
    # xy_extent = ((0,10), (-2, 10))
    # example from https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
    # assumes umap has been run on g and finds decision boundary in projection.
    # trains a set of outlier and unsupervised models on the embedding.
    # Define "classifiers" to be used
    classifiers = {
        "Outlier": neighbors.LocalOutlierFactor(contamination=contamination, novelty=True),
        "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=contamination),
        "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
            contamination=contamination
        ),
        "OCSVM": OneClassSVM(nu=contamination, gamma=gamma),
    }
    logger.info('-'*120)
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        logger.info(f'Fitting {clf_name}:')
        clf.fit(embedding)

    fig, ax = plot_outliers(embedding, classifiers, name, xy_extent, figsize, xy, xytext, border)
    logger.info('-'*100)

    return classifiers, fig, ax



