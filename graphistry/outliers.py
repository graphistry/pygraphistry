from typing import Any, List, Optional, Union

import matplotlib.font_manager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

import random

# import shap
# import dr_explainer as dre # pip install cluster-shapley


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

    print(f'min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}')
    print(f'min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}')
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

    print(f'min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}')
    print(f'min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}')

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
    print('-'*120)
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print(f'Fitting {clf_name}:')
        clf.fit(embedding)

    fig, ax = plot_outliers(embedding, classifiers, name, xy_extent, figsize, xy, xytext, border)
    print('-'*100)

    return classifiers, fig, ax

# #####################################################################################################################
#
#                        Shapely Explanability Modeling
#
# #####################################################################################################################

# def fit_shapely(X, y):
#     # fit the dataset
#     print(f'Fitting ClusterShapely')
#     clusterShapley = dre.ClusterShapley()
#     clusterShapley.fit(X, y)
#     return clusterShapley
#
# def explain_shapely(X, clusterShapley):
#     # compute explanations for data subset
#     shap_values = clusterShapley.transform(X)
#     return shap_values

# def fit_clust(X, y):
#     clust = shap.utils.hclust(X, y, linkage="single")
#     return clust

# def fit_shapely(X, model):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X)
#     return shap_values, explainer

# def fit_explanation(X, shap_values):
#     print(f'Explaining features: {X.columns}')
    
#     c_exp = shap.Explanation(shap_values, data=X, feature_names=X.columns)
#     return c_exp

# def fit_explainer(X, model):
#     # takes outside model and fits it on data
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X)
#     return explainer, shap_values
    
# def _handle_index(shap_values, klass):
#     if klass is None:
#         values = shap_values[klass]
#     else:
#         values = shap_values
#     return values

# def beeswarm(shap_values, klass):
#     # takes explanation fit model and plots as beeswarm
#     shap_values = _handle_index(shap_values, klass)
#     shap.plots.beeswarm(shap_values)

# def waterfall(shap_values, klass):
#     shap_values = _handle_index(shap_values, klass)
#     # visualize the first prediction's explanation
#     shap.plots.waterfall(shap_values)
    
# def force(shap_values, klass):
#     shap_values = _handle_index(shap_values, klass)
#     shap.plots.force(shap_values)

# def bar(shap_values, clust=None):
#     shap.plots.bar(shap_values, clustering=clust, clustering_cutoff=1)



# class HFMixin():
#     def __init__(self):
#         from transformers import pipeline
#         self.pipeline = pipeline


# class SentimentMixin(HFMixin):
#     def __init__(self):
#         self.model = self.pipeline('sentiment-analysis', return_all_scores=True)
        
        
# class ShapelyMixin():
    
#     def __init__(self, X: pd.DataFrame, y:pd.DataFrame, name: str):
#         self.X = X
#         self.y = y
#         self.name = name
#         self.feature_names = X.columns
#         self._model_is_fit = False

    
#     def fit(self, model: Any = None):
#         self.fit_shapely()
#         self.fit_explanation(self.X)
#         if model is not None:
#             self.shap_external_model_scores = self.fit_model_explainer(self.X, model)
            
#     # def fit_shapely(self):
#     #     print('Fitting a Supervised Shapely Cluster model')
#     #     self.cluster_shapely = fit_shapely(self.X, self.y)
#     #
#     # def transform(self, X):
#     #     shap_values = explain_shapely(X, self.cluster_shapely)
#     #     return shap_values

#     # def fit_explanation(self, X):
#     #     print('Fitting a Semi-Supervised Shapely Explanation model '
#     #           '-- it explains the data from the perspective of the shapely_scores of the Shapely Cluster Model')
#     #     #shap_values = self.transform(X)
#     #     self.explanation_model = fit_explanation(X, shap_values, self.feature_names)
        
#     def fit_model_explainer(self, X: pd.DataFrame, model: Any):
#         print('Fitting a Explainer model '
#               '-- it explains the data from the input Model')

#         self.explainer_model, shap_values = fit_explainer(X, model)
#         self._model_is_fit = True
#         return shap_values
    
#     def explain(self, X: pd.DataFrame):
#         if self._model_is_fit:
#             shap_values = self.explainer_model(X)
#             return shap_values
#         else:
#             raise NotImplementedError('No model has been fit, fit instance using `g.explain.fit`')

#     def plot(self, X: pd.DataFrame, kind='bees', klass=None):
#         shap_values = self.explain(X)
#         if kind == 'bees':
#             beeswarm(shap_values, klass)
#         if kind == 'waterfall':
#             waterfall(shap_values, klass)
#         if kind == 'force':
#             force(shap_values, klass)
        
    


# def graphistry_to_outliers(g, name='data', contamination=0.25, gamma=0.35,  xy=(8, 3),  xytext=(5, 1), figsize=(17, 10)):
#     embedding = g._xy
#     classifiers, fig, ax = detect_outliers(embedding, name='data', contamination=0.25, gamma=0.35,  xy=(8, 3),  xytext=(5, 1), figsize=(17, 10))
#     return classifiers, fig, ax


## example use case:

# coin_clfs = {}
# for coin in df.coin:
#     good_index = df.coin == coin
#     g_coin = g2.nodes(g2._nodes[good_index]) # not needed but nice as it will add only the nodes in question to the graphistry plot if you wanted to plot
#     g_coin._xy = g2._xy[good_index]  # set the right embeddings by coin
#      #plots the graph and labels it as the coin — need more work here to point to outlier, but that is candy
#     clfs = detect_outliers(g_coin, name=coin, ..)
#     coin_clfs[coin] = clfs