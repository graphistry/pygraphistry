from typing import Union, Tuple

import pandas as pd
import logging

try: 
    import matplotlib.font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import neighbors
    from sklearn.covariance import EllipticEnvelope
    from sklearn.svm import OneClassSVM
except:
    plt = None  # type: ignore
    np = None  # type: ignore
    neighbors = None  # type: ignore
    EllipticEnvelope = None  # type: ignore
    OneClassSVM = None  # type: ignore


logger = logging.getLogger(__name__)


# #####################################################################################################################
#
#                        Outlier Modeling
#
# #####################################################################################################################


def get_outliers(embedding: Union[np.ndarray, pd.DataFrame], df: pd.DataFrame, outlier_fraction: float = 0.2):
    """get outliers using sklearn's LocalOutlierFactor

    Args:
        embedding (Union[np.ndarray, pd.DataFrame]): umap embedding
        df (pd.DataFrame): dataframe for enrichment
        outlier_fraction (float, optional): fraction of outliers parameter. Defaults to 0.2.

    Returns:
        Tuple: tuple of outlying dataframe, outlier scores, and outlier classifier
    """
    outlier_clf = neighbors.LocalOutlierFactor(contamination=outlier_fraction)
    outlier_scores = outlier_clf.fit_predict(embedding)
    df['outlier'] = outlier_scores
    outlying = df[outlier_scores == -1]

    return outlying, outlier_scores, outlier_clf


def get_embedding_extent(embedding: Union[np.ndarray, pd.DataFrame]):
    """helper function to get the extent of the embedding

    Args:
        embedding (Union[np.ndarray, pd.DataFrame]): umap embedding

    Returns:
        Tuple: tuple of min and max x and y
    """
    mminx, mmaxx = embedding.T[0].min(), embedding.T[0].max()
    mminy, mmaxy = embedding.T[1].min(), embedding.T[1].max()

    deltax = mmaxx - mminx
    deltay = mmaxy - mminy

    logger.info(f"min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}")
    logger.info(f"min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}")
    return mminx, mmaxx, mminy, mmaxy


def plot_outliers(
    embedding: Union[np.ndarray, pd.DataFrame],
    classifiers: dict,
    name: str,
    xy_extent = ((0, 10), (0, 10)),
    figsize = (7, 4),
    xy = (1, 1),
    xytext = (2, 2),
    border = 1,
):
    """
    Plot the decision function for several outliers detection algorithms.
        Although embedding can be any size, it is expected to be 2D umap coordinates
        
    Args:
        embedding (np.ndarray): embedding of the data
        classifiers (dict): dict of classifiers to use, with keys as names and values as sklearn classifiers    
        name (str): name of the dataset
        xy_extent (tuple): extent of the plot
        figsize (tuple): size of the plot
        xy (tuple): xy position of the legend
        xytext (tuple): xy position of the legend text
        border (int): border around the plot
    returns:
        fig (matplotlib.figure.Figure): figure of the plot and axes
    """
    colors = ["m", "r", "b", "g"]
    legend1 = {}
    if xy_extent is None:
        mminx, mmaxx = embedding.T[0].min(), embedding.T[0].max()
        mminy, mmaxy = embedding.T[1].min(), embedding.T[1].max()
    else:
        mminx, mmaxx = xy_extent[0]
        mminy, mmaxy = xy_extent[1]

    deltax = mmaxx - mminx
    deltay = mmaxy - mminy

    logger.info(f"min extent x: {mminx}, max extent x: {mmaxx}, delta: {deltax}")
    logger.info(f"min extent y: {mminy}, max extent y: {mmaxy}, delta: {deltay}")

    # Learn a frontier for outlier detection with several classifiers
    xx, yy = np.meshgrid(
        np.linspace(mminx - border, mmaxx + border, 500),
        np.linspace(mminy - border, mmaxy + border, 500),
    )
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
    plt.scatter(embedding[:, 0], embedding[:, 1], color="black", s=22, alpha=0.75)  # type: ignore
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
        (
            legend1_keys_list[0],
            legend1_keys_list[1],
            legend1_keys_list[2],
            legend1_keys_list[3],
        ),
        loc="upper center",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )
    plt.ylabel("y UMAP")
    plt.xlabel("x UMAP")

    plt.show()

    return fig, ax


def detect_outliers(
    embedding: Union[np.ndarray, pd.DataFrame],
    name: str = "data",
    contamination: float = 0.25,
    gamma: float = 0.35,
    xy_extent: Union[Tuple, None] = None,
    xy = (8, 3),
    xytext = (5, 1),
    figsize = (17, 10),
    border = 1,
): 
    """Train and plot outlier detection algorithms on embedding.
    
    example adapted from https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html

    Args:
        embedding (Union[np.ndarray, pd.DataFrame]): embedding of the data
        name (str, optional): optional name. Defaults to "data".
        contamination (float, optional): contamination parameter. Defaults to 0.25.
        gamma (float, optional): gamma parameter for OneClassSVM. Defaults to 0.35.
        xy_extent (Tuple, optional): if given, sets extent one wishes to plot (ie, zoom in). Defaults to None.
        xy (tuple, optional): annotation pointer (to center of outlier or any other point). Defaults to (8, 3).
        xytext (tuple, optional): annotation text coordinates. Defaults to (5, 1).
        figsize (tuple, optional): size of figure. Defaults to (17, 10).
        border (int, optional): border around extent. Defaults to 1.

    Returns:
        Tuple: Tuple of fit classifier dicts, fig, plot axes
    """
    # xy_extent = ((0,10), (-2, 10))
    # assumes umap has been run on g and finds decision boundary in projection.
    # trains a set of outlier and unsupervised models on the embedding.
    # Define "classifiers" to be used
    classifiers = {
        "Outlier": neighbors.LocalOutlierFactor(
            contamination=contamination, novelty=True
        ),
        "Empirical Covariance": EllipticEnvelope(
            support_fraction=1.0, contamination=contamination
        ),
        "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(
            contamination=contamination
        ),
        "OCSVM": OneClassSVM(nu=contamination, gamma=gamma),
    }
    logger.info("-" * 120)
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        logger.info(f"Fitting {clf_name}:")
        clf.fit(embedding)

    fig, ax = plot_outliers(
        embedding, classifiers, name, xy_extent, figsize, xy, xytext, border
    )
    logger.info("-" * 100)

    return classifiers, fig, ax
