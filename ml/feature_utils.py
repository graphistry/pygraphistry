from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

from dirty_cat import (
    SuperVectorizer,
    SimilarityEncoder,
    TargetEncoder,
    MinHashEncoder,
    GapEncoder,
)
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

import ml.constants as config
from ml.utils import setup_logger

logger = setup_logger(__name__)

encoders_dirty = {
    "similarity": SimilarityEncoder(similarity="ngram"),
    "target": TargetEncoder(handle_unknown="ignore"),
    "minhash": MinHashEncoder(n_components=config.N_HASHERS_DEFAULT),
    "gap": GapEncoder(n_components=config.N_TOPICS_DEFAULT),
    "super": SuperVectorizer(auto_cast=True),
}


def calculate_column_similarity(y, n_points=10):
    # y is a pandas series of labels for a given dataset
    sorted_values = y.sort_values().unique()
    similarity_encoder = SimilarityEncoder(similarity="ngram")
    transformed_values = similarity_encoder.fit_transform(sorted_values.reshape(-1, 1))

    plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=n_points)
    return transformed_values, similarity_encoder, sorted_values


def plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=15):
    # TODO try this with UMAP
    mds = MDS(dissimilarity="precomputed", n_init=10, random_state=42)
    two_dim_data = mds.fit_transform(1 - transformed_values)  # transformed values lie

    random_points = np.random.choice(
        len(similarity_encoder.categories_[0]), n_points, replace=False
    )

    nn = NearestNeighbors(n_neighbors=2).fit(transformed_values)
    _, indices_ = nn.kneighbors(transformed_values[random_points])
    indices = np.unique(indices_.squeeze())

    f, ax = plt.subplots()
    ax.scatter(x=two_dim_data[indices, 0], y=two_dim_data[indices, 1])
    # adding the legend
    for x in indices:
        ax.text(
            x=two_dim_data[x, 0], y=two_dim_data[x, 1], s=sorted_values[x], fontsize=8
        )
    ax.set_title(
        "multi-dimensional-scaling representation using a 3gram similarity matrix"
    )

    f2, ax2 = plt.subplots(figsize=(7, 7))
    cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
    ax2.set_yticks(np.arange(len(indices)))
    ax2.set_xticks(np.arange(len(indices)))
    ax2.set_yticklabels(sorted_values[indices], rotation="30")
    ax2.set_xticklabels(sorted_values[indices], rotation="60", ha="right")
    ax2.xaxis.tick_bottom()
    ax2.set_title("Similarities across categories")
    f2.colorbar(cax2)
    f2.tight_layout()


# #################################################################################
#
#      Pandas Helpers
#
# ###############################################################################
def check_target_not_in_features(df, y, remove=True):
    """

    :param df: model dataframe
    :param y: target dataframe
    :param remove: whether to remove columns from df, default True
    :return: dataframes of model and target
    """
    if y is None:
        return df, y
    remove_cols = []
    if hasattr(y, "columns"):
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    else:
        logger.warning(f"Target is not of type(DataFrame) and has no columns")
    if remove:
        logger.info(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols)
        return tf, y
    return df, y


def featurize_to_torch(df, y, vectorizer, remove=True):
    """
        Vectorize pandas DataFrames of model and target features into Torch compatible tensors
    :param df: DataFrame of model features
    :param y: DataFrame of target features
    :param vectorizer: must impliment (X, y) encoding and output (X_enc, y_enc, sup_vec, sup_label)
    :param remove: whether to remove target from df
    :return:
        data: dict of {feature, target} encodings
        encoders: dict of {feature, target} encoding objects (which store vocabularity and column information)
    """
    df, y = check_target_not_in_features(df, y, remove=remove)
    X_enc, y_enc, sup_vec, sup_label = vectorizer(df, y)
    if y_enc is not None:
        data = {
            config.FEATURE: torch.tensor(X_enc.values),
            config.TARGET: torch.tensor(y_enc.values),
        }
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    encoders = {config.FEATURE: sup_vec, config.TARGET: sup_label}
    return data, encoders



# #################################################################################
#
#      Featurization Functions
#
# ###############################################################################

def process_dirty_dataframes(Xdf, y):
    t = time()
    sup_vec = SuperVectorizer(auto_cast=True)
    sup_label = None
    y_enc = None
    logger.info("Sit tight, this might take a few minutes --")
    X_enc = sup_vec.fit_transform(Xdf)
    logger.info(f"Fitting SuperVectorizer on DATA took {(time()-t)/60:.2f} minutes")
    all_transformers = sup_vec.transformers
    features_transformed = sup_vec.get_feature_names()
    logger.info(f"Shape of data {X_enc.shape}")
    logger.info(f"Transformers: {all_transformers}\n")
    logger.info(f"Transformed Columns: {features_transformed[:20]}...\n")
    X_enc = pd.DataFrame(X_enc, columns=features_transformed)
    X_enc = X_enc.fillna(0)
    if y is not None:
        logger.info(f"Fitting Targets --")
        sup_label = SuperVectorizer(auto_cast=True)
        y_enc = sup_label.fit_transform(y)
        if type(y_enc) == scipy.sparse.csr.csr_matrix:
            y_enc = y_enc.toarray()
        labels_transformed = sup_label.get_feature_names()
        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)
        logger.info(f"Shape of target {y_enc.shape}")
        logger.info(f"Target Transformers used: {sup_label.transformers}")
    return X_enc, y_enc, sup_vec, sup_label


# #################################################################################
#
#      Pandas Helpers
#
# ###############################################################################