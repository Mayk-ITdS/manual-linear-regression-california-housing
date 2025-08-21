from typing import Any
from pandas import DataFrame
from ETL.feature_engineer import FeatureEngineer
import numpy as np
import pandas as pd

def run_transform(df: DataFrame, bounds) -> tuple[FeatureEngineer, DataFrame]:
    ft_eng = FeatureEngineer(df, bounds)
    df_engineered = ft_eng.categorise_coordinates()
    df_engineered,_ = ft_eng.filter_samples(df_engineered)

    return ft_eng, df_engineered

def rank_values(col: np.array) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
    n = col.size
    indexed = list(enumerate(col.tolist()))
    indexed = sorted(indexed, key=lambda x: x[1])
    ranks = [0] * n
    i = 0
    while i < n:
        start = i
        end = i
        while end + 1 < n and indexed[end][1] == indexed[end + 1][1]:
            end += 1
            average_rank = (start + end + 2) / 2
            for j in range(start, end + 1):
                ranks[indexed[j][0]] = average_rank
            i = end + 1

    return np.array(ranks)

def compute_ranks(matrix):
    if matrix.ndim == 1:
        return rank_values(matrix)
    result = np.zeros_like(matrix, dtype=float)
    for col in range(matrix.shape[1]):
        result[:, col] = rank_values(matrix[:, col])
    return result

def split_data(df_eng,proportion=0.8):

    X = df_eng.drop("MedHouseVal", axis=1)
    Y = df_eng["MedHouseVal"]
    number_samples = X.shape[0]
    split_index = int(number_samples * proportion)
    return

def spearman_ranks(df_feats,df_targets):
    X_ranks = compute_ranks(np.array(df_feats))
    Y_ranks = compute_ranks(df_targets)
    return X_ranks, Y_ranks

def spearman_corr(x_ranks,y_ranks):
    number_samples = x_ranks.shape[0]

    # normalizing to z_score with spearman ranks

    X_rank_z = (x_ranks - x_ranks.mean(axis=0)) / x_ranks.std(axis=0)
    Y_rank_z = (y_ranks - y_ranks.mean()) / y_ranks.std()

    # correlation matrices by spearman

    X_spearmann = (X_rank_z.T @ X_rank_z) / (number_samples - 1)
    Y_spearmann = (X_rank_z.T @ Y_rank_z) / (number_samples - 1)
    return X_spearmann,Y_spearmann

def pearson_zscore(x,y):

    # standarization to z_score with pearson only

    X_zscore = (np.array(x) - np.array(x).mean(axis=0)) / np.array(x).std(axis=0)
    Y_zscore = (y - y.mean(axis=0)) / y.std(axis=0)

    return X_zscore, Y_zscore

def pearson_corr(X_zscore, Y_zscore, feature_names) -> tuple[DataFrame, DataFrame]:
    # test of pearson correlation
    number_samples = X_zscore.shape[0]
    features_pearson = (X_zscore.T @ X_zscore) / (number_samples - 1)
    d_features_pearson = pd.DataFrame(features_pearson, index=feature_names, columns=feature_names)
    xy_pearson = (X_zscore.T @ Y_zscore) / (number_samples - 1)
    df_xy_pearson = pd.DataFrame(xy_pearson, index=feature_names, columns=["Target"])

    return d_features_pearson, df_xy_pearson

def one_hot_encoding(df_,drop_one=True):
    for cat in df_["Category"].unique():
        df_[f"cat_{cat}"] = (df_["Category"] == cat).apply(int)
    if drop_one:
        df_ = df_.drop("cat_Mendocino", axis=1)

    return df_