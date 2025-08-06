from typing import Any, Tuple, Callable
import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from sklearn.datasets import fetch_california_housing
import json
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = fetch_california_housing(as_frame=True)

# MedInc        median income in block group
# HouseAge      median house age in block group
# AveRooms      average number of rooms per household
# AveBedrms     average number of bedrooms per household
# Population    block group population
# AveOccup      average number of household members
# Latitude      block group latitude
# Longitude     block group longitude
# MedHouseVal(target)  median house value for California districts,
# expressed in hundreds of thousands of dollars ($100,000)

# This dataset was derived from the 1990 U.S. census, using one row per census
# block group. A block group is the smallest geographical unit for which the U.S.
# Census Bureau publishes sample data (a block group typically has a population
# of 600 to 3,000 people)

np.random.seed(42)

df_full = data.frame
df_full = df_full.sample(frac=1).reset_index(drop=True)

ETLReturnType = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    pd.Index, np.ndarray, np.ndarray
]

def load_model(path: str = "model.json") -> dict | None:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "model.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            model = json.load(f)
        print("Model loaded from file.",path)
        return model
    else:
        print("Model file not found, training will be required.",path)
        return None

def features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["RoomDiff"] = df["AveRooms"] - df["AveBedrms"]
    df["EstimatedHouseholds"] = df["Population"] / df["AveOccup"]
    df["OccupancyScore"] = df["RoomDiff"] / df["AveOccup"]
    df["OccupancyBoost"] = df["RoomDiff"] * df["AveOccup"]
    df.drop(["AveRooms", "AveBedrms", "AveOccup", "Population", "OccupancyBoost"], axis=1, inplace=True)
    print(df.head())

    return df


def etl(df: pd.DataFrame):
    def rank_values(col: np.array) -> ndarray[tuple[int, ...], dtype[Any]]:
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

    feature_names = df.columns.drop("MedHouseVal")

    X = np.array(df.drop("MedHouseVal", axis=1).values)
    Y = df["MedHouseVal"]

    number_samples = df.shape[0]
    split_index = int(number_samples * 0.8)

    # raw data operations
    X_ranks = compute_ranks(X)
    Y_ranks = compute_ranks(Y)

    X_rank_z = (X_ranks - X_ranks.mean(axis=0)) / X_ranks.std(axis=0)
    Y_rank_z = (Y_ranks - Y_ranks.mean()) / Y_ranks.std()
    #standarization
    # pure pearson
    X_zscore = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_zscore = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    # spearmann
    X_spearmann = (X_rank_z.T @ X_rank_z) / (number_samples - 1)
    Y_spearmann = (X_rank_z.T @ Y_rank_z) / (number_samples - 1)

    X_train = X_zscore[:split_index, :]
    X_test = X_zscore[split_index:, :]

    Y_train = Y_zscore[:split_index]
    Y_test = Y_zscore[split_index:]

    #test of pearson correlation
    features_pearson = (X_zscore.T @ X_zscore) / (number_samples - 1)
    df_fpearson = pd.DataFrame(features_pearson, index=feature_names, columns=feature_names)
    xy_pearson = (X_zscore.T @ Y_zscore) / (number_samples - 1)
    df_xy_pearson = pd.DataFrame(xy_pearson, index=feature_names, columns=["Target"])
    return X, Y, X_zscore, Y_zscore, X_train, X_test, Y_train, Y_test, xy_pearson, features_pearson, df_fpearson, df_xy_pearson, X_ranks, Y_ranks, X_spearmann, Y_spearmann, feature_names, X_rank_z, Y_rank_z


def linear_regression(etl: Callable[[pd.DataFrame], ETLReturnType]) -> tuple[pd.DataFrame,pd.DataFrame, dict]:
    X, Y, X_zscore, Y_zscore, X_train, X_test, Y_train, Y_test, xy_pearson, features_pearson, df_fpearson, df_xy_pearson, X_ranks, Y_ranks, X_spearmann, Y_spearmann, feature_names, X_rank_z, Y_rank_z = etl
    bias = 0
    n = X_rank_z.shape[0]
    best_mse = float('inf')
    weights = np.array([ 0.64422243,  0.11740738, -0.80621795, -0.74821601, -0.42069151,  0.06012402,
  0.49112416])
    model = {
        "weights": weights.tolist(),
        "bias": bias,
        "mean_target": Y.mean(),
        "std_target": Y.std(),
        "mean_features": X.mean(axis=0).tolist(),
        "std_features": X.std(axis=0).tolist(),
        "targets": Y.tolist(),
    }
    patience = 20
    no_improvement = 0
    for epoch in range(2000):
        learning_rate = 0.1
        y_pred = X_zscore @ weights + bias
        error = Y_zscore - y_pred
        mse = (error.T @ error) / n
        dL_dw = (2 * X_zscore.T @ (y_pred - Y_zscore)) / n
        dL_db = (2 * np.sum(Y_zscore - y_pred)) / n
        weights -= learning_rate * dL_dw
        bias -= learning_rate * dL_db
        real_pred = y_pred * model["std_target"] + model["mean_target"]
        print(f"Epoch: {epoch}")
        print(f"MSE: {mse}")
        print(f"Prediction (z): {y_pred}")
        print(f"Prediction (real): {real_pred}")
        print(f"Real targets: {Y}")
        print(f"Weights: {weights}\n")

        if mse < best_mse:
            best_mse = mse
            model["weights"] = weights.tolist()
            model["bias"] = float(bias)
            model["real_pred"] = (y_pred * model["std_target"] + model["mean_target"]).tolist()
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping due to no improvement.")
                print(f"Best MSE: {best_mse}")
                print(f"Weights: {weights}")
                break
        indexes = pd.Series(Y)
    df_result = pd.DataFrame({
        "Prediction": model["real_pred"],
        "Target": Y
    },index=Y.indexes)
    df_result["Diff"] = df_result["Prediction"] - df_result["Target"]
    df_result["AbsError"] = abs(df_result["Prediction"] - df_result["Target"])
    df_result_sorted = df_result.sort_values("AbsError",ascending=False)



    model_path = os.path.join(os.path.dirname(__file__),"model.json")
    with open(model_path, "w") as f:
        json.dump(model, f)

    print(df_result.to_string(index=False))
    return df_result,df_result_sorted, model


def show_transformed(etl) -> None:
    X, Y, X_zscore, Y_zscore, X_train, X_test, Y_train, Y_test, xy_pearson, features_pearson, df_fpearson, df_xy_pearson, X_ranks, Y_ranks, X_spearmann, Y_spearmann, feature_names = etl
    print("Features values")
    print(X)
    print("Target values:")
    print(Y)
    print("Ranks X:")
    print(pd.DataFrame(X_ranks, columns=feature_names))
    print("Ranks Y:")
    print(pd.DataFrame(Y_ranks, columns=["Target"]))
    print(pd.DataFrame(X, columns=feature_names).describe())
    print("\nZ-scores (first 7 rows):\n")
    print(pd.DataFrame(X_zscore, columns=feature_names))
    print("\nCorrelation matrix inbetween features (Pearson):\n")
    print(df_fpearson)
    print("\nCorrelation matrix inbetween features (Spearmann):\n")
    print(pd.DataFrame(X_spearmann, index=feature_names, columns=feature_names))
    print("\nSpearman correlation between features and target:\n")
    print(pd.DataFrame(Y_spearmann, index=feature_names, columns=["Target"]))
    print("\nPearson test for each feature with target:\n")
    print(pd.DataFrame(xy_pearson, index=feature_names, columns=["Target"]))
    print("\nTargets:\n")
    print(pd.DataFrame(Y_zscore[:7], columns=["Targets 0-6"]))

model = load_model()

if model is None:
    df_result, df_result_sorted, model = linear_regression(etl(features_engineering(df_full.copy())))
else:
    print("Loaded model — training skipped.")
