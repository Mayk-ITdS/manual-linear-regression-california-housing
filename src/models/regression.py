import time
from pathlib import Path
from typing import Any, Tuple, Callable
import numpy as np
import pandas as pd
import json
import os

MODELS_DIR = Path("./models")

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

class LinearRegression:
    def __init__(self,patience=20,learning_rate=0.01,bias = 0,alpha=0.5):
        self.df_= pd.read_csv("df_zscore_final.csv")
        self.patience = patience
        self.learning_rate = learning_rate
        self.bias = bias
        self.ridge = alpha
        self.I = np.eye(len(self.df_))
    def train(self):
        start_time = time.perf_counter()
        X = self.df_.drop(columns=["MedHouseVal", "Category", "MedHouseValQuartiles", "Unnamed: 0"])
        Y = self.df_["MedHouseVal"]
        n = X.shape[0]
        best_mse = float('inf')
        weights = np.array([ 0.64422243,  0.11740738, -0.80621795, -0.74821601, -0.42069151,  0.06012402,0.49112416])
        model = {
            "weights": weights.tolist(),
            "bias": self.bias,
            "mean_target": Y.mean(),
            "std_target": Y.std(),
            "mean_features": X.mean(axis=0).tolist(),
            "std_features": X.std(axis=0).tolist(),
            "targets": Y.tolist(),
            "r2": 0,
        }
        no_improvement = 0
        for epoch in range(2000):
            weights_norm = (X.T@X + self.ridge*self.I)**(-1)@X.T@Y
            y_pred = X.T
            errors = Y - y_pred
            mse = (errors.T @ errors) / n
            dL_dw = (2 * X.T @ (y_pred - Y)) / n
            dL_db = (2 * np.sum(Y - y_pred)) / n
            weights -= self.learning_rate * dL_dw
            self.bias -= self.learning_rate * dL_db
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
                model["bias"] = float(self.bias)
                model["real_pred"] = (y_pred * model["std_target"] + model["mean_target"]).tolist()
                model["r2"] = 1 - (errors.T @ errors)/np.sum((Y - Y.mean())**2)
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print("Early stopping due to no improvement.")
                    print(f"Best MSE: {best_mse}")
                    print(f"Weights: {weights}")
                    break
        end = time.perf_counter()
        print(f"Elapsed time for regression: {end - start_time}")
        df_result = pd.DataFrame({
            "Prediction": model["real_pred"],
            "Target": Y
        },index=Y.indexes)

        df_result["Diff"] = df_result["Prediction"] - df_result["Target"]
        df_result["AbsError"] = abs(df_result["Prediction"] - df_result["Target"])
        df_result_sorted = df_result.sort_values("AbsError",ascending=False)

        model_path = os.path.join(os.path.dirname(__file__),"model.csv")
        pd.DataFrame(model).to_csv(model_path)

        return df_result,df_result_sorted, model



# def show_transformed(etl) -> None:
#     print("Features values")
#     print(X)
#     print("Target values:")
#     print(Y)
#     print("Ranks X:")
#     print(pd.DataFrame(X_ranks, columns=feature_names))
#     print("Ranks Y:")
#     print(pd.DataFrame(Y_ranks, columns=["Target"]))
#     print(pd.DataFrame(X, columns=feature_names).describe())
#     print("\nZ-scores (first 7 rows):\n")
#     print(pd.DataFrame(X_zscore, columns=feature_names))
#     print("\nCorrelation matrix inbetween features (Pearson):\n")
#     print(df_fpearson)
#     print("\nCorrelation matrix inbetween features (Spearmann):\n")
#     print(pd.DataFrame(X_spearmann, index=feature_names, columns=feature_names))
#     print("\nSpearman correlation between features and target:\n")
#     print(pd.DataFrame(Y_spearmann, index=feature_names, columns=["Target"]))
#     print("\nPearson test for each feature with target:\n")
#     print(pd.DataFrame(xy_pearson, index=feature_names, columns=["Target"]))
#     print("\nTargets:\n")
#     print(pd.DataFrame(Y_zscore[:7], columns=["Targets 0-6"]))


# if model is None:
#     df_result, df_result_sorted, model = models(etl(features_engineering(df_full.copy())))
# else:
#     print("Loaded model — training skipped.")
