import regression
import visualizer
import pandas as pd
import numpy as np

etl_data = regression.etl(regression.features_engineering(regression.df_full.copy()))
model = regression.load_model()

if model is None:
    df_result, df_result_sorted, model = regression.linear_regression(etl_data)
else:
    X = etl_data[0]
    Y = etl_data[1]
    X_zscore = etl_data[2]

    y_pred = X_zscore @ np.array(model["weights"]) + model["bias"]
    real_pred = y_pred * model["std_target"] + model["mean_target"]

    df_result = pd.DataFrame({
        "Prediction": real_pred,
        "Target": Y
    }, index=range(len(Y)))

    df_result["Diff"] = df_result["Prediction"] - df_result["Target"]
    df_result["AbsError"] = abs(df_result["Diff"])
    df_result_sorted = df_result.sort_values("AbsError", ascending=False)

X_zscore = etl_data[2]
Y_zscore = etl_data[3]
df_xy_corr = etl_data[11]
feature_names = etl_data[16]

viz = visualizer.Visualizer(X_zscore, Y_zscore, df_xy_corr, feature_names, df_result)
viz.one_choice()
