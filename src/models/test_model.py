import regression
import visualizer
import pandas as pd
import numpy as np

import regression

etl_data = regression.etl(regression.df_full.copy())
model = regression.load_model()

# if model is None:
#     df_result, df_result_sorted, model = regression.models(etl_data)
# else:
#     # X = etl_data[0]
#     # Y = etl_data[1]
#     # X_zscore = etl_data[2]
#     #
#     # y_pred = X_zscore @ np.array(model["weights"]) + model["bias"]
#     # real_pred = y_pred * model["std_target"] + model["mean_target"]
#     #
#     # df_result = pd.DataFrame({
#     #     "Prediction": real_pred,
#     #     "Target": Y
#     # }, index=range(len(Y)))
#     #
#     # df_result["Diff"] = df_result["Prediction"] - df_result["Target"]
#     # df_result["AbsError"] = abs(df_result["Diff"])
#     # df_result_sorted = df_result.sort_values("AbsError", ascending=False)
X = etl_data[0]
Y = etl_data[1]
X_zscore = etl_data[2]
Y_zscore = etl_data[3]
df_xy_corr = etl_data[11]
feature_names = etl_data[16]
print("\nRaw-data: ")
print(X.head())
print("\nData types")
print(X.dtypes)

for col in X.columns:
    print(f"Unique types in {col}")
    print(X[col].map(type).unique())
print(X.describe(include="all"))
print("\nNan-amount: ")
print(X.isna().sum().sum())

# viz.one_choice()
# viz.prediction_vs_target()
# viz.correlation_plot()
viz = visualizer.Visualizer(X,Y,X_zscore,Y_zscore,df_xy_corr,feature_names,df_result=20)

viz.one_choice()
