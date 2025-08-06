import regression
import matplotlib.pyplot as plt
X, Y, X_zscore, Y_zscore, X_train, X_test, Y_train, Y_test, xy_pearson, features_pearson, df_fpearson, df_xy_corr, X_ranks, Y_ranks, X_spearmann, Y_spearmann, feature_names, X_rank_z, Y_rank_z = regression.etl(regression.features_engineering(regression.df_full.copy()))

class Visualizer:
    def __init__(self,X_z,Y_z,df_xy_corr,feature_names,df_result):
        self.df_result=df_result
        self.X_z=X_z
        self.Y_z=Y_z
        self.df_xy_corr=df_xy_corr
        self.feature_names=feature_names
    def correlation_plot(self):
        self.df_xy_corr.plot(kind='barh', legend=False,figsize=(8,6))
        plt.title("Feature - Target Correlation")
        plt.tight_layout()
        plt.show()

    def scatter_plot(self):
        for i, col in enumerate(self.feature_names):
            plt.figure()
            plt.scatter(self.X_z[:, i], self.Y_z, alpha=0.3)
            plt.title(f"{col} vs MedHouseVal")
            plt.xlabel(col)
            plt.ylabel("MedHouseVal Z-score")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def prediction_vs_target(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df_result["Prediction"], self.df_result["Target"], alpha=0.3)
        plt.plot([self.df_result["Target"].min(), self.df_result["Target"].max()],
                 [self.df_result["Target"].min(), self.df_result["Target"].max()],
                 color="red", linestyle="--", label="Ideal prediction")
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.title("Prediction vs Real Target")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def one_choice(self):
        x = self.df_result.index
        y = self.df_result["AbsError"]
        plt.figure(figsize=(8, 5))
        plt.scatter(x,y, alpha=0.3)
        plt.xlabel("Data row index")
        plt.ylabel("AbsError")
        plt.title("Absolute Error vs Data Index")
        plt.grid(True)
        plt.tight_layout()
        plt.show()