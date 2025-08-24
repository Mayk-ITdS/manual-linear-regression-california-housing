import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ETL.feature_engineer import FeatureEngineer
from ETL import bounds
from ETL.extract import df_full

ft_eng = FeatureEngineer(df_full, bounds)
df_corr = ft_eng.categorise_coordinates()
df_big, df_sm = ft_eng.filter_samples(df_corr)

class Visualizer:
    def __init__(self, X, Y, X_z, Y_z, df_xy_corr, feature_names, df_result, df_big, df_small):
        self.df_result = df_result
        self.X = X
        self.Y = Y
        self.X_z = X_z
        self.Y_z = Y_z
        self.df_xy_corr = df_xy_corr
        self.feature_names = feature_names
        self.df_big = df_big
        self.df_small = df_small

    def correlation_plot(self):
        self.df_xy_corr.plot(kind='barh', legend=False, figsize=(8, 6))
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

    def all_features_vs_target(self):
        df_ = self.df_big
        order = (df_.groupby("Category")["MedHouseVal"].median()
                 .sort_values()
                 .index.tolist())
        cats = order[:8]
        plt.figure(figsize=(10, 5))

        for cat in cats:
            vals = np.sort(df_.loc[df_["Category"].eq(cat), "MedHouseVal"].dropna().values)
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            plt.plot(vals, y, label=f"{cat} (n={len(vals)})")

        plt.xlabel("MedHouseVal")
        plt.ylabel("ECDF")
        plt.title("Empiryczna dystrybuanta MedHouseVal per kategoria")
        plt.legend(frameon=False, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_distribution_by_category(self):
        d_ = self.df_big.drop(["Latitude", "Longitude"], axis=1)
        features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup"]
        cats = sorted(d_["Category"].dropna().unique().tolist())
        palette = sns.color_palette("tab10", n_colors=len(cats))
        palette_map = {c: col for c, col in zip(cats, palette)}

        hue_order = d_["Category"].value_counts().sort_values(ascending=True).index.tolist()
        plt.figure(figsize=(22, 14))

        for i, col in enumerate(features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=d_, x=col, hue="Category", hue_order=hue_order, bins=30, kde=False, multiple="layer",
                         edgecolor="black", palette=palette_map, alpha=0.4, linewidth=1.2)
            plt.title(f"Histogram of {col} by Category")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def feature_vs_target(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        features_var_cat = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                            "Population", "AveOccup","Category"]
        d_big = pd.DataFrame(self.df_big, columns=features_var_cat)
        Y = self.df_big['MedHouseVal']

        high_range_features = ["Category", "MedInc"]
        low_range_features = [col for col in features_var_cat if col not in high_range_features]

        plt.figure(figsize=(10, 5))

        if 'Category' in d_big.columns:
            plt.figure(figsize=(10, 5))
            cat = d_big['Category'].astype('category')
            order = list(cat.cat.categories)
            data = [Y[cat == k] for k in order]
            plt.boxplot(data, labels=order, showfliers=False)
            plt.xlabel("Category")
            plt.ylabel("MedHouseVal")
            plt.title("Target distribution by Category")
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()


    def prediction_vs_target(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df_result["Prediction"], self.df_result["Target"], alpha=0.3)
        plt.plot([self.df_result["Target"].min(), self.df_result["Target"].max()],
                 [self.df_result["Target"].min(), self.df_result["Target"].max()],
                 color="red", linestyle="--", label="Ideal prediction")
        plt.xlabel("Median Income")
        plt.ylabel("Target")
        plt.title("Median Income vs Target")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def each_feature(self):
        for col in self.X.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(self.X[col], bins=100, color="pink", edgecolor="black", alpha=0.3)
            plt.title(f"{col} distribution")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.grid(axis="y", alpha=0.3)
            plt.savefig(f"../../plots/{col}-distribution.png")
            plt.show()

    def one_choice(self):
        x = self.X["Longitude"]
        y = self.X["Latitude"]
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.3)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Latitude vs Longitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("../../plots/Latitude-vs-Longitude.png")
        plt.show()

    def regression_on_two_features(self):
        selected_features = ["Category", "MedInc"]

        for feature in selected_features:
            x = self.df_big[selected_features]
            y = self.df_big['MedHouseVal']

            w, b = np.polyfit(x, y, 1)
            y_pred = w * x + b

            # MSE i R²
            mse = np.mean((y - y_pred) ** 2)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, alpha=0.3, label="Data")
            plt.plot(np.sort(x), w * np.sort(x) + b, color="red", label=f"y={w:.2f}x + {b:.2f}")
            plt.title(f"{feature} vs Target\nMSE={mse:.4f} | R²={r2:.4f}")
            plt.xlabel(f"{feature} (Z-score)")
            plt.ylabel("Target (Z-score)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

if __name__=="__main__":
    viz = Visualizer(x, y, x_zscore, y_zscore, df_xy_corr, feature_names, df_full, df_big, df_sm)
    # viz.show_distribution_by_category()
    # viz.regression_on_two_features()
    viz.feature_vs_target()
    print(df_big)