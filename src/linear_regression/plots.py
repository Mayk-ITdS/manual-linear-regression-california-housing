import regression
import matplotlib.pyplot as plt
X_zscore,Y_zscore,X_train,X_test,Y_train,Y_test, xy_pearson, features_pearson,df_fpearson,df_xy_corr = regression.etl()

df_xy_corr.plot(kind='barh', legend=True)
plt.title("Korelacja cech z targetem")
plt.tight_layout()
plt.show()

for i, col in enumerate(regression.columns[:-1]):
    plt.figure()
    plt.scatter(X_zscore[:, i], Y_zscore, alpha=0.3)
    plt.title(f"{col} vs MedHouseVal")
    plt.xlabel(col)
    plt.ylabel("MedHouseVal")
    plt.grid(True)
    plt.show()