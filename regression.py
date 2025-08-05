from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
data = fetch_california_housing(as_frame=True)
columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude', 'MedHouseVal']
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

#This dataset was derived from the 1990 U.S. census, using one row per census
#block group. A block group is the smallest geographical unit for which the U.S.
#Census Bureau publishes sample data (a block group typically has a population
#of 600 to 3,000 people)
np.random.seed(42)

df_full = data.frame
df_full = df_full.sample(frac=1).reset_index(drop=True)


X = np.array(df_full.drop(["MedHouseVal"],axis=1))
Y = np.array(df_full["MedHouseVal"])

X_zscore = (X - X.mean(axis=0))/X.std(axis=0)
Y_zscore = (Y - Y.mean(axis=0))/Y.std(axis=0)

number_samples = df_full.shape[0]
split_index = int(number_samples * 0.8)

X_train = X_zscore[:split_index,:]
X_test = X_zscore[split_index:,:]

Y_train = Y_zscore[:split_index]
Y_test = Y_zscore[split_index:]

#test of pearson correlation

features_pearson = (X_zscore.T @ X_zscore)/ (number_samples - 1)
df_fpearson = pd.DataFrame(features_pearson,index=columns[:-1],columns=columns[:-1])
xy_pearson = (X_zscore.T @ Y_zscore)/ (number_samples - 1)
print("Z-score (pierwsze 7 wierszy):")
print(X_zscore[:7, :])
print("\nMacierz korelacji między cechami (Pearson):")
print(df_fpearson)
print("\nKorelacja każdej cechy z targetem (Pearson):")
print(xy_pearson)
print("Targets")
print(Y_zscore[:7])