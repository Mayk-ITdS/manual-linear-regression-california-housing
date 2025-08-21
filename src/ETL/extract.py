from sklearn.datasets import fetch_california_housing
import pandas as pd

pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", 200)
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

df_full = data.frame
X_features = df_full.iloc[:,:-1]
Y_target = df_full.iloc[:,-1]

