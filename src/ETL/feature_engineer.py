from datetime import time
from itertools import count
from pathlib import Path
from typing import Tuple, Any
import numpy as np
import pandas as pd
from pandas import DataFrame
from ETL.extract import *
import time
from ETL import bounds


BBox = Tuple[float, float, float, float]

class FeatureEngineer:
    def __init__(self, data: DataFrame, bounds_module: object) -> None:
        self.df = data.copy()
        self.bounds: dict[str, BBox] = bounds_module.bounds_dict
        self.prioritised_categories: list = bounds_module.prioritised_categories
        self.out_col = "Category"
        self.lat_col: str = 'Latitude'
        self.lon_col: str = 'Longitude'
        self.default_label = 'Unknown'

    def categorise_coordinates(self):

        lat = pd.to_numeric(self.df[self.lat_col], errors='coerce')
        lon = pd.to_numeric(self.df[self.lon_col], errors='coerce')

        if self.out_col not in self.df.columns:
            self.df[self.out_col] = self.default_label

        for name in self.prioritised_categories:
            unassigned = self.df[self.out_col].eq(self.default_label)
            box = self.bounds.get(name)
            min_la, max_la, min_lon, max_lon = box
            mask = (
                    unassigned
                    & (lat.between(min_la, max_la))
                    & (lon.between(min_lon, max_lon))
            )
            self.df.loc[mask,self.out_col] = name

        return self.df

    def filter_samples(self,df_eng: DataFrame) -> tuple[DataFrame, DataFrame]:
        start = time.perf_counter()
        sample = df_eng["Category"].value_counts()
        big_samples = sample[sample > 60].index
        small_samples = sample[sample < 60].index
        sm = df_eng["Category"].isin(small_samples)
        m = df_eng["Category"].isin(big_samples)

        df_eng_small = df_eng.loc[sm].copy()
        df_eng_big = df_eng.loc[m].copy()

        df_eng_big["MedHouseValQuartiles"] = pd.NA
        df_eng_big.loc[m,"MedHouseValQuartiles"] = df_eng_big.loc[m].groupby("Category")["MedHouseVal"].transform(lambda x: pd.Categorical(
            pd.cut(
                x,
                bins=[-np.inf,x.quantile(0.25),x.quantile(0.5),x.quantile(0.75),np.inf],
                labels=["Q1","Q2","Q3","Q4"]
            ),
            categories=["Q1","Q2","Q3","Q4"],
            ordered=False
        ))
        end = time.perf_counter()
        print(f"filtering took {end-start} seconds")
        return df_eng_big,df_eng_small

    def big_summary(self, df_eng: DataFrame) -> DataFrame:

        return (df_eng.groupby('Category')
        .agg(
            samples=("MedHouseVal", "count"),
            medIncomeValue=("MedInc", "median"),
            MedHouseValMean=("MedHouseVal", "mean"),
            MedHouseValMedian=("MedHouseVal", "median"),
            MedHouseValStd=("MedHouseVal", "std"),
            population_sum=("Population", "sum"),

        ))
    def small_summary(self,df_eng:DataFrame) -> DataFrame:
        return (df_eng.groupby('Category').agg(
            samples=("MedHouseVal", "count"),
            medIncomeValue=("MedInc", "median"),
            MedHouseValMean=("MedHouseVal", "mean"),
            MedHouseValMedian=("MedHouseVal", "median"),
            MedHouseValStd=("MedHouseVal", "std"),
            population_sum=("Population", "sum"),
        ))

    def write_to_csv(self,df_eng:DataFrame,summary="big_summary") -> None:
        DATA_DIR = Path("../../data")
        PROCESSED_DIR = DATA_DIR / "processed"
        SUMMARY_DIR = DATA_DIR / "summary"

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        df_eng.to_csv(f"{DATA_DIR}/{summary}.csv",encoding="utf-8",index=True)
