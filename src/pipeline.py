from pathlib import Path
from pandas import DataFrame
from ETL import bounds
from ETL.transform import run_transform, pearson_corr, pearson_zscore, one_hot_encoding
from ETL.extract import df_full

DATA_DIR = Path("../data")
PROCESSED_DIR = DATA_DIR / "processed"
SUMMARY_DIR = DATA_DIR / "summary"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
numeric_features = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
def run_pipeline(df: DataFrame,bounds_module,normalization_method)->None:

    ft_eng,df_eng = run_transform(df,bounds_module)
    df_eng.to_csv(PROCESSED_DIR / "df_updated_features.csv")

    x_zscored,y_zscored = pearson_zscore(df_eng[numeric_features],df_eng["MedHouseVal"])

    corr_matrix = normalization_method(x_zscored,y_zscored,numeric_features)
    corr_matrix.to_csv(PROCESSED_DIR / "feats_corr_matrix.csv")

    df_numeric_scaled = DataFrame(x_zscored,columns=numeric_features,index=df_eng.index)
    df_numeric_scaled.to_csv(PROCESSED_DIR / "df_numeric_scaled.csv")

    df_final = one_hot_encoding(df_eng,drop_one=True)
    df_final[numeric_features] = df_numeric_scaled
    df_final["MedHouseVal"] = y_zscored
    df_final.to_csv(PROCESSED_DIR / "df_zscore_final.csv")

    summary_df = ft_eng.big_summary(df_eng)
    summary_df.to_csv(SUMMARY_DIR / "top_samples_summary.csv")

if __name__ == "__main__":
    run_pipeline(df_full,bounds,normalization_method=pearson_corr)