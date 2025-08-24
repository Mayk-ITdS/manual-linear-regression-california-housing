from pandas import DataFrame
from ETL import bounds
from ETL.transform import run_transform, pearson_corr, pearson_zscore, one_hot_encoding, add_interactions
from ETL.extract import df_full
from config import PROCESSED_DIR, SUMMARY_DIR

numeric_features = [
    "MedInc", "EstmIncPerson",
    "HouseAge", "RoomPerPerson",
    "PopulationPerHousehold"
]

categorical_keep = [
    "cat_LA Metro",
    "cat_S.F. Bay Area"
]

interaction_keep = [
    "MedInc_x_cat_LA Metro",
    "MedInc_x_cat_Central Valley",
    "MedInc_x_cat_SoCal",
    "MedInc_x_cat_S.F. Bay Area"
]

selected_features1 = numeric_features + categorical_keep + interaction_keep

numeric_features2 = [
    "EstmIncPerson", "EstmIncPerson_sq",
    "HouseAge", "RoomPerPerson",
    "PopulationPerHousehold"
]

categorical_keep2 = [
    "cat_LA Metro",
    "cat_S.F. Bay Area"
]

interaction_keep2 = [
    "MedInc_x_cat_LA Metro",
    "MedInc_x_cat_Central Valley",
    "MedInc_x_cat_SoCal",
    "MedInc_x_cat_S.F. Bay Area"
]

selected_features2 = numeric_features2 + categorical_keep2 + interaction_keep2
selected_features = [
    "MedInc", "MedInc_sq",
    "EstmIncPerson", "EstmIncPerson_sq",
    "RoomPerPerson", "HouseAge", "PopulationPerHousehold",
    "cat_LA Metro", "cat_S.F. Bay Area",
    "MedInc_x_cat_LA Metro", "MedInc_x_cat_S.F. Bay Area",
    "MedInc_x_cat_Central Valley", "MedInc_x_cat_SoCal"
]
def run_pipeline(df: DataFrame,bounds_module,normalization_method)->None:
    # --- Feature engineering ---
    ft_eng,df_eng = run_transform(df,bounds_module)
    df_eng.to_csv(PROCESSED_DIR / "df_updated_features.csv")

    #one hot for category and interacion with EstIncPerson_sq
    df_add_hot = one_hot_encoding(df_eng, drop_one=True)
    df_ready_to_normalise = add_interactions(df_add_hot)

    # --- Pearson correlation ---
    X = df_ready_to_normalise.drop(columns=["MedHouseVal","Category","MedHouseValQuartiles"])
    Y = df_ready_to_normalise["MedHouseVal"]

    x_zscored,y_zscored = pearson_zscore(X.to_numpy(dtype=float),Y.to_numpy(dtype=float))
    corr_x, corrxy = normalization_method(x_zscored, y_zscored,X.columns)

    corr_x.to_csv(PROCESSED_DIR / "feats_corr_matrix.csv")
    corrxy.to_csv(PROCESSED_DIR / "feats_target_corr.csv")

    #Final normalized dataset
    df_all_scaled = DataFrame(x_zscored,columns=X.columns,index=df_eng.index)
    df_all_scaled["MedHouseVal"] = y_zscored
    df_all_scaled.to_csv(PROCESSED_DIR / "df_numeric_scaled.csv",index=False)

    #Regress-ready
    df_final = df_all_scaled.copy()[selected_features]
    df_final["MedHouseVal"] = y_zscored
    df_final.to_csv(PROCESSED_DIR / "df_zscore_final.csv", index=False)

    # --- Summary ---
    summary_df = ft_eng.big_summary(df_eng)
    summary_df.to_csv(SUMMARY_DIR / "top_samples_summary.csv")

    print(" Pipeline finished. Data ready in:", PROCESSED_DIR, "and", SUMMARY_DIR)

    print("\n=== Sanity check: mean/std per column ===")
    desc = df_final.describe().T[["mean", "std"]].round(4)
    print(desc)
    print("\n=== Sanity check: features correlation ===")
    print(corr_x.round(3))
    print("\n=== Sanity check: target correlation ===")
    print(corrxy.round(3))

    off = desc[(desc["mean"].abs() > 1e-6) | (abs(desc["std"] - 1) > 1e-3)]
    if not off.empty:
        print("\n[WARNING] Some columns not properly standardized:")
        print(off)

if __name__ == "__main__":
    run_pipeline(df_full,bounds,normalization_method=pearson_corr)