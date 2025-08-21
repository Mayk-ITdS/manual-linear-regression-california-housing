from ETL import bounds
from ETL.extract import df_full
from ETL.feature_engineer import FeatureEngineer

ft = FeatureEngineer(data=df_full, bounds_module=bounds)
df_eng = ft.categorise_coordinates()
df_big,df_small = ft.filter_samples(df_eng)
sm_summary = ft.small_summary(df_small)
big_summary = ft.big_summary(df_big)

# ft.write_to_csv(sm_summary,"small_summary")
# ft.write_to_csv(big_summary,"big_summary")

print(big_summary)