import pandas as pd

PARQUET = "a_share_pv_clean_2015_2025_trimmed.parquet"

df = pd.read_parquet(PARQUET)
print(df.info(show_counts=True))
print(df)
print(df.shape)
print(type(df))

