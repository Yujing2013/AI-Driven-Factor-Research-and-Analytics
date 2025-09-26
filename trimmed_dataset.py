import pandas as pd

PARQUET = "a_share_pv_clean_2015_2025.parquet"
out = "a_share_pv_clean_2015_2025_trimmed.parquet"

df = pd.read_parquet(PARQUET)

# # 1) 丢掉没有价格的空行（未来日期或下载缺口）
# df = df.dropna(subset=['open','high','low','close'], how='any')

# 2) 保留到20250925
cutoff = "20250925"   # 今天
df = df.loc[df.index.get_level_values('trade_date') <= cutoff]

df.to_parquet(out)
print("done:", out, df.shape)
