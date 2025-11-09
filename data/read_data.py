import pandas as pd
path = 'data/NIFTY/OPT.parquet'
df = pd.read_parquet(path)
print(df.head())
print(df.tail())
# df['Ticker'] = "NIFTY"
# print(df.head())
# df.to_parquet(path)