from quant_data_lib.Quant_data_lib import get_indices_data, get_options_data, get_OptContract
from datetime import datetime

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 3, 31)
symbols = ['INDIA VIX']
index = get_indices_data(
    start_date=start_date,
    end_date=end_date,
    symbols=symbols
)
print(index)


# startdate = datetime(2025,1,1)
# enddate = datetime(2025,2,28)
# query4 = "get_Contracts_info"
# underlyings = ['NIFTY']
# contracts = get_OptContract(start_date=startdate, end_date=enddate, symbols=underlyings, query_type=query4 )
# contracts = contracts['contracts'].tolist()
# # 0         NIFTY  NIFTY02JAN2521600PE  2025-01-02  21600.00         PE
# # 1         NIFTY  NIFTY02JAN2521650PE  2025-01-02  21650.00         PE
# # 2         NIFTY  NIFTY02JAN2521700PE  2025-01-02  21700.00         PE
# # 3         NIFTY  NIFTY02JAN2521750PE  2025-01-02  21750.00         PE
# # 4         NIFTY  NIFTY02JAN2521800PE  2025-01-02  21800.00         PE

# Opt1 = get_options_data(start_date=startdate, end_date=enddate, contracts=contracts )
# print(contracts)
# import pandas as pd
# df = pd.read_csv(r'C:\Users\kakad\OneDrive\Documents\Quant_Projects\opt_backtestser\data\BANKNIFTY\banknifty.csv')
# index.to_parquet(r'data\INDIA VIX\VIX.parquet')