Data format requireemtns
store data i data/ folder with Underlying Name as a folder. create SPOT, FUT and OPT as parquet files to represnt derrivatives. all available data should be dumped in a single parquet file as of now . (need to change it later)

Example 
data/NIFTY => folder
data/NIFTY/SPOT.parquet
data/NIFTY/FUT.parquet
data/NIFTY/OPT.parquet



start backend server :
```bash
    # activate venv before
    python -m uvicorn frontend.backend:app --host 127.0.0.1 --port 8000 --reload
```

start frontend server 
```bash
    #activate backend server first
    streamlit run frontend/streamlit_app.py
```

streamlit will be started at [http://localhost:8501]






fut columns :
date           object
time           object
underlying     object
contract       object
expirydate     object
expiry_type    object
open           object
high           object
low            object
close          object
volume          int64
dtype: object

spot columns :
Ticker    object
Date      object
Time      object
Open      object
High      object
Low       object
Close     object
dtype: object

OPT COLUMNS 
:
Contract          object
Underlying        object
Expiry_Date       object
Strike           float64
Option_Type       object
Date              object
Time              object
Open             float64
High             float64
Low              float64
Close            float64
Volume             int64
Open_Interest      int64
dtype: object

