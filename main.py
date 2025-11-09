from datetime import datetime
from pathlib import Path
import json
import os
from backtest.engine import Backtester
from strategies.short_straddle_delta_hedge.strategy import ShortStraddleDeltaHedge

def main():
    root_path = Path(__file__).parent
    data_path = root_path / "data"
    
    strategy_name = os.getenv('BACKTEST_STRATEGY', 'short_straddle_delta_hedge')
    strategy_path = root_path / "strategies" / strategy_name
    
    with open(strategy_path / "config.json", "r") as f:
        config = json.load(f)
    
    start_date_str = os.getenv('BACKTEST_START_DATE', '2025-01-01')
    end_date_str = os.getenv('BACKTEST_END_DATE', '2025-01-31')
    
    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except ValueError:
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 31)
    
    if strategy_name == 'short_straddle_delta_hedge':
        strategy = ShortStraddleDeltaHedge(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    backtester = Backtester(data_path, strategy)
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    print(f"Strategy: {config['strategy_name']}")
    print(f"Underlying: {config['underlying']}")
    print(f"Initial Capital: {config['capital']}")
    
    backtester.run(start_date, end_date)

if __name__ == "__main__":
    main()
