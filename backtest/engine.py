from datetime import datetime, time, timedelta
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Optional
from collections import deque
import numpy as np
import math
import logging
from backtest.base_strategy import BaseStrategy, Order, OrderType, OrderSide, Trade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, data_path: Path, strategy: BaseStrategy):
        self.data_path = data_path
        self.strategy = strategy
        try:
            self.strategy.backtester = self
        except Exception:
            pass
        self.current_data: Dict[str, pd.DataFrame] = {}
        self.trades: List[Trade] = []
        self._data_file_cache: Dict[str, pd.DataFrame] = {}
        
        self.transaction_costs = {
            "brokerage_percentage": 0.0003,  
            "stt_percentage": 0.0005,        
            "exchange_charges": 0.0001,      
            "gst": 0.18                     
        }
        
    def _load_data(self, date: datetime.date) -> None:
        self.current_data.clear()
        
        for symbol in self.strategy.subscribed_instruments:
          
            types_to_try = []
            if isinstance(symbol, dict):
                sub = symbol
                underlying = sub.get('underlying')
               
                for t in ('SPOT', 'FUT', 'OPT'):
                    if sub.get(t, False):
                        types_to_try.append(t)
                if not types_to_try:
                    types_to_try = ['SPOT', 'FUT', 'OPT']
                sub_repr = f"{underlying}:{types_to_try}"
            else:
               
                if not isinstance(symbol, str):
                    logger.warning(f"Unsupported subscription type: {type(symbol)} - {symbol}")
                    continue
                parts = symbol.split('_')
                underlying = parts[0]
                suffix = parts[-1].upper() if len(parts) > 1 else ''
                if suffix in ('FUT', 'SPOT', 'OPT'):
                    types_to_try = [suffix]
                    underlying = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
                else:
                    types_to_try = ['SPOT', 'FUT', 'OPT']

            for inst in types_to_try:
                file_name = f"{inst}.parquet"
                file_path = self.data_path / underlying / file_name
                if not file_path.exists():
                    logger.debug(f"No {file_name} for {underlying} at {file_path}")
                    continue

                file_key = str(file_path.resolve())
                try:
                   
                    if file_key not in self._data_file_cache:
                        df_full = pd.read_parquet(file_path)

                        if inst == 'FUT':
                            df_full['timestamp'] = pd.to_datetime(df_full['date'].astype(str) + ' ' + df_full['time'].astype(str))
                            for c in ('open', 'high', 'low', 'close'):
                                if c in df_full.columns:
                                    df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
                            df_full.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'underlying': 'underlying', 'contract': 'contract'}, inplace=True)
                            df_full['instrument'] = 'FUT'
                            df_full['contract'] = df_full['contract'] if 'contract' in df_full.columns else None
                            df_full['underlying'] = df_full['underlying'] if 'underlying' in df_full.columns else underlying

                        elif inst == 'SPOT':
                            df_full['timestamp'] = pd.to_datetime(df_full['Date'].astype(str) + ' ' + df_full['Time'].astype(str))
                            for c in ('Open', 'High', 'Low', 'Close'):
                                if c in df_full.columns:
                                    df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
                            df_full['Volume'] = df_full['Volume'] if 'Volume' in df_full.columns else 0
                            df_full['instrument'] = 'SPOT'
                            df_full['contract'] = None
                            df_full['underlying'] = df_full['Ticker'] if 'Ticker' in df_full.columns else underlying

                        else:  
                            df_full['timestamp'] = pd.to_datetime(df_full['Date'].astype(str) + ' ' + df_full['Time'].astype(str))
                            for c in ('Open', 'High', 'Low', 'Close'):
                                if c in df_full.columns:
                                    df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
                            df_full['Volume'] = df_full['Volume'] if 'Volume' in df_full.columns else 0
                            df_full['instrument'] = 'OPT'
                            df_full['contract'] = df_full['Contract'] if 'Contract' in df_full.columns else None
                            df_full['underlying'] = df_full['Underlying'] if 'Underlying' in df_full.columns else underlying

                        for col in ('Open', 'High', 'Low', 'Close'):
                            if col not in df_full.columns:
                                df_full[col] = pd.NA
                        if 'Volume' not in df_full.columns:
                            df_full['Volume'] = 0

                        df_full.set_index('timestamp', inplace=True)
                        self._data_file_cache[file_key] = df_full
                    else:
                        df_full = self._data_file_cache[file_key]

                    if df_full.empty:
                        logger.warning(f"Data file {file_path} is empty or has no usable timestamp column")
                        continue

                    df_date = df_full[df_full.index.date == date]

                    if df_date.empty:
                        logger.debug(f"No rows for date {date} in {file_path}")
                        continue

                    
                    if inst == 'SPOT':
                        ticker_col = 'Ticker' if 'Ticker' in df_date.columns else 'underlying'
                        ticker_vals = df_date[ticker_col].astype(str).unique()
                        spot_ticker = ticker_vals[0] if len(ticker_vals) > 0 else underlying
                        key = f"{spot_ticker}_SPOT"
                        self.current_data[key] = df_date

                    elif inst == 'FUT':
                        if 'contract' in df_date.columns and df_date['contract'].notna().any():
                            for contract, grp in df_date.groupby('contract'):
                                key = str(contract)
                                self.current_data[key] = grp
                        else:
                            key = f"{underlying}_FUT"
                            self.current_data[key] = df_date

                    else:  
                        if 'contract' in df_date.columns and df_date['contract'].notna().any():
                            for contract, grp in df_date.groupby('contract'):
                                key = str(contract)
                                self.current_data[key] = grp
                        else:
                           
                            if all(x in df_date.columns for x in ('Strike', 'Option_Type', 'Expiry_Date')):
                                df_temp = df_date.copy()
                                df_temp['contract'] = (
                                    df_temp['Underlying'].astype(str) + '_' +
                                    df_temp['Expiry_Date'].astype(str) + '_' +
                                    df_temp['Strike'].astype(str) + '_' +
                                    df_temp['Option_Type'].astype(str)
                                )
                                for contract, grp in df_temp.groupby('contract'):
                                    key = str(contract)
                                    self.current_data[key] = grp
                            else:
                                key = f"{underlying}_OPT"
                                self.current_data[key] = df_date

                    logger.debug(f"Loaded data keys (sample): {list(self.current_data.keys())[:5]}")

                except Exception as e:
                    logger.exception(f"Error loading data for {symbol} from {file_path}: {e}")

    def get_spot_key(self, underlying: str) -> Optional[str]:
       
        up = underlying.upper()
        for key, df in self.current_data.items():
           
            try:
                if df.get('instrument', None) == 'SPOT' and str(df.get('underlying', '')).upper() == up:
                    return key
            except Exception:
                continue
        for key in self.current_data.keys():
            if key.upper().startswith(up) and key.upper().endswith('_SPOT'):
                return key
        return None

    def get_available_option_contracts(self, underlying: str, as_of_date: Optional[datetime.date] = None) -> Dict:
        
        mapping = {}
        up = underlying.upper()
        for key, df in self.current_data.items():
            try:
               
                inst = df['instrument'].iloc[0] if 'instrument' in df.columns and len(df) > 0 else None
                if inst != 'OPT':
                    continue
                und_val = df['underlying'].iloc[0] if 'underlying' in df.columns and len(df) > 0 else None
                if und_val is None or str(und_val).upper() != up:
                    continue

                expiry_col = next((c for c in df.columns if 'expir' in c.lower()), None)
                strike_col = next((c for c in df.columns if c.lower() == 'strike'), None)
                type_col = next((c for c in df.columns if 'option' in c.lower() or c.lower() in ('type', 'option_type')), None)

                expiry = None
                if expiry_col and expiry_col in df.columns and df[expiry_col].notna().any():
                    expiry = pd.to_datetime(df[expiry_col].iloc[0]).date()
                else:
                    parts = str(key).split('_')
                    for p in parts:
                        if len(p) >= 6 and p.isdigit():
                            try:
                                expiry = pd.to_datetime(p, format='%Y%m%d').date()
                                break
                            except Exception:
                                continue

                strike = None
                if strike_col and strike_col in df.columns and df[strike_col].notna().any():
                    strike = float(df[strike_col].iloc[0])
                else:
                    kparts = str(key).split('_')
                    for p in kparts:
                        try:
                            strike = float(p)
                            break
                        except Exception:
                            continue

                opt_type = None
                if type_col and type_col in df.columns and df[type_col].notna().any():
                    opt_type = str(df[type_col].iloc[0])
                else:
                   
                    if key.upper().endswith('_CE'):
                        opt_type = 'CE'
                    elif key.upper().endswith('_PE'):
                        opt_type = 'PE'

                if expiry is None or strike is None or opt_type is None:
                    continue

                mapping.setdefault(expiry, {}).setdefault(str(strike), {})[opt_type] = key
            except Exception as e:
                logger.exception(f'Error processing contract {key}: {e}')
                continue

        return mapping

    def choose_nearest_expiry(self, expiries: List[datetime.date], as_of: datetime.date) -> Optional[datetime.date]:
        
        if not expiries:
            return None
        future = [e for e in expiries if e >= as_of]
        if future:
            return min(future)
        return max(expiries)

    def _calculate_transaction_cost(self, order: Order, fill_price: float) -> float:
        
        value = abs(order.quantity * fill_price)

        brokerage = value * self.transaction_costs["brokerage_percentage"]
        stt = value * self.transaction_costs["stt_percentage"]
        exchange = value * self.transaction_costs["exchange_charges"]
        gst = (brokerage + exchange) * self.transaction_costs["gst"]

        variable_cost = brokerage + stt + exchange + gst

        instrument_type = None
        df = self.current_data.get(order.symbol)
        if df is not None and "instrument" in df.columns and len(df) > 0:
            try:
                instrument_type = str(df["instrument"].iloc[0]).upper()
            except Exception:
                instrument_type = None

        lot_size_cfg = int(self.strategy.config.get("lot_size", 1)) if self.strategy and hasattr(self.strategy, "config") else 1
        
        if lot_size_cfg > 0 and order.quantity % lot_size_cfg == 0:
            num_lots = order.quantity // lot_size_cfg
        else:
           
            num_lots = order.quantity

        fixed_cost = 0.0
        if instrument_type == 'OPT':
            fixed_cost = 25.0 * num_lots  
        elif instrument_type == 'FUT':
            fixed_cost = 100.0 * num_lots 

        return variable_cost + fixed_cost

    def _simulate_order_fill(self, order: Order, timestamp: datetime) -> Optional[float]:
        if order.symbol not in self.current_data:
            logger.error(f"No data available for {order.symbol}")
            return None
            
        data = self.current_data[order.symbol]
        if timestamp not in data.index:
            logger.error(f"No data available for {order.symbol} at {timestamp}")
            return None
            
        current_price = float(data.loc[timestamp, 'Close'])  
        if order.order_type == OrderType.MARKET:
            slippage = 0.001  
            fill_price = float(current_price * (1 + slippage if order.side == OrderSide.BUY else 1 - slippage))
        else:  
            if order.limit_price is None:
                logger.error("Limit price not specified for LIMIT order")
                return None
            if (order.side == OrderSide.BUY and current_price > order.limit_price) or \
               (order.side == OrderSide.SELL and current_price < order.limit_price):
                return None  
            fill_price = float(order.limit_price)
            
        return fill_price

    def place_order(self, order: Order, timestamp: datetime) -> bool:
      
        fill_price = self._simulate_order_fill(order, timestamp)
        if fill_price is None:
            return False
            
        self.strategy.on_trade_update(order, fill_price, timestamp)
        

        try:
            trade = Trade(
                symbol=str(order.symbol),
                timestamp=timestamp.replace(microsecond=0),  
                price=round(float(fill_price), 3),  
                quantity=int(order.quantity),  
                side=order.side
            )
            self.trades.append(trade)
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            logger.error(f"Values: symbol={order.symbol}, price={fill_price}, quantity={order.quantity}")
        
        return True

    def run(self, start_date: datetime, end_date: datetime):
      
      
        strategy_path = Path(self.strategy.__class__.__module__.replace('.', '/')).parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        report_dir = strategy_path / "report" / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

      
        self._run_report_dir = report_dir

        try:
            file_log_path = report_dir / "backtest.log"
            file_handler = logging.FileHandler(file_log_path, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
            file_handler.setFormatter(file_formatter)
          
            root_logger = logging.getLogger()
            if not any(getattr(h, 'baseFilename', None) == str(file_log_path) for h in root_logger.handlers if hasattr(h, 'baseFilename')):
                root_logger.addHandler(file_handler)
            logger.info(f"Backtest run directory: {report_dir}")
        except Exception:
            logger.exception("Failed to configure file logging for backtest run")

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5: 
                logger.info(f"Processing date: {current_date.date()}")
                
                subscribed = self.strategy.on_start(current_date)
                if subscribed:
                    
                    self._load_data(current_date.date())
                    
                    
                    market_start = datetime.combine(current_date.date(), time(9, 15))
                    market_end = datetime.combine(current_date.date(), time(15, 30))
                    
                    current_time = market_start
                    while current_time <= market_end:
                        
                        minute_data = {}
                        for symbol, df in self.current_data.items():
                            if current_time in df.index:
                                minute_data[symbol] = {
                                    'open': df.loc[current_time, 'Open'],
                                    'high': df.loc[current_time, 'High'],
                                    'low': df.loc[current_time, 'Low'],
                                    'close': df.loc[current_time, 'Close'],
                                    'volume': df.loc[current_time, 'Volume']
                                }
                        
                        
                        if minute_data:
                            self.strategy.on_data(current_time, minute_data)
                        
                        current_time += timedelta(minutes=1)
                    
                    
            current_date += timedelta(days=1)
        
        
        self._generate_report()

    def _square_off_all(self, timestamp: datetime):
        
        for symbol, position in list(self.strategy.positions.items()):
            order = Order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                quantity=abs(position.quantity)
            )
            self.place_order(order, timestamp)

    def _generate_report(self):
        if not self.trades:
            logger.warning("No trades to generate report")
            return

        report_dir = getattr(self, '_run_report_dir', None)
        if report_dir is None:
            strategy_path = Path(self.strategy.__class__.__module__.replace('.', '/')).parent
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            report_dir = strategy_path / "report" / timestamp
            report_dir.mkdir(parents=True, exist_ok=True)

        trades_list = []
        for t in self.trades:
            trades_list.append({
                'symbol': t.symbol,
                'timestamp': pd.to_datetime(t.timestamp),
                'price': float(t.price),
                'quantity': int(t.quantity),
                'side': t.side.value if hasattr(t.side, 'value') else str(t.side)
            })
        trades_df = pd.DataFrame(trades_list).sort_values('timestamp').reset_index(drop=True)

        raw_trades_path = report_dir / 'trades.csv'
        if not trades_df.empty:
            raw_export = trades_df.copy()
            raw_export['timestamp'] = raw_export['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            raw_export.to_csv(raw_trades_path, index=False)
            logger.info(f"Raw trades saved to {raw_trades_path}")
        else:
            pd.DataFrame(columns=['symbol','timestamp','price','quantity','side']).to_csv(raw_trades_path, index=False)
            logger.info(f"Empty raw trades file saved to {raw_trades_path}")

        tradebook_rows = []
        for symbol in trades_df['symbol'].unique():
            sym_trades = trades_df[trades_df['symbol'] == symbol].copy()
            sym_trades = sym_trades.reset_index(drop=True)

            open_queue = deque()
            for _, row in sym_trades.iterrows():
                side = row['side']
                qty = int(row['quantity'])
                price = float(row['price'])
                ts = row['timestamp']

                qty_remaining = qty

                while qty_remaining > 0 and open_queue and open_queue[0]['side'] != side:
                    open_trade = open_queue[0]
                    match_qty = min(qty_remaining, open_trade['qty'])

                  
                    if open_trade['timestamp'] <= ts:
                        entry = open_trade
                        exit_trade = {'qty': match_qty, 'price': price, 'timestamp': ts, 'side': side}
                    else:
                        entry = {'qty': match_qty, 'price': price, 'timestamp': ts, 'side': side}
                        exit_trade = open_trade

                    entry_side = entry['side']
                    pnl = (exit_trade['price'] - entry['price']) * (1 if entry_side.upper() == 'BUY' else -1) * match_qty

                    tradebook_rows.append({
                        'symbol': symbol,
                        'entry_timestamp': entry['timestamp'],
                        'entry_price': entry['price'],
                        'entry_qty': match_qty,
                        'entry_side': entry['side'],
                        'exit_timestamp': exit_trade['timestamp'],
                        'exit_price': exit_trade['price'],
                        'exit_qty': match_qty,
                        'exit_side': exit_trade['side'],
                        'pnl': round(pnl, 3)
                    })

                    qty_remaining -= match_qty
                    open_trade['qty'] -= match_qty
                    if open_trade['qty'] == 0:
                        open_queue.popleft()

                if qty_remaining > 0:
                    open_queue.append({'qty': qty_remaining, 'price': price, 'timestamp': ts, 'side': side})

        tradebook_df = pd.DataFrame(tradebook_rows)

        if not tradebook_df.empty:
            tradebook_df['charges'] = 0.0
            for idx, row in tradebook_df.iterrows():
                symbol = row['symbol'].upper()
                
                if symbol.endswith('FUT'):
                    
                    tradebook_df.at[idx, 'charges'] = 100.0 * 2
                elif symbol.endswith('CE') or symbol.endswith('PE'):
                   
                    tradebook_df.at[idx, 'charges'] = 25.0 * 2
            
            tradebook_df['pat'] = tradebook_df['pnl'] - tradebook_df['charges']

        tradebook_path = report_dir / 'tradebook.csv'
        if not tradebook_df.empty:

            tb_export = tradebook_df.copy()
            tb_export['entry_timestamp'] = tb_export['entry_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            tb_export['exit_timestamp'] = tb_export['exit_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            tb_export.to_csv(tradebook_path, index=False)
            logger.info(f"Tradebook saved to {tradebook_path}")
        else:

            pd.DataFrame(columns=['symbol','entry_timestamp','entry_price','entry_qty','entry_side','exit_timestamp','exit_price','exit_qty','exit_side','pnl','charges','pat']).to_csv(tradebook_path, index=False)
            logger.info(f"Empty tradebook saved to {tradebook_path}")

        closed_tb = tradebook_df.copy()
        if closed_tb.empty:
            logger.warning('No closed round-trip trades found for metrics')

            initial_capital = float(self.strategy.config.get('capital', self.strategy.config.get('initial_capital', 1000000)))
            report = {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'initial_capital': initial_capital,
                'return_on_initial_capital': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0
            }
            report_json_path = report_dir / 'report.json'
            with open(report_json_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Report saved to {report_json_path}")
            return

        closed_tb['pnl'] = closed_tb['pnl'].astype(float)
        total_pnl = float(closed_tb['pnl'].sum())
        total_charges = float(closed_tb.get('charges', pd.Series(dtype=float)).sum()) if 'charges' in closed_tb.columns else 0.0
        total_pat = total_pnl - total_charges
        total_trades = len(closed_tb)
        profitable_trades = int((closed_tb['pnl'] > 0).sum())
        win_rate = float(profitable_trades / total_trades * 100) if total_trades > 0 else 0.0

        initial_capital = float(self.strategy.config.get('capital', self.strategy.config.get('initial_capital', 1000000)))
        return_on_initial = total_pnl / initial_capital * 100.0
        return_on_initial_pat = (total_pat / initial_capital * 100.0) if initial_capital else 0.0

        equity_points = closed_tb.groupby('exit_timestamp')['pnl'].sum().sort_index()
        equity = equity_points.cumsum() + initial_capital
        equity.index = pd.to_datetime(equity.index)


        if not equity.empty:
            first_idx = equity.index.min()
            prepend_idx = first_idx - pd.Timedelta(days=1)
            eq_prepend = pd.Series([initial_capital], index=[prepend_idx])
            equity = pd.concat([eq_prepend, equity]).sort_index()

        running_max = equity.cummax()
        drawdown_pos = running_max - equity
        drawdown_pct = drawdown_pos / running_max.replace(0, np.nan)
        max_drawdown = float(drawdown_pos.max()) if not drawdown_pos.empty else 0.0
        max_drawdown_pct = float(drawdown_pct.max() * 100) if not drawdown_pct.empty and not np.isnan(drawdown_pct.max()) else 0.0

      
        drawdown_plot = -drawdown_pos

        daily_pnl = closed_tb.copy()
        daily_pnl['exit_date'] = pd.to_datetime(daily_pnl['exit_timestamp']).dt.date
        daily_returns = daily_pnl.groupby('exit_date')['pnl'].sum() / initial_capital
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        else:
            sharpe = 0.0


        try:
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    name='Equity',
                    line=dict(color='green')
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=drawdown_plot.index,
                    y=drawdown_plot.values,
                    name='Drawdown',
                    line=dict(color='red')
                ),
                row=2,
                col=1
            )

            fig.update_yaxes(title_text='Equity', row=1, col=1)
            fig.update_yaxes(title_text='Drawdown', row=2, col=1, zeroline=True, zerolinewidth=1)
            fig.update_xaxes(title_text='Time', row=2, col=1)

            fig.update_layout(title='Equity and Drawdown', hovermode='x unified', showlegend=True, height=700)

            html_path = report_dir / 'equity_drawdown.html'
            fig.write_html(str(html_path), include_plotlyjs='cdn')
            logger.info(f"Equity/drawdown plot saved to {html_path}")
        except Exception as e:
            logger.warning(f"Plotly not available or failed to generate plot: {e}")

        rows = []
        for symbol in closed_tb['symbol'].unique():
            sym_rows = closed_tb[closed_tb['symbol'] == symbol]
            rows.append({
                'symbol': symbol,
                'closed_trades': len(sym_rows),
                'realized_pnl': float(sym_rows['pnl'].sum()),
                'charges': float(sym_rows['charges'].sum()) if 'charges' in sym_rows.columns else 0.0,
                'pat': float(sym_rows['pat'].sum()) if 'pat' in sym_rows.columns else float(sym_rows['pnl'].sum())
            })
        pnl_summary = pd.DataFrame(rows)
        pnl_summary_path = report_dir / 'pnl_summary.csv'
        pnl_summary.to_csv(pnl_summary_path, index=False)

        report = {
            'total_trades': int(total_trades),
            'profitable_trades': int(profitable_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl_gross': round(total_pnl, 3),
            'total_charges': round(total_charges, 3),
            'total_pat': round(total_pat, 3),
            'initial_capital': round(initial_capital, 2),
            'return_on_initial_capital_gross_pct': round(return_on_initial, 4),
            'return_on_initial_capital_pat_pct': round(return_on_initial_pat, 4),
            'max_drawdown': round(max_drawdown, 3),
            'max_drawdown_pct': round(max_drawdown_pct, 4),
            'sharpe_ratio': round(sharpe, 4)
        }

        report_json_path = report_dir / f"report.json"
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {report_json_path}")

        logger.info("Backtest Report:")
        for metric, value in report.items():
            logger.info(f"{metric}: {value}")
