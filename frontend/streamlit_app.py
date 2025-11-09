import time
import json
from pathlib import Path
import requests
import streamlit as st
from datetime import datetime, timedelta

API_URL = 'http://localhost:8000'
ROOT = 'c:/Users/kakad/OneDrive/Documents/Quant_Projects/opt_backtestser'

st.set_page_config(layout="wide")
st.title('Backtest Runner')

try:
    r = requests.get(f"{API_URL}/strategies")
    r.raise_for_status()
    strategies = r.json().get('strategies', [])
except Exception as e:
    st.error(f"Failed to contact backend: {e}")
    strategies = []

strategy = st.selectbox('Strategy', [''] + strategies)

if strategy:
    try:
        r = requests.get(f"{API_URL}/strategy/{strategy}/config")
        r.raise_for_status()
        config = r.json()
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        config = None
    
    if config:
        st.session_state['config'] = config
        st.session_state['original_config'] = config.copy()  

    config = st.session_state.get('config')
    if config:
        st.subheader('Configuration')
        
        edited = config.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write('**Strategy Settings**')
            edited['underlying'] = st.text_input('Underlying', value=config.get('underlying', 'NIFTY'), key='underlying', help='e.g., NIFTY')
            edited['strategy_name'] = st.text_input('Strategy Name', value=config.get('strategy_name', 'short_straddle_delta_hedge'), key='strategy_name')
            edited['strike_interval'] = st.number_input('Strike Interval', value=int(config.get('strike_interval', 50)), step=10, key='strike_interval')
            edited['lot_size'] = st.number_input('Lot Size', value=int(config.get('lot_size', 75)), step=1, key='lot_size')
        
        with col2:
            st.write('**Capital & Hedging**')
            edited['capital'] = st.number_input('Initial Capital', value=int(config.get('capital', 1000000)), step=100000, key='capital')
            edited['hedge_frequency_minute'] = st.number_input('Hedge Frequency (min)', value=int(config.get('hedge_frequency_minute', 15)), step=1, key='hedge_frequency', min_value=1)
            edited['vix_filter'] = st.checkbox('Enable VIX Filter', value=config.get('vix_filter', False), key='vix_filter')
            if edited['vix_filter']:
                edited['vix_threshold'] = st.number_input('VIX Threshold', value=int(config.get('vix_threshold', 13)), step=1, key='vix_threshold')
            else:
                edited['vix_threshold'] = config.get('vix_threshold', 13)
        
        with col3:
            st.write('**Timing & Other**')
            edited['timeframe'] = st.text_input('Timeframe', value=config.get('timeframe', '1min'), key='timeframe')
            edited['trading_start_time'] = st.text_input('Trading Start Time', value=config.get('trading_start_time', '09:30:00'), key='trading_start_time', help='HH:MM:SS format')
            edited['trading_end_time'] = st.text_input('Trading End Time', value=config.get('trading_end_time', '15:15:00'), key='trading_end_time', help='HH:MM:SS format')
            edited['square_off_time'] = st.text_input('Square Off Time', value=config.get('square_off_time', '15:20:00'), key='square_off_time', help='HH:MM:SS format')
        
        with st.expander('Advanced Settings'):
            edited['quantity'] = st.number_input('Quantity', value=int(config.get('quantity', 10)), step=1, key='quantity')
            st.info('Other config fields (if any) will be preserved')
            st.write('**Current config keys:**', list(config.keys()))
        
        st.subheader('Backtest Date Range')
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            default_start = datetime(2025, 1, 1)
            start_date = st.date_input('Start Date', value=default_start, key='start_date')
        
        with date_col2:
            default_end = datetime(2025, 1, 31)
            end_date = st.date_input('End Date', value=default_end, key='end_date')
        
        
        st.subheader('Actions')
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button('Save config'):
                try:
                    r = requests.put(f"{API_URL}/strategy/{strategy}/config", json=edited)
                    r.raise_for_status()
                    st.success('✓ Config saved')
                    st.session_state['config'] = edited
                    st.session_state['original_config'] = edited.copy()
                except Exception as e:
                    st.error(f"Failed to save config: {e}")
        
        with btn_col2:
            if st.button('Reset to saved'):
                st.session_state['config'] = st.session_state.get('original_config', config).copy()
                st.rerun()
        
        with btn_col3:
            if st.button('Start backtest'):
                 
                if start_date > end_date:
                    st.error('Start date must be before end date')
                else:
                    try:
                        payload = {
                            'strategy': strategy,
                            'config': edited,
                            'start_date': start_date.isoformat(),
                            'end_date': end_date.isoformat()
                        }
                        r = requests.post(f"{API_URL}/start", json=payload)
                        r.raise_for_status()
                        run_id = r.json().get('run_id')
                        st.session_state['run_id'] = run_id
                        st.session_state['backtest_started'] = True
                        st.session_state['log_lines'] = []  
                        st.session_state['last_log_timestamp'] = None  
                        st.session_state['log_start_offset'] = 0 
                        st.session_state['current_run_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
                        st.success(f'✓ Started run {run_id}')
                    except Exception as e:
                        st.error(f"Failed to start backtest: {e}")

if st.session_state.get('backtest_started'):
    run_id = st.session_state.get('run_id')
    if run_id:
        col1, col2, col3 = st.columns([6, 2, 2])
        with col1:
            st.write("**Status:** ▶️ Backtest in progress...")
        with col2:
            if st.button('🛑 Stop', key=f'stop_{run_id}', help='Stop the current backtest'):
                try:
                    stop_resp = requests.post(f"{API_URL}/runs/{run_id}/stop")
                    st.session_state['backtest_started'] = False
                    
                    st.session_state['log_lines'] = []
                    st.session_state['last_log_timestamp'] = None
                    st.session_state['log_start_offset'] = 0
                    st.warning('⚠️ Backtest stopped')
                except Exception as e:
                    st.error(f'Error stopping backtest: {e}')
        
        with st.expander(f'📋 Logs - {run_id[:8]}...', expanded=False):
            
            log_placeholder = st.empty()
            
            current_run_timestamp = st.session_state.get('current_run_start_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            offset = st.session_state.get('log_start_offset', 0)
            lines_all = st.session_state.get('log_lines', [])
            last_timestamp_seen = st.session_state.get('last_log_timestamp', None)
            
            while st.session_state.get('backtest_started'):
                try:
                    r = requests.get(f"{API_URL}/runs/{run_id}/log", params={'offset': offset})
                    r.raise_for_status()
                    data = r.json()
                    new_lines = data.get('lines', [])
                    offset = data.get('offset', offset)
                    
                    if new_lines:
                        
                        newly_added = []
                        for line in new_lines:
                            
                            try:
                                if len(line) >= 19: 
                                    line_timestamp = line[:19] 
                                    if line_timestamp >= current_run_timestamp:
                                       
                                        if last_timestamp_seen is None or line_timestamp >= last_timestamp_seen:
                                            if line not in lines_all: 
                                                newly_added.append(line)
                                                last_timestamp_seen = line_timestamp
                            except Exception:
                                
                                pass
                        
                        if newly_added:
                            lines_all.extend(newly_added)
                            st.session_state['log_lines'] = lines_all
                            st.session_state['last_log_timestamp'] = last_timestamp_seen
                            st.session_state['log_start_offset'] = offset
                        
                        log_text = '\n'.join(lines_all)
                        with log_placeholder.container(border=True):
                            st.markdown(
                                f"""
                                <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f8f8f8; border-radius: 5px; font-family: monospace; font-size: 11px; word-wrap: break-word; white-space: pre-wrap;">
                                {log_text}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    s = requests.get(f"{API_URL}/runs/{run_id}/status").json()
                    if s.get('status') == 'finished':
                        st.session_state['backtest_started'] = False
                        st.success('✓ Backtest completed successfully!')
                        
                        strategy_path = Path(f"{ROOT}/strategies/{strategy}")
                        report_dir = max((strategy_path / 'report').glob('*'), key=lambda x: x.stat().st_mtime)
                        
                        try:
                            with open(report_dir / 'report.json', 'r') as f:
                                report = json.load(f)
                                
                            st.subheader('📊 Backtest Results')
                            metric_cols = st.columns(4)
                            
                            with metric_cols[0]:
                                st.metric('Return on Capital', 
                                    f"{report['return_on_initial_capital_pat_pct']:.2f}%",
                                    delta=f"{report['return_on_initial_capital_gross_pct']:.2f}% (Gross)")
                                
                            with metric_cols[1]:
                                st.metric('Total P&L', 
                                    f"₹{report['total_pat']:,.2f}",
                                    delta=f"Charges: ₹{report['total_charges']:,.2f}")
                                
                            with metric_cols[2]:
                                st.metric('Max Drawdown',
                                    f"₹{report['max_drawdown']:,.2f}",
                                    f"{report['max_drawdown_pct']:.2f}%",
                                    delta_color="inverse")
                                
                            with metric_cols[3]:
                                st.metric('Sharpe Ratio',
                                    f"{report['sharpe_ratio']:.2f}",
                                    f"Win Rate: {report['win_rate']:.1f}%")
                            
                            if (report_dir / 'equity_drawdown.html').exists():
                                st.subheader('📈 Equity Curve')
                                with open(report_dir / 'equity_drawdown.html', 'r') as f:
                                    html_content = f.read()
                                    st.components.v1.html(html_content, height=600)
                            
                            with st.expander('🔍 Detailed Metrics', expanded=False):
                                st.json(report)
                            
                            
                            try:
                                if (report_dir / 'tradebook.csv').exists():
                                    st.subheader('📒 Trade History')
                                    
                                    import pandas as pd
                                    trades_df = pd.read_csv(report_dir / 'tradebook.csv')
                                    
                                    st.download_button(
                                        label="📥 Download Trades CSV",
                                        data=trades_df.to_csv(index=False),
                                        file_name='tradebook.csv',
                                        mime='text/csv',
                                    )
                                    
                                    st.dataframe(
                                        trades_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "entry_time": st.column_config.DatetimeColumn("Entry Time", format="DD-MM-YY HH:mm"),
                                            "exit_time": st.column_config.DatetimeColumn("Exit Time", format="DD-MM-YY HH:mm"),
                                            "entry_price": st.column_config.NumberColumn("Entry Price", format="₹%.2f"),
                                            "exit_price": st.column_config.NumberColumn("Exit Price", format="₹%.2f"),
                                            "pnl": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                                            "quantity": st.column_config.NumberColumn("Quantity", format="%d"),
                                        }
                                    )
                            except Exception as e:
                                st.error(f"Error loading tradebook: {e}")
                            
                        except Exception as e:
                            st.error(f"Error loading report: {e}")
                        
                        break
                except Exception as e:
                    st.error(f"Error fetching logs: {e}")
                    st.session_state['backtest_started'] = False
                    break
                
                time.sleep(0.8)