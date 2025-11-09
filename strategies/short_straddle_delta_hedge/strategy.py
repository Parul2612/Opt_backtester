from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from backtest.base_strategy import BaseStrategy, Order, OrderType, OrderSide
import math
import mibian  
import logging

logger = logging.getLogger(__name__)

class ShortStraddleDeltaHedge(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.atm_strike: Optional[float] = None
        self.current_future: Optional[str] = None
        self.current_call: Optional[str] = None
        self.current_put: Optional[str] = None
        self.hedge_threshold = 0.01  
        self.position_taken = False
        self.last_hedge_time = None
        self.next_hedge_time = None
        self.entry_time = time(9, 30) 
        self.exit_time = time(15, 15)  
        self.hedge_frequency = timedelta(minutes=15) 
        self._current_time = None
        self.vix_filter = config.get('vix_filter', False)
        self.vix_threshold = config.get('vix_threshold', 13)
        
        self.risk_free_rate = 10  
        self.volatility = 12  
        

    def on_start(self, date: datetime) -> List[str]:
     
        
        self.atm_strike = None
        self.current_future = None
        self.current_call = None
        self.current_put = None
        self.position_taken = False
        self.day_end = False
        self.hedge_frequency = timedelta(minutes= self.config['hedge_frequency_minute'])
        self.opt_qty = self.config['quantity']
        
        spec = self.subscribe(self.config['underlying'], futures=True, options=True, spot=True)
        self.subscribed_instruments = [spec]
        
        if self.vix_filter:
            vix_spec = self.subscribe("INDIA VIX", spot=True)
            self.subscribed_instruments.append(vix_spec)
            return [spec, vix_spec]
        
        return [spec]
        
    def on_data(self, timestamp: datetime, data: Dict[str, Dict]) -> None:
        
        if timestamp.weekday() != 3:  
            return

        current_time = timestamp.time()
        self._current_time = timestamp
        if current_time < self.entry_time or current_time > self.exit_time:
            return
            
        if self.vix_filter and not self.position_taken:
            vix_key = None
            try:
                vix_key = self.backtester.get_spot_key("INDIA VIX")
            except Exception:
                return
                
            if not vix_key or vix_key not in data:
                return
                
            vix_value = data[vix_key]['close']
            if vix_value < self.vix_threshold:
                return 
        if current_time >= self.exit_time and not self.day_end:
            self.backtester._square_off_all(timestamp)
            self.position_taken = False
            self.day_end = True
            self.last_hedge_time = None
            self.next_hedge_time = None
            return
            
        underlying = self.config['underlying']
        spot_key = None
        try:
            spot_key = self.backtester.get_spot_key(underlying)
        except Exception:
            return

        if not spot_key or spot_key not in data:
            return
            
        spot_price = data[spot_key]['close']

        if current_time == self.entry_time and not self.position_taken:
            
            mapping = self.backtester.get_available_option_contracts(underlying, as_of_date=timestamp.date())
           
            expiries = sorted(mapping.keys())
            if not expiries:
                return
                
            chosen_expiry = expiries[0]
            strikes_map = mapping.get(chosen_expiry, {})
            
            strikes = [float(s) for s in strikes_map.keys()]
            if not strikes:
                return
                
            self.atm_strike = min(strikes, key=lambda s: abs(s - spot_price))
            
            pair = strikes_map.get(str(self.atm_strike), {})
            self.current_call = pair.get('CE')
            self.current_put = pair.get('PE')

            for k in self.backtester.current_data.keys():
                dfk = self.backtester.current_data[k]
                try:
                    inst = dfk['instrument'].iloc[0] if 'instrument' in dfk.columns else None
                    und_val = dfk['underlying'].iloc[0] if 'underlying' in dfk.columns else None
                    if inst == 'FUT' and str(und_val).upper() == underlying.upper():
                        self.current_future = k
                        break
                except Exception:
                    continue

            if self.current_call and self.current_put and self.current_future:
                lot_size = self.config['lot_size']
                orders = self._place_straddle(timestamp, self.opt_qty * lot_size)
                for order in orders:
                    self.backtester.place_order(order, timestamp)
                self.position_taken = True
                self.last_hedge_time = timestamp
                self.next_hedge_time = timestamp + self.hedge_frequency

        if self.position_taken and timestamp >= self.next_hedge_time:
            net_delta = self._calculate_position_delta(data)
            hedge_order = self._adjust_hedge(timestamp, net_delta, data)
            if hedge_order:
                self.backtester.place_order(hedge_order, timestamp)
            self.last_hedge_time = timestamp
            self.next_hedge_time = timestamp + self.hedge_frequency


    def _get_days_to_expiry(self,
                                current_time: datetime,
                                expiry: datetime.date,
                                expiry_time: time = time(15, 30),
                                include_today: bool = True,
                                method: str = "ceil") -> int:
        
        SECONDS_PER_DAY = 24 * 60 * 60

        expiry_dt = datetime.combine(expiry, expiry_time)
        delta_seconds = (expiry_dt - current_time).total_seconds()

        if delta_seconds <= 0:
            return 0

        frac_days = delta_seconds / SECONDS_PER_DAY  
        if current_time.date() == expiry and include_today:
            
            return 1

        if method == "ceil":
            days_int = math.ceil(frac_days)
        elif method == "floor":
            days_int = math.floor(frac_days)
        elif method == "round":
            days_int = round(frac_days)
        else:
            raise ValueError("method must be 'ceil','floor', or 'round'")

        if include_today:
            if expiry > current_time.date():
                days_int += 1

        return int(days_int)

        
    def _calculate_option_greeks(self, 
                               spot: float, 
                               strike: float, 
                               days_to_expiry: float, 
                               option_type: str,
                               market_price: float) -> Tuple[float, float]:
       
           
        if not all(isinstance(x, (int, float)) for x in [spot, strike, days_to_expiry, market_price]):
            logger.warning("Invalid input types for Greeks calculation")
            return 0.0, self.volatility
        
        if any(x <= 0 for x in [spot, strike, days_to_expiry, market_price]):
            logger.warning(f"Invalid values for Greeks calculation (must be positive) spot :{spot}, strike: {strike}, days_to_expiry: {days_to_expiry}, market_price: {market_price}")
            return 0.0, self.volatility
        
        try:
            spot = float(spot)
            strike = float(strike)
            days_to_expiry = float(days_to_expiry)
            market_price = float(market_price)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting inputs to float: {e}")
            return 0.0, self.volatility
        
        try:
            bs_args = [spot, strike, self.risk_free_rate, days_to_expiry]
            if option_type == 'C':
                bs = mibian.BS(bs_args, callPrice=market_price)
            else:
                bs = mibian.BS(bs_args, putPrice=market_price)
            
            try:
                imp_vol = float(bs.impliedVolatility)
                if not 0.0 <= imp_vol <= 500.0:  
                    raise ValueError("Implied vol out of reasonable range")
            except Exception as e:
                logger.warning(f"Error calculating implied vol: {e}. Using default.")
                imp_vol = self.volatility
            
            bs = mibian.BS(bs_args, volatility=imp_vol)
            delta = bs.callDelta if option_type == 'C' else bs.putDelta
        
            if not isinstance(delta, (int, float)) or abs(delta) > 1:
                raise ValueError("Delta out of valid range")
            
            return float(delta), float(imp_vol)
        
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {e}. Using default values.")
            try:
                bs = mibian.BS([spot, strike, self.risk_free_rate, days_to_expiry], volatility=self.volatility)
                delta = bs.callDelta if option_type == 'C' else bs.putDelta
                if isinstance(delta, (int, float)) and abs(delta) <= 1:
                    return float(delta), self.volatility
            except Exception as e2:
                logger.warning(f"Error using default values: {e2}")
        
            return 0.0, self.volatility
        
    def _find_atm_strike(self, spot_price: float) -> float:

        strike_interval = self.config['strike_interval']
        return round(spot_price / strike_interval) * strike_interval
        
    def _calculate_position_delta(self, data: Dict[str, Dict]) -> float:

        net_delta = 0.0
        
        underlying = self.config['underlying']
        spot_key = self.backtester.get_spot_key(underlying)
        if not spot_key or spot_key not in data:
            return 0.0
            
        spot_price = float(data[spot_key]['close'])
        
        mapping = self.backtester.get_available_option_contracts(underlying)
        if not mapping:
            return 0.0
            
        expiries = sorted(mapping.keys())
        if not expiries:
            return 0.0
            
        current_expiry = expiries[0]  
        days_to_expiry = self._get_days_to_expiry(self._current_time, current_expiry)
        
        if self.current_call in self.positions and self.current_call in data:
            call_price = float(data[self.current_call]['close'])
            call_delta, call_vol = self._calculate_option_greeks(
                spot=spot_price,
                strike=self.atm_strike,
                days_to_expiry=days_to_expiry,
                option_type='C',
                market_price=call_price
            )
            net_delta += -call_delta * self.positions[self.current_call].quantity
            
            self.volatility = call_vol
            
        if self.current_put in self.positions and self.current_put in data:
            put_price = float(data[self.current_put]['close'])
            put_delta, put_vol = self._calculate_option_greeks(
                spot=spot_price,
                strike=self.atm_strike,
                days_to_expiry=days_to_expiry,
                option_type='P',
                market_price=put_price
            )
            
            net_delta += -put_delta * self.positions[self.current_put].quantity
            
        if self.current_future in self.positions:
            net_delta += 1.0 * self.positions[self.current_future].quantity
            
        return net_delta

    def _place_straddle(self, timestamp: datetime, quantity: int) -> List[Order]:

        
        call_order = Order(
            symbol=self.current_call,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=quantity
        )
        
        put_order = Order(
            symbol=self.current_put,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=quantity
        )
        
        return [call_order, put_order]

    def _adjust_hedge(self, timestamp: datetime, net_delta: float, data: Dict[str, Dict]) -> Optional[Order]:

        if self.current_future not in data:
            return None

        try:
            net_delta_val = float(net_delta)
        except Exception:
            logger.warning(f"Net delta not a real number ({net_delta}), skipping hedge adjustment")
            return None

        if not math.isfinite(net_delta_val):
            logger.warning(f"Net delta is not finite ({net_delta_val}), skipping hedge adjustment")
            return None

        lot_size = int(self.config.get('lot_size', 1)) if self.config.get('lot_size') else 1
        if lot_size <= 0:
            logger.warning(f"Invalid lot_size in config: {lot_size}. Skipping hedge adjustment")
            return None
        
        required_futures_underlying = -net_delta_val  
        required_futures_lots = round(required_futures_underlying / lot_size)
        
        required_futures_quantity = required_futures_lots * lot_size
        MAX_FUTURES_QUANTITY = 10_000_000  
        if abs(required_futures_quantity) > MAX_FUTURES_QUANTITY:
            logger.warning(f"Calculated required_futures_quantity {required_futures_quantity} exceeds cap {MAX_FUTURES_QUANTITY}. Skipping hedge adjustment")
            return None
        
        current_futures = 0
        if self.current_future in self.positions:
            current_futures = self.positions[self.current_future].quantity
        
        futures_quantity_diff = required_futures_quantity - current_futures
        
        if futures_quantity_diff != 0:

            side = OrderSide.BUY if futures_quantity_diff > 0 else OrderSide.SELL
            return Order(
                symbol=self.current_future,
                order_type=OrderType.MARKET,
                side=side,
                quantity=abs(futures_quantity_diff)
            )
        return None