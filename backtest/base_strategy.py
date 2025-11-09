from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    limit_price: Optional[float] = None
    
@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    price: float
    quantity: int
    side: OrderSide

class BaseStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.subscribed_instruments: List[str] = []

    def subscribe(self, underlying: str, futures: bool = True, options: bool = True, spot: bool = True) -> Dict:
        
        return {
            'underlying': underlying,
            'FUT': bool(futures),
            'OPT': bool(options),
            'SPOT': bool(spot)
        }

    def on_start(self, date: datetime) -> List[str]:
        
        raise NotImplementedError

    def on_data(self, timestamp: datetime, data: Dict[str, Dict]) -> None:
        
        raise NotImplementedError

    def on_trade_update(self, order: Order, fill_price: float, timestamp: datetime) -> None:
         
        if order.symbol not in self.positions:
            if order.side == OrderSide.BUY:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    entry_time=timestamp
                )
            else:  
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=-order.quantity,
                    entry_price=fill_price,
                    entry_time=timestamp
                )
        else:
            current_pos = self.positions[order.symbol]
            if order.side == OrderSide.BUY:
                current_pos.quantity += order.quantity
            else:  
                current_pos.quantity -= order.quantity

            if current_pos.quantity == 0:
                del self.positions[order.symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
    
        return self.positions.copy()