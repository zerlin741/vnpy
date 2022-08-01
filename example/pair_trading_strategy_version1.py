from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from vnpy.app.portfolio_strategy import StrategyTemplate, StrategyEngine
from vnpy.trader.utility import BarGenerator, extract_vt_symbol, update_ctaposdetails
from vnpy.trader.object import TickData, BarData, TradeData, PositionData, Direction, CtaPostionDetails, Offset
from vnpy.trader.constant import OrderPriceType

class PairTradingStrategy(StrategyTemplate):
    active_vt_symbol = 'T2003.CFFEX'
    passive_vt_symbol = 'TF2003.CFFEX'
    pos_details = {}
    max_pos = 100
    lot_size = 1
    win_tick = 0

    parameters = [
        'active_vt_symbol',
        'passive_vt_symbol',
        'max_pos',
        'win_tick'
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: List[str],
        setting: dict
    ):
        """"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        self.bgs: Dict[str, BarGenerator] = {}
        self.targets: Dict[str, int] = {}
        self.last_tick_time: datetime = None

        self.spread_count: int = 0
        self.spread_data: np.array = np.zeros(100)

        # Obtain contract info
        self.leg1_symbol, self.leg2_symbol = vt_symbols

        def on_bar(bar: BarData):
            """"""
            pass

        for vt_symbol in self.vt_symbols:
            self.targets[vt_symbol] = 0
            self.bgs[vt_symbol] = BarGenerator(on_bar)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.pos_details[self.active_vt_symbol] = CtaPostionDetails(
            extract_vt_symbol(self.active_vt_symbol)[0],
            extract_vt_symbol(self.active_vt_symbol)[1]
        )
        self.pos_details[self.passive_vt_symbol] = CtaPostionDetails(
            extract_vt_symbol(self.passive_vt_symbol)[0],
            extract_vt_symbol(self.passive_vt_symbol)[1]
        )
        self.load_bars(1)
        self.history_pos_details = []

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")


    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.history_pos_details = pd.DataFrame(self.history_pos_details)
        self.history_pos_details.set_index('datetime', inplace=True)
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        if (
            self.last_tick_time
            and self.last_tick_time.minute != tick.datetime.minute
        ):
            bars = {}
            for vt_symbol, bg in self.bgs.items():
                bars[vt_symbol] = bg.generate()
            self.on_bars(bars)

        bg: BarGenerator = self.bgs[tick.vt_symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime

    def on_trade(self, trade: TradeData):
        self.pos_details[trade.vt_symbol] = update_ctaposdetails(self.pos_details[trade.vt_symbol], trade)
        if trade.vt_symbol == self.active_vt_symbol:
            self.active_trade_price_status.update({trade.direction: trade})

    def on_bars(self, bars: Dict[str, BarData]):
        """"""
        active_bar = bars.get(self.active_vt_symbol)
        passive_bar = bars.get(self.passive_vt_symbol)
        if active_bar is None or passive_bar is None:
            return
        active_long_pos = self.pos_details[self.active_vt_symbol].long_pos
        active_short_pos = self.pos_details[self.active_vt_symbol].short_pos
        passive_long_pos = self.pos_details[self.passive_vt_symbol].long_pos
        passive_short_pos = self.pos_details[self.passive_vt_symbol].short_pos
        if active_bar.gateway_name == "OLD":
            current_dt = passive_bar.datetime
        else:
            current_dt = active_bar.datetime
        self.history_pos_details.append({'datetime': datetime.combine(current_dt.date(), current_dt.time()), 'active_long': active_long_pos.volume, 'active_short': active_short_pos.volume,
                                         'passive_long': passive_long_pos.volume, 'passive_short': passive_short_pos.volume})
        print(active_long_pos.volume, active_short_pos.volume, passive_long_pos.volume, passive_short_pos.volume)

        active_pricetick = self.strategy_engine.priceticks[self.active_vt_symbol]
        self.cancel_all()
        # if active_long_pos.volume != 0 and active_short_pos.volume != 0:
        #     raise ValueError()

        if active_long_pos.volume == 0 and active_short_pos.volume == 0:
            self.buy(self.active_vt_symbol, active_bar.bid_price_1, self.lot_size, gateway_name=OrderPriceType.MarketMaking)
            self.short(self.active_vt_symbol, active_bar.ask_price_1, self.lot_size, gateway_name=OrderPriceType.MarketMaking)

        if active_long_pos.volume != 0:
            target_price = sum(active_long_pos.price_per[-self.lot_size:])/self.lot_size + self.win_tick*active_pricetick
            if active_bar.bid_price_1 >= target_price:
                self.sell(self.active_vt_symbol, active_bar.bid_price_1, self.lot_size, gateway_name=OrderPriceType.StopWinClose)
            else:
                if active_long_pos.volume < self.max_pos:
                    self.buy(self.active_vt_symbol, active_bar.bid_price_1, self.lot_size, gateway_name=OrderPriceType.MarketMaking)
                    # self.sell(self.active_vt_symbol, target_price, self.lot_size, gateway_name=OrderPriceType.PassiveClose)
                elif active_long_pos.volume >= self.max_pos:
                    self.sell(self.active_vt_symbol, active_bar.ask_price_1, self.lot_size, gateway_name=OrderPriceType.PassiveClose)

        if active_short_pos.volume != 0:
            if active_bar.ask_price_1+self.win_tick*active_pricetick<= sum(active_short_pos.price_per[-self.lot_size:])/self.lot_size:
                self.cover(self.active_vt_symbol, active_bar.ask_price_1, self.lot_size, gateway_name=OrderPriceType.StopWinClose)
            else:
                if active_short_pos.volume < self.max_pos:
                    self.cover(self.active_vt_symbol, active_bar.bid_price_1, self.lot_size, gateway_name=OrderPriceType.PassiveClose)
                    self.short(self.active_vt_symbol, active_bar.ask_price_1, self.lot_size, gateway_name=OrderPriceType.MarketMaking)
                elif active_short_pos.volume >= self.max_pos:
                    self.cover(self.active_vt_symbol, active_bar.bid_price_1, self.lot_size, gateway_name=OrderPriceType.PassiveClose)
        if passive_long_pos.volume > active_short_pos.volume:
            self.sell(self.passive_vt_symbol, passive_bar.bid_price_1, passive_long_pos.volume-active_short_pos.volume, gateway_name=OrderPriceType.Hedging)
        elif passive_long_pos.volume < active_short_pos.volume:
            self.buy(self.passive_vt_symbol, passive_bar.ask_price_1, abs(passive_long_pos.volume-active_short_pos.volume), gateway_name=OrderPriceType.Hedging)
        if passive_short_pos.volume > active_long_pos.volume:
            self.cover(self.passive_vt_symbol, passive_bar.ask_price_1, passive_short_pos.volume-active_long_pos.volume, gateway_name=OrderPriceType.Hedging)
        elif passive_short_pos.volume < active_long_pos.volume:
            self.short(self.passive_vt_symbol, passive_bar.bid_price_1, abs(passive_short_pos.volume-active_long_pos.volume), gateway_name=OrderPriceType.Hedging)



