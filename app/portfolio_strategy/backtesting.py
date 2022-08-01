from collections import defaultdict
from datetime import date, datetime, timedelta, time
from typing import Dict, List, Set, Tuple
from functools import lru_cache
from copy import copy
import traceback

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame

from vnpy.trader.constant import Direction, Offset, Interval, Status, OrderPriceType
from vnpy.trader.database import database_manager
from vnpy.trader.object import OrderData, TradeData, TickData, TickData, CtaPostionDetails
from vnpy.trader.utility import round_to, extract_vt_symbol, update_ctaposdetails

from .template import StrategyTemplate


INTERVAL_DELTA_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


class BacktestingEngine:
    """"""

    gateway_name = "BACKTESTING"

    def __init__(self):
        """"""
        self.vt_symbols: List[str] = []
        self.start: datetime = None
        self.end: datetime = None

        self.rates: Dict[str, float] = 0
        self.slippages: Dict[str, float] = 0
        self.sizes: Dict[str, float] = 1
        self.priceticks: Dict[str, float] = 0

        self.capital: float = 1_000_000
        self.risk_free: float = 0.02

        self.strategy: StrategyTemplate = None
        self.bars: Dict[str, TickData] = {}
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0
        self.history_data: Dict[Tuple, TickData] = {}
        # self.dts: Set[datetime] = set()
        self.dts = defaultdict(set)
        self.lastest_data = {}

        self.limit_order_count = 0
        self.limit_orders = {}
        self.active_limit_orders = {}

        self.trade_count = 0
        self.trades = {}

        self.logs = []

        self.daily_results = {}
        self.daily_df = None

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.bars = {}
        self.datetime = None

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()
        self.pnl_per_trade = []

        self.logs.clear()
        self.daily_results.clear()
        self.daily_df = None

    def set_parameters(
        self,
        vt_symbols: List[str],
        interval: Interval,
        start: datetime,
        rates: Dict[str, float],
        slippages: Dict[str, float],
        sizes: Dict[str, float],
        priceticks: Dict[str, float],
        capital: int = 0,
        end: datetime = None,
        risk_free: float = 0
    ) -> None:
        """"""
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.rates = rates
        self.slippages = slippages
        self.sizes = sizes
        self.priceticks = priceticks

        self.start = start
        self.end = end
        self.capital = capital
        self.risk_free = risk_free

    def add_strategy(self, strategy_class: type, setting: dict) -> None:
        """"""
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )

    def load_data(self) -> None:
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # Clear previously loaded history data
        self.history_data.clear()
        self.dts.clear()
        market_open_time_func = lambda x: time(9,15) if x<date(2020,7,20) else time(9,30)

        # Load 30 days of data each time and allow for progress update
        progress_delta = timedelta(days=30)
        total_delta = self.end - self.start
        interval_delta = INTERVAL_DELTA_MAP[self.interval]

        for vt_symbol in self.vt_symbols:
            start_bar_time , last_bar_time = None, None
            start = self.start
            end = self.start + progress_delta
            progress = 0

            data_count = 0
            while start < self.end:
                end = min(end, self.end)  # Make sure end time stays within set range

                data = load_tick_data(
                    vt_symbol,
                    self.interval,
                    start,
                    end
                )

                for bar in data:
                    if bar.datetime.time() >= market_open_time_func(bar.datetime.date()) and bar.datetime.time() <= time(11,30):
                        label = bar.datetime.date().strftime("%Y%m%d.am")
                    elif bar.datetime.time() >= time(13,0) and bar.datetime.time() <= time(15,15):
                        label = bar.datetime.date().strftime("%Y%m%d.pm")
                    else:
                        label = False
                    if label:
                        if self.lastest_data.get(vt_symbol) is None:
                            self.lastest_data.update({vt_symbol: [label, bar]})
                        else:
                            last_label = self.lastest_data.get(vt_symbol)
                            if last_label[0] != label:
                                self.lastest_data.update({vt_symbol: [label, bar]})
                            else:
                                while bar.datetime - self.lastest_data[vt_symbol][1].datetime > timedelta(seconds=0.7):
                                    last_bar = copy(self.lastest_data[vt_symbol][1])
                                    last_bar.datetime += timedelta(seconds=0.5)
                                    self.lastest_data.update({vt_symbol: [label, last_bar]})
                                    self.history_data[(last_bar.datetime, vt_symbol)] = last_bar
                                    self.dts[last_bar.datetime.date()].add(last_bar.datetime)
                                self.lastest_data.update({vt_symbol: [label, bar]})

                        self.dts[bar.datetime.date()].add(bar.datetime)
                        self.history_data[(bar.datetime, vt_symbol)] = bar
                        start_bar_time = start_bar_time or bar.datetime
                        data_count += 1
                        last_bar_time = bar.datetime
                progress += progress_delta / total_delta
                progress = min(progress, 1)
                progress_bar = "#" * int(progress * 10)
                self.output(f"{vt_symbol}加载进度：{progress_bar} [{progress:.0%}]")

                start = end + interval_delta
                end += (progress_delta + interval_delta)

            self.output(f"first bar time is {start_bar_time}, last bar time is {last_bar_time}")

            self.output(f"{vt_symbol}历史数据加载完成，数据量：{data_count}")

        self.output("所有历史数据加载完成")

    def run_backtesting(self, init_pos=None) -> None:
        """"""
        self.strategy.on_init(init_pos)

        # Generate sorted datetime list
        days = [day for day in self.dts]
        days.sort()
        dts = {k: sorted(list(v)) for k,v in self.dts.items()}
        # dts = list(self.dts)
        # dts.sort()

        # Use the first [days] of history data for initializing strategy
        day_count = 0
        ix = 0

        for ix, day in enumerate(days):
            if day_count >= self.days:
                break
            for dt in dts[day]:
                try:
                    self.new_bars(dt)
                except Exception:
                    self.output("触发异常，回测终止")
                    self.output(traceback.format_exc())
                    return
            day_count += 1


        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # Use the rest of history data for running backtesting
        for day in days[ix:]:
            for dt in dts[day]:
                try:
                    self.new_bars(dt)
                except Exception:
                    self.output("触发异常，回测终止")
                    self.output(traceback.format_exc())
                    return
            self.strategy.cancel_all()
            self.strategy.on_daily_end()
        self.update_trading_end(dt)
        self.strategy.on_stop()
        self.output("历史数据回放结束")

    def calculate_result(self) -> None:
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)


        # Calculate daily result by iteration.
        pre_closes = {}
        start_poses = {}
        pre_pos_details = self.strategy.inited_pos_detail
        pnl_per_trade = []

        for d, daily_result in self.daily_results.items():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                pre_pos_details,
                self.sizes,
                self.rates,
                self.slippages,
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses
            pre_pos_details = daily_result.end_pos_details
            pnl_per_trade.extend(daily_result.pnl_per_trade)
        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields = [
                "date", "trade_count", "turnover",
                "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl",
                "readable_end_pos", "order_price_analysis"
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")
        self.pnl_per_trade = pnl_per_trade
        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> None:
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            avg_win = 0
            avg_loss = 0
            win_ratio = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_cancel_order_nums = self.strategy.cancal_order_nums
            daily_cancel_order_nums = total_cancel_order_nums / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * 240
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                daily_risk_free = self.risk_free / np.sqrt(240)
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(240)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_net_pnl / max_drawdown
            if self.pnl_per_trade:
                win_trade = [x for x in self.pnl_per_trade if x>0]
                loss_trade = [x for x in self.pnl_per_trade if x<0]
                win_ratio = len(win_trade)/(len(win_trade)+len(loss_trade))
                avg_win = sum(win_trade)/len(win_trade)
                avg_loss = sum(loss_trade)/len(loss_trade)

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")
            self.output(f"总胜率：\t{win_ratio:,.4f}")
            self.output(f"平均单笔盈利：\t{avg_win:,.2f}")
            self.output(f"平均单笔亏损：\t{avg_loss:,.2f}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")
            self.output(f"日均撤单次数: \t{daily_cancel_order_nums}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "win_ratio": win_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "daily_cancel_order": daily_cancel_order_nums,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")
        return statistics

    def show_chart(self, df: DataFrame = None) -> None:
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def update_daily_close(self, bars: Dict[str, TickData], dt: datetime) -> None:
        """"""
        d = dt.date()

        close_prices = {}
        for bar in bars.values():
            close_prices[bar.vt_symbol] = (bar.ask_price_1 + bar.bid_price_1) / 2
        for v in close_prices.values():
            if v <= 1:
                return
        daily_result = self.daily_results.get(d, None)

        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def update_trading_end(self, dt):
        d = dt.date()
        self.daily_results[d].trading_end = True

    def new_bars(self, dt: datetime) -> None:
        """"""
        # if self.strategy.trading and self.datetime and self.datetime.day != dt.day:
        #     self.update_daily_close(self.bars, self.datetime)
        self.datetime = dt

        # self.bars.clear()
        for vt_symbol in self.vt_symbols:
            bar = self.history_data.get((dt, vt_symbol), None)
            if bar and (bar.ask_price_1 <= 1 or bar.bid_price_1<=1):
                bar = None
            # If bar data of vt_symbol at dt exists
            if bar:
                self.bars[vt_symbol] = bar
            # Otherwise, use previous data to backfill
            elif vt_symbol in self.bars:
                old_bar = self.bars[vt_symbol]

                bar = TickData(
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    ask_price_1=old_bar.ask_price_1,
                    bid_price_1=old_bar.bid_price_1,
                    low_price=old_bar.low_price,
                    high_price=old_bar.high_price,
                    ask_volume_1=old_bar.ask_volume_1,
                    bid_volume_1=old_bar.bid_volume_1,
                    open_price=old_bar.open_price,
                    gateway_name='OLD'
                )
                self.bars[vt_symbol] = bar

        self.cross_limit_order()
        self.strategy.on_bars(self.bars)
        if self.strategy.trading:
            self.update_daily_close(self.bars, dt)

    def cross_limit_order(self) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        for order in list(self.active_limit_orders.values()):
            bar = self.bars[order.vt_symbol]
            if bar.gateway_name == 'OLD':
                return

            # long_cross_price = bar.low_price
            # short_cross_price = bar.high_price
            long_cross_price = bar.ask_price_1
            short_cross_price = bar.bid_price_1
            # long_best_price = bar.open_price
            # short_best_price = bar.open_price

            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push order update with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            # if long_cross:
            #     trade_price = min(order.price, long_best_price)
            # else:
            #     trade_price = max(order.price, short_best_price)
            trade_price = order.price

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=order.gateway_name,
            )
            self.strategy.on_trade(trade, self.bars)
            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def load_bars(
        self,
        strategy: StrategyTemplate,
        days: int,
        interval: Interval
    ) -> None:
        """"""
        self.days = days

    def send_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool,
        gateway_name: str
    ) -> List[str]:
        """"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order
        # print("send limit order", order.symbol, order.direction.value, order.offset.value, order.price, order.gateway_name)

        return [order.vt_orderid]

    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid not in self.active_limit_orders:
            return
        order = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: StrategyTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> List[TradeData]:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> List[OrderData]:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> List["PortfolioDailyResult"]:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


class ContractDailyResult:
    """"""

    def __init__(self, result_date: date, close_price: float):
        """"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: List[TradeData] = []
        self.trade_count: int = 0
        self.order_price_analysis = {}

        self.start_pos: float = 0
        self.end_pos: float = 0
        self.end_pos_details = None

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        pre_pos_detail: CtaPostionDetails,
        size: int,
        rate: float,
        slippage: float,
        trading_end: bool=False
    ) -> None:
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        self.pre_close = pre_close or 0
        leverage = 0.05
        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos
        self.pnl_per_trade = []
        # self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        order_price_analysis = defaultdict(dict)

        for trade in self.trades:
            trading_pnl = 0
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change
            turnover = trade.volume * size * trade.price

            if trade.offset == Offset.CLOSE:
                if trade.direction == Direction.SHORT:
                    trading_pnl = (trade.price*trade.volume - sum(pre_pos_detail.long_pos.price_per[-trade.volume:]))*size
                elif trade.direction == Direction.LONG:
                    trading_pnl = (sum(pre_pos_detail.short_pos.price_per[-trade.volume:]) - trade.price*trade.volume) * size
                self.pnl_per_trade.append(trading_pnl)
            self.trading_pnl += trading_pnl
            order_price_analysis[trade.gateway_name]['count'] = order_price_analysis[trade.gateway_name].get('count', 0) +trade.volume
            order_price_analysis[trade.gateway_name]['pnl'] = order_price_analysis[trade.gateway_name].get('pnl', 0) + trading_pnl

            self.slippage += trade.volume * size * slippage
            self.turnover += turnover
            self.commission += turnover * rate
            pre_pos_detail = update_ctaposdetails(pre_pos_detail, trade)
        # end_pos_value = (pre_pos_detail.long_pos.volume * self.close_price * size \
        #                 + pre_pos_detail.short_pos.volume * self.close_price * size) * leverage
        self.holding_pnl = (self.close_price*pre_pos_detail.long_pos.volume - sum(pre_pos_detail.long_pos.price_per))*size\
                           +(sum(pre_pos_detail.short_pos.price_per)-self.close_price*pre_pos_detail.short_pos.volume)*size
        pre_pos_detail.long_pos.price_per = [self.close_price for x in pre_pos_detail.long_pos.price_per]
        pre_pos_detail.short_pos.price_per = [self.close_price for x in pre_pos_detail.short_pos.price_per]
        self.total_pnl = self.trading_pnl + self.holding_pnl

        # Net pnl takes account of commission and slippage cost
        # self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage
        self.end_pos_details = pre_pos_detail
        self.order_price_analysis = {k: v for k, v in order_price_analysis.items() if v and v['count']>0}

    def update_close_price(self, close_price: float) -> None:
        """"""
        self.close_price = close_price


class PortfolioDailyResult:
    """"""

    def __init__(self, result_date: date, close_prices: Dict[str, float]):
        """"""
        self.date: date = result_date
        self.close_prices: Dict[str, float] = close_prices
        self.pre_closes: Dict[str, float] = {}
        self.start_poses: Dict[str, float] = {}
        self.end_poses: Dict[str, float] = {}
        self.end_pos_details: Dict[str, CtaPostionDetails] = {}

        self.contract_results: Dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(result_date, close_price)

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0
        self.readable_end_pos: str = ''
        self.order_price_analysis: Dict = {}
        self.trading_end = False
        self.pnl_per_trade = []

    def add_trade(self, trade: TradeData) -> None:
        """"""
        contract_result = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: Dict[str, float],
        start_poses: Dict[str, float],
        pre_pos_details: Dict[str, CtaPostionDetails],
        sizes: Dict[str, float],
        rates: Dict[str, float],
        slippages: Dict[str, float],
    ) -> None:
        """"""
        self.pre_closes = pre_closes

        for vt_symbol, contract_result in self.contract_results.items():
            symbol, exchange = extract_vt_symbol(vt_symbol)
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                pre_pos_details.get(vt_symbol, CtaPostionDetails(symbol, exchange)),
                sizes[vt_symbol],
                rates[vt_symbol],
                slippages[vt_symbol],
                self.trading_end
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.slippage += contract_result.slippage
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl
            self.end_pos_details[vt_symbol] = contract_result.end_pos_details
            self.end_poses[vt_symbol] = contract_result.end_pos
            self.readable_end_pos += f'{vt_symbol} long: {contract_result.end_pos_details.long_pos.volume}, short: {contract_result.end_pos_details.short_pos.volume} ;'
            self.order_price_analysis[vt_symbol] = contract_result.order_price_analysis
            self.pnl_per_trade.extend(contract_result.pnl_per_trade)

    def update_close_prices(self, close_prices: Dict[str, float]) -> None:
        """"""
        self.close_prices = close_prices

        for vt_symbol, close_price in close_prices.items():
            contract_result = self.contract_results.get(vt_symbol, None)
            if contract_result:
                contract_result.update_close_price(close_price)


@lru_cache(maxsize=999)
def load_bar_data(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime
):
    """"""
    symbol, exchange = extract_vt_symbol(vt_symbol)

    return database_manager.load_bar_data(
        symbol, exchange, interval, start, end
    )

@lru_cache(maxsize=999)
def load_tick_data(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime
):
    """"""
    symbol, exchange = extract_vt_symbol(vt_symbol)

    return database_manager.load_tick_data(
        symbol, exchange, start, end
    )