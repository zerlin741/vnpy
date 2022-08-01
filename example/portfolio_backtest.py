import os
from datetime import datetime, time
from importlib import reload
import pandas as pd
from pathlib import Path
from tabulate import tabulate

import sys
sys.path.append(os.getcwd())

from tools.format import format_int
from tools.vnpy_util import *
import vnpy.app.portfolio_strategy
reload(vnpy.app.portfolio_strategy)

from vnpy.app.portfolio_strategy import BacktestingEngine
from vnpy.trader.constant import Interval

from strategies.pair_trading_strategy_version1 import PairTradingStrategy as strategy_version1
from strategies.pair_trading_strategy_version2 import PairTradingStrategy as strategy_version2
from strategies.pair_trading_strategy_version3 import PairTradingStrategy as strategy_version3
from strategies.pair_trading_strategy_versionz import PairTradingStrategy as strategy_versionz
from strategies.pair_trading_strategy_version4 import PairTradingStrategy as strategy_version4
from strategies.pair_trading_strategy_version5 import PairTradingStrategy as strategy_version5
from strategies.pair_trading_strategy_version6 import PairTradingStrategy as strategy_version6
from strategies.pair_trading_strategy_version7 import PairTradingStrategy as strategy_version7
from strategies.pair_trading_strategy_version8 import PairTradingStrategy as strategy_version8
from strategies.pair_trading_strategy_version9 import PairTradingStrategy as strategy_version9
from strategies.pair_trading_strategy_version10 import PairTradingStrategy as strategy_version10
from strategies.pair_trading_strategy_version11 import PairTradingStrategy as strategy_version11
from strategies.pair_trading_strategy_version12 import PairTradingStrategy as strategy_version12
from strategies.pair_trading_strategy_version13 import PairTradingStrategy as strategy_version13
from strategies.pair_trading_strategy_version14 import PairTradingStrategy as strategy_version14
from strategies.pair_trading_strategy_version15 import PairTradingStrategy as strategy_version15
from strategies.pair_trading_strategy_version16 import PairTradingStrategy as strategy_version16
from strategies.pair_trading_strategy_version17 import PairTradingStrategy as strategy_version17
from strategies.pair_trading_strategy_version18 import PairTradingStrategy as strategy_version18
from strategies.pair_trading_strategy_version19 import PairTradingStrategy as strategy_version19
from strategies.pair_trading_strategy_version20 import PairTradingStrategy as strategy_version20
from strategies.pair_trading_strategy_version21 import PairTradingStrategy as strategy_version21
from strategies.pair_trading_strategy_version22 import PairTradingStrategy as strategy_version22
from strategies.pair_trading_strategy_version23 import PairTradingStrategy as strategy_version23
from strategies.pair_trading_strategy_version24 import PairTradingStrategy as strategy_version24
from strategies.pair_trading_strategy_version25 import PairTradingStrategy as strategy_version25
from strategies.pair_trading_strategy_version26 import PairTradingStrategy as strategy_version26
from strategies.pair_trading_strategy_version27 import PairTradingStrategy as strategy_version27
from strategies.pair_trading_strategy_version28 import PairTradingStrategy as strategy_version28
from strategies.pair_trading_strategy_version29 import PairTradingStrategy as strategy_version29
from strategies.pair_trading_strategy_version30 import PairTradingStrategy as strategy_version30
from strategies.pair_trading_strategy_version31 import PairTradingStrategy as strategy_version31
from strategies.pair_trading_strategy_version32 import PairTradingStrategy as strategy_version32


"""
TF主力时间：
2006 [20200220 - 20200513]
2009 [20200514 - 20200818]
2012 [20200819 - 20201118]
2103 [20201119 - ]
"""

def run_backtest(active_symbol, passive_symbol, start_date, end_date, bk_params, version):
    params = {
        'active_vt_symbol': f'{active_symbol}CSV.CFFEX',
        'passive_vt_symbol': f'{passive_symbol}CSV.CFFEX'
    }
    symbols = list(params.values())
    start_date = start_date
    end_date = end_date
    params.update({
        'version': version
    })
    params.update(bk_params)
    version = version
    init_pos = {
        params['active_vt_symbol']: {
            'long': {
                'volume': 44,
                'price': 99.9
            }
        },
        params['passive_vt_symbol']: {
            'short': {
                'volume': 45,
                'price': 100.275
            }
        }
    }
    init_pos = False





    strategy_map = {
        'version1': strategy_version1,
        'version2': strategy_version2,
        'version3': strategy_version3,
        'versionz': strategy_versionz,
        'version4': strategy_version4,
        'version5': strategy_version5,
        'version6': strategy_version6,
        'version7': strategy_version7,
        'version8': strategy_version8,
        'version9': strategy_version9,
        'version10': strategy_version10,
        'version11': strategy_version11,
        'version12': strategy_version12,
        'version13': strategy_version13,
        'version14': strategy_version14,
        'version15': strategy_version15,
        'version16': strategy_version16,
        'version17': strategy_version17,
        'version18': strategy_version18,
        'version19': strategy_version19,
        'version20': strategy_version20,
        'version21': strategy_version21,
        'version22': strategy_version22,
        'version23': strategy_version23,
        'version24': strategy_version24,
        'version25': strategy_version25,
        'version26': strategy_version26,
        'version27': strategy_version27,
        'version28': strategy_version28,
        'version29': strategy_version29,
        'version30': strategy_version30,
        'version31': strategy_version31,
        'version32': strategy_version32
    }
    _rates = {
        'TF': 5e-6,
        'TS': 2.5e-6,
        'T': 5e-6
    }

    _sizes = {
        'T': 10000,
        'TF': 10000,
        'TS': 20000
    }

    _priceticks = {
        'T': 0.005,
        'TF': 0.005,
        'TS': 0.005
    }
    strategy = strategy_map[version]


    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbols=symbols,
        interval=Interval.MINUTE,
        start=start_date,
        end=end_date,
        rates={
            symbols[0]: _rates[get_base_symbol_from_vt(symbols[0])],
            symbols[1]: _rates[get_base_symbol_from_vt(symbols[1])]
        },
        slippages={
            symbols[0]: 0,
            symbols[1]: 0
        },
        sizes={
            symbols[0]: _sizes[get_base_symbol_from_vt(symbols[0])],
            symbols[1]: _sizes[get_base_symbol_from_vt(symbols[1])]
        },
        priceticks={
            symbols[0]: _priceticks[get_base_symbol_from_vt(symbols[0])],
            symbols[1]: _priceticks[get_base_symbol_from_vt(symbols[1])]
        },
        capital=1_000_000,
    )


    engine.add_strategy(strategy, params)
    engine.load_data()
    engine.run_backtesting(init_pos=init_pos)
    df = engine.calculate_result()
    df1 = engine.calculate_statistics()
    # df2 = engine.strategy.daily_pos_detail
    engine.show_chart()
    res = df[['trading_pnl', 'holding_pnl']]

    df1.update(params)
    pd.set_option('display.max_colwidth', -1)
    df.loc['average'] = df.mean()

    df = format_int(df)
    df1 = format_int(df1)
    print(pd.Series(df1))
    print(tabulate(df, headers=df.columns, tablefmt='psql'))
    df2=pd.DataFrame()

    save_name = f"{params['active_vt_symbol']}-{params['passive_vt_symbol']}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}_{version}_{datetime.now().time().strftime('%H%M%S')}"
    save_path = os.path.join(os.getcwd(), 'results')
    xlsx_path = fr'{save_path}\{save_name}.xlsx'
    xlsx_path = Path(xlsx_path).as_posix()
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
    for d, sheetname in zip([pd.Series(df1), df, df2], ['summary', 'daily', 'average_pos_daily']):
        d.to_excel(writer, sheet_name=sheetname, index=True)
    writer.save()
    return res

if __name__ == '__main__':
    active_symbol = 'TF2003'
    passive_symbol = 'TF2006'
    start_date = datetime(2020,1,1)
    end_date = datetime(2020,2,21)
    bk_params = {
        'max_pos': 5,
        'win_tick': 158,
        'stop_tick': 108,
        'std_num': 1.5,
        'spread_n': 120,
        'safe_pos': 1
    }
    version = 'version32'
    res = run_backtest(active_symbol, passive_symbol, start_date, end_date, bk_params, version)
    # df = pd.DataFrame(res)
    # df['profit'] = None
    # df.columns = ['symbol', 'price', 'direction', 'offset', 'gateway', 'cci', 'ma1', 'ppo1','ma2', 'ppo2','ma3', 'ppo3','profit']
    # from vnpy.trader.object import TickData, BarData, TradeData, OrderData, Direction, CtaPostionDetails, Offset
    #
    # for i in df.index:
    #     if 'TF' not in df.loc[i, 'symbol']:
    #         continue
    #     if df.loc[i, 'offset'] == Offset.OPEN:
    #         if df.loc[i, 'direction'] == Direction.LONG:
    #             sig = 1
    #         else:
    #             sig = -1
    #         left = df.loc[i:]
    #         left = left[left['offset'] == Offset.CLOSE]
    #         if len(left) == 0:
    #             continue
    #         closep = left['price'].iloc[0]
    #         p = (closep - df.loc[i, 'price']) * sig
    #         df.loc[i, 'profit'] = p
    # df_summary=df[df['profit']!=None]
    # df_summary=df[~df['profit'].isna()]
    # df_summary=df[(~df['profit'].isna())&(df['profit']!=0)]
    # df_pos = df[df['profit']>0]
    # df_neg = df[df['profit']<0]
    # df_pos_long = df_pos[df_pos['direction']==Direction.LONG]
    # df_pos_short = df_pos[df_pos['direction']==Direction.SHORT]
    # df_neg_short = df_pos[df_pos['direction']==Direction.SHORT]
    # df_neg_long = df_neg[df_neg['direction']==Direction.LONG]
    # df_neg_short = df_neg[df_neg['direction']==Direction.SHORT]
    #
    # a = df_pos_long.describe()
    # b = df_neg_long.describe()
    #
    # for c in ['ppo1', 'ppo2', 'ppo3']:
    #     r = df_summary[((df_summary[c]<0)&(df_summary['direction']==Direction.LONG))|\
    #                     ((df_summary[c]>0)&(df_summary['direction']==Direction.SHORT))]
    #     print(r.profit.mean(), len(r[r['profit']>0])/len(r))