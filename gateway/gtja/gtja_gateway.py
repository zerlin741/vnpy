"""
"""

import sys
import pytz
import json
from datetime import datetime
from typing import Dict

from vnpy.api.hft import TdApi
from vnpy.api.sip import (
    MdApi,
    AccountType_Stock,
    PositionSide_Long,
    PositionSide_Short,
    OrderType_LMT,
    OrderFlag_Security_Normal,
    OrderStatus_PendingNew,
    OrderStatus_New,
    OrderStatus_PartiallyFilled,
    OrderStatus_Filled,
    OrderStatus_PendingCancel,
    OrderStatus_Canceling,
    OrderStatus_CancelFilled,
    OrderStatus_PartiallyCanceled,
    OrderStatus_Rejected,
    OrderStatus_CancelRejected,
    MKtype_SH,
    MKtype_SZ,
)
from vnpy.trader.constant import (
    Direction,
    Exchange,
    Product,
    Status,
)
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.utility import round_to, get_folder_path
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
)
from vnpy.trader.event import EVENT_TIMER

from.terminal_info import get_terminal_info


MAX_FLOAT = sys.float_info.max
CHINA_TZ = pytz.timezone("Asia/Shanghai")

MK_GTJA2VT: Dict[int, Exchange] = {
    MKtype_SH: Exchange.SSE,
    MKtype_SZ: Exchange.SZSE

}
MK_VT2GTJA: Dict[Exchange, int] = {v: k for k, v in MK_GTJA2VT.items()}

SH_PRODUCT_GTJA2VT: Dict[str, Product] = {
    "ES": Product.EQUITY,
    "D": Product.BOND,
    "RWS": Product.OPTION,
    "FF": Product.FUTURES,
    "EU": Product.FUND
}

SZ_PRODUCT_GTJA2VT: Dict[int, Product] = {
    1: Product.EQUITY,
    2: Product.EQUITY,
    3: Product.EQUITY,
    4: Product.EQUITY,
    5: Product.BOND,
    6: Product.BOND,
    7: Product.BOND,
    8: Product.BOND,
    9: Product.BOND,
    10: Product.BOND,
    11: Product.BOND,
    12: Product.BOND,
    13: Product.BOND,
    14: Product.ETF,
    15: Product.ETF,
    16: Product.ETF,
    17: Product.ETF,
    18: Product.ETF,
    19: Product.ETF,
    20: Product.ETF,
    21: Product.ETF,
    22: Product.ETF,
    23: Product.FUND,
    24: Product.FUND,
    25: Product.FUND,
    26: Product.FUND,
    28: Product.OPTION,
    29: Product.OPTION,
    30: Product.OPTION,
    33: Product.EQUITY,
    34: Product.BOND,
    35: Product.BOND,

}

ORDERSTATUS_GTJA2VT: Dict[int, Status] = {
    OrderStatus_PendingNew: Status.SUBMITTING,
    OrderStatus_New: Status.NOTTRADED,
    OrderStatus_PartiallyFilled: Status.PARTTRADED,
    OrderStatus_Filled: Status.ALLTRADED,
    OrderStatus_PendingCancel: Status.SUBMITTING,
    OrderStatus_Canceling: Status.CANCELLED,
    OrderStatus_CancelFilled: Status.CANCELLED,
    OrderStatus_PartiallyCanceled: Status.CANCELLED,
    OrderStatus_Rejected: Status.REJECTED,
    OrderStatus_CancelRejected: Status.REJECTED,
}

EXCHANGE_GTJA2VT: Dict[str, Exchange] = {
    "SH": Exchange.SSE,
    "SZ": Exchange.SZSE
}
EXCHANGE_VT2GTJA: Dict[Exchange, str] = {
    v: k for k, v in EXCHANGE_GTJA2VT.items()
}
DIRECTION_GTJA2VT: Dict[int, Direction] = {
    PositionSide_Long: Direction.LONG,
    PositionSide_Short: Direction.SHORT
}
DIRECTION_VT2GTJA: Dict[Direction, int] = {
    v: k for k, v in DIRECTION_GTJA2VT.items()
}

symbol_name_map: Dict[str, str] = {}
symbol_pricetick_map: Dict[str, float] = {}


class GtjaGateway(BaseGateway):
    """
    VN Trader Gateway for CTP .
    """

    default_setting = {
        "???????????????": "",
        "????????????": "",
        "???????????????": "",
        "????????????": "",
        "????????????": "",
        "???????????????": "",
        "??????": "",
        "???????????????": "",
        "????????????": "",
        "???????????????": "",
        "????????????": ""
    }

    exchanges = [Exchange.SSE, Exchange.SZSE]

    def __init__(self, event_engine):
        """Constructor"""
        super().__init__(event_engine, "GTJA")

        self.td_api = GtjaTdApi(self)
        self.md_api = GtjaMdApi(self)

    def connect(self, setting: dict):
        """"""
        td_userid = setting["???????????????"]
        td_password = setting["????????????"]
        td_ip = setting["???????????????"]
        td_port = setting["????????????"]
        orgid = setting["????????????"]
        barnchid = setting["???????????????"]
        system_id = setting["??????"]

        md_userid = setting["???????????????"]
        md_password = setting["????????????"]
        md_ip = setting["???????????????"]
        md_port = setting["????????????"]

        self.td_api.connect(
            td_userid,
            td_password,
            orgid,
            barnchid,
            system_id,
            td_ip,
            td_port
        )

        self.md_api.connect(
            md_ip,
            md_port,
            md_userid,
            md_password,
        )

    def subscribe(self, req: SubscribeRequest) -> None:
        """"""
        self.md_api.subscrbie(req)

    def send_order(self, req: OrderRequest) -> str:
        """"""
        self.td_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        self.td_api.cancel_order(req)

    def query_account(self) -> None:
        """"""
        self.td_api.query_account()

    def query_position(self):
        """"""
        self.td_api.query_position()

    def close(self) -> None:
        """"""
        self.td_api.close()
        self.md_api.close()

    def write_error(self, msg: str, error: dict):
        """"""
        error_id = error["err_code"]
        error_msg = error["err_msg"]
        msg = f"{msg}????????????{error_id}????????????{error_msg}"
        self.write_log(msg)

    def process_timer_event(self, event):
        """"""
        self.count += 1
        if self.count < 2:
            return
        self.count = 0

        func = self.query_functions.pop(0)
        func()
        self.query_functions.append(func)

    def init_query(self):
        """"""
        self.count = 0
        self.query_functions = [self.query_account, self.query_position]
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)


class GtjaMdApi(MdApi):

    def __init__(self, gateway: GtjaGateway):
        """"""
        super().__init__()

        self.gateway: GtjaGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.userid: str = ""
        self.password: str = ""
        self.client_id: int = 0
        self.server_ip: str = ""
        self.server_port: int = 0
        self.protocol: int = 0
        self.session_id: int = 0
        self.date: str = ""

        self.connect_status: bool = False
        self.login_status: bool = False

        self.sse_inited: bool = False
        self.szse_inited: bool = False

    def onDisconnected(self, reason: int) -> None:
        """"""
        self.connect_status = False
        self.login_status = False
        self.gateway.write_log(f"???????????????????????????, ??????{reason}")

        self.login_server()

    def onSubscribe(self, error):
        """"""
        if error["errcode"]:
            self.gateway.write_log(
                f"????????????????????????{error['errcode']}?????????{error['errstr']}"
            )

    def onDepthMarketData(self, mk_type: int, symbol: str, data: dict) -> None:
        """"""
        pass

    def onMarketData(self, mk_type: int, symbol: str, data: dict) -> None:
        """"""
        timestamp = f"{self.date}{str(data['nTime'])}"
        dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S%f")
        dt = CHINA_TZ.localize(dt)

        tick = TickData(
            symbol=symbol,
            exchange=MK_GTJA2VT[mk_type],
            datetime=dt,
            volume=data["iVolume"],
            last_price=data["uMatch"] / 10000,
            limit_up=data["uHighLimited"] / 10000,
            limit_down=data["uLowLimited"] / 10000,
            open_price=data["uOpen"] / 10000,
            high_price=data["uHigh"] / 10000,
            low_price=data["uLow"] / 10000,
            pre_close=data["uPreClose"] / 10000,
            gateway_name=self.gateway_name
        )

        tick.bid_price_1, tick.bid_price_2, tick.bid_price_3, tick.bid_price_4, tick.bid_price_5 = data["bid"][0:5]
        tick.ask_price_1, tick.ask_price_2, tick.ask_price_3, tick.ask_price_4, tick.ask_price_5 = data["ask"][0:5]
        tick.bid_volume_1, tick.bid_volume_2, tick.bid_volume_3, tick.bid_volume_4, tick.bid_volume_5 = data["bid_qty"][0:5]
        tick.ask_volume_1, tick.ask_volume_2, tick.ask_volume_3, tick.ask_volume_4, tick.ask_volume_5 = data["ask_qty"][0:5]

        pricetick = symbol_pricetick_map.get(tick.vt_symbol, 0)
        if pricetick:
            tick.bid_price_1 = round_to(tick.bid_price_1 / 10000, pricetick)
            tick.bid_price_2 = round_to(tick.bid_price_2 / 10000, pricetick)
            tick.bid_price_3 = round_to(tick.bid_price_3 / 10000, pricetick)
            tick.bid_price_4 = round_to(tick.bid_price_4 / 10000, pricetick)
            tick.bid_price_5 = round_to(tick.bid_price_5 / 10000, pricetick)
            tick.ask_price_1 = round_to(tick.ask_price_1 / 10000, pricetick)
            tick.ask_price_2 = round_to(tick.ask_price_2 / 10000, pricetick)
            tick.ask_price_3 = round_to(tick.ask_price_3 / 10000, pricetick)
            tick.ask_price_4 = round_to(tick.ask_price_4 / 10000, pricetick)
            tick.ask_price_5 = round_to(tick.ask_price_5 / 10000, pricetick)

        tick.name = symbol_name_map.get(tick.vt_symbol, tick.symbol)
        self.gateway.on_tick(tick)

    def connect(
        self,
        ip: str,
        port: int,
        username: str,
        password: str,
    ) -> None:
        """"""
        cfg = generate_cfg(ip, port, username, password)

        self.date = datetime.now().strftime("%Y%m%d")

        # Create API object
        if not self.connect_status:
            self.createMdApi(cfg, False)
            n = self.login()
            self.query_contract()

            if not n:
                self.gateway.write_log("???????????????????????????")
                self.connect_status = True
                self.login_status = True
            else:
                self.gateway.write_log(f"??????????????????????????????{n}")
        else:
            self.gateway.write_log("??????????????????????????????????????????")

    def close(self) -> None:
        """"""
        if self.connect_status:
            self.exit()

    def subscrbie(self, req: SubscribeRequest) -> None:
        """"""
        if self.login_status:
            gtja_exchange = MK_VT2GTJA.get(req.exchange, "")
            self.subscribeMarketData(gtja_exchange, req.symbol)

    def query_contract(self) -> None:
        """
        0 -> SSE, SZSE
        1 -> SSE,
        2 -> SZSE
        """
        self.subscribeBaseInfo(0)

    def onSHBaseInfo(self, code, data):
        contract = ContractData(
            gateway_name=self.gateway_name,
            symbol=str(code),
            exchange=Exchange.SSE,
            name=data["szStkNameZN"].split(" ")[0],
            product=SH_PRODUCT_GTJA2VT.get(
                data["szStkClass"].split(" ")[0], Product.EQUITY
            ),
            min_volume=data["i64BuyNumUnit"],
            pricetick=data["i64PriceLevel"] / 10000,
            size=1,
        )
        self.gateway.on_contract(contract)
        symbol_name_map[contract.vt_symbol] = contract.name
        symbol_pricetick_map[contract.vt_symbol] = contract.pricetick

    def onSZBaseInfo(self, code, data):
        contract = ContractData(
            gateway_name=self.gateway_name,
            symbol=str(code),
            exchange=Exchange.SZSE,
            name=data["sSymbol"],
            product=SZ_PRODUCT_GTJA2VT.get(
                data["usSecurityType"], Product.EQUITY
            ),
            min_volume=data["i64BuyQtyUnit"],
            pricetick=data.get("i64PriceTick", 100) / 10000,
            size=1,
        )

        self.gateway.on_contract(contract)
        symbol_name_map[contract.vt_symbol] = contract.name
        symbol_pricetick_map[contract.vt_symbol] = contract.pricetick


class GtjaTdApi(TdApi):
    """"""

    def __init__(self, gateway):
        """Constructor"""
        super().__init__()

        self.gateway: GtjaGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.reqid: int = 0
        self.order_count: int = 0

        self.connect_status: bool = False
        self.login_status: bool = False

        self.userid: str = ""
        self.password: str = ""
        self.orders: Dict[str, OrderData] = {}

        self.orderid_sysid_map = {}

        self.prefix = ""

    def _new_orderid(self) -> str:
        """"""
        self.order_count += 1
        orderid = f"{self.date}_{self.order_count}"
        return orderid

    def onError(self, error: dict, reqid: int) -> None:
        """"""
        self.gateway.write_error("??????", error)

    def onRiskNotify(self, data: dict) -> None:
        """"""
        status = data["alarm_status"]
        rule = data["alarm_rule"]
        self.gateway.write_log(f"???????????????????????????{status}?????????{rule}")

    def onDisconnect(self) -> None:
        """"""
        self.login_status = False
        self.gateway.write_log(f"???????????????????????????")

    def onLogin(self, data: dict, error: dict) -> None:
        """"""
        if not error["err_code"]:
            self.login_status = True
            self.gateway.write_log(f"???????????????????????????")

            self.query_order()
            self.query_trade()
            self.gateway.init_query()
        else:
            self.gateway.write_error("???????????????????????????", error)

    def onTradeReport(self, data) -> None:
        """"""
        exchange, symbol = data["symbol"].split(".")
        dt = f"{data['trade_date']} {data['trade_time']}"
        orderid = data["cl_order_id"]
        sysid = data["order_id"]

        if not orderid and sysid:
            orderid = self._new_orderid()

        dt = datetime.strptime(dt, "%Y%m%d %H%M%S%f")
        dt = CHINA_TZ.localize(dt)

        trade = TradeData(
            tradeid=data["report_no"],
            orderid=orderid,
            gateway_name=self.gateway_name,
            symbol=symbol,
            exchange=EXCHANGE_GTJA2VT[exchange],
            direction=DIRECTION_GTJA2VT[data["side"]],
            price=data["price"] / 10000,
            volume=data["volume"],
            datetime=dt,
        )

        order = self.orders[orderid]
        order.datetime = trade.datetime
        order.traded += trade.volume
        if order.traded == order.volume:
            order.status = Status.ALLTRADED
        else:
            order.status = Status.PARTTRADED

        self.gateway.on_order(order)
        self.gateway.on_trade(trade)

    def onOrderRsp(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ):
        """"""
        orderid = data["cl_order_id"]
        order = self.orders[orderid]

        if error["err_code"]:
            self.gateway.write_error("??????????????????", error)

            order.status = Status.REJECTED
            dt = datetime.now()
            dt = CHINA_TZ.localize(dt)
            order.datetime = dt
            self.gateway.on_order(order)
        else:
            sysid = data["order_id"]
            self.orderid_sysid_map[orderid] = sysid

    def onCancelRsp(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool
    ) -> None:
        """"""
        if error["err_code"]:
            self.gateway.write_error("??????????????????", error)

    def onQueryPositionRsp(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool,
        pos: str
    ) -> None:
        """"""
        exchange, symbol = data["symbol"].split(".")

        pos = PositionData(
            gateway_name=self.gateway_name,
            symbol=symbol,
            exchange=EXCHANGE_GTJA2VT[exchange],
            direction=Direction.NET,
            volume=data["volume"],
            price=data["cost_price"] / 10000,
            pnl=data["total_income"] / 10000,
        )
        self.gateway.on_position(pos)

    def onQueryCashRsp(self, data: dict, error: dict, reqid: int) -> None:
        """"""
        account = AccountData(
            accountid=data["account_id"],
            balance=data["total_amount"] / 10000,
            frozen=(data["total_amount"] - data["avail_amount"]) / 10000,
            gateway_name=self.gateway_name
        )
        self.gateway.on_account(account)

    def onQueryOrderRsp(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool,
        pos: str
    ) -> None:
        """"""
        if error["err_code"] != 14020:
            exchange, symbol = data["symbol"].split(".")

            orderid = data["cl_order_id"]
            sysid = data["order_id"]
            if not orderid:
                orderid = sysid
            else:
                self.orderid_sysid_map[orderid] = sysid

            dt = f"{data['order_date']} {data['order_time']}"
            dt = datetime.strptime(dt, "%Y%m%d %H%M%S%f")
            dt = CHINA_TZ.localize(dt)

            order = OrderData(
                orderid=orderid,
                gateway_name=self.gateway_name,
                symbol=symbol,
                exchange=EXCHANGE_GTJA2VT[exchange],
                direction=DIRECTION_GTJA2VT[data["side"]],
                price=data["price"] / 10000,
                volume=data["volume"],
                status=ORDERSTATUS_GTJA2VT[data["order_status"]],
                datetime=dt,
            )
            self.orders[orderid] = order
            self.gateway.on_order(order)

        if last:
            self.gateway.write_log("????????????????????????")

    def onQueryTradeRsp(
        self,
        data: dict,
        error: dict,
        reqid: int,
        last: bool,
        pos: str
    ) -> None:
        """"""
        if error["err_code"] != 14020:
            exchange, symbol = data["symbol"].split(".")

            orderid = data["cl_order_id"]
            sysid = data["order_id"]
            if not orderid:
                orderid = sysid

            dt = f"{data['trade_date']} {data['trade_time']}"
            dt = datetime.strptime(dt, "%Y%m%d %H%M%S%f")
            dt = CHINA_TZ.localize(dt)

            trade = TradeData(
                tradeid=data["report_no"],
                orderid=orderid,
                gateway_name=self.gateway_name,
                symbol=symbol,
                exchange=EXCHANGE_GTJA2VT[exchange],
                direction=DIRECTION_GTJA2VT[data["side"]],
                price=data["price"] / 10000,
                volume=data["volume"],
                datetime=dt,
            )
            self.gateway.on_trade(trade)

        if last:
            self.gateway.write_log("????????????????????????")

    def connect(
        self,
        userid: str,
        password: str,
        orgid: str,
        branchid: str,
        system_id: str,
        host: str,
        port: str,
    ) -> None:
        """
        Start connection to server.
        """
        self.userid = userid
        self.password = password
        self.prefix = datetime.now().strftime("%Y%m%d%H%M%S")

        if not self.connect_status:
            path = get_folder_path(self.gateway_name.lower())
            self.setLogConfig(str(path))
            self.createTraderApi()

            account_info = {
                "account_id": userid,
                "account_type": AccountType_Stock,
                "account_pwd": password,
                "cust_orgid": orgid,
                "cust_branchid": branchid,
                "cl_system_id": system_id,
            }

            self.login(host, int(port), account_info, get_terminal_info())

            self.connect_status = True

    def send_order(self, req: OrderRequest) -> str:
        """
        Send new order.
        """
        self.order_count += 1
        suffix = str(self.order_count).rjust(6, "0")
        orderid = f"{self.prefix}_{suffix}"

        exchange = EXCHANGE_VT2GTJA[req.exchange]
        gtja_symbol = f"{exchange}.{req.symbol}"

        order_req = {
            "cl_order_id": orderid,
            "symbol": gtja_symbol,
            "order_type": OrderType_LMT,
            "side": DIRECTION_VT2GTJA[req.direction],
            "volume": int(req.volume),
            "price": int(req.price * 10000),
            "order_flag": OrderFlag_Security_Normal,

        }

        self.reqid += 1
        self.order(order_req, self.reqid)

        order = req.create_order_data(orderid, self.gateway_name)
        self.orders[orderid] = order
        self.gateway.on_order(order)
        return order.vt_orderid

    def query_order(self) -> None:
        """"""
        self.reqid += 1
        self.queryOrders("", 500, self.reqid, 0)

    def query_trade(self) -> None:
        """"""
        self.reqid += 1
        self.queryTrades("", 500, self.reqid)

    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel existing order.
        """
        self.reqid += 1
        sysid = self.orderid_sysid_map.get(req.orderid, req.orderid)

        cancel_req = {"order_id": sysid}
        self.cancelOrder(cancel_req, self.reqid)

    def query_account(self) -> None:
        """
        Query account balance data.
        """
        self.reqid += 1
        self.queryCash(self.reqid)

    def query_position(self) -> None:
        """
        Query position holding data.
        """

        self.reqid += 1
        self.queryPositions("", 500, self.reqid)

    def close(self) -> None:
        """"""
        if self.connect_status:
            self.exit()


def adjust_price(price: float) -> float:
    """"""
    if price == MAX_FLOAT:
        price = 0
    return price


def generate_cfg(ip: str, port: int, username: str, password: str) -> str:
    """"""
    setting = {
        "ip0": ip,
        "port0": port,
        "connect_mode": "NR",
        "username": username,
        "password": password
    }
    cfg = json.dumps(setting, separators=("|", ":"))
    return cfg
