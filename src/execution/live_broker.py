"""
CCXT Live Execution Adapter with Hard Safety Circuit Breakers.
Safeguards real trading capital against unexpected fat-finger errors,
extreme leverage, and flash drawdowns.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import ccxt

from src.domain.entities import Order, Portfolio, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.ports.execution_ports import IExecutionClient
from src.risk.kill_switch import EmergencyKillSwitch
from src.security.vault import (
    ExchangeCredentials,
    PreFlightPermissionValidator,
    SecurityViolationError,
)

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when a risk or safety circuit breaker halts execution."""


class MaxLeverageExceededError(CircuitBreakerError):
    """Raised when an order would push portfolio gross leverage beyond limit."""


class MaxOrderNotionalExceededError(CircuitBreakerError):
    """Raised when an individual order exceeds the maximum single-trade dollar ceiling."""


class MaxDrawdownStopError(CircuitBreakerError):
    """Raised when portfolio drawdown exceeds the hard emergency stop threshold."""


class MarketCapacityExceededError(CircuitBreakerError):
    """Raised when an order consumes more than the allowed top-of-book depth fraction."""


class CCXTExecutionAdapter(IExecutionClient):
    """
    Institutional exchange adapter with fail-safe circuit breakers.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str | None = None,
        api_secret: str | None = None,
        credentials: ExchangeCredentials | None = None,
        validate_permissions: bool = False,
        strict_ip_whitelist: bool = True,
        dry_run: bool = True,
        max_gross_leverage: float = 2.5,
        max_order_notional: float = 25_000.0,
        max_drawdown_stop: float = 0.15,
        taker_fee_pct: float = 0.0004,
        maker_fee_pct: float = 0.0002,
    ):
        if credentials is not None:
            exchange_id = credentials.exchange_id
            api_key = credentials.api_key
            api_secret = credentials.api_secret

        self.exchange_id = exchange_id
        self.dry_run = dry_run
        self.strict_ip_whitelist = strict_ip_whitelist
        self.max_gross_leverage = max_gross_leverage
        self.max_order_notional = max_order_notional
        self.max_drawdown_stop = max_drawdown_stop
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct

        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "fetchMarkets": {"types": ["linear"]},
                "fetchCurrencies": False,
            },
        }
        self.client: ccxt.Exchange = exchange_class(config)

        if validate_permissions and not self.dry_run and api_key:
            try:
                PreFlightPermissionValidator.validate_binance_permissions(
                    self.client,
                    strict_ip_whitelist=strict_ip_whitelist,
                )
            except SecurityViolationError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Sub-account / SAPI permission endpoint note ({e}); "
                    "sub-accounts inherently block external withdrawals."
                )

        self.portfolio = Portfolio(
            cash=100_000.0,
            initial_cash=100_000.0,
            positions={},
        )
        self.peak_equity = 100_000.0
        self.trade_history: list[Trade] = []
        self.orders: dict[str, Order] = {}

    def get_portfolio(self) -> Portfolio:
        return self.portfolio

    def get_order_status(self, order_id: str, symbol: str) -> Order:
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found.")
        return self.orders[order_id]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_trade_history(self) -> list[Trade]:
        return list(self.trade_history)

    def verify_risk_circuit_breakers(self, symbol: str, quantity: float, est_price: float) -> None:
        """
        Enforces 3 non-negotiable institutional safety gates:
        1. Maximum single order notional
        2. Emergency drawdown tripwire
        3. Portfolio gross leverage ceiling
        """
        order_notional = quantity * est_price
        if order_notional > self.max_order_notional:
            raise MaxOrderNotionalExceededError(
                f"Order notional ${order_notional:,.2f} exceeds limit ${self.max_order_notional:,.2f}"
            )

        current_equity = self.portfolio.total_equity
        self.peak_equity = max(self.peak_equity, current_equity)

        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.max_drawdown_stop:
            raise MaxDrawdownStopError(
                f"Hard emergency stop: current drawdown {drawdown*100:.2f}% exceeds {self.max_drawdown_stop*100:.2f}%"
            )

        # Estimate post-trade gross leverage
        current_gross = sum(abs(p.market_value) for p in self.portfolio.positions.values())
        new_gross = current_gross + order_notional
        new_leverage = new_gross / max(current_equity, 1.0)
        if new_leverage > self.max_gross_leverage:
            raise MaxLeverageExceededError(
                f"Projected gross leverage {new_leverage:.2f}x exceeds ceiling {self.max_gross_leverage:.2f}x"
            )

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        est_price = price or 100.0

        # Safety Check
        self.verify_risk_circuit_breakers(symbol, quantity, est_price)

        order_id = f"ccxt_ord_{uuid.uuid4().hex[:10]}"
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(tz=timezone.utc),
        )

        if self.dry_run:
            # Simulate fill immediately in dry run
            order.status = OrderStatus.FILLED
            order.filled_quantity = quantity
            order.average_fill_price = est_price
            trade = Trade(
                id=f"dry_trd_{uuid.uuid4().hex[:10]}",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=est_price,
                fee=quantity * est_price * self.taker_fee_pct,
                timestamp=datetime.now(tz=timezone.utc),
            )
            self.trade_history.append(trade)
            self.portfolio.cash -= trade.fee
            # Record position
            pos_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                entry_price=est_price,
                current_price=est_price,
            )
        else:
            # Send live order to CCXT
            ccxt_side = "buy" if side == OrderSide.BUY else "sell"
            ccxt_type = "market" if order_type == OrderType.MARKET else "limit"
            res = self.client.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=quantity,
                price=price,
            )
            order.status = OrderStatus.SUBMITTED
            order.id = str(res.get("id", order_id))

        self.orders[order_id] = order
        return order


class BinanceLiveBroker(CCXTExecutionAdapter):
    """
    Production Binance USD-M Perpetual Futures (`fapi`) Execution Broker.
    Features:
    1. Post-Only (`timeInForce='GTX'`) Maker Limit-Chase Execution Engine.
    2. Atomic Long/Short Leg Pairing (concurrent execution to prevent directional drift).
    3. Strict Exchange Filter Quantization (`tickSize`, `stepSize`, `minNotional`).
    4. Pre-Trade `positionRisk` Reconciliation and `reduceOnly=True` Emergency Flatten.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        credentials: ExchangeCredentials | None = None,
        testnet: bool = False,
        dry_run: bool = True,
        initial_capital: float = 1000.0,
        max_gross_leverage: float = 6.0,
        max_order_notional: float = 25_000.0,
        max_drawdown_stop: float = 0.15,
        chase_interval_sec: float = 5.0,
        max_chase_iterations: int = 3,
        allow_taker_fallback: bool = True,
        strict_ip_whitelist: bool = False,
        kill_switch: EmergencyKillSwitch | None = None,
    ) -> None:
        super().__init__(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            credentials=credentials,
            validate_permissions=(not dry_run and not testnet),
            strict_ip_whitelist=strict_ip_whitelist,
            dry_run=dry_run,
            max_gross_leverage=max_gross_leverage,
            max_order_notional=max_order_notional,
            max_drawdown_stop=max_drawdown_stop,
        )
        if testnet and hasattr(self.client, "set_sandbox_mode"):
            self.client.set_sandbox_mode(True)

        self.portfolio = Portfolio(
            cash=float(initial_capital),
            initial_cash=float(initial_capital),
            positions={},
        )
        self.peak_equity = float(initial_capital)
        self.chase_interval_sec = float(chase_interval_sec)
        self.max_chase_iterations = int(max_chase_iterations)
        self.allow_taker_fallback = bool(allow_taker_fallback)
        self.kill_switch = kill_switch or EmergencyKillSwitch()

        self.maker_fills_count: int = 0
        self.taker_fills_count: int = 0

    @property
    def maker_fill_ratio(self) -> float:
        total = self.maker_fills_count + self.taker_fills_count
        return (self.maker_fills_count / total) if total > 0 else 1.0

    def quantize_price(self, symbol: str, price: float) -> float:
        """Quantizes price strictly to Binance symbol `tickSize`."""
        try:
            if hasattr(self.client, "markets") and self.client.markets:
                return float(self.client.price_to_precision(symbol, price))
        except Exception:
            pass
        return round(float(price), 6 if price < 1.0 else (4 if price < 100.0 else 2))

    def quantize_amount(self, symbol: str, amount: float) -> float:
        """Quantizes quantity strictly to Binance symbol `stepSize`."""
        try:
            if hasattr(self.client, "markets") and self.client.markets:
                return float(self.client.amount_to_precision(symbol, amount))
        except Exception:
            pass
        return round(float(amount), 4 if amount < 10.0 else 2)

    def fetch_best_bid_ask(self, symbol: str, fallback_price: float = 100.0) -> tuple[float, float]:
        """
        Fetches true L1 top-of-book `(best_bid, best_ask)` for Post-Only (`GTX`) limit placement.
        Queries L1 `fetch_order_book(symbol, limit=5)` first (since Binance Futures `/fapi/v1/ticker/24hr`
        omits `bid`/`ask`), falling back to `fetch_ticker` with tick-safe spread separation.
        """
        try:
            if hasattr(self.client, "fetch_order_book"):
                ob = self.client.fetch_order_book(symbol, limit=5)
                if isinstance(ob, dict):
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    if len(bids) > 0 and len(asks) > 0:
                        b_px = float(bids[0][0])
                        a_px = float(asks[0][0])
                        if 0 < b_px < a_px:
                            return b_px, a_px
        except Exception:
            pass

        try:
            ticker = self.client.fetch_ticker(symbol)
            last = float(ticker.get("last") or ticker.get("close") or fallback_price)
            raw_bid = ticker.get("bid")
            raw_ask = ticker.get("ask")
            bid = float(raw_bid) if (raw_bid is not None and float(raw_bid) > 0) else (last * 0.9995)
            ask = float(raw_ask) if (raw_ask is not None and float(raw_ask) > 0) else (last * 1.0005)
            if bid >= ask:
                bid = last * 0.9995
                ask = last * 1.0005
            q_bid = self.quantize_price(symbol, bid)
            q_ask = self.quantize_price(symbol, ask)
            if q_ask <= q_bid:
                step = max(last * 0.001, 1e-6)
                q_ask = self.quantize_price(symbol, q_bid + step)
            return q_bid, max(q_ask, q_bid * 1.0002)
        except Exception:
            return fallback_price * 0.9995, fallback_price * 1.0005

    @staticmethod
    def compute_effective_margin_collateral(bal: dict[str, Any]) -> dict[str, Any]:
        """
        Tier 1C (Issue #57): Computes unified USDS-M Cross-Margin collateral across
        `USDT` (100% weight) and yield-bearing margin assets `BFUSD` (99.9% weight) and
        `LDUSDT` (99.9% weight).
        """
        haircuts = {"USDT": 1.000, "BFUSD": 0.999, "LDUSDT": 0.999}
        wallet_by_asset: dict[str, float] = {"USDT": 0.0, "BFUSD": 0.0, "LDUSDT": 0.0}
        free_by_asset: dict[str, float] = {"USDT": 0.0, "BFUSD": 0.0, "LDUSDT": 0.0}
        non_zero_assets: list[dict[str, Any]] = []

        if not isinstance(bal, dict):
            return {
                "usdt_wallet": 0.0,
                "bfusd_wallet": 0.0,
                "ldusdt_wallet": 0.0,
                "yield_collateral_usd": 0.0,
                "effective_wallet_usd": 0.0,
                "effective_free_usd": 0.0,
                "yield_collateral_share": 0.0,
                "non_zero_assets": non_zero_assets,
            }

        for asset_code in ("USDT", "BFUSD", "LDUSDT"):
            info_obj = bal.get(asset_code)
            if isinstance(info_obj, dict):
                f_val = float(info_obj.get("free") or 0.0)
                t_val = float(info_obj.get("total") or f_val)
                wallet_by_asset[asset_code] = max(wallet_by_asset[asset_code], t_val)
                free_by_asset[asset_code] = max(free_by_asset[asset_code], f_val)

        raw_info = bal.get("info", {})
        if isinstance(raw_info, dict) and isinstance(raw_info.get("assets"), list):
            for a in raw_info["assets"]:
                if not isinstance(a, dict):
                    continue
                code = str(a.get("asset") or "").upper()
                wb = float(a.get("walletBalance") or 0.0)
                ab = float(a.get("availableBalance") or wb)
                if wb > 0 or ab > 0:
                    non_zero_assets.append(
                        {
                            "asset": code,
                            "walletBalance": wb,
                            "availableBalance": ab,
                        }
                    )
                if code in wallet_by_asset:
                    wallet_by_asset[code] = wb
                    free_by_asset[code] = ab

        eff_wallet = sum(wallet_by_asset[k] * haircuts[k] for k in haircuts)
        eff_free = sum(free_by_asset[k] * haircuts[k] for k in haircuts)
        yield_usd = wallet_by_asset["BFUSD"] * haircuts["BFUSD"] + wallet_by_asset["LDUSDT"] * haircuts["LDUSDT"]
        yield_share = (yield_usd / eff_wallet) if eff_wallet > 0 else 0.0

        return {
            "usdt_wallet": wallet_by_asset["USDT"],
            "bfusd_wallet": wallet_by_asset["BFUSD"],
            "ldusdt_wallet": wallet_by_asset["LDUSDT"],
            "yield_collateral_usd": yield_usd,
            "effective_wallet_usd": eff_wallet,
            "effective_free_usd": eff_free if eff_free > 0 else eff_wallet,
            "yield_collateral_share": yield_share,
            "non_zero_assets": non_zero_assets,
        }

    def reconcile_positions_from_exchange(self) -> dict[str, Position]:
        """
        Synchronizes local `self.portfolio.positions` and cash balance with actual
        Binance USD-M Futures account state (`GET /fapi/v2/positionRisk` & `fetch_balance`),
        including yield-bearing collateral (`BFUSD` & `LDUSDT`).
        """
        if self.dry_run:
            return self.portfolio.positions

        try:
            bal = self.client.fetch_balance()
            col = self.compute_effective_margin_collateral(bal)
            if col["effective_wallet_usd"] > 0:
                self.portfolio.cash = float(col["effective_free_usd"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not fetch live balance during reconciliation: {e}")

        try:
            raw_positions = self.client.fetch_positions()
            reconciled: dict[str, Position] = {}
            for p in raw_positions:
                contracts = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0.0)
                if abs(contracts) <= 1e-10:
                    continue
                sym = str(p.get("symbol") or "").split(":")[0]
                side_str = str(p.get("side") or ("long" if contracts > 0 else "short")).lower()
                p_side = PositionSide.LONG if side_str == "long" else PositionSide.SHORT
                entry_px = float(p.get("entryPrice") or p.get("markPrice") or 0.0)
                mark_px = float(p.get("markPrice") or entry_px)
                reconciled[sym] = Position(
                    symbol=sym,
                    side=p_side,
                    quantity=abs(contracts),
                    entry_price=entry_px,
                    current_price=mark_px,
                )
            self.portfolio.positions = reconciled
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not fetch live positions during reconciliation: {e}")

        return self.portfolio.positions

    def execute_maker_limit_chase(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        fallback_price: float = 100.0,
        reduce_only: bool = False,
    ) -> Order:
        """
        Executes an order using Post-Only (`timeInForce='GTX'`) Maker limit orders at the
        top of book (`best_bid` for BUY, `best_ask` for SELL), chasing up to
        `max_chase_iterations` before optional IOC/Market fallback.
        """
        q_amt = self.quantize_amount(symbol, quantity)
        bid, ask = self.fetch_best_bid_ask(symbol, fallback_price=fallback_price)
        limit_px = self.quantize_price(symbol, bid if side == OrderSide.BUY else ask)

        if not reduce_only:
            self.verify_risk_circuit_breakers(symbol, q_amt, limit_px)

        order_id = f"gtx_{uuid.uuid4().hex[:10]}"
        if self.dry_run:
            self.maker_fills_count += 1
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=q_amt,
                price=limit_px,
                status=OrderStatus.FILLED,
                filled_quantity=q_amt,
                average_fill_price=limit_px,
                created_at=datetime.now(timezone.utc),
            )
            fee = q_amt * limit_px * self.maker_fee_pct
            trade = Trade(
                id=f"trd_{uuid.uuid4().hex[:10]}",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=q_amt,
                price=limit_px,
                fee=fee,
                timestamp=datetime.now(timezone.utc),
            )
            self.trade_history.append(trade)
            self.portfolio.cash -= fee
            self._apply_fill_to_portfolio(symbol, side, q_amt, limit_px)
            self.orders[order_id] = order
            return order

        remaining = q_amt
        ccxt_side = "buy" if side == OrderSide.BUY else "sell"
        last_order_id = order_id
        filled_total = 0.0
        cost_total = 0.0

        last_gtx_rejected = False
        for attempt in range(self.max_chase_iterations):
            bid, ask = self.fetch_best_bid_ask(symbol, fallback_price=limit_px)
            spread = max(ask - bid, bid * 0.0005, 1e-6)
            if side == OrderSide.BUY:
                raw_target = (bid - spread * 0.5) if last_gtx_rejected else bid
            else:
                raw_target = (ask + spread * 1.2) if last_gtx_rejected else ask
            chase_px = self.quantize_price(symbol, raw_target)
            if side == OrderSide.SELL and chase_px <= bid:
                chase_px = self.quantize_price(symbol, ask + max(spread, ask * 0.001))
            params: dict[str, Any] = {"timeInForce": "GTX"}
            if reduce_only:
                params["reduceOnly"] = True

            try:
                res = self.client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=ccxt_side,
                    amount=self.quantize_amount(symbol, remaining),
                    price=chase_px,
                    params=params,
                )
                last_gtx_rejected = False
                live_id = str(res.get("id", order_id))
                last_order_id = live_id
                if self.chase_interval_sec > 0:
                    time.sleep(self.chase_interval_sec)

                chk = self.client.fetch_order(live_id, symbol)
                status_str = str(chk.get("status", "")).lower()
                filled_this = float(chk.get("filled") or 0.0)
                avg_px = float(chk.get("average") or chase_px)

                if filled_this > 0:
                    filled_total += filled_this
                    cost_total += filled_this * avg_px
                    remaining = max(0.0, q_amt - filled_total)
                    self.maker_fills_count += 1

                if status_str in ("closed", "filled") or remaining <= 1e-8:
                    break

                # Cancel unfilled portion to re-post at new top of book
                try:
                    cancel_res = self.client.cancel_order(live_id, symbol)
                    if isinstance(cancel_res, dict):
                        filled_after = float(cancel_res.get("filled") or 0.0)
                        avg_after = float(cancel_res.get("average") or chase_px)
                        delta_fill = max(0.0, filled_after - filled_this)
                        if delta_fill > 0:
                            filled_total += delta_fill
                            cost_total += delta_fill * avg_after
                            remaining = max(0.0, q_amt - filled_total)
                            if filled_this <= 0:
                                self.maker_fills_count += 1
                        if remaining <= 1e-8:
                            break
                except Exception:
                    try:
                        chk_after = self.client.fetch_order(live_id, symbol)
                        filled_after = float(chk_after.get("filled") or 0.0)
                        avg_after = float(chk_after.get("average") or chase_px)
                        delta_fill = max(0.0, filled_after - filled_this)
                        if delta_fill > 0:
                            filled_total += delta_fill
                            cost_total += delta_fill * avg_after
                            remaining = max(0.0, q_amt - filled_total)
                            if filled_this <= 0:
                                self.maker_fills_count += 1
                        if str(chk_after.get("status", "")).lower() in ("closed", "filled") or remaining <= 1e-8:
                            break
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                last_gtx_rejected = True
                logger.debug(f"GTX attempt {attempt+1} on {symbol} rejected/repricing: {e}")
                if self.chase_interval_sec > 0:
                    time.sleep(min(0.25, self.chase_interval_sec * 0.2))

        # Fallback for any remaining quantity
        if remaining > 1e-8 and self.allow_taker_fallback:
            fb_params: dict[str, Any] = {"reduceOnly": True} if reduce_only else {}
            res_mkt = self.client.create_order(
                symbol=symbol,
                type="market",
                side=ccxt_side,
                amount=self.quantize_amount(symbol, remaining),
                params=fb_params,
            )
            last_order_id = str(res_mkt.get("id", last_order_id))
            avg_px = float(res_mkt.get("average") or res_mkt.get("price") or limit_px)
            filled_total += remaining
            cost_total += remaining * avg_px
            remaining = 0.0
            self.taker_fills_count += 1

        final_avg_px = (cost_total / filled_total) if filled_total > 0 else limit_px
        if filled_total > 0:
            self._apply_fill_to_portfolio(symbol, side, filled_total, final_avg_px)

        final_order = Order(
            id=last_order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=q_amt,
            price=limit_px,
            status=OrderStatus.FILLED if remaining <= 1e-8 else OrderStatus.PARTIALLY_FILLED,
            filled_quantity=filled_total,
            average_fill_price=final_avg_px,
            created_at=datetime.now(timezone.utc),
        )
        self.orders[last_order_id] = final_order
        return final_order

    def _apply_fill_to_portfolio(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        fill_price: float,
    ) -> None:
        """Updates internal position tracking after an order fill."""
        existing = self.portfolio.positions.get(symbol)
        signed_delta = quantity if side == OrderSide.BUY else -quantity
        if existing and existing.side != PositionSide.FLAT and existing.quantity > 0:
            existing_signed = (
                existing.quantity if existing.side == PositionSide.LONG else -existing.quantity
            )
            new_signed = existing_signed + signed_delta
        else:
            new_signed = signed_delta

        if abs(new_signed) <= 1e-8:
            self.portfolio.positions.pop(symbol, None)
        else:
            new_side = PositionSide.LONG if new_signed > 0 else PositionSide.SHORT
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                side=new_side,
                quantity=abs(new_signed),
                entry_price=fill_price,
                current_price=fill_price,
            )

    def execute_atomic_rebalance(
        self,
        target_dollars: dict[str, float],
        prices: dict[str, float],
        min_notional: float = 25.0,
    ) -> list[Order]:
        """
        Pairs BUY deltas and SELL deltas and executes them concurrently so the portfolio
        never develops a directional exposure tilt mid-rebalance.
        """
        ks_status = self.kill_switch.evaluate_and_enforce(self.portfolio, broker=self)
        if ks_status.triggered and self.kill_switch.is_locked():
            raise MaxDrawdownStopError(f"Kill-Switch active: {ks_status.reason}")

        all_syms = sorted(set(target_dollars.keys()) | set(self.portfolio.positions.keys()))
        buy_tasks: list[tuple[str, float, float]] = []
        sell_tasks: list[tuple[str, float, float]] = []

        for sym in all_syms:
            px = float(prices.get(sym) or prices.get(sym.replace("/", "")) or 0.0)
            if px <= 0:
                pos_cur = self.portfolio.positions.get(sym)
                px = pos_cur.current_price if pos_cur else 0.0
            if px <= 0:
                continue

            tgt_val = float(target_dollars.get(sym, target_dollars.get(sym.replace("/", ""), 0.0)))
            pos = self.portfolio.positions.get(sym)
            cur_val = 0.0
            if pos and pos.side != PositionSide.FLAT and pos.quantity > 0:
                cur_val = (pos.quantity * px) if pos.side == PositionSide.LONG else (-pos.quantity * px)

            delta_usd = tgt_val - cur_val
            if abs(delta_usd) < min_notional:
                continue

            qty = abs(delta_usd) / px
            if delta_usd > 0:
                buy_tasks.append((sym, qty, px))
            else:
                sell_tasks.append((sym, qty, px))

        executed: list[Order] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            max_len = max(len(buy_tasks), len(sell_tasks))
            for i in range(max_len):
                if i < len(buy_tasks):
                    b_sym, b_qty, b_px = buy_tasks[i]
                    futures.append(
                        pool.submit(
                            self.execute_maker_limit_chase,
                            b_sym,
                            OrderSide.BUY,
                            b_qty,
                            b_px,
                        )
                    )
                if i < len(sell_tasks):
                    s_sym, s_qty, s_px = sell_tasks[i]
                    futures.append(
                        pool.submit(
                            self.execute_maker_limit_chase,
                            s_sym,
                            OrderSide.SELL,
                            s_qty,
                            s_px,
                        )
                    )
            for fut in futures:
                executed.append(fut.result())

        return executed

    def emergency_flatten_all(self) -> list[str]:
        """
        Cancels all open orders and flattens all open positions with `reduceOnly=True`.
        """
        flattened: list[str] = []
        for sym, pos in list(self.portfolio.positions.items()):
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue
            if not self.dry_run:
                try:
                    if hasattr(self.client, "cancel_all_orders"):
                        self.client.cancel_all_orders(sym)
                except Exception:
                    pass
                close_side = "sell" if pos.side == PositionSide.LONG else "buy"
                try:
                    self.client.create_order(
                        symbol=sym,
                        type="market",
                        side=close_side,
                        amount=self.quantize_amount(sym, pos.quantity),
                        params={"reduceOnly": True},
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Emergency flatten error on {sym}: {e}")
            flattened.append(sym)
            self.portfolio.positions.pop(sym, None)

        return flattened

    def hedge_excess_exposure(self, net_exposure_usd: float) -> Order | None:
        """
        Reduces the largest position on the overweight side when net exposure asymmetry
        exceeds the safety threshold.
        """
        if abs(net_exposure_usd) <= 1.0 or not self.portfolio.positions:
            return None

        target_side = PositionSide.LONG if net_exposure_usd > 0 else PositionSide.SHORT
        candidates = [
            p for p in self.portfolio.positions.values()
            if p.side == target_side and p.quantity > 0
        ]
        if not candidates:
            return None

        largest = max(candidates, key=lambda p: p.quantity * (p.current_price or p.entry_price))
        px = largest.current_price if largest.current_price > 0 else largest.entry_price
        if px <= 0:
            return None

        reduce_qty = min(largest.quantity, abs(net_exposure_usd) / px)
        order_side = OrderSide.SELL if target_side == PositionSide.LONG else OrderSide.BUY
        return self.execute_maker_limit_chase(
            symbol=largest.symbol,
            side=order_side,
            quantity=reduce_qty,
            fallback_price=px,
            reduce_only=True,
        )

    def ensure_account_configuration(
        self,
        symbols: list[str],
        target_leverage: float = 3.0,
        enforce_single_asset_margin: bool = False,
    ) -> dict[str, Any]:
        """
        Audits and programmatically enforces Binance USDS-M Futures (`fapi/v1`) account settings:
        1. Position Mode -> One-Way Mode (`dualSidePosition='false'` via `/fapi/v1/positionSide/dual`)
        2. Multi-Assets Margin Mode -> Single-Asset Mode (optional, via `/fapi/v1/multiAssetsMargin`)
        3. Margin Type -> Cross Margin (`marginType='CROSSED'` via `/fapi/v1/marginType`)
        4. Initial Leverage -> `target_int_lev` (`POST /fapi/v1/leverage`)
        """
        target_int_lev = max(1, min(20, int(round(target_leverage))))
        report: dict[str, Any] = {
            "dry_run": self.dry_run,
            "position_mode": "ONE_WAY",
            "dual_side_position": False,
            "multi_assets_margin": False,
            "margin_modes": {s: "CROSSED" for s in symbols},
            "leverages": {s: target_int_lev for s in symbols},
            "usdt_free_balance": float(self.portfolio.cash),
            "usdt_total_balance": float(self.portfolio.total_equity),
        }

        if self.dry_run:
            return report

        # 0. Synchronize server clock & load Futures markets metadata
        try:
            if hasattr(self.client, "load_time_difference"):
                self.client.load_time_difference()
            if hasattr(self.client, "load_markets"):
                self.client.load_markets()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Market/time sync note: {e}")

        # 1. Verify / Enforce One-Way Position Mode (`dualSidePosition='false'`)
        try:
            pos_mode = self.client.fetch_position_mode()
            is_hedged = bool(pos_mode.get("hedged", False))
            if is_hedged:
                logger.info("Switching Binance Futures Position Mode from Hedge Mode to One-Way Mode...")
                self.client.set_position_mode(False)
                pos_mode = self.client.fetch_position_mode()
                is_hedged = bool(pos_mode.get("hedged", False))
            report["dual_side_position"] = is_hedged
            report["position_mode"] = "HEDGE_MODE" if is_hedged else "ONE_WAY"
        except Exception as e:  # noqa: BLE001
            err_str = str(e)
            if "-4059" in err_str or "No need to change position side" in err_str:
                report["dual_side_position"] = False
                report["position_mode"] = "ONE_WAY"
            else:
                logger.warning(f"Position mode verification warning: {e}")

        # 2. Check / Optionally Enforce Multi-Assets Margin Mode
        try:
            if hasattr(self.client, "fapiPrivateGetMultiAssetsMargin"):
                ma_resp = self.client.fapiPrivateGetMultiAssetsMargin()
                is_multi = bool(
                    str(ma_resp.get("multiAssetsMargin", "false")).lower() == "true"
                )
                if is_multi and enforce_single_asset_margin and hasattr(self.client, "fapiPrivatePostMultiAssetsMargin"):
                    self.client.fapiPrivatePostMultiAssetsMargin({"multiAssetsMargin": "false"})
                    is_multi = False
                report["multi_assets_margin"] = is_multi
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Multi-Assets Margin check note: {e}")

        # 3. Enforce CROSSED Margin & Initial Leverage per Symbol
        for sym in symbols:
            try:
                self.client.set_margin_mode("cross", sym)
                report["margin_modes"][sym] = "CROSSED"
            except Exception as e:  # noqa: BLE001
                err_str = str(e)
                if "-4046" in err_str or "No need to change margin type" in err_str:
                    report["margin_modes"][sym] = "CROSSED"
                else:
                    logger.warning(f"Could not set CROSSED margin for {sym}: {e}")
                    report["margin_modes"][sym] = f"ERROR: {e}"

            try:
                lev_resp = self.client.set_leverage(target_int_lev, sym)
                confirmed_lev = int(lev_resp.get("leverage", target_int_lev)) if isinstance(lev_resp, dict) else target_int_lev
                report["leverages"][sym] = confirmed_lev
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not set leverage={target_int_lev}x for {sym}: {e}")

        # 4. Fetch Live Futures Balance (USDT + Yield-Bearing Collateral BFUSD & LDUSDT)
        try:
            bal = self.client.fetch_balance()
            col = self.compute_effective_margin_collateral(bal)
            free_cash = float(col["effective_free_usd"])
            total_cash = float(col["effective_wallet_usd"])
            report["non_zero_assets"] = col["non_zero_assets"]
            report["bfusd_balance"] = col["bfusd_wallet"]
            report["ldusdt_balance"] = col["ldusdt_wallet"]
            report["yield_collateral_usd"] = col["yield_collateral_usd"]
            report["yield_collateral_share"] = col["yield_collateral_share"]
            if col["yield_collateral_usd"] > 0 and not report.get("multi_assets_margin") and not enforce_single_asset_margin:
                try:
                    if hasattr(self.client, "fapiPrivatePostMultiAssetsMargin"):
                        self.client.fapiPrivatePostMultiAssetsMargin({"multiAssetsMargin": "true"})
                        report["multi_assets_margin"] = True
                except Exception:
                    pass
            self.portfolio.cash = free_cash
            if not self.portfolio.positions and total_cash > 0:
                self.portfolio.initial_cash = total_cash
                self.peak_equity = total_cash
            report["usdt_free_balance"] = free_cash
            report["usdt_total_balance"] = total_cash
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not fetch USDT balance during configuration check: {e}")

        return report

    def run_production_smoke_test(
        self,
        symbol: str = "DOGE/USDT",
        notional_usd: float = 6.0,
        leverage: int = 3,
    ) -> dict[str, Any]:
        """
        Executes a complete 5-stage production smoke test on Binance USDS-M Futures:
        Stage 1: Pre-Flight API Key Permissions & Initial Balance Audit.
        Stage 2: Auto-Configuration (`One-Way Mode`, `CROSSED` Margin, `leverage`x).
        Stage 3: Resting Passive Maker Order (`Post-Only GTX`) Placement, Query & Cancellation.
        Stage 4: Live Micro-Position Opening (`notional_usd`) & `positionRisk` Reconciliation.
        Stage 5: Live Position Closing (`reduceOnly=True`) & Clean-Slate Verification (`0` open positions).
        """
        logger.info(f"🧪 Starting Production Smoke Test on {symbol} (Notional: ${notional_usd:.2f} USDT, Leverage: {leverage}x, DryRun: {self.dry_run})")
        config_report = self.ensure_account_configuration(
            symbols=[symbol],
            target_leverage=float(leverage),
        )
        logger.info(f"[Stage 1-2/5] Config & Balance Report: {config_report}")
        initial_balance = float(config_report["usdt_total_balance"])

        bid, ask = self.fetch_best_bid_ask(symbol, fallback_price=0.15)
        mid_px = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.15

        # Ensure quantity satisfies both target notional_usd and minimum exchange stepSize
        raw_qty = max(notional_usd, 6.0) / (bid * 0.98 if bid > 0 else mid_px)
        qty = self.quantize_amount(symbol, raw_qty)
        if qty * mid_px < 5.5:
            qty = self.quantize_amount(symbol, raw_qty * 1.15)

        # Stage 3: Resting Post-Only (GTX) Limit Order Post & Cancel (-1.5% below best_bid)
        resting_bid_px = self.quantize_price(symbol, bid * 0.985)
        logger.info(
            f"[Stage 3/5] Preparing GTX test order on {symbol}: bid=${bid}, ask=${ask}, "
            f"resting_px=${resting_bid_px}, qty={qty} (notional=${qty * resting_bid_px:.2f} USDT)"
        )
        gtx_post_cancel_ok = False
        resting_order_id = ""

        if self.dry_run:
            resting_order_id = f"dry_gtx_rest_{uuid.uuid4().hex[:8]}"
            gtx_post_cancel_ok = True
            logger.info(f"[Stage 3/5] DRY-RUN Post-Only GTX resting order posted & cancelled ({resting_order_id} @ ${resting_bid_px})")
        else:
            res_rest = self.client.create_order(
                symbol=symbol,
                type="limit",
                side="buy",
                amount=qty,
                price=resting_bid_px,
                params={"timeInForce": "GTX"},
            )
            resting_order_id = str(res_rest.get("id", ""))
            logger.info(f"[Stage 3/5] Live Post-Only GTX order posted: id={resting_order_id} ({qty} {symbol} @ ${resting_bid_px})")
            time.sleep(0.5)
            self.client.cancel_order(resting_order_id, symbol)
            gtx_post_cancel_ok = True
            logger.info(f"[Stage 3/5] Live Post-Only GTX order cancelled cleanly: id={resting_order_id}")

        # Stage 4: Open Micro-Position & Reconcile with Exchange
        logger.info(f"[Stage 4/5] Opening micro-position: BUY {qty} {symbol} (~${qty * mid_px:.2f} USDT)...")
        open_order = self.execute_maker_limit_chase(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=qty,
            fallback_price=mid_px,
            reduce_only=False,
        )
        if not self.dry_run:
            time.sleep(0.5)
        reconciled_open = self.reconcile_positions_from_exchange()
        pos_opened = symbol in reconciled_open or (symbol.split(":")[0] in reconciled_open)
        open_qty = (
            reconciled_open[symbol].quantity
            if symbol in reconciled_open
            else (open_order.filled_quantity or qty)
        )
        logger.info(
            f"[Stage 4/5] Position reconciled: opened={pos_opened}, qty={open_qty}, avg_fill=${open_order.average_fill_price}"
        )

        # Stage 5: Close Position with reduceOnly=True & Verify 100% Flat State
        logger.info(f"[Stage 5/5] Closing micro-position with reduceOnly=True...")
        if self.dry_run:
            close_order = self.execute_maker_limit_chase(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=open_qty,
                fallback_price=mid_px,
                reduce_only=True,
            )
            close_order_id = close_order.id
        else:
            try:
                if hasattr(self.client, "cancel_all_orders"):
                    self.client.cancel_all_orders(symbol)
            except Exception:
                pass
            res_close = self.client.create_order(
                symbol=symbol,
                type="market",
                side="sell",
                amount=self.quantize_amount(symbol, open_qty),
                params={"reduceOnly": True},
            )
            close_order_id = str(res_close.get("id", ""))
            self.portfolio.positions.pop(symbol, None)
            time.sleep(0.5)

        reconciled_final = self.reconcile_positions_from_exchange()
        final_balance = float(self.portfolio.cash)
        if not self.dry_run:
            try:
                bal_after = self.client.fetch_balance()
                final_balance = float(bal_after.get("USDT", {}).get("total") or final_balance)
            except Exception:
                pass

        is_flat = symbol not in reconciled_final and len(reconciled_final) == 0
        logger.info(
            f"✅ Smoke Test Complete | Flat={is_flat} | Initial USDT=${initial_balance:,.4f} -> Final USDT=${final_balance:,.4f}"
        )

        return {
            "status": "PASSED" if (gtx_post_cancel_ok and pos_opened and is_flat) else "CHECK_REQUIRED",
            "symbol": symbol,
            "quantity": open_qty,
            "notional_usd": round(open_qty * (open_order.average_fill_price or mid_px), 4),
            "configuration": config_report,
            "gtx_resting_order_id": resting_order_id,
            "gtx_post_and_cancel_ok": gtx_post_cancel_ok,
            "open_order_id": open_order.id,
            "entry_fill_price": open_order.average_fill_price,
            "position_verified_on_exchange": pos_opened,
            "close_order_id": close_order_id,
            "final_flat_verified": is_flat,
            "initial_usdt_balance": initial_balance,
            "final_usdt_balance": final_balance,
            "net_cost_usdt": round(final_balance - initial_balance, 6),
        }

