"""Narrow v5 ETF policy and saved-output admission; no account execution."""
from decimal import Context, ROUND_HALF_UP, localcontext

from ..core.contracts import Document, fields, integer, require
from ..core.etf_buy_hold import BUY_HOLD_VERSION, validate_buy_hold_policy
from ..core.portfolio import PORTFOLIO_VERSION, decimal
from .etf_grid import price_tick_for, validate_etf_grid
from .profiles import daily_open_profile

ROTATION_POLICY = dict(contract_version="etf_rotation_policy_v1", schedule="weekly_first_trading_session")


def validate_v5_plan(plan, profile, universe):
    policy = plan["portfolio_policy"]
    if policy == ROTATION_POLICY:
        require(plan["signal_frame"] is not None, "saved rotation Signal required")
    else:
        validate_buy_hold_policy(policy)
        require(plan["signal_frame"] is None and policy["entry_session"] == plan["start_session"] and
                policy["security_id"] in universe, "entry policy/Signal/scope mismatch")
    require(plan["initial_account"]["positions"] == {}, "new v5 ETF account starts with empty holdings")
    require(profile == daily_open_profile(unknown_status_policy=profile["unknown_status_policy"],
            price_limit_policy=profile["price_limit_policy"], slippage_bps=profile["slippage_bps"],
            price_grid_policy=profile["price_grid_policy"]), "changed ETF v2 execution policy")
    validate_etf_grid(profile, universe, plan["price_unit"])


def _validate_saved_entry(wire, policy):
    require(len(wire["decisions"]) == 1, "one entry decision required")
    decision = wire["decisions"][0]
    calendar = wire["plan"]["market_replay"]["calendar"]
    entry = policy["entry_session"]
    require(decision["contract_version"] == BUY_HOLD_VERSION and decision["signal_ref"] is None and
            decision["portfolio_policy_ref"] == wire["portfolio_policy_ref"] and
            decision["trade_session"] == entry and
            decision["reference_session"] == calendar[calendar.index(entry) - 1] and
            decision["selected_security_id"] == policy["security_id"], "saved entry decision differs")
    integer(decision["expected_account_version"])
    intents, orders, fills = decision["intents"], wire["orders"], wire["fills"]
    require(all(type(items) is list for items in (intents, orders, fills)) and
            decision["status"] in ("NO_DECISION", "DECISION_COMPLETE"), "saved entry decision shape differs")
    if decision["status"] == "NO_DECISION":
        require(decision["targets"] == {} and not intents, "saved no-entry decision differs")
    else:
        fields(decision["targets"], policy["security_id"])
        target = decision["targets"][policy["security_id"]]
        integer(target)
        require(target % 100 == 0 and bool(intents) == bool(target), "saved entry target differs")
    require(len(intents) <= 1 and len(orders) == len(intents) and len(fills) <= len(orders),
            "entry retry or missing order forbidden")
    if not intents:
        return
    intent, order = intents[0], orders[0]
    integer(intent["expected_account_version"]); integer(order["expected_account_version"])
    integer(order["quantity"], 1)
    require(intent["side"] == "BUY" and intent["security_id"] == policy["security_id"] and
            intent["valid_until"] == entry and
            intent["expected_account_version"] == decision["expected_account_version"], "saved entry intent differs")
    integer(intent["quantity"], 1)
    require(intent["quantity"] == decision["targets"][policy["security_id"]] and
            intent["quantity"] % 100 == 0 and all(order[key] == value for key, value in intent.items()) and
            order["session"] == entry and order["order_id"] == wire["run_id"] + ":order:0", "saved entry order differs")
    integer(order["filled_quantity"]); integer(order["unfilled_quantity"])
    filled = order["filled_quantity"]
    require(filled + order["unfilled_quantity"] == intent["quantity"] and filled % 100 == 0 and
            bool(fills) == bool(filled) and order["status"] ==
            ("EXPIRED" if not filled else "FILLED" if not order["unfilled_quantity"] else "PARTIAL_EXPIRED"),
            "saved entry fill quantity/status differs")
    if fills:
        fill = fills[0]
        integer(fill["quantity"], 1)
        require(fill["order_id"] == order["order_id"] and fill["fill_id"] == order["order_id"] + ":fill:0" and
                fill["side"] == "BUY" and fill["security_id"] == policy["security_id"] and
                fill["session"] == entry and fill["quantity"] == filled, "saved entry fill link differs")


def validate_saved_v5(wire):
    from .backtest import BacktestRequest, MarketReplay, _validate
    from .unit_splits import validate_saved_applications
    plan = wire["plan"]
    require(wire["runtime_version"] == "axiom.backtest/5" and plan["contract_version"] == "backtest_request_v5",
            "saved ETF v5 tuple mismatch")
    _validate(BacktestRequest.from_dict(plan))
    policy = plan["portfolio_policy"]
    hold = policy["contract_version"] == "etf_buy_and_hold_policy_v1"
    require(wire["core_version"] == (BUY_HOLD_VERSION if hold else PORTFOLIO_VERSION) and
            wire["portfolio_policy_ref"] == Document.from_dict(policy).identity and
            wire["signal_ref"] == (None if hold else plan["signal_frame"]["signal_run_ref"]) and
            wire["market_ref"] == MarketReplay.from_dict(plan["market_replay"]).identity and
            wire["profile_ref"] == Document.from_dict(plan["profile"]).identity and
            wire["price_unit"] == plan["price_unit"] == "CNY/fund unit" and wire["quantity_unit"] == "fund units",
            "saved ETF v5 policy/input/unit binding mismatch")
    if hold:
        _validate_saved_entry(wire, policy)
    profile = plan["profile"]
    for fill in wire["fills"]:
        require(fill["price_grid_ref"] == profile["price_grid_ref"] and
                fill["price_tick"] == price_tick_for(profile, fill["security_id"]) and
                fill["price_rounding"] == "adverse_tick", "saved ETF fill grid differs")
        for name in ("raw_slipped_price", "rounding_delta", "effective_slippage_bps"):
            decimal(fill[name])
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            require(decimal(fill["price"]) > 0 and decimal(fill["price"]) % decimal(fill["price_tick"]) == 0 and
                    decimal(fill["effective_slippage_bps"]) >= decimal(profile["slippage_bps"]), "saved ETF fill price differs")
    validate_saved_applications(wire)
