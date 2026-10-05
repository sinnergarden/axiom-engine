"""Narrow v5 ETF policy and saved-output admission; no account execution."""
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
        require(len(wire["decisions"]) == 1, "one entry decision required")
        decision = wire["decisions"][0]
        require(decision["contract_version"] == BUY_HOLD_VERSION and decision["signal_ref"] is None and
                decision["portfolio_policy_ref"] == wire["portfolio_policy_ref"] and
                decision["trade_session"] == policy["entry_session"], "saved entry decision differs")
        for intent in decision["intents"]:
            require(intent["side"] == "BUY" and intent["security_id"] == policy["security_id"] and
                    intent["valid_until"] == policy["entry_session"], "saved entry intent differs")
            integer(intent["quantity"], 1)
            require(intent["quantity"] % 100 == 0, "entry lot differs")
        require(len(decision["intents"]) <= 1 and all(o["session"] == policy["entry_session"] for o in wire["orders"]),
                "entry retry forbidden")
    profile = plan["profile"]
    for fill in wire["fills"]:
        require(fill["price_grid_ref"] == profile["price_grid_ref"] and
                fill["price_tick"] == price_tick_for(profile, fill["security_id"]) and
                fill["price_rounding"] == "adverse_tick", "saved ETF fill grid differs")
        for name in ("raw_slipped_price", "rounding_delta", "effective_slippage_bps"):
            decimal(fill[name])
        require(decimal(fill["price"]) > 0 and decimal(fill["price"]) % decimal(fill["price_tick"]) == 0 and
                decimal(fill["effective_slippage_bps"]) >= decimal(profile["slippage_bps"]), "saved ETF fill price differs")
    validate_saved_applications(wire)
