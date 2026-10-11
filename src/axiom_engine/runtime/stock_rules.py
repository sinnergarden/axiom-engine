"""Pure frozen CSI300 policy and date-effective stock fee factories."""
from copy import deepcopy

from ..core.contracts import Document, fields, require
from ..core.portfolio import decimal
from ..core.stock_portfolio import BUDGET_BASIS, validate_top_k
from ..core.stock_rules import (stock_execution_rules, validate_execution_rules, interval_at,
    _period, _sources, _intervals, CSI300_ELIGIBILITY, CANDIDATE_POLICY)

FEE_LIMITATIONS = ["Fees apply only within the explicit source-verified window; an open effective interval does not verify the future.",
                  "Broker commission and its minimum are separate declared model assumptions."]


def validate_fee_schedule(wire):
    fields(wire, "contract_version currency money_unit intervals sources verified_from verified_through limitations")
    require(wire["contract_version"] == "stock_fee_schedule_v1" and wire["currency"] == "CNY" and
            wire["money_unit"] == "CNY_fen" and wire["limitations"] == FEE_LIMITATIONS, "unsupported stock fee schedule")
    _period(wire["verified_from"], wire["verified_through"])
    require(type(wire["intervals"]) is list and bool(wire["intervals"]), "frozen stock fee intervals required")
    for row in wire["intervals"]:
        fields(row, "effective_from effective_to sell_stamp_tax_rate transfer_fee_rate source_keys")
        for name in ("sell_stamp_tax_rate", "transfer_fee_rate"):
            require(decimal(row[name], minimum=0) <= 1, "invalid stock fee rate")
    intervals = _intervals(wire["intervals"], _sources(wire["sources"]))[None]
    interval_at(intervals, wire["verified_from"], label="stock fee")
    interval_at(intervals, wire["verified_through"], label="stock fee")
    for left, right in zip(intervals, intervals[1:]):
        if left["effective_to"] < wire["verified_through"] and right["effective_from"] > wire["verified_from"]:
            require(left["effective_to"] == right["effective_from"], "gap in verified stock fee window")
    return intervals


def stock_fee_schedule(*, intervals: list[dict], sources: list[dict], verified_from: str, verified_through: str) -> dict:
    wire = deepcopy(dict(contract_version="stock_fee_schedule_v1", currency="CNY", money_unit="CNY_fen",
        intervals=intervals, sources=sources, verified_from=verified_from, verified_through=verified_through,
        limitations=FEE_LIMITATIONS))
    validate_fee_schedule(wire)
    return wire


def fee_at(wire, day):
    require(wire["verified_from"] <= day <= wire["verified_through"], "stock fill outside verified fee window")
    return interval_at(wire["intervals"], day, label="stock fee")


def csi300_stock_portfolio_policy(*, top_k: int, execution_universe: list[str], execution_rules: dict) -> dict:
    validate_execution_rules(execution_rules)
    require(type(execution_universe) is list and execution_universe == execution_rules["universe"],
            "full frozen CSI300 execution union required")
    validate_top_k(top_k, execution_universe)
    return dict(eligibility_id=CSI300_ELIGIBILITY, top_k=top_k, rebalance="weekly_first_trading_session",
                budget_basis=BUDGET_BASIS, candidate_policy=CANDIDATE_POLICY,
                stock_execution_rules_ref=Document.from_dict(execution_rules).identity)


def strategy_support_ref(universe, universe_id):
    from ..core.stock_strategy import strategy_support_ref as core_support_ref
    return core_support_ref(universe, universe_id)


def strategy_stock_portfolio_policy(*, strategy_spec, universe_id, execution_universe, execution_rules):
    """The original PIT membership query must independently admit this pool ID."""
    from ..core.stock_strategy import validate_strategy_spec
    validate_execution_rules(execution_rules)
    require(type(execution_universe) is list and execution_universe == execution_rules["universe"],
            "complete frozen strategy execution union required")
    validate_strategy_spec(strategy_spec)
    validate_top_k(strategy_spec["top_k"], execution_universe)
    strategy_support_ref(execution_universe, universe_id)
    return dict(contract_version="stock_portfolio_policy_v4", eligibility_id="pit_a_share_strategy_v1",
                universe_id=universe_id, top_k=strategy_spec["top_k"], rebalance="weekly_first_trading_session",
                budget_basis=BUDGET_BASIS, candidate_policy=CANDIDATE_POLICY,
                stock_execution_rules_ref=Document.from_dict(execution_rules).identity,
                strategy_spec=deepcopy(strategy_spec), strategy_spec_ref=Document.from_dict(strategy_spec).identity)


def validate_full_stock_policy(policy, *, execution_universe, execution_rules):
    if policy.get("contract_version") == "stock_portfolio_policy_v4":
        expected = strategy_stock_portfolio_policy(strategy_spec=policy.get("strategy_spec"),
            universe_id=policy.get("universe_id"), execution_universe=execution_universe, execution_rules=execution_rules)
    else:
        expected = csi300_stock_portfolio_policy(top_k=policy.get("top_k"), execution_universe=execution_universe,
                                               execution_rules=execution_rules)
    require(policy == expected, "stock portfolio policy identity mismatch")
    return policy


def portfolio_support_ref(policy, universe):
    from ..core.stock_rules import support_ref
    return (strategy_support_ref(universe, policy["universe_id"])
            if policy.get("contract_version") == "stock_portfolio_policy_v4" else support_ref(universe))


def stock_core_version(policy):
    return "axiom.stock_portfolio/4" if policy.get("contract_version") == "stock_portfolio_policy_v4" else "axiom.stock_portfolio/3"
