"""Validate saved strategy identities and clocks without calling a planner."""
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction

from ..core.contracts import Document, fields, integer, require
from ..core.portfolio import decimal
from ..core.stock_portfolio import instant
from ..core.stock_strategy import validate_strategy_state


def validate_saved_strategy_decision(decision, policy, universe, previous_state, positions):
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _validate(decision, policy, universe, previous_state, positions)


def _validate(decision, policy, universe, previous_state, positions):
    targets = decision.get("strategy_targets")
    fields(targets, "contract_version strategy_spec_ref status selected_security_ids target_weights allocation_ratios preserve_quantities force_exit_security_ids next_state trace")
    require(targets["contract_version"] == "stock_strategy_targets_v1" and
            decision.get("strategy_spec_ref") == targets["strategy_spec_ref"] == policy["strategy_spec_ref"] and
            decision.get("strategy_targets_ref") == Document.from_dict(targets).identity and
            decision.get("strategy_state_before") == previous_state and
            decision.get("strategy_state_before_ref") == Document.from_dict(previous_state).identity and
            decision.get("strategy_state_after_ref") == Document.from_dict(targets["next_state"]).identity,
            "saved strategy identity/state mismatch")
    validate_strategy_state(targets["next_state"], universe)
    require(targets["next_state"]["check_index"] == previous_state["check_index"]+1, "saved strategy check index mismatch")
    weights, preserve, forced = targets["target_weights"], targets["preserve_quantities"], targets["force_exit_security_ids"]
    require(type(weights) is dict and type(preserve) is dict and type(forced) is list and len(set(forced)) == len(forced) and
            (set(weights) | set(preserve) | set(forced)) <= set(universe) and
            not ((set(weights) | set(preserve)) & set(forced)) and not (set(weights) & set(preserve)),
            "saved strategy target domains conflict")
    require(all(type(v) is str and decimal(v) >= 0 for v in weights.values()), "invalid saved target weights")
    allocations = targets["allocation_ratios"]
    require(type(allocations) is dict and set(allocations) == set(weights), "saved allocation ratio scope mismatch")
    ratios = []
    for s, allocation in allocations.items():
        fields(allocation, "numerator denominator")
        integer(allocation["numerator"]); integer(allocation["denominator"], 1)
        ratio = Fraction(allocation["numerator"], allocation["denominator"])
        require(allocation == dict(numerator=ratio.numerator, denominator=ratio.denominator) and
                decimal(weights[s]) == Decimal(ratio.numerator)/ratio.denominator,
                "saved exact allocation ratio/weight mismatch")
        ratios.append(ratio)
    require(sum(ratios, Fraction(0)) <= 1, "saved allocation ratios exceed budget")
    for value in preserve.values(): integer(value)
    require(all(decision.get("targets", {}).get(s, 0) == 0 for s in forced) and
            all(intent["security_id"] not in set(forced) | set(preserve) for intent in decision["intents"] if intent["side"] == "BUY"),
            "saved exit/preserved target was repurchased")
    require(type(targets["selected_security_ids"]) is list and
            len(set(targets["selected_security_ids"])) == len(targets["selected_security_ids"]) <= policy["top_k"] and
            set(targets["selected_security_ids"]) <= set(universe) and
            not (set(targets["selected_security_ids"]) & set(forced)) and
            targets["status"] in ("NO_DECISION", "DECISION_COMPLETE"), "invalid saved strategy selection")
    if decision["status"] == "DECISION_COMPLETE":
        require(all(decision["targets"].get(s, 0) == q for s,q in preserve.items()), "saved preserved quantity changed")
    feature = decision["feature_session"]
    prior = {p["security_id"]: p for p in positions if p["session"] == feature}
    for item in targets["trace"]:
        if item.get("reason") not in ("WEEKLY_CLOSE_TAKE_PROFIT", "WEEKLY_CLOSE_STOP_LOSS"):
            continue
        s, observed = item["security_id"], item["observation"]
        p = prior.get(s)
        require(p is not None and observed["session"] == feature == p["mark_session"] and
                observed["price_basis"] == "original_session_units" and observed["price"] == p["mark_price"] and
                observed["cost_minor"] == p["cost_minor"] and
                observed["owned_quantity"] == p["quantity"]+p.get("pending_share_quantity", 0) and
                observed["committed_sequence"] == p["committed_sequence"] and
                observed["source_refs"] == p["mark_source_refs"] and
                instant(observed["available_at"]) <= instant(feature+"T20:30:00+08:00"),
                "saved weekly-close exit does not match prior equity/price units")
        value = decimal(observed["price"])*observed["owned_quantity"]*100/observed["cost_minor"]-1
        require(decimal(item["observed_return"]) == value, "saved exit return does not reconcile")
        overlay = policy["strategy_spec"]["exit_overlay"]
        require(overlay is not None and s in forced and
                (value >= decimal(overlay["take_profit"]) if item["reason"] == "WEEKLY_CLOSE_TAKE_PROFIT" else
                 value <= -decimal(overlay["stop_loss"])), "saved exit threshold mismatch")
    return targets["next_state"]


def validate_saved_strategy_capacity(wire, rows):
    """Count actual shares, pending grants and registered stock-right slots."""
    capacity = wire["request_manifest"]["portfolio_policy"]["top_k"]
    quantities, pending, reservations = {}, {}, {}
    events = wire["account_events"].get("equity_facts", {}).get("actions", [])
    actions = {a["economic_event_id"]: a for a in events}
    for row in rows["position_ledger"]:
        s, event = row["security_id"], row["source_event_id"]
        quantities[s] = quantities.get(s, 0)+row["quantity_delta"]
        pending[s] = pending.get(s, 0)+row.get("pending_share_delta", 0)
        if row["reason"] == "EQUITY_REGISTER" and row.get("record_quantity", 0):
            total = actions[event]["stock_distribution_shares_per_share"]
            if total is None or decimal(total) > 0: reservations[event] = s
        elif row["reason"] in ("EQUITY_EX", "EQUITY_LISTING"):
            reservations.pop(event, None)
        owned = {s for s,q in quantities.items() if q} | {s for s,q in pending.items() if q} | set(reservations.values())
        require(len(owned) <= capacity, "saved account exceeds owned-position capacity")
        if not quantities[s]: quantities.pop(s)
        if not pending[s]: pending.pop(s)
