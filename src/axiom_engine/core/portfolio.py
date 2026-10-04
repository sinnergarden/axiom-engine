"""Pure, explicit Top1 rotation decisions. No execution prices, I/O or clock."""
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

from .contracts import (Document, digest, fields, integer, number, require,
                        session, text, timestamp)

PORTFOLIO_VERSION = "axiom.rotation/1"


class SignalFrame(Document):
    """Final Research scores; construction alone is not execution admission."""


class PortfolioDecision(Document):
    """Targets, account-version-bound intents and explanations, never fills."""


def decimal(value, *, minimum=None):
    require(type(value) is str, "decimal values must be strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    require(result.is_finite() and (minimum is None or result >= minimum),
            "invalid decimal range")
    return result


def minor(value):
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate_signals(frame):
    wire = frame.to_dict()
    fields(wire, "contract_version signal_run_ref signal_stage score_semantics universe rows")
    require(wire["contract_version"] == "signal_frame_v1" and
            wire["signal_stage"] == "final" and
            wire["score_semantics"] == "momentum_20d", "unsupported signal contract/stage")
    digest(wire["signal_run_ref"])
    universe = wire["universe"]
    require(type(universe) is list and bool(universe) and
            len(set(universe)) == len(universe), "unique explicit signal universe required")
    for security in universe:
        text(security)
    require(type(wire["rows"]) is list, "signal rows required")
    indexed = {}
    for row in wire["rows"]:
        fields(row, "security_id session knowledge_cutoff available_at score valid invalid_reason source_refs")
        require(row["security_id"] in universe, "signal outside universe")
        session(row["session"])
        timestamp(row["knowledge_cutoff"]); timestamp(row["available_at"])
        require(row["available_at"] <= row["knowledge_cutoff"], "signal unavailable at cutoff")
        require(row["knowledge_cutoff"][:10] == row["session"], "cutoff must be on feature session")
        require(type(row["valid"]) is bool, "signal validity required")
        if row["valid"]:
            number(row["score"])
            require(row["invalid_reason"] is None, "valid signal has invalid reason")
        else:
            require(row["score"] is None, "invalid signal cannot carry score")
            text(row["invalid_reason"])
        require(type(row["source_refs"]) is list and bool(row["source_refs"]), "signal refs required")
        for ref in row["source_refs"]:
            digest(ref)
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate signal key")
        indexed[key] = row
    return wire, indexed


def plan_rotation(frame, *, account, context):
    """Top1 positive score, weekly caller schedule, previous-close lot sizing.

    Estimated sale proceeds can fund a planned buy. Runtime must still constrain
    execution by actual sell fills and cash. Invalid coverage produces a traced
    no-decision, never an implicit zero score or liquidation.
    """
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _plan_rotation(frame, account, context)


def _plan_rotation(frame, account, context):
    wire, rows = validate_signals(frame)
    fields(context, "trade_session feature_session decision_time knowledge_cutoff reference_prices lot_size commission_rate minimum_commission_minor tax_rate slippage_bps account_state_version")
    session(context["trade_session"]); session(context["feature_session"])
    timestamp(context["decision_time"]); timestamp(context["knowledge_cutoff"])
    require(context["feature_session"] < context["trade_session"] and
            context["knowledge_cutoff"] < context["decision_time"] and
            context["decision_time"][:10] == context["trade_session"], "invalid decision clock")
    fields(account, "cash_minor positions version")
    integer(account["cash_minor"]); integer(account["version"])
    require(account["version"] == context["account_state_version"], "account version conflict")
    require(type(account["positions"]) is dict, "positions required")
    for security, position in account["positions"].items():
        require(security in wire["universe"], "held security outside supported universe")
        fields(position, "quantity sellable_quantity")
        integer(position["quantity"]); integer(position["sellable_quantity"])
        require(position["sellable_quantity"] <= position["quantity"], "invalid sellable quantity")
    integer(context["lot_size"], 1)
    integer(context["minimum_commission_minor"])
    rate = decimal(context["commission_rate"], minimum=0)
    decimal(context["tax_rate"], minimum=0); decimal(context["slippage_bps"], minimum=0)
    require(rate <= 1, "invalid commission rate")
    result = {"contract_version": PORTFOLIO_VERSION, "feature_session": context["feature_session"],
              "trade_session": context["trade_session"], "signal_ref": wire["signal_run_ref"],
              "expected_account_version": account["version"], "status": "DECISION_COMPLETE",
              "selected_security_id": None, "targets": {}, "intents": [], "trace": []}
    batch = [rows.get((context["feature_session"], sec)) for sec in wire["universe"]]
    require(all(row is not None for row in batch), "incomplete decision signal coverage")
    for row in batch:
        require(row["knowledge_cutoff"] == context["knowledge_cutoff"] and
                row["available_at"] <= context["decision_time"], "future/inconsistent signal cutoff")
    if any(not row["valid"] for row in batch):
        result["status"] = "NO_DECISION"
        result["trace"] = [{"reason": "INVALID_SIGNAL", "security_id": row["security_id"],
                            "detail": row["invalid_reason"]} for row in batch if not row["valid"]]
        return PortfolioDecision.from_dict(result)
    candidates = sorted((row for row in batch if row["score"] > 0),
                        key=lambda row: (-row["score"], row["security_id"]))
    selected = candidates[0]["security_id"] if candidates else None
    result["selected_security_id"] = selected
    prices = {}
    needed = set(account["positions"]) | ({selected} if selected else set())
    for security in sorted(needed):
        quote = context["reference_prices"].get(security)
        require(quote is not None, "missing held/candidate reference price")
        fields(quote, "price session available_at source_refs")
        session(quote["session"]); timestamp(quote["available_at"])
        require(quote["session"] <= context["feature_session"] and
                quote["available_at"] <= context["knowledge_cutoff"], "future reference price")
        require(bool(quote["source_refs"]), "reference price refs required")
        for ref in quote["source_refs"]:
            digest(ref)
        price = decimal(quote["price"], minimum=0)
        require(price > 0, "positive reference price required")
        prices[security] = price
    wealth = Decimal(account["cash_minor"]) + sum(
        Decimal(p["quantity"]) * prices[s] * 100 for s, p in account["positions"].items())
    lot = context["lot_size"]
    target = int(wealth / (prices[selected] * 100)) // lot * lot if selected else 0
    for security in sorted(needed):
        result["targets"][security] = target if security == selected else 0
    for security in sorted(account["positions"]):
        position = account["positions"][security]
        desired = result["targets"].get(security, 0)
        quantity = min(max(0, position["quantity"] - desired), position["sellable_quantity"])
        if quantity:
            result["intents"].append({"security_id": security, "side": "SELL", "quantity": quantity})
        if position["quantity"] - desired > quantity:
            result["trace"].append({"security_id": security, "reason": "T_PLUS_ONE_OR_UNSELLABLE",
                                    "quantity": position["quantity"] - desired - quantity})
    if selected:
        current = account["positions"].get(selected, {"quantity": 0})["quantity"]
        quantity = max(0, target - current) // lot * lot
        if quantity:
            result["intents"].append({"security_id": selected, "side": "BUY", "quantity": quantity})
    result["trace"].append({"reason": "POSITIVE_TOP1" if selected else "NO_POSITIVE_SCORE",
                            "reference_nav_minor": minor(wealth), "tie_break": "security_id_asc",
                            "sizing": "previous_observed_close", "cash_check": "actual_fill_cash"})
    decision_id = PortfolioDecision.from_dict({"account": account, "context": context,
                                               "signal_ref": wire["signal_run_ref"]}).identity
    for index, intent in enumerate(result["intents"]):
        intent.update({"intent_id": decision_id + ":" + str(index),
                       "expected_account_version": account["version"], "valid_until": context["trade_session"]})
    return PortfolioDecision.from_dict(result)
