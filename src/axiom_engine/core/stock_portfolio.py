"""Pure weekly Top5 planning over unchanged, saved Research predictions."""
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
import re

from .contracts import Document, digest, fields, integer, number, require, session, text
from .portfolio import PortfolioDecision, decimal, minor

STOCK_PORTFOLIO_VERSION = "axiom.stock_portfolio/1"
BUDGET_BASIS = "available_cash_plus_previous_close_positions_excluding_receivables"


class StockPredictionFrame(Document):
    """Raw normalized-target predictions; scores are not return percentages."""


def instant(value):
    require(type(value) is str and "T" in value, "aware timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid aware timestamp") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None, "aware timestamp required")
    return result


def source_ref(value):
    require(type(value) is str and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value), "invalid owner source ref")


def validate_stock_predictions(frame):
    wire = frame.to_dict()
    fields(wire, "contract_version signal_run_ref signal_stage score_semantics score_unit feature_ref model_ref limitations universe rows")
    require(wire["contract_version"] == "stock_prediction_run_v1" and wire["signal_stage"] == "prediction_raw" and
            wire["score_semantics"] == "forward_5_session_cs_zscore_prediction" and wire["score_unit"] == "dimensionless",
            "unsupported stock prediction semantics")
    for name in ("signal_run_ref", "feature_ref", "model_ref"):
        digest(wire[name])
    require(type(wire["limitations"]) is list and all(type(x) is str for x in wire["limitations"]), "prediction limitations required")
    universe = wire["universe"]
    require(type(universe) is list and bool(universe) and len(set(universe)) == len(universe), "explicit prediction union required")
    for security in universe:
        text(security)
    require(type(wire["rows"]) is list and bool(wire["rows"]), "prediction rows required")
    indexed, groups = {}, {}
    for row in wire["rows"]:
        fields(row, "security_id session knowledge_cutoff available_at score valid invalid_reason source_refs member")
        require(row["security_id"] in universe, "prediction outside union")
        session(row["session"])
        cutoff, available = instant(row["knowledge_cutoff"]), instant(row["available_at"])
        require(cutoff == instant(row["session"] + "T20:30:00+08:00") and available <= cutoff,
                "prediction clock conflict")
        require(type(row["valid"]) is bool and type(row["member"]) is bool, "explicit validity/member flags required")
        if row["valid"]:
            number(row["score"])
            require(row["invalid_reason"] is None, "valid prediction has invalid reason")
        else:
            require(row["score"] is None, "invalid prediction cannot carry score")
            text(row["invalid_reason"])
        require(type(row["source_refs"]) is list and bool(row["source_refs"]), "prediction source refs required")
        for ref in row["source_refs"]:
            source_ref(ref)
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate prediction key")
        indexed[key] = row
        groups.setdefault(row["session"], set()).add(row["security_id"])
    require(all(keys == set(universe) for keys in groups.values()), "incomplete prediction union coverage")
    return wire, indexed


def plan_stock_portfolio(frame, *, account, context):
    """Equal deployable-wealth targets; no execution facts, receivables or I/O."""
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _plan(frame, account, context)


def _plan(frame, account, context):
    wire, rows = validate_stock_predictions(frame)
    fields(context, "trade_session feature_session decision_time knowledge_cutoff reference_prices lot_size commission_rate minimum_commission_minor slippage_bps account_state_version supported_security_ids supported_universe_ref")
    session(context["trade_session"]); session(context["feature_session"])
    cutoff, decision = instant(context["knowledge_cutoff"]), instant(context["decision_time"])
    require(context["feature_session"] < context["trade_session"] and
            cutoff == instant(context["feature_session"] + "T20:30:00+08:00") and
            decision == instant(context["trade_session"] + "T08:55:00+08:00"), "invalid stock decision clock")
    supported = context["supported_security_ids"]
    require(type(supported) is list and bool(supported) and len(set(supported)) == len(supported) and
            set(supported) <= set(wire["universe"]), "explicit supported prediction subset required")
    digest(context["supported_universe_ref"])
    fields(account, "cash_minor positions version")
    integer(account["cash_minor"]); integer(account["version"])
    require(account["version"] == context["account_state_version"], "account version conflict")
    require(type(account["positions"]) is dict, "positions required")
    for security, position in account["positions"].items():
        require(security in supported, "held security outside execution scope")
        fields(position, "quantity sellable_quantity")
        integer(position["quantity"]); integer(position["sellable_quantity"])
        require(position["sellable_quantity"] <= position["quantity"], "invalid sellable quantity")
    require(context["lot_size"] == 100, "stock buy lot must be 100 shares")
    integer(context["minimum_commission_minor"])
    require(decimal(context["commission_rate"], minimum=0) <= 1, "invalid commission rate")
    require(decimal(context["slippage_bps"], minimum=0) < 10000, "invalid slippage")
    batch = [rows.get((context["feature_session"], security)) for security in wire["universe"]]
    require(all(row is not None for row in batch), "missing previous-session prediction")
    for row in batch:
        require(instant(row["knowledge_cutoff"]) == cutoff and instant(row["available_at"]) <= decision,
                "future/inconsistent prediction cutoff")
    eligible = [row for row in batch if row["member"] and row["security_id"] in supported]
    result = {"contract_version": STOCK_PORTFOLIO_VERSION, "feature_session": context["feature_session"],
        "trade_session": context["trade_session"], "signal_ref": wire["signal_run_ref"],
        "supported_universe_ref": context["supported_universe_ref"], "expected_account_version": account["version"],
        "status": "DECISION_COMPLETE", "selected_security_ids": [], "targets": {}, "intents": [], "trace": []}
    invalid = [row for row in eligible if not row["valid"]]
    if invalid or len(eligible) < 5:
        result["status"] = "NO_DECISION"
        result["trace"] = [{"reason": "INVALID_SIGNAL", "security_id": row["security_id"], "detail": row["invalid_reason"]} for row in invalid]
        if len(eligible) < 5:
            result["trace"].append({"reason": "INSUFFICIENT_ELIGIBLE_MEMBERS", "count": len(eligible)})
        return PortfolioDecision.from_dict(result)
    selected = [row["security_id"] for row in sorted(eligible, key=lambda row: (-row["score"], row["security_id"]))[:5]]
    result["selected_security_ids"] = selected
    needed = set(selected) | set(account["positions"])
    prices = {}
    for security in sorted(needed):
        quote = context["reference_prices"].get(security)
        require(quote is not None, "missing stock previous-close reference")
        fields(quote, "price session available_at source_refs")
        require(quote["session"] == context["feature_session"] and instant(quote["available_at"]) <= cutoff,
                "future or stale stock sizing reference")
        require(type(quote["source_refs"]) is list and bool(quote["source_refs"]), "reference provenance required")
        for ref in quote["source_refs"]:
            source_ref(ref)
        prices[security] = decimal(quote["price"], minimum=0)
        require(prices[security] > 0, "positive reference price required")
    budget = Decimal(account["cash_minor"]) + sum(Decimal(position["quantity"]) * prices[security] * 100
                                                  for security, position in account["positions"].items())
    for security in sorted(needed):
        result["targets"][security] = int(budget / 5 / (prices[security] * 100)) // 100 * 100 if security in selected else 0
    for security, position in sorted(account["positions"].items()):
        reduction = max(0, position["quantity"] - result["targets"].get(security, 0))
        quantity = min(reduction, position["sellable_quantity"])
        if quantity:
            result["intents"].append({"security_id": security, "side": "SELL", "quantity": quantity})
        if reduction > quantity:
            result["trace"].append({"reason": "T_PLUS_ONE_OR_UNSELLABLE", "security_id": security, "quantity": reduction - quantity})
    for security in sorted(selected):
        current = account["positions"].get(security, {"quantity": 0})["quantity"]
        quantity = max(0, result["targets"][security] - current) // 100 * 100
        if quantity:
            result["intents"].append({"security_id": security, "side": "BUY", "quantity": quantity})
    result["trace"].append({"reason": "RAW_TOP5", "eligible_count": len(eligible), "tie_break": "security_id_asc",
        "budget_basis": BUDGET_BASIS, "reference_budget_minor": minor(budget), "sizing": "previous_native_close",
        "score_semantics": wire["score_semantics"], "cash_check": "actual_fill_cash"})
    identity = Document.from_dict({"contract": STOCK_PORTFOLIO_VERSION, "frame": wire, "context": context, "account": account}).identity
    for index, intent in enumerate(result["intents"]):
        intent.update(intent_id=identity + ":" + str(index), expected_account_version=account["version"], valid_until=context["trade_session"])
    return PortfolioDecision.from_dict(result)
