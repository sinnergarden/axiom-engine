"""Pure weekly TopK planning over unchanged, saved Research predictions."""
from datetime import datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
import re

from .contracts import Document, digest, fields, integer, number, require, session, text
from .portfolio import PortfolioDecision, decimal, minor
from .stock_rules import (validate_execution_rules, rule_at, listed, floor_quantity, legal_quantity,
                          support_ref as csi300_support_ref, CSI300_ELIGIBILITY, CANDIDATE_POLICY)

STOCK_PORTFOLIO_VERSION = "axiom.stock_portfolio/1"
TOPK_PORTFOLIO_VERSION = "axiom.stock_portfolio/2"
CSI300_PORTFOLIO_VERSION = "axiom.stock_portfolio/3"
BUDGET_BASIS = "available_cash_plus_previous_close_positions_excluding_receivables"


def validate_top_k(top_k, supported):
    require(type(top_k) is int and 1 <= top_k <= len(supported),
            "top_k must be a positive integer within the frozen execution universe")


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


def _prediction_clock_v2(row, cutoff, available):
    feature_cutoff = instant(row["feature_knowledge_cutoff"])
    model_available = instant(row["simulated_model_available_at"])
    start = instant(row["session"] + "T00:00:00+08:00")
    require(start <= feature_cutoff < start + timedelta(days=1) and
            start <= cutoff < start + timedelta(days=1), "prediction/feature clock belongs to another session")
    require(available == cutoff and feature_cutoff <= cutoff and model_available < cutoff,
            "v2 simulated prediction clock conflict")
    feature_available = row["feature_available_at"]
    require(feature_available is not None or row["valid"] is False, "valid prediction lacks Feature availability")
    if feature_available is not None:
        require(instant(feature_available) <= feature_cutoff, "Feature dependency unavailable at its original cutoff")
    return feature_cutoff, model_available


def validate_stock_predictions(frame):
    """Validate saved v1/v2 neutral predictions; account admission is separate."""
    wire = frame.to_dict()
    v2 = wire.get("contract_version") == "stock_prediction_run_v2"
    fields(wire, "contract_version signal_run_ref signal_stage score_semantics score_unit feature_ref model_ref limitations universe rows" +
           (" fold_spec_ref clock_basis" if v2 else ""))
    require(wire["contract_version"] in ("stock_prediction_run_v1", "stock_prediction_run_v2") and wire["signal_stage"] == "prediction_raw" and
            wire["score_semantics"] == "forward_5_session_cs_zscore_prediction" and wire["score_unit"] == "dimensionless",
            "unsupported stock prediction semantics")
    for name in ("signal_run_ref", "feature_ref", "model_ref"):
        digest(wire[name])
    if v2:
        digest(wire["fold_spec_ref"])
        require(wire["clock_basis"] == "declared_simulation", "unsupported prediction clock basis")
    require(type(wire["limitations"]) is list and all(type(x) is str for x in wire["limitations"]), "prediction limitations required")
    universe = wire["universe"]
    require(type(universe) is list and bool(universe) and len(set(universe)) == len(universe), "explicit prediction union required")
    for security in universe:
        text(security)
    require(type(wire["rows"]) is list and bool(wire["rows"]), "prediction rows required")
    indexed, groups, clocks = {}, {}, {}
    model_clock = None
    for row in wire["rows"]:
        fields(row, "security_id session knowledge_cutoff available_at score valid invalid_reason source_refs member" +
               (" feature_knowledge_cutoff feature_available_at simulated_model_available_at" if v2 else ""))
        require(row["security_id"] in universe, "prediction outside union")
        session(row["session"])
        cutoff, available = instant(row["knowledge_cutoff"]), instant(row["available_at"])
        if v2:
            feature_cutoff, model_available = _prediction_clock_v2(row, cutoff, available)
            clock = (feature_cutoff, cutoff)
            require(clocks.setdefault(row["session"], clock) == clock and
                    (model_clock is None or model_clock == model_available), "inconsistent saved prediction clocks")
            model_clock = model_available
        else:
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
        if v2:
            require(all(wire[name] in row["source_refs"] for name in ("feature_ref", "model_ref")),
                    "v2 prediction lacks Feature/model source refs")
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate prediction key")
        indexed[key] = row
        groups.setdefault(row["session"], set()).add(row["security_id"])
    require(all(keys == set(universe) for keys in groups.values()), "incomplete prediction union coverage")
    return wire, indexed


def plan_stock_portfolio(frame, *, account, context, top_k=None):
    """Equal deployable-wealth targets; no execution facts, receivables or I/O."""
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _plan(frame, account, context, top_k)


def _plan_admitted_stock_portfolio(wire, rows, *, account, context, top_k):
    """Runtime-private reuse after its full entry admission; no public bypass tag."""
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _plan(None, account, context, top_k, admitted=(wire, rows))


def _plan(frame, account, context, top_k, admitted=None):
    wire, rows = validate_stock_predictions(frame) if admitted is None else admitted
    v2 = wire["contract_version"] == "stock_prediction_run_v2"
    full = "stock_execution_rules" in context
    if v2:
        require(top_k is not None and "feature_knowledge_cutoff" in context,
                "v2 neutral predictions only; account clock consumption is not admitted without explicit TopK and feature clock")
        if admitted is None:
            unsigned = dict(wire); reference = unsigned.pop("signal_run_ref")
            require(Document.from_dict(unsigned).identity == reference, "Saved v2 prediction identity mismatch")
    fields(context, "trade_session feature_session decision_time knowledge_cutoff reference_prices commission_rate minimum_commission_minor slippage_bps account_state_version supported_security_ids supported_universe_ref" +
           (" portfolio_policy stock_execution_rules stock_execution_rules_ref" if full else " lot_size") +
           (" feature_knowledge_cutoff" if v2 else ""))
    session(context["trade_session"]); session(context["feature_session"])
    cutoff, decision = instant(context["knowledge_cutoff"]), instant(context["decision_time"])
    require(context["feature_session"] < context["trade_session"] and
            cutoff == instant(context["feature_session"] + ("T21:00:00+08:00" if v2 else "T20:30:00+08:00")) and
            decision == instant(context["trade_session"] + "T08:55:00+08:00"), "invalid stock decision clock")
    feature_cutoff = instant(context["feature_knowledge_cutoff"]) if v2 else cutoff
    require(feature_cutoff == instant(context["feature_session"] + "T20:30:00+08:00"), "invalid stock Feature cutoff")
    supported = context["supported_security_ids"]
    require(type(supported) is list and bool(supported) and len(set(supported)) == len(supported) and
            set(supported) <= set(wire["universe"]), "explicit supported prediction subset required")
    digest(context["supported_universe_ref"])
    rules, rule_index = (context["stock_execution_rules"], None) if full else (None, None)
    if full:
        rule_index = validate_execution_rules(rules)
        require(supported == wire["universe"] == rules["universe"] and
                context["supported_universe_ref"] == csi300_support_ref(supported) and
                context["stock_execution_rules_ref"] == Document.from_dict(rules).identity,
                "complete CSI300 rules/union identity required")
        require(context["portfolio_policy"] == {"eligibility_id": CSI300_ELIGIBILITY, "top_k": top_k,
            "rebalance": "weekly_first_trading_session", "budget_basis": BUDGET_BASIS,
            "candidate_policy": CANDIDATE_POLICY, "stock_execution_rules_ref": context["stock_execution_rules_ref"]},
            "explicit CSI300 valid-member policy required")
        require(top_k is not None, "CSI300 policy requires explicit top_k")
    legacy = top_k is None
    k = 5 if legacy else top_k
    if not legacy:
        validate_top_k(k, supported)
    version = CSI300_PORTFOLIO_VERSION if full else (STOCK_PORTFOLIO_VERSION if legacy else TOPK_PORTFOLIO_VERSION)
    fields(account, "cash_minor positions version")
    integer(account["cash_minor"]); integer(account["version"])
    require(account["version"] == context["account_state_version"], "account version conflict")
    require(type(account["positions"]) is dict, "positions required")
    for security, position in account["positions"].items():
        require(security in supported, "held security outside execution scope")
        fields(position, "quantity sellable_quantity")
        integer(position["quantity"]); integer(position["sellable_quantity"])
        require(position["sellable_quantity"] <= position["quantity"], "invalid sellable quantity")
    if not full:
        require(context["lot_size"] == 100, "stock buy lot must be 100 shares")
    integer(context["minimum_commission_minor"])
    require(decimal(context["commission_rate"], minimum=0) <= 1, "invalid commission rate")
    require(decimal(context["slippage_bps"], minimum=0) < 10000, "invalid slippage")
    batch = [rows.get((context["feature_session"], security)) for security in wire["universe"]]
    require(all(row is not None for row in batch), "missing previous-session prediction")
    for row in batch:
        require(instant(row["knowledge_cutoff"]) == cutoff and instant(row["available_at"]) <= decision,
                "future/inconsistent prediction cutoff")
        if v2:
            require(instant(row["feature_knowledge_cutoff"]) == feature_cutoff and
                    instant(row["simulated_model_available_at"]) < cutoff, "inconsistent v2 decision clock")
        if full and row["member"]:
            require(listed(rule_index[0][row["security_id"]], context["feature_session"]),
                    "PIT member contradicts native stock listing interval")
    eligible = [row for row in batch if row["member"] and row["security_id"] in supported]
    result = {"contract_version": version, "feature_session": context["feature_session"],
        "trade_session": context["trade_session"], "signal_ref": wire["signal_run_ref"],
        "supported_universe_ref": context["supported_universe_ref"], "expected_account_version": account["version"],
        "status": "DECISION_COMPLETE", "selected_security_ids": [], "targets": {}, "intents": [], "trace": []}
    if not legacy:
        result["top_k"] = k
    if full:
        result["stock_execution_rules_ref"] = context["stock_execution_rules_ref"]
    if v2:
        result["prediction_clock"] = {"clock_basis": wire["clock_basis"],
            "feature_knowledge_cutoff": batch[0]["feature_knowledge_cutoff"],
            "inference_cutoff": batch[0]["knowledge_cutoff"],
            "simulated_model_available_at": batch[0]["simulated_model_available_at"],
            "model_ref": wire["model_ref"], "fold_spec_ref": wire["fold_spec_ref"]}
    invalid = [row for row in eligible if not row["valid"]]
    valid_count = len(eligible) if legacy else sum(row["valid"] for row in eligible)
    if full:
        result["trace"].append({"reason": "VALID_MEMBER_CANDIDATES", "candidate_policy": CANDIDATE_POLICY,
            "pit_member_count": len(eligible), "valid_candidate_count": valid_count,
            "excluded_invalid_member_count": len(invalid), "excluded_invalid_members": [
                {"security_id": row["security_id"], "invalid_reason": row["invalid_reason"]} for row in invalid]})
    if (invalid and not full) or valid_count < k:
        result["status"] = "NO_DECISION"
        if not full:
            result["trace"] = [{"reason": "INVALID_SIGNAL", "security_id": row["security_id"], "detail": row["invalid_reason"]} for row in invalid]
        if valid_count < k:
            result["trace"].append({"reason": "INSUFFICIENT_ELIGIBLE_MEMBERS", "count": valid_count,
                                    **({} if legacy else {"top_k": k})})
        return PortfolioDecision.from_dict(result)
    candidates = [row for row in eligible if row["valid"]] if full else eligible
    selected = [row["security_id"] for row in sorted(candidates, key=lambda row: (-row["score"], row["security_id"]))[:k]]
    result["selected_security_ids"] = selected
    held = {s for s, p in account["positions"].items() if p["quantity"]} if full else set(account["positions"])
    needed = set(selected) | held
    prices = {}
    missing = []
    for security in sorted(needed):
        quote = context["reference_prices"].get(security)
        if full and quote is None:
            missing.append(security)
            continue
        require(quote is not None, "missing stock previous-close reference")
        fields(quote, "price session available_at source_refs")
        if full:
            session(quote["session"])
            require(quote["session"] <= context["feature_session"] and instant(quote["available_at"]) <= feature_cutoff,
                    "future stock sizing reference")
            if quote["session"] != context["feature_session"]:
                missing.append(security)
                continue
        require(quote["session"] == context["feature_session"] and instant(quote["available_at"]) <= feature_cutoff,
                "future or stale stock sizing reference")
        require(type(quote["source_refs"]) is list and bool(quote["source_refs"]), "reference provenance required")
        for ref in quote["source_refs"]:
            source_ref(ref)
        prices[security] = decimal(quote["price"], minimum=0)
        require(prices[security] > 0, "positive reference price required")
    if missing:
        result["status"] = "NO_DECISION"
        result["trace"].append({"reason": "MISSING_SIZING_REFERENCE", "security_ids": missing,
                                 "required_session": context["feature_session"]})
        return PortfolioDecision.from_dict(result)
    budget = Decimal(account["cash_minor"]) + sum(Decimal(position["quantity"]) * prices[security] * 100
                                                  for security, position in account["positions"].items() if security in held)
    for security in sorted(needed):
        raw = int(budget / k / (prices[security] * 100)) if security in selected else 0
        if full:
            rule = rule_at(rules, security, context["trade_session"], validated=rule_index)
            result["targets"][security] = floor_quantity(raw, rule["buy_minimum"], rule["buy_increment"])
            if security in selected and not result["targets"][security]:
                result["trace"].append({"reason": "TARGET_BELOW_MINIMUM_QUANTITY", "security_id": security,
                    "raw_target_quantity": raw, "minimum": rule["buy_minimum"]})
        else:
            result["targets"][security] = raw // 100 * 100
    for security, position in sorted(account["positions"].items()):
        reduction = max(0, position["quantity"] - result["targets"].get(security, 0))
        quantity = min(reduction, position["sellable_quantity"])
        available = quantity
        if full and quantity:
            rule = rule_at(rules, security, context["trade_session"], validated=rule_index)
            quantity = legal_quantity(reduction, "SELL", rule, held=position["quantity"],
                                      sellable=position["sellable_quantity"], apply_maximum=False)
            if quantity < available:
                result["trace"].append({"reason": "BELOW_MINIMUM_ORDER_QUANTITY" if not quantity else "ORDER_QUANTITY_INCREMENT",
                    "security_id": security, "side": "SELL", "quantity": available - quantity})
        if quantity:
            result["intents"].append({"security_id": security, "side": "SELL", "quantity": quantity})
        if reduction > available:
            result["trace"].append({"reason": "T_PLUS_ONE_OR_UNSELLABLE", "security_id": security, "quantity": reduction - available})
    for security in sorted(selected):
        current = account["positions"].get(security, {"quantity": 0})["quantity"]
        increase = max(0, result["targets"][security] - current)
        if full:
            rule = rule_at(rules, security, context["trade_session"], validated=rule_index)
            quantity = legal_quantity(increase, "BUY", rule, apply_maximum=False)
            if quantity < increase:
                result["trace"].append({"reason": "BELOW_MINIMUM_ORDER_QUANTITY" if not quantity else "ORDER_QUANTITY_INCREMENT",
                    "security_id": security, "side": "BUY", "quantity": increase - quantity})
        else:
            quantity = increase // 100 * 100
        if quantity:
            result["intents"].append({"security_id": security, "side": "BUY", "quantity": quantity})
    result["trace"].append({"reason": "RAW_TOP5" if legacy else "RAW_TOP_K", "eligible_count": len(eligible),
        **({} if legacy else {"top_k": k}), "tie_break": "security_id_asc",
        "budget_basis": BUDGET_BASIS, "reference_budget_minor": minor(budget), "sizing": "previous_native_close",
        "score_semantics": wire["score_semantics"], "cash_check": "actual_fill_cash"})
    identity = Document.from_dict({"contract": version, **({"frame_ref": wire["signal_run_ref"]} if v2 else {"frame": wire}), "context": context, "account": account,
                                  **({} if legacy else {"top_k": k})}).identity
    for index, intent in enumerate(result["intents"]):
        intent.update(intent_id=identity + ":" + str(index), expected_account_version=account["version"], valid_until=context["trade_session"])
    return PortfolioDecision.from_dict(result)
