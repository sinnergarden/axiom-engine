"""One fixed entry policy, pure previous-close sizing, no Signal fabrication."""
import re
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from .contracts import Document, digest, fields, integer, require, session, timestamp
from .portfolio import PortfolioDecision, decimal

BUY_HOLD_VERSION = "axiom.etf_buy_and_hold/1"


def etf_buy_and_hold_policy(*, security_id, entry_session):
    require(type(security_id) is str and re.fullmatch(r"cn\.etf\.(SSE|SZSE)\.\d{6}\.\d{8}", security_id),
            "canonical ETF identity required")
    session(entry_session)
    return dict(contract_version="etf_buy_and_hold_policy_v1", security_id=security_id,
        entry_session=entry_session, budget="1", schedule="entry_session_once",
        partial_fill_policy="expire_no_retry", cash_dividend_policy="retain_cash",
        terminal_policy="mark_open_position")


def validate_buy_hold_policy(policy):
    fields(policy, "contract_version security_id entry_session budget schedule partial_fill_policy cash_dividend_policy terminal_policy")
    require(policy == etf_buy_and_hold_policy(security_id=policy["security_id"], entry_session=policy["entry_session"]),
            "unsupported buy-and-hold policy")


def plan_etf_buy_and_hold(policy, *, account, context):
    """Submit the entry-day BUY once; Runtime owns actual fill cash and expiry."""
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        validate_buy_hold_policy(policy)
        fields(context, "trade_session reference_session decision_time reference_cutoff reference_prices lot_size commission_rate minimum_commission_minor tax_rate slippage_bps account_state_version")
        session(context["trade_session"]); session(context["reference_session"])
        timestamp(context["decision_time"]); timestamp(context["reference_cutoff"])
        require(context["trade_session"] == policy["entry_session"] and
                context["reference_session"] < context["trade_session"] and
                context["reference_cutoff"] == context["reference_session"] + "T12:30:00Z" and
                context["decision_time"] == context["trade_session"] + "T00:55:00Z",
                "entry and previous-close decision clocks required")
        fields(account, "cash_minor positions version")
        integer(account["cash_minor"]); integer(account["version"])
        require(account["positions"] == {} and account["version"] == context["account_state_version"],
                "empty entry account at current version required")
        require(type(context["reference_prices"]) is dict and type(context["lot_size"]) is int and context["lot_size"] == 100,
                "reference quotes and 100-unit lot required")
        integer(context["minimum_commission_minor"]); integer(context["account_state_version"])
        for key in ("commission_rate", "tax_rate", "slippage_bps"):
            decimal(context[key], minimum=0)
        require(decimal(context["commission_rate"]) <= 1 and decimal(context["tax_rate"]) <= 1 and
                decimal(context["slippage_bps"]) < 10000, "invalid execution cost range")
        security = policy["security_id"]
        policy_ref = Document.from_dict(policy).identity
        result = dict(contract_version=BUY_HOLD_VERSION, trade_session=context["trade_session"],
            reference_session=context["reference_session"], signal_ref=None, portfolio_policy_ref=policy_ref,
            expected_account_version=account["version"], status="DECISION_COMPLETE",
            selected_security_id=security, targets={}, intents=[], trace=[])
        quote = context["reference_prices"].get(security)
        if quote is not None:
            fields(quote, "price session available_at source_refs")
            session(quote["session"])
        if quote is None or quote["session"] != context["reference_session"]:
            result.update(status="NO_DECISION", trace=[dict(reason="ENTRY_REFERENCE_UNAVAILABLE", security_id=security)])
            return PortfolioDecision.from_dict(result)
        timestamp(quote["available_at"])
        require(quote["available_at"] <= context["reference_cutoff"] and type(quote["source_refs"]) is list and bool(quote["source_refs"]),
                "entry reference unavailable at cutoff")
        for ref in quote["source_refs"]: digest(ref)
        price = decimal(quote["price"], minimum=0)
        require(price > 0, "positive entry reference required")
        target = int(Decimal(account["cash_minor"]) / (price * 100 * context["lot_size"])) * context["lot_size"]
        result["targets"] = {security: target}
        result["trace"] = [dict(reason="ENTRY_ONCE" if target else "ZERO_LOT_BUDGET",
            security_id=security, sizing="strict_previous_session_close", cash_check="actual_fill_cash")]
        if target:
            decision_ref = Document.from_dict(dict(account=account, context=context, portfolio_policy_ref=policy_ref)).identity
            result["intents"] = [dict(intent_id=decision_ref + ":0", security_id=security, side="BUY", quantity=target,
                expected_account_version=account["version"], valid_until=context["trade_session"])]
        return PortfolioDecision.from_dict(result)
