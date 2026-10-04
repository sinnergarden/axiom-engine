"""Frozen cached-signal daily replay, Core decisions, SimBroker and accounting."""
from datetime import date
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

from .._implementation import IMPLEMENTATION_REF
from ..core import SignalFrame, plan_rotation
from ..core.contracts import Document, digest, fields, integer, require, session, text, timestamp
from ..core.portfolio import PORTFOLIO_VERSION, decimal, minor, validate_signals
from .accounting import AccountLedger

RUNTIME_VERSION = "axiom.backtest/1"


class MarketReplay(Document):
    """Explicit unadjusted daily simulation facts and cash-dividend events."""


class BacktestRequest(Document):
    """Frozen inputs, scope, initial account and experimental execution profile."""


class BacktestRun(Document):
    """One immutable, complete offline result; no live account side effects."""


def _validate(request):
    plan = request.to_dict()
    fields(plan, "contract_version account_id start_session end_session signal_frame market_replay initial_account profile")
    require(plan["contract_version"] == "backtest_request_v1", "unsupported request")
    text(plan["account_id"]); session(plan["start_session"]); session(plan["end_session"])
    signal, signals = validate_signals(SignalFrame.from_dict(plan["signal_frame"]))
    market = plan["market_replay"]
    fields(market, "contract_version price_basis calendar universe rows cash_dividends source_refs source_evidence limitations")
    require(market["contract_version"] == "market_replay_v1" and market["price_basis"] == "unadjusted", "unadjusted replay required")
    calendar = market["calendar"]
    require(type(calendar) is list and bool(calendar) and calendar == sorted(set(calendar)), "explicit ordered calendar required")
    for day in calendar:
        session(day)
    require(set(market["universe"]) == set(signal["universe"]) and
            len(market["universe"]) == len(signal["universe"]), "signal/market universe mismatch")
    require(plan["start_session"] in calendar and plan["end_session"] in calendar and
            calendar.index(plan["start_session"]) > 0 and plan["start_session"] <= plan["end_session"], "scope needs prior trading session")
    require(type(market["limitations"]) is list and type(market["source_refs"]) is list and bool(market["source_refs"]), "market provenance required")
    for ref in market["source_refs"]:
        digest(ref)
    indexed = {}
    for row in market["rows"]:
        fields(row, "security_id session open close volume_units limit_up limit_down close_available_at market_state state_reason source_refs")
        require(row["security_id"] in signal["universe"] and row["session"] in calendar, "market key outside scope")
        for name in ("open", "close", "limit_up", "limit_down"):
            if row[name] is not None:
                require(decimal(row[name], minimum=0) > 0, "positive market price required")
        if row["volume_units"] is not None:
            require(decimal(row["volume_units"], minimum=0) >= 0, "invalid volume")
        if row["limit_up"] is not None and row["limit_down"] is not None:
            lower, upper = decimal(row["limit_down"]), decimal(row["limit_up"])
            require(lower <= upper, "contradictory price limits")
            for name in ("open", "close"):
                require(row[name] is None or lower <= decimal(row[name]) <= upper,
                        "market price outside declared limits")
        timestamp(row["close_available_at"])
        require(row["market_state"] in ("normal_trading", "unknown_status", "suspended", "source_gap"),
                "unsupported market state, listing or calendar scope")
        require(row["state_reason"] is None or type(row["state_reason"]) is str, "invalid state reason")
        require(row["close_available_at"][:10] >= row["session"], "close timestamp before session")
        require(bool(row["source_refs"]), "row provenance required")
        for ref in row["source_refs"]:
            digest(ref)
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate market key")
        indexed[key] = row
    require(set(indexed) == {(d, s) for d in calendar for s in signal["universe"]}, "incomplete market key coverage")
    actions = market["cash_dividends"]
    seen = set()
    require(type(actions) is list, "explicit corporate action scope required")
    for action in actions:
        fields(action, "event_id security_id record_session ex_session pay_session cash_per_unit source_refs")
        text(action["event_id"])
        require(action["event_id"] not in seen and action["security_id"] in signal["universe"], "duplicate/unknown action")
        seen.add(action["event_id"])
        for key in ("record_session", "ex_session", "pay_session"):
            session(action[key])
        require(action["record_session"] < action["ex_session"] <= action["pay_session"], "unsupported dividend date order")
        require(action["record_session"] in calendar, "record date absent from calendar")
        if action["record_session"] >= plan["start_session"] and action["ex_session"] <= plan["end_session"]:
            require(action["ex_session"] in calendar and (action["pay_session"] > plan["end_session"] or
                    action["pay_session"] in calendar), "dividend cash phase outside trading calendar unsupported")
        decimal(action["cash_per_unit"], minimum=0)
        require(bool(action["source_refs"]), "action provenance required")
        for ref in action["source_refs"]:
            digest(ref)
        require(not (plan["initial_account"]["positions"] and action["record_session"] < plan["start_session"] <= action["pay_session"]),
                "initial dividend entitlements require an explicit supported seed")
    profile = plan["profile"]
    fields(profile, "contract_version lot_size settlement_sessions commission_rate minimum_commission_minor tax_rate slippage_bps participation_rate decision_time_utc execution approximation unknown_status_policy limitation")
    require(profile["contract_version"] == "daily_open_profile_v1" and
            profile["execution"] == "open" and profile["approximation"] == "daily_volume_proxy", "unsupported execution profile")
    integer(profile["lot_size"], 1); integer(profile["settlement_sessions"])
    integer(profile["minimum_commission_minor"])
    for name in ("commission_rate", "tax_rate", "slippage_bps", "participation_rate"):
        decimal(profile[name], minimum=0)
    require(decimal(profile["participation_rate"]) <= 1 and decimal(profile["commission_rate"]) <= 1 and
            decimal(profile["tax_rate"]) <= 1 and decimal(profile["slippage_bps"]) < 10000, "invalid execution range")
    timestamp(plan["start_session"] + "T" + profile["decision_time_utc"])
    require(profile["decision_time_utc"] < "01:30:00Z", "decision must precede mainland open")
    text(profile["limitation"])
    require(profile["unknown_status_policy"] == "block", "unknown market status must block execution")
    fields(plan["initial_account"], "cash_minor positions")
    return plan, signal, signals, market, calendar, indexed, profile


def _fees(price, quantity, side, profile):
    gross = minor(price * quantity * 100)
    commission = max(profile["minimum_commission_minor"], minor(Decimal(gross) * decimal(profile["commission_rate"])))
    tax = minor(Decimal(gross) * decimal(profile["tax_rate"])) if side == "SELL" else 0
    return gross, commission, tax


def _simulate(intent, row, ledger, profile, day, run_id, order_index):
    order = {**intent, "order_id": run_id + ":order:" + str(order_index), "session": day,
             "status": "EXPIRED", "filled_quantity": 0, "unfilled_quantity": intent["quantity"],
             "reason": None, "execution": "daily_open_approximation"}
    order["market_state"] = row["market_state"]
    order["state_reason"] = row["state_reason"]
    if row["market_state"] != "normal_trading":
        order["reason"] = "UNKNOWN_MARKET_STATUS" if row["market_state"] == "unknown_status" else "NOT_TRADING"
        return order
    if any(row[key] is None for key in ("open", "volume_units", "limit_up", "limit_down")):
        order["reason"] = "MISSING_EXECUTION_FACT"
        return order
    opening = decimal(row["open"])
    buy = intent["side"] == "BUY"
    price = opening * (1 + decimal(profile["slippage_bps"]) / 10000 * (1 if buy else -1))
    if (price > decimal(row["limit_up"]) or price < decimal(row["limit_down"]) or
            (buy and price == decimal(row["limit_up"])) or (not buy and price == decimal(row["limit_down"]))):
        order["reason"] = "PRICE_LIMIT"
        return order
    lot = profile["lot_size"]
    cap = int(decimal(row["volume_units"]) * decimal(profile["participation_rate"]))
    quantity = min(intent["quantity"], cap)
    if buy or quantity < intent["quantity"]:
        quantity = quantity // lot * lot
    if not buy:
        quantity = min(quantity, ledger.positions.get(intent["security_id"], {"sellable_quantity": 0})["sellable_quantity"])
    if buy and quantity:
        low, high = 0, quantity // lot
        while low < high:
            mid = (low + high + 1) // 2
            gross, commission, tax = _fees(price, mid * lot, "BUY", profile)
            if gross + commission + tax <= ledger.cash:
                low = mid
            else:
                high = mid - 1
        quantity = low * lot
    if not quantity:
        order["reason"] = "INSUFFICIENT_CASH" if buy and cap >= lot else "NO_VOLUME_OR_SELLABLE_QUANTITY"
        return order
    gross, commission, tax = _fees(price, quantity, intent["side"], profile)
    if not buy and ledger.cash + gross < commission + tax:
        order["reason"] = "INSUFFICIENT_CASH_FOR_FEES"
        return order
    fill = {"fill_id": order["order_id"] + ":fill:0", "order_id": order["order_id"], "session": day,
            "security_id": intent["security_id"], "side": intent["side"], "quantity": quantity,
            "price": str(price), "reference_open": str(opening), "gross_minor": gross,
            "commission_minor": commission, "tax_minor": tax, "fee_minor": commission + tax,
            "slippage_minor": minor(abs(price - opening) * quantity * 100),
            "cash_delta_minor": -gross - commission - tax if buy else gross - commission - tax,
            "source_refs": row["source_refs"]}
    ledger.apply_fill(fill)
    order.update({"filled_quantity": quantity, "unfilled_quantity": intent["quantity"] - quantity,
                  "status": "FILLED" if quantity == intent["quantity"] else "PARTIAL_EXPIRED",
                  "reason": None if quantity == intent["quantity"] else "CASH_OR_VOLUME_CAP",
                  "committed_sequence": ledger.sequence})
    return order


def run_backtest(request):
    """Run an explicit frozen request; no Data/Research import, discovery or I/O."""
    require(isinstance(request, BacktestRequest), "BacktestRequest required")
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _run(request)


def _run(request):
    plan, signal, signals, market, calendar, rows, profile = _validate(request)
    run_id = Document.from_dict({"request": plan, "core": PORTFOLIO_VERSION, "runtime": RUNTIME_VERSION,
                                "implementation_ref": IMPLEMENTATION_REF}).identity
    ledger = AccountLedger(cash_minor=plan["initial_account"]["cash_minor"], calendar=calendar,
                           settlement_sessions=profile["settlement_sessions"], positions=plan["initial_account"]["positions"])
    quotes, marks, entitlements = {}, {}, {}
    nav, positions, orders, decisions = [], [], [], []
    initial_value = None
    for index, day in enumerate(calendar):
        if day > plan["end_session"]:
            break
        if day >= plan["start_session"]:
            if initial_value is None:
                require(all(s in marks for s in ledger.positions), "initial positions lack observed price")
                initial_value = ledger.cash + sum(minor(decimal(marks[s]["price"]) * p["quantity"] * 100)
                                                   for s, p in ledger.positions.items())
                require(initial_value > 0, "positive initial NAV required")
            ledger.advance(day)
            for phase, key in (("EX", "ex_session"), ("PAY", "pay_session")):
                for action in sorted(market["cash_dividends"], key=lambda a: a["event_id"]):
                    if action[key] == day and action["event_id"] in entitlements:
                        ledger.dividend(action, phase, entitlements.get(action["event_id"], 0))
            previous = calendar[index - 1]
            if date.fromisoformat(day).isocalendar()[:2] != date.fromisoformat(previous).isocalendar()[:2]:
                first = signals.get((previous, signal["universe"][0]))
                require(first is not None, "missing previous-session signal")
                context = {"trade_session": day, "feature_session": previous,
                           "decision_time": day + "T" + profile["decision_time_utc"],
                           "knowledge_cutoff": first["knowledge_cutoff"], "reference_prices": quotes,
                           "account_state_version": ledger.sequence,
                           **{key: profile[key] for key in ("lot_size", "commission_rate", "minimum_commission_minor", "tax_rate", "slippage_bps")}}
                decision = plan_rotation(SignalFrame.from_dict(signal), account=ledger.account(), context=context).to_dict()
                decisions.append(decision)
                for intent in decision["intents"]:
                    orders.append(_simulate(intent, rows[day, intent["security_id"]], ledger, profile, day, run_id, len(orders)))
            for action in market["cash_dividends"]:
                if action["record_session"] == day:
                    entitlements[action["event_id"]] = ledger.positions.get(action["security_id"], {"quantity": 0})["quantity"]
        for security in sorted(signal["universe"]):
            row = rows[day, security]
            if row["close"] is not None:
                quote = {"price": row["close"], "session": day, "available_at": row["close_available_at"], "source_refs": row["source_refs"]}
                quotes[security] = quote
                marks[security] = quote
        if day < plan["start_session"]:
            continue
        ledger.sequence += 1  # commit the session valuation at one shared watermark
        value = 0
        for security, position in sorted(ledger.positions.items()):
            if not position["quantity"]:
                continue
            require(security in marks, "held position lacks observed valuation price")
            mark = marks[security]
            amount = minor(decimal(mark["price"]) * position["quantity"] * 100)
            value += amount
            positions.append({"session": day, "security_id": security, **position,
                "mark_price": mark["price"], "mark_session": mark["session"],
                "is_stale": mark["session"] != day, "stale_sessions": index - calendar.index(mark["session"]),
                "mark_source_refs": mark["source_refs"], "market_value_minor": amount,
                "committed_sequence": ledger.sequence})
        receivable = sum(ledger.receivables.values())
        total = ledger.cash + value + receivable
        nav.append({"session": day, "cash_minor": ledger.cash, "market_value_minor": value,
            "receivable_minor": receivable, "nav_minor": total,
            "nav_index": str(Decimal(total) / initial_value), "committed_sequence": ledger.sequence})
    peak, drawdown = initial_value, Decimal(0)
    for point in nav:
        peak = max(peak, point["nav_minor"])
        drawdown = min(drawdown, Decimal(point["nav_minor"]) / peak - 1)
    result = {"contract_version": "backtest_run_v1", "run_id": run_id, "account_id": plan["account_id"],
        "status": "COMPLETE", "plan": plan, "signal_ref": signal["signal_run_ref"],
        "market_ref": MarketReplay.from_dict(market).identity, "profile_ref": Document.from_dict(profile).identity,
        "core_version": PORTFOLIO_VERSION, "runtime_version": RUNTIME_VERSION,
        "implementation_ref": IMPLEMENTATION_REF,
        "committed_sequence": ledger.sequence, "initial_nav_minor": initial_value,
        "final_account": {"cash_minor": ledger.cash, "receivable_minor": sum(ledger.receivables.values()),
                          "positions": ledger.positions, "committed_sequence": ledger.sequence},
        "decisions": decisions, "orders": orders, "fills": ledger.fills,
        "cash_ledger": ledger.cash_ledger, "position_ledger": ledger.position_ledger,
        "nav": nav, "positions": positions,
        "metrics": {"total_return": str(Decimal(nav[-1]["nav_minor"]) / initial_value - 1),
                    "max_drawdown": str(drawdown), "total_fees_minor": sum(f["fee_minor"] for f in ledger.fills),
                    "turnover_minor": sum(f["gross_minor"] for f in ledger.fills), "fill_count": len(ledger.fills),
                    "unfilled_order_count": sum(o["unfilled_quantity"] > 0 for o in orders)},
        "limitations": [profile["limitation"], "Daily open-price approximation; full-day volume is an execution-side capacity proxy, not known opening liquidity.",
                         "Day orders expire after one simulated fill; no live broker, SQLite recovery, split or delisting support.",
                         "Dividend handling supports explicit record/ex/pay cash events; unsupported economic events must be rejected by the caller adapter.",
                         "Metrics cover the fixed short sample only; no OOS, CAGR or Sharpe claim.", *market["limitations"]]}
    result["content_digest"] = Document.from_dict(result).identity
    return BacktestRun.from_dict(result)


def save_backtest_run(run, path):
    """Save only a complete deterministic result; refuse a conflicting overwrite."""
    require(isinstance(run, BacktestRun), "BacktestRun required")
    path = Path(path)
    if path.exists():
        require(path.read_text() == run.payload + "\n", "conflicting saved BacktestRun")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(run.payload + "\n")


def load_backtest_run(path):
    """Read a saved result and check content/input identity without recomputation."""
    saved = BacktestRun(Path(path).read_text())
    wire = saved.to_dict()
    require(wire.get("contract_version") == "backtest_run_v1" and wire.get("status") == "COMPLETE", "unsupported saved result")
    recorded_digest = wire.pop("content_digest", None)
    require(recorded_digest == Document.from_dict(wire).identity, "saved result content digest mismatch")
    require(wire["run_id"] == Document.from_dict({"request": wire["plan"],
            "core": wire["core_version"], "runtime": wire["runtime_version"],
            "implementation_ref": wire["implementation_ref"]}).identity,
            "saved run identity mismatch")
    return saved
