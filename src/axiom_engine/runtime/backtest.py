"""Frozen cached-signal daily replay, Core decisions, SimBroker and accounting."""
from datetime import date
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, localcontext
from pathlib import Path
import re

from .._implementation import IMPLEMENTATION_REF
from ..core import SignalFrame, plan_rotation, StockPredictionFrame, plan_stock_portfolio
from ..core.stock_portfolio import STOCK_PORTFOLIO_VERSION, TOPK_PORTFOLIO_VERSION, CSI300_PORTFOLIO_VERSION
from ..core.stock_portfolio import _plan_admitted_stock_portfolio
from ..core.contracts import Document, digest, fields, integer, require, session, text, timestamp
from ..core.portfolio import PORTFOLIO_VERSION, decimal, minor, validate_signals
from .accounting import AccountLedger
from .profiles import STATUS_GAP_REASONS
from .unit_splits import UNIT_SPLIT_POLICY, suspension_events, validate_unit_splits

RUNTIME_VERSION = "axiom.backtest/2"
LEGACY_RUNTIME_VERSION = "axiom.backtest/1.1"


class MarketReplay(Document):
    """Unadjusted daily facts, cash dividends and bounded ETF unit events."""


class BacktestRequest(Document):
    """Frozen inputs, scope, initial account and experimental execution profile."""


class BacktestRun(Document):
    """One immutable, complete offline result; no live account side effects."""


def _validate(request, *, decoded_plan=None):
    plan = request.to_dict() if decoded_plan is None else decoded_plan
    if plan.get("contract_version") in ("backtest_request_v3", "backtest_request_v4", "backtest_request_v6"):
        from .stock_inputs import validate_stock_request
        return validate_stock_request(plan)
    v5 = plan.get("contract_version") == "backtest_request_v5"
    v2 = plan.get("contract_version") in ("backtest_request_v2", "backtest_request_v5")
    fields(plan, "contract_version account_id start_session end_session signal_frame market_replay initial_account profile" +
           (" unit_split_policy" if v2 else "") + (" portfolio_policy price_unit" if v5 else ""))
    require(plan["contract_version"] in ("backtest_request_v1", "backtest_request_v2", "backtest_request_v5"), "unsupported request")
    if v2:
        require(plan["unit_split_policy"] == UNIT_SPLIT_POLICY, "unsupported unit split policy")
    text(plan["account_id"]); session(plan["start_session"]); session(plan["end_session"])
    market = plan["market_replay"]
    if v5 and plan["signal_frame"] is None:
        from ..core.etf_buy_hold import validate_buy_hold_policy
        validate_buy_hold_policy(plan["portfolio_policy"])
        universe = market["universe"]
        require(type(universe) is list and bool(universe) and all(type(s) is str for s in universe) and
                len(set(universe)) == len(universe), "unique explicit ETF universe required")
        signal, signals = None, {}
    else:
        signal, signals = validate_signals(SignalFrame.from_dict(plan["signal_frame"]))
        universe = signal["universe"]
    fields(market, "contract_version price_basis calendar universe rows cash_dividends source_refs source_evidence limitations" +
           (" unit_splits" if v2 else ""))
    require(market["contract_version"] == ("market_replay_v2" if v2 else "market_replay_v1") and
            market["price_basis"] == "unadjusted", "matching unadjusted replay required")
    calendar = market["calendar"]
    require(type(calendar) is list and bool(calendar) and calendar == sorted(set(calendar)), "explicit ordered calendar required")
    for day in calendar:
        session(day)
    require(set(market["universe"]) == set(universe) and
            len(market["universe"]) == len(universe), "signal/market universe mismatch")
    require(plan["start_session"] in calendar and plan["end_session"] in calendar and
            calendar.index(plan["start_session"]) > 0 and plan["start_session"] <= plan["end_session"], "scope needs prior trading session")
    require(type(market["limitations"]) is list and type(market["source_refs"]) is list and bool(market["source_refs"]), "market provenance required")
    for ref in market["source_refs"]:
        digest(ref)
    indexed = {}
    for row in market["rows"]:
        fields(row, "security_id session open close volume_units limit_up limit_down close_available_at market_state state_reason source_refs")
        require(row["security_id"] in universe and row["session"] in calendar, "market key outside scope")
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
        if v5:
            for name in ("open", "close"):
                if row[name] is not None:
                    require(row["limit_up"] is None or decimal(row[name]) <= decimal(row["limit_up"]), "market price above known limit")
                    require(row["limit_down"] is None or decimal(row[name]) >= decimal(row["limit_down"]), "market price below known limit")
        timestamp(row["close_available_at"])
        require(row["market_state"] in ("normal_trading", "unknown_status", "suspended", "source_gap"),
                "unsupported market state, listing or calendar scope")
        require(row["state_reason"] is None or type(row["state_reason"]) is str, "invalid state reason")
        require(row["close_available_at"][:10] >= row["session"], "close timestamp before session")
        require(bool(row["source_refs"]), "row provenance required")
        for ref in row["source_refs"]:
            digest(ref)
            if v2:
                require(ref in market["source_refs"], "unbound v2 market row reference")
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate market key")
        indexed[key] = row
    require(set(indexed) == {(d, s) for d in calendar for s in universe}, "incomplete market key coverage")
    actions = market["cash_dividends"]
    seen = set()
    require(type(actions) is list, "explicit corporate action scope required")
    for action in actions:
        fields(action, "event_id security_id record_session ex_session pay_session cash_per_unit source_refs")
        text(action["event_id"])
        require(action["event_id"] not in seen and action["security_id"] in universe, "duplicate/unknown action")
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
    fields(profile, "contract_version lot_size settlement_sessions commission_rate minimum_commission_minor tax_rate slippage_bps participation_rate decision_time_utc execution approximation unknown_status_policy limitation" +
           (" price_limit_policy price_grid_policy price_grid_ref price_grid" if v5 else ""))
    require(profile["contract_version"] == ("daily_open_profile_v2" if v5 else "daily_open_profile_v1") and
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
    require(profile["unknown_status_policy"] in ("block", "etf_daily_observed"), "unsupported unknown-status policy")
    if profile["unknown_status_policy"] == "etf_daily_observed":
        require(all(re.fullmatch(r"cn\.etf\.(SSE|SZSE)\.\d{6}\.\d{8}", s) for s in universe),
                "observed-daily profile requires canonical ETF identities")
    fields(plan["initial_account"], "cash_minor positions")
    if v5:
        from .etf_inputs import validate_v5_plan
        validate_v5_plan(plan, profile, universe)
    if v2:
        validate_unit_splits(market, plan["start_session"], plan["end_session"])
    return plan, signal, signals, market, calendar, indexed, profile


def _fees(price, quantity, side, profile):
    gross = minor(price * quantity * 100)
    commission = max(profile["minimum_commission_minor"], minor(Decimal(gross) * decimal(profile["commission_rate"])))
    tax = minor(Decimal(gross) * decimal(profile["tax_rate"])) if side == "SELL" else 0
    return gross, commission, tax


def _fee_components(price, quantity, side, profile, day=None):
    full = profile["contract_version"] == "stock_daily_open_profile_v2"
    if profile["contract_version"] not in ("stock_daily_open_profile_v1", "stock_daily_open_profile_v2"):
        return (*_fees(price, quantity, side, profile), 0)
    if quantity == 0:
        return 0, 0, 0, 0
    if full:
        from .stock_rules import fee_at
        rates = fee_at(profile["stock_fee_schedule"], day)
    else:
        rates = profile
    gross = minor(price * quantity * 100)
    commission = max(profile["minimum_commission_minor"], minor(Decimal(gross) * decimal(profile["commission_rate"])))
    stamp = minor(Decimal(gross) * decimal(rates["sell_stamp_tax_rate"])) if side == "SELL" else 0
    transfer = minor(Decimal(gross) * decimal(rates["transfer_fee_rate"]))
    return gross, commission, stamp, transfer


def _simulate(intent, row, ledger, profile, day, run_id, order_index, unit_splits=None, stock_market=None,
              stock_rules_index=None):
    order = {**intent, "order_id": run_id + ":order:" + str(order_index), "session": day,
             "status": "EXPIRED", "filled_quantity": 0, "unfilled_quantity": intent["quantity"],
             "reason": None, "execution": "daily_open_approximation"}
    order["market_state"] = row["market_state"]
    order["state_reason"] = row["state_reason"]
    order["execution_admission"] = "CONFIRMED_STATUS"
    full = profile["contract_version"] == "stock_daily_open_profile_v2"
    stock = profile["contract_version"] in ("stock_daily_open_profile_v1", "stock_daily_open_profile_v2")
    if full:
        order.update(requested_quantity=intent["quantity"], submitted_quantity=0,
                     unsubmitted_quantity=intent["quantity"], quantity=0, unfilled_quantity=0,
                     stock_execution_rules_ref=profile["stock_execution_rules_ref"], quantity_rule_effective_from=None,
                     submission_reason=None)
    volume_field = "volume_shares" if stock else "volume_units"
    if stock:
        from .stock_inputs import applicable_blocks
        order["execution_evidence_cutoff"] = row["execution_evidence_cutoff"]
        order["field_available_at"] = row["field_available_at"]
        if full and row["_stock_listed"]:
            from ..core.stock_rules import rule_at, legal_quantity
            rule = rule_at(profile["stock_execution_rules"], intent["security_id"], day, validated=stock_rules_index)
            order["quantity_rule_effective_from"] = rule["effective_from"]
        if full and (not row["_stock_listed"] or not row["_stock_factor_valid"]):
            order.update(execution_admission="BLOCKED", reason="MISSING_STOCK_LIFECYCLE_CAPABILITY",
                         capability_missing_reason=row["_stock_factor_missing_reason"],
                         capability_source_ref=row["_stock_factor_source_ref"])
            return order
        blocks = applicable_blocks(stock_market, intent["security_id"], day)
        if blocks:
            order.update(execution_admission="BLOCKED", reason="UNSUPPORTED_STOCK_ACTION", action_blocks=blocks)
            return order
    if unit_splits is not None:
        order["announced_suspension_event_ids"] = suspension_events(unit_splits, intent["security_id"], day)
        if order["announced_suspension_event_ids"]:
            order.update(execution_admission="BLOCKED", reason="ANNOUNCED_SUSPENSION")
            return order
    if row["market_state"] != "normal_trading":
        observed_etf = (profile["unknown_status_policy"] in ("etf_daily_observed", "stock_daily_observed") and
                        row["market_state"] == "unknown_status" and row["state_reason"] in STATUS_GAP_REASONS)
        if not observed_etf:
            order["execution_admission"] = "BLOCKED"
            order["reason"] = "UNKNOWN_MARKET_STATUS" if row["market_state"] == "unknown_status" else "NOT_TRADING"
            return order
        order["execution_admission"] = "STOCK_OBSERVED_DAILY_ASSUMPTION" if stock else "ETF_OBSERVED_DAILY_ASSUMPTION"
    etf_v2 = profile["contract_version"] == "daily_open_profile_v2"
    required = ("open", volume_field) if etf_v2 and profile["price_limit_policy"] == "known_only" else ("open", volume_field, "limit_up", "limit_down")
    if any(row[key] is None for key in required):
        order["reason"] = "MISSING_EXECUTION_FACT"
        return order
    if Decimal(row[volume_field]) <= 0 and not full:
        order["reason"] = "NO_VOLUME"
        return order
    opening = decimal(row["open"])
    buy = intent["side"] == "BUY"
    price = opening * (1 + decimal(profile["slippage_bps"]) / 10000 * (1 if buy else -1))
    rounding = {}
    if etf_v2:
        from .etf_grid import price_tick_for
        tick = decimal(price_tick_for(profile, intent["security_id"]))
        if opening % tick != 0:
            order["reason"] = "PRICE_TICK"
            return order
        raw = price
        price = (raw / tick).to_integral_value(rounding=ROUND_CEILING if buy else ROUND_FLOOR) * tick
        rounding = dict(raw_slipped_price=str(raw), price_tick=str(tick), price_grid_ref=profile["price_grid_ref"],
            price_rounding="adverse_tick", rounding_delta=str(price - raw),
            effective_slippage_bps=str(((price / opening - 1) if buy else (1 - price / opening)) * 10000))
        order.update(rounding)
        if price <= 0:
            order["reason"] = "INVALID_EXECUTION_PRICE"
            return order
    if stock and price % decimal(rule["price_tick"] if full else profile["price_tick"]) != 0:
        order["reason"] = "PRICE_TICK"
        return order
    if ((row["limit_up"] is not None and (price > decimal(row["limit_up"]) or (buy and price == decimal(row["limit_up"])))) or
            (row["limit_down"] is not None and (price < decimal(row["limit_down"]) or (not buy and price == decimal(row["limit_down"]))))):
        order["reason"] = "PRICE_LIMIT"
        return order
    cap = int(Decimal(row[volume_field]) * decimal(profile["participation_rate"]))
    if stock and not full:
        cap = min(cap, profile["maximum_order_quantity"])
    if full:
        held = ledger.positions.get(intent["security_id"], {"quantity": 0, "sellable_quantity": 0})
        submitted = legal_quantity(intent["quantity"], intent["side"], rule,
            held=held["quantity"], sellable=held["sellable_quantity"])
        if submitted < intent["quantity"]:
            order["submission_reason"] = ("ORDER_QUANTITY_MAXIMUM" if intent["quantity"] > rule["daily_proxy_maximum"] else
                ("T_PLUS_ONE_OR_UNSELLABLE" if not buy and held["sellable_quantity"] < intent["quantity"] else
                 "BELOW_MINIMUM_ORDER_QUANTITY" if not submitted else "ORDER_QUANTITY_INCREMENT"))
        if buy and submitted:
            before_cash = submitted
            minimum, increment = rule["buy_minimum"], rule["buy_increment"]
            def affordable(q):
                return sum(_fee_components(price, q, "BUY", profile, day)) <= ledger.cash
            if not affordable(minimum):
                submitted = 0
            else:
                low, high = 0, (submitted - minimum) // increment
                while low < high:
                    mid = (low + high + 1) // 2
                    if affordable(minimum + mid * increment): low = mid
                    else: high = mid - 1
                submitted = minimum + low * increment
            if submitted < before_cash:
                order["submission_reason"] = "INSUFFICIENT_CASH"
        order.update(quantity=submitted, submitted_quantity=submitted,
                     unsubmitted_quantity=intent["quantity"] - submitted, unfilled_quantity=submitted)
        if not submitted:
            order["reason"] = order["submission_reason"] or "NO_LEGAL_SELLABLE_QUANTITY"
            return order
        # Legality applies to the submitted order; native daily capacity can produce one-share partial fills.
        quantity = min(submitted, cap)
    else:
        lot = profile["lot_size"]
        quantity = min(intent["quantity"], cap)
        if buy:
            quantity = quantity // lot * lot
        else:
            held = ledger.positions.get(intent["security_id"], {"quantity": 0, "sellable_quantity": 0})
            quantity = min(quantity, held["sellable_quantity"])
            full_exit = (intent["quantity"] == held["quantity"] == held["sellable_quantity"] and cap >= held["quantity"])
            if not full_exit:
                quantity = quantity // lot * lot
    if buy and quantity and not full:
        low, high = 0, quantity // lot
        while low < high:
            mid = (low + high + 1) // 2
            gross, commission, tax, transfer = _fee_components(price, mid * lot, "BUY", profile)
            if gross + commission + tax + transfer <= ledger.cash:
                low = mid
            else:
                high = mid - 1
        quantity = low * lot
    if not quantity:
        order["reason"] = "NO_VOLUME" if full else ("INSUFFICIENT_CASH" if buy and cap >= lot else "NO_VOLUME_OR_SELLABLE_QUANTITY")
        return order
    gross, commission, tax, transfer = _fee_components(price, quantity, intent["side"], profile, day)
    fee = commission + tax + transfer
    require(not buy or gross + fee <= ledger.cash, "stock fill cash capacity mismatch")
    if not buy and ledger.cash + gross < fee:
        order["reason"] = "INSUFFICIENT_CASH_FOR_FEES"
        return order
    fill = {"fill_id": order["order_id"] + ":fill:0", "order_id": order["order_id"], "session": day,
            "security_id": intent["security_id"], "side": intent["side"], "quantity": quantity,
            "price": str(price), "reference_open": str(opening), "gross_minor": gross,
            "commission_minor": commission, "tax_minor": tax, "fee_minor": fee,
            "slippage_minor": minor(abs(price - opening) * quantity * 100),
            "cash_delta_minor": -gross - fee if buy else gross - fee,
            "source_refs": row["source_refs"]}
    fill.update({"market_state": row["market_state"], "state_reason": row["state_reason"],
                 "execution_admission": order["execution_admission"]})
    fill.update(rounding)
    if stock:
        fill.update(stamp_tax_minor=tax, transfer_fee_minor=transfer, quantity_unit="shares",
                    execution_evidence_cutoff=row["execution_evidence_cutoff"], field_available_at=row["field_available_at"])
    if full:
        from .stock_rules import fee_at
        fill.update(stock_execution_rules_ref=profile["stock_execution_rules_ref"],
                    quantity_rule_effective_from=rule["effective_from"], stock_fee_schedule_ref=profile["stock_fee_schedule_ref"],
                    fee_interval_effective_from=fee_at(profile["stock_fee_schedule"], day)["effective_from"])
    ledger.apply_fill(fill)
    ordered = order["submitted_quantity"] if full else intent["quantity"]
    order.update({"filled_quantity": quantity, "unfilled_quantity": ordered - quantity,
                  "status": "FILLED" if quantity == ordered else "PARTIAL_EXPIRED",
                  "reason": None if quantity == ordered else "CASH_OR_VOLUME_CAP",
                  "committed_sequence": ledger.sequence})
    return order


def run_backtest(request, *, limits=None):
    """Run an explicit frozen request; no Data/Research import, discovery or I/O."""
    require(isinstance(request, BacktestRequest), "BacktestRequest required")
    decoded_plan = None
    if limits is not None:
        from .stock_schedule import _resource_preflight
        decoded_plan = _resource_preflight(request, limits)
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _run(request, decoded_plan=decoded_plan)


def _run(request, *, decoded_plan=None, admitted=None, storage=None):
    plan, signal, signals, market, calendar, rows, profile = (
        _validate(request, decoded_plan=decoded_plan) if admitted is None else admitted)
    streaming = storage is not None
    v5 = plan["contract_version"] == "backtest_request_v5"
    v2 = plan["contract_version"] in ("backtest_request_v2", "backtest_request_v5")
    v4 = plan["contract_version"] == "backtest_request_v4"
    v6 = plan["contract_version"] in ("backtest_request_v6", "backtest_request_v7")
    scheduled = v4 or v6
    stock = plan["contract_version"] in ("backtest_request_v3", "backtest_request_v4", "backtest_request_v6", "backtest_request_v7")
    hold = v5 and plan["portfolio_policy"]["contract_version"] == "etf_buy_and_hold_policy_v1"
    from ..core.etf_buy_hold import BUY_HOLD_VERSION, plan_etf_buy_and_hold
    runtime = "axiom.backtest/6" if v6 else ("axiom.backtest/5" if v5 else ("axiom.backtest/4" if v4 else ("axiom.backtest/3" if stock else (RUNTIME_VERSION if v2 else LEGACY_RUNTIME_VERSION))))
    core_version = CSI300_PORTFOLIO_VERSION if v6 else (BUY_HOLD_VERSION if hold else (TOPK_PORTFOLIO_VERSION if stock else PORTFOLIO_VERSION))
    rules_index = None
    if v6:
        from ..core.stock_rules import validate_execution_rules
        rules_index = storage.audit.globals["rules_index"] if streaming else validate_execution_rules(profile["stock_execution_rules"])
    splits = market.get("unit_splits", [])
    run_id = storage.run_id if streaming else Document.from_dict({"request": plan, "core": core_version, "runtime": runtime,
                                "implementation_ref": IMPLEMENTATION_REF}).identity
    ledger = AccountLedger(cash_minor=plan["initial_account"]["cash_minor"], calendar=calendar,
                           settlement_sessions=profile["settlement_sessions"], positions=plan["initial_account"]["positions"],
                           output_guard=storage.reserve_output if streaming else None)
    quotes, marks, entitlements = {}, {}, {}
    registrations, applications, mark_basis = {}, [], {}
    nav, positions, orders, decisions = [], [], [], []
    order_index = 0
    calendar_index = {day: index for index, day in enumerate(calendar)}
    if streaming:
        storage.attach(ledger, nav, positions, orders, decisions)
    def append_output(kind, row):
        if streaming:
            storage.reserve_output(kind, row)
        {"nav": nav, "positions": positions, "orders": orders, "decisions": decisions}[kind].append(row)
    initial_value = None
    stopped = None
    lifecycle = None
    if v6:
        lifecycle = dict(storage.audit.lifecycle) if streaming else dict(pre_listing_null=0, listed_nonmember_gap=0, member_gap=0, held_gap=0)
        if not streaming:
            for row in rows.values():
                gap = not row["_stock_factor_valid"] or any(row[name] is None for name in
                    ("open", "close", "volume_shares", "limit_up", "limit_down"))
                if row["market_state"] == "not_listed": lifecycle["pre_listing_null"] += 1
                elif row["_stock_listed"] and gap: lifecycle["member_gap" if row["_stock_member"] else "listed_nonmember_gap"] += 1
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
            if stock:
                from .stock_inputs import applicable_blocks
                if v6:
                    lifecycle["held_gap"] += sum(1 for security, position in ledger.positions.items() if position["quantity"] and
                        (not rows[day, security]["_stock_listed"] or not rows[day, security]["_stock_factor_valid"] or
                         any(rows[day, security][name] is None for name in ("open", "close", "volume_shares", "limit_up", "limit_down"))))
                    gaps = [{"security_id": security, "listed": rows[day, security]["_stock_listed"],
                        "factor_missing_reason": rows[day, security]["_stock_factor_missing_reason"],
                        "factor_source_ref": rows[day, security]["_stock_factor_source_ref"]}
                        for security, position in ledger.positions.items() if position["quantity"] and
                        (not rows[day, security]["_stock_listed"] or not rows[day, security]["_stock_factor_valid"])]
                    if gaps:
                        stopped = {"session": day, "reason": "HELD_MISSING_STOCK_LIFECYCLE_CAPABILITY", "gaps": gaps,
                                   "committed_sequence": ledger.sequence}
                        if streaming:
                            storage.flush(day, "STOPPED_BEFORE_NAV", ledger.sequence)
                        break
                blocking = [block for security, position in ledger.positions.items() if position["quantity"]
                            for block in applicable_blocks(market, security, day)]
                if blocking:
                    stopped = {"session": day, "reason": "HELD_UNSUPPORTED_STOCK_ACTION", "blocks": blocking,
                               "committed_sequence": ledger.sequence}
                    if streaming:
                        storage.flush(day, "STOPPED_BEFORE_NAV", ledger.sequence)
                    break
            for phase, key in (("EX", "ex_session"), ("PAY", "pay_session")):
                for action in sorted(market["cash_dividends"], key=lambda a: a["event_id"]):
                    if action[key] == day and action["event_id"] in entitlements:
                        ledger.dividend(action, phase, entitlements.get(action["event_id"], 0))
            previous = calendar[index - 1]
            rebalance = day == plan["start_session"] if hold else date.fromisoformat(day).isocalendar()[:2] != date.fromisoformat(previous).isocalendar()[:2]
            if rebalance:
                active, active_rows = signals[day] if scheduled else (signal, signals)
                context = {"trade_session": day, "decision_time": day + "T" + profile["decision_time_utc"], "reference_prices": quotes,
                           "account_state_version": ledger.sequence,
                           **{key: profile[key] for key in (("commission_rate", "minimum_commission_minor", "slippage_bps") if v6 else
                                                          ("lot_size", "commission_rate", "minimum_commission_minor", "slippage_bps"))}}
                if hold:
                    context.update(reference_session=previous, reference_cutoff=previous + "T12:30:00Z", tax_rate=profile["tax_rate"])
                    decision = plan_etf_buy_and_hold(plan["portfolio_policy"], account=ledger.account(), context=context).to_dict()
                else:
                    first = active_rows.get((previous, active["universe"][0]))
                    require(first is not None, "missing previous-session signal")
                    context.update(feature_session=previous, knowledge_cutoff=first["knowledge_cutoff"])
                if stock:
                    context.update(supported_security_ids=plan["execution_universe"], supported_universe_ref=plan["supported_universe_ref"])
                    if v6:
                        context.update(portfolio_policy=plan["portfolio_policy"], stock_execution_rules=profile["stock_execution_rules"],
                                       stock_execution_rules_ref=plan["stock_execution_rules_ref"])
                    if scheduled:
                        context["feature_knowledge_cutoff"] = first["feature_knowledge_cutoff"]
                        decision = _plan_admitted_stock_portfolio(active, active_rows, account=ledger.account(),
                            context=context, top_k=plan["portfolio_policy"]["top_k"], rules_index=rules_index).to_dict()
                    else:
                        decision = plan_stock_portfolio(StockPredictionFrame.from_dict(signal), account=ledger.account(),
                            context=context, top_k=plan["portfolio_policy"]["top_k"]).to_dict()
                elif not hold:
                    context["tax_rate"] = profile["tax_rate"]
                    decision = plan_rotation(SignalFrame.from_dict(signal), account=ledger.account(), context=context).to_dict()
                if v2 or stock:
                    decision["reference_prices"] = {s: dict(q) for s, q in quotes.items()}
                append_output("decisions", decision)
                for intent in decision["intents"]:
                    order = _simulate(intent, rows[day, intent["security_id"]], ledger, profile, day, run_id, order_index,
                                      splits if v2 else None, market if stock else None, stock_rules_index=rules_index)
                    order_index += 1
                    append_output("orders", order)
                    if streaming:
                        storage.observe_order(order)
                if streaming:
                    # These borrow the current source block or pending output.
                    # Drop them before a later session can advance that block.
                    active = active_rows = first = context = decision = intent = order = None
            for action in market["cash_dividends"]:
                if action["record_session"] == day:
                    entitlements[action["event_id"]] = ledger.positions.get(action["security_id"], {"quantity": 0})["quantity"]
            for item in splits:
                event = item["event"]
                if event["record_date"] == day:
                    position = ledger.positions.get(event["security_id"], {"quantity": 0, "sellable_quantity": 0, "cost_minor": 0})
                    require(not any(lot["security_id"] == event["security_id"] for lot in ledger.pending) and
                            position["quantity"] == position["sellable_quantity"],
                            "unit split registration requires fully settled entitlement")
                    registrations[event["event_id"]] = {"sequence": ledger.sequence, "position": dict(position)}
        for security in sorted(market["universe"]):
            row = rows[day, security]
            if row["close"] is not None:
                quote = {"price": row["close"], "session": day, "available_at": row["close_available_at"], "source_refs": row["source_refs"]}
                quotes[security] = quote
                marks[security] = quote
                mark_basis[security] = None
        if streaming:
            row = None
        if day < plan["start_session"]:
            continue
        for item in sorted(splits, key=lambda i: i["event"]["event_id"]):
            event = item["event"]
            if event["effective_date"] == day:
                registration = registrations.get(event["event_id"])
                require(registration is not None, "unit split registration absent")
                application = ledger.unit_split(item, registration, quotes.get(event["security_id"]))
                if application is not None:
                    applications.append(application)
                    quotes[event["security_id"]] = marks[event["security_id"]] = application["normalized_quote"]
                    mark_basis[event["security_id"]] = event["event_id"]
        ledger.sequence += 1  # commit the session valuation at one shared watermark
        value = 0
        for security, position in sorted(ledger.positions.items()):
            if not position["quantity"]:
                continue
            require(security in marks, "held position lacks observed valuation price")
            mark = marks[security]
            amount = minor(decimal(mark["price"]) * position["quantity"] * 100)
            value += amount
            point = {"session": day, "security_id": security, **position,
                "mark_price": mark["price"], "mark_session": mark["session"],
                "is_stale": mark["session"] != day, "stale_sessions": index - calendar_index[mark["session"]],
                "mark_source_refs": mark["source_refs"], "market_value_minor": amount,
                "committed_sequence": ledger.sequence}
            if v2:
                point["mark_basis_event_id"] = mark_basis.get(security)
            if v6:
                point["stale_reason"] = rows[day, security]["_stock_close_missing_reason"] if mark["session"] != day else None
            append_output("positions", point)
        if streaming:
            point = None
        receivable = sum(ledger.receivables.values())
        total = ledger.cash + value + receivable
        append_output("nav", {"session": day, "cash_minor": ledger.cash, "market_value_minor": value,
            "receivable_minor": receivable, "nav_minor": total,
            "nav_index": str(Decimal(total) / initial_value), "committed_sequence": ledger.sequence})
        if streaming:
            storage.observe_nav(nav[-1], initial_value)
            storage.flush(day, "SESSION_COMMITTED", ledger.sequence)
    if streaming:
        limitations = [profile["limitation"],
            "Daily open-price approximation; full-day volume is an execution-side capacity proxy, not known opening liquidity.",
            "Day orders expire after one simulated fill; no live broker, SQLite recovery, general stock quantity-action or delisting support.",
            "Dividend handling supports explicit record/ex/pay cash events; unsupported economic events must be rejected by the caller adapter.",
            "Metrics cover the frozen stock sample; separate saved evaluation owns annualization.",
            *market["limitations"], *signal["limitations"], *plan["limitations"]]
        return storage.result(ledger=ledger, initial_value=initial_value, stopped=stopped,
                              lifecycle=lifecycle, limitations=limitations)
    peak, drawdown = initial_value, Decimal(0)
    for point in nav:
        peak = max(peak, point["nav_minor"])
        drawdown = min(drawdown, Decimal(point["nav_minor"]) / peak - 1)
    result = {"contract_version": "backtest_run_v6" if v6 else ("backtest_run_v5" if v5 else ("backtest_run_v4" if v4 else ("backtest_run_v3" if stock else ("backtest_run_v2" if v2 else "backtest_run_v1")))), "run_id": run_id, "account_id": plan["account_id"],
        "status": "BLOCKED" if stopped else "COMPLETE", "plan": plan, "signal_ref": None if hold else (signal["schedule_ref"] if scheduled else signal["signal_run_ref"]),
        "market_ref": MarketReplay.from_dict(market).identity, "profile_ref": Document.from_dict(profile).identity,
        "core_version": core_version, "runtime_version": runtime,
        "implementation_ref": IMPLEMENTATION_REF,
        "committed_sequence": ledger.sequence, "initial_nav_minor": initial_value,
        "final_account": {"cash_minor": ledger.cash, "receivable_minor": sum(ledger.receivables.values()),
                          "positions": ledger.positions, "committed_sequence": ledger.sequence},
        "decisions": decisions, "orders": orders, "fills": ledger.fills,
        "cash_ledger": ledger.cash_ledger, "position_ledger": ledger.position_ledger,
        "nav": nav, "positions": positions,
        "metrics": {"total_return": None if stopped else str(Decimal(nav[-1]["nav_minor"]) / initial_value - 1),
                    "max_drawdown": None if stopped else str(drawdown), "total_fees_minor": sum(f["fee_minor"] for f in ledger.fills),
                    "turnover_minor": sum(f["gross_minor"] for f in ledger.fills), "fill_count": len(ledger.fills),
                    "unfilled_order_count": sum(o["unfilled_quantity"] > 0 for o in orders)},
        "limitations": [profile["limitation"], "Daily open-price approximation; full-day volume is an execution-side capacity proxy, not known opening liquidity.",
                         "Day orders expire after one simulated fill; no live broker, SQLite recovery, split or delisting support.",
                         "Dividend handling supports explicit record/ex/pay cash events; unsupported economic events must be rejected by the caller adapter.",
                         "Metrics cover the fixed short sample only; no OOS, CAGR or Sharpe claim.", *market["limitations"]]}
    if profile["unknown_status_policy"] == "etf_daily_observed":
        result["limitations"].insert(0, "EXPLICIT ETF DAILY APPROXIMATION: UNKNOWN status is retained, not promoted to normal_trading; missing-status executions assume observed daily open/volume/limits and cannot establish opening liquidity. Not a live execution profile.")
    if v2:
        result["unit_split_applications"] = applications
        result["limitations"] = [s.replace("split or delisting support", "general corporate-action or delisting support") for s in result["limitations"]]
        result["limitations"] = [s.replace("Metrics cover the fixed short sample only; no OOS, CAGR or Sharpe claim.",
            "Metrics cover the frozen account scope; any annualization belongs to a separate saved evaluation.") for s in result["limitations"]]
        result["limitations"].extend([
            "ETF unit replacement applies visible planned arrangements under a single-holder fully settled EOD simulation model; APPLIED is not issuer implemented status.",
            "Date-only next-open availability is best effort; actual receipt remains unchanged and strict historical PIT is not established.",
            "Issuer not_stated phase remains native; EOD and no extra T+1 are Runtime conventions. UNKNOWN and missing bars remain; disclosed full-session suspension is an additional hard block."])
    if v5:
        result.update(portfolio_policy_ref=Document.from_dict(plan["portfolio_policy"]).identity,
                      price_unit=plan["price_unit"], quantity_unit="fund units")
    if stock:
        result.update(quantity_unit="shares", price_unit="CNY/share", stopped=stopped,
                      admission_ref=plan["admission_ref"], supported_universe_ref=plan["supported_universe_ref"])
        if v6:
            result["stock_execution_rules_ref"] = plan["stock_execution_rules_ref"]
            result["lifecycle_admission"] = lifecycle
            result["metrics"].update(unsubmitted_order_count=sum(o["unsubmitted_quantity"] > 0 for o in orders),
                unsubmitted_quantity=sum(o["unsubmitted_quantity"] for o in orders),
                incomplete_order_count=sum(o["unsubmitted_quantity"] + o["unfilled_quantity"] > 0 for o in orders))
        result["limitations"] += signal["limitations"]
        result["limitations"].insert(0, profile["limitation"])
        result["limitations"] = [value.replace("split or delisting support", "general stock quantity-action or delisting support")
            .replace("Metrics cover the fixed short sample only; no OOS, CAGR or Sharpe claim.",
                     "Metrics cover the frozen stock sample; separate saved evaluation owns annualization.") for value in result["limitations"]]
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
    v4 = wire.get("contract_version") == "backtest_run_v4"
    v6 = wire.get("contract_version") == "backtest_run_v6"
    stock = wire.get("contract_version") in ("backtest_run_v3", "backtest_run_v4", "backtest_run_v6")
    require(wire.get("contract_version") in ("backtest_run_v1", "backtest_run_v2", "backtest_run_v3", "backtest_run_v4", "backtest_run_v5", "backtest_run_v6") and
            wire.get("status") in (("COMPLETE", "BLOCKED") if stock else ("COMPLETE",)), "unsupported saved result")
    recorded_digest = wire.pop("content_digest", None)
    require(recorded_digest == Document.from_dict(wire).identity, "saved result content digest mismatch")
    require(wire["run_id"] == Document.from_dict({"request": wire["plan"],
            "core": wire["core_version"], "runtime": wire["runtime_version"],
            "implementation_ref": wire["implementation_ref"]}).identity,
            "saved run identity mismatch")
    if wire["contract_version"] == "backtest_run_v2":
        require(wire["runtime_version"] == "axiom.backtest/2" and wire["plan"]["contract_version"] == "backtest_request_v2",
                "saved v2 tuple mismatch")
        _validate(BacktestRequest.from_dict(wire["plan"]))
        from .unit_splits import validate_saved_applications
        validate_saved_applications(wire)
    if wire["contract_version"] == "backtest_run_v5":
        from .etf_inputs import validate_saved_v5
        validate_saved_v5(wire)
    if stock:
        require(wire["runtime_version"] == ("axiom.backtest/6" if v6 else ("axiom.backtest/4" if v4 else "axiom.backtest/3")) and
                wire["core_version"] in ((CSI300_PORTFOLIO_VERSION,) if v6 else (STOCK_PORTFOLIO_VERSION, TOPK_PORTFOLIO_VERSION)) and
                wire["plan"]["contract_version"] == ("backtest_request_v6" if v6 else ("backtest_request_v4" if v4 else "backtest_request_v3")), "saved stock tuple mismatch")
        from .stock_inputs import validate_saved_stock_core, validate_stock_request
        validate_stock_request(wire["plan"], legacy_saved_top5=wire["core_version"] == STOCK_PORTFOLIO_VERSION)
        validate_saved_stock_core(wire)
        require((wire["status"] == "BLOCKED") == (wire["stopped"] is not None), "saved stock stop status mismatch")
    return saved
