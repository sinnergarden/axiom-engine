"""Validate saved strategy identities and clocks without calling a planner."""
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from copy import deepcopy
import os

from ..core.contracts import Document, canonical, digest, fields, integer, require
from ..core.portfolio import decimal
from ..core.stock_portfolio import instant
from ..core.stock_strategy import (initial_strategy_state, validate_strategy_state,
    validate_market_observation, market_gate_transition)

RISK_INPUT_LIMIT_BYTES = 4*1024*1024


def validate_strategy_risk_input(risk_input, reference, spec, scope):
    """Bound the original inline native dependency separately from execution market inputs."""
    from .stock_evidence import native_ref
    gate=spec["market_gate"]
    fields(risk_input,"contract_version native_ref view")
    require(risk_input["contract_version"]=="stock_market_risk_input_v1", "unsupported market risk input")
    digest(reference);digest(risk_input["native_ref"])
    require(len(canonical(risk_input).encode())<=RISK_INPUT_LIMIT_BYTES, "inline market risk input exceeds bound")
    require(Document.from_dict(risk_input).identity==reference, "market risk input identity mismatch")
    view=risk_input["view"]
    fields(view,"context field_meta records")
    require(native_ref(view)==risk_input["native_ref"], "market risk original native identity mismatch")
    context=view["context"];query=context["query"]
    require(context["domain"]=="benchmark_daily" and context["contract_version"]=="data_batch_v1" and
        query["fields"]==["close"] and query["symbols"]==[gate["security_id"]] and
        query["purpose"]=="decision_facts" and query["price_basis"]=="unadjusted" and
        query["pit_policy"] in ("operational_pit_v1","market_pit_safe_v1","best_effort_vendor_v1"),
        "market risk requires original decision facts")
    days=query["sessions"];n=gate["window_sessions"]
    require(type(days) is list and days==sorted(set(days)) and len(days)==n-1+len(scope["calendar"])-1 and
        days[n-1:]==scope["calendar"][:-1] and all(d<scope["anchor_session"] for d in days[:n-1]),
        "market risk needs complete preheat and prior-session scope")
    require(query["cutoff_by_session"]=={d:d+"T20:30:00+08:00" for d in days},
        "market risk per-session knowledge cutoff mismatch")
    fields(view["field_meta"],"close")
    field=view["field_meta"]["close"]
    fields(field,"dtype unit by_key")
    require(field["dtype"]=="float64" and field["unit"]=="index points", "market risk native close unit mismatch")
    records=view["records"];metadata=field["by_key"]
    require(type(records) is list and type(metadata) is list and len(records)==len(metadata)==len(days),
        "market risk native coverage mismatch")
    points=[]
    for day,row,meta in zip(days,records,metadata):
        fields(row,"security_id session close")
        for key in ("revision_id","raw_batch_id","availability_basis"):
            require(type(meta[key]) is str and bool(meta[key]), "market risk native provenance required")
        instant(meta["first_observed_at"])
        require(row["security_id"]==meta["security_id"]==gate["security_id"] and
            row["session"]==meta["session"]==day and row["close"] is not None and
            meta["missing_reason"] is None and instant(meta["usable_from"])<=instant(query["cutoff_by_session"][day]),
            "missing or unavailable market risk fact")
        require(type(row["close"]) in (int,float,str) and decimal(str(row["close"]))>0,
                "positive native risk close required")
        points.append(dict(session=day,close=row["close"],**{k:meta[k] for k in
            ("usable_from","revision_id","raw_batch_id","first_observed_at","availability_basis","evidence_ref")}))
    return points


class BoundMarketRiskInput:
    """One invocation's small admitted observation index, never a persisted PASS carrier."""
    def __init__(self, request):
        spec=request["portfolio_policy"]["strategy_spec"]
        self.reference=request["risk_input_ref"];self.native_ref=request["risk_input"]["native_ref"]
        self.points=validate_strategy_risk_input(request["risk_input"],self.reference,spec,request["scope"])
        self.window=spec["market_gate"]["window_sessions"];self.security=spec["market_gate"]["security_id"]
        self.by_day={p["session"]:i for i,p in enumerate(self.points)};self.pid=os.getpid()

    def observation(self, feature, reference):
        require(os.getpid()==self.pid and reference==self.reference, "risk input differs from admitted binding")
        index=self.by_day.get(feature,-1)
        require(index>=self.window-1,"risk observation lacks prior window")
        wire=dict(contract_version="stock_market_observation_v1",risk_input_ref=self.reference,
            native_ref=self.native_ref,security_id=self.security,feature_session=feature,
            knowledge_cutoff=feature+"T20:30:00+08:00",points=deepcopy(self.points[index-self.window+1:index+1]))
        wire["observation_ref"]=Document.from_dict(wire).identity
        return wire


def registered_market_risk_rights(ledger, actions):
    """Actual pre-EX stock rights, including registrations based on pending shares."""
    reserved = set()
    for action in actions:
        state = ledger.equity_entitlements.get(action["economic_event_id"])
        if state is None or state["ex_applied"] or not (
                state["record_quantity"] or state.get("record_pending_quantity", 0)):
            continue
        total = action["stock_distribution_shares_per_share"]
        if total is None or decimal(total) > 0:
            reserved.add(action["security_id"])
    return sorted(reserved)


def _saved_market_gate(decision, spec, previous_state, universe):
    targets = decision["strategy_targets"]
    traces = [t for t in targets["trace"] if t.get("reason") == "MARKET_GATE"]
    require(len(traces) == 1, "saved market gate trace required")
    trace = traces[0]
    require([t for t in decision["trace"] if t.get("reason") == "MARKET_GATE"] == traces,
            "saved market gate display trace differs from targets")
    fields(trace, "reason observation risk_input_ref risk_on average_close mode_before mode_after ownership")
    risk_on, average = validate_market_observation(trace["observation"], spec["market_gate"],
        decision["feature_session"], trace["risk_input_ref"])
    require(type(trace["risk_on"]) is bool and trace["risk_on"] == risk_on and
            trace["average_close"] == average and
            decision.get("risk_input_ref") == trace["risk_input_ref"] and
            decision.get("market_observation_ref") == trace["observation"]["observation_ref"],
            "saved market gate observation/result mismatch")
    ownership = trace["ownership"]
    fields(ownership, "positions pending_share_quantities reserved_security_ids registered_stock_right_security_ids")
    require(type(ownership["positions"]) is dict and type(ownership["pending_share_quantities"]) is dict,
            "saved market ownership maps required")
    for position in ownership["positions"].values():
        fields(position, "quantity sellable_quantity")
        integer(position["quantity"], 1); integer(position["sellable_quantity"])
        require(position["quantity"] >= position["sellable_quantity"], "saved market sellability mismatch")
    for quantity in ownership["pending_share_quantities"].values(): integer(quantity, 1)
    for key in ("reserved_security_ids", "registered_stock_right_security_ids"):
        require(type(ownership[key]) is list and ownership[key] == sorted(set(ownership[key])),
                "saved registered market rights required")
    owned = (set(ownership["positions"]) | set(ownership["pending_share_quantities"]) |
             set(ownership["reserved_security_ids"]) | set(ownership["registered_stock_right_security_ids"]))
    require(owned <= set(universe), "saved market ownership outside scope")
    mode, suppressed = market_gate_transition(previous_state["market_mode"], risk_on, owned)
    require(trace["mode_before"] == previous_state["market_mode"] and trace["mode_after"] == mode and
            targets["next_state"]["market_mode"] == mode, "saved market latch transition mismatch")
    if suppressed:
        require(not targets["selected_security_ids"] and not targets["target_weights"] and
                not targets["allocation_ratios"] and not targets["preserve_quantities"] and
                targets["force_exit_security_ids"] == sorted(owned) and
                not any(i["side"] == "BUY" for i in decision["intents"]),
                "saved market liquidation refilled before flat risk-on")
    return trace


def validate_saved_strategy_decision(decision, policy, universe, previous_state, positions):
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _validate(decision, policy, universe, previous_state, positions)


def _validate(decision, policy, universe, previous_state, positions):
    gated = "market_gate" in policy["strategy_spec"]
    # The saved-reader entry historically starts with the legacy initializer.
    # Derive the gated first state from its request spec inside this validator.
    if gated and previous_state == initial_strategy_state():
        previous_state = initial_strategy_state(policy["strategy_spec"])
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
    require(("market_mode" in previous_state) == gated and ("market_mode" in targets["next_state"]) == gated,
            "saved market state/spec mismatch")
    if gated:
        _saved_market_gate(decision, policy["strategy_spec"], previous_state, universe)
    else:
        require(not (set(decision) & {"risk_input_ref", "market_observation_ref"}) and
                not any(t.get("reason") == "MARKET_GATE" for t in targets["trace"]),
                "ungated saved strategy carries market gate fields")
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


def _saved_market_decision_versions(rows):
    """Use validated session boundaries and Runtime's existing pre-trade event phases.

    SESSION_COMMITTED includes the closing NAV sequence. SETTLEMENT, equity
    EX/LISTING and cash EX/PAY precede planning; FILL and REGISTER follow it.
    This binds the decision clock without creating another account replay.
    """
    phases = {p["session"]:p["committed_sequence"] for p in rows["session_phases"]}
    versions = {}; previous = 0
    for day in sorted(phases):
        versions[day] = previous
        previous = phases[day]
    starts = dict(versions)
    for kind, reasons in (("position_ledger", {"SETTLEMENT", "EQUITY_EX", "EQUITY_LISTING"}),
                          ("cash_ledger", {"DIVIDEND_EX", "DIVIDEND_PAY"})):
        for row in rows[kind]:
            if row["reason"] in reasons:
                day = row["session"]
                require(day in versions and starts[day] < row["sequence"] <= phases[day],
                        "saved pre-trade event differs from session boundary")
                versions[day] = max(versions[day],row["sequence"])
    return versions


def validate_saved_strategy_capacity(wire, rows):
    """Count actual shares, pending grants and registered stock-right slots."""
    request = wire["request_manifest"]
    capacity = request["portfolio_policy"]["top_k"]
    quantities, pending, reservations = {}, {}, {}
    gated = "market_gate" in request["portfolio_policy"]["strategy_spec"]
    if gated:
        bound = BoundMarketRiskInput(request)
        decision_versions = _saved_market_decision_versions(rows)
        sellable, risk_reservations = {}, {}
        decisions = iter(rows["decisions"])
        decision = next(decisions, None)
        previous_version = -1
        def check_at_version(item):
            nonlocal previous_version
            version = item.get("expected_account_version")
            integer(version)
            require(previous_version <= version <= wire["committed_sequence"],
                    "saved market decision watermark order mismatch")
            require(version == decision_versions.get(item["trade_session"]),
                    "saved market decision version differs from pre-trade session boundary")
            previous_version = version
            trace = next(t for t in item["strategy_targets"]["trace"] if t.get("reason") == "MARKET_GATE")
            require(item["risk_input_ref"] == request["risk_input_ref"] and
                    trace["observation"] == bound.observation(item["feature_session"], item["risk_input_ref"]),
                    "saved market observation differs from original native dependency")
            actual = dict(positions={s:dict(quantity=q,sellable_quantity=sellable.get(s,0))
                            for s,q in quantities.items() if q},
                pending_share_quantities={s:q for s,q in pending.items() if q},
                reserved_security_ids=sorted(set(reservations.values())),
                registered_stock_right_security_ids=sorted(set(risk_reservations.values())))
            require(trace["ownership"] == actual,
                    "saved market ownership differs at decision account version")
    events = wire["account_events"].get("equity_facts", {}).get("actions", [])
    actions = {a["economic_event_id"]: a for a in events}
    for row in rows["position_ledger"]:
        if gated:
            while decision is not None and decision["expected_account_version"] < row["sequence"]:
                check_at_version(decision)
                decision = next(decisions, None)
        s, event = row["security_id"], row["source_event_id"]
        quantities[s] = quantities.get(s, 0)+row["quantity_delta"]
        pending[s] = pending.get(s, 0)+row.get("pending_share_delta", 0)
        if gated:
            sellable[s] = sellable.get(s, 0)+row["sellable_delta"]
            if row["reason"] == "EQUITY_REGISTER" and (row.get("record_quantity",0) or row.get("record_pending_quantity",0)):
                total = actions[event]["stock_distribution_shares_per_share"]
                if total is None or decimal(total)>0: risk_reservations[event] = s
            elif row["reason"] in ("EQUITY_EX", "EQUITY_LISTING"):
                risk_reservations.pop(event, None)
        if row["reason"] == "EQUITY_REGISTER" and row.get("record_quantity", 0):
            total = actions[event]["stock_distribution_shares_per_share"]
            if total is None or decimal(total) > 0: reservations[event] = s
        elif row["reason"] in ("EQUITY_EX", "EQUITY_LISTING"):
            reservations.pop(event, None)
        owned = {s for s,q in quantities.items() if q} | {s for s,q in pending.items() if q} | set(reservations.values())
        require(len(owned) <= capacity, "saved account exceeds owned-position capacity")
        if not quantities[s]: quantities.pop(s)
        if not pending[s]: pending.pop(s)
    if gated:
        while decision is not None:
            check_at_version(decision)
            decision = next(decisions, None)
