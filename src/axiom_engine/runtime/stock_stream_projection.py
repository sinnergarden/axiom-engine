"""Validate a saved v7 output projection without reopening its large sources.

This loader checks saved output, linkage and fees from the small profile. It
does not establish native-price, PIT or prediction-source numerical validity.
That is the separate full-source audit's responsibility.
"""
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from hashlib import sha256
import json
from pathlib import Path

from ..core.contracts import (ContractError, _pairs, _walk, digest, fields, integer,
                              require, session)
from ..core.portfolio import decimal, minor
from .stock_stream_outputs import PHASES, RESULT_KINDS, RESULT_PART_VERSION, _json_pieces


_LOADER_TOKEN = object()
RUN_FIELDS = ("contract_version run_id account_id status request_manifest request_ref source_audit "
              "source_audit_ref signal_ref market_ref profile_ref core_version runtime_version "
              "implementation_ref committed_sequence initial_nav_minor final_account stopped "
              "lifecycle_admission metrics limitations result_parts account_events account_events_ref content_digest")


@dataclass(frozen=True, init=False)
class SavedRunProjection:
    """Only the loader constructs this validated view; it is not a v6 run."""
    wire: dict
    rows: dict
    profile: dict
    calendar: tuple
    _digests: tuple

    def __init__(self, wire, rows, profile, calendar, *, _token=None):
        require(_token is _LOADER_TOKEN, "use load_stock_backtest_projection")
        object.__setattr__(self, "wire", wire)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "calendar", tuple(calendar))
        object.__setattr__(self, "_digests", (_hash_wire(wire), _hash_wire(rows),
                                             _hash_wire(profile), _hash_wire(tuple(calendar))))

    @property
    def input_run_ref(self):
        return {key: self.wire[key] for key in ("run_id", "content_digest", "committed_sequence")}

    @property
    def account_events(self):
        return self.wire["account_events"]

    def verify(self):
        """Check the shared in-memory view without I/O or whole-graph bytes."""
        require(self._digests == (_hash_wire(self.wire), _hash_wire(self.rows),
                                 _hash_wire(self.profile), _hash_wire(self.calendar)),
                "saved projection was modified after loading")
        return self


def _hash_wire(wire):
    hashed = sha256()
    try:
        for _ in _walk(wire):
            pass
        for piece in _json_pieces(wire):
            hashed.update(piece)
    except (UnicodeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid saved canonical JSON") from exc
    return "sha256:" + hashed.hexdigest()


def _bounded_object(path, *, maximum, read_size):
    """Stat before allocation and enforce the bound again while reading."""
    path = Path(path)
    integer(maximum, 1); integer(read_size, 1)
    expected_size = path.stat().st_size
    require(0 < expected_size <= maximum, "saved object exceeds decoded/read budget")
    buffer = bytearray()
    byte_hash = sha256()
    with path.open("rb") as saved:
        while True:
            remaining = maximum - len(buffer)
            piece = saved.read(min(read_size, remaining) if remaining else 1)
            if not piece:
                break
            require(len(buffer) + len(piece) <= maximum, "saved object grew beyond decoded budget")
            byte_hash.update(piece)
            buffer.extend(piece)
    require(len(buffer) == expected_size and path.stat().st_size == expected_size,
            "saved object size changed during read")
    try:
        wire = json.loads(buffer, object_pairs_hook=_pairs)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ContractError("invalid saved JSON") from exc
    require(type(wire) is dict, "saved object must be JSON object")
    return wire, len(buffer), "sha256:" + byte_hash.hexdigest()


def _artifact(ref):
    fields(ref, "artifact_type artifact_id contract_version manifest_uri content_digest")
    for name in ("artifact_type", "artifact_id", "contract_version", "manifest_uri"):
        require(type(ref[name]) is str and bool(ref[name]), "invalid saved ArtifactRef")
    digest(ref["content_digest"])


def _run_header(wire):
    from .stock_stream_contracts import validate_manifest
    require("account_events" in wire and "account_events_ref" in wire,
            "saved account event view/ref required")
    fields(wire, RUN_FIELDS)
    require(wire["contract_version"] == "backtest_run_v7" and
            wire["runtime_version"] == "axiom.backtest/7" and
            wire["core_version"] == "axiom.stock_portfolio/3", "invalid saved v7 version tuple")
    require(wire["status"] in ("COMPLETE", "BLOCKED"), "invalid saved run status")
    for key in ("run_id", "request_ref", "source_audit_ref", "signal_ref", "market_ref",
                "profile_ref", "implementation_ref", "account_events_ref", "content_digest"):
        digest(wire[key])
    unsigned = {key: value for key, value in wire.items() if key != "content_digest"}
    require(wire["content_digest"] == _hash_wire(unsigned), "saved v7 content digest mismatch")
    request = wire["request_manifest"]
    validate_manifest(request)
    require(wire["request_ref"] == request["request_ref"] and
            wire["account_id"] == request["account_id"] and
            wire["signal_ref"] == request["prediction_input"]["prediction_ref"] and
            wire["market_ref"] == request["market_input"]["market_ref"] and
            wire["profile_ref"] == request["profile_input"]["profile_ref"],
            "saved v7 request/reference binding mismatch")
    require(wire["run_id"] == _hash_wire({"request_ref": wire["request_ref"],
            "core_version": wire["core_version"], "runtime_version": wire["runtime_version"],
            "implementation_ref": wire["implementation_ref"]}), "saved v7 run identity mismatch")
    audit = wire["source_audit"]
    from .stock_stream_contracts import validate_source_audit
    validate_source_audit(audit,request)
    require(
            wire["source_audit_ref"] == _hash_wire(audit) and
            all(audit[name] == wire[name] for name in
                ("request_ref", "market_ref", "profile_ref", "implementation_ref")) and
            audit["prediction_ref"] == wire["signal_ref"], "saved source-audit content binding mismatch")
    require(type(audit["counts"]) is dict and type(audit["limitations"]) is list,
            "saved source-audit counts/limitations required")
    for count in audit["counts"].values():
        integer(count)
    _account_events(wire, request, audit)
    integer(wire["committed_sequence"]); integer(wire["initial_nav_minor"], 1)
    require(wire["initial_nav_minor"] == request["initial_account"]["cash_minor"],
            "saved initial account binding mismatch")
    require(type(wire["result_parts"]) is list and bool(wire["result_parts"]), "saved result parts required")
    require((wire["stopped"] is None) == (wire["status"] == "COMPLETE"), "saved status/stopped mismatch")
    return request


def _account_events(wire, request, audit):
    """The small saved view is a binding, not another native-source audit."""
    from .stock_inputs import validate_cash_action
    events = wire["account_events"]
    fields(events, "contract_version request_ref market_ref profile_ref cash_dividends source_refs limitations")
    require(events["contract_version"] == "stock_account_events_v1" and
            wire["account_events_ref"] == _hash_wire(events) and
            all(events[name] == wire[name] for name in ("request_ref", "market_ref", "profile_ref")),
            "saved account event content/reference binding mismatch")
    refs = events["source_refs"]
    execution_refs = {item["native_ref"] for item in request["market_input"]["native_inputs"]
                      if item["role"] == "execution"}
    require(type(refs) is list and bool(refs) and len(set(refs)) == len(refs) and
            set(refs) <= execution_refs, "saved account event refs outside execution inputs")
    for ref in refs:
        digest(ref)
    require(type(events["cash_dividends"]) is list and
            type(audit["counts"].get("cash_actions")) is int and
            len(events["cash_dividends"]) == audit["counts"]["cash_actions"],
            "saved account cash action count mismatch")
    require(type(events["limitations"]) is list and
            all(type(value) is str for value in events["limitations"]), "saved account event limitations required")
    scope = request["scope"]
    calendar = set(scope["calendar"])
    anchor, end = scope["anchor_session"], scope["end_session"]
    seen = set()
    for action in events["cash_dividends"]:
        validate_cash_action(action, scope["execution_universe"], refs)
        require(action["event_id"] not in seen, "duplicate saved cash action")
        seen.add(action["event_id"])
        require(action["record_session"] in calendar or action["record_session"] < anchor,
                "saved cash record session absent from calendar")
        for name in ("ex_session", "pay_session"):
            require(action[name] is None or action[name] > end or action[name] < anchor or
                    action[name] in calendar, "saved cash phase calendar mismatch")
    return events


def _part_descriptor(ref, index, previous, calendar, sequence):
    fields(ref, "artifact part_index start_session end_session first_committed_sequence last_committed_sequence previous_part_digest row_counts")
    _artifact(ref["artifact"])
    require(ref["artifact"]["contract_version"] == RESULT_PART_VERSION and
            ref["part_index"] == index and type(ref["part_index"]) is int and
            ref["previous_part_digest"] == previous, "saved part order/hash chain mismatch")
    for name in ("start_session", "end_session"):
        session(ref[name])
        require(ref[name] in calendar, "saved part session outside run calendar")
    require(ref["start_session"] <= ref["end_session"], "invalid saved part range")
    first, last = ref["first_committed_sequence"], ref["last_committed_sequence"]
    integer(first); integer(last)
    require(first <= last <= sequence, "saved part sequence outside run watermark")
    fields(ref["row_counts"], " ".join(RESULT_KINDS))
    for count in ref["row_counts"].values():
        integer(count)


def _part_rows(part, ref, run_id):
    fields(part, "contract_version run_id part_index rows content_digest")
    require(part["contract_version"] == RESULT_PART_VERSION and part["run_id"] == run_id and
            type(part["part_index"]) is int and part["part_index"] == ref["part_index"],
            "saved result part/run binding mismatch")
    require(part["content_digest"] == _hash_wire({key: value for key, value in part.items()
                                                if key != "content_digest"}),
            "saved result part content digest mismatch")
    fields(part["rows"], " ".join(RESULT_KINDS))
    for kind, rows in part["rows"].items():
        require(type(rows) is list and len(rows) == ref["row_counts"][kind], "saved part row count mismatch")
        for row in rows:
            require(type(row) is dict, "saved result row must be object")
            day = row.get("session", row.get("trade_session"))
            require(type(day) is str and ref["start_session"] <= day <= ref["end_session"],
                    "saved result row outside part range")
            watermark = row.get("sequence", row.get("committed_sequence"))
            if watermark is not None:
                integer(watermark)
                require(watermark <= ref["last_committed_sequence"], "saved result row exceeds part watermark")
    phases = part["rows"]["session_phases"]
    require(bool(phases), "saved result part lacks committed phase")
    for marker in phases:
        fields(marker, "session phase committed_sequence")
        require(marker["phase"] in PHASES and
                ref["first_committed_sequence"] <= marker["committed_sequence"] <=
                ref["last_committed_sequence"], "saved phase differs from part watermark")
    require(min(p["session"] for p in phases) == ref["start_session"] and
            max(p["session"] for p in phases) == ref["end_session"] and
            min(p["committed_sequence"] for p in phases) == ref["first_committed_sequence"] and
            max(p["committed_sequence"] for p in phases) == ref["last_committed_sequence"],
            "saved phase/range binding mismatch")
    return part["rows"]


def _saved_business(wire, rows, profile):
    """Saved arithmetic/link reconciliation, not an account/planner replay."""
    from ..core.stock_rules import rule_at, validate_execution_rules
    from .backtest import _fee_components
    from .stock_rules import fee_at
    request = wire["request_manifest"]
    calendar = request["scope"]["calendar"]
    day_index = {day: index for index, day in enumerate(calendar)}
    rules_ref = request["profile_input"]["stock_execution_rules_ref"]
    fee_ref = request["profile_input"]["stock_fee_schedule_ref"]
    rules = validate_execution_rules(profile["stock_execution_rules"])
    frames = {frame["signal_run_ref"]: frame for frame in request["prediction_input"]["frames"]}
    require(len(frames) == len(request["prediction_input"]["frames"]), "duplicate saved Signal reference")
    intents = {}
    decision_days = set()
    execution_universe = set(request["scope"]["execution_universe"])
    for decision in rows["decisions"]:
        day = decision.get("trade_session")
        require(day in day_index and day_index[day] > 0 and day not in decision_days and
                decision.get("feature_session") == calendar[day_index[day] - 1] and
                decision.get("contract_version") == wire["core_version"] and
                decision.get("top_k") == request["portfolio_policy"]["top_k"] and
                type(decision.get("top_k")) is int and
                decision.get("stock_execution_rules_ref") == rules_ref,
                "saved decision/version/session/rule binding mismatch")
        decision_days.add(day)
        frame = frames.get(decision.get("signal_ref"))
        clock = decision.get("prediction_clock")
        require(frame is not None and type(clock) is dict,'Saved decision/Signal metadata required')
        if frame.get('kind')=='derived':
            require(clock.get('contract_version')=='stock_signal_clock_v2' and clock.get('kind')=='derived' and
                    clock.get('parent_signal_refs')=={a:p['signal_run_ref'] for a,p in frame['parent_inputs'].items()} and
                    all(clock.get(k)==frame[k] for k in ('signal_plan_ref','score_ref','implementation_ref','signal_stage')) and
                    'model_ref' not in clock and 'fold_spec_ref' not in clock,'Saved Derived parent metadata mismatch')
            for name in ('signal_plan_ref','score_ref','implementation_ref'): digest(clock.get(name))
        else:
            require(clock.get('model_ref')==frame['model_ref'] and clock.get('fold_spec_ref')==frame['fold_spec_ref'],
                    'Saved decision/Signal metadata mismatch')
            if clock.get('contract_version')=='stock_signal_clock_v2':
                require(clock.get('kind')=='raw' and clock.get('signal_stage')=='raw_prediction','Saved raw Signal stage mismatch')
                digest(clock.get('label_spec_ref'))
                require(clock['label_spec_ref']==wire['source_audit'].get('prediction_targets',{}).get(decision['signal_ref']),
                        'Saved raw target differs from admitted LabelSpec')
        require(type(decision.get("intents")) is list, "saved decision intents required")
        for intent in decision["intents"]:
            key = intent.get("intent_id")
            require(type(key) is str and key and key not in intents, "duplicate/invalid saved intent")
            integer(intent.get("quantity"), 1)
            require(intent.get("security_id") in execution_universe and intent.get("side") in ("BUY", "SELL") and
                    intent.get("valid_until") == day, "saved intent security/side/validity mismatch")
            integer(intent.get("expected_account_version"))
            intents[key] = (day, intent)
    orders = {}
    seen_intents = set()
    for index, order in enumerate(rows["orders"]):
        key = order.get("order_id")
        require(key == wire["run_id"] + ":order:" + str(index) and key not in orders and
                order.get("stock_execution_rules_ref") == rules_ref, "saved order ID/rule mismatch")
        linked = intents.get(order.get("intent_id"))
        require(linked is not None and order["intent_id"] not in seen_intents and
                order.get("session") == linked[0] and order.get("requested_quantity") == linked[1]["quantity"] and
                all(order.get(name) == linked[1].get(name) for name in
                    ("security_id", "side", "expected_account_version", "valid_until")),
                "saved order differs from linked intent")
        seen_intents.add(order["intent_id"])
        for name in ("requested_quantity", "submitted_quantity", "unsubmitted_quantity", "filled_quantity",
                     "unfilled_quantity", "quantity"):
            integer(order.get(name))
        require(order["requested_quantity"] == order["submitted_quantity"] + order["unsubmitted_quantity"] and
                order["quantity"] == order["submitted_quantity"] == order["filled_quantity"] + order["unfilled_quantity"],
                "saved order submission quantities do not reconcile")
        if order["submitted_quantity"]:
            rule = rule_at(profile["stock_execution_rules"], order["security_id"], order["session"], validated=rules)
            require(order.get("quantity_rule_effective_from") == rule["effective_from"] and
                    order["submitted_quantity"] <= rule["daily_proxy_maximum"], "saved order quantity provenance mismatch")
            if order["side"] == "BUY":
                require(order["submitted_quantity"] >= rule["buy_minimum"] and
                        (order["submitted_quantity"] - rule["buy_minimum"]) % rule["buy_increment"] == 0,
                        "illegal saved buy quantity")
        orders[key] = order
    require(seen_intents == set(intents), "saved intent lacks order")
    fills = {}
    for fill in rows["fills"]:
        key = fill.get("fill_id")
        order = orders.get(fill.get("order_id"))
        require(order is not None and key == fill["order_id"] + ":fill:0" and key not in fills and
                fill.get("session") == order["session"] and fill.get("security_id") == order["security_id"] and
                fill.get("side") == order["side"] and type(fill.get("quantity")) is int and
                0 < fill["quantity"] == order["filled_quantity"], "saved fill/order linkage mismatch")
        require(order.get("committed_sequence") == fill.get("sequence"),
                "saved order/fill watermark mismatch")
        interval = fee_at(profile["stock_fee_schedule"], fill["session"])
        rule = rule_at(profile["stock_execution_rules"], fill["security_id"], fill["session"], validated=rules)
        require(fill.get("stock_execution_rules_ref") == rules_ref and
                fill.get("quantity_rule_effective_from") == rule["effective_from"] and
                fill.get("stock_fee_schedule_ref") == fee_ref and
                fill.get("fee_interval_effective_from") == interval["effective_from"], "saved fill fee/rule provenance mismatch")
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            price = decimal(fill["price"], minimum=0)
            require(price > 0, "saved fill price must be positive")
            gross, commission, stamp, transfer = _fee_components(price, fill["quantity"],
                                                                fill["side"], profile, fill["session"])
        require(tuple(fill.get(name) for name in ("gross_minor", "commission_minor", "tax_minor", "stamp_tax_minor",
                                                "transfer_fee_minor", "fee_minor")) ==
                (gross, commission, stamp, stamp, transfer, commission + stamp + transfer),
                "saved actual-fill fee mismatch")
        expected_cash = -gross - commission - stamp - transfer if fill["side"] == "BUY" else gross - commission - stamp - transfer
        require(fill.get("cash_delta_minor") == expected_cash, "saved fill cash/fee mismatch")
        integer(fill.get("sequence"), 1)
        fills[key] = fill
    require(all(bool(order["filled_quantity"]) == (key + ":fill:0" in fills) for key, order in orders.items()),
            "saved order lacks actual fill")
    for kind in ("fills", "cash_ledger", "position_ledger"):
        previous = 0
        for row in rows[kind]:
            integer(row.get("sequence"), 1)
            require(previous < row["sequence"] <= wire["committed_sequence"], "saved ledger sequence mismatch")
            previous = row["sequence"]
    _saved_balances(wire, rows, fills)
    _saved_cash_actions(wire, rows)
    metrics = wire["metrics"]
    expected = dict(total_fees_minor=sum(fill["fee_minor"] for fill in fills.values()),
                    turnover_minor=sum(fill["gross_minor"] for fill in fills.values()), fill_count=len(fills),
                    unfilled_order_count=sum(order["unfilled_quantity"] > 0 for order in orders.values()),
                    unsubmitted_order_count=sum(order["unsubmitted_quantity"] > 0 for order in orders.values()),
                    unsubmitted_quantity=sum(order["unsubmitted_quantity"] for order in orders.values()),
                    incomplete_order_count=sum(order["unsubmitted_quantity"] + order["unfilled_quantity"] > 0
                                               for order in orders.values()))
    require(all(type(metrics.get(key)) is int and metrics[key] == value for key, value in expected.items()),
            "saved metrics differ from business rows")


def _saved_cash_actions(wire, rows):
    """Check record ownership and saved EX/PAY without inferring missing facts."""
    actions = wire["account_events"]["cash_dividends"]
    action_map = {action["event_id"]: action for action in actions}
    cash_events = {}
    for row in rows["cash_ledger"]:
        if row["reason"] not in ("DIVIDEND_EX", "DIVIDEND_PAY"):
            continue
        key = (row["reason"], row["source_event_id"])
        require(row["source_event_id"] in action_map and key not in cash_events,
                "saved cash event lacks unique account action")
        cash_events[key] = row
    scope = wire["request_manifest"]["scope"]
    nav = {point["session"]: point for point in rows["nav"]}
    snapshots = {(point["session"], point["security_id"]): point for point in rows["positions"]}
    quantities, cursor = {}, 0
    expected = set()
    # Fills and EOD positions establish record ownership. Cash/receivables never
    # supply an omitted action, record date, rate or payment date.
    for action in sorted(actions, key=lambda item: (item["record_session"], item["event_id"])):
        record = action["record_session"]
        while cursor < len(rows["fills"]) and rows["fills"][cursor]["session"] <= record:
            fill = rows["fills"][cursor]
            security = fill["security_id"]
            quantities[security] = quantities.get(security, 0) + fill["quantity"] * (1 if fill["side"] == "BUY" else -1)
            require(quantities[security] >= 0, "saved record ownership has negative quantity")
            cursor += 1
        registered = record in nav
        if not registered:
            continue
        quantity = quantities.get(action["security_id"], 0)
        snapshot = snapshots.get((record, action["security_id"]))
        require((0 if snapshot is None else snapshot.get("quantity")) == quantity,
                "saved record entitlement disagrees with EOD position/fills")
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            amount = minor(decimal(action["cash_before_tax_per_share"], minimum=0) * quantity * 100)
        ex_key = ("DIVIDEND_EX", action["event_id"])
        pay_key = ("DIVIDEND_PAY", action["event_id"])
        ex = cash_events.get(ex_key)
        if action["ex_session"] in nav:
            require(ex is not None and ex["session"] == action["ex_session"] and
                    ex["cash_delta_minor"] == 0 and ex["receivable_delta_minor"] == amount and
                    ex["sequence"] > nav[record]["committed_sequence"] and
                    ex["sequence"] < nav[action["ex_session"]]["committed_sequence"],
                    "saved owned dividend lacks matching EX recognition")
            expected.add(ex_key)
            if action["pay_session"] is not None and action["pay_session"] in nav:
                pay = cash_events.get(pay_key)
                require(pay is not None and pay["session"] == action["pay_session"] and
                        pay["cash_delta_minor"] == amount and pay["receivable_delta_minor"] == -amount and
                        ex["sequence"] < pay["sequence"] < nav[action["pay_session"]]["committed_sequence"],
                        "saved dividend PAY does not match entitlement/EX")
                expected.add(pay_key)
    require(set(cash_events) == expected, "saved EX/PAY lies outside registered account action phases")


def _saved_balances(wire, rows, fills):
    calendar = wire["request_manifest"]["scope"]["calendar"]
    start, end = (wire["request_manifest"]["scope"][key] for key in ("start_session", "end_session"))
    nav_days = [day for day in calendar if start <= day <= end]
    if wire["status"] == "COMPLETE":
        require([point["session"] for point in rows["nav"]] == nav_days, "saved NAV coverage mismatch")
    else:
        require([point["session"] for point in rows["nav"]] == nav_days[:len(rows["nav"])],
                "blocked NAV is not a calendar prefix")
        stopped_day = wire["stopped"].get("session")
        require(len(rows["nav"]) < len(nav_days) and stopped_day == nav_days[len(rows["nav"])],
                "blocked stopped session is not the first missing NAV")
    phases = {}
    previous_day, previous_sequence = None, 0
    for marker in rows["session_phases"]:
        day, sequence = marker["session"], marker["committed_sequence"]
        require(previous_day is None or day >= previous_day, "saved phase session goes backwards")
        require(sequence >= previous_sequence, "saved phase sequence goes backwards")
        require(day not in phases or phases[day] == marker, "conflicting saved session phase")
        phases[day] = marker
        previous_day, previous_sequence = day, sequence
    require(previous_sequence == wire["committed_sequence"], "saved final phase watermark mismatch")
    expected_days = [point["session"] for point in rows["nav"]]
    if wire["status"] == "BLOCKED":
        expected_days.append(stopped_day)
        require(phases[stopped_day]["phase"] == "STOPPED_BEFORE_NAV", "blocked run lacks stopped phase")
    require(set(phases) == set(expected_days), "saved phase coverage differs from output sessions")
    final = wire["final_account"]
    fields(final, "cash_minor receivable_minor positions committed_sequence")
    require(final["committed_sequence"] == wire["committed_sequence"], "saved final account watermark mismatch")
    cash, receivable, cursor = wire["initial_nav_minor"], 0, 0
    fill_cash = set()
    points = {point["session"]: point for point in rows["nav"]}
    previous_nav_sequence = 0
    for point in rows["nav"]:
        sequence = point["committed_sequence"]
        integer(sequence, 1)
        require(sequence > previous_nav_sequence, "saved NAV sequence must increase")
        previous_nav_sequence = sequence
        marker = phases.get(point["session"])
        require(marker is not None and marker["phase"] == "SESSION_COMMITTED" and
                marker["committed_sequence"] == sequence, "saved NAV phase/watermark mismatch")
        while cursor < len(rows["cash_ledger"]) and rows["cash_ledger"][cursor]["sequence"] <= sequence:
            cash, receivable = _cash_row(rows["cash_ledger"][cursor], cash, receivable, fills, fill_cash)
            cursor += 1
        require(point["cash_minor"] == cash and point["receivable_minor"] == receivable and
                point["nav_minor"] == cash + receivable + point["market_value_minor"], "saved NAV cash/value mismatch")
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            require(decimal(point["nav_index"]) == Decimal(point["nav_minor"]) / wire["initial_nav_minor"],
                    "saved NAV index mismatch")
    while cursor < len(rows["cash_ledger"]):
        cash, receivable = _cash_row(rows["cash_ledger"][cursor], cash, receivable, fills, fill_cash)
        cursor += 1
    require(fill_cash == set(fills) and final["cash_minor"] == cash and final["receivable_minor"] == receivable,
            "saved final cash ledger mismatch")
    positions = {}
    fill_positions = set()
    for row in rows["position_ledger"]:
        position = positions.setdefault(row["security_id"], {"quantity": 0, "sellable_quantity": 0, "cost_minor": 0})
        for name, delta in (("quantity", "quantity_delta"), ("sellable_quantity", "sellable_delta"),
                            ("cost_minor", "cost_delta_minor")):
            require(type(row.get(delta)) is int, "invalid saved position delta")
            position[name] += row[delta]
        if row["reason"] == "FILL":
            key = row["source_event_id"]
            fill = fills.get(key)
            require(fill is not None and key not in fill_positions and row["sequence"] == fill["sequence"] and
                    row["session"] == fill["session"] and row["security_id"] == fill["security_id"] and
                    row["quantity_delta"] == fill["quantity"] * (1 if fill["side"] == "BUY" else -1),
                    "saved position ledger/fill mismatch")
            fill_positions.add(key)
        else:
            require(row["reason"] == "SETTLEMENT", "unsupported saved stock position ledger event")
        require(position["quantity"] >= position["sellable_quantity"] >= 0 and position["cost_minor"] >= 0,
                "saved position balance invalid")
    require(fill_positions == set(fills) and final["positions"] == positions, "saved final position ledger mismatch")
    saved_positions = {}
    values_by_session = {}
    for point in rows["positions"]:
        nav = points.get(point["session"])
        key = (point["session"], point["security_id"])
        require(nav is not None and key not in saved_positions and point["committed_sequence"] == nav["committed_sequence"],
                "saved position/NAV watermark mismatch")
        saved_positions[key] = point
        integer(point.get("market_value_minor"))
        values_by_session[point["session"]] = values_by_session.get(point["session"], 0) + point["market_value_minor"]
    for point in rows["nav"]:
        require(point["market_value_minor"] == values_by_session.get(point["session"], 0),
                "saved position/NAV value mismatch")
    if wire["status"] == "COMPLETE":
        require(rows["nav"] and rows["nav"][-1]["committed_sequence"] == wire["committed_sequence"],
                "saved final NAV watermark mismatch")


def _cash_row(row, cash, receivable, fills, fill_cash):
    for name in ("cash_delta_minor", "receivable_delta_minor", "balance_minor"):
        require(type(row.get(name)) is int, "invalid saved cash ledger amount")
    cash += row["cash_delta_minor"]
    receivable += row["receivable_delta_minor"]
    require(cash >= 0 and receivable >= 0 and row["balance_minor"] == cash, "saved cash balance mismatch")
    if row["reason"] == "FILL":
        key = row["source_event_id"]
        fill = fills.get(key)
        require(fill is not None and key not in fill_cash and row["sequence"] == fill["sequence"] and
                row["session"] == fill["session"] and row["cash_delta_minor"] == fill["cash_delta_minor"] and
                row["receivable_delta_minor"] == 0, "saved cash ledger/fill mismatch")
        fill_cash.add(key)
    else:
        require(row["reason"] in ("DIVIDEND_EX", "DIVIDEND_PAY"), "unsupported saved stock cash event")
    return cash, receivable


def _load_stock_backtest_projection(path, *, artifact_reader, limits):
    from .stock_stream_contracts import validate_limits
    validate_limits(limits)
    require(callable(artifact_reader), "local artifact locator required")
    # The run is itself charged once to the physical result budget, including
    # the public saver's terminal LF. Its known size can reject before parsing.
    run_size = Path(path).stat().st_size
    require(0 < run_size <= limits["max_result_bytes"], "saved projection exceeds total result budget")
    wire, run_bytes, _ = _bounded_object(path, maximum=min(limits["max_block_bytes"], limits["max_result_bytes"]),
                                        read_size=limits["max_read_bytes"])
    request = _run_header(wire)
    profile_artifact = request["profile_input"]["artifact"]
    _artifact(profile_artifact)
    profile_path = Path(artifact_reader(profile_artifact))
    profile_size = profile_path.stat().st_size
    require(0 < profile_size <= limits["max_block_bytes"], "small profile exceeds decoded budget")
    total = run_bytes + profile_size
    require(total <= limits["max_result_bytes"], "saved projection exceeds total result budget")
    parts = []
    previous = None
    previous_day = None
    previous_sequence = 0
    calendar = request["scope"]["calendar"]
    # All physical sizes and descriptor row counts are checked before the first
    # result part is parsed or its business rows are materialized.
    for index, ref in enumerate(wire["result_parts"]):
        _part_descriptor(ref, index, previous, calendar, wire["committed_sequence"])
        require(previous_day is None or ref["start_session"] >= previous_day,
                "saved result part session order mismatch")
        require(ref["first_committed_sequence"] >= previous_sequence, "saved result part watermark order mismatch")
        part_path = Path(artifact_reader(ref["artifact"]))
        size = part_path.stat().st_size
        require(0 < size <= min(limits["max_block_bytes"], limits["max_result_part_bytes"]),
                "saved result part exceeds decoded/part budget")
        total += size
        require(total <= limits["max_result_bytes"], "saved projection exceeds total result budget")
        require(sum(ref["row_counts"].values()) <= size, "impossible saved part row-count declaration")
        parts.append((ref, part_path, size))
        previous = ref["artifact"]["content_digest"]
        previous_day = ref["end_session"]
        previous_sequence = ref["last_committed_sequence"]
    require(previous_sequence == wire["committed_sequence"], "saved result final watermark mismatch")
    profile, observed_size, _ = _bounded_object(profile_path, maximum=limits["max_block_bytes"],
                                               read_size=limits["max_read_bytes"])
    require(observed_size == profile_size and _hash_wire(profile) == profile_artifact["content_digest"] ==
            wire["profile_ref"], "saved profile content/reference mismatch")
    require(profile.get("contract_version") == "stock_daily_open_profile_v2" and
            profile.get("stock_execution_rules_ref") == request["profile_input"]["stock_execution_rules_ref"] ==
            _hash_wire(profile["stock_execution_rules"]) and
            profile.get("stock_fee_schedule_ref") == request["profile_input"]["stock_fee_schedule_ref"] ==
            _hash_wire(profile["stock_fee_schedule"]), "saved profile rule/fee reference mismatch")
    from .stock_rules import validate_fee_schedule
    validate_fee_schedule(profile["stock_fee_schedule"])
    rows = {kind: [] for kind in RESULT_KINDS}
    observed_total = run_bytes + observed_size
    for ref, part_path, expected_size in parts:
        part, size, byte_digest = _bounded_object(part_path,
            maximum=min(limits["max_block_bytes"], limits["max_result_part_bytes"]),
            read_size=limits["max_read_bytes"])
        require(size == expected_size and byte_digest == ref["artifact"]["content_digest"] and
                _hash_wire(part) == ref["artifact"]["content_digest"], "saved result part byte/content mismatch")
        observed_total += size
        require(observed_total <= limits["max_result_bytes"], "saved projection grew beyond total result budget")
        decoded = _part_rows(part, ref, wire["run_id"])
        for kind in RESULT_KINDS:
            rows[kind].extend(decoded[kind])
        del part, decoded
    _saved_business(wire, rows, profile)
    return SavedRunProjection(wire, rows, profile, calendar, _token=_LOADER_TOKEN)


def load_stock_backtest_projection(path, *, artifact_reader, limits):
    try:
        return _load_stock_backtest_projection(path, artifact_reader=artifact_reader, limits=limits)
    except (KeyError, TypeError, RecursionError, OverflowError) as exc:
        raise ContractError("malformed saved stock projection") from exc
