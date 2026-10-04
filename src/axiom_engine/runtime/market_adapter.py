"""Read-only Data facade adapter for the explicit small daily ETF replay.

No raw access, feature generation, revision selection, fetching or publication.
Unknown status is retained; Runtime admits it only under an explicit ETF profile.
"""
from datetime import date, datetime, timedelta, timezone

from ..core.contracts import Document, require
from .backtest import MarketReplay


def _utc(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(instant.tzinfo is not None, "Data availability must be timezone-aware")
    if instant.microsecond:
        instant += timedelta(seconds=1)
    return instant.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _batch(batch, snapshot):
    wire = batch.to_json()
    context = wire["context"]
    require(context["contract_version"] == "data_batch_v1" and context["snapshot_id"] == snapshot and
            context["query"]["purpose"] == "market_replay", "unpinned or wrong-purpose DataBatch")
    return wire


def read_etf_market_replay(data, *, snapshot, universe, first_session, end_session):
    """Map public states/read_market/events results, preserving actual contexts.

    `first_session` is the preceding signal session, not the evaluation start.
    The supported terminal ETF source contains cash dividends, not split facts;
    unexplained factor changes fail admission. No historical completeness claim.
    """
    from axiom_data import EventQuery, QuerySpec  # only this optional adapter needs Data
    require(snapshot not in ("current", "latest", ""), "concrete Snapshot required")
    dates = []
    cursor, last = date.fromisoformat(first_session), date.fromisoformat(end_session)
    while cursor <= last:
        dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    cutoffs = {day: day + "T20:30:00+08:00" for day in dates}
    state_query = QuerySpec("market_daily", ("close",), tuple(universe), tuple(dates),
                            "best_effort_vendor_v1", cutoffs, purpose="market_replay")
    states = _batch(data.states(snapshot=snapshot, query=state_query), snapshot)
    state_meta = {(row["session"], row["security_id"]): row
                  for row in states["field_meta"]["market_state"]["by_key"]}
    state_rows = {(row["session"], row["security_id"]): row for row in states["records"]}
    calendar = []
    for day in dates:
        day_states = [state_rows[day, security]["market_state"] for security in universe]
        if set(day_states) == {"calendar_closed"}:
            continue
        require("calendar_closed" not in day_states, "exchange calendars disagree")
        for security, state in zip(universe, day_states):
            reason = state_meta[day, security]["missing_reason"] or ""
            require(state in ("normal_trading", "suspended", "source_gap", "unknown_status") and
                    not reason.startswith(("calendar_", "identity_")),
                    "unknown calendar, listing or unsupported event scope")
        calendar.append(day)
    require(calendar and calendar[0] == first_session and calendar[-1] == end_session,
            "explicit first/end trading sessions required")
    def read(domain, fields):
        query = QuerySpec(domain, fields, tuple(universe), tuple(calendar),
                          "best_effort_vendor_v1", {d: cutoffs[d] for d in calendar}, purpose="market_replay")
        return _batch(data.read_market(snapshot=snapshot, query=query), snapshot)
    prices = read("market_daily", ("open", "close", "volume_units"))
    limits = read("price_limits", ("up_limit", "down_limit"))
    factors = read("adjustment_factors", ("factor",))
    actions = _batch(data.events(snapshot=snapshot, query=EventQuery(
        "corporate_actions", ("cash_dividend_per_unit", "record_date", "pay_date", "ex_date"),
        tuple(universe), first_session, end_session, cutoffs[end_session],
        "best_effort_vendor_v1", "ex_date", purpose="market_replay")), snapshot)
    # All four value domains must retain their declared native unit.
    for batch, name, unit in ((prices, "open", "CNY/fund unit"), (prices, "close", "CNY/fund unit"),
                              (prices, "volume_units", "fund units"), (actions, "cash_dividend_per_unit", "CNY/fund unit")):
        require(batch["field_meta"][name]["unit"] == unit, "unexpected ETF unit")
    limit_rows = {(r["session"], r["security_id"]): r for r in limits["records"]}
    close_meta = {(r["session"], r["security_id"]): r for r in prices["field_meta"]["close"]["by_key"]}
    refs = [Document.from_dict(batch).identity for batch in (states, prices, limits, factors, actions)]
    source_evidence = [{"reference": ref, "context": batch["context"]}
                       for ref, batch in zip(refs, (states, prices, limits, factors, actions))]
    rows = []
    for row in prices["records"]:
        key = row["session"], row["security_id"]
        state, reason = state_rows[key]["market_state"], state_meta[key]["missing_reason"]
        availability = close_meta[key].get("usable_from")
        require(row["close"] is None or availability is not None, "observed close lacks availability")
        # Missing price remains null; timestamp is only the query's missing observation boundary.
        result = {"security_id": row["security_id"], "session": row["session"],
                  "market_state": state, "state_reason": reason,
                  "close_available_at": _utc(availability or cutoffs[row["session"]]), "source_refs": refs[:3]}
        for name in ("open", "close", "volume_units"):
            result[name] = None if row[name] is None else str(row[name])
        for name, source_name in (("limit_up", "up_limit"), ("limit_down", "down_limit")):
            value = limit_rows[key][source_name]
            result[name] = None if value is None else str(value)
        rows.append(result)
    cash_dividends = []
    for action in actions["records"]:
        if action["process_status"] != "实施":
            continue
        require(all(action[name] is not None for name in ("cash_dividend_per_unit", "record_date", "ex_date", "pay_date")),
                "incomplete implemented dividend; cannot silently drop economic event")
        require(action["record_date"] >= first_session, "prior dividend entitlement scope unsupported")
        cash_dividends.append({"event_id": Document.from_dict(action).identity,
            "security_id": action["security_id"], "record_session": action["record_date"],
            "ex_session": action["ex_date"], "pay_session": action["pay_date"],
            "cash_per_unit": str(action["cash_dividend_per_unit"]), "source_refs": [refs[-1]]})
    action_keys = {(a["security_id"], a["ex_session"]) for a in cash_dividends}
    previous_factors = {}
    for row in sorted(factors["records"], key=lambda r: (r["session"], r["security_id"])):
        security, factor = row["security_id"], row["factor"]
        require(factor is not None and factor > 0, "missing factor blocks corporate-action capability audit")
        if security in previous_factors and factor != previous_factors[security]:
            require((security, row["session"]) in action_keys, "unexplained factor change: split/revision unsupported")
        previous_factors[security] = factor
    limitations = sorted({item for batch in (states, prices, limits, factors, actions)
                          for item in batch["context"].get("limitations", [])})
    limitations.append("UNKNOWN security status is preserved; observed price/volume is not normal-trading evidence. Admission belongs to the explicit Runtime profile.")
    limitations.append("Terminal ETF source supplies cash distributions; factor audit is a capability check, not proof of complete split/delisting history.")
    return MarketReplay.from_dict({"contract_version": "market_replay_v1", "price_basis": "unadjusted",
        "calendar": calendar, "universe": list(universe), "rows": rows, "cash_dividends": cash_dividends,
        "source_refs": refs, "source_evidence": source_evidence, "limitations": limitations})
