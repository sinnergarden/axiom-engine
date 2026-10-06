"""Map frozen public stock DataBatches; never collect, train or rewrite facts."""
from ..core.contracts import Document, require
from ..core.portfolio import decimal
from ..core.stock_portfolio import instant
from .stock_inputs import TAX_CONVENTION
from .stock_evidence import native_ref, pack_stock_batches, scoped_bundle
from ..core.stock_rules import validate_execution_rules, listed, LIFECYCLE_POLICY

STOCK_EVENT_FIELDS = ("implementation_announcement_date", "record_date", "ex_date",
    "cash_dividend_before_tax_per_share", "bonus_shares_per_share", "capital_transfer_shares_per_share",
    "source_issue", "source_candidate_count", "candidate_economic_dates")
EVENT_KEY = ("security_id", "report_period", "announcement_date", "process_status")


def _native_query(batch, *, domain, names, universe, calendar, time_field=None, states=False):
    context, query = batch["context"], batch["context"]["query"]
    require(context["domain"] == ("market_state_diagnostics" if states else domain) and query["symbols"] == universe and
            query["pit_policy"] == "best_effort_vendor_v1", "stock native query domain/scope/policy mismatch")
    if time_field:
        require(set(query) == {"fields", "symbols", "start", "end", "cutoff", "pit_policy", "time_field", "purpose", "filters"} and
                query["fields"] == list(names) and query["time_field"] == time_field and
                query["start"] == calendar[0] and query["end"] == calendar[-1] and
                instant(query["cutoff"]) == instant(calendar[-1] + "T20:30:00+08:00") and
                query["filters"] == {}, "stock action query closure mismatch")
    else:
        require(set(query) == {"fields", "symbols", "sessions", "cutoff_by_session", "pit_policy", "purpose",
                              "price_basis", "adjustment_anchor", "universe_id", "policy_by_session"} and
                query["price_basis"] == "unadjusted" and query["adjustment_anchor"] is None and
                query["universe_id"] is None and query["policy_by_session"] is None,
                "stock native query basis mismatch")
        allowed = {tuple(names)}
        if domain == "market_daily" and not states:
            allowed.add(("open", "high", "low", "close", "volume_shares", "amount_cny"))
        require(tuple(query["fields"]) in allowed, "stock native query fields mismatch")
        sessions = query["sessions"]
        require(sessions == sorted(set(sessions)) and
                (set(calendar) <= set(sessions) if states else sessions == calendar) and
                set(query["cutoff_by_session"]) == set(sessions), "stock native query calendar mismatch")
        require(all(instant(query["cutoff_by_session"][day]) == instant(day + "T20:30:00+08:00")
                    for day in sessions), "stock native query cutoff mismatch")


def _visible(meta, cutoff):
    # Data native metadata uses missing_reason/usable_from; synthetic readers may additionally carry status.
    return (meta.get("status", "value") == "value" and meta.get("missing_reason") is None and
            meta.get("usable_from") is not None and instant(meta["usable_from"]) <= instant(cutoff))


def _membership_index(batch, *, universe, calendar, identities):
    """Admit original PIT flags; membership is never inferred from prices or listing."""
    context, query = batch["context"], batch["context"]["query"]
    require(context["contract_version"] == "data_batch_v1" and context["domain"] == "universe_membership" and
            query["purpose"] == "decision_facts" and query["pit_policy"] == "best_effort_vendor_v1" and
            query["fields"] == ["is_member"] and query["symbols"] == universe and query["sessions"] == calendar and
            query["universe_id"] == "csi300" and set(query["cutoff_by_session"]) == set(calendar),
            "original PIT stock membership query required")
    require(all(instant(query["cutoff_by_session"][d]) == instant(d + "T20:30:00+08:00") for d in calendar),
            "stock membership must retain original Feature cutoff")
    return _membership_rows(batch, universe=universe, calendar=calendar, identities=identities)


def _membership_rows(batch, *, universe, calendar, identities):
    """Validate admitted original membership row views without rebinding query."""
    rows, meta = {}, {}
    for native in batch["records"]:
        key = native["session"], native["security_id"]
        require(key not in rows and type(native["is_member"]) is bool, "duplicate or missing PIT stock member flag")
        rows[key] = native
    for entry in batch["field_meta"]["is_member"]["by_key"]:
        key = entry["session"], entry["security_id"]
        require(key not in meta, "duplicate PIT stock membership metadata")
        meta[key] = entry
    expected = {(d, s) for d in calendar for s in universe}
    require(set(rows) == set(meta) == expected, "incomplete PIT stock membership grid")
    for (day, security), row in rows.items():
        require(_visible(meta[day, security], day + "T20:30:00+08:00"), "unavailable PIT stock member flag")
        require(not row["is_member"] or listed(identities[security], day), "PIT member outside listing lifecycle")
    return rows


def stock_market_from_batches(*, batches, universe, calendar, execution_rules=None, membership_batch=None):
    """Preserve complete native batches, including nonimplemented uncertainty."""
    from .backtest import MarketReplay
    return MarketReplay.from_dict(_stock_market_wire(batches=batches, universe=universe, calendar=calendar,
        execution_rules=execution_rules, membership_batch=membership_batch))


def _keyed_stock(batch):
    out = {}
    for row in batch["records"]:
        key = row["session"], row["security_id"]
        require(key not in out, "duplicate native stock key")
        out[key] = row
    return out


def _stock_metadata(batch, name, keys=("session", "security_id")):
    result = {}
    for row in batch["field_meta"][name]["by_key"]:
        key = tuple(row[k] for k in keys)
        require(key not in result, "duplicate native stock field metadata")
        result[key] = row
    return result

def _project_stock_rows(*, batches, universe, calendar, identities, source_refs,
                        membership_batch=None, expanded=False):
    """Project admitted original row views with the same v6 visibility rules."""
    states, prices, limits, factors = batches
    full = identities is not None
    p, s, l, f = map(_keyed_stock, (prices, states, limits, factors))
    meta = {name: _stock_metadata(batch, native) for name, batch, native in
        (("open", prices, "open"), ("close", prices, "close"), ("volume_shares", prices, "volume_shares"),
         ("limit_up", limits, "up_limit"), ("limit_down", limits, "down_limit"), ("market_state", states, "market_state"))}
    rows = []
    for day in calendar:
        for security in universe:
            key = day, security
            require(all(key in index for index in (p, s, l, f)), "missing native stock execution-union key")
            require(all(key in index for index in meta.values()), "missing native stock field metadata")
            for name, native, source in (("open", "open", p), ("close", "close", p), ("volume_shares", "volume_shares", p),
                                         ("limit_up", "up_limit", l), ("limit_down", "down_limit", l)):
                require(source[key][native] is None or _visible(meta[name][key], day + "T20:30:00+08:00"),
                        "unavailable observed stock market value")
                if full and source[key][native] is None:
                    require(bool(meta[name][key].get("missing_reason")), "null stock fact requires native missing reason")
            reason = meta["market_state"][key]["missing_reason"]
            allowed = ("normal_trading", "unknown_status", "suspended", "source_gap")
            if full:
                allowed += ("not_listed", "delisted")
                is_listed = listed(identities[security], day)
                require((s[key]["market_state"] not in ("not_listed", "delisted")) == is_listed,
                        "native stock state contradicts canonical listing lifecycle")
                require(is_listed or all(p[key][n] is None for n in ("open", "close", "volume_shares")),
                        "unlisted stock carries observed market facts")
            require(s[key]["market_state"] in allowed and
                    not (reason or "").startswith(("identity_", "calendar_", "listing_")), "stock lifecycle/calendar admission failed")
            availability = {name: index[key].get("usable_from") for name, index in meta.items()}
            row = {"security_id": security, "session": day, "market_state": s[key]["market_state"], "state_reason": reason,
                "source_refs": source_refs[:3], "execution_evidence_cutoff": day + "T20:30:00+08:00",
                "field_available_at": availability, "close_available_at": availability["close"] or day + "T20:30:00+08:00"}
            for name in ("open", "close"):
                row[name] = None if p[key][name] is None else str(p[key][name])
            row["volume_shares"] = p[key]["volume_shares"]
            for name, native in (("limit_up", "up_limit"), ("limit_down", "down_limit")):
                row[name] = None if l[key][native] is None else str(l[key][native])
            rows.append(row)
    if expanded:
        require(full and membership_batch is not None, "expanded stock rows need original membership")
        members = _membership_rows(membership_batch, universe=universe, calendar=calendar, identities=identities)
        factor_meta = _stock_metadata(factors, "factor")
        close_meta = _stock_metadata(prices, "close")
        rows = [{**row, "_stock_listed": listed(identities[row["security_id"]], row["session"]),
            "_stock_factor_valid": f[row["session"], row["security_id"]]["factor"] is not None,
            "_stock_factor_missing_reason": factor_meta[row["session"], row["security_id"]].get("missing_reason"),
            "_stock_factor_source_ref": source_refs[3],
            "_stock_close_missing_reason": close_meta[row["session"], row["security_id"]].get("missing_reason"),
            "_stock_member": members[row["session"], row["security_id"]]["is_member"]} for row in rows]
    return rows


def _project_stock_actions(*, batches, universe, source_refs, max_event_bytes=None):
    """Project original action views; retain uncertainty and source identities."""
    ex_actions, record_actions = batches
    actions, diagnostics, blocks, seen_native = {}, [], [], {}
    event_bytes = 0
    def reserve(value, previous=None):
        nonlocal event_bytes
        if max_event_bytes is not None:
            from .stock_stream_outputs import canonical_size
            from ..core.contracts import integer
            integer(max_event_bytes, 1)
            size = canonical_size(value, max_event_bytes)
            old_size = 0 if previous is None else canonical_size(previous, max_event_bytes)
            require(event_bytes + size - old_size <= max_event_bytes,
                    "stock event projection budget exceeded before addition")
            event_bytes += size - old_size
    def diagnostic_row(value):
        reserve(value)
        diagnostics.append(value)
    def block(security, day, available, reason, ref):
        item = {"security_id": security, "effective_session": day, "available_at": available,
                "reason": reason, "source_refs": [ref]}
        if item not in blocks:
            reserve(item)
            blocks.append(item)
    for batch, ref in zip((ex_actions, record_actions), source_refs[4:6]):
        action_meta = {name: _stock_metadata(batch, name, EVENT_KEY) for name in STOCK_EVENT_FIELDS}
        seen_query = set()
        for native in batch["records"]:
            security = native["security_id"]
            require(security in universe, "stock event outside execution scope")
            business_key = tuple(native[name] for name in EVENT_KEY)
            require(business_key not in seen_query, "duplicate native stock action business key")
            seen_query.add(business_key)
            require(business_key not in seen_native or seen_native[business_key] == native,
                    "conflicting native stock action payload across queries")
            seen_native[business_key] = native
            require(all(business_key in index for index in action_meta.values()), "missing stock action field metadata")
            dates = [native.get(name) for name in ("record_date", "ex_date")]
            available = [index[business_key]["usable_from"] for index in action_meta.values()
                         if index[business_key].get("usable_from")]
            require(bool(available), "stock event visibility proof required")
            available_at = max(available, key=instant)
            require(instant(available_at) <= instant(batch["context"]["query"]["cutoff"]), "future stock action evidence")
            diagnostic = {"native_record": native, "available_at": available_at, "source_refs": [ref], "policy": "observed_implemented_only"}
            if native["process_status"] in ("预案", "股东大会通过"):
                diagnostic["admission"] = "NONIMPLEMENTED_DIAGNOSTIC_ONLY"
                diagnostic_row(diagnostic)
                continue
            if native["process_status"] != "实施":
                diagnostic["admission"] = "UNKNOWN_IMPLEMENTATION_STATUS"
                diagnostic_row(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            if native.get("source_issue"):
                diagnostic["admission"] = "AMBIGUOUS_IMPLEMENTED_ACTION"
                diagnostic_row(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            needed = ("implementation_announcement_date", "record_date", "ex_date", "cash_dividend_before_tax_per_share",
                      "bonus_shares_per_share", "capital_transfer_shares_per_share")
            require(all(_visible(action_meta[name][business_key], batch["context"]["query"]["cutoff"])
                        for name in needed if native.get(name) is not None), "unavailable implemented stock action fact")
            quantity_rates = [native.get(name) for name in ("bonus_shares_per_share", "capital_transfer_shares_per_share")]
            if any(value is None or value != 0 for value in quantity_rates):
                diagnostic["admission"] = "UNSUPPORTED_QUANTITY_ACTION"
                diagnostic_row(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            cash = native.get("cash_dividend_before_tax_per_share")
            if any(day is None for day in dates) or cash is None or cash <= 0:
                diagnostic["admission"] = "MISSING_IMPLEMENTED_CASH_FACT"
                diagnostic_row(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            identity = Document.from_dict(native).identity
            action = {"contract_version": "stock_cash_action_v1", "event_id": identity, "security_id": security,
                "record_session": native["record_date"], "ex_session": native["ex_date"], "pay_session": None,
                "cash_before_tax_per_share": str(cash), "tax_convention": TAX_CONVENTION,
                "available_at": available_at, "source_refs": [ref]}
            if identity in actions:
                revised = {**actions[identity], "source_refs": sorted(set(actions[identity]["source_refs"] + [ref]))}
                reserve(revised, actions[identity])
                actions[identity]["source_refs"] = revised["source_refs"]
            else:
                reserve(action)
                actions[identity] = action
    return sorted(actions.values(), key=lambda a: a["event_id"]), diagnostics, blocks


def _stock_market_wire(*, batches, universe, calendar, saved_evidence=None, coverage_bundle=None,
                      execution_rules=None, membership_batch=None):
    require(len(batches) == 6, "stock states/prices/limits/factors/two action queries required")
    states, prices, limits, factors, ex_actions, record_actions = batches
    snapshot = prices["context"]["snapshot_id"]
    full = execution_rules is not None
    require(full == (membership_batch is not None), "stock rules and original PIT membership must be supplied together")
    if full:
        identities, _ = validate_execution_rules(execution_rules)
        require(execution_rules["universe"] == universe and execution_rules["calendar"] == calendar,
                "stock execution rules scope mismatch")
        require(membership_batch["context"]["snapshot_id"] == snapshot, "stock member Snapshot mismatch")
        _membership_index(membership_batch, universe=universe, calendar=calendar, identities=identities)
    for batch in batches:
        require(batch["context"]["contract_version"] == "data_batch_v1" and
                batch["context"]["snapshot_id"] == snapshot and
                batch["context"]["query"]["purpose"] == "market_replay", "stock batch Snapshot/purpose mismatch")
    for batch, domain, names, state in ((states, "market_daily", ("close",), True),
            (prices, "market_daily", ("open", "close", "volume_shares"), False),
            (limits, "price_limits", ("up_limit", "down_limit"), False),
            (factors, "adjustment_factors", ("factor",), False)):
        _native_query(batch, domain=domain, names=names, universe=universe, calendar=calendar, states=state)
    for batch, time_field in ((ex_actions, "ex_date"), (record_actions, "record_date")):
        _native_query(batch, domain="corporate_actions", names=STOCK_EVENT_FIELDS,
                      universe=universe, calendar=calendar, time_field=time_field)
    for batch, name, unit in ((prices, "open", "CNY/share"), (prices, "close", "CNY/share"),
            (prices, "volume_shares", "shares"), (limits, "up_limit", "CNY/share"),
            (limits, "down_limit", "CNY/share"), (factors, "factor", "dimensionless"),
            (ex_actions, "cash_dividend_before_tax_per_share", "CNY/share"),
            (record_actions, "cash_dividend_before_tax_per_share", "CNY/share")):
        require(batch["field_meta"][name]["unit"] == unit, "unexpected native stock unit")
    all_batches = list(batches) + ([membership_batch] if full else [])
    source_refs = ([native_ref(batch) for batch in all_batches] if saved_evidence is None else
                   [entry["reference"] for entry in saved_evidence])
    rows = _project_stock_rows(batches=batches[:4], universe=universe, calendar=calendar,
                              identities=identities if full else None, source_refs=source_refs)
    cash_actions, diagnostics, blocks = _project_stock_actions(batches=batches[4:6], universe=universe,
                                                            source_refs=source_refs)
    f = _keyed_stock(factors)
    def block(security, day, available, reason, ref):
        item = {"security_id": security, "effective_session": day, "available_at": available,
                "reason": reason, "source_refs": [ref]}
        if item not in blocks:
            blocks.append(item)
    factor_meta = _stock_metadata(factors, "factor")
    for security in universe:
        previous = None
        for day in calendar:
            factor = f[day, security]["factor"]
            if full:
                require((day, security) in factor_meta, "missing native stock factor metadata")
                if factor is None:
                    require(bool(factor_meta[day, security].get("missing_reason")), "null stock factor requires native missing reason")
                    previous = None
                    continue
                require(listed(identities[security], day), "unlisted stock carries observed factor")
            require((day, security) in factor_meta and _visible(factor_meta[day, security], day + "T20:30:00+08:00"),
                    "unavailable stock factor capability evidence")
            require(factor is not None and decimal(str(factor), minimum=0) > 0, "missing stock factor capability evidence")
            # A same-day cash EX cannot prove the magnitude or quantity effect of a factor transition.
            if previous is not None and factor != previous:
                block(security, day, factor_meta[day, security]["usable_from"], "UNEXPLAINED_FACTOR_CHANGE", source_refs[3])
            previous = factor
    unique_refs = list(dict.fromkeys(source_refs))
    if saved_evidence is None:
        evidence, coverage_bundle = pack_stock_batches([all_batches[source_refs.index(ref)] for ref in unique_refs], unique_refs)
    else:
        evidence = saved_evidence
    limitations = sorted({value for batch in all_batches for value in batch["context"].get("limitations", [])})
    limitations += ["OBSERVED IMPLEMENTED ACTIONS ONLY: nonimplemented uncertainty remains diagnostic; no complete action-history claim.",
        "Stock cash source lacks PAY dates; gross-before-tax receivables remain pending until verified native payment.",
        "Native UNKNOWN and actual field availability are retained; daily open/volume/limits are retrospective execution evidence."]
    wire = {"contract_version": "market_replay_v3", "price_basis": "unadjusted",
        "calendar": list(calendar), "universe": list(universe), "rows": rows,
        "cash_dividends": cash_actions,
        "action_diagnostics": diagnostics, "action_blocks": blocks, "source_refs": unique_refs,
        "source_evidence": evidence, "limitations": limitations}
    if coverage_bundle:
        wire["coverage_bundle"] = coverage_bundle
    if full:
        wire.update(contract_version="market_replay_v4", stock_execution_rules_ref=Document.from_dict(execution_rules).identity,
                    membership_ref=native_ref(membership_batch), lifecycle_policy=LIFECYCLE_POLICY)
    return wire


def read_stock_market_replay(data, *, snapshot, universe, calendar):
    """Six bounded read-only public calls at each declared retrospective cutoff."""
    from axiom_data import QuerySpec, EventQuery
    cutoffs = {day: day + "T20:30:00+08:00" for day in calendar}
    require(snapshot not in ("", "current", "latest"), "fixed stock Snapshot required")
    def read(domain, names):
        query = QuerySpec(domain, tuple(names), tuple(universe), tuple(calendar), "best_effort_vendor_v1", cutoffs, purpose="market_replay")
        return data.read_market(snapshot=snapshot, query=query).to_json()
    query = QuerySpec("market_daily", ("close",), tuple(universe), tuple(calendar), "best_effort_vendor_v1", cutoffs, purpose="market_replay")
    batches = [data.states(snapshot=snapshot, query=query).to_json(),
        read("market_daily", ("open", "close", "volume_shares")), read("price_limits", ("up_limit", "down_limit")),
        read("adjustment_factors", ("factor",))]
    for time_field in ("ex_date", "record_date"):
        query = EventQuery("corporate_actions", STOCK_EVENT_FIELDS, tuple(universe), calendar[0], calendar[-1],
                           cutoffs[calendar[-1]], "best_effort_vendor_v1", time_field, purpose="market_replay")
        batches.append(data.events(snapshot=snapshot, query=query).to_json())
    return stock_market_from_batches(batches=batches, universe=universe, calendar=calendar)


def stock_dividend_scope(run):
    """Freeze observed stock record-date evidence; never infer unknown PAY."""
    from .evaluation import DividendScope
    from .stock_stream_projection import SavedRunProjection
    if isinstance(run, SavedRunProjection):
        run.verify()
        wire = run.wire
        plan, events = wire["request_manifest"]["scope"], wire["account_events"]
        return DividendScope.from_dict({"contract_version": "dividend_scope_v3",
            "start_session": plan["start_session"], "end_session": plan["end_session"],
            "knowledge_cutoff": plan["end_session"] + "T12:30:00Z", "universe": plan["execution_universe"],
            "coverage": "observed_records_only", "actions": [a for a in events["cash_dividends"]
                if plan["start_session"] <= a["record_session"] <= plan["end_session"]],
            "source_refs": events["source_refs"], "account_events_ref": wire["account_events_ref"],
            "limitations": events["limitations"]})
    wire = run.to_dict()
    require(wire["contract_version"] in ("backtest_run_v3", "backtest_run_v4", "backtest_run_v6"), "stock saved run required")
    plan, market = wire["plan"], wire["plan"]["market_replay"]
    evidence = [entry for entry in market["source_evidence"] if entry["batch"]["context"]["domain"] == "corporate_actions"]
    scope = {"contract_version": "dividend_scope_v2", "start_session": plan["start_session"],
        "end_session": plan["end_session"], "knowledge_cutoff": plan["end_session"] + "T12:30:00Z",
        "universe": plan["execution_universe"], "coverage": "observed_records_only",
        "actions": [a for a in market["cash_dividends"] if plan["start_session"] <= a["record_session"] <= plan["end_session"]],
        "source_refs": [entry["reference"] for entry in evidence], "source_evidence": evidence,
        "limitations": ["Observed implemented stock cash actions only; absent actions do not establish completeness.",
                        "Unknown PAY remains pending; gross dividends exclude personal holding-period taxes."]}
    bundle = scoped_bundle(evidence, market.get("coverage_bundle", []))
    if bundle:
        scope["coverage_bundle"] = bundle
    return DividendScope.from_dict(scope)
