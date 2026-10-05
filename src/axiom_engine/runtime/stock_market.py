"""Map frozen public stock DataBatches; never collect, train or rewrite facts."""
from ..core.contracts import Document, require
from ..core.portfolio import decimal
from ..core.stock_portfolio import instant
from .stock_inputs import TAX_CONVENTION
from .stock_evidence import native_ref, pack_stock_batches, scoped_bundle

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


def stock_market_from_batches(*, batches, universe, calendar):
    """Preserve complete native batches, including nonimplemented uncertainty."""
    from .backtest import MarketReplay
    return MarketReplay.from_dict(_stock_market_wire(batches=batches, universe=universe, calendar=calendar))


def _stock_market_wire(*, batches, universe, calendar, saved_evidence=None, coverage_bundle=None):
    require(len(batches) == 6, "stock states/prices/limits/factors/two action queries required")
    states, prices, limits, factors, ex_actions, record_actions = batches
    snapshot = prices["context"]["snapshot_id"]
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
    def keyed(batch):
        out = {}
        for row in batch["records"]:
            key = row["session"], row["security_id"]
            require(key not in out, "duplicate native stock key")
            out[key] = row
        return out
    def metadata(batch, name, keys=("session", "security_id")):
        result = {}
        for row in batch["field_meta"][name]["by_key"]:
            key = tuple(row[k] for k in keys)
            require(key not in result, "duplicate native stock field metadata")
            result[key] = row
        return result
    source_refs = ([native_ref(batch) for batch in batches] if saved_evidence is None else
                   [entry["reference"] for entry in saved_evidence])
    p, s, l, f = map(keyed, (prices, states, limits, factors))
    meta = {name: metadata(batch, native) for name, batch, native in
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
            reason = meta["market_state"][key]["missing_reason"]
            require(s[key]["market_state"] in ("normal_trading", "unknown_status", "suspended", "source_gap") and
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
    actions, diagnostics, blocks, seen_native = {}, [], [], {}
    def block(security, day, available, reason, ref):
        item = {"security_id": security, "effective_session": day, "available_at": available,
                "reason": reason, "source_refs": [ref]}
        if item not in blocks:
            blocks.append(item)
    for batch, ref in zip((ex_actions, record_actions), source_refs[-2:]):
        action_meta = {name: metadata(batch, name, EVENT_KEY) for name in STOCK_EVENT_FIELDS}
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
                diagnostics.append(diagnostic)
                continue
            if native["process_status"] != "实施":
                diagnostic["admission"] = "UNKNOWN_IMPLEMENTATION_STATUS"
                diagnostics.append(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            if native.get("source_issue"):
                diagnostic["admission"] = "AMBIGUOUS_IMPLEMENTED_ACTION"
                diagnostics.append(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            needed = ("implementation_announcement_date", "record_date", "ex_date", "cash_dividend_before_tax_per_share",
                      "bonus_shares_per_share", "capital_transfer_shares_per_share")
            require(all(_visible(action_meta[name][business_key], batch["context"]["query"]["cutoff"])
                        for name in needed if native.get(name) is not None), "unavailable implemented stock action fact")
            quantity_rates = [native.get(name) for name in ("bonus_shares_per_share", "capital_transfer_shares_per_share")]
            if any(value is None or value != 0 for value in quantity_rates):
                diagnostic["admission"] = "UNSUPPORTED_QUANTITY_ACTION"
                diagnostics.append(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            cash = native.get("cash_dividend_before_tax_per_share")
            if any(day is None for day in dates) or cash is None or cash <= 0:
                diagnostic["admission"] = "MISSING_IMPLEMENTED_CASH_FACT"
                diagnostics.append(diagnostic)
                block(security, native.get("ex_date"), available_at, diagnostic["admission"], ref)
                continue
            identity = Document.from_dict(native).identity
            action = {"contract_version": "stock_cash_action_v1", "event_id": identity, "security_id": security,
                "record_session": native["record_date"], "ex_session": native["ex_date"], "pay_session": None,
                "cash_before_tax_per_share": str(cash), "tax_convention": TAX_CONVENTION,
                "available_at": available_at, "source_refs": [ref]}
            if identity in actions:
                actions[identity]["source_refs"] = sorted(set(actions[identity]["source_refs"] + [ref]))
            else:
                actions[identity] = action
    factor_meta = metadata(factors, "factor")
    for security in universe:
        previous = None
        for day in calendar:
            factor = f[day, security]["factor"]
            require((day, security) in factor_meta and _visible(factor_meta[day, security], day + "T20:30:00+08:00"),
                    "unavailable stock factor capability evidence")
            require(factor is not None and decimal(str(factor), minimum=0) > 0, "missing stock factor capability evidence")
            # A same-day cash EX cannot prove the magnitude or quantity effect of a factor transition.
            if previous is not None and factor != previous:
                block(security, day, factor_meta[day, security]["usable_from"], "UNEXPLAINED_FACTOR_CHANGE", source_refs[3])
            previous = factor
    unique_refs = list(dict.fromkeys(source_refs))
    if saved_evidence is None:
        evidence, coverage_bundle = pack_stock_batches([batches[source_refs.index(ref)] for ref in unique_refs], unique_refs)
    else:
        evidence = saved_evidence
    limitations = sorted({value for batch in batches for value in batch["context"].get("limitations", [])})
    limitations += ["OBSERVED IMPLEMENTED ACTIONS ONLY: nonimplemented uncertainty remains diagnostic; no complete action-history claim.",
        "Stock cash source lacks PAY dates; gross-before-tax receivables remain pending until verified native payment.",
        "Native UNKNOWN and actual field availability are retained; daily open/volume/limits are retrospective execution evidence."]
    wire = {"contract_version": "market_replay_v3", "price_basis": "unadjusted",
        "calendar": list(calendar), "universe": list(universe), "rows": rows,
        "cash_dividends": sorted(actions.values(), key=lambda a: a["event_id"]),
        "action_diagnostics": diagnostics, "action_blocks": blocks, "source_refs": unique_refs,
        "source_evidence": evidence, "limitations": limitations}
    if coverage_bundle:
        wire["coverage_bundle"] = coverage_bundle
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
    wire = run.to_dict()
    require(wire["contract_version"] in ("backtest_run_v3", "backtest_run_v4"), "stock saved run required")
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
