"""Frozen stock quantity rules; pure validation and integer sizing only."""
import re

from .contracts import Document, digest, fields, integer, require, session, text
from .portfolio import decimal

BOARDS = frozenset(("SSE_MAIN", "SZSE_MAIN", "SZSE_CHINEXT", "SSE_STAR"))
CSI300_ELIGIBILITY = "csi300_pit_a_share_v1"
CANDIDATE_POLICY = "member_valid_finite_v1"
MAXIMUM_POLICY = "min_limit_market_maximum_v1"
LIFECYCLE_POLICY = "listing_member_position_v1"
RULE_FIELDS = ("board effective_from effective_to buy_minimum buy_increment sell_minimum sell_increment "
               "full_residual_exit_allowed limit_order_maximum market_order_maximum daily_proxy_maximum "
               "price_tick settlement_sessions source_keys")
RULE_LIMITATIONS = ["Frozen source-backed board classification is derived, not native historical board metadata.",
    "The smaller limit/market maximum is a conservative daily-open research assumption, not a real market order."]


def _period(start, through):
    session(start); session(through)
    require(start <= through, "invalid verified stock interval")


def _sources(sources):
    require(type(sources) is list and bool(sources), "frozen rule sources required")
    keys = set()
    for source in sources:
        fields(source, "source_key url content_sha256 clause")
        text(source["source_key"]); text(source["clause"])
        digest(source["content_sha256"])
        require(type(source["url"]) is str and source["url"].startswith("https://"), "HTTPS rule source required")
        require(source["source_key"] not in keys, "duplicate rule source key")
        keys.add(source["source_key"])
    return keys


def _source_keys(value, keys):
    require(type(value) is list and bool(value) and all(type(v) is str for v in value) and
            len(set(value)) == len(value) and set(value) <= keys,
            "unbound classification or rule source")


def _intervals(rows, keys, *, board=False):
    require(type(rows) is list and bool(rows), "frozen stock intervals required")
    grouped = {}
    for row in rows:
        session(row["effective_from"])
        if row["effective_to"] is not None:
            session(row["effective_to"])
            require(row["effective_from"] < row["effective_to"], "empty stock rule interval")
        _source_keys(row["source_keys"], keys)
        grouped.setdefault(row["board"] if board else None, []).append(row)
    for intervals in grouped.values():
        ordered = sorted(intervals, key=lambda r: r["effective_from"])
        require(intervals == ordered, "ordered stock rule intervals required")
        for left, right in zip(ordered, ordered[1:]):
            require(left["effective_to"] is not None and left["effective_to"] <= right["effective_from"],
                    "overlapping stock rule intervals")
    return grouped


def listed(identity, day):
    return identity["listing_date"] <= day and (identity["delisting_date"] is None or day < identity["delisting_date"])


def interval_at(rows, day, *, label="stock rule"):
    matched = [r for r in rows if r["effective_from"] <= day and
               (r["effective_to"] is None or day < r["effective_to"])]
    require(len(matched) == 1, "missing or overlapping " + label + " interval")
    return matched[0]


def _identity_event_evidence(batches, universe, through):
    """Validate original public retrospective event evidence, not historic receipt."""
    from .stock_portfolio import instant
    masters = [(ref, batch) for ref, batch in batches.items() if batch['context']['domain']=='security_master']
    events = [(ref, batch) for ref, batch in batches.items() if batch['context']['domain']=='listing_events']
    require(len(masters)==len(events)==1, "identity v2 needs one original master and event batch")
    master_ref, master = masters[0]; event_ref, batch = events[0]
    query, original = batch['context']['query'], master['context']['query']
    require(batch['context']['snapshot_id']==master['context']['snapshot_id'] and
            query['symbols']==original['symbols']==universe and query['start']=='1900-01-01' and
            query['end']==through and query['cutoff']==original['cutoff'] and
            original['pit_policy']=='operational_pit_v1' and original['purpose']=='historical_exploration' and
            original['start']=='1900-01-01' and original['end']==through and
            query['pit_policy']=='operational_pit_v1' and query['purpose']=='historical_exploration' and
            query['time_field']=='event_date' and query['filters']=={'event_type':'delisting'} and
            query['fields']==['exchange','listing_date','delisting_date','event_date','event_state'],
            "identity event query/Snapshot/observation closure mismatch")
    selected = {}
    indexed = {}
    for name in query['fields']:
        indexed[name] = {}
        for meta in batch['field_meta'][name]['by_key']:
            key = meta['security_id'], meta['event_type']
            require(key not in indexed[name], "duplicate identity event metadata")
            indexed[name][key] = meta
    for row in batch['records']:
        security = row['security_id']; key = security, row['event_type']
        require(security in universe and security not in selected and row['event_type']=='delisting' and
                row['event_state']=='value' and row['event_date']==row['delisting_date'] and
                query['start']<=row['event_date']<=through, "ambiguous/invalid original delisting event")
        metadata = []
        for name in query['fields']:
            require(key in indexed[name], "missing identity event metadata")
            meta = indexed[name][key]
            require(meta.get('status')=='value' and meta.get('missing_reason') is None and
                    all(type(meta.get(n)) is str and bool(meta[n]) for n in
                        ('revision_id','raw_batch_id','usable_from','availability_basis','first_observed_at')) and
                    meta['availability_basis']=='first_observed_at' and
                    instant(meta['usable_from'])==instant(meta['first_observed_at'])<=instant(query['cutoff']),
                    "unavailable/unbound original identity event")
            metadata.append(meta)
        proof = {n:metadata[0][n] for n in ('revision_id','raw_batch_id','usable_from','availability_basis','first_observed_at')}
        require(all(all(m[n]==value for n,value in proof.items()) for m in metadata),
                "identity event fields select different revisions")
        selected[security] = (row, {'source_ref':event_ref,'event_date':row['event_date'],**proof})
    require(all(set(values)=={(s,'delisting') for s in selected} for values in indexed.values()),
            "identity event metadata grid mismatch")
    return master_ref, event_ref, selected


def validate_execution_rules(wire):
    fields(wire, "contract_version universe calendar identity_input identity_input_ref quantity_rules sources "
                 "verified_from verified_through limitations")
    require(wire["contract_version"] == "stock_execution_rules_v1" and wire["limitations"] == RULE_LIMITATIONS,
            "unsupported frozen stock quantity policy")
    _period(wire["verified_from"], wire["verified_through"])
    universe, calendar = wire["universe"], wire["calendar"]
    require(type(universe) is list and bool(universe) and all(type(s) is str for s in universe) and
            universe == sorted(set(universe)), "sorted stock union required")
    require(type(calendar) is list and bool(calendar) and all(type(d) is str for d in calendar) and
            calendar == sorted(set(calendar)), "ordered stock calendar required")
    for day in calendar:
        session(day)
        require(wire["verified_from"] <= day <= wire["verified_through"], "stock session outside verified rule window")
    sources = _sources(wire["sources"])
    identity = wire["identity_input"]
    fields(identity, "contract_version rows source_refs source_evidence limitations")
    v2 = identity["contract_version"] == "stock_execution_identity_v2"
    require(identity["contract_version"] in ("stock_execution_identity_v1", "stock_execution_identity_v2") and
            wire["identity_input_ref"] == Document.from_dict(identity).identity, "stock identity input mismatch")
    require(type(identity["limitations"]) is list and all(type(x) is str for x in identity["limitations"]),
            "identity provenance limitations required")
    require(type(identity["source_refs"]) is list and bool(identity["source_refs"]) and
            len(set(identity["source_refs"])) == len(identity["source_refs"]), "identity source closure required")
    native = {}; batches = {}
    require(type(identity["source_evidence"]) is list and bool(identity["source_evidence"]), "native identity evidence required")
    for entry in identity["source_evidence"]:
        fields(entry, "reference batch"); digest(entry["reference"])
        batch = entry["batch"]
        require(entry["reference"] not in native and Document.from_dict(batch).identity == entry["reference"],
                "stock native identity hash mismatch")
        require(batch.get("context", {}).get("contract_version") == "data_batch_v1" and
                batch["context"].get("domain") in (("security_master","listing_events") if v2 else ("security_master",)),
                "native security master/event required")
        records = {}
        require(type(batch.get("records")) is list, "native identity records required")
        for row in batch["records"]:
            security = row.get("security_id")
            require(type(security) is str and security not in records, "duplicate native identity")
            records[security] = row
        native[entry["reference"]] = records
        batches[entry["reference"]] = batch
    require(set(native) == set(identity["source_refs"]), "unbound native identity source")
    if v2:
        master_ref, event_ref, events = _identity_event_evidence(batches, universe, wire['verified_through'])
    index = {}
    require(type(identity["rows"]) is list, "stock classification rows required")
    for row in identity["rows"]:
        fields(row, "security_id exchange board instrument_kind listing_date delisting_date classification_source_keys source_refs")
        security = row["security_id"]
        require(type(security) is str, "canonical stock identity required")
        match = re.fullmatch(r"cnstock\.(\d{6})\.(SH|SZ)\.(\d{8})", security)
        require(match is not None and security not in index and row["instrument_kind"] == "A_SHARE" and
                row["board"] in BOARDS, "unknown or conflicting stock classification")
        require(row["exchange"] == ("SSE" if match[2] == "SH" else "SZSE") and
                row["board"].startswith(row["exchange"] + "_"), "canonical stock exchange/board mismatch")
        session(row["listing_date"])
        require(match[3] == row["listing_date"].replace("-", ""), "canonical stock listing identity mismatch")
        if row["delisting_date"] is not None:
            session(row["delisting_date"])
            require(row["listing_date"] < row["delisting_date"], "invalid stock listing interval")
        _source_keys(row["classification_source_keys"], sources)
        require(type(row["source_refs"]) is list and bool(row["source_refs"]) and
                len(set(row["source_refs"])) == len(row["source_refs"]) and set(row["source_refs"]) <= set(native),
                "stock classification lacks native identity binding")
        for ref in ([master_ref] if v2 else row["source_refs"]):
            original = native[ref].get(security)
            require(original is not None and all(original.get(name) == row[name] for name in
                    (("exchange", "listing_date") if v2 else ("exchange", "listing_date", "delisting_date"))),
                    "classification differs from native identity")
        proof = None
        if v2:
            event = events.get(security)
            require(row['source_refs']==[master_ref]+([event_ref] if event else []), "identity row event source closure mismatch")
            if event:
                original_event, proof = event
                require(all(original_event[n]==row[n] for n in ('exchange','listing_date','delisting_date')) and
                        original['delisting_date'] in (None,row['delisting_date']), "identity boundary contradicts original sources")
            else:
                require(row['delisting_date']==original['delisting_date'], "identity boundary lacks original event")
        # Board is an explicit source-backed mapping. Code prefixes do not filter the union;
        # documented exceptional assignments, such as 302132 on ChiNext, remain representable.
        index[security] = ({**row,'_stock_identity_version':2,'_stock_delisting_evidence':proof} if v2 else row)
    require(list(index) == universe, "complete ordered canonical stock identity union required")
    require(type(wire["quantity_rules"]) is list and bool(wire["quantity_rules"]), "frozen stock quantity intervals required")
    for rule in wire["quantity_rules"]:
        fields(rule, RULE_FIELDS)
        require(rule["board"] in BOARDS, "unknown quantity-rule board")
        for name in ("buy_minimum", "buy_increment", "sell_minimum", "sell_increment",
                     "limit_order_maximum", "market_order_maximum", "daily_proxy_maximum", "settlement_sessions"):
            integer(rule[name], 1)
        require(type(rule["full_residual_exit_allowed"]) is bool and rule["settlement_sessions"] == 1 and
                rule["daily_proxy_maximum"] == min(rule["limit_order_maximum"], rule["market_order_maximum"]) and
                rule["daily_proxy_maximum"] >= max(rule["buy_minimum"], rule["sell_minimum"]), "invalid stock quantity policy")
        require(decimal(rule["price_tick"], minimum=0) > 0, "positive stock tick required")
    grouped = _intervals(wire["quantity_rules"], sources, board=True)
    for security in universe:
        for day in calendar:
            if listed(index[security], day):
                interval_at(grouped.get(index[security]["board"], []), day)
    return index, grouped


def stock_execution_rules(*, universe, calendar, identity_input, quantity_rules, sources, verified_from, verified_through):
    wire = Document.from_dict(dict(contract_version="stock_execution_rules_v1", universe=universe, calendar=calendar,
        identity_input=identity_input, identity_input_ref=Document.from_dict(identity_input).identity,
        quantity_rules=quantity_rules, sources=sources, verified_from=verified_from, verified_through=verified_through,
        limitations=RULE_LIMITATIONS)).to_dict()
    validate_execution_rules(wire)
    return wire


def rule_at(wire, security, day, *, validated=None):
    index, grouped = validate_execution_rules(wire) if validated is None else validated
    require(security in index and wire["verified_from"] <= day <= wire["verified_through"] and listed(index[security], day),
            "stock order outside verified listing/rule window")
    return interval_at(grouped.get(index[security]["board"], []), day)


def floor_quantity(quantity, minimum, increment):
    integer(quantity); integer(minimum, 1); integer(increment, 1)
    return 0 if quantity < minimum else minimum + (quantity - minimum) // increment * increment


def legal_quantity(quantity, side, rule, *, held=0, sellable=0, apply_maximum=True):
    integer(quantity); integer(held); integer(sellable)
    require(side in ("BUY", "SELL") and sellable <= held, "invalid stock order side/position")
    cap = min(quantity, rule["daily_proxy_maximum"]) if apply_maximum else quantity
    if side == "SELL":
        cap = min(cap, sellable)
        if (rule["full_residual_exit_allowed"] and quantity == held == sellable and cap == held):
            return cap
    prefix = "buy" if side == "BUY" else "sell"
    return floor_quantity(cap, rule[prefix + "_minimum"], rule[prefix + "_increment"])


def support_ref(universe):
    return Document.from_dict({"eligibility_id": CSI300_ELIGIBILITY, "security_ids": list(universe)}).identity
