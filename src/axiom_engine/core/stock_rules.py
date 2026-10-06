"""Frozen stock quantity rules; pure validation and integer sizing only."""
from copy import deepcopy
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
    require(type(value) is list and bool(value) and len(set(value)) == len(value) and set(value) <= keys,
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


def validate_execution_rules(wire):
    fields(wire, "contract_version universe calendar identity_input identity_input_ref quantity_rules sources "
                 "verified_from verified_through limitations")
    require(wire["contract_version"] == "stock_execution_rules_v1" and wire["limitations"] == RULE_LIMITATIONS,
            "unsupported frozen stock quantity policy")
    _period(wire["verified_from"], wire["verified_through"])
    universe, calendar = wire["universe"], wire["calendar"]
    require(type(universe) is list and bool(universe) and universe == sorted(set(universe)), "sorted stock union required")
    require(type(calendar) is list and bool(calendar) and calendar == sorted(set(calendar)), "ordered stock calendar required")
    for day in calendar:
        session(day)
        require(wire["verified_from"] <= day <= wire["verified_through"], "stock session outside verified rule window")
    sources = _sources(wire["sources"])
    identity = wire["identity_input"]
    fields(identity, "contract_version rows source_refs source_evidence limitations")
    require(identity["contract_version"] == "stock_execution_identity_v1" and
            wire["identity_input_ref"] == Document.from_dict(identity).identity, "stock identity input mismatch")
    require(type(identity["limitations"]) is list and all(type(x) is str for x in identity["limitations"]),
            "identity provenance limitations required")
    require(type(identity["source_refs"]) is list and bool(identity["source_refs"]) and
            len(set(identity["source_refs"])) == len(identity["source_refs"]), "identity source closure required")
    native = {}
    require(type(identity["source_evidence"]) is list and bool(identity["source_evidence"]), "native identity evidence required")
    for entry in identity["source_evidence"]:
        fields(entry, "reference batch"); digest(entry["reference"])
        batch = entry["batch"]
        require(entry["reference"] not in native and Document.from_dict(batch).identity == entry["reference"],
                "stock native identity hash mismatch")
        require(batch.get("context", {}).get("contract_version") == "data_batch_v1" and
                batch["context"].get("domain") == "security_master", "native security master required")
        records = {}
        require(type(batch.get("records")) is list, "native identity records required")
        for row in batch["records"]:
            security = row.get("security_id")
            require(type(security) is str and security not in records, "duplicate native identity")
            records[security] = row
        native[entry["reference"]] = records
    require(set(native) == set(identity["source_refs"]), "unbound native identity source")
    index = {}
    require(type(identity["rows"]) is list, "stock classification rows required")
    for row in identity["rows"]:
        fields(row, "security_id exchange board instrument_kind listing_date delisting_date classification_source_keys source_refs")
        security = row["security_id"]
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
        for ref in row["source_refs"]:
            original = native[ref].get(security)
            require(original is not None and all(original.get(name) == row[name] for name in
                    ("exchange", "listing_date", "delisting_date")), "classification differs from native identity")
        # Board is an explicit source-backed mapping. Code prefixes do not filter the union;
        # documented exceptional assignments, such as 302132 on ChiNext, remain representable.
        index[security] = row
    require(list(index) == universe, "complete ordered canonical stock identity union required")
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
    wire = deepcopy(dict(contract_version="stock_execution_rules_v1", universe=universe, calendar=calendar,
        identity_input=identity_input, identity_input_ref=Document.from_dict(identity_input).identity,
        quantity_rules=quantity_rules, sources=sources, verified_from=verified_from, verified_through=verified_through,
        limitations=RULE_LIMITATIONS))
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
