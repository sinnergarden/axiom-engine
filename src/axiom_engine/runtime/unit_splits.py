"""Bounded ETF unit replacement inputs; issuer facts stay separate from models."""
from math import gcd
import re

from ..core.contracts import Document, digest, fields, integer, require, session, text, timestamp

UNIT_SPLIT_POLICY = "etf_settled_holder_eod_v1"
UNIT_SPLIT_PHASE = "EOD_AFTER_CLOSE_BEFORE_NAV"
EVENT_FIELDS = tuple("security_id event_id event_type announcement_date announcement_precision process_status record_date effective_date effective_phase new_price_basis_session new_price_basis_basis ratio_numerator ratio_denominator quantity_rounding quantity_rounding_scope suspension_start suspension_end suspension_scope resume_session document_refs extraction_version revision_id revision_sequence first_observed_at raw_batch_id source_available_at evidence_ref".split())
EVENT_VALUE_FIELDS = tuple(name for name in EVENT_FIELDS if name not in ("security_id", "event_id"))


def eod(day):
    return day + "T12:30:00Z"


def validate_unit_splits(market, start, end):
    items = market["unit_splits"]
    require(type(items) is list, "explicit unit split scope required")
    evidence = {item["reference"]: item for item in market["source_evidence"]}
    snapshots = {item["context"]["snapshot_id"] for item in market["source_evidence"] if "context" in item}
    require(len(snapshots) == 1, "unit inputs require one fixed market Snapshot")
    seen = set()
    for item in items:
        fields(item, "event available_at source_refs")
        event = item["event"]
        fields(event, " ".join(EVENT_FIELDS))
        text(event["event_id"])
        require(event["event_id"] not in seen, "duplicate unit split event")
        seen.add(event["event_id"])
        require(event["security_id"] in market["universe"] and
                re.fullmatch(r"cn\.etf\.(SSE|SZSE)\.\d{6}\.\d{8}", event["security_id"]), "unit replacement requires canonical ETF")
        require(event["event_type"] == "unit_split" and event["process_status"] in ("planned", "implemented"),
                "unsupported unit replacement event/status")
        require(event["effective_phase"] in ("not_stated", "end_of_day"), "unsupported unit replacement phase")
        for name in ("announcement_date", "record_date", "effective_date", "new_price_basis_session"):
            session(event[name])
        require(event["announcement_precision"] == "day", "unsupported disclosure precision")
        require(event["announcement_date"] <= event["record_date"], "unit plan announced after registration")
        require(start <= event["record_date"] <= event["effective_date"] <= end and
                event["record_date"] in market["calendar"] and event["effective_date"] in market["calendar"] and
                event["new_price_basis_session"] > event["effective_date"], "unsupported unit replacement date scope")
        following = [day for day in market["calendar"] if day > event["effective_date"]]
        require(not following or following[0] == event["new_price_basis_session"], "unsupported intervening old-unit trading sessions")
        text(event["new_price_basis_basis"])
        require(not any(a["security_id"] == event["security_id"] and a["record_session"] == event["effective_date"]
                        for a in market["cash_dividends"]), "cash record and unit EOD collision unsupported")
        integer(event["ratio_numerator"], 1); integer(event["ratio_denominator"], 1)
        require(event["ratio_numerator"] > event["ratio_denominator"], "unit consolidation or unchanged ratio unsupported")
        require(gcd(event["ratio_numerator"], event["ratio_denominator"]) == 1, "unit ratio must be reduced")
        require((event["quantity_rounding"], event["quantity_rounding_scope"]) in
                (("not_stated", None), ("ceiling_to_whole_fund_unit", "registered_holder_units")), "unsupported holder rounding")
        suspension = [event[k] for k in ("suspension_start", "suspension_end", "suspension_scope", "resume_session")]
        if any(v is not None for v in suspension):
            require(all(v is not None for v in suspension) and event["suspension_scope"] == "full_session",
                    "unsupported suspension scope")
            for name in ("suspension_start", "suspension_end", "resume_session"):
                session(event[name])
            require(event["suspension_start"] <= event["suspension_end"] < event["resume_session"], "invalid suspension interval")
        for name in ("document_refs", "extraction_version", "revision_id", "first_observed_at", "raw_batch_id"):
            text(event[name])
        integer(event["revision_sequence"], 1)
        timestamp(item["available_at"])
        require(item["available_at"] <= eod(event["record_date"]), "unit plan unavailable at registration")
        require(type(item["source_refs"]) is list and bool(item["source_refs"]), "unit event proof required")
        for ref in item["source_refs"]:
            digest(ref)
            require(ref in market["source_refs"] and ref in evidence and "batch" in evidence[ref], "unit event lacks saved public batch")
            batch = evidence[ref]["batch"]
            require(Document.from_dict(batch).identity == ref, "unit event batch digest mismatch")
            context, query = batch["context"], batch["context"]["query"]
            require(context["domain"] == "fund_share_conversions" and query["purpose"] == "market_replay" and
                    query["time_field"] == "effective_date" and query["pit_policy"] == "best_effort_vendor_v1" and
                    context["snapshot_id"] in snapshots, "unit event batch scope mismatch")
            from .market_adapter import _utc
            require(_utc(query["cutoff"]) == eod(event["record_date"]), "unit registration query cutoff mismatch")
            require(event in batch["records"], "unit event differs from public selected record")
            usable = []
            for name in EVENT_VALUE_FIELDS:
                meta = [r for r in batch["field_meta"][name]["by_key"] if
                        r["event_id"] == event["event_id"] and r["security_id"] == event["security_id"]]
                require(len(meta) == 1 and meta[0]["revision_id"] == event["revision_id"] and
                        meta[0]["usable_from"] is not None, "unit field revision/availability mismatch")
                if event[name] is not None:
                    require(meta[0]["status"] == "value", "unit economic field missing")
                usable.append(_utc(meta[0]["usable_from"]))
            require(item["available_at"] == max(usable), "unit event availability mismatch")
    return items


def validate_saved_applications(run):
    """Validate saved links/shape without applying a ratio or calculating prices."""
    items = {i["event"]["event_id"]: i for i in run["plan"]["market_replay"]["unit_splits"]}
    require(type(run.get("unit_split_applications")) is list, "saved applications required")
    seen, previous = set(), -1
    ledger = [row for row in run["position_ledger"] if row["reason"] == "UNIT_SPLIT"]
    watermarks = {point["session"]: point["committed_sequence"] for point in run["nav"]}
    require(len(ledger) == len(run["unit_split_applications"]), "unit application/ledger count mismatch")
    for application, row in zip(run["unit_split_applications"], ledger):
        fields(application, "event_id security_id session phase sequence status record_sequence record_quantity before_quantity after_quantity before_sellable_quantity after_sellable_quantity cost_minor rounding_extra_fraction original_quote normalized_quote before_market_value_minor after_market_value_minor rounding_value_minor source_refs")
        event_id = application["event_id"]
        require(event_id in items and event_id not in seen, "duplicate or unbound saved application")
        seen.add(event_id)
        event = items[event_id]["event"]
        require(application["security_id"] == event["security_id"] and application["session"] == event["effective_date"] and
                application["phase"] == UNIT_SPLIT_PHASE and application["status"] in ("APPLIED", "NO_ENTITLEMENT"),
                "saved unit application scope mismatch")
        for name in ("sequence", "record_sequence", "record_quantity", "before_quantity", "after_quantity",
                     "before_sellable_quantity", "after_sellable_quantity", "cost_minor", "before_market_value_minor", "after_market_value_minor"):
            integer(application[name])
        require(type(application["rounding_value_minor"]) is int, "invalid saved rounding value")
        fields(application["rounding_extra_fraction"], "numerator denominator")
        integer(application["rounding_extra_fraction"]["numerator"])
        integer(application["rounding_extra_fraction"]["denominator"], 1)
        require(previous < application["sequence"] < run["committed_sequence"] and
                application["record_sequence"] < application["sequence"] and
                application["session"] in watermarks and application["sequence"] < watermarks[application["session"]],
                "invalid unit application ordering")
        require(application["record_quantity"] == application["before_quantity"] == application["before_sellable_quantity"] and
                application["after_quantity"] == application["after_sellable_quantity"] and
                application["status"] == ("APPLIED" if application["before_quantity"] else "NO_ENTITLEMENT"),
                "saved settled entitlement mismatch")
        previous = application["sequence"]
        require(row == {"sequence": application["sequence"], "session": application["session"],
                "security_id": application["security_id"], "quantity_delta": application["after_quantity"] - application["before_quantity"],
                "sellable_delta": application["after_sellable_quantity"] - application["before_sellable_quantity"],
                "cost_delta_minor": 0, "reason": "UNIT_SPLIT", "source_event_id": event_id}, "saved unit ledger mismatch")
        for name in ("original_quote", "normalized_quote"):
            quote = application[name]
            fields(quote, "price session available_at source_refs")
            text(quote["price"]); session(quote["session"]); timestamp(quote["available_at"])
            require(type(quote["source_refs"]) is list and bool(quote["source_refs"]), "saved quote refs required")
            for ref in quote["source_refs"]:
                digest(ref)
                require(ref in run["plan"]["market_replay"]["source_refs"], "unbound saved unit quote reference")
        require(type(application["source_refs"]) is list and bool(application["source_refs"]), "saved application refs required")
        for ref in application["source_refs"]:
            digest(ref)
            require(ref in run["plan"]["market_replay"]["source_refs"], "unbound saved unit application reference")
    require(seen == set(items), "saved run omitted unit application")
    for position in run["positions"]:
        require("mark_basis_event_id" in position and
                (position["mark_basis_event_id"] is None or position["mark_basis_event_id"] in seen), "unbound mark unit basis")
    for order in run["orders"]:
        require(type(order.get("announced_suspension_event_ids")) is list and
                set(order["announced_suspension_event_ids"]) <= set(items), "unbound suspension evidence")
    for decision in run["decisions"]:
        require(type(decision.get("reference_prices")) is dict, "saved Core reference prices required")


def suspension_events(items, security, day):
    opening = day + "T01:30:00Z"
    return [item["event"]["event_id"] for item in items if item["event"]["security_id"] == security and
            item["available_at"] <= opening and item["event"]["suspension_scope"] == "full_session" and
            item["event"]["suspension_start"] <= day <= item["event"]["suspension_end"]]
