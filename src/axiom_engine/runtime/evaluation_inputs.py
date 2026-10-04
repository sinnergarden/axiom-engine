"""Exact native DataBatch projections shared by adapters and input validation.

Only keys, units, clocks and stored values are checked here. There is no PIT
revision selection, fetching, account replay or financial metric calculation.
"""
from ..core.contracts import Document, require
from .market_adapter import _utc


def evidence_batch(wire):
    evidence = wire["source_evidence"]
    require(len(evidence) == 1 and "batch" in evidence[0] and len(wire["source_refs"]) == 1,
            "complete native DataBatch proof required")
    batch = evidence[0]["batch"]
    ref = Document.from_dict(batch).identity
    require(evidence[0]["reference"] == ref and wire["source_refs"] == [ref], "unbound native DataBatch")
    return batch


def _context(batch, domain):
    context = batch["context"]
    query = context["query"]
    require(context.get("contract_version") == "data_batch_v1" and context.get("domain") == domain and
            bool(context.get("snapshot_id")) and bool(context.get("reader_version")) and
            query.get("purpose") == "market_replay" and query.get("pit_policy") == "best_effort_vendor_v1",
            "wrong native DataBatch domain/policy/provenance")
    return context, query


def benchmark_rows(batch, days):
    _, query = _context(batch, "benchmark_daily")
    require(query.get("fields") == ["close"] and query.get("symbols") == ["000300.SH"] and
            query.get("sessions") == days and query.get("price_basis") == "unadjusted" and
            query.get("adjustment_anchor") is None and query.get("universe_id") is None and
            query.get("policy_by_session") is None and query.get("cutoff_by_session") ==
            {d: d + "T20:30:00+08:00" for d in days}, "native benchmark query scope mismatch")
    metadata = batch["field_meta"]["close"]
    require(metadata["unit"] == "index points", "native index points required")
    key = lambda r: (r["session"], r["security_id"])
    records = {key(r): r for r in batch["records"]}
    meta = {key(r): r for r in metadata["by_key"]}
    expected = {(d, "000300.SH") for d in days}
    require(len(records) == len(batch["records"]) and set(records) == expected and
            len(meta) == len(metadata["by_key"]) and set(meta) == expected, "native benchmark key coverage mismatch")
    ref = Document.from_dict(batch).identity
    rows = []
    for day in days:
        close, observed = records[day, "000300.SH"]["close"], meta[day, "000300.SH"]
        require(close is None or observed.get("usable_from") is not None, "benchmark value lacks availability")
        rows.append({"session": day, "close": None if close is None else str(close),
            "available_at": None if observed.get("usable_from") is None else _utc(observed["usable_from"]),
            "valid": close is not None, "missing_reason": (observed.get("missing_reason") or "SOURCE_MISSING") if close is None else None,
            "source_refs": [ref]})
    return rows


def dividend_actions(batch, universe, start, end):
    context, query = _context(batch, "corporate_actions")
    names = ["cash_dividend_per_unit", "record_date", "pay_date", "ex_date"]
    require(query.get("fields") == names and query.get("symbols") == universe and
            query.get("start") == start and query.get("end") == end and query.get("time_field") == "record_date" and
            query.get("filters") == {} and _utc(query["cutoff"]) == end + "T12:30:00Z" and
            context.get("logical_key") == ["security_id", "announcement_date", "process_status"],
            "native dividend query scope/cutoff mismatch")
    require(batch["field_meta"]["cash_dividend_per_unit"]["unit"] == "CNY/fund unit", "native cash dividend unit required")
    key = lambda r: (r["security_id"], r["announcement_date"], r["process_status"])
    keys = {key(r) for r in batch["records"]}
    require(len(keys) == len(batch["records"]), "duplicate native dividend record")
    metadata = {}
    for name in names:
        by_key = batch["field_meta"][name]["by_key"]
        metadata[name] = {key(r): r for r in by_key}
        require(len(metadata[name]) == len(by_key) and set(metadata[name]) == keys, "native dividend metadata coverage mismatch")
    ref, actions = Document.from_dict(batch).identity, []
    for row in batch["records"]:
        require(row["security_id"] in universe and row["record_date"] is not None and
                start <= row["record_date"] <= end, "native record outside dividend observation scope")
        if row["process_status"] != "实施":
            continue
        require(all(row[n] is not None and metadata[n][key(row)].get("status") == "value" and
                metadata[n][key(row)].get("missing_reason") is None for n in names), "incomplete known dividend observation")
        usable = [metadata[n][key(row)].get("usable_from") for n in names]
        require(all(usable), "dividend observation lacks availability")
        available = max(_utc(t) for t in usable)
        require(available <= end + "T12:30:00Z", "native dividend is later than knowledge cutoff")
        actions.append({"event_id": Document.from_dict(row).identity, "security_id": row["security_id"],
            "record_session": row["record_date"], "ex_session": row["ex_date"], "pay_session": row["pay_date"],
            "cash_per_unit": str(row["cash_dividend_per_unit"]), "available_at": available, "source_refs": [ref]})
    return sorted(actions, key=lambda a: a["event_id"])
