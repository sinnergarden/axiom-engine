"""Thin optional Data readers for frozen P10 evidence; no source fetching."""
from ..core.contracts import Document, require, session
from .evaluation import BenchmarkSeries, DividendScope
from .evaluation_inputs import benchmark_rows, dividend_actions
from .market_adapter import _batch, _utc


def read_csi300_benchmark(data, *, snapshot, sessions):
    """Retain native closes, explicit missing keys and complete DataBatch proof."""
    from axiom_data import QuerySpec
    require(snapshot not in ("", "current", "latest"), "concrete Snapshot required")
    days = list(sessions)
    require(len(days) >= 2 and days == sorted(set(days)), "ordered anchor/NAV sessions required")
    for day in days:
        session(day)
    query = QuerySpec("benchmark_daily", ("close",), ("000300.SH",), tuple(days),
        "best_effort_vendor_v1", {d: d + "T20:30:00+08:00" for d in days}, purpose="market_replay")
    batch = _batch(data.read_market(snapshot=snapshot, query=query), snapshot)
    ref = Document.from_dict(batch).identity
    return BenchmarkSeries.from_dict({"contract_version": "benchmark_series_v1", "security_id": "000300.SH",
        "series_kind": "price_index_excluding_dividends", "unit": "index points", "calendar": days,
        "rows": benchmark_rows(batch, days), "source_refs": [ref], "source_evidence": [{"reference": ref, "batch": batch}],
        "limitations": [*batch["context"].get("limitations", []),
            "Native CSI300 price-index close excludes dividends; no total-return series is supplied."]})


def read_dividend_scope(data, *, snapshot, universe, start_session, end_session):
    """End-known implemented cash events by record date, including future EX.

    This evidence may classify pending income. It cannot correct old account
    facts or establish supplier completeness/unknown future announcements.
    """
    from axiom_data import EventQuery
    require(snapshot not in ("", "current", "latest"), "concrete Snapshot required")
    session(start_session); session(end_session)
    require(start_session <= end_session and bool(universe), "explicit dividend observation range required")
    cutoff = end_session + "T20:30:00+08:00"
    query = EventQuery("corporate_actions", ("cash_dividend_per_unit", "record_date", "pay_date", "ex_date"),
        tuple(universe), start_session, end_session, cutoff, "best_effort_vendor_v1", "record_date", purpose="market_replay")
    batch = _batch(data.events(snapshot=snapshot, query=query), snapshot)
    ref = Document.from_dict(batch).identity
    return DividendScope.from_dict({"contract_version": "dividend_scope_v1", "start_session": start_session,
        "end_session": end_session, "knowledge_cutoff": _utc(cutoff), "universe": list(universe),
        "coverage": "observed_records_only", "actions": dividend_actions(batch, list(universe), start_session, end_session),
        "source_refs": [ref], "source_evidence": [{"reference": ref, "batch": batch}],
        "limitations": [*batch["context"].get("limitations", []),
            "Record-date scope contains implemented cash events visible at the fixed end cutoff; unknown future announcements are excluded.",
            "OBSERVED RECORDS ONLY: this frozen terminal source does not certify completeness of pending dividend history."]})
