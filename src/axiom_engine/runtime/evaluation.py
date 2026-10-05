"""P10 evaluation of saved account observations, without account replay.

Definitions are frozen in axiom-docs Trade §11.1 and §11.2 (c7d28ff).
Monetary amounts remain CNY fen; ratios are not rounded for presentation.
"""
from calendar import monthrange
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import Document, digest, fields, integer, require, session, timestamp
from ..core.portfolio import decimal
from .backtest import BacktestRequest, BacktestRun
from .episode_evaluation import evaluate_episodes
from .evaluation_inputs import evidence_batch, benchmark_rows, dividend_actions
from .period_evaluation import period_metrics, validate_period_metrics

EVALUATION_VERSION = "axiom.evaluation/1"
LONG_EVALUATION_VERSION = "axiom.evaluation/2"


class BenchmarkSeries(Document):
    """Pinned CSI300 native close observations, including the prior anchor."""


class DividendScope(Document):
    """End-cutoff record-date observations; no supplier completeness claim."""


class EvaluationSpec(Document):
    """The supported daily evaluation definition, including fixed P&L bins."""


class EvaluationReport(Document):
    """Saved metrics bound to one immutable account result and input evidence."""


def daily_evaluation_spec():
    return EvaluationSpec.from_dict({
        "contract_version": "evaluation_spec_v1", "frequency": "daily",
        "benchmark_security_id": "000300.SH", "benchmark_series_kind": "price_index_excluding_dividends",
        "anchor": "previous_session", "drawdown_peak": "initial_nav_included",
        "monthly_partial_policy": "separate_observed_return", "episode_definition": "position_0_nonzero_0",
        "dividend_recognition": "ex_income_record_entitlement_pay_transfer",
        "return_denominator": "cumulative_buy_cost_including_fees", "weighting": "equal_closed_episode",
        "external_cash_flows": "reject", "annualization": "none", "missing_policy": "null_no_fill",
        "pnl_distribution": {"metric": "net_pnl_minor", "unit": "CNY fen",
            "edges_minor": [-100000, -50000, -10000, 0, 10000, 50000, 100000],
            "interval": "left_closed_right_open", "minimum_episodes": 10}})


def long_history_evaluation_spec():
    """Fixed v2 profile; the daily v1 factory and saved results stay unchanged."""
    wire = daily_evaluation_spec().to_dict()
    wire.update(contract_version="evaluation_spec_v2", annualization={
        "method": "geometric_cagr", "day_count": "actual_actual_calendar_year_split",
        "interval": "start_inclusive_end_exclusive", "start_anchor": "previous_session_initial_nav",
        "end_anchor": "last_saved_nav_session", "minimum_year_fraction": "1"})
    return EvaluationSpec.from_dict(wire)


def _verify_run(run):
    require(isinstance(run, BacktestRun), "saved BacktestRun required")
    wire = run.to_dict()
    require(wire.get("contract_version") in ("backtest_run_v1", "backtest_run_v2", "backtest_run_v3") and wire.get("status") == "COMPLETE",
            "complete saved account result required")
    recorded = wire.pop("content_digest", None)
    require(recorded == Document.from_dict(wire).identity, "saved result content digest mismatch")
    require(wire["run_id"] == Document.from_dict({"request": wire["plan"], "core": wire["core_version"],
            "runtime": wire["runtime_version"], "implementation_ref": wire["implementation_ref"]}).identity,
            "saved run identity mismatch")
    wire["content_digest"] = recorded
    if wire["contract_version"] == "backtest_run_v2":
        from .backtest import _validate
        from .unit_splits import validate_saved_applications
        require(wire["runtime_version"] == "axiom.backtest/2" and wire["plan"]["contract_version"] == "backtest_request_v2",
                "saved v2 tuple mismatch")
        _validate(BacktestRequest.from_dict(wire["plan"]))
        validate_saved_applications(wire)
    if wire["contract_version"] == "backtest_run_v3":
        from .stock_inputs import validate_stock_request
        require(wire["runtime_version"] == "axiom.backtest/3" and
                wire["core_version"] in ("axiom.stock_portfolio/1", "axiom.stock_portfolio/2") and
                wire["stopped"] is None, "complete stock tuple required")
        if wire["core_version"] == "axiom.stock_portfolio/1":
            require(type(wire["plan"]["portfolio_policy"]["top_k"]) is int and
                    wire["plan"]["portfolio_policy"]["top_k"] == 5, "legacy stock Core only supports Top5")
        validate_stock_request(wire["plan"])
    return wire


def _provenance(wire):
    require(type(wire["source_refs"]) is list and bool(wire["source_refs"]), "input provenance required")
    for ref in wire["source_refs"]:
        digest(ref)
    require(type(wire["source_evidence"]) is list and type(wire["limitations"]) is list,
            "input evidence/limitations required")
    snapshots = set()
    if "coverage_bundle" in wire:
        from .stock_evidence import native_batches
        batches = native_batches(wire["source_evidence"], wire["coverage_bundle"])
        require({entry["reference"] for entry in wire["source_evidence"]} == set(wire["source_refs"]), "stock provenance closure mismatch")
        for batch in batches:
            require(batch["context"]["query"]["purpose"] == "market_replay", "wrong-purpose evaluation evidence")
            snapshots.add(batch["context"]["snapshot_id"])
        return snapshots
    for evidence in wire["source_evidence"]:
        require(evidence["reference"] in wire["source_refs"], "unbound source evidence")
        if "batch" in evidence:
            require(Document.from_dict(evidence["batch"]).identity == evidence["reference"],
                    "DataBatch evidence digest mismatch")
            context = evidence["batch"]["context"]
        else:
            context = evidence["context"]
        require(context["query"]["purpose"] == "market_replay", "wrong-purpose evaluation evidence")
        snapshots.add(context["snapshot_id"])
    return snapshots


def _validate_benchmark(benchmark, required_sessions, snapshots):
    require(isinstance(benchmark, BenchmarkSeries), "BenchmarkSeries required")
    wire = benchmark.to_dict()
    fields(wire, "contract_version security_id series_kind unit calendar rows source_refs source_evidence limitations")
    require(wire["contract_version"] == "benchmark_series_v1" and wire["security_id"] == "000300.SH" and
            wire["series_kind"] == "price_index_excluding_dividends" and wire["unit"] == "index points",
            "CSI300 native price-index observations required")
    require(wire["calendar"] == required_sessions, "benchmark must include exact prior anchor and NAV sessions")
    observed_snapshots = _provenance(wire)
    require(not snapshots or observed_snapshots == snapshots, "benchmark/account Snapshot mismatch")
    require(type(wire["rows"]) is list and [r["session"] for r in wire["rows"]] == required_sessions,
            "benchmark key coverage mismatch; missing values must be explicit nulls")
    for row in wire["rows"]:
        fields(row, "session close available_at valid missing_reason source_refs")
        session(row["session"])
        require(type(row["valid"]) is bool and row["valid"] == (row["close"] is not None), "inconsistent benchmark validity")
        require(type(row["source_refs"]) is list and bool(row["source_refs"]) and
                set(row["source_refs"]) <= set(wire["source_refs"]), "unbound benchmark observation")
        if row["valid"]:
            require(decimal(row["close"], minimum=0) > 0 and row["missing_reason"] is None,
                    "positive native benchmark close required")
            timestamp(row["available_at"])
            require(row["available_at"] <= row["session"] + "T12:30:00Z", "benchmark observation beyond its cutoff")
        else:
            require(type(row["missing_reason"]) is str and bool(row["missing_reason"]), "missing benchmark needs reason")
    require(wire["rows"] == benchmark_rows(evidence_batch(wire), required_sessions),
            "benchmark projection does not match native DataBatch")
    return wire


def _validate_scope(scope, run, snapshots):
    if scope is None:
        return None
    require(isinstance(scope, DividendScope), "DividendScope required")
    wire = scope.to_dict()
    plan = run["plan"]
    fields(wire, "contract_version start_session end_session knowledge_cutoff universe coverage actions source_refs source_evidence limitations" +
           (" coverage_bundle" if plan["contract_version"] == "backtest_request_v3" and "coverage_bundle" in wire else ""))
    if plan["contract_version"] == "backtest_request_v3":
        from .stock_inputs import validate_cash_action
        require(wire["contract_version"] == "dividend_scope_v2" and wire["coverage"] == "observed_records_only" and
                wire["start_session"] == plan["start_session"] and wire["end_session"] == plan["end_session"] and
                wire["universe"] == plan["execution_universe"] and
                wire["knowledge_cutoff"] == plan["end_session"] + "T12:30:00Z", "stock dividend scope mismatch")
        require(_provenance(wire) == snapshots, "stock dividend Snapshot mismatch")
        expected = [a for a in plan["market_replay"]["cash_dividends"]
                    if plan["start_session"] <= a["record_session"] <= plan["end_session"]]
        require(wire["actions"] == expected, "stock dividend scope differs from frozen account actions")
        expected_evidence = [entry for entry in plan["market_replay"]["source_evidence"]
                             if entry["batch"]["context"]["domain"] == "corporate_actions"]
        require(wire["source_evidence"] == expected_evidence and
                wire["source_refs"] == [entry["reference"] for entry in expected_evidence], "stock dividend evidence mismatch")
        from .stock_evidence import scoped_bundle
        require(wire.get("coverage_bundle", []) == scoped_bundle(expected_evidence, plan["market_replay"].get("coverage_bundle", [])),
                "stock dividend coverage closure mismatch")
        for action in wire["actions"]:
            validate_cash_action(action, wire["universe"], wire["source_refs"])
        return wire
    require(wire["contract_version"] == "dividend_scope_v1" and wire["coverage"] == "observed_records_only" and
            wire["start_session"] == plan["start_session"] and wire["end_session"] == plan["end_session"] and
            set(wire["universe"]) == set(plan["signal_frame"]["universe"]) and
            len(wire["universe"]) == len(plan["signal_frame"]["universe"]), "dividend observation scope mismatch")
    timestamp(wire["knowledge_cutoff"])
    require(wire["knowledge_cutoff"] == plan["end_session"] + "T12:30:00Z", "frozen end knowledge cutoff required")
    observed_snapshots = _provenance(wire)
    require(not snapshots or observed_snapshots == snapshots, "dividend/account Snapshot mismatch")
    seen = set()
    for action in wire["actions"]:
        fields(action, "event_id security_id record_session ex_session pay_session cash_per_unit available_at source_refs")
        digest(action["event_id"])
        require(action["event_id"] not in seen and action["security_id"] in wire["universe"], "duplicate/unknown dividend observation")
        seen.add(action["event_id"])
        for key in ("record_session", "ex_session", "pay_session"):
            session(action[key])
        require(wire["start_session"] <= action["record_session"] <= wire["end_session"] and
                action["record_session"] < action["ex_session"] <= action["pay_session"], "invalid dividend observation dates")
        timestamp(action["available_at"])
        require(action["available_at"] <= wire["knowledge_cutoff"], "later announcement is not known at evaluation end")
        decimal(action["cash_per_unit"], minimum=0)
        require(bool(action["source_refs"]) and set(action["source_refs"]) <= set(wire["source_refs"]), "unbound dividend observation")
    require(wire["actions"] == dividend_actions(evidence_batch(wire), wire["universe"], wire["start_session"], wire["end_session"]),
            "dividend projection does not match native DataBatch")
    return wire


def _account_series(run):
    plan = run["plan"]
    calendar = plan["market_replay"]["calendar"]
    days = [d for d in calendar if plan["start_session"] <= d <= plan["end_session"]]
    require([r["session"] for r in run["nav"]] == days, "saved NAV coverage mismatch")
    integer(run["initial_nav_minor"], 1)
    peak, sequence, series = run["initial_nav_minor"], -1, []
    for point in run["nav"]:
        integer(point["nav_minor"]); integer(point["committed_sequence"])
        require(point["committed_sequence"] > sequence, "saved NAV watermarks must increase")
        sequence = point["committed_sequence"]
        peak = max(peak, point["nav_minor"])
        series.append({"session": point["session"], "nav_minor": point["nav_minor"],
            "nav_index": point["nav_index"], "peak_nav_minor": peak,
            "drawdown": str(Decimal(point["nav_minor"]) / peak - 1), "committed_sequence": sequence})
    require(sequence == run["committed_sequence"], "saved final watermark mismatch")
    watermarks = {r["session"]: r["committed_sequence"] for r in series}
    for point in run["positions"]:
        require(point["session"] in watermarks and point["committed_sequence"] == watermarks[point["session"]],
                "saved position/NAV watermarks mismatch")
    require(all(e["reason"] in ("FILL", "DIVIDEND_EX", "DIVIDEND_PAY") for e in run["cash_ledger"]),
            "external cash flows unsupported by daily evaluation")
    return calendar, days, series


def _monthly(run, calendar, series):
    by_day = {r["session"]: r for r in series}
    anchor = calendar[calendar.index(run["plan"]["start_session"]) - 1]
    observations = {anchor: {"nav_minor": run["initial_nav_minor"]}, **by_day}
    out = []
    for month in sorted({d[:7] for d in by_day}):
        full_days = [d for d in calendar if d.startswith(month)]
        observed = [d for d in by_day if d.startswith(month)]
        year, number = map(int, month.split("-"))
        civil_first, civil_last = month + "-01", month + f"-{monthrange(year, number)[1]:02d}"
        index = calendar.index(observed[0])
        boundary = calendar[index - 1] if index else None
        beginning = observations.get(boundary, {}).get("nav_minor")
        end = by_day[observed[-1]]["nav_minor"]
        complete = (calendar[0] <= civil_first and calendar[-1] >= civil_last and observed == full_days and
                    boundary is not None and boundary[:7] < month)
        value = None if beginning is None or beginning <= 0 else str(Decimal(end) / beginning - 1)
        status = "MISSING" if value is None else ("COMPLETE" if complete else "PARTIAL")
        out.append({"month": month, "status": status, "first_session": observed[0], "last_session": observed[-1],
            "boundary_session": boundary, "start_nav_minor": beginning, "end_nav_minor": end,
            "return": value if status == "COMPLETE" else None, "observed_return": value,
            "reason": None if status == "COMPLETE" else ("MISSING_OR_ZERO_BOUNDARY_NAV" if value is None else "MONTH_BOUNDARY_OR_ACCOUNT_COVERAGE_PARTIAL"),
            "committed_sequence": by_day[observed[-1]]["committed_sequence"]})
    return out


def _benchmark(wire):
    anchor = wire["rows"][0]
    base = None if anchor["close"] is None else decimal(anchor["close"])
    peak, previous, worst, complete = base, base, Decimal(0), base is not None
    series = []
    for row in wire["rows"][1:]:
        close = None if row["close"] is None else decimal(row["close"])
        if close is None:
            complete, peak = False, None
        drawdown = None
        if close is not None and peak is not None:
            peak = max(peak, close)
            drawdown = close / peak - 1
            worst = min(worst, drawdown)
        series.append({"session": row["session"], "close": row["close"],
            "nav_index": None if close is None or base is None else str(close / base),
            "daily_return": None if close is None or previous is None else str(close / previous - 1),
            "drawdown": None if drawdown is None else str(drawdown), "valid": row["valid"],
            "missing_reason": row["missing_reason"], "source_refs": row["source_refs"]})
        previous = close
    last = wire["rows"][-1]["close"]
    return {"security_id": wire["security_id"], "series_kind": wire["series_kind"], "unit": wire["unit"],
        "anchor_session": anchor["session"], "anchor_close": anchor["close"],
        "status": "COMPLETE" if complete else ("PARTIAL" if any(r["valid"] for r in series) else "MISSING"),
        "series": series, "total_return": None if base is None or last is None else str(decimal(last) / base - 1),
        "max_drawdown": str(worst) if complete else None}


def _distribution(episodes, rule):
    values = [e["net_pnl_minor"] for e in episodes if e["statistics_eligible"]]
    enough = len(values) >= rule["minimum_episodes"]
    edges = [None, *rule["edges_minor"], None]
    bins = []
    if enough:
        for low, high in zip(edges, edges[1:]):
            bins.append({"lower_minor": low, "upper_minor": high,
                "count": sum((low is None or value >= low) and (high is None or value < high) for value in values)})
    return {"status": "AVAILABLE" if enough else "INSUFFICIENT_SAMPLE", "metric": rule["metric"],
        "unit": rule["unit"], "included_episode_count": len(values),
        "minimum_episodes": rule["minimum_episodes"], "bins": bins}


def _identity(wire):
    return Document.from_dict({key: wire[key] for key in ("input_run_ref", "spec_ref", "benchmark_ref",
        "dividend_scope_ref", "evaluation_version", "implementation_ref")}).identity


def evaluate_backtest(run, *, benchmark, spec, dividend_scope=None):
    """Explicit P10 calculation from saved observations; no feed, broker or Core."""
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        return _evaluate(run, benchmark, spec, dividend_scope)


def _evaluate(run, benchmark, spec, dividend_scope):
    saved = _verify_run(run)
    require(isinstance(spec, EvaluationSpec), "EvaluationSpec required")
    long_period = spec.payload == long_history_evaluation_spec().payload
    require(long_period or spec.payload == daily_evaluation_spec().payload, "unsupported EvaluationSpec")
    if long_period:
        calendar = saved["plan"]["market_replay"]["calendar"]
        require(bool(saved["nav"]) and calendar == sorted(set(calendar)) and
                saved["nav"][0]["session"] in calendar and calendar.index(saved["nav"][0]["session"]) > 0,
                "saved NAV requires an ordered frozen calendar and strict prior anchor")
    calendar, days, series = _account_series(saved)
    anchor = calendar[calendar.index(days[0]) - 1]
    snapshots = _provenance(saved["plan"]["market_replay"])
    bench_input = _validate_benchmark(benchmark, [anchor, *days], snapshots)
    scope = _validate_scope(dividend_scope, saved, snapshots)
    input_ref = {key: saved[key] for key in ("run_id", "content_digest", "committed_sequence")}
    episodes, metrics = evaluate_episodes(saved, scope, input_ref)
    monthly = _monthly(saved, calendar, series)
    bench = _benchmark(bench_input)
    limitations = [*saved["limitations"], *bench_input["limitations"],
        "CSI300 is a price index excluding dividends; account NAV includes recognized dividends. These are different return bases.",
        "Cumulative buy cost including fees is the episode return denominator; equal episode weighting is not IRR or a time-weighted account return.",
        ("CAGR uses the declared initial-wealth clock and actual calendar-year fractions; simulated annualized returns are not predictions. Input-run metric-scope statements describe that saved run; qualified CAGR is provided only by this v2 evaluation, with all original data, execution and dividend limitations retained. No Sharpe or confidence claim; distribution minimum count is a display threshold."
         if long_period else "No annualization, Sharpe or confidence claim; distribution minimum count is a display threshold.")]
    if scope is None:
        limitations.append("DIVIDEND COVERAGE UNKNOWN: saved ex-date events cannot establish absence of known pending ex-date income; closed metrics describe observed income only.")
    else:
        limitations.extend(scope["limitations"])
    wire = {"contract_version": "evaluation_report_v2" if long_period else "evaluation_report_v1", "input_run_ref": input_ref,
        **{key: saved[key] for key in ("signal_ref", "market_ref", "profile_ref")},
        "spec_ref": spec.identity, "spec": spec.to_dict(), "benchmark_ref": benchmark.identity,
        "benchmark_input": bench_input, "dividend_scope_ref": None if scope is None else dividend_scope.identity,
        "dividend_scope": scope, "evaluation_version": LONG_EVALUATION_VERSION if long_period else EVALUATION_VERSION, "implementation_ref": IMPLEMENTATION_REF,
        "status": "COMPLETE" if scope is not None and bench["status"] == "COMPLETE" and all(m["status"] == "COMPLETE" for m in monthly) else "PARTIAL",
        "series": series, "monthly_returns": monthly, "episodes": episodes, "episode_metrics": metrics,
        "benchmark": bench, "pnl_distribution": _distribution(episodes, spec.to_dict()["pnl_distribution"]),
        "limitations": list(dict.fromkeys(limitations))}
    if long_period:
        wire["period_metrics"] = period_metrics(saved, series, bench, anchor=anchor)
    wire["evaluation_ref"] = _identity(wire)
    wire["content_digest"] = Document.from_dict(wire).identity
    return EvaluationReport.from_dict(wire)


def _verify_report(report):
    require(isinstance(report, EvaluationReport), "EvaluationReport required")
    wire = report.to_dict()
    require(wire.get("contract_version") in ("evaluation_report_v1", "evaluation_report_v2") and
            wire.get("status") in ("COMPLETE", "PARTIAL"), "unsupported saved evaluation")
    recorded = wire.pop("content_digest", None)
    require(recorded == Document.from_dict(wire).identity, "saved evaluation content digest mismatch")
    long_period = wire["contract_version"] == "evaluation_report_v2"
    expected_spec = long_history_evaluation_spec() if long_period else daily_evaluation_spec()
    require(wire["spec"] == expected_spec.to_dict() and wire["evaluation_version"] ==
            (LONG_EVALUATION_VERSION if long_period else EVALUATION_VERSION), "saved spec/report/evaluation version mismatch")
    require(("period_metrics" in wire) == long_period, "saved period metrics contract mismatch")
    if long_period:
        validate_period_metrics(wire["period_metrics"])
    fields(wire["input_run_ref"], "run_id content_digest committed_sequence")
    for key in ("run_id", "content_digest"):
        digest(wire["input_run_ref"][key])
    integer(wire["input_run_ref"]["committed_sequence"])
    require(wire["spec_ref"] == EvaluationSpec.from_dict(wire["spec"]).identity and
            wire["benchmark_ref"] == BenchmarkSeries.from_dict(wire["benchmark_input"]).identity and
            wire["dividend_scope_ref"] == (None if wire["dividend_scope"] is None else DividendScope.from_dict(wire["dividend_scope"]).identity),
            "saved evaluation input identity mismatch")
    digest(wire["implementation_ref"])
    require(wire["evaluation_ref"] == _identity(wire), "saved evaluation identity mismatch")


def save_backtest_evaluation(report, path):
    """Write one immutable result, refusing a different result at the same path."""
    _verify_report(report)
    path = Path(path)
    payload = report.payload + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as output:
            output.write(payload)
    except FileExistsError:
        require(path.read_text() == payload, "conflicting saved EvaluationReport")


def load_backtest_evaluation(path):
    """Read and verify saved bytes/refs; never calculate business metrics."""
    saved = EvaluationReport(Path(path).read_text())
    _verify_report(saved)
    return saved
