"""Saved account analysis under Trade §11.3; no account or Data execution."""
from datetime import date
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import Document, digest, fields, integer, require, text
from ..core.portfolio import decimal
from .period_evaluation import _window

ANALYSIS_VERSION = "axiom.evaluation/3"
BENCHMARK_KEYS = ("CSI300", "NASDAQ100", "SSE_COMPOSITE")
PRESERVED = ("input_run_ref", "signal_ref", "market_ref", "profile_ref", "series", "monthly_returns",
             "episodes", "episode_metrics", "pnl_distribution", "benchmark", "period_metrics",
             "benchmark_ref", "benchmark_input", "dividend_scope_ref", "dividend_scope")


def analysis_evaluation_spec(*, risk_free):
    """A required explicit CNY effective annual rate, including a named assumption."""
    from .evaluation import EvaluationSpec, long_history_evaluation_spec
    fields(risk_free, "currency annual_effective_rate source")
    require(risk_free["currency"] == "CNY" and decimal(risk_free["annual_effective_rate"]) > -1,
            "explicit CNY effective annual rate greater than -1 required")
    text(risk_free["source"])
    spec = long_history_evaluation_spec().to_dict()
    spec.update(contract_version="evaluation_spec_v3", risk_free=dict(risk_free),
        rolling_window_sessions=20, return_distribution={"metric":"net_return", "unit":"fraction",
        "edges":[("-" if i < 0 else "") + f"0.{abs(i):02d}" for i in range(-20,21,2)],
        "interval":"left_closed_right_open", "minimum_episodes":10},
        benchmark_keys=list(BENCHMARK_KEYS), benchmark_alignment="exact_native_session_date_no_fill",
        cross_currency_relative_policy="null_without_fx", cross_market_clock="retrospective_local_session")
    return EvaluationSpec.from_dict(spec)


def _sample_std(values):
    mean = sum(values, Decimal(0)) / len(values)
    return mean, (sum(((v-mean)**2 for v in values), Decimal(0)) / (len(values)-1)).sqrt()


def _drawdown(base):
    series = base["series"]
    worst = min(decimal(p["drawdown"]) for p in series)
    out = dict(status="NO_DRAWDOWN", peak_session=None, trough_session=None, peak_nav_minor=None,
        trough_nav_minor=None, drawdown=str(worst), elapsed_calendar_days=None,
        peak_is_initial_anchor=False, recovery_session=None, recovery_status="NOT_APPLICABLE")
    if worst >= 0:
        return out
    i = next(i for i,p in enumerate(series) if decimal(p["drawdown"]) == worst)
    trough = series[i]
    peak_value = trough["peak_nav_minor"]
    peaks = [p for p in series[:i] if p["nav_minor"] == peak_value]
    peak_day = peaks[-1]["session"] if peaks else base["period_metrics"]["window"]["anchor_session"]
    require(peaks or peak_value == base["period_metrics"]["account"]["initial_nav_minor"],
            "drawdown peak lacks saved observation or initial anchor")
    recovery = next((p["session"] for p in series[i+1:] if p["nav_minor"] >= peak_value), None)
    out.update(status="AVAILABLE", peak_session=peak_day, trough_session=trough["session"],
        peak_nav_minor=peak_value, trough_nav_minor=trough["nav_minor"],
        elapsed_calendar_days=(date.fromisoformat(trough["session"])-date.fromisoformat(peak_day)).days,
        peak_is_initial_anchor=not bool(peaks), recovery_session=recovery,
        recovery_status="OPEN" if recovery is None else "RECOVERED")
    return out


def _returns(base):
    prior = base["period_metrics"]["account"]["initial_nav_minor"]
    values = []
    for point in base["series"]:
        value = point["nav_minor"]
        values.append(None if prior <= 0 else Decimal(value)/prior-1)
        prior = value
    return values


def _risk(base, risk_free, returns):
    window = base["period_metrics"]["window"]
    years = decimal(window["year_fraction"])
    n = len(returns)
    rate = decimal(risk_free["annual_effective_rate"])
    prior = window["anchor_session"]
    excess = []
    for point, ret in zip(base["series"], returns):
        fraction = decimal(_window(prior, point["session"])["year_fraction"])
        excess.append(None if ret is None else ret - ((1+rate)**fraction-1))
        prior = point["session"]
    status, value, factor = "AVAILABLE", None, None
    if any(v is None for v in excess): status = "MISSING_RETURN"
    elif years < 1: status = "INSUFFICIENT_SPAN"
    elif n < 30: status = "INSUFFICIENT_OBSERVATIONS"
    else:
        mean,std = _sample_std(excess)
        factor = str((Decimal(n)/years).sqrt())
        if std == 0: status = "ZERO_VOLATILITY"
        else: value = str(mean/std*decimal(factor))
    account = base["period_metrics"]["account"]
    cstatus, calmar = account["cagr_status"], None
    if account["cagr"] is not None:
        dd = abs(decimal(account["max_drawdown"]))
        cstatus = "ZERO_DRAWDOWN" if dd == 0 else "AVAILABLE"
        if dd: calmar = str(decimal(account["cagr"])/dd)
    return {"sharpe":dict(status=status,value=value,observations=n,year_fraction=str(years),
        annualization_factor=factor,risk_free=dict(risk_free)), "calmar":dict(status=cstatus,value=calmar)}


def _return_distribution(base, rule):
    values = [decimal(e["net_return"]) for e in base["episodes"] if e["statistics_eligible"]]
    edges = [None, *map(decimal,rule["edges"]), None]
    enough = len(values) >= rule["minimum_episodes"]
    bins = []
    if enough:
        for low,high in zip(edges,edges[1:]):
            bins.append(dict(lower=None if low is None else f"{low:.2f}",
                upper=None if high is None else f"{high:.2f}",
                count=sum((low is None or value >= low) and (high is None or value < high) for value in values)))
    return {**rule, "status":"AVAILABLE" if enough else "INSUFFICIENT_SAMPLE",
            "included_episode_count":len(values), "bins":bins}


def _series(base, returns):
    initial = base["period_metrics"]["account"]["initial_nav_minor"]
    wealth = [initial,*[p["nav_minor"] for p in base["series"]]]
    out=[]
    for i,p in enumerate(base["series"]):
        status,value,vol = "INSUFFICIENT_WINDOW",None,None
        if i >= 19:
            samples = returns[i-19:i+1]
            status = "MISSING_RETURN" if any(v is None for v in samples) or wealth[i-19] <= 0 else "AVAILABLE"
            if status == "AVAILABLE":
                value = str(Decimal(p["nav_minor"])/wealth[i-19]-1)
                vol = str(_sample_std(samples)[1])
        out.append(dict(session=p["session"],committed_sequence=p["committed_sequence"],
            account_cumulative_return=str(Decimal(p["nav_minor"])/initial-1),
            rolling_return_20=value,rolling_volatility_20=vol,rolling_status=status))
    return out


def _execution_summary(saved):
    gross = sum(f["gross_minor"] for f in saved["fills"])
    fees = sum(f["fee_minor"] for f in saved["fills"])
    mean_nav = Decimal(sum(p["nav_minor"] for p in saved["nav"]))/len(saved["nav"])
    return dict(turnover_status="AVAILABLE" if mean_nav > 0 else "ZERO_MEAN_NAV",
        two_sided_turnover=None if mean_nav == 0 else str(Decimal(gross)/mean_nav),
        fee_ratio=str(Decimal(fees)/saved["initial_nav_minor"]),
        gross_traded_minor=gross,fees_minor=fees,mean_nav_minor=str(mean_nav),
        initial_nav_minor=saved["initial_nav_minor"])


def _concentration(saved):
    by_day={p["session"]:[] for p in saved["nav"]}
    for p in saved["positions"]:
        by_day[p["session"]].append(p)
    out=[]
    for p in saved["nav"]:
        positions=by_day[p["session"]]
        require(all(v["committed_sequence"] == p["committed_sequence"] for v in positions),
                "concentration position/NAV watermark mismatch")
        largest=max(positions,key=lambda v:(v["market_value_minor"],v["security_id"]),default=None)
        out.append(dict(session=p["session"],committed_sequence=p["committed_sequence"],
            status="ZERO_NAV" if p["nav_minor"] == 0 else "AVAILABLE",
            security_id=None if largest is None else largest["security_id"],
            maximum_single_security_weight=None if p["nav_minor"] == 0 else
            str(Decimal(0 if largest is None else largest["market_value_minor"])/p["nav_minor"])))
    return out


def _episode_points(base):
    return [dict(episode_id=e["episode_id"],net_return=e["net_return"],entry_session=e["entry_session"],
        exit_session=e["exit_session"],holding_calendar_days=(date.fromisoformat(e["exit_session"])-
        date.fromisoformat(e["entry_session"])).days) for e in base["episodes"] if e["statistics_eligible"]]


def _execution_trace(saved):
    orders, fills = {}, {}
    for order in saved.get("orders",[]):
        orders.setdefault(order.get("intent_id"),[]).append(order)
    for fill in saved["fills"]:
        fills.setdefault(fill.get("order_id"),[]).append(fill)
    trace=[]
    if "decisions" not in saved:
        return None
    for index,decision in enumerate(saved["decisions"]):
        linked=[]
        for intent in decision.get("intents",[]):
            matched=[]
            for order in orders.get(intent.get("intent_id"),[]) if intent.get("intent_id") is not None else []:
                actual=fills.get(order.get("order_id"),[]) if order.get("order_id") is not None else []
                matched.append(dict(order_id=order.get("order_id"),status=order.get("status"),reason=order.get("reason"),
                    requested_quantity=order.get("quantity"),filled_quantity=order.get("filled_quantity"),
                    unfilled_quantity=order.get("unfilled_quantity"),execution_admission=order.get("execution_admission"),
                    committed_sequence=order.get("committed_sequence"),fill_ids=[f.get("fill_id") for f in actual],fills=actual))
            linked.append(dict(intent_id=intent.get("intent_id"),intent=intent,orders=matched))
        trace.append(dict(decision_index=index,trade_session=decision.get("trade_session"),
            feature_session=decision.get("feature_session"),status=decision.get("status"),
            selected_security_id=decision.get("selected_security_id"),selected_security_ids=decision.get("selected_security_ids"),
            top_k=decision.get("top_k"),targets=decision.get("targets"),trace=decision.get("trace"),intent_links=linked))
    return trace


def _identity(wire):
    return Document.from_dict({k:wire[k] for k in ("input_run_ref","base_evaluation_ref",
        "base_evaluation_content_digest","benchmark_refs","spec_ref","dividend_scope_ref",
        "evaluation_version","implementation_ref")}).identity


def evaluate_saved_analysis(run, base_report, *, benchmarks, spec):
    """Only consume the saved run/base; never invoke the account or old evaluator."""
    with localcontext(Context(prec=40,rounding=ROUND_HALF_UP)):
        return _evaluate(run,base_report,benchmarks,spec)


def _evaluate(run, base_report, benchmarks, spec):
    from .evaluation import EvaluationReport, EvaluationSpec, _verify_run, _verify_report
    from .analysis_benchmarks import benchmark_comparisons
    saved = _verify_run(run)
    base = _verify_report(base_report)
    require(base["contract_version"] == "evaluation_report_v2", "saved v2 base evaluation required")
    require(base["input_run_ref"] == {k:saved[k] for k in ("run_id","content_digest","committed_sequence")} and
            all(base[k] == saved[k] for k in ("signal_ref","market_ref","profile_ref")),
            "base evaluation belongs to a different saved account")
    require(isinstance(spec,EvaluationSpec), "EvaluationSpec required")
    sw = spec.to_dict()
    require(sw == analysis_evaluation_spec(risk_free=sw.get("risk_free")).to_dict(), "unsupported analysis spec")
    comparisons, inputs, refs = benchmark_comparisons(base,benchmarks)
    returns = _returns(base)
    wire = {k:v for k,v in base.items() if k not in ("content_digest","evaluation_ref")}
    wire.update(contract_version="evaluation_report_v3",evaluation_version=ANALYSIS_VERSION,
        implementation_ref=IMPLEMENTATION_REF,spec_ref=spec.identity,spec=sw,
        base_evaluation_ref=base["evaluation_ref"],base_evaluation_content_digest=base["content_digest"],base_evaluation=base,
        benchmark_inputs=inputs,benchmark_refs=refs,benchmark_comparisons=comparisons,
        status="PARTIAL" if base["status"] == "PARTIAL" or any(c["status"] != "COMPLETE" for c in comparisons.values()) else "COMPLETE",
        drawdown_interval=_drawdown(base),return_distribution=_return_distribution(base,sw["return_distribution"]),
        risk_metrics=_risk(base,sw["risk_free"],returns),analysis_series=_series(base,returns),
        execution_summary=_execution_summary(saved),concentration_series=_concentration(saved),
        episode_points=_episode_points(base),execution_trace=_execution_trace(saved),
        limitations=[*base["limitations"],
            "Base v2 scope statements (including no Sharpe) describe the original evaluation; risk values here follow v3 statuses and explicit assumptions.",
            "Cross-market curves use exact native date labels retrospectively, without same-instant availability or FX comparison."])
    wire["evaluation_ref"]=_identity(wire)
    wire["content_digest"]=Document.from_dict(wire).identity
    return EvaluationReport.from_dict(wire)


def verify_analysis_wire(wire):
    """Saved reader verifies schema and reference closure, never business metrics."""
    from .evaluation import _verify_report_wire, EvaluationSpec
    base = _verify_report_wire(wire["base_evaluation"])
    require(base["contract_version"] == "evaluation_report_v2" and
        wire["base_evaluation_ref"] == base["evaluation_ref"] and
        wire["base_evaluation_content_digest"] == base["content_digest"], "unbound saved base evaluation")
    require(all(wire[k] == base[k] for k in PRESERVED), "saved v2 facts changed inside analysis")
    spec=wire["spec"]
    require(spec == analysis_evaluation_spec(risk_free=spec.get("risk_free")).to_dict() and
        wire["spec_ref"] == EvaluationSpec.from_dict(spec).identity and wire["evaluation_version"] == ANALYSIS_VERSION,
        "saved analysis spec/version mismatch")
    fields(wire["benchmark_inputs"]," ".join(BENCHMARK_KEYS));fields(wire["benchmark_refs"]," ".join(BENCHMARK_KEYS))
    for k in BENCHMARK_KEYS:
        native=wire["benchmark_inputs"][k]
        if k == "NASDAQ100" or (k == "SSE_COMPOSITE" and native is None):
            require(native is None and wire["benchmark_refs"][k] is None and
                    wire["benchmark_comparisons"][k]["status"] == "SOURCE_UNAVAILABLE",
                    "new benchmark native contract awaits the Data owner handoff")
        elif k == "SSE_COMPOSITE":
            from .retrospective_benchmark import validate_sse_for_base
            validate_sse_for_base(native,base)
        require(wire["benchmark_refs"][k] == (None if native is None else Document.from_dict(native).identity),
                "saved benchmark input reference mismatch")
    require(wire["benchmark_inputs"]["CSI300"] == base["benchmark_input"], "original CSI300 benchmark changed")
    from .evaluation import _validate_benchmark_wire
    _validate_benchmark_wire(wire["benchmark_inputs"]["CSI300"],
        [base["period_metrics"]["window"]["anchor_session"],*[p["session"] for p in base["series"]]],set())
    digest(wire["implementation_ref"])
    require(wire["evaluation_ref"] == _identity(wire), "saved analysis identity mismatch")
    require([p["session"] for p in wire["analysis_series"]] == [p["session"] for p in base["series"]] and
        [p["committed_sequence"] for p in wire["analysis_series"]] == [p["committed_sequence"] for p in base["series"]],
        "saved analysis series/watermark scope mismatch")
    _validate_outputs(wire,base)


def _nullable_decimal(value):
    if value is not None:
        decimal(value)


def _validate_outputs(wire, base):
    """Structural checks only; none of the analytical functions are invoked."""
    from ..core.contracts import session
    added="base_evaluation_ref base_evaluation_content_digest base_evaluation benchmark_inputs benchmark_refs benchmark_comparisons drawdown_interval return_distribution risk_metrics analysis_series execution_summary concentration_series episode_points execution_trace"
    require(set(wire) == (set(base)-{"content_digest"}) | set(added.split()), "saved analysis report shape mismatch")
    risk=wire["risk_metrics"];fields(risk,"sharpe calmar")
    sharpe=risk["sharpe"];fields(sharpe,"status value observations year_fraction annualization_factor risk_free")
    require(sharpe["status"] in ("AVAILABLE","MISSING_RETURN","INSUFFICIENT_SPAN","INSUFFICIENT_OBSERVATIONS","ZERO_VOLATILITY") and
            (sharpe["value"] is not None) == (sharpe["status"] == "AVAILABLE"), "invalid saved Sharpe status/value")
    _nullable_decimal(sharpe["value"]);_nullable_decimal(sharpe["annualization_factor"])
    integer(sharpe["observations"])
    require(sharpe["observations"] == len(base["series"]) and sharpe["year_fraction"] == base["period_metrics"]["window"]["year_fraction"] and
            sharpe["risk_free"] == wire["spec"]["risk_free"], "saved Sharpe input metadata mismatch")
    missing=any(p["nav_minor"] <= 0 for p in base["series"][:-1])
    expected=("MISSING_RETURN" if missing else "INSUFFICIENT_SPAN" if decimal(sharpe["year_fraction"]) < 1 else
              "INSUFFICIENT_OBSERVATIONS" if sharpe["observations"] < 30 else None)
    require(sharpe["status"] == expected if expected is not None else sharpe["status"] in ("AVAILABLE","ZERO_VOLATILITY"),
            "saved Sharpe eligibility/status mismatch")
    calmar=risk["calmar"];fields(calmar,"status value")
    require(calmar["status"] in ("AVAILABLE","INSUFFICIENT_SPAN","MISSING_BOUNDARY","ZERO_DRAWDOWN") and
            (calmar["value"] is not None) == (calmar["status"] == "AVAILABLE"), "invalid saved Calmar status/value")
    _nullable_decimal(calmar["value"])
    account=base["period_metrics"]["account"]
    expected=(account["cagr_status"] if account["cagr"] is None else
              "ZERO_DRAWDOWN" if decimal(account["max_drawdown"]).is_zero() else "AVAILABLE")
    require(calmar["status"] == expected,"saved Calmar eligibility/status mismatch")
    dd=wire["drawdown_interval"]
    fields(dd,"status peak_session trough_session peak_nav_minor trough_nav_minor drawdown elapsed_calendar_days peak_is_initial_anchor recovery_session recovery_status")
    require(dd["status"] in ("AVAILABLE","NO_DRAWDOWN") and type(dd["peak_is_initial_anchor"]) is bool and
            dd["drawdown"] == base["period_metrics"]["account"]["max_drawdown"], "saved drawdown scope mismatch")
    for name in ("peak_session","trough_session","recovery_session"):
        if dd[name] is not None: session(dd[name])
    for name in ("peak_nav_minor","trough_nav_minor","elapsed_calendar_days"):
        if dd[name] is not None: integer(dd[name])
    require(dd["recovery_status"] in ("NOT_APPLICABLE","OPEN","RECOVERED") and
            (dd["recovery_session"] is not None) == (dd["recovery_status"] == "RECOVERED"), "invalid saved recovery")
    dist=wire["return_distribution"]
    fields(dist,"metric unit edges interval minimum_episodes status included_episode_count bins")
    rule=wire["spec"]["return_distribution"]
    require(all(dist[k] == rule[k] for k in rule) and dist["status"] in ("AVAILABLE","INSUFFICIENT_SAMPLE"), "saved return distribution rule mismatch")
    integer(dist["included_episode_count"])
    require(dist["included_episode_count"] == base["episode_metrics"]["eligible_closed_count"], "saved distribution sample scope mismatch")
    edges=[None,*rule["edges"],None]
    if dist["status"] == "AVAILABLE":
        require(dist["included_episode_count"] >= rule["minimum_episodes"] and len(dist["bins"]) == len(edges)-1,
                "saved distribution sample/bins mismatch")
        for point,low,high in zip(dist["bins"],edges,edges[1:]):
            fields(point,"lower upper count");integer(point["count"])
            require((point["lower"],point["upper"]) == (low,high), "saved distribution edges mismatch")
        require(sum(p["count"] for p in dist["bins"]) == dist["included_episode_count"], "saved distribution count mismatch")
    else:
        require(dist["included_episode_count"] < rule["minimum_episodes"] and dist["bins"] == [], "invalid insufficient saved distribution")
    for point in wire["analysis_series"]:
        fields(point,"session committed_sequence account_cumulative_return rolling_return_20 rolling_volatility_20 rolling_status")
        require(point["rolling_status"] in ("AVAILABLE","INSUFFICIENT_WINDOW","MISSING_RETURN"), "invalid saved rolling status")
        decimal(point["account_cumulative_return"])
        for name in ("rolling_return_20","rolling_volatility_20"):
            require((point[name] is not None) == (point["rolling_status"] == "AVAILABLE"), "saved rolling null/status mismatch")
            _nullable_decimal(point[name])
    summary=wire["execution_summary"]
    fields(summary,"turnover_status two_sided_turnover fee_ratio gross_traded_minor fees_minor mean_nav_minor initial_nav_minor")
    require(summary["turnover_status"] in ("AVAILABLE","ZERO_MEAN_NAV") and
            (summary["two_sided_turnover"] is not None) == (summary["turnover_status"] == "AVAILABLE"), "saved turnover status/value mismatch")
    for name in ("gross_traded_minor","fees_minor","initial_nav_minor"): integer(summary[name])
    for name in ("two_sided_turnover","fee_ratio","mean_nav_minor"): _nullable_decimal(summary[name])
    require([(p["session"],p["committed_sequence"]) for p in wire["concentration_series"]] ==
            [(p["session"],p["committed_sequence"]) for p in base["series"]], "saved concentration scope/watermark mismatch")
    for point in wire["concentration_series"]:
        fields(point,"session committed_sequence status security_id maximum_single_security_weight")
        require(point["status"] in ("AVAILABLE","ZERO_NAV") and
            (point["maximum_single_security_weight"] is not None) == (point["status"] == "AVAILABLE"), "invalid saved concentration")
        _nullable_decimal(point["maximum_single_security_weight"])
    for point in wire["episode_points"]:
        fields(point,"episode_id net_return entry_session exit_session holding_calendar_days")
        digest(point["episode_id"]);decimal(point["net_return"]);session(point["entry_session"]);session(point["exit_session"])
        integer(point["holding_calendar_days"])
    require(wire["execution_trace"] is None or type(wire["execution_trace"]) is list, "saved trace must be observations or null")
    for index,point in enumerate(wire["execution_trace"] or []):
        fields(point,"decision_index trade_session feature_session status selected_security_id selected_security_ids top_k targets trace intent_links")
        require(point["decision_index"] == index, "saved decision trace order mismatch")
        for link in point["intent_links"]:
            fields(link,"intent_id intent orders")
            require(link["intent_id"] == link["intent"].get("intent_id"), "saved intent trace link mismatch")
            for order in link["orders"]:
                fields(order,"order_id status reason requested_quantity filled_quantity unfilled_quantity execution_admission committed_sequence fill_ids fills")
                require(order["fill_ids"] == [f.get("fill_id") for f in order["fills"]] and
                    all(f.get("order_id") == order["order_id"] for f in order["fills"]), "saved order/fill trace link mismatch")
    fields(wire["benchmark_comparisons"]," ".join(BENCHMARK_KEYS))
    expected="PARTIAL" if base["status"] == "PARTIAL" or any(
        c["status"] != "COMPLETE" for c in wire["benchmark_comparisons"].values()) else "COMPLETE"
    require(wire["status"] == expected,"saved analysis completeness status mismatch")
    for key,comparison in wire["benchmark_comparisons"].items():
        require(comparison["input_ref"] == wire["benchmark_refs"][key] and
            comparison["status"] in ("COMPLETE","PARTIAL","SOURCE_UNAVAILABLE") and
            comparison["clock_scope"] == "RETROSPECTIVE_LOCAL_SESSION", "saved comparison input/status/clock mismatch")
        if comparison["status"] == "SOURCE_UNAVAILABLE":
            require(wire["benchmark_inputs"][key] is None and comparison["native_series"] == [] and comparison["series"] == [],
                    "unavailable benchmark cannot contain observations")
        else:
            native=wire["benchmark_inputs"][key]
            require(key in ("CSI300","SSE_COMPOSITE") and comparison["security_id"] == native["security_id"] and
                comparison["currency"] == "CNY" and comparison["timezone"] == "Asia/Shanghai" and
                comparison["return_basis"] == native["series_kind"] and comparison["native_calendar"] == native["calendar"],
                "saved comparison native metadata mismatch")
            if key == "SSE_COMPOSITE":
                require(all(comparison[output] == native[source] for output,source in
                    (("observation_snapshot_id","snapshot_id"),("observation_cutoff","knowledge_cutoff"),
                    ("observation_pit_policy","pit_policy"),("observation_purpose","purpose"))), "saved SSE observation scope differs")
            require([p["native_session"] for p in comparison["native_series"]] == native["calendar"],
                    "saved comparison native dates mismatch")
            points=[*comparison["native_series"],*comparison["series"]]
            percentages=["benchmark_cumulative_return" in point for point in points]
            require(not any(percentages) or all(percentages),"partial benchmark percentage projection")
            if all(percentages):
                for point in points:
                    require((point["benchmark_cumulative_return"] is None)==(point["normalized_index"] is None),
                        "benchmark percentage null/boundary differs")
                    _nullable_decimal(point["benchmark_cumulative_return"])
            for point,row in zip(comparison["native_series"],native["rows"]):
                require(all(point[name] == row[name] for name in ("close","available_at","source_refs")),
                        "saved comparison native observations mismatch")
            indexed={p["session"]:p for p in native["rows"]}
            anchor_day=base["period_metrics"]["window"]["anchor_session"]
            anchor=indexed.get(anchor_day)
            anchor_close=None if anchor is None else anchor["close"]
            require(comparison["anchor_session"]==anchor_day and comparison["anchor_close"]==anchor_close,
                    "saved comparison anchor raw boundary differs")
            if key == "SSE_COMPOSITE":require(all(percentages),"new SSE contract requires saved percentages")
            for point in comparison["native_series"]:
                missing=anchor_close is None or point["close"] is None
                require((point["normalized_index"] is None)==missing,"native normalized boundary/null differs")
                _nullable_decimal(point["normalized_index"])
            require([p["account_session"] for p in comparison["series"]] == [p["session"] for p in base["series"]],
                    "saved comparison account dates mismatch")
            for point in comparison["series"]:
                require(point["native_session"] in (None,point["account_session"]), "saved cross-market date filling forbidden")
                row=indexed.get(point["account_session"])
                missing=anchor_close is None or row is None or row["close"] is None
                require((point["normalized_index"] is None)==missing,"projected normalized boundary/null differs")
                require(point["native_session"] == (None if row is None else row["session"]) and
                    all(point[name] == (None if row is None else row[name]) for name in ("close","available_at")) and
                    point["source_refs"] == ([] if row is None else row["source_refs"]),
                    "saved comparison projected native observations mismatch")
                relative_status=("MISSING_BOUNDARY" if anchor_close is None else "MISSING_OBSERVATION") if missing else (
                    "PRICE_INDEX_PROXY" if comparison["currency"] == "CNY" else "FX_REQUIRED")
                require(point["relative_status"]==relative_status and
                    (point["account_relative_wealth"] is None)==(missing or comparison["currency"] != "CNY"),
                    "saved relative wealth raw boundary/status differs")
                for name in ("close","normalized_index","account_relative_wealth"): _nullable_decimal(point[name])
            complete=anchor_close is not None and all(indexed.get(p["session"]) is not None and
                indexed[p["session"]]["close"] is not None for p in base["series"])
            require(comparison["status"]==("COMPLETE" if complete else "PARTIAL"),"saved benchmark missing/status differs")
