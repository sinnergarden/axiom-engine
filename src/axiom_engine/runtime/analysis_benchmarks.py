"""Exact native-date display comparisons, without forward fill or FX inference."""
from decimal import Decimal

from ..core.contracts import fields, require
from ..core.portfolio import decimal
from .analysis_evaluation import BENCHMARK_KEYS


def _project_native(base, native, input_ref):
    """Internal normalized native observations; adapters own native input admission."""
    anchor_day = base["period_metrics"]["window"]["anchor_session"]
    indexed = {r["session"]:r for r in native["rows"]}
    anchor = indexed.get(anchor_day)
    initial = None if anchor is None or anchor["close"] is None else decimal(anchor["close"])
    by_day = {p["session"]:p for p in base["series"]}
    native_series=[]
    for row in native["rows"]:
        close = None if row["close"] is None else decimal(row["close"])
        native_series.append(dict(native_session=row["session"],close=row["close"],
            available_at=row["available_at"],source_refs=row["source_refs"],
            normalized_index=None if initial is None or close is None else str(close/initial),
            benchmark_cumulative_return=None if initial is None or close is None else str(close/initial-1)))
    projected=[]
    for day,p in by_day.items():
        row=indexed.get(day)
        close=None if row is None or row["close"] is None else decimal(row["close"])
        norm=None if close is None or initial is None else close/initial
        if norm is None:
            relative,status=None,"MISSING_BOUNDARY" if initial is None else "MISSING_OBSERVATION"
        elif native["currency"] != "CNY":
            relative,status=None,"FX_REQUIRED"
        else:
            relative,status=str(decimal(p["nav_index"])/norm-1),"PRICE_INDEX_PROXY"
        projected.append(dict(account_session=day,native_session=None if row is None else row["session"],
            close=None if row is None else row["close"],available_at=None if row is None else row["available_at"],
            source_refs=[] if row is None else row["source_refs"],normalized_index=None if norm is None else str(norm),
            benchmark_cumulative_return=None if norm is None else str(norm-1),
            account_relative_wealth=relative,relative_status=status))
    complete=initial is not None and all(p["normalized_index"] is not None for p in projected)
    return dict(input_ref=input_ref,security_id=native["security_id"],currency=native["currency"],
        return_basis=native["series_kind"],native_calendar=native["calendar"],timezone=native["timezone"],
        clock_scope="RETROSPECTIVE_LOCAL_SESSION",status="COMPLETE" if complete else "PARTIAL",
        unavailable_reason=None,anchor_session=anchor_day,anchor_close=None if anchor is None else anchor["close"],
        native_series=native_series,series=projected)


def _add_drawdowns(base, comparison, key):
    """Copy CSI authority; calculate SSE once from the admitted native prices."""
    anchor = comparison["anchor_session"]
    if key == "CSI300":
        values = {anchor: "0" if comparison["anchor_close"] is not None else None,
                  **{p["session"]: p["drawdown"] for p in base["benchmark"]["series"]}}
        worst = base["benchmark"]["max_drawdown"]
    else:
        peak = None if comparison["anchor_close"] is None else decimal(comparison["anchor_close"])
        values, worst, complete = {}, Decimal(0), peak is not None
        for point in comparison["native_series"]:
            close = None if point["close"] is None else decimal(point["close"])
            if close is None:
                peak, complete = None, False
            value = None
            if peak is not None:
                peak = max(peak, close)
                value = close / peak - 1
                worst = min(worst, value)
            values[point["native_session"]] = None if value is None else str(value)
        worst = str(worst) if complete else None
    comparison.update(projection_version="benchmark_comparison_v2", max_drawdown=worst)
    for point in comparison["native_series"]:
        point["benchmark_drawdown"] = values[point["native_session"]]
    for point in comparison["series"]:
        point["benchmark_drawdown"] = values.get(point["account_session"])


def benchmark_comparisons(base, benchmarks, *, projection_version="benchmark_comparison_v1"):
    from .evaluation import BenchmarkSeries, _validate_benchmark_wire
    fields(benchmarks," ".join(BENCHMARK_KEYS))
    anchor=base["period_metrics"]["window"]["anchor_session"]
    days=[anchor,*[p["session"] for p in base["series"]]]
    inputs,refs,comparisons={},{},{}
    for key in BENCHMARK_KEYS:
        value=benchmarks[key]
        if value is None:
            require(key != "CSI300","original CSI300 input is required")
            inputs[key],refs[key]=None,None
            comparisons[key]=dict(input_ref=None,security_id=None,currency=None,return_basis=None,
                native_calendar=None,timezone=None,clock_scope="RETROSPECTIVE_LOCAL_SESSION",
                status="SOURCE_UNAVAILABLE",unavailable_reason="OWNER_NATIVE_INPUT_NOT_SUPPLIED",
                anchor_session=anchor,anchor_close=None,native_series=[],series=[])
            if projection_version == "benchmark_comparison_v2":
                comparisons[key].update(projection_version=projection_version, max_drawdown=None)
            continue
        require(isinstance(value,BenchmarkSeries),"BenchmarkSeries required")
        native=value.to_dict()
        if key == "CSI300":
            require(native == base["benchmark_input"] and value.identity == base["benchmark_ref"],
                    "original CSI300 benchmark cannot be replaced")
            _validate_benchmark_wire(native,days,set())
        else:
            require(key == "SSE_COMPOSITE", "direct Nasdaq input unsupported; use the independent saved ETF account")
            from .retrospective_benchmark import validate_sse_for_base
            validate_sse_for_base(native,base)
        inputs[key],refs[key]=native,value.identity
        comparisons[key]=_project_native(base,{**native,"currency":"CNY","timezone":"Asia/Shanghai"},value.identity)
        if projection_version == "benchmark_comparison_v2":
            _add_drawdowns(base, comparisons[key], key)
        if key == "SSE_COMPOSITE":
            comparisons[key].update(observation_snapshot_id=native["snapshot_id"],observation_cutoff=native["knowledge_cutoff"],
                observation_pit_policy=native["pit_policy"],observation_purpose=native["purpose"])
    return comparisons,inputs,refs
