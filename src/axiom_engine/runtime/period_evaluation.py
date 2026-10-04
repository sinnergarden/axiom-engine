"""Bounded saved-wealth CAGR under Trade §11.2, never account replay."""
from calendar import isleap
from datetime import date
from decimal import Decimal

from ..core.contracts import fields, integer, require, session
from ..core.portfolio import decimal

DAY_COUNT = "actual_actual_calendar_year_split"


def _window(anchor, end):
    session(anchor); session(end)
    first, last = date.fromisoformat(anchor), date.fromisoformat(end)
    require(first < last, "positive saved evaluation interval required")
    cursor, segments = first, []
    while cursor < last:
        boundary = last if cursor.year == last.year else date(cursor.year + 1, 1, 1)
        segments.append({"year": cursor.year, "days": (boundary - cursor).days,
                         "year_days": 366 if isleap(cursor.year) else 365})
        cursor = boundary
    elapsed = (last - first).days
    require(sum(s["days"] for s in segments) == elapsed, "year segments must cover the interval")
    years = sum((Decimal(s["days"]) / s["year_days"] for s in segments), Decimal(0))
    return {"anchor_session": anchor, "end_session": end, "elapsed_calendar_days": elapsed,
            "day_count": DAY_COUNT, "year_segments": segments, "year_fraction": str(years)}


def _cagr(start, end, years):
    if start is None or end is None:
        return {"cagr_status": "MISSING_BOUNDARY", "cagr": None,
                "cagr_reason": "MISSING_ANCHOR_OR_END_VALUE"}
    require(start.is_finite() and end.is_finite() and start > 0 and end >= 0,
            "positive initial wealth and nonnegative final wealth required")
    if years < 1:
        return {"cagr_status": "INSUFFICIENT_SPAN", "cagr": None,
                "cagr_reason": "YEAR_FRACTION_BELOW_ONE"}
    value = Decimal(-1) if end == 0 else (end / start) ** (Decimal(1) / years) - 1
    require(value.is_finite(), "finite CAGR required")
    return {"cagr_status": "AVAILABLE", "cagr": str(value), "cagr_reason": None}


def period_metrics(run, series, benchmark, *, anchor):
    """Called only inside the evaluator's independent Decimal context."""
    window = _window(anchor, series[-1]["session"])
    years = decimal(window["year_fraction"])
    initial, final = run["initial_nav_minor"], series[-1]["nav_minor"]
    initial_close, final_close = benchmark["anchor_close"], benchmark["series"][-1]["close"]
    return {"window": window,
        "account": {**_cagr(Decimal(initial), Decimal(final), years),
            "initial_nav_minor": initial, "final_nav_minor": final,
            "total_return": str(Decimal(final) / initial - 1),
            "max_drawdown": str(min(decimal(p["drawdown"]) for p in series))},
        "benchmark": {**_cagr(None if initial_close is None else decimal(initial_close),
                               None if final_close is None else decimal(final_close), years),
            "anchor_close": initial_close, "end_close": final_close,
            "total_return": benchmark["total_return"], "max_drawdown": benchmark["max_drawdown"]}}


def validate_period_metrics(wire):
    """Validate saved v2 shape and values without recomputing metrics."""
    fields(wire, "window account benchmark")
    window = wire["window"]
    fields(window, "anchor_session end_session elapsed_calendar_days day_count year_segments year_fraction")
    session(window["anchor_session"]); session(window["end_session"])
    require(window["anchor_session"] < window["end_session"] and window["day_count"] == DAY_COUNT,
            "unsupported saved evaluation interval")
    integer(window["elapsed_calendar_days"], 1)
    require(decimal(window["year_fraction"]) > 0 and type(window["year_segments"]) is list and
            bool(window["year_segments"]), "positive saved year fraction/segments required")
    for segment in window["year_segments"]:
        fields(segment, "year days year_days")
        integer(segment["year"], 1); integer(segment["days"], 1)
        require(type(segment["year_days"]) is int and segment["year_days"] in (365, 366), "invalid saved year days")
    for name, endpoints in (("account", "initial_nav_minor final_nav_minor"),
                            ("benchmark", "anchor_close end_close")):
        leg = wire[name]
        fields(leg, "cagr_status cagr cagr_reason total_return max_drawdown " + endpoints)
        require(leg["cagr_status"] in ("AVAILABLE", "INSUFFICIENT_SPAN", "MISSING_BOUNDARY"), "invalid saved CAGR status")
        if leg["cagr_status"] == "AVAILABLE":
            decimal(leg["cagr"])
            require(leg["cagr_reason"] is None, "available CAGR cannot have a missing reason")
        else:
            require(leg["cagr"] is None and type(leg["cagr_reason"]) is str and bool(leg["cagr_reason"]),
                    "unavailable CAGR needs an explicit reason")
        for key in ("total_return", "max_drawdown"):
            if leg[key] is not None:
                decimal(leg[key])
    integer(wire["account"]["initial_nav_minor"], 1)
    integer(wire["account"]["final_nav_minor"])
    for key in ("anchor_close", "end_close"):
        if wire["benchmark"][key] is not None:
            require(decimal(wire["benchmark"][key]) > 0, "positive native benchmark close required")
