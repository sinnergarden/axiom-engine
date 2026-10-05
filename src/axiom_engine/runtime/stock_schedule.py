"""Frozen saved-fold clocks and an entry-local lookup for the sole Runtime."""
from ..core.contracts import Document, digest, fields, require, session
from ..core.stock_portfolio import StockPredictionFrame, instant, validate_stock_predictions


class StockPredictionSchedule(Document):
    """Original predictions and small model metadata; no training payloads."""


CLOCK_POLICY = {"contract_version": "stock_prediction_clock_policy_v1",
    "feature_cutoff_local_time": "20:30:00", "inference_cutoff_local_time": "21:00:00",
    "decision_local_time": "08:55:00", "execution": "next_exchange_session_open",
    "clock_basis": "declared_simulation"}
LIMITATIONS = ["Model and prediction clocks are declared simulation, not historical completion evidence.",
               "Research's saved-fold loader admits training labels, booster and Feature parents; Runtime consumes the original saved predictions."]


def _verify_model(model):
    fields(model, "contract_version dataset_ref feature_ref label_ref raw_label_refs fit_cutoff simulated_available_at clock_basis ordered_features feature_selection catalog_ref target_semantics parameters num_boost_round environment implementation_ref booster_digest feature_normalization label_normalization model_ref")
    require(model["contract_version"] == "stock_model_release_v2" and
            model["clock_basis"] == "declared_simulation", "Saved stock model contract required")
    unsigned = dict(model); reference = unsigned.pop("model_ref")
    digest(reference)
    require(Document.from_dict(unsigned).identity == reference, "Saved model metadata identity mismatch")


def _compile_folds(folds, calendar):
    require(type(calendar) is list and bool(calendar) and calendar == sorted(set(calendar)), "Frozen ordered exchange calendar required")
    for day in calendar:
        session(day)
    require(type(folds) is list and bool(folds), "Nonempty ordered saved folds required")
    previous = {day: calendar[i-1] for i, day in enumerate(calendar) if i}
    universe, trade_map, schedule, refs = None, {}, [], set()
    last = None
    for fold in folds:
        fields(fold, "fold_ref fold_spec model prediction_frame")
        digest(fold["fold_ref"])
        require(fold["fold_ref"] not in refs, "Duplicate saved fold ref")
        refs.add(fold["fold_ref"])
        spec, model = fold["fold_spec"], fold["model"]
        require(type(spec) is dict and spec.get("contract_version") in
                ("stock_ml_fold_spec_v1", "stock_ml_fold_spec_v2"), "Saved fold spec required")
        fields(spec, "contract_version training_window fit_session fit_cutoff simulated_model_available_at oos_trade_sessions inference_cutoff_by_session evaluation_cutoff")
        _verify_model(model)
        wire, rows = validate_stock_predictions(StockPredictionFrame.from_dict(fold["prediction_frame"]))
        require(wire["contract_version"] == "stock_prediction_run_v2", "Schedule requires original v2 predictions")
        unsigned = dict(wire); signal_ref = unsigned.pop("signal_run_ref")
        require(Document.from_dict(unsigned).identity == signal_ref, "Saved prediction identity mismatch")
        spec_ref = Document.from_dict(spec).identity
        require(wire["fold_spec_ref"] == spec_ref and wire["model_ref"] == model["model_ref"] and
                wire["feature_ref"] == model["feature_ref"] and wire["score_semantics"] == model["target_semantics"],
                "Saved prediction/model/fold linkage mismatch")
        if universe is None:
            universe = wire["universe"]
        require(wire["universe"] == universe, "Saved folds must share the frozen ordered prediction union")
        trades = spec["oos_trade_sessions"]
        require(type(trades) is list and bool(trades) and trades == sorted(set(trades)) and
                all(day in previous for day in trades), "Fold OOS exchange sessions required")
        require(last is None or last < trades[0], "Overlapping or unordered saved folds")
        last = trades[-1]
        features = [previous[day] for day in trades]
        clocks = spec["inference_cutoff_by_session"]
        require(type(clocks) is dict and set(clocks) == set(features) and
                {day for day, _ in rows} == set(features), "Fold prediction sessions differ from strict exchange predecessors")
        session(spec["fit_session"])
        require(spec["fit_session"] in calendar and
                instant(model["fit_cutoff"]) == instant(spec["fit_cutoff"]) and
                instant(model["simulated_available_at"]) == instant(spec["simulated_model_available_at"]) ==
                instant(spec["fit_session"] + "T20:45:00+08:00") and
                instant(spec["fit_cutoff"]) < instant(model["simulated_available_at"]), "Fold fit/model clock conflict")
        for trade, feature in zip(trades, features):
            infer, decision = instant(feature + "T21:00:00+08:00"), instant(trade + "T08:55:00+08:00")
            require(instant(clocks[feature]) == infer and
                    instant(model["simulated_available_at"]) < infer <= decision, "Fold inference/decision clock conflict")
            for security in universe:
                row = rows[feature, security]
                require(instant(row["feature_knowledge_cutoff"]) == instant(feature + "T20:30:00+08:00") and
                        instant(row["knowledge_cutoff"]) == instant(row["available_at"]) == infer and
                        instant(row["simulated_model_available_at"]) == instant(model["simulated_available_at"]),
                        "Saved row differs from fixed account clock policy")
            require(trade not in trade_map, "Duplicate schedule trade session")
            trade_map[trade] = (wire, rows)
            schedule.append({"trade_session": trade, "feature_session": feature,
                             "signal_run_ref": signal_ref, "fold_spec_ref": spec_ref})
    require([s["trade_session"] for s in schedule] == calendar[calendar.index(schedule[0]["trade_session"]):
            calendar.index(schedule[-1]["trade_session"])+1], "Gap in saved fold exchange schedule")
    return universe, schedule, trade_map


def stock_prediction_schedule(*, folds: list[dict], calendar: list[str]) -> StockPredictionSchedule:
    """Bind finite original saved folds; never fit, predict, query or run an account."""
    universe, schedule, _ = _compile_folds(folds, calendar)
    wire = {"contract_version": "stock_prediction_schedule_v1", "clock_policy": dict(CLOCK_POLICY),
            "calendar": calendar, "universe": universe, "folds": folds,
            "trade_schedule": schedule, "limitations": list(LIMITATIONS)}
    wire["schedule_ref"] = Document.from_dict(wire).identity
    return StockPredictionSchedule.from_dict(wire)


def _admit_schedule(wire, calendar):
    fields(wire, "contract_version schedule_ref clock_policy calendar universe folds trade_schedule limitations")
    require(wire["contract_version"] == "stock_prediction_schedule_v1" and
            wire["clock_policy"] == CLOCK_POLICY and wire["calendar"] == calendar and
            wire["limitations"] == LIMITATIONS, "Saved schedule policy/calendar mismatch")
    unsigned = dict(wire); reference = unsigned.pop("schedule_ref")
    require(Document.from_dict(unsigned).identity == reference, "Saved schedule identity mismatch")
    universe, schedule, admitted = _compile_folds(wire["folds"], calendar)
    require(wire["universe"] == universe and wire["trade_schedule"] == schedule, "Saved trade schedule differs from original folds")
    return wire, admitted


def _resource_preflight(request, limits):
    """Optional caller budget is operational; it does not change run identity."""
    fields(limits, "max_folds max_prediction_rows max_market_rows max_input_bytes")
    from ..core.contracts import integer
    for value in limits.values():
        integer(value)
    input_bytes = len(request.payload.encode("utf-8"))
    require(input_bytes <= limits["max_input_bytes"],
            f"Stock resource budget exceeded: max_input_bytes={input_bytes} > {limits['max_input_bytes']}")
    plan = request.to_dict()
    require(plan.get("contract_version") == "backtest_request_v4", "Resource limits require stock request v4")
    folds = plan["prediction_schedule"]["folds"]
    counts = {"max_folds": len(folds), "max_prediction_rows": sum(len(f["prediction_frame"]["rows"]) for f in folds),
              "max_market_rows": len(plan["market_replay"]["rows"])}
    for name, value in counts.items():
        require(value <= limits[name], f"Stock resource budget exceeded: {name}={value} > {limits[name]}")
    return plan
