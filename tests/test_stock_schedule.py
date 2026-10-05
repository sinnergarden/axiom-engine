"""Synthetic saved-fold admission and continuous single-account golden cases."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, StockPredictionFrame, plan_stock_portfolio
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, run_backtest, stock_prediction_schedule,
    stock_portfolio_policy, save_backtest_run, load_backtest_run, evaluate_backtest,
    long_history_evaluation_spec, stock_dividend_scope)
from axiom_engine.runtime import (analysis_evaluation_spec, evaluate_saved_analysis,
    build_fill_display, save_backtest_evaluation, load_backtest_evaluation,
    save_fill_display, load_fill_display)
from axiom_engine.runtime import stock_schedule, stock_inputs, stock_market
import test_stocks as legacy

CALENDAR = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
            "2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]


def seal(wire, name):
    wire = deepcopy(wire)
    wire.pop(name, None)
    wire[name] = Document.from_dict(wire).identity
    return wire


def fold(calendar, trades, reverse=False):
    fit = calendar[calendar.index(trades[0])-1]
    features = [calendar[calendar.index(d)-1] for d in trades]
    spec = {"contract_version": "stock_ml_fold_spec_v2", "training_window": {
        "unit": "calendar_years", "length": 2, "end": "previous_fit_session",
        "start": "fit_date_minus_years_inclusive", "leap_day": "clamp_feb_28"},
        "fit_session": fit, "fit_cutoff": fit + "T20:30:00+08:00",
        "simulated_model_available_at": fit + "T20:45:00+08:00", "oos_trade_sessions": trades,
        "inference_cutoff_by_session": {d: d + "T21:00:00+08:00" for d in features},
        "evaluation_cutoff": calendar[-1] + "T22:00:00+08:00"}
    model = seal({"contract_version": "stock_model_release_v2", "dataset_ref": legacy.REF,
        "feature_ref": legacy.REF, "label_ref": legacy.REF, "raw_label_refs": [legacy.REF],
        "fit_cutoff": spec["fit_cutoff"], "simulated_available_at": spec["simulated_model_available_at"],
        "clock_basis": "declared_simulation", "ordered_features": ["synthetic"], "feature_selection": {},
        "catalog_ref": legacy.REF, "target_semantics": "forward_5_session_cs_zscore_prediction",
        "parameters": {"seed": 42}, "num_boost_round": 100, "environment": {"kind": "synthetic"},
        "implementation_ref": legacy.REF, "booster_digest": legacy.REF,
        "feature_normalization": "same_date_visible_members_cs_zscore_no_fit", "label_normalization": {}}, "model_ref")
    with patch.object(legacy, "DAYS", features):
        frame = legacy.predictions([i if reverse else -i for i in range(6)])
    frame.update(contract_version="stock_prediction_run_v2", model_ref=model["model_ref"],
                 fold_spec_ref=Document.from_dict(spec).identity, clock_basis="declared_simulation",
                 limitations=["Synthetic schedule test; no fitted model or historical training evidence."])
    for row in frame["rows"]:
        row.update(feature_knowledge_cutoff=row["knowledge_cutoff"], feature_available_at=row["available_at"],
                   simulated_model_available_at=model["simulated_available_at"],
                   knowledge_cutoff=row["session"] + "T21:00:00+08:00", available_at=row["session"] + "T21:00:00+08:00",
                   source_refs=[legacy.REF, model["model_ref"]])
    return {"fold_ref": Document.from_dict({"synthetic_fold_spec": spec}).identity,
            "fold_spec": spec, "model": model, "prediction_frame": seal(frame, "signal_run_ref")}


def scheduled_request(calendar=None, split=6, k=3, cash=False):
    calendar = calendar or CALENDAR
    folds = [fold(calendar, calendar[1:split]), fold(calendar, calendar[split:], reverse=True)]
    schedule = stock_prediction_schedule(folds=folds, calendar=calendar).to_dict()
    with patch.object(legacy, "DAYS", calendar):
        event = {**legacy.cash_event(), "record_date": calendar[1], "ex_date": calendar[2]}
        plan = legacy.request(inputs=legacy.native_inputs([event]) if cash else None).to_dict()
    plan["contract_version"] = "backtest_request_v4"
    plan.pop("signal_frame")
    plan.update(account_id=f"synthetic-50w-top{k}", prediction_schedule=schedule,
                initial_account={"cash_minor": 50000000, "positions": {}},
                portfolio_policy=stock_portfolio_policy(top_k=k, execution_universe=legacy.SECURITIES))
    proof = plan["admission_evidence"]
    for name in ("signal_run_ref", "model_ref", "feature_ref", "listing_identity_equal_all_83", "admission_ref"):
        proof.pop(name)
    proof.update(contract_version="stock_snapshot_pair_admission_v2", prediction_schedule_ref=schedule["schedule_ref"],
        prediction_refs=[{"fold_ref": f["fold_ref"], **{n: f["prediction_frame"][n] for n in
            ("fold_spec_ref", "signal_run_ref", "model_ref", "feature_ref")}} for f in folds],
        listing_identity_checks={"checked_rows": 6, "paired_rows": 6, "mismatches": []})
    proof["prediction_membership"].update(compared=6*(len(calendar)-1), saved_prediction_rows=6*(len(calendar)-1))
    basis = proof["previous_close_basis_checks"]
    basis.pop("equal_all_83_23"); basis.pop("listing_suffix_and_SZSE_identity_all_83")
    basis.update(equal_rows=6*len(calendar), listing_identity_checked_rows=6*len(calendar), listing_identity_mismatches=[])
    proof = seal(proof, "admission_ref")
    plan.update(admission_evidence=proof, admission_ref=proof["admission_ref"])
    return BacktestRequest.from_dict(plan)


def reseal_schedule(plan):
    for f in plan["prediction_schedule"]["folds"]:
        f["prediction_frame"]["fold_spec_ref"] = Document.from_dict(f["fold_spec"]).identity
        f["prediction_frame"] = seal(f["prediction_frame"], "signal_run_ref")
    plan["prediction_schedule"] = seal(plan["prediction_schedule"], "schedule_ref")
    proof = plan["admission_evidence"]
    proof["prediction_schedule_ref"] = plan["prediction_schedule"]["schedule_ref"]
    proof["prediction_refs"] = [{"fold_ref": f["fold_ref"], **{n: f["prediction_frame"][n] for n in
        ("fold_spec_ref", "signal_run_ref", "model_ref", "feature_ref")}} for f in plan["prediction_schedule"]["folds"]]
    plan["admission_evidence"] = seal(proof, "admission_ref")
    plan["admission_ref"] = plan["admission_evidence"]["admission_ref"]


class StockScheduleTests(unittest.TestCase):
    def test_continuous_account_two_top_k_and_admission_once(self):
        request = scheduled_request(); original = request.payload
        with patch.object(stock_schedule, "validate_stock_predictions", wraps=stock_schedule.validate_stock_predictions) as frames, \
             patch.object(stock_schedule, "_verify_model", wraps=stock_schedule._verify_model) as models, \
             patch.object(stock_inputs, "native_batches", wraps=stock_inputs.native_batches) as native:
            three = run_backtest(request).to_dict()
        self.assertEqual((frames.call_count, models.call_count, native.call_count), (2, 2, 1))
        five = run_backtest(scheduled_request(k=5)).to_dict()
        self.assertEqual(request.payload, original)
        self.assertEqual(three["initial_nav_minor"], 50000000)
        self.assertEqual(three["signal_ref"], five["signal_ref"])
        self.assertNotEqual(three["run_id"], five["run_id"])
        self.assertNotEqual(three["fills"], five["fills"])
        self.assertEqual(three["runtime_version"], "axiom.backtest/4")
        self.assertEqual([d["trade_session"] for d in three["decisions"]], ["2024-01-08", "2024-01-15"])
        self.assertEqual([d["signal_ref"] for d in three["decisions"]],
                         [f["prediction_frame"]["signal_run_ref"] for f in three["plan"]["prediction_schedule"]["folds"]])
        self.assertGreater(three["decisions"][1]["expected_account_version"], 0)
        self.assertTrue(any(f["side"] == "SELL" and f["session"] == "2024-01-15" for f in three["fills"]))
        self.assertEqual(len(three["nav"]), 10)
        self.assertEqual(three["nav"][-1]["nav_minor"], 50000000-three["metrics"]["total_fees_minor"])
        self.assertEqual(three["contract_version"], "backtest_run_v4")

    def test_fold_switch_midweek_does_not_rebalance(self):
        result = run_backtest(scheduled_request(split=3)).to_dict()
        self.assertEqual([d["trade_session"] for d in result["decisions"]], ["2024-01-08", "2024-01-15"])
        self.assertEqual(result["plan"]["prediction_schedule"]["trade_schedule"][2]["trade_session"], "2024-01-10")

    def test_long_holiday_uses_actual_exchange_predecessor(self):
        calendar = ["2024-02-08", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22",
                    "2024-02-23", "2024-02-26", "2024-02-27"]
        result = run_backtest(scheduled_request(calendar=calendar)).to_dict()
        self.assertEqual(result["decisions"][0]["feature_session"], "2024-02-08")
        self.assertEqual(result["decisions"][0]["trade_session"], "2024-02-19")

    def test_admission_rejects_model_ref_clocks_union_overlap_and_gap_before_ledger(self):
        for mutation in ("model_hash", "fit_model", "model_infer", "feature_late", "infer_late", "union", "overlap", "gap", "proof_count"):
            plan = scheduled_request().to_dict(); schedule = plan["prediction_schedule"]
            first, second = schedule["folds"]
            if mutation == "model_hash": first["model"]["parameters"]["seed"] = 1
            if mutation == "fit_model": first["fold_spec"]["fit_cutoff"] = first["fold_spec"]["simulated_model_available_at"]
            if mutation == "model_infer": first["model"]["simulated_available_at"] = CALENDAR[0]+"T21:00:00+08:00"
            if mutation == "feature_late": first["prediction_frame"]["rows"][0]["feature_available_at"] = CALENDAR[0]+"T20:31:00+08:00"
            if mutation == "infer_late": first["prediction_frame"]["rows"][0]["available_at"] = CALENDAR[1]+"T09:00:00+08:00"
            if mutation == "union": second["prediction_frame"]["universe"].reverse()
            if mutation == "overlap": second["fold_spec"]["oos_trade_sessions"].insert(0, CALENDAR[5])
            if mutation == "gap": first["fold_spec"]["oos_trade_sessions"].remove(CALENDAR[3])
            if mutation == "proof_count": plan["admission_evidence"]["listing_identity_checks"]["checked_rows"] -= 1
            reseal_schedule(plan)
            with self.subTest(mutation=mutation), patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("ledger started")):
                with self.assertRaises(ContractError): run_backtest(BacktestRequest.from_dict(plan))

    def test_core_original_clocks_no_decision_and_late_previous_close(self):
        frame = scheduled_request().to_dict()["prediction_schedule"]["folds"][0]["prediction_frame"]
        with patch.object(legacy, "DAYS", CALENDAR): ctx = legacy.context()
        ctx.update(knowledge_cutoff=CALENDAR[0]+"T13:00:00Z", feature_knowledge_cutoff=CALENDAR[0]+"T12:30:00Z")
        account = {"cash_minor": 50000000, "positions": {}, "version": 0}
        result = plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account=account, context=ctx, top_k=3).to_dict()
        self.assertEqual(result["prediction_clock"]["inference_cutoff"], CALENDAR[0]+"T21:00:00+08:00")
        self.assertEqual(result["selected_security_ids"], legacy.SECURITIES[:3])
        ctx["reference_prices"][legacy.SECURITIES[0]]["available_at"] = CALENDAR[0]+"T20:31:00+08:00"
        with self.assertRaisesRegex(ContractError, "sizing reference"):
            plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account=account, context=ctx, top_k=3)
        frame["rows"][0].update(valid=False, score=None, invalid_reason="MISSING")
        frame = seal(frame, "signal_run_ref")
        result = plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account=account, context=ctx, top_k=3).to_dict()
        self.assertEqual(result["status"], "NO_DECISION")
        self.assertIn("prediction_clock", result)

    def test_saved_v4_loader_and_old_evaluation_consume_only_saved_values(self):
        run = run_backtest(scheduled_request())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"run.json"; save_backtest_run(run, path)
            with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("replay")), \
                 patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("ledger")):
                loaded = load_backtest_run(path)
            self.assertEqual(loaded.payload, run.payload)
            with patch.object(legacy, "DAYS", CALENDAR): benchmark = legacy.benchmark()
            report = evaluate_backtest(loaded, benchmark=benchmark, spec=long_history_evaluation_spec(),
                                       dividend_scope=stock_dividend_scope(loaded)).to_dict()
            self.assertEqual(report["input_run_ref"]["run_id"], run.to_dict()["run_id"])
            self.assertIsNone(report["period_metrics"]["account"]["cagr"])

    def test_saved_analysis_and_fill_display_keep_v4_inputs_and_short_sample_nulls(self):
        from test_fill_display import files
        from test_analysis_evaluation import RF
        run = run_backtest(scheduled_request())
        with patch.object(legacy, "DAYS", CALENDAR): benchmark = legacy.benchmark()
        base = evaluate_backtest(run, benchmark=benchmark, spec=long_history_evaluation_spec(), dividend_scope=stock_dividend_scope(run))
        with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("replay")):
            analysis = evaluate_saved_analysis(run, base, benchmarks={"CSI300": benchmark, "SSE_COMPOSITE": None, "NASDAQ100": None},
                                              spec=analysis_evaluation_spec(risk_free=RF))
            wire = analysis.to_dict()
            self.assertIsNone(wire["risk_metrics"]["sharpe"]["value"])
            self.assertIsNone(wire["risk_metrics"]["calmar"]["value"])
            self.assertEqual(wire["input_run_ref"]["run_id"], run.to_dict()["run_id"])
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                display = build_fill_display(run, display=files(root/"data", run))
                self.assertEqual(display.to_dict()["status"], "COMPLETE")
                save_backtest_evaluation(analysis, root/"evaluation.json")
                save_fill_display(display, root/"display.json")
                self.assertEqual(load_backtest_evaluation(root/"evaluation.json").payload, analysis.payload)
                self.assertEqual(load_fill_display(root/"display.json").payload, display.payload)

    def test_v4_stock_cash_receivable_and_unknown_payment_survive_saved_evaluation(self):
        from test_analysis_evaluation import RF
        run = run_backtest(scheduled_request(cash=True))
        self.assertGreater(run.to_dict()["final_account"]["receivable_minor"], 0)
        with patch.object(legacy, "DAYS", CALENDAR): benchmark = legacy.benchmark()
        base = evaluate_backtest(run, benchmark=benchmark, spec=long_history_evaluation_spec(), dividend_scope=stock_dividend_scope(run))
        self.assertEqual(base.to_dict()["episode_metrics"]["payment_unknown_count"], 1)
        report = evaluate_saved_analysis(run, base, benchmarks={"CSI300": benchmark, "SSE_COMPOSITE": None, "NASDAQ100": None},
                                         spec=analysis_evaluation_spec(risk_free=RF))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"evaluation.json"; save_backtest_evaluation(report, path)
            with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("replay")):
                self.assertEqual(load_backtest_evaluation(path).payload, report.payload)

    def test_caller_resource_budget_rejects_before_ledger_without_fold_cap(self):
        request = scheduled_request()
        limits = {"max_folds": 2, "max_prediction_rows": 60, "max_market_rows": 66,
                  "max_input_bytes": len(request.payload.encode())}
        self.assertEqual(run_backtest(request, limits=limits).payload, run_backtest(request).payload)
        for key in limits:
            with self.subTest(budget=key), patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("ledger started")):
                with self.assertRaisesRegex(ContractError, "budget exceeded"):
                    run_backtest(request, limits={**limits, key: limits[key]-1})
        folds = [fold(CALENDAR, [d]) for d in CALENDAR[1:]]
        schedule = stock_prediction_schedule(folds=folds, calendar=CALENDAR).to_dict()
        self.assertEqual(len(schedule["folds"]), 10)

    def test_resealed_saved_mapping_tamper_is_rejected_without_replay(self):
        wire = run_backtest(scheduled_request()).to_dict()
        wire["decisions"][0]["trade_session"] = "2024-01-09"
        wire = seal(wire, "content_digest")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.json"; path.write_text(Document.from_dict(wire).payload)
            with self.assertRaisesRegex(ContractError, "trade mapping"):
                load_backtest_run(path)

    def test_equivalent_aware_clocks_in_unordered_rows_keep_saved_original_row(self):
        plan = scheduled_request().to_dict()
        frame = plan["prediction_schedule"]["folds"][0]["prediction_frame"]
        row = frame["rows"][1]
        for name in ("knowledge_cutoff", "available_at", "feature_knowledge_cutoff", "simulated_model_available_at"):
            from axiom_engine.core.stock_portfolio import instant
            from datetime import timezone
            row[name] = instant(row[name]).astimezone(timezone.utc).isoformat()
        frame["rows"][0], frame["rows"][1] = frame["rows"][1], frame["rows"][0]
        reseal_schedule(plan)
        # Refresh the serialized reference in the saved trade mapping as well.
        for entry in plan["prediction_schedule"]["trade_schedule"]:
            if entry["trade_session"] in plan["prediction_schedule"]["folds"][0]["fold_spec"]["oos_trade_sessions"]:
                entry["signal_run_ref"] = plan["prediction_schedule"]["folds"][0]["prediction_frame"]["signal_run_ref"]
        reseal_schedule(plan)
        run = run_backtest(BacktestRequest.from_dict(plan))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"mixed.json"; save_backtest_run(run, path)
            self.assertEqual(load_backtest_run(path).payload, run.payload)

    def test_new_pair_counts_reject_float_and_bool(self):
        for section, name, value in (("listing_identity_checks", "checked_rows", 6.0),
                                     ("listing_identity_checks", "paired_rows", True),
                                     ("previous_close_basis_checks", "equal_rows", 66.0),
                                     ("previous_close_basis_checks", "listing_identity_checked_rows", 66.0)):
            plan = scheduled_request().to_dict()
            plan["admission_evidence"][section][name] = value
            reseal_schedule(plan)
            with self.subTest(section=section, name=name), self.assertRaises(ContractError):
                stock_inputs.validate_stock_request(plan)

    def test_public_and_entry_admitted_core_agree_without_rehashing_frame_per_intent(self):
        request = scheduled_request()
        original_hash = Document.from_dict
        identities = []
        def observe(value):
            if value.get("contract") == "axiom.stock_portfolio/2": identities.append(value)
            return original_hash(value)
        with patch.object(Document, "from_dict", side_effect=observe):
            run = run_backtest(request).to_dict()
        self.assertEqual(len(identities), 2)
        self.assertTrue(all("frame_ref" in value and "frame" not in value for value in identities))
        decision = deepcopy(run["decisions"][0]); quotes = decision.pop("reference_prices")
        plan = run["plan"]; frame = plan["prediction_schedule"]["folds"][0]["prediction_frame"]
        context = {"trade_session": CALENDAR[1], "feature_session": CALENDAR[0],
            "decision_time": CALENDAR[1]+"T00:55:00Z", "knowledge_cutoff": frame["rows"][0]["knowledge_cutoff"],
            "feature_knowledge_cutoff": frame["rows"][0]["feature_knowledge_cutoff"], "reference_prices": quotes,
            "account_state_version": decision["expected_account_version"], "supported_security_ids": legacy.SECURITIES,
            "supported_universe_ref": plan["supported_universe_ref"], **{k: plan["profile"][k] for k in
                ("lot_size", "commission_rate", "minimum_commission_minor", "slippage_bps")}}
        public = plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account={"cash_minor": 50000000,
            "positions": {}, "version": decision["expected_account_version"]}, context=context, top_k=3).to_dict()
        self.assertEqual(public, decision)
