"""Trade §11.2 hand arithmetic and saved-only v1/v2 compatibility."""
from datetime import date, timedelta
from decimal import Context, Decimal, Inexact, ROUND_DOWN, getcontext, localcontext, setcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BenchmarkSeries, EvaluationReport, daily_evaluation_spec,
    long_history_evaluation_spec, evaluate_backtest, load_backtest_evaluation, save_backtest_evaluation)
from test_evaluation import saved, trade, benchmark, missing_benchmark, rehash, scope
from test_backtest import DAYS, REF


def marked_wealth(anchor="2020-01-01", end="2022-01-01", final=1210000):
    """Synthetic saved marks on an existing ledger's fully invested position."""
    first = (date.fromisoformat(anchor) + timedelta(days=1)).isoformat()
    run = saved([trade(first, "BUY", "A", 1000, "10")], days=[anchor, first, end])
    wire = run.to_dict()
    wire["nav"][-1].update(nav_minor=final, market_value_minor=final,
                          nav_index=str(Decimal(final) / wire["initial_nav_minor"]))
    wire["positions"][-1]["market_value_minor"] = final
    return rehash(wire)


def benchmark_end(run, value):
    wire = benchmark(run).to_dict()
    batch = wire["source_evidence"][0]["batch"]
    batch["records"][-1]["close"] = value
    ref = Document.from_dict(batch).identity
    wire["source_evidence"][0]["reference"] = ref
    wire["source_refs"] = [ref]
    for row in wire["rows"]:
        row["source_refs"] = [ref]
    wire["rows"][-1]["close"] = str(value)
    return BenchmarkSeries.from_dict(wire)


def evaluate(run, bench=None):
    return evaluate_backtest(run, benchmark=bench or benchmark(run), spec=long_history_evaluation_spec())


def report_with_hashes(wire):
    wire.pop("content_digest", None)
    wire["evaluation_ref"] = Document.from_dict({k: wire[k] for k in (
        "input_run_ref", "spec_ref", "benchmark_ref", "dividend_scope_ref", "evaluation_version", "implementation_ref")}).identity
    wire["content_digest"] = Document.from_dict(wire).identity
    return EvaluationReport.from_dict(wire)


class LongEvaluationTests(unittest.TestCase):
    def test_v2_factory_changes_only_version_and_annualization(self):
        daily = daily_evaluation_spec().to_dict()
        long = long_history_evaluation_spec().to_dict()
        self.assertEqual(long.pop("contract_version"), "evaluation_spec_v2")
        self.assertEqual(long.pop("annualization"), dict(method="geometric_cagr",
            day_count="actual_actual_calendar_year_split", interval="start_inclusive_end_exclusive",
            start_anchor="previous_session_initial_nav", end_anchor="last_saved_nav_session", minimum_year_fraction="1"))
        daily.pop("contract_version"); daily.pop("annualization")
        self.assertEqual(long, daily)

    def test_leap_year_split_and_four_hand_calculated_wealth_cases(self):
        for terminal, expected in ((1210000, "0.1"), (640000, "-0.2"), (1000000, "0"), (0, "-1")):
            with self.subTest(terminal=terminal):
                w = evaluate(marked_wealth(final=terminal)).to_dict()
                period = w["period_metrics"]
                self.assertEqual(period["window"], dict(anchor_session="2020-01-01", end_session="2022-01-01",
                    elapsed_calendar_days=731, day_count="actual_actual_calendar_year_split",
                    year_segments=[dict(year=2020, days=366, year_days=366), dict(year=2021, days=365, year_days=365)],
                    year_fraction="2"))
                self.assertEqual(period["account"]["cagr_status"], "AVAILABLE")
                self.assertEqual(Decimal(period["account"]["cagr"]), Decimal(expected))
                self.assertIsNone(period["account"]["cagr_reason"])
                self.assertEqual(period["account"]["final_nav_minor"], terminal)
                self.assertEqual(Decimal(period["account"]["max_drawdown"]), min(Decimal(p["drawdown"]) for p in w["series"]))
                self.assertEqual(w["status"], "PARTIAL")
                self.assertTrue(all(m["status"] == "PARTIAL" for m in w["monthly_returns"]))
                self.assertEqual(w["episodes"][0]["status"], "OPEN")
                self.assertFalse(any("No annualization" in item for item in w["limitations"]))

    def test_natural_anniversary_can_be_short_and_zero_is_not_annualized(self):
        run = marked_wealth("2020-07-01", "2021-07-01", final=0)
        period = evaluate(run).to_dict()["period_metrics"]
        self.assertEqual(period["window"]["elapsed_calendar_days"], 365)
        self.assertEqual(period["window"]["year_segments"], [dict(year=2020, days=184, year_days=366),
                                                              dict(year=2021, days=181, year_days=365)])
        with localcontext(Context(prec=40)):
            self.assertEqual(Decimal(period["window"]["year_fraction"]), Decimal(184)/366 + Decimal(181)/365)
        self.assertLess(Decimal(period["window"]["year_fraction"]), 1)
        self.assertEqual(period["account"]["cagr_status"], "INSUFFICIENT_SPAN")
        self.assertIsNone(period["account"]["cagr"])
        self.assertEqual(period["account"]["total_return"], "-1")

    def test_missing_endpoints_precede_short_span_and_are_not_filled(self):
        run = marked_wealth("2020-07-01", "2021-07-01")
        for index in (0, -1):
            with self.subTest(index=index):
                period = evaluate(run, missing_benchmark(run, index)).to_dict()["period_metrics"]
                self.assertEqual(period["benchmark"]["cagr_status"], "MISSING_BOUNDARY")
                self.assertIsNone(period["benchmark"]["cagr"])
                self.assertIsNone(period["benchmark"]["total_return"])
                self.assertEqual(period["account"]["cagr_status"], "INSUFFICIENT_SPAN")

    def test_benchmark_hand_return_and_middle_gap_keep_cagr_separate_from_drawdown(self):
        run = marked_wealth()
        period = evaluate(run, benchmark_end(run, 121)).to_dict()["period_metrics"]
        self.assertEqual(Decimal(period["benchmark"]["cagr"]), Decimal("0.1"))
        self.assertEqual(Decimal(period["benchmark"]["total_return"]), Decimal("0.21"))
        self.assertEqual(Decimal(period["benchmark"]["max_drawdown"]), 0)
        w = evaluate(run, missing_benchmark(run, 1)).to_dict()
        self.assertEqual(w["period_metrics"]["benchmark"]["cagr_status"], "AVAILABLE")
        self.assertIsNone(w["period_metrics"]["benchmark"]["max_drawdown"])
        self.assertEqual(w["benchmark"]["status"], "PARTIAL")
        self.assertEqual(w["status"], "PARTIAL")

    def test_initial_peak_includes_fee_and_short_span_retains_total_and_drawdown(self):
        run = saved([trade(DAYS[1], "BUY", "A", 100, "10", 500)])
        period = evaluate(run).to_dict()["period_metrics"]
        self.assertEqual(period["account"]["cagr_status"], "INSUFFICIENT_SPAN")
        self.assertEqual(period["account"]["total_return"], "-0.0005")
        self.assertEqual(period["account"]["max_drawdown"], "-0.0005")

    def test_negative_nav_external_flow_and_absent_prior_anchor_reject(self):
        for change in ("negative", "initial_zero", "external_flow", "no_anchor"):
            with self.subTest(change=change):
                wire = marked_wealth().to_dict()
                if change == "negative":
                    wire["nav"][-1]["nav_minor"] = -1
                elif change == "initial_zero":
                    wire["initial_nav_minor"] = 0
                elif change == "external_flow":
                    wire["cash_ledger"].append(dict(reason="DEPOSIT"))
                else:
                    wire["plan"]["market_replay"]["calendar"].pop(0)
                    wire["run_id"] = Document.from_dict(dict(request=wire["plan"], core=wire["core_version"],
                        runtime=wire["runtime_version"], implementation_ref=wire["implementation_ref"])).identity
                run = rehash(wire)
                with self.assertRaises(ContractError):
                    evaluate(run, benchmark(marked_wealth()))

    def test_decimal_context_repeat_and_no_account_execution(self):
        run = marked_wealth(final=1234567)
        expected = evaluate(run).payload
        before = getcontext().copy()
        try:
            getcontext().prec = 6; getcontext().rounding = ROUND_DOWN; getcontext().traps[Inexact] = True
            with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("account replay")):
                self.assertEqual(evaluate(run).payload, expected)
                self.assertEqual(evaluate(run).payload, expected)
            self.assertEqual((getcontext().prec, getcontext().rounding, getcontext().traps[Inexact]), (6, ROUND_DOWN, True))
        finally:
            setcontext(before)

    def test_both_stored_versions_load_without_metrics_or_current_implementation(self):
        run = marked_wealth()
        v1 = evaluate_backtest(run, benchmark=benchmark(run), spec=daily_evaluation_spec(), dividend_scope=scope(run))
        old = v1.to_dict(); old["implementation_ref"] = REF
        v1 = report_with_hashes(old)
        v2 = evaluate(run)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/"v1.json", Path(directory)/"v2.json"]
            for path, report in zip(paths, (v1, v2)):
                save_backtest_evaluation(report, path)
            before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
            with patch("axiom_engine.runtime.evaluation.evaluate_backtest", side_effect=AssertionError("reader computes")), \
                 patch("axiom_engine.runtime.evaluation.period_metrics", side_effect=AssertionError("reader annualizes")), \
                 patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("reader replays")):
                self.assertEqual(load_backtest_evaluation(paths[0]).payload, v1.payload)
                self.assertNotIn("period_metrics", load_backtest_evaluation(paths[0]).to_dict())
                self.assertEqual(load_backtest_evaluation(paths[1]).payload, v2.payload)
            self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
            with self.assertRaises(ContractError):
                save_backtest_evaluation(v2, paths[0])

    def test_loader_rejects_rehashed_version_pair_and_period_contract_mismatches(self):
        report = evaluate(marked_wealth())
        for change in ("spec", "evaluation_version", "period_absent", "status", "nonfinite"):
            with self.subTest(change=change):
                wire = report.to_dict()
                if change == "spec":
                    wire["spec"] = daily_evaluation_spec().to_dict()
                    wire["spec_ref"] = daily_evaluation_spec().identity
                elif change == "evaluation_version":
                    wire["evaluation_version"] = "axiom.evaluation/1"
                elif change == "period_absent":
                    del wire["period_metrics"]
                elif change == "status":
                    wire["period_metrics"]["account"]["cagr_status"] = "UNKNOWN"
                else:
                    wire["period_metrics"]["account"]["cagr"] = "NaN"
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"tampered.json"
                    path.write_text(report_with_hashes(wire).payload)
                    with self.assertRaises(ContractError):
                        load_backtest_evaluation(path)


if __name__ == "__main__":
    unittest.main()
