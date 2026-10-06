"""The bounded adapters preserve the existing account's exact business facts."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import weakref

from axiom_engine.core.contracts import ContractError, canonical
from axiom_engine.runtime.backtest import BacktestRequest, run_backtest, save_backtest_run
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.stock_stream import run_stock_backtest, stock_run_id
from axiom_engine.runtime.stock_stream_inputs import StockInputSource
from axiom_engine.runtime.stock_stream_outputs import StockResultSink
from axiom_engine.runtime.stock_stream_projection import load_stock_backtest_projection
from axiom_engine.runtime.stock_schedule import stock_prediction_schedule
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from axiom_engine.runtime.evaluation import evaluate_backtest, long_history_evaluation_spec
from axiom_engine.runtime.analysis_evaluation import evaluate_saved_analysis, analysis_evaluation_spec
from axiom_engine.runtime.stock_market import stock_dividend_scope
from axiom_engine.runtime.fill_display import build_fill_display, save_fill_display, load_fill_display
from test_stock_stream_inputs import source_fixture, LIMITS
from test_csi300_runtime import full_request
from test_csi300 import BOARD_IDS, rules_for
from test_analysis_evaluation import RF
from test_stock_schedule import fold, seal
from test_stock_stream_projection import cash_request
from test_fill_display import files as display_files
import test_stocks as legacy_fixture


def execute(root, manifest, block_sessions=2, limits=LIMITS):
    request = BacktestRequest.from_dict(manifest)
    sink = StockResultSink(root / "parts", run_id=stock_run_id(manifest))
    run = run_stock_backtest(request, source=StockInputSource(), sink=sink,
                             block_sessions=block_sessions, limits=limits)
    path = root / "run.json"
    save_backtest_run(run, path)
    projection = load_stock_backtest_projection(path, artifact_reader=lambda ref: Path(ref["manifest_uri"]),
                                                 limits=limits)
    return run.to_dict(), projection, sink


def map_run_ids(value, old, new):
    if type(value) is dict:
        return {key: map_run_ids(item, old, new) for key, item in value.items()}
    if type(value) is list:
        return [map_run_ids(item, old, new) for item in value]
    if type(value) is str and value.startswith(old + ":"):
        return new + value[len(old):]
    return value


class StockStreamTests(unittest.TestCase):
    def test_runtime_retires_block_borrowers_before_next_source_decode(self):
        class Tracked(dict):
            pass

        test = self
        class ProbeSource(StockInputSource):
            probing = False

            def __init__(self):
                super().__init__()
                self.borrowers = []
                self.probes = 0

            def iter_blocks(self, *args, **kwargs):
                self.probing = True
                try:
                    yield from super().iter_blocks(*args, **kwargs)
                finally:
                    self.probing = False

            def _block(self, *args, **kwargs):
                if self.probing:
                    test.assertTrue(all(ref() is None for ref in self.borrowers),
                                    "previous block still live before next source decode")
                    self.probes += 1
                block = super()._block(*args, **kwargs)
                if not self.probing:
                    return block

                def track(value):
                    value = Tracked(value)
                    self.borrowers.append(weakref.ref(value))
                    return value

                return replace(block,
                    market_rows={key: track(row) for key, row in block.market_rows.items()},
                    signals={day: (header, track({key: track(row) for key, row in indexed.items()}))
                             for day, (header, indexed) in block.signals.items()})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, legacy = source_fixture(root)
            source = ProbeSource()
            sink = StockResultSink(root / "parts", run_id=stock_run_id(manifest))
            run = run_stock_backtest(BacktestRequest.from_dict(manifest), source=source,
                sink=sink, block_sessions=1, limits=LIMITS).to_dict()
            self.assertEqual(source.probes, len(manifest["scope"]["calendar"]))
            self.assertTrue(all(ref() is None for ref in source.borrowers))
            old = run_backtest(BacktestRequest.from_dict(legacy)).to_dict()
            self.assertEqual(run["final_account"], old["final_account"])
            self.assertEqual(run["metrics"], old["metrics"])

    def test_total_result_budget_closes_before_seal_and_loads_at_exact_boundary(self):
        for profile_lf in (False, True):
            with self.subTest(profile_lf=profile_lf), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); manifest, _ = source_fixture(root)
                profile = Path(manifest["profile_input"]["artifact"]["manifest_uri"])
                if profile_lf:
                    with profile.open("ab") as saved:
                        saved.write(b"\n")
                output = root / "out"
                wire, _, sink = execute(output, manifest)
                parts_bytes = sink.total_bytes
                total = parts_bytes + (output / "run.json").stat().st_size + profile.stat().st_size
                self.assertEqual((output / "run.json").stat().st_size,
                    len(canonical(wire).encode("utf-8")) + 1)
                for maximum in (total, total - 1, parts_bytes):
                    with self.subTest(maximum=maximum):
                        shutil.rmtree(output)
                        limits = {**LIMITS, "max_result_bytes": maximum}
                        if maximum == total:
                            loaded, projection, _ = execute(output, manifest, limits=limits)
                            self.assertEqual(loaded, wire)
                            self.assertEqual(projection.wire, wire)
                        else:
                            failed_sink = StockResultSink(output / "parts", run_id=stock_run_id(manifest))
                            with patch("axiom_engine.runtime.stock_stream.BacktestRun.from_dict") as seal:
                                with self.assertRaisesRegex(ContractError, "before run seal"):
                                    run_stock_backtest(BacktestRequest.from_dict(manifest), source=StockInputSource(),
                                        sink=failed_sink, block_sessions=2, limits=limits)
                                seal.assert_not_called()
                            self.assertFalse((output / "run.json").exists())

    def test_two_blocks_match_v6_and_other_layouts_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, legacy = source_fixture(root)
            old = run_backtest(BacktestRequest.from_dict(legacy)).to_dict()
            first, projection, sink = execute(root / "two", manifest)
            daily, other, _ = execute(root / "daily", manifest, 1,
                {**LIMITS, "max_result_part_bytes": 6000})
            whole, third, _ = execute(root / "whole", manifest, 4)
            self.assertEqual(first["run_id"], daily["run_id"])
            self.assertEqual(first["run_id"], whole["run_id"])
            for group in ("nav", "positions", "decisions", "orders", "fills", "cash_ledger", "position_ledger"):
                self.assertEqual(projection.rows[group], other.rows[group], group)
                self.assertEqual(projection.rows[group], third.rows[group], group)
                self.assertEqual(projection.rows[group], map_run_ids(old[group], old["run_id"], first["run_id"]), group)
            for name in ("status", "initial_nav_minor", "final_account", "metrics", "committed_sequence", "lifecycle_admission", "stopped"):
                self.assertEqual(first[name], old[name], name)
                self.assertEqual(first[name], daily[name], name)
                self.assertEqual(first[name], whole[name], name)
            self.assertEqual(len(projection.rows["decisions"]), 2)
            self.assertEqual(len(projection.rows["nav"]), 3)
            self.assertEqual(sink.buffered_bytes, 0)
            self.assertNotEqual(first["content_digest"], daily["content_digest"])

    def test_output_drain_preserves_t1_idempotency_and_global_order_numbers(self):
        seen = []
        original = AccountLedger.drain_outputs
        def observe(ledger, counts):
            seen.append((ledger.sequence, deepcopy(ledger.pending), len(ledger._applied), dict(counts)))
            return original(ledger, counts)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root)
            with patch.object(AccountLedger, "drain_outputs", observe):
                wire, projection, _ = execute(root / "out", manifest, 1)
            self.assertTrue(any(pending for _, pending, _, _ in seen))
            self.assertTrue(any(applied for _, _, applied, _ in seen))
            orders = projection.rows["orders"]
            self.assertEqual([o["order_id"] for o in orders],
                [wire["run_id"] + ":order:" + str(i) for i in range(len(orders))])
            self.assertTrue(any(r["reason"] == "SETTLEMENT" for r in projection.rows["position_ledger"]))

    def test_fold_switch_and_top_k_preserve_saved_signal_identity_and_exact_trades(self):
        boards = {security: board for board, security in BOARD_IDS.items()}
        boards.update({"cnstock.000101.SZ.20000101": "SZSE_MAIN", "cnstock.000102.SZ.20000101": "SZSE_MAIN"})
        plan = full_request(rules=rules_for(boards=boards)).to_dict()
        calendar, universe = plan["market_replay"]["calendar"], plan["execution_universe"]
        with patch.object(legacy_fixture, "SECURITIES", universe):
            folds = [fold(calendar, calendar[1:3]), fold(calendar, calendar[3:], reverse=True)]
        plan["prediction_schedule"] = stock_prediction_schedule(folds=folds, calendar=calendar).to_dict()
        proof = plan["admission_evidence"]
        proof.update(prediction_schedule_ref=plan["prediction_schedule"]["schedule_ref"],
            prediction_refs=[{"fold_ref": f["fold_ref"], **{key: f["prediction_frame"][key]
                for key in ("fold_spec_ref", "signal_run_ref", "feature_ref", "model_ref")}} for f in folds])
        plan["admission_evidence"] = seal(proof, "admission_ref")
        plan["admission_ref"] = plan["admission_evidence"]["admission_ref"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root, plan)
            old = run_backtest(BacktestRequest.from_dict(plan)).to_dict()
            wire, projection, _ = execute(root / "out", manifest, 2)
            self.assertEqual(projection.rows["decisions"], old["decisions"])
            self.assertEqual([d["signal_ref"] for d in projection.rows["decisions"]],
                [f["prediction_frame"]["signal_run_ref"] for f in folds])
            self.assertTrue(any(fill["side"] == "SELL" and fill["session"] == calendar[-1]
                                for fill in projection.rows["fills"]))
            self.assertEqual(wire["final_account"], old["final_account"])
            five = deepcopy(manifest)
            five["account_id"] += "-top5"
            five["portfolio_policy"]["top_k"] = 5
            five["request_ref"] = logical_ref(five, "request_ref")
            five_wire, five_projection, _ = execute(root / "five", five, 1)
            self.assertEqual(wire["signal_ref"], five_wire["signal_ref"])
            self.assertEqual(wire["market_ref"], five_wire["market_ref"])
            self.assertEqual(wire["initial_nav_minor"], 50000000)
            self.assertEqual(five_wire["initial_nav_minor"], 50000000)
            self.assertNotEqual(wire["run_id"], five_wire["run_id"])
            self.assertNotEqual([d["selected_security_ids"] for d in projection.rows["decisions"]],
                                [d["selected_security_ids"] for d in five_projection.rows["decisions"]])
            self.assertNotEqual(projection.rows["fills"], five_projection.rows["fills"])

    def test_bad_tail_fails_before_ledger_and_sink(self):
        def damage(batches, member):
            batches[1]["records"][-1]["open"] = -10.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root, full_request(mutate_native=damage).to_dict())
            sink = StockResultSink(root / "parts", run_id=stock_run_id(manifest))
            with patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("ledger started")), \
                 patch.object(sink, "append", side_effect=AssertionError("sink started")):
                with self.assertRaises(ContractError):
                    run_stock_backtest(BacktestRequest.from_dict(manifest), source=StockInputSource(),
                                       sink=sink, block_sessions=2, limits=LIMITS)

    def test_projection_uses_existing_v2_v3_algorithms_without_source_io_or_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, legacy = source_fixture(root)
            old = run_backtest(BacktestRequest.from_dict(legacy))
            wire, projection, _ = execute(root / "out", manifest)
            with patch.object(legacy_fixture, "DAYS", manifest["scope"]["calendar"]):
                benchmark = legacy_fixture.benchmark()
            old_base = evaluate_backtest(old, benchmark=benchmark, spec=long_history_evaluation_spec()).to_dict()
            with patch.object(StockInputSource, "audit", side_effect=AssertionError("source audit")), \
                 patch.object(Path, "open", side_effect=AssertionError("source/result I/O")), \
                 patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("account replay")):
                base = evaluate_backtest(projection, benchmark=benchmark, spec=long_history_evaluation_spec())
                report = evaluate_saved_analysis(projection, base,
                    benchmarks=dict(CSI300=benchmark, SSE_COMPOSITE=None, NASDAQ100=None),
                    spec=analysis_evaluation_spec(risk_free=RF)).to_dict()
            for key in ("series", "monthly_returns", "episode_metrics", "benchmark", "pnl_distribution", "period_metrics"):
                self.assertEqual(base.to_dict()[key], old_base[key], key)
            self.assertEqual(base.to_dict()["input_run_ref"], projection.input_run_ref)
            self.assertEqual(report["input_run_ref"], projection.input_run_ref)
            self.assertIsNone(report["risk_metrics"]["sharpe"]["value"])
            self.assertEqual(report["execution_summary"]["fees_minor"], wire["metrics"]["total_fees_minor"])

    def test_sink_failure_does_not_drain_unconfirmed_rows_or_seal_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root)
            sink = StockResultSink(root / "parts", run_id=stock_run_id(manifest))
            def fail(**kwargs):
                raise ContractError("synthetic sink failure")
                yield
            with patch.object(sink, "append", fail), patch.object(AccountLedger, "drain_outputs", side_effect=AssertionError("drained")):
                with self.assertRaisesRegex(ContractError, "sink failure"):
                    run_stock_backtest(BacktestRequest.from_dict(manifest), source=StockInputSource(),
                                       sink=sink, block_sessions=2, limits=LIMITS)
            self.assertFalse((root / "run.json").exists())

    def test_event_scope_and_evaluation_keep_original_record_ex_and_unknown_pay(self):
        for ex_day in ("2024-01-03", "2024-01-09"):
            with self.subTest(ex=ex_day), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, legacy = source_fixture(root, cash_request(ex=ex_day).to_dict())
                old = run_backtest(BacktestRequest.from_dict(legacy))
                wire, projection, _ = execute(root / "out", manifest, 1)
                self.assertEqual(wire["final_account"], old.to_dict()["final_account"])
                self.assertEqual(projection.rows["cash_ledger"],
                    map_run_ids(old.to_dict()["cash_ledger"], old.to_dict()["run_id"], wire["run_id"]))
                with patch.object(legacy_fixture, "DAYS", manifest["scope"]["calendar"]):
                    benchmark = legacy_fixture.benchmark()
                old_base = evaluate_backtest(old, benchmark=benchmark, spec=long_history_evaluation_spec(),
                    dividend_scope=stock_dividend_scope(old)).to_dict()
                with patch.object(Path, "open", side_effect=AssertionError("large parent I/O")), \
                     patch.object(StockInputSource, "audit", side_effect=AssertionError("source audit")):
                    scope = stock_dividend_scope(projection)
                    base = evaluate_backtest(projection, benchmark=benchmark, spec=long_history_evaluation_spec(),
                                             dividend_scope=scope).to_dict()
                scope_wire = scope.to_dict()
                self.assertEqual(scope_wire["contract_version"], "dividend_scope_v3")
                self.assertEqual(scope_wire["account_events_ref"], wire["account_events_ref"])
                self.assertEqual(scope_wire["coverage"], "observed_records_only")
                self.assertEqual(scope_wire["actions"], stock_dividend_scope(old).to_dict()["actions"])
                self.assertIsNone(scope_wire["actions"][0]["pay_session"])
                for key in ("series", "episode_metrics", "pnl_distribution", "period_metrics"):
                    self.assertEqual(base[key], old_base[key], key)

    def test_saved_fill_display_keeps_native_prices_and_v7_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, legacy = source_fixture(root)
            old = run_backtest(BacktestRequest.from_dict(legacy))
            wire, projection, _ = execute(root / "out", manifest)
            display = display_files(root / "display", old)
            original = old.to_dict()
            old_display = build_fill_display(old, display=display).to_dict()
            with patch.object(Path, "open", side_effect=AssertionError("large parent I/O")), \
                 patch.object(StockInputSource, "audit", side_effect=AssertionError("source audit")), \
                 patch("axiom_engine.runtime.backtest.AccountLedger", side_effect=AssertionError("account replay")):
                report = build_fill_display(projection, display=display)
            saved = report.to_dict()
            self.assertEqual(saved["status"], "COMPLETE")
            self.assertEqual(saved["input_run_ref"], projection.input_run_ref)
            self.assertEqual(saved["coordinates"], map_run_ids(old_display["coordinates"],
                original["run_id"], wire["run_id"]))
            self.assertEqual(saved["fills"], projection.rows["fills"])
            self.assertEqual(saved["consumed_input"]["account_unit_contract"]["backtest_contract_version"],
                             "backtest_run_v7")
            self.assertEqual(old.to_dict(), original)
            output = root / "fill-display.json"
            save_fill_display(report, output)
            with patch("axiom_engine.runtime.fill_display.read_review_display", side_effect=AssertionError("Data")), \
                 patch("axiom_engine.runtime.fill_display._coordinate", side_effect=AssertionError("recompute")):
                self.assertEqual(load_fill_display(output).payload, report.payload)

    def test_pending_raw_history_is_guarded_before_addition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root)
            limits = {**LIMITS, "max_result_buffer_bytes": 6000, "max_result_part_bytes": 6000}
            sink = StockResultSink(root / "parts", run_id=stock_run_id(manifest))
            with patch.object(AccountLedger, "drain_outputs", side_effect=AssertionError("unconfirmed drain")):
                with self.assertRaisesRegex(ContractError, "before history addition"):
                    run_stock_backtest(BacktestRequest.from_dict(manifest), source=StockInputSource(),
                                       sink=sink, block_sessions=1, limits=limits)
            self.assertEqual(sink.total_bytes, 0)
            self.assertFalse((root / "run.json").exists())

    def test_blocked_session_retains_settlement_without_inventing_nav(self):
        star = BOARD_IDS["SSE_STAR"]
        def top_star(frame):
            for row in frame["rows"]:
                if row["security_id"] == star: row["score"] = 100
        def factor_change(batches, member):
            for row in batches[3]["records"]:
                if row["session"] >= "2024-01-03" and row["security_id"] == star:
                    row["factor"] = 1.1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, legacy = source_fixture(root, full_request(k=1, mutate_frame=top_star, mutate_native=factor_change).to_dict())
            old = run_backtest(BacktestRequest.from_dict(legacy)).to_dict()
            wire, projection, _ = execute(root / "out", manifest)
            self.assertEqual(wire["status"], "BLOCKED")
            self.assertEqual(wire["stopped"], old["stopped"])
            self.assertEqual(projection.rows["nav"], old["nav"])
            self.assertEqual(projection.rows["position_ledger"], map_run_ids(old["position_ledger"], old["run_id"], wire["run_id"]))
            self.assertEqual(projection.rows["session_phases"][-1]["phase"], "STOPPED_BEFORE_NAV")
            self.assertGreater(wire["committed_sequence"], projection.rows["nav"][-1]["committed_sequence"])


if __name__ == "__main__":
    unittest.main()
