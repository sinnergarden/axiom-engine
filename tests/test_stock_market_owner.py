"""One admitted market, distinct saved signals, unchanged account executor."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document, canonical
from axiom_engine.runtime import (BacktestRequest, StockInputSource, StockMarketSpec,
    admit_stock_market_inputs, bind_stock_prediction_inputs)
from axiom_engine.runtime import stock_stream_inputs
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_stock_owned_inputs import request_for, run_owned
from test_stock_stream_inputs import LIMITS, artifact_file
from test_stock_stream import execute


def seal_request(plan):
    for part, field in (("market_input", "market_ref"), ("prediction_input", "prediction_ref")):
        plan[part][field] = logical_ref(plan[part], field)
    plan["request_ref"] = logical_ref(plan, "request_ref")
    return plan


def different_signal(root, original, *, change=None):
    root.mkdir()
    plan = deepcopy(original); plan["account_id"] += "-different-signal"
    for i, item in enumerate(plan["prediction_input"]["frames"]):
        old_model = item["model_ref"]
        model = Document(Path(item["model_metadata_artifact"]["manifest_uri"]).read_text()).to_dict()
        model["parameters"]["seed"] += 1
        model.pop("model_ref"); model["model_ref"] = Document.from_dict(model).identity
        item["model_ref"] = model["model_ref"]
        item["model_metadata_artifact"] = artifact_file(root, "different-model-"+str(i), model)
        frame = Document(Path(item["prediction_artifact"]["manifest_uri"]).read_text()).to_dict()
        frame["model_ref"] = model["model_ref"]
        for row in frame["rows"]:
            if row["score"] is not None:
                row["score"] = -row["score"]
            row["source_refs"] = [model["model_ref"] if ref == old_model else ref for ref in row["source_refs"]]
        if change:
            change(frame)
        frame.pop("signal_run_ref"); frame["signal_run_ref"] = Document.from_dict(frame).identity
        item["signal_run_ref"] = frame["signal_run_ref"]
        item["prediction_artifact"] = artifact_file(root, "different-prediction-"+str(i), frame)
    return seal_request(plan)


def new_basis(root, original, mutate=None):
    root.mkdir(); plan = deepcopy(original)
    plan["market_input"]["model_snapshot_id"] = "synthetic-second-model-snapshot"
    for i, entry in enumerate(plan["market_input"]["native_inputs"]):
        if entry["role"] != "prediction_basis":
            continue
        value = Document(Path(entry["artifact"]["manifest_uri"]).read_text()).to_dict()
        value["context"]["snapshot_id"] = plan["market_input"]["model_snapshot_id"]
        if mutate:
            mutate(value)
        entry["artifact"] = artifact_file(root, "second-basis-"+str(i), value)
        entry["native_ref"] = Document.from_dict(value).identity
    return seal_request(plan)


def market_for(plan, **kwargs):
    return admit_stock_market_inputs(StockMarketSpec.from_request(BacktestRequest.from_dict(plan)),
        source=kwargs.pop("source", StockInputSource()), block_sessions=2, limits=LIMITS,
        max_market_bytes=kwargs.pop("max_market_bytes", 2_000_000), **kwargs)


def bind(market, plan, **kwargs):
    return bind_stock_prediction_inputs(market, BacktestRequest.from_dict(plan),
        source=kwargs.pop("source", StockInputSource()), limits=kwargs.pop("limits", LIMITS),
        max_signal_bytes=kwargs.pop("max_signal_bytes", 2_000_000), **kwargs)


class StockMarketOwnerTests(unittest.TestCase):
    def test_market_before_signal_and_two_distinct_signals_scan_market_once_and_match_full_oracles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = request_for(root); second = different_signal(root/"second", first)
            oracles = [execute(root/("out-"+str(i)), p)[:2] for i, p in enumerate((first, second))]
            for i in range(2):
                shutil.rmtree(root/("out-"+str(i)))
            scans = []; original_init = stock_stream_inputs._CanonicalIndex.__init__
            def init(index, artifact, *args, **kwargs):
                scans.append(str(artifact["manifest_uri"]) if artifact else str(kwargs.get("descriptor_path")))
                return original_init(index, artifact, *args, **kwargs)
            spec_wire = StockMarketSpec.from_request(BacktestRequest.from_dict(first)).to_dict()
            self.assertNotIn("prediction_input", spec_wire)
            self.assertNotIn("model_snapshot_id", spec_wire["market_input"])
            # All Signal paths can be unavailable during market admission.
            saved = []
            for artifact in first["prediction_input"]["frames"]:
                path = Path(artifact["prediction_artifact"]["manifest_uri"])
                saved.append((path, path.read_bytes())); path.unlink()
            with patch.object(stock_stream_inputs._CanonicalIndex, "__init__", init):
                market = admit_stock_market_inputs(StockMarketSpec.from_dict(spec_wire), source=StockInputSource(),
                    block_sessions=2, limits=LIMITS, max_market_bytes=2_000_000)
                for path, raw in saved:
                    path.write_bytes(raw)
                with market:
                    market_paths = {d["artifact"]["manifest_uri"] for d in first["market_input"]["native_inputs"] if d["role"] == "execution"}
                    market_paths.add(first["profile_input"]["artifact"]["manifest_uri"])
                    self.assertEqual(len(scans), len(market_paths))
                    self.assertEqual(set(scans), market_paths)
                    market_operations = market.statistics["source_operations"]
                    # Remove originals: both the shared basis and execution facts
                    # must now come from retained admitted native views.
                    for path in market_paths-{first["profile_input"]["artifact"]["manifest_uri"]}:
                        Path(path).unlink()
                    with ExitStack() as stack:
                        with patch("axiom_engine.core.stock_rules.validate_execution_rules", side_effect=AssertionError("shared rules revalidated")):
                            a = stack.enter_context(bind(market, first))
                            b = stack.enter_context(bind(market, second))
                        stats = market.statistics
                        self.assertEqual(stats["source_admissions"], 1)
                        self.assertEqual((stats["basis_pairings"], stats["basis_reuses"], stats["signal_bindings"]), (1, 1, 2))
                        self.assertEqual(stats["source_operations"], market_operations)
                        self.assertTrue(all(scans.count(p) == 1 for p in market_paths))
                        self.assertLess(a.statistics["owned_bytes"]+b.statistics["owned_bytes"], market.statistics["owned_bytes"])
                        with a.execution_scope():
                            with self.assertRaisesRegex(ContractError, "sequential"):
                                run_owned(root/"busy", second, b)
                        with self.assertRaisesRegex(ContractError, "Signal borrows"):
                            market.close()
                        runs = []
                        for i, (plan, inputs) in enumerate(((first, a), (second, b))):
                            wire, projection = run_owned(root/("out-"+str(i)), plan, inputs)
                            self.assertEqual(wire, oracles[i][0])
                            self.assertEqual(projection.rows, oracles[i][1].rows)
                            runs.append(wire)
                        self.assertNotEqual(runs[0]["signal_ref"], runs[1]["signal_ref"])
                        self.assertNotEqual(runs[0]["run_id"], runs[1]["run_id"])
                        self.assertNotEqual(runs[0]["final_account"], runs[1]["final_account"])
                    self.assertEqual(market.statistics["signal_borrows"], 0)

    def test_new_basis_pairs_retained_six_prices_factor_member_and_provenance_then_reuses_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = request_for(root)
            good = new_basis(root/"good", first)
            oracle, projection, _ = execute(root/"out", good)
            shutil.rmtree(root/"out")
            changes = {
                "high": lambda v: v["records"][0].update(high=123.0) if v["context"]["domain"] == "market_daily" else None,
                "factor": lambda v: v["records"][0].update(factor=2.0) if v["context"]["domain"] == "adjustment_factors" else None,
                "member": lambda v: v["records"][0].update(is_member=False) if v["context"]["domain"] == "universe_membership" else None,
                "provenance": lambda v: v["field_meta"]["close"]["by_key"][0].update(revision_id="different") if v["context"]["domain"] == "market_daily" else None,
                "unit": lambda v: v["field_meta"]["high"].update(unit="USD/share") if v["context"]["domain"] == "market_daily" else None}
            bad = [new_basis(root/name, first, change) for name, change in changes.items()]
            with market_for(first) as market:
                for path in {i["artifact"]["manifest_uri"] for i in first["market_input"]["native_inputs"] if i["role"] == "execution"}:
                    Path(path).unlink()
                for plan in bad:
                    with patch("axiom_engine.runtime.backtest.AccountLedger") as ledger:
                        with self.assertRaises(ContractError):
                            bind(market, plan)
                        ledger.assert_not_called()
                    self.assertEqual(market.statistics["signal_borrows"], 0)
                    self.assertEqual(market.statistics["basis_proofs"], 0)
                with bind(market, good) as inputs:
                    wire, actual = run_owned(root/"out", good, inputs)
                    self.assertEqual(wire, oracle); self.assertEqual(actual.rows, projection.rows)
                for entry in good["market_input"]["native_inputs"]:
                    if entry["role"] == "prediction_basis":
                        Path(entry["artifact"]["manifest_uri"]).unlink()
                with bind(market, good):
                    self.assertEqual(market.statistics["basis_reuses"], 1)

    def test_bad_signal_or_market_binding_and_quotas_release_only_failed_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = request_for(root)
            bad_member = different_signal(root/"bad-member", first, change=lambda f: f["rows"][0].update(member=False))
            bad_clock = different_signal(root/"bad-clock", first, change=lambda f: f["rows"][0].update(
                available_at=f["rows"][0]["session"]+"T22:00:00+08:00"))
            with market_for(first) as market:
                for plan in (bad_member, bad_clock):
                    with self.assertRaises(ContractError):
                        bind(market, plan)
                    self.assertEqual(market.statistics["signal_borrows"], 0)
                for mutate in (lambda p: p.update(clock_policy="bad"),
                    lambda p: p["market_input"].update(execution_snapshot_id="different"),
                    lambda p: p["scope"]["execution_universe"].reverse()):
                    plan = deepcopy(first); mutate(plan); seal_request(plan)
                    with self.assertRaises(ContractError):
                        bind(market, plan)
                opened = []; original = tempfile.TemporaryFile
                def temporary(*args, **kwargs):
                    value = original(*args, **kwargs); opened.append(value); return value
                with patch("axiom_engine.runtime.stock_market_owner.tempfile.TemporaryFile", temporary):
                    with self.assertRaisesRegex(ContractError, "max_signal_bytes"):
                        bind(market, first, max_signal_bytes=20_000)
                self.assertTrue(opened); self.assertTrue(all(v.closed for v in opened))
                self.assertEqual(market.statistics["signal_borrows"], 0)
                with bind(market, first) as inputs:
                    wire, _ = run_owned(root/"valid", first, inputs)
                    self.assertEqual(wire["status"], "COMPLETE")
                for value in (True, 0, -1):
                    with self.assertRaises(ContractError):
                        bind(market, first, max_signal_bytes=value)
            market.close()
            with self.assertRaisesRegex(ContractError, "closed"):
                bind(market, first)

    def test_started_constructor_failure_closes_signal_store_and_keeps_market_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = request_for(Path(tmp)); opened = []; original = tempfile.TemporaryFile
            def temporary(*args, **kwargs):
                value = original(*args, **kwargs); opened.append(value); return value
            with market_for(first) as market:
                with patch("axiom_engine.runtime.stock_market_owner.tempfile.TemporaryFile", temporary), \
                        patch("axiom_engine.runtime.stock_owned_inputs.AdmittedStockInputs", side_effect=MemoryError("synthetic allocation failure")):
                    # Patch only the constructor used inside _bind; the public
                    # source-type guard still needs the real class.
                    from axiom_engine.runtime.stock_market_owner import _bind
                    from axiom_engine.runtime.stock_stream_contracts import validate_manifest
                    from axiom_engine.runtime.stock_market_owner import _release_source
                    source = StockInputSource()
                    try:
                        with market._scope(), self.assertRaises(MemoryError):
                            _bind(market, validate_manifest(BacktestRequest.from_dict(first)), source, LIMITS, 2_000_000)
                    finally:
                        _release_source(source)
                self.assertTrue(opened); self.assertTrue(all(v.closed for v in opened))
                self.assertEqual(market.statistics["signal_borrows"], 0)
                self.assertEqual(market.statistics["basis_proofs"], 0)
                with bind(market, first):
                    self.assertEqual(market.statistics["signal_bindings"], 1)

    def test_market_quota_accounts_for_retained_proof_and_rejects_before_signal_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = request_for(Path(tmp))
            with market_for(first) as market:
                exact = market.statistics["owned_bytes"]
            with market_for(first, max_market_bytes=exact) as market:
                with patch("axiom_engine.runtime.stock_market_owner.tempfile.TemporaryFile") as temporary:
                    with self.assertRaisesRegex(ContractError, "max_market_bytes"):
                        bind(market, first)
                    temporary.assert_not_called()
                self.assertEqual(market.statistics["signal_borrows"], 0)
                self.assertEqual(market.statistics["owned_bytes"], exact)

    @unittest.skipUnless(hasattr(os, "fork"), "process ownership")
    def test_market_rejects_fork_before_lock_or_store_access(self):
        class ForbiddenLock:
            def acquire(self, **kwargs):
                raise AssertionError("child touched inherited market lock")
        with tempfile.TemporaryDirectory() as tmp:
            first = request_for(Path(tmp))
            with market_for(first) as market:
                read_fd, write_fd = os.pipe(); child = os.fork()
                if child == 0:
                    os.close(read_fd); market._lock = ForbiddenLock(); results = []
                    for action in (lambda: market.statistics, lambda: market._scope().__enter__(), market.close,
                                   lambda: bind(market, first), market.__enter__):
                        try:
                            action(); results.append(False)
                        except ContractError as exc:
                            results.append("creating process" in str(exc))
                        except Exception:
                            results.append(False)
                    os.write(write_fd, canonical({"results": results}).encode()); os.close(write_fd); os._exit(0)
                os.close(write_fd); raw = os.read(read_fd, 4096); os.close(read_fd)
                _, status = os.waitpid(child, 0)
                self.assertEqual(status, 0)
                self.assertTrue(all(Document(raw.decode()).to_dict()["results"]))
                with bind(market, first):
                    self.assertEqual(market.statistics["signal_borrows"], 1)


if __name__ == "__main__":
    unittest.main()
