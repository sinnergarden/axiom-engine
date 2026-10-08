"""One original byte admission, isolated accounts, and unchanged exact oracle."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.core.contracts import ContractError
from axiom_engine.runtime import (BacktestRequest, StockInputSource, StockResultSink,
    admit_stock_inputs, run_stock_backtest, stock_run_id, save_backtest_run,
    load_stock_backtest_projection)
from axiom_engine.runtime.stock_stream_contracts import logical_ref, read_budget
from test_stock_stream_inputs import source_fixture, LIMITS
from test_stock_stream import execute
from test_csi300_runtime import full_request
from test_csi300 import BOARD_IDS, rules_for


def request_for(root):
    boards = {security: board for board, security in BOARD_IDS.items()}
    boards.update({"cnstock.000101.SZ.20000101": "SZSE_MAIN", "cnstock.000102.SZ.20000101": "SZSE_MAIN"})
    return source_fixture(root, full_request(rules=rules_for(boards=boards)).to_dict())[0]


def imported(manifest, source=None, **kwargs):
    return admit_stock_inputs(BacktestRequest.from_dict(manifest), source=source or StockInputSource(),
        block_sessions=2, limits=LIMITS, max_owned_bytes=kwargs.pop("max_owned_bytes", 2_000_000), **kwargs)


def run_owned(root, manifest, inputs, limits=LIMITS, *, block_sessions=2):
    sink = StockResultSink(root/"parts", run_id=stock_run_id(manifest))
    result = run_stock_backtest(BacktestRequest.from_dict(manifest), source=inputs,
        sink=sink, block_sessions=block_sessions, limits=limits)
    save_backtest_run(result, root/"run.json")
    projection = load_stock_backtest_projection(root/"run.json",
        artifact_reader=lambda ref: Path(ref["manifest_uri"]), limits=limits)
    return result.to_dict(), projection


class StockOwnedInputTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "fork"), "fork process-ownership counterexample")
    def test_fork_rejects_every_entry_before_lock_or_shared_cursor_and_parent_still_reads(self):
        class ForbiddenLock:
            def acquire(self, **kwargs):
                raise AssertionError("child touched lock before PID rejection")
            def release(self):
                raise AssertionError("child released lock before PID rejection")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = request_for(Path(tmp))
            with imported(manifest) as inputs:
                store = inputs._AdmittedStockInputs__store
                fd = store.fileno()
                before = os.lseek(fd, 0, os.SEEK_CUR)
                counters = inputs.statistics
                scope = inputs.execution_scope(); scope.__enter__()
                inherited = inputs.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS))
                read_fd, write_fd = os.pipe()
                child = os.fork()
                if child == 0:
                    os.close(read_fd)
                    inputs._AdmittedStockInputs__lock = ForbiddenLock()
                    result = {}
                    actions = {
                        "inventory": lambda: inputs.inventory(manifest),
                        "statistics": lambda: inputs.statistics,
                        "scope_enter": lambda: inputs.execution_scope().__enter__(),
                        "inherited_scope_exit": lambda: scope.__exit__(None, None, None),
                        "audit": lambda: inputs.audit(manifest, block_sessions=2,
                            read_budget=read_budget(LIMITS), limits=LIMITS, implementation_ref=IMPLEMENTATION_REF),
                        "iter_blocks": lambda: inputs.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)),
                        "inherited_iterator": lambda: next(inherited),
                        "context_enter": inputs.__enter__, "context_exit": lambda: inputs.__exit__(None, None, None),
                        "close": inputs.close}
                    for name, action in actions.items():
                        try:
                            action()
                            result[name] = "ACCEPTED"
                        except ContractError as exc:
                            result[name] = "PID_REJECTED" if "creating process" in str(exc) else str(exc)
                        except Exception as exc:
                            result[name] = repr(exc)
                    result["cursor_unchanged"] = os.lseek(fd, 0, os.SEEK_CUR) == before
                    os.write(write_fd, json.dumps(result).encode())
                    os.close(write_fd)
                    os._exit(0)
                os.close(write_fd)
                try:
                    raw = bytearray()
                    while piece := os.read(read_fd, 4096):
                        raw.extend(piece)
                    _, status = os.waitpid(child, 0)
                    self.assertEqual(status, 0)
                    result = json.loads(raw)
                    self.assertTrue(result.pop("cursor_unchanged"))
                    self.assertEqual(set(result.values()), {"PID_REJECTED"})
                    self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), before)
                    self.assertEqual(inputs.statistics, counters)
                    audit = inputs.audit(manifest, block_sessions=2, read_budget=read_budget(LIMITS),
                        limits=LIMITS, implementation_ref=IMPLEMENTATION_REF)
                    self.assertEqual(audit.receipt["request_ref"], manifest["request_ref"])
                    blocks = inputs.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS))
                    try:
                        self.assertEqual(next(blocks).sessions, tuple(manifest["scope"]["calendar"][:2]))
                    finally:
                        blocks.close()
                finally:
                    os.close(read_fd)
                    inherited.close()
                    scope.__exit__(None, None, None)

    def test_one_byte_read_limit_preserves_owned_output_and_serialization_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            oracle, _, _ = execute(root/"out", manifest); shutil.rmtree(root/"out")
            limits = {**LIMITS, "max_read_bytes": 1}
            with admit_stock_inputs(BacktestRequest.from_dict(manifest), source=StockInputSource(),
                    block_sessions=2, limits=limits, max_owned_bytes=2_000_000) as inputs:
                wire, _ = run_owned(root/"out", manifest, inputs, limits)
                self.assertEqual(wire, oracle)
                self.assertLessEqual(inputs.statistics["source_decoded_bytes_peak"], limits["max_block_bytes"])

    def test_top3_top5_and_cash_match_complete_original_wire_with_one_source_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = request_for(root)
            five = deepcopy(first); five["account_id"] += "-five"; five["portfolio_policy"]["top_k"] = 5
            cash = deepcopy(first); cash["account_id"] += "-cash"; cash["initial_account"]["cash_minor"] //= 2
            cases = [first, five, cash]
            for item in cases:
                item["request_ref"] = logical_ref(item, "request_ref")
            expected = []
            for i, item in enumerate(cases):
                wire, projection, _ = execute(root/str(i), item)
                expected.append((wire, projection.rows)); shutil.rmtree(root/str(i))
            source = StockInputSource()
            with patch.object(source, "audit", wraps=source.audit) as audit, \
                    patch.object(source, "iter_blocks", wraps=source.iter_blocks) as blocks:
                inputs = imported(first, source)
                self.assertEqual(audit.call_count, 0); self.assertEqual(blocks.call_count, 0)
                self.assertEqual(inputs._AdmittedStockInputs__market.statistics["source_admissions"], 1)
                self.assertEqual(source._indexes, {})
                before_decode = inputs.statistics["owned_record_decodes"]
                with inputs, patch.object(source, "audit", side_effect=AssertionError("original source reopened")), \
                        patch("axiom_engine.runtime.stock_stream_inputs._CanonicalIndex", side_effect=AssertionError("scanner reused")):
                    for i, item in enumerate(cases):
                        wire, projection = run_owned(root/str(i), item, inputs)
                        self.assertEqual(wire, expected[i][0])
                        self.assertEqual(projection.rows, expected[i][1])
                self.assertEqual(inputs.statistics["source_admissions"], 1)
                self.assertEqual(inputs.statistics["account_bindings"], 3)
                self.assertEqual(inputs.statistics["owned_record_decodes"]-before_decode, 6*(1+inputs.statistics["owned_blocks"]))
            self.assertNotEqual(expected[0][0]["final_account"], expected[1][0]["final_account"])
            self.assertEqual(expected[0][0]["signal_ref"], expected[1][0]["signal_ref"])

    def test_owned_bytes_survive_source_mutation_and_decoded_aliases_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            oracle, _, _ = execute(root/"run", manifest); shutil.rmtree(root/"run")
            with imported(manifest) as inputs:
                path = Path(manifest["market_input"]["native_inputs"][0]["artifact"]["manifest_uri"])
                path.write_bytes(b"{}")
                with inputs.execution_scope():
                    audit = inputs.audit(manifest, block_sessions=2, read_budget=read_budget(LIMITS),
                        limits=LIMITS, implementation_ref=IMPLEMENTATION_REF)
                    audit.globals["profile"]["commission_rate"] = "0.9"
                    audit.receipt["counts"]["market_rows"] = 0
                    blocks = inputs.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS))
                    block = next(blocks)
                    next(iter(block.market_rows.values()))["open"] = "999999"
                    blocks.close()
                wire, _ = run_owned(root/"run", manifest, inputs)
                self.assertEqual(wire, oracle)
                with self.assertRaises(ContractError):
                    execute(root/"changed-source", manifest)

    def test_all_other_admission_capability_changes_and_invalid_top_k_fail_before_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            with imported(manifest) as inputs:
                changes = [lambda p: p["portfolio_policy"].update(top_k=True),
                    lambda p: p["portfolio_policy"].update(top_k=0),
                    lambda p: p["portfolio_policy"].update(top_k=100),
                    lambda p: p["portfolio_policy"].update(risk_cap="0.1"),
                    lambda p: p["portfolio_policy"].update(budget_basis="another"),
                    lambda p: p.update(clock_policy="another"),
                    lambda p: p.update(stock_action_policy="another"),
                    lambda p: p["scope"]["execution_universe"].reverse(),
                    lambda p: p["profile_input"].update(profile_ref="sha256:"+"f"*64),
                    lambda p: p["prediction_input"]["frames"][0].update(feature_ref="sha256:"+"f"*64),
                    lambda p: p["market_input"].update(execution_snapshot_id="another")]
                for change in changes:
                    candidate = deepcopy(manifest); change(candidate)
                    candidate["market_input"]["market_ref"] = logical_ref(candidate["market_input"], "market_ref")
                    candidate["prediction_input"]["prediction_ref"] = logical_ref(candidate["prediction_input"], "prediction_ref")
                    candidate["request_ref"] = logical_ref(candidate, "request_ref")
                    with patch("axiom_engine.runtime.backtest.AccountLedger") as ledger:
                        with self.assertRaises(ContractError):
                            run_owned(root/"bad", candidate, inputs)
                        ledger.assert_not_called()

    def test_budget_close_exclusive_scope_and_failure_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            opened = []
            original = tempfile.TemporaryFile
            def temporary(*args, **kwargs):
                value = original(*args, **kwargs); opened.append(value); return value
            with patch("axiom_engine.runtime.stock_market_owner.tempfile.TemporaryFile", side_effect=temporary):
                with self.assertRaisesRegex(ContractError, "max_market_bytes|quota"):
                    imported(manifest, max_owned_bytes=1)
            self.assertTrue(all(item.closed for item in opened))
            inputs = imported(manifest)
            with inputs.execution_scope():
                with self.assertRaisesRegex(ContractError, "sequential"):
                    run_owned(root/"busy", manifest, inputs)
                with self.assertRaisesRegex(ContractError, "during account"):
                    inputs.close()
            with self.assertRaisesRegex(ContractError, "fresh result sink"):
                run_stock_backtest(BacktestRequest.from_dict(manifest), source=inputs,
                    sink=object(), block_sessions=2, limits=LIMITS)
            with self.assertRaises(ContractError):
                run_owned(root/"small", manifest, inputs, {**LIMITS, "max_block_bytes": 40000})
            wire, _ = run_owned(root/"valid", manifest, inputs)
            self.assertEqual(wire["status"], "COMPLETE")
            inputs.close(); inputs.close()
            with self.assertRaisesRegex(ContractError, "closed"):
                run_owned(root/"closed", manifest, inputs)


if __name__ == "__main__":
    unittest.main()
