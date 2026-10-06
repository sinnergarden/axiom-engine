"""Synthetic byte budgets and immutable output receipts; no input suppliers."""
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document, canonical
from axiom_engine.runtime import stock_stream_outputs
from axiom_engine.runtime.stock_stream_outputs import (RESULT_KINDS, StockResultSink,
                                                       bounded_canonical_bytes)


RUN = "sha256:" + "1" * 64
BUDGET = dict(max_part_bytes=4096, max_buffer_bytes=4096, max_total_bytes=100000)
DAY = "2024-01-31"


def nav(index=0, day=DAY):
    return {"session": day, "committed_sequence": index, "nav_minor": 50000000 - index,
            "cash_minor": 50000000 - index, "market_value_minor": 0, "receivable_minor": 0}


class StockOutputTests(unittest.TestCase):
    def test_canonical_stream_matches_original_json_unicode_and_nested_rows(self):
        row = {"中文": ["\n\t\"\\", "€漢🙂" * 1200, True, None, -0.0, 1e-9],
               "b": {"z": 1, "a": "\u0000"}}
        expected = canonical(row).encode("utf-8")
        self.assertEqual(bounded_canonical_bytes(row, len(expected)), expected)
        with self.assertRaises(ContractError):
            bounded_canonical_bytes(row, len(expected) - 1)

    def test_receipts_bind_exact_bytes_document_and_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            refs = list(sink.append(session=DAY, phase="SESSION_COMMITTED",
                        rows=iter([("nav", nav())]), committed_sequence=0, write_budget=BUDGET))
            second = "2024-02-01"
            refs += list(sink.append(session=second, phase="STOPPED_BEFORE_NAV",
                        rows=iter([]), committed_sequence=1, write_budget=BUDGET))
            self.assertEqual(sink.finish(write_budget=BUDGET), [])
            self.assertEqual(sink.buffered_bytes, 0)
            self.assertEqual([r["part_index"] for r in refs], [0, 1])
            self.assertIsNone(refs[0]["previous_part_digest"])
            self.assertEqual(refs[1]["previous_part_digest"], refs[0]["artifact"]["content_digest"])
            for ref in refs:
                path = Path(ref["artifact"]["manifest_uri"])
                raw = path.read_bytes()
                wire = json.loads(raw)
                self.assertEqual(raw, canonical(wire).encode())
                self.assertEqual(Document.from_dict(wire).identity, ref["artifact"]["content_digest"])
                recorded = wire.pop("content_digest")
                self.assertEqual(recorded, Document.from_dict(wire).identity)
                self.assertEqual(set(ref["row_counts"]), set(RESULT_KINDS))
                self.assertEqual(ref["row_counts"], {k: len(v) for k, v in wire["rows"].items()})

    def test_mid_session_receipt_precedes_consumption_of_following_rows(self):
        budget = dict(max_part_bytes=1000, max_buffer_bytes=1000, max_total_bytes=100000)
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            consumed = []

            def rows():
                for index in range(15):
                    consumed.append(index)
                    yield "positions", {"session": DAY, "committed_sequence": 15,
                                        "security_id": str(index), "note": "x" * 90}

            receipts = sink.append(session=DAY, phase="SESSION_COMMITTED", rows=rows(),
                                   committed_sequence=15, write_budget=budget)
            first = next(receipts)
            self.assertLess(len(consumed), 15)
            self.assertEqual(sink.buffered_bytes, 0)
            refs = [first, *receipts]
            all_rows = []
            for ref in refs:
                path = Path(ref["artifact"]["manifest_uri"])
                self.assertLessEqual(path.stat().st_size, budget["max_part_bytes"])
                all_rows.extend(json.loads(path.read_bytes())["rows"]["positions"])
            self.assertEqual([r["security_id"] for r in all_rows], list(map(str, range(15))))
            self.assertEqual(sum(r["row_counts"]["positions"] for r in refs), 15)

    def test_single_large_row_is_rejected_before_any_file_or_row_buffer(self):
        budget = dict(max_part_bytes=1000, max_buffer_bytes=700, max_total_bytes=10000)
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            with self.assertRaisesRegex(ContractError, "budget"):
                list(sink.append(session=DAY, phase="SESSION_COMMITTED", rows=iter([
                    ("decisions", {"trade_session": DAY, "huge": "x" * 10000})]),
                    committed_sequence=0, write_budget=budget))
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertLess(sink.buffered_bytes, 700)

    def test_write_failure_has_no_receipt_and_preserves_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            real_open = Path.open

            def fail_write(path, *args, **kwargs):
                if args and args[0] == "xb":
                    raise OSError("synthetic disk failure")
                return real_open(path, *args, **kwargs)

            receipts = sink.append(session=DAY, phase="SESSION_COMMITTED",
                rows=iter([("nav", nav())]), committed_sequence=0, write_budget=BUDGET)
            with patch.object(Path, "open", fail_write):
                with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                    next(receipts)
            self.assertGreater(sink.buffered_bytes, 0)
            self.assertEqual(sink._index, 0)
            self.assertEqual(list(Path(folder).iterdir()), [])
            with self.assertRaises(ContractError):
                sink.finish(write_budget=BUDGET)

    def test_total_budget_rejects_before_next_file_and_existing_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            ref = list(sink.append(session=DAY, phase="SESSION_COMMITTED",
                rows=iter([("nav", nav())]), committed_sequence=0, write_budget=BUDGET))[0]
            existing = Path(ref["artifact"]["manifest_uri"]).read_bytes()
            limited = {**BUDGET, "max_total_bytes": sink.total_bytes + 1}
            with self.assertRaisesRegex(ContractError, "total result budget"):
                list(sink.append(session="2024-02-01", phase="SESSION_COMMITTED",
                     rows=iter([]), committed_sequence=0, write_budget=limited))
            other = StockResultSink(folder, run_id=RUN)
            with self.assertRaises(FileExistsError):
                list(other.append(session=DAY, phase="SESSION_COMMITTED", rows=iter([]),
                                  committed_sequence=0, write_budget=BUDGET))
            self.assertEqual(Path(ref["artifact"]["manifest_uri"]).read_bytes(), existing)

    def test_bad_budget_bool_bad_phase_and_clock_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            for budget, phase in (({**BUDGET, "max_part_bytes": True}, "SESSION_COMMITTED"),
                                  (BUDGET, "ARBITRARY_PAUSE")):
                with self.assertRaises(ContractError):
                    list(sink.append(session=DAY, phase=phase, rows=iter([]),
                                     committed_sequence=0, write_budget=budget))
            list(sink.append(session=DAY, phase="SESSION_COMMITTED", rows=iter([]),
                             committed_sequence=2, write_budget=BUDGET))
            with self.assertRaises(ContractError):
                list(sink.append(session="2024-01-30", phase="SESSION_COMMITTED", rows=iter([]),
                                 committed_sequence=2, write_budget=BUDGET))

    def test_impossible_marker_budget_is_checked_before_encoding(self):
        for overrides in ({"max_part_bytes": 1}, {"max_total_bytes": 1}):
            with tempfile.TemporaryDirectory() as folder:
                sink = StockResultSink(folder, run_id=RUN)
                with patch.object(stock_stream_outputs, "bounded_canonical_bytes", side_effect=AssertionError("allocated")):
                    with self.assertRaisesRegex(ContractError, "budget"):
                        list(sink.append(session=DAY, phase="SESSION_COMMITTED", rows=iter([]),
                            committed_sequence=0, write_budget={**BUDGET, **overrides}))
                self.assertEqual(sink.buffered_bytes, 0)

    def test_runtime_pending_rows_share_buffer_budget_before_sink_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            sink._external_pending_bytes = lambda: BUDGET["max_buffer_bytes"]
            with patch.object(stock_stream_outputs, "bounded_canonical_bytes", side_effect=AssertionError("allocated")):
                with self.assertRaisesRegex(ContractError, "unified result buffer"):
                    list(sink.append(session=DAY, phase="SESSION_COMMITTED", rows=iter([("nav", nav())]),
                                     committed_sequence=0, write_budget=BUDGET))
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(sink.buffered_bytes, 0)

    def test_sink_rechecks_runtime_pending_budget_before_row_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            external = [0]
            sink._external_pending_bytes = lambda: external[0]
            def rows():
                external[0] = BUDGET["max_buffer_bytes"]
                yield "nav", nav()
            with self.assertRaisesRegex(ContractError, "unified result buffer"):
                list(sink.append(session=DAY, phase="SESSION_COMMITTED", rows=rows(),
                                 committed_sequence=0, write_budget=BUDGET))
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(len(sink._rows["nav"]), 0)

    def test_cross_part_live_encoded_buffers_obey_budget_after_receipt(self):
        budget = dict(max_part_bytes=2000, max_buffer_bytes=1000, max_total_bytes=10000)
        with tempfile.TemporaryDirectory() as folder:
            sink = StockResultSink(folder, run_id=RUN)
            buffers, peaks = [], []

            class TrackedBuffer(bytearray):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    buffers.append(weakref.ref(self))

                def extend(self, piece):
                    super().extend(piece)
                    # Measure actual live buffers, including any orphaned local
                    # reference after sink._encoded_bytes has been reset.
                    live = sum(len(ref()) for ref in buffers if ref() is not None)
                    peaks.append(live + sink._scratch_bytes)

            rows = []
            for index in range(3):
                row = {"session": DAY, "committed_sequence": 3, "security_id": str(index), "padding": ""}
                row["padding"] = "x" * (555 - len(canonical(row).encode()))
                self.assertEqual(len(canonical(row).encode()), 555)
                rows.append(("positions", row))
            with patch.object(stock_stream_outputs, "bytearray", TrackedBuffer, create=True):
                receipts = sink.append(session=DAY, phase="SESSION_COMMITTED", rows=iter(rows),
                                       committed_sequence=3, write_budget=budget)
                refs = []
                for receipt in receipts:
                    refs.append(receipt)
                    self.assertEqual(sum(len(ref()) for ref in buffers if ref() is not None), 0)
            self.assertEqual(len(refs), 3)
            self.assertLessEqual(max(peaks), budget["max_buffer_bytes"])


if __name__ == "__main__":
    unittest.main()
