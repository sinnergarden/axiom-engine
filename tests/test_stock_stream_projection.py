"""Small v7 saved projections: output audit only, no large source reads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document, canonical
from axiom_engine.runtime import BacktestRun, run_backtest, save_backtest_run
from axiom_engine.runtime.stock_inputs import ACTION_POLICY
from axiom_engine.runtime.stock_schedule import CLOCK_POLICY
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from axiom_engine.runtime.stock_stream_outputs import RESULT_KINDS, StockResultSink
from axiom_engine.runtime.stock_stream_projection import (SavedRunProjection,
                                                         load_stock_backtest_projection)
from test_csi300_runtime import full_request
import test_stocks as legacy


REF = "sha256:" + "d" * 64
LIMITS = dict(max_folds=10, max_prediction_rows=1000, max_market_rows=1000,
              max_input_bytes=2000000, max_read_bytes=701, max_block_bytes=200000,
              max_result_part_bytes=100000, max_result_buffer_bytes=100000,
              max_result_bytes=2000000)


def artifact(path, wire, kind="small_profile"):
    return {"artifact_type": kind, "artifact_id": Document.from_dict(wire).identity,
            "contract_version": wire.get("contract_version", "synthetic"),
            "manifest_uri": str(path), "content_digest": Document.from_dict(wire).identity}


def save(path, wire):
    Path(path).write_text(canonical(wire))


def fixture(folder, request=None):
    root = Path(folder)
    request = full_request() if request is None else request
    plan = request.to_dict()
    old = run_backtest(request).to_dict()
    profile_path = root / "profile.json"
    save(profile_path, plan["profile"])
    profile = {"artifact": artifact(profile_path, plan["profile"]),
               "profile_ref": old["profile_ref"],
               "stock_execution_rules_ref": plan["profile"]["stock_execution_rules_ref"],
               "stock_fee_schedule_ref": plan["profile"]["stock_fee_schedule_ref"]}
    # Deliberately nonexistent parent payloads prove ordinary evaluation does
    # not open training, prediction or native Data sources.
    parent = {"artifact_type": "large_parent", "artifact_id": REF, "contract_version": "synthetic",
              "manifest_uri": str(root / "forbidden-large-parent.json"), "content_digest": REF}
    market = {"contract_version": "stock_market_input_refs_v1", "market_ref": REF,
              "model_snapshot_id": "s_model", "execution_snapshot_id": "s_execution",
              "warmup_sessions": [plan["market_replay"]["calendar"][0]],
              "price_basis": "unadjusted", "projection_version": "market_replay_v4",
              "native_inputs": [{"role": "execution", "artifact": parent, "native_ref": ref}
                                for ref in plan["market_replay"]["source_refs"][4:6]]}
    market["market_ref"] = logical_ref(market, "market_ref")
    frames = []
    for fold in plan["prediction_schedule"]["folds"]:
        frame = fold["prediction_frame"]
        frames.append({"fold_ref": fold["fold_ref"], **{name: frame[name] for name in
            ("fold_spec_ref", "model_ref", "feature_ref", "signal_run_ref")},
            "fold_spec_artifact": parent, "model_metadata_artifact": parent,
            "prediction_artifact": parent})
    predictions = {"contract_version": "stock_prediction_input_refs_v1", "prediction_ref": REF, "frames": frames}
    predictions["prediction_ref"] = logical_ref(predictions, "prediction_ref")
    calendar = plan["market_replay"]["calendar"]
    manifest = {"contract_version": "backtest_request_v7", "request_ref": REF, "account_id": plan["account_id"],
        "scope": {"start_session": calendar[1], "end_session": calendar[-1], "anchor_session": calendar[0],
                  "calendar": calendar, "prediction_universe": plan["prediction_universe"],
                  "execution_universe": plan["execution_universe"], "supported_universe_ref": plan["supported_universe_ref"]},
        "initial_account": plan["initial_account"], "portfolio_policy": plan["portfolio_policy"],
        "profile_input": profile, "market_input": market, "prediction_input": predictions,
        "stock_action_policy": ACTION_POLICY, "clock_policy": CLOCK_POLICY, "limitations": ["Synthetic only"]}
    manifest["request_ref"] = logical_ref(manifest, "request_ref")
    run_id = Document.from_dict({"request_ref": manifest["request_ref"], "core_version": old["core_version"],
        "runtime_version": "axiom.backtest/7", "implementation_ref": old["implementation_ref"]}).identity
    replacements = {}
    for index, order in enumerate(old["orders"]):
        replacements[order["order_id"]] = run_id + ":order:" + str(index)
        replacements[order["order_id"] + ":fill:0"] = run_id + ":order:" + str(index) + ":fill:0"

    def remap(value):
        if type(value) is dict:
            return {key: remap(item) for key, item in value.items()}
        if type(value) is list:
            return [remap(item) for item in value]
        return replacements.get(value, value) if type(value) is str else value

    rows = {kind: remap(old[kind]) for kind in RESULT_KINDS[:-1]}
    sink = StockResultSink(root / "parts", run_id=run_id)
    refs = []
    budget = {"max_part_bytes": LIMITS["max_result_part_bytes"],
              "max_buffer_bytes": LIMITS["max_result_buffer_bytes"], "max_total_bytes": LIMITS["max_result_bytes"]}
    for point in rows["nav"]:
        day = point["session"]
        refs += list(sink.append(session=day, phase="SESSION_COMMITTED",
            committed_sequence=point["committed_sequence"], write_budget=budget,
            rows=((kind, row) for kind in RESULT_KINDS[:-1] for row in rows[kind]
                  if row.get("session", row.get("trade_session")) == day)))
    sink.finish(write_budget=budget)
    audit = {"contract_version": "stock_input_audit_v1", "request_ref": manifest["request_ref"],
             "market_ref": market["market_ref"], "prediction_ref": predictions["prediction_ref"],
             "profile_ref": profile["profile_ref"], "implementation_ref": old["implementation_ref"],
             "counts": {"input_files": 3, "cash_actions": len(plan["market_replay"]["cash_dividends"])},
             "limitations": ["Synthetic saved audit"]}
    events = {"contract_version": "stock_account_events_v1", "request_ref": manifest["request_ref"],
              "market_ref": market["market_ref"], "profile_ref": profile["profile_ref"],
              "cash_dividends": plan["market_replay"]["cash_dividends"],
              "source_refs": plan["market_replay"]["source_refs"][4:6], "limitations": ["Synthetic saved events"]}
    wire = {key: deepcopy(old[key]) for key in ("account_id", "status", "core_version", "implementation_ref",
        "committed_sequence", "initial_nav_minor", "final_account", "stopped", "lifecycle_admission", "metrics", "limitations")}
    wire.update(contract_version="backtest_run_v7", run_id=run_id, request_manifest=manifest,
        request_ref=manifest["request_ref"], source_audit=audit, source_audit_ref=Document.from_dict(audit).identity,
        signal_ref=predictions["prediction_ref"], market_ref=market["market_ref"], profile_ref=profile["profile_ref"],
        runtime_version="axiom.backtest/7", result_parts=refs, account_events=events,
        account_events_ref=Document.from_dict(events).identity)
    wire["content_digest"] = Document.from_dict(wire).identity
    path = root / "run.json"
    save(path, wire)
    return path, wire, rows


def reseal_run(path, wire):
    wire.pop("content_digest", None)
    wire["content_digest"] = Document.from_dict(wire).identity
    save(path, wire)


def mutate_part(path, wire, index, mutation):
    ref = wire["result_parts"][index]
    part_path = Path(ref["artifact"]["manifest_uri"])
    part = json.loads(part_path.read_bytes())
    mutation(part)
    part.pop("content_digest")
    part["content_digest"] = Document.from_dict(part).identity
    save(part_path, part)
    ref["artifact"]["content_digest"] = Document.from_dict(part).identity
    if index + 1 < len(wire["result_parts"]):
        wire["result_parts"][index + 1]["previous_part_digest"] = ref["artifact"]["content_digest"]
    reseal_run(path, wire)


def cash_request(*, record="2024-01-02", ex="2024-01-03", security_index=0):
    def inputs(batches, membership):
        event = legacy.cash_event()
        event.update(security_id=membership["context"]["query"]["symbols"][security_index],
                     record_date=record, ex_date=ex)
        for index, time_field in ((4, "ex_date"), (5, "record_date")):
            with patch.object(legacy, "DAYS", batches[0]["context"]["query"]["sessions"]), \
                 patch.object(legacy, "SECURITIES", membership["context"]["query"]["symbols"]):
                batches[index] = legacy.batch("corporate_actions", legacy.STOCK_EVENT_FIELDS,
                    [deepcopy(event)], {"cash_dividend_before_tax_per_share": "CNY/share"}, time_field=time_field)
    return full_request(mutate_native=inputs)


def reseal_events(path, wire):
    wire["account_events_ref"] = Document.from_dict(wire["account_events"]).identity
    reseal_run(path, wire)


class ProjectionTests(unittest.TestCase):
    def test_load_business_rows_without_opening_large_parents(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, rows = fixture(folder)
            located = []

            def reader(ref):
                located.append(ref["manifest_uri"])
                self.assertNotIn("forbidden", ref["manifest_uri"])
                return Path(ref["manifest_uri"])

            projection = load_stock_backtest_projection(path, artifact_reader=reader, limits=LIMITS)
            self.assertIsInstance(projection, SavedRunProjection)
            self.assertEqual(projection.wire, wire)
            self.assertEqual(projection.input_run_ref, {key: wire[key] for key in
                                                      ("run_id", "content_digest", "committed_sequence")})
            for kind in RESULT_KINDS[:-1]:
                self.assertEqual(projection.rows[kind], rows[kind])
            self.assertEqual(len(located), 1 + len(wire["result_parts"]))
            self.assertIs(projection.verify(), projection)
            projection.rows["nav"][0]["nav_minor"] += 1
            with self.assertRaisesRegex(ContractError, "modified after loading"):
                projection.verify()

    def test_bad_tail_part_byte_binding_and_missing_tail_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            tail = Path(wire["result_parts"][-1]["artifact"]["manifest_uri"])
            tail.write_bytes(tail.read_bytes() + b" ")
            with self.assertRaisesRegex(ContractError, "byte/content"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
            wire["result_parts"].pop()
            reseal_run(path, wire)
            with self.assertRaisesRegex(ContractError, "final watermark"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_rehashed_business_fee_corruption_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            mutate_part(path, wire, 0, lambda part: part["rows"]["fills"][0].update(fee_minor=999999))
            with self.assertRaisesRegex(ContractError, "actual-fill fee"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_total_budget_rejects_before_opening_any_result_part(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            low = {**LIMITS, "max_result_bytes": path.stat().st_size +
                   Path(wire["request_manifest"]["profile_input"]["artifact"]["manifest_uri"]).stat().st_size + 1}
            opened = []
            real_open = Path.open

            def observe(target, *args, **kwargs):
                opened.append(str(target))
                return real_open(target, *args, **kwargs)

            with patch.object(Path, "open", observe):
                with self.assertRaisesRegex(ContractError, "total result budget"):
                    load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=low)
            self.assertFalse(any("part-" in name for name in opened))

    def test_physical_total_includes_public_run_lf_and_external_profile_once(self):
        for profile_lf in (False, True):
            with self.subTest(profile_lf=profile_lf), tempfile.TemporaryDirectory() as folder:
                _, wire, _ = fixture(folder)
                profile_path = Path(wire["request_manifest"]["profile_input"]["artifact"]["manifest_uri"])
                canonical_profile_size = profile_path.stat().st_size
                if profile_lf:
                    with profile_path.open("ab") as saved:
                        saved.write(b"\n")
                path = Path(folder) / "public-run.json"
                save_backtest_run(BacktestRun.from_dict(wire), path)
                self.assertEqual(path.read_bytes(), canonical(wire).encode() + b"\n")
                self.assertEqual(profile_path.stat().st_size, canonical_profile_size + profile_lf)
                part_bytes = sum(Path(ref["artifact"]["manifest_uri"]).stat().st_size
                                 for ref in wire["result_parts"])
                total = path.stat().st_size + profile_path.stat().st_size + part_bytes
                exact = {**LIMITS, "max_result_bytes": total}
                view = load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=exact)
                self.assertEqual(view.wire, wire)
                opened = []
                real_open = Path.open
                def observe(target, *args, **kwargs):
                    opened.append(str(target))
                    return real_open(target, *args, **kwargs)
                with patch.object(Path, "open", observe):
                    with self.assertRaisesRegex(ContractError, "total result budget"):
                        load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"],
                                                       limits={**exact, "max_result_bytes": total - 1})
                self.assertFalse(any("part-" in name for name in opened))
                self.assertNotIn(str(profile_path), opened)

    def test_run_alone_exceeding_total_budget_rejects_before_open_or_parse(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = fixture(folder)
            with patch.object(Path, "open", side_effect=AssertionError("payload opened before known budget check")):
                with self.assertRaisesRegex(ContractError, "total result budget"):
                    load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"],
                                                   limits={**LIMITS, "max_result_bytes": path.stat().st_size - 1})

    def test_rehashed_row_count_and_saved_cash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            wire["result_parts"][0]["row_counts"]["nav"] += 1
            reseal_run(path, wire)
            with self.assertRaisesRegex(ContractError, "row count"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            mutate_part(path, wire, 0, lambda part: part["rows"]["nav"][0].update(cash_minor=1))
            with self.assertRaisesRegex(ContractError, "NAV cash"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_constructor_and_tampered_audit_rejected(self):
        with self.assertRaises(ContractError):
            SavedRunProjection({}, {}, {}, [])
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder)
            wire["source_audit"]["counts"]["input_files"] += 1
            reseal_run(path, wire)
            with self.assertRaisesRegex(ContractError, "source-audit content"):
                load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_record_ex_unknown_payment_matches_saved_entitlement_without_parent_io(self):
        with tempfile.TemporaryDirectory() as folder:
            path, wire, _ = fixture(folder, cash_request())
            view = load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
            action = view.account_events["cash_dividends"][0]
            self.assertIsNone(action["pay_session"])
            ex = next(row for row in view.rows["cash_ledger"] if row["reason"] == "DIVIDEND_EX")
            position = next(row for row in view.rows["positions"]
                            if row["session"] == action["record_session"] and row["security_id"] == action["security_id"])
            self.assertEqual(ex["receivable_delta_minor"], position["quantity"] * 10)
            self.assertEqual(wire["final_account"]["receivable_minor"], ex["receivable_delta_minor"])
            self.assertFalse(any(row["reason"] == "DIVIDEND_PAY" for row in view.rows["cash_ledger"]))

    def test_known_future_ex_keeps_original_action_without_recognizing_income(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = fixture(folder, cash_request(ex="2024-01-09"))
            view = load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
            self.assertEqual(view.account_events["cash_dividends"][0]["ex_session"], "2024-01-09")
            self.assertEqual(view.wire["final_account"]["receivable_minor"], 0)
            self.assertFalse(any(row["reason"].startswith("DIVIDEND_") for row in view.rows["cash_ledger"]))

    def test_anchor_record_does_not_create_an_initial_entitlement(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = fixture(folder, cash_request(record="2023-12-29", ex="2024-01-02"))
            view = load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
            self.assertEqual(view.account_events["cash_dividends"][0]["record_session"], view.calendar[0])
            self.assertFalse(any(row["reason"] == "DIVIDEND_EX" for row in view.rows["cash_ledger"]))
            self.assertTrue(any(row["reason"] == "FILL" for row in view.rows["cash_ledger"]))

    def test_in_period_zero_entitlement_retains_original_zero_amount_ex(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _, _ = fixture(folder, cash_request(security_index=-1))
            view = load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)
            ex = next(row for row in view.rows["cash_ledger"] if row["reason"] == "DIVIDEND_EX")
            self.assertEqual((ex["cash_delta_minor"], ex["receivable_delta_minor"]), (0, 0))

    def test_missing_event_view_or_ref_is_explicitly_rejected(self):
        for missing in ("account_events", "account_events_ref"):
            with tempfile.TemporaryDirectory() as folder:
                path, wire, _ = fixture(folder)
                wire.pop(missing)
                reseal_run(path, wire)
                with self.assertRaisesRegex(ContractError, "event view/ref required"):
                    load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_event_ref_counts_source_and_economics_fail_even_after_run_rehash(self):
        for mutation, message in (
            (lambda wire: wire.update(account_events_ref=REF), "event content/reference"),
            (lambda wire: wire["account_events"].update(request_ref=REF), "event content/reference"),
            (lambda wire: wire["account_events"].update(source_refs=[REF]), "refs outside execution"),
            (lambda wire: wire["account_events"].update(cash_dividends=[]), "cash action count"),
            (lambda wire: wire["account_events"]["cash_dividends"][0].update(cash_before_tax_per_share="0.2"),
             "matching EX recognition"),
            (lambda wire: wire["account_events"]["cash_dividends"][0].update(pay_session="2024-01-08"),
             "PAY does not match")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as folder:
                path, wire, _ = fixture(folder, cash_request())
                mutation(wire)
                if message != "event content/reference" or wire["account_events_ref"] != REF:
                    wire["account_events_ref"] = Document.from_dict(wire["account_events"]).identity
                reseal_run(path, wire)
                with self.assertRaisesRegex(ContractError, message):
                    load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)

    def test_record_quantity_and_unreferenced_cash_action_are_not_inferred_from_balance(self):
        for change, message in (("record", "record entitlement"), ("event", "unique account action")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path, wire, _ = fixture(folder, cash_request())
                action = wire["account_events"]["cash_dividends"][0]
                if change == "record":
                    def mutation(part):
                        position = next(row for row in part["rows"]["positions"]
                                        if row["security_id"] == action["security_id"])
                        position["quantity"] += 1
                    mutate_part(path, wire, 0, mutation)
                else:
                    def mutation(part):
                        cash = next(row for row in part["rows"]["cash_ledger"] if row["reason"] == "DIVIDEND_EX")
                        cash["source_event_id"] = REF
                    mutate_part(path, wire, 1, mutation)
                with self.assertRaisesRegex(ContractError, message):
                    load_stock_backtest_projection(path, artifact_reader=lambda ref: ref["manifest_uri"], limits=LIMITS)


if __name__ == "__main__":
    unittest.main()
