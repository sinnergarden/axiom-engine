"""Small native saved-artifact source admission; no Data or Research runs."""
from copy import deepcopy
from array import array
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document, canonical
from axiom_engine.runtime.stock_evidence import native_batches, native_ref
from axiom_engine.runtime.stock_schedule import CLOCK_POLICY
from axiom_engine.runtime.stock_inputs import validate_stock_request
from axiom_engine.runtime.stock_stream_contracts import logical_ref, read_budget
from axiom_engine.runtime.stock_stream_inputs import StockInputSource, _CanonicalIndex
from axiom_engine.runtime import stock_stream_inputs as stream_inputs
from test_csi300_runtime import full_request
from test_csi300 import REF
import test_stocks as legacy


LIMITS = dict(max_folds=5, max_prediction_rows=1000, max_market_rows=1000,
              max_input_bytes=2_000_000, max_read_bytes=4096, max_block_bytes=500_000,
              max_result_part_bytes=100_000, max_result_buffer_bytes=100_000,
              max_result_bytes=2_000_000)


def artifact_file(root, name, wire):
    raw = canonical(wire)
    path = root / (name + ".json")
    path.write_text(raw, encoding="utf-8")
    return dict(artifact_type=name, artifact_id=name, contract_version=wire.get("contract_version", "1"),
                manifest_uri=str(path), content_digest=Document(raw).identity)


def source_fixture(root, plan=None):
    """Persist existing synthetic v6 parents; no new source/Signal identity."""
    plan = plan or full_request().to_dict()
    market = plan["market_replay"]
    batches = native_batches(market["source_evidence"], market.get("coverage_bundle", []))
    native = []
    for i, batch in enumerate(batches):
        artifact = artifact_file(root, "native-" + str(i), batch)
        native.append(dict(role="execution", artifact=artifact, native_ref=native_ref(batch)))
        if i in (1, 3, 6):
            native.append(dict(role="prediction_basis", artifact=artifact, native_ref=native_ref(batch)))
    # The old synthetic v6 evidence only declared warmup pairing.  Give this
    # fixture actual separately saved original warmup batches for the v7 audit.
    warmup = ["2023-12-28"]
    for i in (1, 3):
        batch = deepcopy(batches[i]); anchor = market["calendar"][0]
        batch["records"] = [dict(row, session=warmup[0]) for row in batch["records"] if row["session"] == anchor]
        for field in batch["field_meta"].values():
            field["by_key"] = [dict(row, session=warmup[0], usable_from=(row["usable_from"] or "").replace(anchor, warmup[0]) or None)
                               for row in field["by_key"] if row["session"] == anchor]
        batch["context"]["query"].update(sessions=warmup, cutoff_by_session={warmup[0]: warmup[0]+"T20:30:00+08:00"})
        artifact = artifact_file(root, "warmup-" + str(i), batch)
        for role in ("execution", "prediction_basis"):
            native.append(dict(role=role, artifact=artifact, native_ref=native_ref(batch)))
    frames = []
    for i, fold in enumerate(plan["prediction_schedule"]["folds"]):
        frame = fold["prediction_frame"]
        frames.append(dict(fold_ref=fold["fold_ref"], **{name: frame[name] for name in
            ("fold_spec_ref", "model_ref", "feature_ref", "signal_run_ref")},
            fold_spec_artifact=artifact_file(root, "fold-spec-"+str(i), fold["fold_spec"]),
            model_metadata_artifact=artifact_file(root, "model-"+str(i), fold["model"]),
            prediction_artifact=artifact_file(root, "prediction-"+str(i), frame)))
    profile = plan["profile"]
    manifest = dict(contract_version="backtest_request_v7", account_id=plan["account_id"],
        scope=dict(start_session=plan["start_session"], end_session=plan["end_session"],
                   anchor_session=market["calendar"][0], calendar=market["calendar"],
                   prediction_universe=plan["prediction_universe"], execution_universe=plan["execution_universe"],
                   supported_universe_ref=plan["supported_universe_ref"]),
        initial_account=plan["initial_account"], portfolio_policy=plan["portfolio_policy"],
        profile_input=dict(artifact=artifact_file(root, "profile", profile), profile_ref=Document.from_dict(profile).identity,
                           **{name: profile[name] for name in ("stock_execution_rules_ref", "stock_fee_schedule_ref")}),
        market_input=dict(contract_version="stock_market_input_refs_v1", model_snapshot_id=batches[1]["context"]["snapshot_id"],
                          execution_snapshot_id=batches[1]["context"]["snapshot_id"], warmup_sessions=warmup,
                          price_basis="unadjusted", projection_version="market_replay_v4", native_inputs=native),
        prediction_input=dict(contract_version="stock_prediction_input_refs_v1", frames=frames),
        stock_action_policy=plan["stock_action_policy"], clock_policy=CLOCK_POLICY, limitations=["Synthetic v7 source fixture"])
    manifest["market_input"]["market_ref"] = logical_ref(manifest["market_input"], "market_ref")
    manifest["prediction_input"]["prediction_ref"] = logical_ref(manifest["prediction_input"], "prediction_ref")
    manifest["request_ref"] = logical_ref(manifest, "request_ref")
    return manifest, plan


def admitted(source, manifest, block_sessions=2, limits=LIMITS):
    return source.audit(manifest, block_sessions=block_sessions, read_budget=read_budget(limits),
                        limits=limits, implementation_ref=REF)


def rebind_native(manifest, name, mutate):
    entry = next(item for item in manifest["market_input"]["native_inputs"] if item["artifact"]["artifact_id"] == name)
    path = Path(entry["artifact"]["manifest_uri"])
    wire = Document(path.read_text()).to_dict(); mutate(wire)
    path.write_text(canonical(wire))
    reference = Document.from_dict(wire).identity
    for item in manifest["market_input"]["native_inputs"]:
        if item["artifact"]["manifest_uri"] == str(path):
            item["native_ref"] = reference; item["artifact"]["content_digest"] = reference
    manifest["market_input"]["market_ref"] = logical_ref(manifest["market_input"], "market_ref")
    manifest["request_ref"] = logical_ref(manifest, "request_ref")


def nested_fold_fixture(root, *, spec_version=None, wrapper_version=None,
                        manifest_version="stock_ml_fold_manifest_v1"):
    """The saved Research wire shape, with synthetic unselected parent data."""
    manifest, plan = source_fixture(root)
    item = manifest["prediction_input"]["frames"][0]
    spec = Document(Path(item["fold_spec_artifact"]["manifest_uri"]).read_text()).to_dict()
    if spec_version is not None:
        spec["contract_version"] = spec_version
    item["fold_spec_ref"] = Document.from_dict(spec).identity
    item["fold_spec_artifact"]["content_digest"] = item["fold_spec_ref"]
    folder = root/"saved-fold"; folder.mkdir()
    for artifact, name in ((item["model_metadata_artifact"], "model.json"), (item["prediction_artifact"], "predictions.json")):
        wire = Document(Path(artifact["manifest_uri"]).read_text()).to_dict()
        if name == "predictions.json":
            wire["fold_spec_ref"] = item["fold_spec_ref"]
            wire.pop("signal_run_ref"); wire["signal_run_ref"] = Document.from_dict(wire).identity
            item["signal_run_ref"] = wire["signal_run_ref"]
        artifact["content_digest"] = Document.from_dict(wire).identity
        (folder/name).write_text(canonical(wire)+"\n")
        artifact["manifest_uri"] = str(folder/name)
    definition = {"version": "axiom.stock_ml_fold/1", "fold_spec_ref": item["fold_spec_ref"], "fold_spec": spec,
                  "input_manifest": {"synthetic_unselected_training_descriptors": [{"source": "x"*150} for _ in range(1000)]}}
    refs = {name: item.get(name, REF) for name in ("feature_ref", "label_ref", "dataset_ref", "model_ref", "signal_run_ref", "evidence_ref")}
    version = wrapper_version or {"stock_ml_fold_spec_v1": "stock_ml_fold_v1", "stock_ml_fold_spec_v2": "stock_ml_fold_v2"}[spec["contract_version"]]
    fold = {"contract_version": version, "definition": definition,
            "definition_ref": Document.from_dict(definition).identity, "status": "COMPLETE", **refs,
            "engine_admission": {"neutral_validation": "NOT_PERFORMED_BY_BUILDER", "runtime": "UNSUPPORTED_V2"},
            "limitations": ["Synthetic original saved Research fold wrapper"]}
    fold["fold_ref"] = Document.from_dict({"definition_ref": fold["definition_ref"], **refs}).identity
    fold["content_digest"] = Document.from_dict(fold).identity
    (folder/"fold.json").write_text(canonical(fold)+"\n")
    file_names = ("fold.json", "feature-slice.json", "label-slice.json", "dataset.json", "model.json",
                  "predictions.json", "signal-evidence.json", "booster.txt")
    descriptor = {"contract_version": manifest_version, "fold_ref": fold["fold_ref"],
                  "files": {name: "sha256:"+hashlib.sha256((folder/name).read_bytes()).hexdigest()
                            if (folder/name).exists() else REF for name in file_names}}
    (folder/"manifest.json").write_text(canonical(descriptor)+"\n")
    item["fold_ref"] = fold["fold_ref"]
    item["fold_spec_artifact"]["manifest_uri"] = str(folder/"fold.json")+"#definition/fold_spec"
    manifest["prediction_input"]["prediction_ref"] = logical_ref(manifest["prediction_input"], "prediction_ref")
    manifest["request_ref"] = logical_ref(manifest, "request_ref")
    return manifest, plan, folder


class StockStreamInputTests(unittest.TestCase):
    def test_native_and_audit_views_retire_before_next_decode_and_reservation_close(self):
        class WeakMap(dict):
            pass
        class WeakList(list):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp)); source = StockInputSource()
            tracked = {}; retired = []; prediction_checks = []; view_stages = {}; active_stages = []
            def watch(stage, label, value):
                tracked.setdefault(id(stage), []).append((label, weakref.ref(value)))
                return value
            original_native = source._native
            original_block = source._block
            original_grid = stream_inputs._grid
            original_rows = _CanonicalIndex.rows
            original_close = stream_inputs._DecodedStage.close
            original_enter = stream_inputs._DecodedStage.__enter__
            original_exit = stream_inputs._DecodedStage.__exit__
            from axiom_engine.core import stock_portfolio
            original_validate = stock_portfolio.validate_stock_predictions
            def native(role, kind, days, budget, *, reservation=None):
                view, used, bindings = original_native(role, kind, days, budget, reservation=reservation)
                wrapped = watch(reservation, "native:"+kind, WeakMap(view))
                view_stages[id(wrapped)] = reservation
                wrapped["records"] = watch(reservation, "records:"+kind, WeakList(view["records"]))
                wrapped["field_meta"] = {name: {**field, "by_key": watch(reservation, "by_key:"+kind, WeakList(field["by_key"]))}
                                         for name, field in view["field_meta"].items()}
                return wrapped, used, bindings
            def grid(batch, days, universe):
                records, metadata = original_grid(batch, days, universe); stage = view_stages[id(batch)]
                records = watch(stage, "grid:records", WeakMap({key: watch(stage, "grid:row", WeakMap(row)) for key, row in records.items()}))
                metadata = watch(stage, "grid:metadata", WeakMap({name: {key: watch(stage, "grid:fact", WeakMap(row)) for key, row in rows.items()}
                                                                for name, rows in metadata.items()}))
                return records, metadata
            def validate(frame):
                wire, checked = original_validate(frame); stage = active_stages[-1]
                rows = {key: watch(stage, "checked:row", WeakMap(row)) for key, row in checked.items()}
                wire = watch(stage, "checked:wire", WeakMap({**wire, "rows": list(rows.values())}))
                checked = watch(stage, "checked:index", WeakMap(rows))
                return wire, checked
            def block(manifest, days, budget):
                value = original_block(manifest, days, budget); stage = value._reservation
                market = watch(stage, "block:market", WeakMap({key: watch(stage, "block:market_row", WeakMap(row))
                                                              for key, row in value.market_rows.items()}))
                signals = {day: (header, watch(stage, "block:prediction_index", WeakMap(indexed)))
                           for day, (header, indexed) in value.signals.items()}
                return replace(value, market_rows=market, signals=signals)
            def rows(index, path, days, budget=None, *, reservation=None):
                if path == ("rows",):
                    labels = [(label, ref()) for label, ref in tracked.get(id(reservation), []) if label.startswith(("native:", "records:", "by_key:"))]
                    self.assertTrue(labels)
                    self.assertTrue(all(value is None for _, value in labels), labels)
                    prediction_checks.append(len(labels))
                return original_rows(index, path, days, budget, reservation=reservation)
            def close(stage):
                refs = tracked.pop(id(stage), [])
                self.assertTrue(all(ref() is None for _, ref in refs), [label for label, ref in refs if ref() is not None])
                retired.extend(label for label, _ in refs)
                return original_close(stage)
            def enter(stage):
                active_stages.append(stage)
                return original_enter(stage)
            def exit(stage, *args):
                try:
                    return original_exit(stage, *args)
                finally:
                    self.assertIs(active_stages.pop(), stage)
            with patch.object(source, "_native", new=native), patch.object(source, "_block", new=block), \
                    patch.object(stream_inputs, "_grid", new=grid), patch.object(stock_portfolio, "validate_stock_predictions", new=validate), \
                    patch.object(_CanonicalIndex, "rows", new=rows), patch.object(stream_inputs._DecodedStage, "close", new=close), \
                    patch.object(stream_inputs._DecodedStage, "__enter__", new=enter), patch.object(stream_inputs._DecodedStage, "__exit__", new=exit):
                admitted(source, manifest)
            self.assertTrue(prediction_checks)
            self.assertIn("native:membership", retired)
            self.assertIn("checked:row", retired)
            self.assertIn("block:prediction_index", retired)
            self.assertEqual(tracked, {})
            self.assertEqual(active_stages, [])
            self.assertEqual(source._memory.temporary_bytes, 0)

    def test_fold_identity_proof_has_scratch_and_profile_physical_size_is_bound(self):
        from axiom_engine.runtime import stock_schedule
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp)); source = StockInputSource()
            profile_path = Path(manifest["profile_input"]["artifact"]["manifest_uri"])
            profile_path.write_bytes(profile_path.read_bytes()+b"\n")
            verify = stock_schedule._verify_model; observations = []
            def checked(model):
                frame = sys._getframe(1)
                while frame is not None and frame.f_code.co_name != "_frames":
                    frame = frame.f_back
                self.assertIsNotNone(frame)
                expected = 3*(stream_inputs._encoded_size(frame.f_locals["spec"])+stream_inputs._encoded_size(model))
                self.assertEqual(source._memory.temporary_bytes, expected)
                observations.append(expected)
                return verify(model)
            with patch.object(stock_schedule, "_verify_model", side_effect=checked):
                audit = admitted(source, manifest)
            self.assertTrue(observations)
            self.assertEqual(audit.globals["profile_bytes"], profile_path.stat().st_size)
            self.assertNotIn("profile_bytes", audit.receipt)
            # The iteration entry checks even a cached, already admitted profile.
            profile_path.write_bytes(profile_path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ContractError, "changed after admission"):
                next(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))

    def test_nested_original_v1_v2_wrappers_require_the_matching_spec_version(self):
        for wrapper, spec, accepted in (("stock_ml_fold_v1", "stock_ml_fold_spec_v1", True),
                                        ("stock_ml_fold_v2", "stock_ml_fold_spec_v2", True),
                                        ("stock_ml_fold_v1", "stock_ml_fold_spec_v2", False),
                                        ("stock_ml_fold_v2", "stock_ml_fold_spec_v1", False)):
            with self.subTest(wrapper=wrapper, spec=spec), tempfile.TemporaryDirectory() as tmp:
                manifest, _, _ = nested_fold_fixture(Path(tmp), spec_version=spec, wrapper_version=wrapper)
                source = StockInputSource()
                if accepted:
                    admitted(source, manifest)
                    blocks = list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))
                    self.assertTrue(any(block.signals for block in blocks))
                else:
                    with self.assertRaisesRegex(ContractError, "wrapper identity mismatch"):
                        admitted(source, manifest)
                    self.assertFalse(source._audited)

    def test_nested_matrix_v3_accepts_only_exact_manifest_wrapper_spec_combinations(self):
        accepted = {("stock_ml_fold_manifest_v1", "stock_ml_fold_v1", "stock_ml_fold_spec_v1"),
                    ("stock_ml_fold_manifest_v1", "stock_ml_fold_v2", "stock_ml_fold_spec_v2"),
                    ("stock_ml_fold_manifest_v2", "stock_ml_fold_v3", "stock_ml_fold_spec_v1"),
                    ("stock_ml_fold_manifest_v2", "stock_ml_fold_v3", "stock_ml_fold_spec_v2")}
        from itertools import product
        for versions in product(("stock_ml_fold_manifest_v1", "stock_ml_fold_manifest_v2", "unknown"),
                                ("stock_ml_fold_v1", "stock_ml_fold_v2", "stock_ml_fold_v3"),
                                ("stock_ml_fold_spec_v1", "stock_ml_fold_spec_v2")):
            with self.subTest(versions=versions), tempfile.TemporaryDirectory() as tmp:
                descriptor, wrapper, spec = versions
                manifest, _, _ = nested_fold_fixture(Path(tmp), manifest_version=descriptor,
                                                     wrapper_version=wrapper, spec_version=spec)
                source = StockInputSource()
                if versions in accepted:
                    admitted(source, manifest)
                    blocks = list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))
                    self.assertTrue(any(block.signals for block in blocks))
                else:
                    with self.assertRaises(ContractError):
                        admitted(source, manifest)
                    self.assertFalse(source._audited)

    def test_nested_matrix_v3_keeps_original_parent_selector_stage_and_clock_gates(self):
        for spec in ("stock_ml_fold_spec_v1", "stock_ml_fold_spec_v2"):
            for failure in ("child", "selector", "parent", "raw_file", "model_file", "stage", "clock"):
                with self.subTest(spec=spec, failure=failure), tempfile.TemporaryDirectory() as tmp:
                    manifest, _, folder = nested_fold_fixture(Path(tmp), manifest_version="stock_ml_fold_manifest_v2",
                        wrapper_version="stock_ml_fold_v3", spec_version=spec)
                    item = manifest["prediction_input"]["frames"][0]
                    if failure == "child":
                        item["fold_spec_artifact"]["content_digest"] = REF
                    elif failure == "selector":
                        item["fold_spec_artifact"]["manifest_uri"] = str(folder/"fold.json")+"#definition/input_manifest"
                    elif failure == "stage":
                        item["feature_ref"] = Document.from_dict({"unrelated_stage": True}).identity
                    elif failure == "parent":
                        wire = Document((folder/"fold.json").read_text()).to_dict()
                        wire["definition"]["input_manifest"]["tampered"] = True
                        wire.pop("content_digest"); wire["content_digest"] = Document.from_dict(wire).identity
                        (folder/"fold.json").write_text(canonical(wire)+"\n")
                    elif failure == "clock":
                        # Rebind valid hashes so rejection exercises the clock gate.
                        wire = Document((folder/"model.json").read_text()).to_dict()
                        wire["simulated_available_at"] = wire["fit_cutoff"]
                        wire.pop("model_ref"); wire["model_ref"] = Document.from_dict(wire).identity
                        (folder/"model.json").write_text(canonical(wire)+"\n")
                        item["model_ref"] = wire["model_ref"]
                        item["model_metadata_artifact"]["content_digest"] = Document.from_dict(wire).identity
                        prediction = Document((folder/"predictions.json").read_text()).to_dict()
                        prediction["model_ref"] = wire["model_ref"]
                        prediction.pop("signal_run_ref"); prediction["signal_run_ref"] = Document.from_dict(prediction).identity
                        item["signal_run_ref"] = prediction["signal_run_ref"]
                        item["prediction_artifact"]["content_digest"] = Document.from_dict(prediction).identity
                        (folder/"predictions.json").write_text(canonical(prediction)+"\n")
                        wrapper = Document((folder/"fold.json").read_text()).to_dict()
                        wrapper.update(model_ref=wire["model_ref"], signal_run_ref=prediction["signal_run_ref"])
                        refs = {name: wrapper[name] for name in ("feature_ref", "label_ref", "dataset_ref", "model_ref", "signal_run_ref", "evidence_ref")}
                        wrapper["fold_ref"] = Document.from_dict({"definition_ref": wrapper["definition_ref"], **refs}).identity
                        wrapper.pop("content_digest"); wrapper["content_digest"] = Document.from_dict(wrapper).identity
                        item["fold_ref"] = wrapper["fold_ref"]
                        (folder/"fold.json").write_text(canonical(wrapper)+"\n")
                        descriptor = Document((folder/"manifest.json").read_text()).to_dict()
                        descriptor["fold_ref"] = wrapper["fold_ref"]
                        for name in ("fold.json", "model.json", "predictions.json"):
                            descriptor["files"][name] = "sha256:"+hashlib.sha256((folder/name).read_bytes()).hexdigest()
                        (folder/"manifest.json").write_text(canonical(descriptor)+"\n")
                    else:
                        wire = Document((folder/"manifest.json").read_text()).to_dict()
                        wire["files"]["fold.json" if failure == "raw_file" else "model.json"] = REF
                        (folder/"manifest.json").write_text(canonical(wire)+"\n")
                    manifest["prediction_input"]["prediction_ref"] = logical_ref(manifest["prediction_input"], "prediction_ref")
                    manifest["request_ref"] = logical_ref(manifest, "request_ref")
                    source = StockInputSource()
                    with self.assertRaisesRegex(ContractError, "model|Model|clock" if failure == "clock" else ".*"):
                        admitted(source, manifest)
                    self.assertFalse(source._audited)

    def test_nested_saved_spec_reads_only_selected_small_child_and_original_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, plan, folder = nested_fold_fixture(Path(tmp))
            source = StockInputSource()
            with patch.object(Path, "open", side_effect=AssertionError("inventory opens payload")):
                inventory = source.inventory(manifest)
            self.assertEqual(len(inventory["files"]), 14)
            loads = json.loads; decoded = []
            def tracked(raw, *args, **kwargs):
                decoded.append(len(raw)); return loads(raw, *args, **kwargs)
            with patch("axiom_engine.runtime.stock_stream_inputs.json.loads", side_effect=tracked):
                audit = admitted(source, manifest)
                blocks = list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))
            self.assertLess(max(decoded), 10000)
            self.assertGreater((folder/"fold.json").stat().st_size, 150000)
            self.assertEqual(audit.receipt["counts"]["prediction_rows"], len(plan["prediction_schedule"]["folds"][0]["prediction_frame"]["rows"]))
            headers = [header for block in blocks for header, _ in block.signals.values()]
            self.assertTrue(all(header["signal_run_ref"] == plan["prediction_schedule"]["folds"][0]["prediction_frame"]["signal_run_ref"] for header in headers))
            binding = next(binding for binding in blocks[0].bindings if "selector" in binding)
            self.assertEqual(binding["child_ref"], manifest["prediction_input"]["frames"][0]["fold_spec_ref"])
            self.assertNotEqual(binding["parent_ref"], binding["child_ref"])
            self.assertNotEqual(binding["parent_ref"], binding["file_digest"])
            self.assertEqual(binding["file_digest"], "sha256:"+hashlib.sha256((folder/"fold.json").read_bytes()).hexdigest())
            self.assertFalse((folder/"feature-slice.json").exists())

    def test_nested_saved_spec_rejects_child_parent_and_raw_manifest_mismatches(self):
        for failure in ("child", "parent", "raw_file", "model_file", "selector"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                manifest, _, folder = nested_fold_fixture(Path(tmp)); item = manifest["prediction_input"]["frames"][0]
                if failure == "child":
                    item["fold_spec_artifact"]["content_digest"] = REF
                elif failure == "selector":
                    item["fold_spec_artifact"]["manifest_uri"] = str(folder/"fold.json")+"#definition/input_manifest"
                elif failure == "parent":
                    wire = Document((folder/"fold.json").read_text()).to_dict(); wire["definition"]["input_manifest"]["tampered"] = True
                    wire.pop("content_digest"); wire["content_digest"] = Document.from_dict(wire).identity
                    (folder/"fold.json").write_text(canonical(wire)+"\n")
                else:
                    wire = Document((folder/"manifest.json").read_text()).to_dict()
                    wire["files"]["fold.json" if failure == "raw_file" else "model.json"] = REF
                    (folder/"manifest.json").write_text(canonical(wire)+"\n")
                manifest["prediction_input"]["prediction_ref"] = logical_ref(manifest["prediction_input"], "prediction_ref")
                manifest["request_ref"] = logical_ref(manifest, "request_ref")
                source = StockInputSource()
                with self.assertRaises(ContractError):
                    admitted(source, manifest)
                self.assertFalse(source._audited)

    def test_row_limit_stops_before_second_row_decode_or_span_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = artifact_file(Path(tmp), "row-limit", {"records": [
                {"security_id": "a", "session": "2024-01-02"},
                {"security_id": "b", "session": "2024-01-02", "oversize": "x"*10000}]})
            arrays = []
            def tracked_array(kind):
                item = array(kind); arrays.append(item); return item
            with patch("axiom_engine.runtime.stock_stream_inputs.array", side_effect=tracked_array):
                with self.assertRaisesRegex(ContractError, "row budget exceeded before compact index growth"):
                    _CanonicalIndex(artifact, dict(max_read_bytes=16, max_decoded_bytes=256), row_limit=1)
            self.assertEqual([len(item) for item in arrays], [2])

    def test_native_record_count_cannot_grow_beyond_original_query_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp))
            rebind_native(manifest, "native-1", lambda wire: wire["records"].append(deepcopy(wire["records"][-1])))
            with self.assertRaisesRegex(ContractError, "row budget exceeded before compact index growth"):
                admitted(StockInputSource(), manifest)

    def test_prediction_index_growth_is_capped_by_original_fold_oos_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp)); item = manifest["prediction_input"]["frames"][0]
            artifact = item["prediction_artifact"]; path = Path(artifact["manifest_uri"])
            wire = Document(path.read_text()).to_dict()
            wire["rows"].append({"oversized_unexpected_row": "x"*600000})
            path.write_text(canonical(wire)); artifact["content_digest"] = Document.from_dict(wire).identity
            manifest["prediction_input"]["prediction_ref"] = logical_ref(manifest["prediction_input"], "prediction_ref")
            manifest["request_ref"] = logical_ref(manifest, "request_ref")
            with self.assertRaisesRegex(ContractError, "row budget exceeded before compact index growth"):
                admitted(StockInputSource(), manifest)

    def test_single_terminal_lf_keeps_content_identity_and_binds_actual_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root)
            before = manifest["request_ref"]
            for path in root.iterdir():
                path.write_bytes(path.read_bytes()+b"\n")
            source = StockInputSource(); admitted(source, manifest)
            blocks = list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))
            self.assertEqual(manifest["request_ref"], before)
            for binding in blocks[0].bindings:
                self.assertNotEqual(binding["file_digest"], binding["parent_ref"])
            artifact = manifest["profile_input"]["artifact"]
            index = source._indexes[(str(Path(artifact["manifest_uri"]).resolve()), artifact["content_digest"])]
            self.assertEqual(index.file_digest, "sha256:"+hashlib.sha256(Path(artifact["manifest_uri"]).read_bytes()).hexdigest())
            Path(artifact["manifest_uri"]).write_bytes(Path(artifact["manifest_uri"]).read_bytes()+b"\n")
            with self.assertRaisesRegex(ContractError, "Trailing or noncanonical"):
                _CanonicalIndex(artifact, read_budget(LIMITS))

    def test_cumulative_headers_reject_even_when_each_scalar_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = artifact_file(Path(tmp), "multi-header", {
                "context": {"a": "x"*70, "b": "y"*70}, "records": [], "field_meta": {}})
            with self.assertRaisesRegex(ContractError, "cumulative decoded budget"):
                _CanonicalIndex(artifact, dict(max_read_bytes=16, max_decoded_bytes=100))
            index = _CanonicalIndex(artifact, dict(max_read_bytes=16, max_decoded_bytes=1000))
            index._memory.limit = 100
            with self.assertRaisesRegex(ContractError, "before cache allocation"):
                index.header()

    def test_scanner_rows_charge_raw_and_decoded_and_drop_previous_row(self):
        row = {"session": "2024-01-02", "security_id": "a", **{"f"+str(i): "x"*100 for i in range(5)}}
        row_size = len(canonical(row).encode())
        self.assertEqual(row_size, 582)
        for array_kind in ("records", "by_key"):
            with self.subTest(array_kind=array_kind), tempfile.TemporaryDirectory() as tmp:
                wire = {"records": [row, row]} if array_kind == "records" else {"field_meta": {"value": {"by_key": [row, row]}}, "records": []}
                artifact = artifact_file(Path(tmp), "scan-rows", wire)
                for limit in (1000, 1300):
                    with self.assertRaisesRegex(ContractError, "cumulative decoded budget"):
                        _CanonicalIndex(artifact, dict(max_read_bytes=125, max_decoded_bytes=limit))
                loads = json.loads; observations = []
                decoder_observations = []
                def decoded_profile(frame, event, value):
                    if event != "return" or frame.f_code.co_name != "decode" or type(value) is not dict or value.get("security_id") != "a":
                        return
                    caller = frame.f_back
                    while caller is not None and caller.f_code.co_name != "_value":
                        caller = caller.f_back
                    self.assertIsNotNone(caller)
                    index = caller.f_locals["self"]
                    active = len(caller.f_locals["raw"]) + len(frame.f_locals["s"].encode()) + row_size + len(index._buffer)
                    self.assertLessEqual(active, index._memory.temporary_bytes+index._memory.global_bytes)
                    self.assertLessEqual(active, 2000)
                    decoder_observations.append(active)
                def observed(raw, *args, **kwargs):
                    value = loads(raw, *args, **kwargs)
                    if type(value) is dict and value.get("security_id") == "a":
                        frame = sys._getframe(1)
                        while frame is not None and frame.f_code.co_name != "_value":
                            frame = frame.f_back
                        self.assertIsNotNone(frame)
                        self.assertEqual(frame.f_code.co_name, "_value")
                        self.assertIsNone(frame.f_locals.get("row"))
                        index = frame.f_locals["self"]
                        active = len(raw)+row_size+len(index._buffer)
                        self.assertLessEqual(active, index._memory.temporary_bytes+index._memory.global_bytes)
                        self.assertLessEqual(index._memory.temporary_bytes+index._memory.global_bytes, 2000)
                        observations.append(active)
                    return value
                with patch("axiom_engine.runtime.stock_stream_inputs.json.loads", side_effect=observed):
                    previous_profile = sys.getprofile(); sys.setprofile(decoded_profile)
                    try:
                        index = _CanonicalIndex(artifact, dict(max_read_bytes=125, max_decoded_bytes=2000))
                    finally:
                        sys.setprofile(previous_profile)
                self.assertEqual(len(observations), 2)
                self.assertEqual(len(decoder_observations), 2)
                self.assertEqual(index._memory.temporary_bytes, 0)

    def test_unknown_unselected_texts_are_checked_then_released_and_scalar_growth_is_guarded(self):
        for size, admitted_ok in ((10000, True), (40000, False)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                manifest, _ = source_fixture(Path(tmp))
                unknown = {"contract_type": "Unknown", "contract_version": "1", "metadata": {},
                           "unknown_id": "i"*size, "reason": "r"*size, "required_evidence": "e"*size}
                rebind_native(manifest, "native-1", lambda wire: wire["context"].setdefault("coverage", {}).update(extra=unknown))
                loads = json.loads; observations = []
                def observed(raw, *args, **kwargs):
                    frame = sys._getframe(1)
                    while frame is not None:
                        if frame.f_code.co_name == "_value":
                            flags = frame.f_locals.get("reserved_values", {})
                            self.assertTrue(all(type(flag) is bool for flag in flags.values()))
                        frame = frame.f_back
                    value = loads(raw, *args, **kwargs)
                    if type(value) is str and len(value) == size:
                        observations.append(len(value))
                    return value
                source = StockInputSource()
                with patch("axiom_engine.runtime.stock_stream_inputs.json.loads", side_effect=observed):
                    if admitted_ok:
                        admitted(source, manifest, limits={**LIMITS, "max_block_bytes": 200000})
                        self.assertGreaterEqual(len(observations), 3)
                    else:
                        with self.assertRaisesRegex(ContractError, "cumulative decoded budget"):
                            admitted(source, manifest, limits={**LIMITS, "max_block_bytes": 100000})
                        self.assertEqual(observations, [])
                        self.assertFalse(source._audited)
                    self.assertEqual(source._memory.temporary_bytes, 0)

    def test_reserved_contract_type_null_is_present_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp))
            artifact = next(item["artifact"] for item in manifest["market_input"]["native_inputs"] if item["artifact"]["artifact_id"] == "native-1")
            path = Path(artifact["manifest_uri"]); wire = Document(path.read_text()).to_dict()
            wire["context"].setdefault("coverage", {})["extra"] = {"contract_type": None}
            # Canonical lexical bytes may still contain an invalid reserved
            # node; bind by raw hash so admission, rather than Document, checks it.
            raw = json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            path.write_bytes(raw); reference = "sha256:"+hashlib.sha256(raw).hexdigest()
            for item in manifest["market_input"]["native_inputs"]:
                if item["artifact"]["manifest_uri"] == str(path):
                    item["native_ref"] = reference; item["artifact"]["content_digest"] = reference
            manifest["market_input"]["market_ref"] = logical_ref(manifest["market_input"], "market_ref")
            manifest["request_ref"] = logical_ref(manifest, "request_ref")
            source = StockInputSource()
            with self.assertRaisesRegex(ContractError, "Unsupported reserved contract_type"):
                admitted(source, manifest)
            self.assertFalse(source._audited)

    def test_research_types_are_opaque_only_at_original_fold_configuration_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec={'contract_version':'stock_ml_fold_spec_v3'}
            typed={'contract_type':'TrainingSpec','contract_version':'1','name':'saved original'}
            for case in ('original','wrong_path','wrong_type','wrong_version','whole_document'):
                wire={'definition':{'fold_spec':spec,'training_spec':deepcopy(typed)}}
                if case=='wrong_path':wire['definition']['annotation']=wire['definition'].pop('training_spec')
                if case=='wrong_type':wire['definition']['training_spec']['contract_type']='AccountPolicy'
                if case=='wrong_version':wire['definition']['training_spec']['contract_version']='2'
                raw=json.dumps(wire,sort_keys=True,separators=(',',':')).encode()
                folder=root/case;folder.mkdir();path=folder/'fold.json';path.write_bytes(raw)
                artifact=dict(artifact_type='StockFoldSpec',artifact_id='original',contract_version='stock_ml_fold_spec_v3',
                    manifest_uri=str(path)+'#definition/fold_spec',content_digest=Document.from_dict(spec).identity)
                if case=='whole_document':artifact.update(manifest_uri=str(path),content_digest='sha256:'+hashlib.sha256(raw).hexdigest())
                budget=dict(max_read_bytes=64,max_decoded_bytes=10000)
                if case=='original':
                    index=_CanonicalIndex(artifact,budget)
                    self.assertEqual(index.value(('definition','fold_spec')),spec)
                    self.assertEqual(index.file_digest,'sha256:'+hashlib.sha256(raw).hexdigest())
                else:
                    with self.assertRaises(ContractError):_CanonicalIndex(artifact,budget)

    def test_audit_cumulative_live_budget_rejects_old_25000_byte_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp)); source = StockInputSource()
            with self.assertRaisesRegex(ContractError, "decoded budget"):
                admitted(source, manifest, limits={**LIMITS, "max_block_bytes": 25000})
            self.assertFalse(source._audited)
            self.assertEqual(source._memory.temporary_bytes, 0)

    def test_action_queries_require_market_replay_purpose_for_both_legs(self):
        for name in ("native-4", "native-5"):
            for purpose in ("training", "decision_facts"):
                with self.subTest(name=name, purpose=purpose), tempfile.TemporaryDirectory() as tmp:
                    manifest, _ = source_fixture(Path(tmp))
                    rebind_native(manifest, name, lambda wire: wire["context"]["query"].update(purpose=purpose))
                    with self.assertRaisesRegex(ContractError, "action source role"):
                        admitted(StockInputSource(), manifest)

    def test_equal_wrong_warmup_availability_does_not_admit_either_source(self):
        for name, field in (("warmup-1", "open"), ("warmup-3", "factor")):
            for when in ("2023-12-28T20:31:00+08:00", "invalid-timestamp"):
                with self.subTest(name=name, when=when), tempfile.TemporaryDirectory() as tmp:
                    manifest, _ = source_fixture(Path(tmp))
                    rebind_native(manifest, name, lambda wire: [entry.update(usable_from=when)
                        for entry in wire["field_meta"][field]["by_key"]])
                    with self.assertRaisesRegex(ContractError, "warmup"):
                        admitted(StockInputSource(), manifest)

    def test_inventory_only_stats_and_deduplicates_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp))
            source = StockInputSource()
            with patch.object(Path, "open", side_effect=AssertionError("inventory opened payload")):
                inventory = source.inventory(manifest)
            self.assertEqual(inventory["folds"], 1)
            self.assertEqual(len(inventory["files"]), 13)
            self.assertEqual(inventory["input_bytes"], sum(p.stat().st_size for p in Path(tmp).iterdir()))

    def test_two_blocks_match_original_v6_expanded_values_headers_and_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, plan = source_fixture(Path(tmp))
            source = StockInputSource(); audit = admitted(source, manifest)
            _, _, original_signals, market, calendar, original_rows, profile = validate_stock_request(plan)
            actual_rows, signals = {}, {}
            blocks = list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))
            self.assertEqual(len(blocks), 2)
            for block in blocks:
                actual_rows.update(block.market_rows); signals.update(block.signals)
            self.assertEqual(actual_rows, original_rows)
            self.assertEqual(audit.globals["profile"], profile)
            for key in ("cash_dividends", "action_diagnostics", "action_blocks", "source_refs"):
                self.assertEqual(audit.globals["market_header"][key], market[key])
            for day, (header, rows) in signals.items():
                original, full_rows = original_signals[day]
                self.assertEqual(header, {k: v for k, v in original.items() if k != "rows"})
                feature = calendar[calendar.index(day)-1]
                self.assertEqual(rows, {k: v for k, v in full_rows.items() if k[0] == feature})
            self.assertEqual(audit.receipt["counts"]["market_rows"], len(original_rows))
            self.assertEqual(audit.receipt["counts"]["prediction_rows"], 12)
            self.assertTrue(all(block.bindings for block in blocks))
            daily = list(source.iter_blocks(manifest, block_sessions=1, read_budget=read_budget(LIMITS)))
            self.assertEqual({key: row for b in daily for key, row in b.market_rows.items()}, actual_rows)

    def test_bad_last_block_fails_full_audit_before_any_execution(self):
        def damage(batches, member):
            batches[1]["records"][-1]["open"] = -10.0
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp), full_request(mutate_native=damage).to_dict())
            source = StockInputSource()
            with self.assertRaises(ContractError):
                admitted(source, manifest)
            with self.assertRaisesRegex(ContractError, "complete first-pass"):
                list(source.iter_blocks(manifest, block_sessions=2, read_budget=read_budget(LIMITS)))

    def test_second_pass_detects_replacement_without_rescanning_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp)); source = StockInputSource(); admitted(source, manifest)
            with patch.object(_CanonicalIndex, "__init__", side_effect=AssertionError("rescanned parent")):
                list(source.iter_blocks(manifest, block_sessions=1, read_budget=read_budget(LIMITS)))
            file = Path(manifest["market_input"]["native_inputs"][1]["artifact"]["manifest_uri"])
            raw = file.read_bytes(); file.write_bytes(raw.replace(b'10.0', b'11.0', 1))
            with self.assertRaisesRegex(ContractError, "changed after admission"):
                list(source.iter_blocks(manifest, block_sessions=1, read_budget=read_budget(LIMITS)))

    def test_factor_change_on_block_boundary_retains_original_block(self):
        def damage(batches, member):
            for row in batches[3]["records"]:
                if row["session"] >= "2024-01-03": row["factor"] = 1.1
        with tempfile.TemporaryDirectory() as tmp:
            manifest, plan = source_fixture(Path(tmp), full_request(mutate_native=damage).to_dict())
            audit = admitted(StockInputSource(), manifest)
            self.assertEqual(audit.globals["market_header"]["action_blocks"], plan["market_replay"]["action_blocks"])
            self.assertEqual(len(audit.globals["market_header"]["action_blocks"]), 4)

    def test_multiple_factor_transitions_keep_original_security_major_order(self):
        def damage(batches, member):
            for row in batches[3]["records"]:
                if row["session"] == "2024-01-03": row["factor"] = 1.1
        with tempfile.TemporaryDirectory() as tmp:
            manifest, plan = source_fixture(Path(tmp), full_request(mutate_native=damage).to_dict())
            audit = admitted(StockInputSource(), manifest)
            self.assertEqual(audit.globals["market_header"]["action_blocks"], plan["market_replay"]["action_blocks"])
            self.assertEqual(audit.receipt["counts"]["action_blocks"], 8)

    def test_original_cash_action_and_diagnostic_views_cross_block(self):
        def action_input(batches, member):
            event = legacy.cash_event()
            event["security_id"] = member["context"]["query"]["symbols"][0]
            event["record_date"] = "2024-01-02"; event["ex_date"] = "2024-01-03"
            for i in (4, 5):
                with patch.object(legacy, "DAYS", batches[0]["context"]["query"]["sessions"]), \
                     patch.object(legacy, "SECURITIES", member["context"]["query"]["symbols"]):
                    batches[i] = legacy.batch("corporate_actions", legacy.STOCK_EVENT_FIELDS, [deepcopy(event)],
                        {"cash_dividend_before_tax_per_share": "CNY/share"},
                        time_field="ex_date" if i == 4 else "record_date")
        with tempfile.TemporaryDirectory() as tmp:
            manifest, plan = source_fixture(Path(tmp), full_request(mutate_native=action_input).to_dict())
            audit = admitted(StockInputSource(), manifest)
            for key in ("cash_dividends", "action_diagnostics", "action_blocks"):
                self.assertEqual(audit.globals["market_header"][key], plan["market_replay"][key])
            self.assertEqual(audit.receipt["counts"]["cash_actions"], 1)

    def test_dual_source_provenance_difference_is_not_a_saved_pass_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, _ = source_fixture(root)
            item = next(item for item in manifest["market_input"]["native_inputs"]
                        if item["role"] == "prediction_basis" and item["artifact"]["artifact_id"] == "native-1")
            wire = Document(Path(item["artifact"]["manifest_uri"]).read_text()).to_dict()
            wire["field_meta"]["open"]["by_key"][-1]["revision_sequence"] = 3
            item["artifact"] = artifact_file(root, "different-provenance", wire)
            item["native_ref"] = native_ref(wire)
            manifest["market_input"]["market_ref"] = logical_ref(manifest["market_input"], "market_ref")
            manifest["request_ref"] = logical_ref(manifest, "request_ref")
            with self.assertRaisesRegex(ContractError, "pairing difference"):
                admitted(StockInputSource(), manifest)

    def test_large_coverage_is_scanned_without_allocating_its_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = artifact_file(Path(tmp), "native-coverage", {"context": {"coverage": {"many": ["x"*100]*1000}},
                "records": [], "field_meta": {}})
            original = __import__("json").loads
            lengths = []
            def bounded(raw, *args, **kwargs):
                lengths.append(len(raw))
                self.assertLessEqual(len(raw), 128)
                return original(raw, *args, **kwargs)
            with patch("axiom_engine.runtime.stock_stream_inputs.json.loads", side_effect=bounded):
                indexed = _CanonicalIndex(artifact, dict(max_read_bytes=16, max_decoded_bytes=512))
            self.assertGreater(indexed.size, 100000)
            self.assertTrue(lengths)
            self.assertEqual(indexed.groups, {})

    def test_budget_guards_reject_before_payload_open_or_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = source_fixture(Path(tmp))
            source = StockInputSource()
            with patch.object(Path, "open", side_effect=AssertionError("opened before inventory guard")):
                with self.assertRaisesRegex(ContractError, "inventory"):
                    admitted(source, manifest, limits={**LIMITS, "max_input_bytes": 1})
            artifact = artifact_file(Path(tmp), "huge-atom", {"value": "x"*10000})
            with self.assertRaisesRegex(ContractError, "cumulative decoded budget"):
                _CanonicalIndex(artifact, dict(max_read_bytes=16, max_decoded_bytes=128))
            for size in (True, 0, -1):
                with self.assertRaises(ContractError): admitted(source, manifest, block_sessions=size)


if __name__ == "__main__":
    unittest.main()
