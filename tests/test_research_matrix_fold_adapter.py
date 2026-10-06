"""Opt-in, tiny cross-repository public writer acceptance; no real ML or Data.

Set AXIOM_RESEARCH_SOURCE to the explicitly fixed Research source checkout.
Research's existing synthetic Data/Feature/model fixtures replace expensive
stages; its production public prepare/build/save/load paths create the artifacts.
This verifies Engine's original saved prediction closure, not native market or
account admission. Research is not an Engine runtime dependency.
"""
import builtins
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_portfolio import StockPredictionFrame, validate_stock_predictions
from axiom_engine.runtime.stock_schedule import stock_prediction_schedule
from axiom_engine.runtime.stock_stream_inputs import StockInputSource, _DecodedMemory, _frames


@unittest.skipUnless(os.environ.get("AXIOM_RESEARCH_SOURCE"), "explicit Research source required")
class ResearchMatrixFoldAdapterTests(unittest.TestCase):
    def test_public_writer_v3_original_saved_closure_for_both_spec_versions(self):
        research = Path(os.environ["AXIOM_RESEARCH_SOURCE"]).resolve()
        sys.path[:0] = [str(research/"src"), str(research/"tests")]
        import axiom_research
        self.assertTrue(Path(axiom_research.__file__).resolve().is_relative_to(research))
        from axiom_research import (prepare_stock_ml_batch_inputs, load_stock_ml_batch_inputs,
            build_stock_ml_fold_from_saved_inputs, load_stock_ml_fold)
        from axiom_research.stock_artifacts import _read, file_digest
        from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query
        from test_stock_folds import backend
        receipts = []
        for version in ("stock_ml_fold_spec_v1", "stock_ml_fold_spec_v2"):
            with self.subTest(spec=version), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                # Artificial sparse sessions keep two calendar years small;
                # they never represent a real exchange calendar.
                with patch("test_stock_feature_inputs.timedelta",
                           side_effect=lambda *, days: timedelta(days=days*(6 if version.endswith("v2") else 1))):
                    fixture = PrepareFeatureFixture(root)
                spec = fixture.folds()[0]
                if version.endswith("v2"):
                    spec.update(contract_version=version, training_window={"unit": "calendar_years", "length": 2,
                        "end": "previous_fit_session", "start": "fit_date_minus_years_inclusive", "leap_day": "clamp_feb_28"})
                saved = fixture.matrix(); data = PublicDataFixture(fixture.spec)
                module = types.ModuleType("axiom_data"); module.QuerySpec = Query; module.adjust_prices = data.adjust
                with patch.dict(sys.modules, {"axiom_data": module}), \
                     patch("axiom_research.stock_ml._implementation", return_value=fixture.implementation), \
                     patch("axiom_research.stock_ml._environment", return_value=fixture.environment):
                    inputs = prepare_stock_ml_batch_inputs(data, feature_inputs=saved, fold_specs=[spec],
                        destination=root/"prepared", preparation_options={"row_block_sessions": 32,
                            "column_block": 32, "maximum_resident_bytes": 64*1024**2,
                            "normalization_backend": "core_cs_batch_v1"})
                saved.close(); fold = inputs["folds"][0]; metrics = {}
                with load_stock_ml_batch_inputs(inputs) as batch, \
                     patch("axiom_research.feature_catalog.load_feature_catalog", return_value=fixture.catalog), \
                     patch("axiom_research.stock_ml._implementation", return_value=fixture.implementation), \
                     patch("axiom_research.stock_ml._environment", return_value=fixture.environment), \
                     patch("axiom_research.stock_training.fit_predict_stock_model", side_effect=backend) as fake_model:
                    published = build_stock_ml_fold_from_saved_inputs(fold["input_manifest"],
                        fold_spec=spec, destination=root/"folds", metrics=metrics, batch=batch)
                    self.assertEqual(fake_model.call_count, 1)
                before = {p: (file_digest(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in root.rglob("*") if p.is_file()}
                self.assertLess(sum(size for _, size, _ in before.values()), 16*1024**2)
                with patch("axiom_research.stock_training.fit_predict_stock_model", side_effect=AssertionError("readonly model executed")):
                    loaded = load_stock_ml_fold(published.path)
                wrapper = loaded.to_dict(); model = loaded.model(); predictions = loaded.predictions()
                descriptor = _read(loaded.path/"manifest.json")
                self.assertEqual((descriptor["contract_version"], wrapper["contract_version"], spec["contract_version"]),
                                 ("stock_ml_fold_manifest_v2", "stock_ml_fold_v3", version))
                self.assertEqual(loaded.identity, published.identity)
                def artifact(name, wire, selector=""):
                    return {"artifact_type": name, "artifact_id": name, "contract_version": wire["contract_version"],
                        "manifest_uri": str(loaded.path/name)+selector, "content_digest": Document.from_dict(wire).identity}
                item = {name: wrapper[name] for name in ("fold_ref", "model_ref", "feature_ref", "signal_run_ref")}
                item.update(fold_spec_ref=Document.from_dict(spec).identity,
                    fold_spec_artifact=artifact("fold.json", spec, "#definition/fold_spec"),
                    model_metadata_artifact=artifact("model.json", model),
                    prediction_artifact=artifact("predictions.json", predictions))
                trades = spec["oos_trade_sessions"]
                manifest = {"scope": {"calendar": fixture.calendar, "prediction_universe": predictions["universe"],
                    "start_session": trades[0], "end_session": trades[-1]}, "prediction_input": {"frames": [item]}}
                source = StockInputSource(); budget = {"max_read_bytes": 4096, "max_decoded_bytes": 1024**2}
                source._memory = _DecodedMemory(budget["max_decoded_bytes"])
                allowed = {loaded.path/name for name in ("manifest.json", "fold.json", "model.json", "predictions.json")}
                source._inventory_sizes = {str(p): p.stat().st_size for p in allowed}
                prediction_path = str(loaded.path/"predictions.json")
                source._scan_row_limits[prediction_path] = len(predictions["rows"])
                source._prediction_paths = {prediction_path}; source._prediction_remaining = len(predictions["rows"])
                original_open = Path.open; original_import = builtins.__import__; opened = set()
                def checked_open(path, *args, **kwargs):
                    self.assertIn(path, allowed, "Engine opened training or other unselected parent")
                    opened.add(path); return original_open(path, *args, **kwargs)
                def checked_import(name, *args, **kwargs):
                    if name.startswith(("axiom_research", "axiom_data", "qlib", "lightgbm", "pandas")):
                        raise AssertionError("Engine imported upstream executor "+name)
                    return original_import(name, *args, **kwargs)
                with patch.object(Path, "open", checked_open), patch("builtins.__import__", side_effect=checked_import):
                    frames, trade_map = _frames(source, manifest, budget)
                    frame = frames[0]; restored = []
                    for trade in trades:
                        feature = trade_map[trade][1]
                        with source._memory.stage() as stage:
                            rows, _, _ = frame["index"].rows(("rows",), [feature], budget, reservation=stage)
                            _, checked = validate_stock_predictions(StockPredictionFrame.from_dict({**frame["header"], "rows": rows}))
                            restored.extend(checked.values())
                    schedule = stock_prediction_schedule(folds=[{"fold_ref": loaded.identity, "fold_spec": spec,
                        "model": model, "prediction_frame": predictions}], calendar=fixture.calendar)
                self.assertEqual(restored, predictions["rows"])
                self.assertEqual(opened, allowed)
                self.assertEqual(frame["parent_binding"]["file_digest"], descriptor["files"]["fold.json"])
                self.assertEqual(frame["header"]["signal_run_ref"], predictions["signal_run_ref"])
                self.assertFalse(source._audited, "prediction-only closure is not full native source admission")
                self.assertEqual(before, {p: (file_digest(p), p.stat().st_size, p.stat().st_mtime_ns) for p in before})
                receipts.append({"versions": [descriptor["contract_version"], wrapper["contract_version"], version],
                    "fold_ref": loaded.identity, "fold_spec_ref": item["fold_spec_ref"], "signal_run_ref": item["signal_run_ref"],
                    "schedule_ref": schedule.to_dict()["schedule_ref"], "saved_file_digests": descriptor["files"],
                    "synthetic_total_bytes": sum(size for _, size, _ in before.values()),
                    "training_rows": _read(loaded.path/"dataset.json")["training_row_count"],
                    "prediction_rows": len(restored), "engine_opened_files": sorted(p.name for p in opened),
                    "original_sha_size_mtime_unchanged": True, "fake_backend_calls": 1,
                    "real_supplier_feature_fit_predict_account_calls": 0, "full_native_source_admission": False})
        if os.environ.get("AXIOM_MATRIX_FOLD_RECEIPT"):
            Path(os.environ["AXIOM_MATRIX_FOLD_RECEIPT"]).write_text(json.dumps({
                "scope": "synthetic public Research writer/loader to Engine original prediction closure",
                "implementation_ref": IMPLEMENTATION_REF, "research_source": str(research), "cases": receipts},
                ensure_ascii=False, sort_keys=True, indent=2)+"\n")


if __name__ == "__main__":
    unittest.main()
