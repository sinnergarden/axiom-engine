"""Owner-calculated v2 benchmark drawdown projections and saved v3 compatibility."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.runtime import (BenchmarkSeries, analysis_evaluation_spec, evaluate_saved_analysis,
    save_backtest_evaluation, load_backtest_evaluation)
from test_analysis_evaluation import wealth, analysis, RF, rehash_report
from test_retrospective_benchmark import native, admitted


class BenchmarkDrawdownTests(unittest.TestCase):
    def setUp(self):
        self.run=wealth([1100000,1200000,1150000])
        self.old,self.base=analysis(self.run)

    def report(self, values):
        batch,receipt=native(self.base.to_dict(),values)
        return evaluate_saved_analysis(self.run,self.base,benchmarks={
            "CSI300":BenchmarkSeries.from_dict(self.base.to_dict()["benchmark_input"]),
            "SSE_COMPOSITE":admitted(batch,receipt),"NASDAQ100":None},
            spec=analysis_evaluation_spec(risk_free=RF,benchmark_projection_version="benchmark_comparison_v2"))

    def test_anchor_repeated_peak_and_maximum_with_original_csi_values(self):
        result=self.report([100.,120.,108.,120.]).to_dict()
        sse=result["benchmark_comparisons"]["SSE_COMPOSITE"]
        self.assertEqual([p["benchmark_drawdown"] for p in sse["native_series"]],["0","0","-0.1","0"])
        self.assertEqual(sse["max_drawdown"],"-0.1")
        self.assertEqual([p["benchmark_drawdown"] for p in sse["series"]],["0","-0.1","0"])
        csi=result["benchmark_comparisons"]["CSI300"];owner=self.base.to_dict()["benchmark"]
        self.assertEqual([p["benchmark_drawdown"] for p in csi["series"]],[p["drawdown"] for p in owner["series"]])
        self.assertEqual(csi["max_drawdown"],owner["max_drawdown"])
        unavailable=result["benchmark_comparisons"]["NASDAQ100"]
        self.assertEqual(unavailable["projection_version"],"benchmark_comparison_v2")
        self.assertIsNone(unavailable["max_drawdown"])
        self.assertEqual(unavailable["series"],[])
        self.assertNotEqual(result["spec_ref"],self.old.to_dict()["spec_ref"])
        self.assertEqual(result["base_evaluation_ref"],self.base.to_dict()["evaluation_ref"])

    def test_missing_peak_stays_null_even_when_cumulative_return_resumes(self):
        result=self.report([100.,120.,None,130.]).to_dict()["benchmark_comparisons"]["SSE_COMPOSITE"]
        self.assertEqual([p["benchmark_drawdown"] for p in result["native_series"]],["0","0",None,None])
        self.assertIsNone(result["max_drawdown"])
        self.assertEqual(result["series"][-1]["benchmark_cumulative_return"],"0.3")
        missing=self.report([None,120.,108.,120.]).to_dict()["benchmark_comparisons"]["SSE_COMPOSITE"]
        self.assertTrue(all(p["benchmark_drawdown"] is None for p in missing["native_series"]))
        self.assertIsNone(missing["max_drawdown"])

    def test_saved_loader_never_calls_projection_or_analytical_functions(self):
        report=self.report([100.,120.,108.,120.])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save_backtest_evaluation(report,root/"new.json");save_backtest_evaluation(self.old,root/"old.json")
            with patch("axiom_engine.runtime.analysis_benchmarks._add_drawdowns",side_effect=AssertionError("recomputed")), \
                 patch("axiom_engine.runtime.analysis_benchmarks._project_native",side_effect=AssertionError("reprojected")), \
                 patch("axiom_engine.runtime.analysis_evaluation._drawdown",side_effect=AssertionError("recomputed account")):
                self.assertEqual(load_backtest_evaluation(root/"new.json").payload,report.payload)
                self.assertEqual(load_backtest_evaluation(root/"old.json").payload,self.old.payload)
        self.assertNotIn("benchmark_projection_version",self.old.to_dict()["spec"])
        self.assertNotIn("projection_version",self.old.to_dict()["benchmark_comparisons"]["CSI300"])

    def test_resealed_projection_marker_gap_and_csi_owner_mismatch_are_rejected(self):
        original=self.report([100.,120.,None,130.]).to_dict()
        changes=[lambda w:w["benchmark_comparisons"]["SSE_COMPOSITE"].update(projection_version="benchmark_comparison_v1"),
                 lambda w:w["benchmark_comparisons"]["SSE_COMPOSITE"]["native_series"][-1].update(benchmark_drawdown="0"),
                 lambda w:w["benchmark_comparisons"]["CSI300"]["series"][0].update(benchmark_drawdown="-0.01"),
                 lambda w:w["benchmark_comparisons"]["SSE_COMPOSITE"].update(max_drawdown="0.1")]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.json"
            for change in changes:
                w=deepcopy(original);change(w);path.write_text(rehash_report(w).payload)
                with self.assertRaises(ContractError):load_backtest_evaluation(path)

    def test_spec_projection_version_rejects_unknown_and_bool(self):
        for value in (True,None,"benchmark_comparison_v3"):
            with self.assertRaises(ContractError):analysis_evaluation_spec(risk_free=RF,benchmark_projection_version=value)
