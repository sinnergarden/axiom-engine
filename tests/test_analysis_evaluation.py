"""Hand arithmetic for saved-only analysis, boundaries and v1/v2/v3 readers."""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Context, Decimal, Inexact, ROUND_DOWN, getcontext, localcontext, setcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (EvaluationReport, BenchmarkSeries, analysis_evaluation_spec,
    evaluate_saved_analysis, evaluate_backtest, long_history_evaluation_spec,
    daily_evaluation_spec, save_backtest_evaluation, load_backtest_evaluation,
    stock_portfolio_policy, run_backtest, BacktestRequest)
from axiom_engine.runtime.analysis_evaluation import (_return_distribution, _execution_summary,
    _concentration, _identity)
from axiom_engine.runtime.analysis_benchmarks import _project_native
from test_evaluation import saved, trade, benchmark, rehash
from test_stocks import request as stock_request, benchmark as stock_benchmark

RF=dict(currency="CNY",annual_effective_rate="0",source="EXPLICIT_ZERO_ASSUMPTION")


def wealth(values, *, anchor="2020-01-01", step=1):
    first=date.fromisoformat(anchor)
    days=[anchor,*[(first+timedelta(days=step*(i+1))).isoformat() for i in range(len(values))]]
    run=saved([trade(days[1],"BUY","A",1000,"10")],days=days)
    wire=run.to_dict()
    with localcontext(Context(prec=40)):
        for nav,pos,value in zip(wire["nav"],wire["positions"],values):
            nav.update(nav_minor=value,market_value_minor=value,nav_index=str(Decimal(value)/1000000))
            pos["market_value_minor"]=value
    return rehash(wire)


def analysis(run, rf=None):
    bench=benchmark(run)
    base=evaluate_backtest(run,benchmark=bench,spec=long_history_evaluation_spec())
    return evaluate_saved_analysis(run,base,benchmarks=dict(CSI300=bench,SSE_COMPOSITE=None,NASDAQ100=None),
        spec=analysis_evaluation_spec(risk_free=RF if rf is None else rf)),base


def rehash_report(wire):
    wire.pop("content_digest",None)
    wire["evaluation_ref"]=_identity(wire)
    wire["content_digest"]=Document.from_dict(wire).identity
    return EvaluationReport.from_dict(wire)


class AnalysisTests(unittest.TestCase):
    def test_factory_requires_explicit_cny_rate_and_source(self):
        with self.assertRaises(TypeError): analysis_evaluation_spec()
        for bad in (None,{},dict(RF,currency="USD"),dict(RF,annual_effective_rate="-1"),
                    dict(RF,annual_effective_rate="NaN"),dict(RF,annual_effective_rate=0),dict(RF,source="")):
            with self.subTest(bad=bad),self.assertRaises(ContractError): analysis_evaluation_spec(risk_free=bad)
        self.assertNotEqual(analysis_evaluation_spec(risk_free=RF).identity,
                            analysis_evaluation_spec(risk_free=dict(RF,annual_effective_rate="0.03")).identity)

    def test_drawdown_first_deepest_trough_recent_equal_peak_and_recovery(self):
        report,_=analysis(wealth([1000000,1200000,1200000,900000,1000000,900000,1200000]))
        d=report.to_dict()["drawdown_interval"]
        self.assertEqual(d,dict(status="AVAILABLE",peak_session="2020-01-04",trough_session="2020-01-05",
            peak_nav_minor=1200000,trough_nav_minor=900000,drawdown="-0.25",elapsed_calendar_days=1,
            peak_is_initial_anchor=False,recovery_session="2020-01-08",recovery_status="RECOVERED"))
        report,_=analysis(wealth([999500,990000]))
        d=report.to_dict()["drawdown_interval"]
        self.assertEqual((d["peak_session"],d["peak_is_initial_anchor"],d["recovery_session"],d["recovery_status"]),
                         ("2020-01-01",True,None,"OPEN"))
        self.assertEqual(analysis(wealth([1000000]*2))[0].to_dict()["drawdown_interval"]["status"],"NO_DRAWDOWN")

    def test_sharpe_hand_sample_frequency_and_rf_leap_fraction(self):
        # Forty observed returns over 400 calendar days, with two repeated returns.
        values=[1000000 if i%2 else 1100000 for i in range(40)]
        report,base=analysis(wealth(values,step=10))
        with localcontext(Context(prec=40)):
            returns=[Decimal("0.1") if i%2==0 else Decimal(1000000)/1100000-1 for i in range(40)]
            mean=sum(returns)/40
            std=(sum((r-mean)**2 for r in returns)/39).sqrt()
            years=Decimal(366)/366+Decimal(34)/365
            expected=mean/std*(Decimal(40)/years).sqrt()
            risk=report.to_dict()["risk_metrics"]["sharpe"]
            self.assertEqual(risk["year_fraction"],str(years))
            self.assertEqual(Decimal(risk["value"]),expected)
        nonzero,_=analysis(wealth(values,step=10),dict(RF,annual_effective_rate="0.05",source="TEST_RATE"))
        self.assertLess(Decimal(nonzero.to_dict()["risk_metrics"]["sharpe"]["value"]),Decimal(risk["value"]))
        self.assertEqual(base.to_dict()["period_metrics"]["account"]["max_drawdown"],report.to_dict()["drawdown_interval"]["drawdown"])

    def test_risk_status_precedence_zero_vol_zero_dd_and_negative_calmar(self):
        for values,step,status in (([1000000]*40,1,"INSUFFICIENT_SPAN"),
            ([1000000]*29,14,"INSUFFICIENT_OBSERVATIONS"),([1000000]*40,10,"ZERO_VOLATILITY"),
            ([1000000,0,0],1,"MISSING_RETURN")):
            report,_=analysis(wealth(values,step=step))
            r=report.to_dict()["risk_metrics"]["sharpe"]
            self.assertEqual((r["status"],r["value"]),(status,None))
        self.assertEqual(analysis(wealth([1000000]*40,step=10))[0].to_dict()["risk_metrics"]["calmar"],dict(status="ZERO_DRAWDOWN",value=None))
        report,base=analysis(wealth([1000000]*39+[900000],step=10))
        calmar=report.to_dict()["risk_metrics"]["calmar"]
        self.assertEqual(calmar["status"],"AVAILABLE");self.assertLess(Decimal(calmar["value"]),0)
        short=analysis(wealth([0]))[0].to_dict()
        self.assertEqual(short["risk_metrics"]["calmar"],dict(status="INSUFFICIENT_SPAN",value=None))
        self.assertEqual(short["concentration_series"][0]["status"],"ZERO_NAV")

    def test_distribution_open_tails_zero_and_20_session_anchor(self):
        spec=analysis_evaluation_spec(risk_free=RF).to_dict()
        values=["-0.21","-0.20","-0.18","-0.0001","0","0.0199","0.02","0.1999","0.20","0.30"]
        raw=dict(episodes=[dict(statistics_eligible=True,net_return=v) for v in values])
        with localcontext(Context(prec=40)):
            dist=_return_distribution(raw,spec["return_distribution"])
        self.assertEqual(len(dist["bins"]),22)
        self.assertEqual([dist["bins"][i]["count"] for i in (0,1,2,10,11,12,20,21)],[1,1,1,1,2,1,1,2])
        self.assertEqual(sum(b["count"] for b in dist["bins"]),10)
        self.assertEqual(_return_distribution(dict(episodes=raw["episodes"][:9]),spec["return_distribution"])["bins"],[])
        report,_=analysis(wealth([1000000+i*1000 for i in range(21)]))
        points=report.to_dict()["analysis_series"]
        self.assertIsNone(points[18]["rolling_return_20"])
        self.assertEqual(points[19]["rolling_return_20"],"0.019")
        self.assertEqual(points[20]["rolling_return_20"],"0.02")
        self.assertEqual(points[20]["account_cumulative_return"],"0.02")
        self.assertIsNotNone(points[19]["rolling_volatility_20"])

    def test_exact_date_missing_points_and_native_usd_curve_without_fx(self):
        report,base=analysis(wealth([1000000]*3))
        native=benchmark(wealth([1000000]*3)).to_dict()
        native.update(currency="USD",timezone="America/New_York")
        native["rows"].pop(1)  # Native calendar gap must not reuse the anchor.
        native["calendar"].pop(1)
        projected=_project_native(base.to_dict(),native,"sha256:"+"b"*64)
        self.assertEqual(projected["series"][0]["relative_status"],"MISSING_OBSERVATION")
        self.assertIsNone(projected["series"][0]["native_session"])
        self.assertEqual(projected["series"][1]["relative_status"],"FX_REQUIRED")
        self.assertIsNone(projected["series"][1]["account_relative_wealth"])
        self.assertEqual(projected["series"][1]["normalized_index"],"1.02")
        w=report.to_dict()
        self.assertEqual(w["benchmark_refs"]["NASDAQ100"],None)
        self.assertEqual(w["benchmark_comparisons"]["NASDAQ100"]["status"],"SOURCE_UNAVAILABLE")

    def test_genuine_decision_intent_order_fill_links_and_economics(self):
        plan=stock_request().to_dict()
        plan["portfolio_policy"]=stock_portfolio_policy(top_k=3,execution_universe=plan["execution_universe"])
        run=run_backtest(BacktestRequest.from_dict(plan));b=stock_benchmark()
        base=evaluate_backtest(run,benchmark=b,spec=long_history_evaluation_spec())
        report=evaluate_saved_analysis(run,base,benchmarks=dict(CSI300=b,SSE_COMPOSITE=None,NASDAQ100=None),spec=analysis_evaluation_spec(risk_free=RF))
        wire=report.to_dict();raw=run.to_dict()
        for point,decision in zip(wire["execution_trace"],raw["decisions"]):
            self.assertEqual(point["targets"],decision["targets"])
            self.assertEqual(point["trace"],decision["trace"])
            for link in point["intent_links"]:
                expected=[o for o in raw["orders"] if o["intent_id"]==link["intent_id"]]
                self.assertEqual(len(link["orders"]),len(expected))
                for order,original in zip(link["orders"],expected):
                    self.assertEqual(order["requested_quantity"],original["quantity"])
                    self.assertEqual(order["committed_sequence"],original.get("committed_sequence"))
                    self.assertEqual(order["fills"],[f for f in raw["fills"] if f["order_id"]==original["order_id"]])
        summary=wire["execution_summary"]
        with localcontext(Context(prec=40)):
            self.assertEqual(Decimal(summary["two_sided_turnover"]),Decimal(sum(f["gross_minor"] for f in raw["fills"]))*len(raw["nav"])/sum(p["nav_minor"] for p in raw["nav"]))
            self.assertEqual(Decimal(summary["fee_ratio"]),Decimal(sum(f["fee_minor"] for f in raw["fills"]))/1000000)
        strict=stock_request(strict=True).to_dict();strict["portfolio_policy"]=plan["portfolio_policy"]
        r=run_backtest(BacktestRequest.from_dict(strict));b=stock_benchmark()
        e=evaluate_backtest(r,benchmark=b,spec=long_history_evaluation_spec())
        s=evaluate_saved_analysis(r,e,benchmarks=dict(CSI300=b,SSE_COMPOSITE=None,NASDAQ100=None),spec=analysis_evaluation_spec(risk_free=RF)).to_dict()
        self.assertTrue(all(o["fill_ids"]==[] for d in s["execution_trace"] for i in d["intent_links"] for o in i["orders"]))
        self.assertEqual(_execution_summary(dict(fills=[],nav=[dict(nav_minor=0)],initial_nav_minor=100))["turnover_status"],"ZERO_MEAN_NAV")

    def test_run_identity_base_binding_reader_versions_and_no_execution(self):
        run=wealth([1000000]*40,step=10);report,base=analysis(run)
        b=benchmark(run);mapping=dict(CSI300=b,SSE_COMPOSITE=None,NASDAQ100=None)
        with patch("axiom_engine.runtime.backtest.run_backtest",side_effect=AssertionError("replay")), \
             patch("axiom_engine.runtime.evaluation.evaluate_backtest",side_effect=AssertionError("old evaluator")):
            repeated=evaluate_saved_analysis(run,base,benchmarks=mapping,spec=analysis_evaluation_spec(risk_free=RF))
        self.assertEqual(repeated.payload,report.payload)
        self.assertEqual(report.to_dict()["base_evaluation"],base.to_dict())
        with self.assertRaises(ContractError):
            evaluate_saved_analysis(wealth([900000]*40,step=10),base,benchmarks=mapping,spec=analysis_evaluation_spec(risk_free=RF))
        changed=dict(mapping,CSI300=benchmark(wealth([1000000]*3)))
        with self.assertRaises(ContractError):
            evaluate_saved_analysis(run,base,benchmarks=changed,spec=analysis_evaluation_spec(risk_free=RF))
        v1=evaluate_backtest(run,benchmark=b,spec=daily_evaluation_spec())
        with tempfile.TemporaryDirectory() as tmp:
            for name,value in (("v1",v1),("v2",base),("v3",report)):
                path=Path(tmp)/name;save_backtest_evaluation(value,path)
                with patch("axiom_engine.runtime.analysis_evaluation._risk",side_effect=AssertionError("reader computes")), \
                     patch("axiom_engine.runtime.analysis_evaluation._drawdown",side_effect=AssertionError("reader computes")):
                    self.assertEqual(load_backtest_evaluation(path).payload,value.payload)
            wire=report.to_dict();wire["benchmark_refs"]["CSI300"]="sha256:"+"b"*64
            path=Path(tmp)/"bad";path.write_text(rehash_report(wire).payload)
            with self.assertRaises(ContractError):load_backtest_evaluation(path)
        before=getcontext().copy()
        try:
            getcontext().prec=6;getcontext().rounding=ROUND_DOWN;getcontext().traps[Inexact]=True
            self.assertEqual(evaluate_saved_analysis(run,base,benchmarks=mapping,spec=analysis_evaluation_spec(risk_free=RF)).payload,report.payload)
        finally:setcontext(before)

    def test_factory_and_reader_ignore_low_precision_decimal_context(self):
        run=wealth([1000000]*40,step=10);report,_=analysis(run)
        expected=analysis_evaluation_spec(risk_free=RF).payload
        before=getcontext().copy()
        try:
            getcontext().prec=1;getcontext().rounding=ROUND_DOWN;getcontext().traps[Inexact]=True
            self.assertEqual(analysis_evaluation_spec(risk_free=RF).payload,expected)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"v3";save_backtest_evaluation(report,path)
                self.assertEqual(load_backtest_evaluation(path).payload,report.payload)
        finally:setcontext(before)

    def test_reader_refuses_unadmitted_new_benchmark_even_after_rebinding(self):
        report,_=analysis(wealth([1000000]*2))
        wire=report.to_dict();native=dict(contract_version="not_an_admitted_benchmark",made_up=True)
        ref=Document.from_dict(native).identity
        wire["benchmark_inputs"]["SSE_COMPOSITE"]=native;wire["benchmark_refs"]["SSE_COMPOSITE"]=ref
        wire["benchmark_comparisons"]["SSE_COMPOSITE"]=dict(wire["benchmark_comparisons"]["CSI300"],input_ref=ref)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad";path.write_text(rehash_report(wire).payload)
            with self.assertRaises(ContractError):load_backtest_evaluation(path)

    def test_reader_checks_watermark_and_original_benchmark_facts(self):
        report,_=analysis(wealth([1000000]*3))
        for change in ("watermark","close","native_close","currency"):
            wire=report.to_dict();comparison=wire["benchmark_comparisons"]["CSI300"]
            if change=="watermark":wire["concentration_series"][0]["committed_sequence"]=999999
            elif change=="close":comparison["series"][0]["close"]="12345"
            elif change=="native_close":comparison["native_series"][0]["close"]="12345"
            else:
                comparison["currency"]="EUR"
                for point in comparison["series"]:point["account_relative_wealth"]=None
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"bad";path.write_text(rehash_report(wire).payload)
                with self.assertRaises(ContractError):load_backtest_evaluation(path)

    def test_reader_checks_saved_risk_eligibility_and_partial_state(self):
        report,_=analysis(wealth([1000000]*3))
        for change in ("sharpe","calmar","complete"):
            wire=report.to_dict()
            if change=="sharpe":wire["risk_metrics"]["sharpe"].update(status="AVAILABLE",value="99",annualization_factor="1")
            elif change=="calmar":wire["risk_metrics"]["calmar"].update(status="AVAILABLE",value="1")
            else:wire["status"]="COMPLETE"
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"bad";path.write_text(rehash_report(wire).payload)
                with self.assertRaises(ContractError):load_backtest_evaluation(path)
