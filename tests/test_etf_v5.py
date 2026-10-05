"""Small synthetic v5 policies through the single owner Runtime and loaders."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, daily_open_profile, etf_buy_and_hold_policy,
    BacktestRun, run_backtest, save_backtest_run, load_backtest_run, evaluate_backtest, long_history_evaluation_spec,
    analysis_evaluation_spec, evaluate_saved_analysis, save_backtest_evaluation, load_backtest_evaluation,
    build_fill_display, save_fill_display, load_fill_display)
from axiom_engine.runtime.unit_splits import UNIT_SPLIT_POLICY
from test_backtest import etf_fixture, DAYS, REF
from test_unit_splits import split_fixture, bind_event
from test_evaluation import benchmark, scope
from test_analysis_evaluation import RF


def request(*, hold=False, bps="5", limits="require_both", price="0.500", cash=50000000):
    w = etf_fixture()
    w.update(contract_version="backtest_request_v5", unit_split_policy=UNIT_SPLIT_POLICY,
        price_unit="CNY/fund unit", portfolio_policy={"contract_version":"etf_rotation_policy_v1",
        "schedule":"weekly_first_trading_session"}, initial_account={"cash_minor":cash,"positions":{}})
    w["market_replay"].update(contract_version="market_replay_v2", unit_splits=[],
        source_evidence=[{"reference":REF,"context":{"snapshot_id":"s_synthetic","query":{"purpose":"market_replay"}}}])
    for row in w["market_replay"]["rows"]:
        row.update(open=price, close=price, volume_units="1000000000", limit_down="0.001")
    w["profile"] = daily_open_profile(unknown_status_policy="etf_daily_observed", price_grid_policy="etf_price_grid_v1",
        price_limit_policy=limits, slippage_bps=bps)
    if hold:
        w["portfolio_policy"] = etf_buy_and_hold_policy(security_id=w["market_replay"]["universe"][0], entry_session=w["start_session"])
        w["signal_frame"] = None
    return w


def run(w):
    return run_backtest(BacktestRequest.from_dict(w))


class ETFV5Tests(unittest.TestCase):
    def test_legacy_factory_exact_bytes_and_explicit_new_policies(self):
        old = etf_fixture()["profile"]
        self.assertEqual(Document.from_dict(old).payload, Document.from_dict(daily_open_profile(unknown_status_policy="etf_daily_observed")).payload)
        self.assertNotIn("price_grid", old)
        for kwargs in ({"slippage_bps":"5"},{"price_limit_policy":"known_only"},{"slippage_bps":True},
                       {"price_grid_policy":"etf_price_grid_v1","slippage_bps":"2"}):
            with self.assertRaises(ContractError): daily_open_profile(**kwargs)

    def test_controlled_zero_vs_five_with_actual_rounded_price_and_cash(self):
        from axiom_engine.runtime.accounting import AccountLedger
        zero_plan, five_plan = request(bps="0"), request()
        zero_plan["account_id"], five_plan["account_id"] = "synthetic-etf-zero", "synthetic-etf-five"
        zero = run(zero_plan).to_dict()
        with patch("axiom_engine.runtime.backtest.AccountLedger", wraps=AccountLedger) as ledger:
            five = run(five_plan).to_dict()
        self.assertEqual(ledger.call_count, 1)
        self.assertEqual(zero["signal_ref"], five["signal_ref"])
        self.assertEqual(zero["market_ref"], five["market_ref"])
        self.assertNotEqual(zero["plan"]["account_id"], five["plan"]["account_id"])
        self.assertNotEqual(zero["run_id"], five["run_id"])
        self.assertEqual(zero["fills"][0]["price"], "0.500")
        self.assertEqual(five["fills"][0]["price"], "0.501")
        self.assertEqual(Decimal(five["fills"][0]["effective_slippage_bps"]), 20)
        self.assertEqual(next(f for f in five["fills"] if f["side"] == "SELL")["price"], "0.499")
        self.assertEqual(five["final_account"]["cash_minor"], 50000000 + sum(f["cash_delta_minor"] for f in five["fills"]))
        for f in five["fills"]:
            self.assertEqual(f["cash_delta_minor"], (-f["gross_minor"]-f["fee_minor"]) if f["side"]=="BUY" else f["gross_minor"]-f["fee_minor"])

    def test_hold_never_consumes_signal_and_partial_fill_expires_once(self):
        w = request(hold=True)
        for row in w["market_replay"]["rows"]:
            if row["session"] == w["start_session"]: row["volume_units"]="1000"
        with patch("axiom_engine.runtime.backtest.validate_signals", side_effect=AssertionError("Signal invented")), \
             patch("axiom_engine.runtime.backtest.plan_rotation", side_effect=AssertionError("rotation called")):
            result = run(w).to_dict()
        self.assertIsNone(result["signal_ref"])
        self.assertEqual(result["core_version"], "axiom.etf_buy_and_hold/1")
        self.assertEqual(len(result["decisions"]), 1)
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["status"], "PARTIAL_EXPIRED")
        self.assertEqual(result["fills"][0]["quantity"], 100)
        self.assertEqual(result["final_account"]["positions"][w["portfolio_policy"]["security_id"]]["quantity"],100)

    def test_off_grid_limit_after_rounding_and_cash_after_rounding(self):
        cases = [("0.5005",50000000,None,"PRICE_TICK"),("0.500",50000000,"0.501","PRICE_LIMIT"),
                 ("1.000",10000,None,"INSUFFICIENT_CASH")]
        for price,cash,upper,reason in cases:
            with self.subTest(reason=reason):
                w=request(hold=True,price=price,cash=cash)
                if upper:
                    for row in w["market_replay"]["rows"]:
                        if row["session"]==w["start_session"]:row["limit_up"]=upper
                result=run(w).to_dict()
                self.assertEqual(result["fills"],[])
                self.assertEqual(result["orders"][0]["reason"],reason)
                self.assertEqual(len(result["orders"]),1)
                self.assertEqual(result["final_account"]["cash_minor"],cash)

    def test_known_only_keeps_null_limits_and_other_hard_blocks(self):
        for policy,reason in (("require_both","MISSING_EXECUTION_FACT"),("known_only","CASH_OR_VOLUME_CAP")):
            w=request(hold=True,limits=policy)
            for row in w["market_replay"]["rows"]:row.update(limit_up=None,limit_down=None)
            result=run(w).to_dict()
            self.assertEqual(result["orders"][0]["reason"],reason)
            self.assertTrue(all(row["limit_up"] is None for row in result["plan"]["market_replay"]["rows"]))
            self.assertEqual(bool(result["fills"]),policy=="known_only")
        for state,expected in (("suspended","NOT_TRADING"),("source_gap","NOT_TRADING")):
            w=request(hold=True,limits="known_only")
            for row in w["market_replay"]["rows"]:
                if row["session"]==w["start_session"]:row.update(market_state=state,limit_up=None,limit_down=None)
            self.assertEqual(run(w).to_dict()["orders"][0]["reason"],expected)
        w=request(hold=True,limits="known_only")
        for row in w["market_replay"]["rows"]:
            row.update(limit_down=None,limit_up="0.500")
        self.assertEqual(run(w).to_dict()["orders"][0]["reason"],"PRICE_LIMIT")

    def test_unit_policy_grid_and_entry_scope_reject_before_ledger(self):
        changes=[lambda w:w.update(price_unit="CNY/share"),
                 lambda w:w["profile"]["price_grid"]["rules"][0].update(tick_size="0.01"),
                 lambda w:w["portfolio_policy"].update(entry_session=DAYS[2]),
                 lambda w:w["initial_account"].update(positions={"A":{}})]
        for change in changes:
            w=request(hold=True);change(w)
            with patch("axiom_engine.runtime.backtest.AccountLedger",side_effect=AssertionError("ledger started")):
                with self.assertRaises(ContractError):run(w)

    def test_missing_strict_previous_close_and_late_reference(self):
        w=request(hold=True)
        for row in w["market_replay"]["rows"]:
            if row["session"]==DAYS[0]:row["close"]=None
        r=run(w).to_dict()
        self.assertEqual(r["decisions"][0]["status"],"NO_DECISION")
        self.assertEqual(r["decisions"][0]["trace"][0]["reason"],"ENTRY_REFERENCE_UNAVAILABLE")
        self.assertEqual(r["fills"],[])
        w=request(hold=True)
        for row in w["market_replay"]["rows"]:
            if row["session"]==DAYS[0]:row["close_available_at"]=DAYS[0]+"T12:31:00Z"
        with self.assertRaisesRegex(ContractError,"reference unavailable"):run(w)

    def test_positive_hold_split_and_cash_dividend_are_not_reinvested(self):
        w=split_fixture(quantity=0)
        old=w["market_replay"]["universe"][0];sid="cn.etf.SSE.513100.20130515"
        w["market_replay"]["universe"][0]=sid
        for row in w["market_replay"]["rows"]:
            if row["security_id"]==old:row["security_id"]=sid
        event=deepcopy(w["market_replay"]["unit_splits"][0]["event"]);event["security_id"]=sid
        bind_event(w,event)
        w.update(contract_version="backtest_request_v5",signal_frame=None,price_unit="CNY/fund unit",
            initial_account={"cash_minor":50000000,"positions":{}},
            portfolio_policy=etf_buy_and_hold_policy(security_id=sid,entry_session=w["start_session"]))
        w["profile"]=daily_open_profile(unknown_status_policy="etf_daily_observed",price_grid_policy="etf_price_grid_v1",slippage_bps="5")
        w["market_replay"]["cash_dividends"]=[dict(event_id="synthetic-cash",security_id=sid,record_session=DAYS[2],
            ex_session=DAYS[3],pay_session=DAYS[4],cash_per_unit="0.1",source_refs=[REF])]
        r=run(w).to_dict();bought=r["fills"][0]["quantity"];application=r["unit_split_applications"][0]
        self.assertEqual(len(r["fills"]),1)
        self.assertGreater(application["before_quantity"],0)
        self.assertEqual(application["after_quantity"],5*bought)
        self.assertEqual(application["before_market_value_minor"],application["after_market_value_minor"])
        self.assertEqual(r["final_account"]["positions"][sid]["cost_minor"],application["cost_minor"])
        self.assertEqual(r["final_account"]["cash_minor"],50000000+r["fills"][0]["cash_delta_minor"]+bought*10)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"run.json";save_backtest_run(run_doc:=BacktestRun.from_dict(r),path)
            self.assertEqual(load_backtest_run(path).payload,run_doc.payload)

    def test_public_saved_evaluation_and_fill_display_accept_v5_without_replay(self):
        from test_fill_display import files
        for hold in (False,True):
            result=run(request(hold=hold));w=result.to_dict()
            bench=benchmark(result)
            base=evaluate_backtest(result,benchmark=bench,spec=long_history_evaluation_spec(),dividend_scope=scope(result))
            report=evaluate_saved_analysis(result,base,benchmarks={"CSI300":bench,"SSE_COMPOSITE":None,"NASDAQ100":None},
                spec=analysis_evaluation_spec(risk_free=RF,benchmark_projection_version="benchmark_comparison_v2"))
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);save_backtest_run(result,root/"run.json");save_backtest_evaluation(report,root/"eval.json")
                with patch("test_fill_display.SNAPSHOT","s_synthetic"):
                    display=build_fill_display(result,display=files(root/"data",result,unit="CNY/fund unit"))
                save_fill_display(display,root/"display.json")
                with patch("axiom_engine.runtime.backtest.run_backtest",side_effect=AssertionError("replay")), \
                     patch("axiom_engine.runtime.backtest.AccountLedger",side_effect=AssertionError("ledger")):
                    self.assertEqual(load_backtest_run(root/"run.json").payload,result.payload)
                    self.assertEqual(load_backtest_evaluation(root/"eval.json").payload,report.payload)
                    self.assertEqual(load_fill_display(root/"display.json").payload,display.payload)
                self.assertEqual(report.to_dict()["signal_ref"],w["signal_ref"])
                bad=deepcopy(w);bad["portfolio_policy_ref"]=REF;bad.pop("content_digest")
                bad["content_digest"]=Document.from_dict(bad).identity
                (root/"bad.json").write_text(Document.from_dict(bad).payload)
                with self.assertRaisesRegex(ContractError,"binding mismatch"):load_backtest_run(root/"bad.json")
