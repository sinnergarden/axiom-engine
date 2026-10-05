"""TopK uses the same planner/ledger; the default Core retains its saved ABI."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, StockPredictionFrame, plan_stock_portfolio
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, run_backtest, stock_portfolio_policy,
    save_backtest_run, load_backtest_run, evaluate_backtest, long_history_evaluation_spec)
from test_stocks import predictions, context, request, SECURITIES, benchmark


class TopKTests(unittest.TestCase):
    def plan(self, k=None, frame=None):
        return plan_stock_portfolio(StockPredictionFrame.from_dict(frame or predictions()),
            account={"cash_minor":1000000,"positions":{},"version":0}, context=context(), top_k=k)

    def test_legacy_default_bytes_and_explicit_five_are_separate(self):
        old = self.plan()
        self.assertEqual(old.identity, "sha256:60ca03a58e7f81ca930527348fa2de46b1082b30bbf77e7701f85730e321ea3c")
        explicit = self.plan(5).to_dict()
        self.assertEqual(explicit["contract_version"], "axiom.stock_portfolio/2")
        self.assertEqual(explicit["targets"], old.to_dict()["targets"])
        self.assertNotEqual(explicit["intents"][0]["intent_id"], old.to_dict()["intents"][0]["intent_id"])

    def test_generic_boundaries_equal_wealth_and_ties(self):
        for k, size in ((1,1000),(3,300),(5,200),(6,100)):
            result = self.plan(k, predictions([-1]*6)).to_dict()
            self.assertEqual(result["selected_security_ids"], SECURITIES[:k])
            self.assertEqual(list(result["targets"].values()), [size]*k)
            self.assertEqual(result["trace"][-1]["top_k"], k)
        for k in (True,False,0,-1,7,3.0,"3"):
            with self.subTest(k=k), self.assertRaises(ContractError):
                self.plan(k)
            with self.assertRaises(ContractError):
                stock_portfolio_policy(top_k=k, execution_universe=SECURITIES)

    def test_shortage_and_invalid_do_not_shrink_k(self):
        frame = predictions()
        for row in frame["rows"]:
            if row["security_id"] in SECURITIES[2:]: row["member"] = False
        result = self.plan(3,frame).to_dict()
        self.assertEqual(result["status"], "NO_DECISION")
        self.assertEqual(result["intents"], [])
        self.assertEqual(result["trace"], [{"reason":"INSUFFICIENT_ELIGIBLE_MEMBERS","count":2,"top_k":3}])
        frame = predictions(); frame["rows"][0].update(valid=False,score=None,invalid_reason="MISSING")
        self.assertEqual(self.plan(3,frame).to_dict()["trace"][0]["reason"], "INVALID_SIGNAL")

    def test_runtime_two_k_runs_share_signal_and_keep_old_tuple_loader(self):
        plan = request().to_dict(); original = deepcopy(plan)
        results = []
        for k in (3,5):
            variant = deepcopy(plan)
            variant["portfolio_policy"] = stock_portfolio_policy(top_k=k,execution_universe=variant["execution_universe"])
            results.append(run_backtest(BacktestRequest.from_dict(variant)))
        three,five = [r.to_dict() for r in results]
        self.assertNotEqual(three["run_id"],five["run_id"])
        self.assertEqual(three["signal_ref"],five["signal_ref"])
        self.assertEqual(three["plan"]["initial_account"],five["plan"]["initial_account"])
        self.assertEqual(three["core_version"],"axiom.stock_portfolio/2")
        self.assertEqual(len(three["decisions"][0]["selected_security_ids"]),3)
        self.assertEqual(len(five["decisions"][0]["selected_security_ids"]),5)
        self.assertNotEqual(three["fills"],five["fills"])
        self.assertEqual(plan,original)
        with tempfile.TemporaryDirectory() as tmp:
            for k,r in zip((3,5),results):
                path=Path(tmp)/f"{k}.json"; save_backtest_run(r,path)
                with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("replay")):
                    self.assertEqual(load_backtest_run(path).payload,r.payload)
            # A saved /1 tuple can only represent its original Top5 policy.
            wire=five.copy(); wire["core_version"]="axiom.stock_portfolio/1"
            wire["run_id"]=Document.from_dict({"request":wire["plan"],"core":wire["core_version"],
                "runtime":wire["runtime_version"],"implementation_ref":wire["implementation_ref"]}).identity
            wire.pop("content_digest");wire["content_digest"]=Document.from_dict(wire).identity
            path=Path(tmp)/"legacy.json";path.write_text(Document.from_dict(wire).payload)
            with self.assertRaises(ContractError): load_backtest_run(path)
            legacy = load_backtest_run(Path(__file__).parent/"fixtures/top5_run_v1.json")
            self.assertEqual(legacy.to_dict()["core_version"],"axiom.stock_portfolio/1")
            evaluate_backtest(legacy,benchmark=benchmark(),spec=long_history_evaluation_spec())
        self.assertEqual(evaluate_backtest(results[0],benchmark=benchmark(),spec=long_history_evaluation_spec()).to_dict()["input_run_ref"]["run_id"],three["run_id"])
