"""Synthetic strategy/clock/equity/capacity and compatibility scenarios."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import (ContractError, StockPredictionFrame, ConfiguredStockStrategy,
    initial_strategy_state, plan_stock_portfolio, stock_strategy_spec)
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import stock_portfolio_policy, BacktestRequest, StockInputSource
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.backtest import _simulate
from axiom_engine.runtime.stock_rules import portfolio_support_ref
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from axiom_engine.runtime.stock_strategy_saved import validate_saved_strategy_decision
from test_csi300 import rules_for, core_context, frame_for, fees_for, REF, DAYS
from test_stock_stream_inputs import source_fixture
from test_stock_stream import execute
from test_stock_market_owner import market_for, bind
from test_stock_owned_inputs import run_owned
from test_stock_equity import fixture as equity_fixture


def overlay():
    return dict(take_profit="0.20", stop_loss="0.10",
        cost_basis="fee_inclusive_ledger_cost_per_owned_share_v1", check="rebalance_sessions_only",
        cooldown_rebalance_sessions=1, reentry="after_flat_and_cooldown")


def inputs(*, positions=None, cash=1_000_000, price="10", pending=None, observations=None, invalid=()):
    universe = rules_for()["universe"]
    return dict(feature_session=DAYS[0], trade_session=DAYS[1], universe=universe,
        candidates=[dict(security_id=s, score=None if s in invalid else len(universe)-i,
                        member=True, valid=s not in invalid) for i,s in enumerate(universe)],
        positions=positions or {}, pending_share_quantities=pending or {}, cash_minor=cash,
        reference_prices={s:dict(price=price,session=DAYS[0],available_at=DAYS[0]+"T12:00:00Z",source_refs=[REF]) for s in universe},
        risk_observations=observations or {}, constraints=dict(max_positions=1,reserved_security_ids=[],
        quantity_rules={s:dict(buy_minimum=100,buy_increment=100,sell_minimum=100,sell_increment=100,
                              daily_proxy_maximum=1_000_000) for s in universe}))


def risk(*, price="10", cost=100_000, owned=100):
    return dict(price=price,session=DAYS[0],available_at=DAYS[0]+"T12:00:00Z",source_refs=[REF],
                cost_minor=cost,owned_quantity=owned,price_basis="original_session_units",committed_sequence=1)


def decide(spec, data, state=None):
    data=deepcopy(data); data["constraints"]["max_positions"]=spec["top_k"]
    return ConfiguredStockStrategy(spec).plan(inputs=data,state=state or initial_strategy_state()).to_dict()


def configured(manifest, spec, *, universe_id="csi300"):
    manifest=deepcopy(manifest)
    profile=Document(Path(manifest["profile_input"]["artifact"]["manifest_uri"]).read_text()).to_dict()
    manifest["portfolio_policy"]=stock_portfolio_policy(strategy_spec=spec,universe_id=universe_id,
        execution_universe=manifest["scope"]["execution_universe"],execution_rules=profile["stock_execution_rules"])
    manifest["scope"]["supported_universe_ref"]=portfolio_support_ref(manifest["portfolio_policy"],manifest["scope"]["execution_universe"])
    manifest["request_ref"]=logical_ref(manifest,"request_ref")
    return manifest


class StockStrategyTests(unittest.TestCase):
    def test_configuration_and_legacy_factory_remain_explicit(self):
        from axiom_engine.runtime.stock_inputs import ELIGIBILITY_ID
        from axiom_engine.core.stock_portfolio import BUDGET_BASIS
        universe=["cnstock.000101.SZ.20000101"]
        self.assertEqual(stock_portfolio_policy(top_k=1,execution_universe=universe),
            dict(eligibility_id=ELIGIBILITY_ID,top_k=1,rebalance="weekly_first_trading_session",budget_basis=BUDGET_BASIS))
        for k in (True,0,-1,"5"):
            with self.assertRaises(ContractError): stock_strategy_spec(top_k=k)
        with self.assertRaises(ContractError): stock_strategy_spec(selection="rank_buffer",top_k=5,rank_buffer={"keep_rank":4})
        with self.assertRaises(ContractError): stock_strategy_spec(top_k=1,exit_overlay={**overlay(),"check":"intraday"})

    def test_three_compositions_change_targets_without_signal_or_io(self):
        a,b,*_=rules_for()["universe"]
        data=inputs(positions={b:dict(quantity=100,sellable_quantity=100)},price="12",observations={b:risk(price="12")})
        before=deepcopy(data)
        with patch("builtins.open",side_effect=AssertionError("strategy I/O")):
            top=decide(stock_strategy_spec(top_k=1),data)
            buffered=decide(stock_strategy_spec(selection="rank_buffer",top_k=1,rank_buffer={"keep_rank":10}),data)
            stopped=decide(stock_strategy_spec(top_k=1,exit_overlay=overlay()),data)
        self.assertEqual(top["selected_security_ids"],[a])
        self.assertEqual(buffered["selected_security_ids"],[b])
        self.assertEqual(stopped["force_exit_security_ids"],[b])
        self.assertNotIn(b,stopped["target_weights"])
        self.assertEqual(data,before)

    def test_future_information_and_inconsistent_equity_units_are_rejected(self):
        s=rules_for()["universe"][0]
        data=inputs(positions={s:dict(quantity=100,sellable_quantity=100)},observations={s:risk()})
        for mutation in ("quote","risk_clock","unit","different_close"):
            bad=deepcopy(data)
            if mutation=="quote": bad["reference_prices"][s]["available_at"]=DAYS[1]+"T00:00:00Z"
            if mutation=="risk_clock": bad["risk_observations"][s]["session"]=DAYS[1]
            if mutation=="unit": bad["risk_observations"][s]["price_basis"]="guessed_adjustment"
            if mutation=="different_close": bad["risk_observations"][s]["price"]="12"
            with self.subTest(mutation=mutation),self.assertRaises(ContractError):
                decide(stock_strategy_spec(top_k=1,exit_overlay=overlay()),bad)

    def test_confirmed_bonus_quantity_and_missing_cost_do_not_invent_exits(self):
        s=rules_for()["universe"][0]
        spec=stock_strategy_spec(top_k=1,exit_overlay=overlay())
        data=inputs(positions={s:dict(quantity=100,sellable_quantity=100)},pending={s:100},price="5",
                    observations={s:risk(price="5",owned=200)})
        result=decide(spec,data)
        self.assertEqual(result["force_exit_security_ids"],[])
        self.assertEqual(result["preserve_quantities"],{s:100})
        before_ex=deepcopy(data)
        before_ex["reference_prices"][s]["price"]="10"
        before_ex["risk_observations"][s]=risk(price="10",owned=100)
        self.assertEqual(decide(spec,before_ex)["force_exit_security_ids"],[])
        data["positions"][s]=dict(quantity=200,sellable_quantity=200);data["pending_share_quantities"]={}
        self.assertEqual(decide(spec,data)["force_exit_security_ids"],[])
        for cost,price in ((0,"5"),(100_000,None)):
            bad=deepcopy(data);bad["risk_observations"][s].update(cost_minor=cost,price=price)
            result=decide(spec,bad)
            self.assertEqual(result["force_exit_security_ids"],[])
            self.assertTrue(any(t["reason"]=="UNUSABLE_WEEKLY_CLOSE_RISK_BASIS" for t in result["trace"]))

    def test_exit_survives_insufficient_candidates_and_cooldown_waits_until_flat(self):
        s,*others=rules_for()["universe"]
        spec=stock_strategy_spec(top_k=1,exit_overlay=overlay())
        data=inputs(positions={s:dict(quantity=100,sellable_quantity=0)},price="8",observations={s:risk(price="8")},invalid=others)
        first=decide(spec,data)
        self.assertEqual(first["force_exit_security_ids"],[s])
        self.assertEqual(first["status"],"DECISION_COMPLETE")
        insufficient=decide(stock_strategy_spec(top_k=2,exit_overlay=overlay()),data)
        self.assertEqual(insufficient["force_exit_security_ids"],[s])
        self.assertTrue(any(t["reason"]=="INSUFFICIENT_ELIGIBLE_MEMBERS" for t in insufficient["trace"]))
        waiting=decide(spec,data,first["next_state"])
        self.assertEqual(waiting["force_exit_security_ids"],[s])
        flat=inputs(invalid=others)
        one=decide(spec,flat,waiting["next_state"])
        two=decide(spec,flat,one["next_state"])
        three=decide(spec,flat,two["next_state"])
        self.assertNotIn(s,one["target_weights"]);self.assertNotIn(s,two["target_weights"])
        self.assertEqual(three["target_weights"],{s:"1"})
        plain=decide(stock_strategy_spec(top_k=2),flat)
        buffered=decide(stock_strategy_spec(selection="rank_buffer",top_k=2,rank_buffer={"keep_rank":10}),flat)
        self.assertEqual(plain["status"],"NO_DECISION")
        self.assertEqual(buffered["target_weights"],{s:"0.5"})

    def test_frozen_budget_and_failed_sells_cannot_expand_position_count(self):
        rules=rules_for();a,b,c,*_=rules["universe"]
        data=inputs(positions={c:dict(quantity=800,sellable_quantity=0)},cash=200_000)
        result=decide(stock_strategy_spec(top_k=2),data)
        self.assertEqual(result["preserve_quantities"],{c:800})
        self.assertEqual(Decimal(result["target_weights"][a]),Decimal("0.2"))
        context=core_context(rules,2)
        context.update(portfolio_policy=stock_portfolio_policy(strategy_spec=stock_strategy_spec(top_k=2),
            universe_id="csi300",execution_universe=rules["universe"],execution_rules=rules),
            strategy_state=initial_strategy_state(),pending_share_quantities={},reserved_security_ids=[],risk_observations={})
        decision=plan_stock_portfolio(StockPredictionFrame.from_dict(frame_for(rules)),
            account=dict(cash_minor=200_000,positions=data["positions"],version=0),context=context,top_k=2).to_dict()
        self.assertEqual(decision["targets"][c],800);self.assertEqual(decision["targets"][a],200)
        from axiom_engine.runtime import stock_daily_open_profile_v2
        profile=stock_daily_open_profile_v2(execution_rules=rules,fee_schedule=fees_for())
        ledger=AccountLedger(cash_minor=1_000_000,calendar=DAYS,settlement_sessions=1,
            positions={c:dict(quantity=100,sellable_quantity=100,cost_minor=100_000)})
        row=dict(market_state="suspended",state_reason="synthetic",execution_evidence_cutoff=DAYS[1]+"T20:30:00+08:00",
            field_available_at={},source_refs=[REF],open="10",close="10",volume_shares=10000,
            limit_up="11",limit_down="9",_stock_listed=True,_stock_factor_valid=True)
        sell=dict(security_id=c,side="SELL",quantity=100,intent_id="sell",valid_until=DAYS[1],expected_account_version=0)
        _simulate(sell,row,ledger,profile,DAYS[1],"synthetic",0,stock_market={"action_blocks":[]},position_capacity=1)
        row["market_state"]="normal_trading"
        buy={**sell,"security_id":a,"side":"BUY","intent_id":"buy"}
        rejected=_simulate(buy,row,ledger,profile,DAYS[1],"synthetic",1,stock_market={"action_blocks":[]},position_capacity=1)
        self.assertEqual(rejected["reason"],"MAX_OWNED_POSITIONS")
        ledger.positions.clear()
        rejected=_simulate(buy,row,ledger,profile,DAYS[1],"synthetic",2,stock_market={"action_blocks":[]},
            position_capacity=1,pending_share_quantities={c:100})
        self.assertEqual(rejected["reason"],"MAX_OWNED_POSITIONS");self.assertEqual(ledger.fills,[])
        from test_stock_equity import seeded, action
        from axiom_engine.runtime.stock_strategy_rights import registered_share_reservations
        registered=seeded()
        reservations=registered_share_reservations(registered,[action("s")])
        self.assertEqual(reservations,["s"])
        rejected=_simulate(buy,row,ledger,profile,DAYS[1],"synthetic",3,stock_market={"action_blocks":[]},
            position_capacity=1,reserved_security_ids=reservations)
        self.assertEqual(rejected["reason"],"MAX_OWNED_POSITIONS")

    def test_core_preserve_only_rights_initialize_zero_targets_without_buying(self):
        rules=rules_for();security=rules["universe"][-1]
        spec=stock_strategy_spec(top_k=1)
        for kind in ("pending_only","registered_right_only"):
            with self.subTest(kind=kind):
                context=core_context(rules,1)
                context.update(portfolio_policy=stock_portfolio_policy(strategy_spec=spec,universe_id="csi300",
                    execution_universe=rules["universe"],execution_rules=rules),reference_prices={},
                    strategy_state=initial_strategy_state(),risk_observations={},
                    pending_share_quantities={security:100} if kind=="pending_only" else {},
                    reserved_security_ids=[security] if kind=="registered_right_only" else [])
                result=plan_stock_portfolio(StockPredictionFrame.from_dict(frame_for(rules)),
                    account=dict(cash_minor=1_000_000,positions={},version=0),context=context,top_k=1).to_dict()
                self.assertEqual(result["status"],"DECISION_COMPLETE")
                self.assertEqual(result["selected_security_ids"],[security])
                self.assertEqual(result["strategy_targets"]["preserve_quantities"],{security:0})
                self.assertEqual(result["strategy_targets"]["target_weights"],{})
                self.assertEqual(result["targets"],{security:0})
                self.assertEqual(result["intents"],[])

    def test_top_k_three_exact_lot_boundary_matches_legacy_targets(self):
        rules=rules_for();context=core_context(rules,3)
        frame=StockPredictionFrame.from_dict(frame_for(rules))
        account=dict(cash_minor=300_000,positions={},version=0)
        legacy=plan_stock_portfolio(frame,account=account,context=context,top_k=3).to_dict()
        context.update(portfolio_policy=stock_portfolio_policy(strategy_spec=stock_strategy_spec(top_k=3),
            universe_id="csi300",execution_universe=rules["universe"],execution_rules=rules),
            strategy_state=initial_strategy_state(),pending_share_quantities={},reserved_security_ids=[],risk_observations={})
        result=plan_stock_portfolio(frame,account=account,context=context,top_k=3).to_dict()
        self.assertEqual(result["targets"],legacy["targets"])
        self.assertEqual(list(result["targets"].values()),[100,100,100])
        self.assertEqual([i["quantity"] for i in result["intents"]],[100,100,100])
        self.assertTrue(all(r==dict(numerator=1,denominator=3) for r in result["strategy_targets"]["allocation_ratios"].values()))

    def test_same_saved_signal_three_configs_reuse_market_and_load_without_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_=source_fixture(root)
            signal_ref=manifest["prediction_input"]["prediction_ref"]
            specs=[stock_strategy_spec(top_k=2),stock_strategy_spec(selection="rank_buffer",top_k=2,rank_buffer={"keep_rank":10}),
                   stock_strategy_spec(top_k=2,exit_overlay=overlay())]
            with market_for(manifest) as market:
                run_ids=[]
                for i,spec in enumerate(specs):
                    request=configured(manifest,spec)
                    wire,expected,_=execute(root/f"cold-{i}",request)
                    with bind(market,request) as bound:
                        other,loaded=run_owned(root/f"owned-{i}",request,bound)
                    self.assertEqual(other["run_id"],wire["run_id"])
                    self.assertEqual(loaded.rows,expected.rows)
                    self.assertEqual(wire["core_version"],"axiom.stock_portfolio/4")
                    self.assertEqual(wire["signal_ref"],signal_ref)
                    run_ids.append(wire["run_id"])
                self.assertEqual(len(set(run_ids)),3)
                self.assertEqual(market.statistics["source_admissions"],1)
            from axiom_engine.runtime import load_stock_backtest_projection
            from test_stock_stream_inputs import LIMITS
            with patch("axiom_engine.runtime.backtest._run",side_effect=AssertionError("account replay")),\
                 patch.object(ConfiguredStockStrategy,"plan",side_effect=AssertionError("strategy replay")):
                load_stock_backtest_projection(root/"cold-0"/"run.json",artifact_reader=lambda ref:Path(ref["manifest_uri"]),limits=LIMITS)

    def test_saved_equity_strategy_state_and_original_membership_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,_=equity_fixture(root)
            request=configured(manifest,stock_strategy_spec(top_k=2,exit_overlay={**overlay(),"take_profit":"0.00001","stop_loss":"0.00001"}))
            wire,projection,_=execute(root/"out",request)
            self.assertTrue(any(t["reason"] in ("WEEKLY_CLOSE_TAKE_PROFIT","WEEKLY_CLOSE_STOP_LOSS")
                for d in projection.rows["decisions"] for t in d["trace"]))
            decision=deepcopy(projection.rows["decisions"][0])
            decision["strategy_targets"]["next_state"]["check_index"]+=1
            decision["strategy_targets_ref"]=Document.from_dict(decision["strategy_targets"]).identity
            decision["strategy_state_after_ref"]=Document.from_dict(decision["strategy_targets"]["next_state"]).identity
            with self.assertRaisesRegex(ContractError,"check index"):
                validate_saved_strategy_decision(decision,request["portfolio_policy"],request["scope"]["execution_universe"],
                    initial_strategy_state(),projection.rows["positions"])
            bad=configured(manifest,stock_strategy_spec(top_k=2),universe_id="csi800")
            with self.assertRaisesRegex(ContractError,"membership"):
                execute(root/"wrong-pool",bad)
            self.assertEqual(wire["status"],"COMPLETE")


if __name__ == "__main__": unittest.main()
