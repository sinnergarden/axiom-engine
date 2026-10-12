"""One bounded market-gate sequence and its saved native/journal closure."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, ConfiguredStockStrategy, initial_strategy_state, stock_strategy_spec
from axiom_engine.core.contracts import Document, canonical
from axiom_engine.runtime.stock_stream_contracts import logical_ref, validate_manifest
from axiom_engine.runtime.stock_strategy_saved import (BoundMarketRiskInput,
    registered_market_risk_rights, validate_saved_strategy_capacity)
from test_stock_strategy import inputs, configured
from test_stock_stream_inputs import LIMITS
from test_stock_stream import execute
from test_stock_equity import fixture as equity_fixture, seeded, action

GATE = dict(rule="close_above_sma",security_id="000300.SH",window_sessions=120,
            check="rebalance_sessions_only")


def dependency(calendar):
    before=[];day=date.fromisoformat(calendar[0])
    while len(before)<119:
        day-=timedelta(days=1)
        if day.weekday()<5: before.append(day.isoformat())
    days=list(reversed(before))+calendar[:-1]
    metadata=[dict(security_id="000300.SH",session=d,usable_from=d+"T20:00:00+08:00",
        revision_id="synthetic-revision",raw_batch_id="synthetic-batch",
        first_observed_at=d+"T20:00:00+08:00",availability_basis="synthetic",evidence_ref=None,
        missing_reason=None) for d in days]
    view=dict(context=dict(contract_version="data_batch_v1",domain="benchmark_daily",
        query=dict(fields=["close"],symbols=["000300.SH"],purpose="decision_facts",price_basis="unadjusted",
            pit_policy="best_effort_vendor_v1",sessions=days,
            cutoff_by_session={d:d+"T20:30:00+08:00" for d in days})),
        records=[dict(security_id="000300.SH",session=d,close=10.0) for d in days],
        field_meta=dict(close=dict(dtype="float64",unit="index points",by_key=metadata)))
    return dict(contract_version="stock_market_risk_input_v1",native_ref=Document.from_dict(view).identity,view=view)


def bind_risk(request, risk):
    request=deepcopy(request)
    request.update(risk_input=risk,risk_input_ref=Document.from_dict(risk).identity)
    if "request_ref" in request: request["request_ref"]=logical_ref(request,"request_ref")
    return request


def bare_request(calendar):
    return dict(scope=dict(calendar=calendar,anchor_session=calendar[0]),
                portfolio_policy=dict(strategy_spec=stock_strategy_spec(top_k=1,market_gate=GATE)))


def gated_inputs(*, bound, risk_on, state, positions=None, pending=None, registered=None):
    data=inputs(positions=positions,pending=pending)
    data["risk_input_ref"]=bound.reference
    observation=bound.observation(data["feature_session"],bound.reference)
    if risk_on:
        observation["points"][-1]["close"]=20.0
        observation["observation_ref"]=Document.from_dict({k:v for k,v in observation.items() if k!="observation_ref"}).identity
    data["market_observation"]=observation
    data["constraints"]["registered_stock_right_security_ids"]=registered or []
    return data


class StockMarketGateTests(unittest.TestCase):
    def test_legacy_wire_and_one_liquidation_latch_sequence(self):
        legacy=dict(contract_version="stock_strategy_spec_v1",selection="top_k",top_k=1,rank_buffer=None,exit_overlay=None)
        self.assertEqual(canonical(stock_strategy_spec(top_k=1)),canonical(legacy))
        self.assertEqual(initial_strategy_state(legacy),initial_strategy_state())
        data=inputs();calendar=[data["feature_session"],data["trade_session"]]
        request=bind_risk(bare_request(calendar),dependency(calendar));bound=BoundMarketRiskInput(request)
        spec=request["portfolio_policy"]["strategy_spec"];strategy=ConfiguredStockStrategy(spec)
        state=initial_strategy_state(spec);security=data["universe"][0]
        def step(on, **ownership):
            nonlocal state
            result=strategy.plan(inputs=gated_inputs(bound=bound,risk_on=on,state=state,**ownership),state=state).to_dict()
            state=result["next_state"]
            return result
        self.assertEqual(step(True)["target_weights"],{security:"1"})
        # Equality is off. A failed/unsellable exit remains owned even when the market recovers.
        off=step(False,positions={security:dict(quantity=100,sellable_quantity=0)})
        self.assertEqual(off["force_exit_security_ids"],[security]);self.assertEqual(state["market_mode"],"exit_pending")
        failed=step(True,positions={security:dict(quantity=100,sellable_quantity=100)})
        self.assertEqual(failed["target_weights"],{});self.assertEqual(state["market_mode"],"exit_pending")
        for ownership in (dict(pending={security:10}),dict(registered=[security])):
            remaining=step(True,**ownership)
            self.assertEqual(remaining["force_exit_security_ids"],[security]);self.assertEqual(remaining["selected_security_ids"],[])
        self.assertEqual(step(True)["target_weights"],{security:"1"});self.assertEqual(state["market_mode"],"risk_on")
        self.assertEqual(step(False)["target_weights"],{});self.assertEqual(state["market_mode"],"risk_off")
        self.assertEqual(step(True)["target_weights"],{security:"1"})

    def test_window_availability_and_risk_rebinding(self):
        data=inputs();calendar=[data["feature_session"],data["trade_session"]]
        request=bind_risk(bare_request(calendar),dependency(calendar));bound=BoundMarketRiskInput(request)
        observation=bound.observation(calendar[0],bound.reference)
        self.assertEqual(len(observation["points"]),120)
        short=gated_inputs(bound=bound,risk_on=False,state=initial_strategy_state(request["portfolio_policy"]["strategy_spec"]))
        short["market_observation"]["points"].pop(0)
        short["market_observation"]["observation_ref"]=Document.from_dict({k:v for k,v in short["market_observation"].items() if k!="observation_ref"}).identity
        with self.assertRaisesRegex(ContractError,"complete market window"):
            ConfiguredStockStrategy(request["portfolio_policy"]["strategy_spec"]).plan(inputs=short,
                state=initial_strategy_state(request["portfolio_policy"]["strategy_spec"]))
        for damage in ("missing","late"):
            risk=deepcopy(request["risk_input"])
            if damage=="missing": risk["view"]["records"][0]["close"]=None
            else: risk["view"]["field_meta"]["close"]["by_key"][0]["usable_from"]=calendar[1]+"T20:00:00+08:00"
            risk["native_ref"]=Document.from_dict(risk["view"]).identity
            with self.assertRaisesRegex(ContractError,"missing or unavailable"):
                BoundMarketRiskInput(bind_risk(request,risk))
        changed=deepcopy(request["risk_input"]);changed["view"]["records"][-1]["close"]=11.0
        changed["native_ref"]=Document.from_dict(changed["view"]).identity
        other=bind_risk(request,changed)
        with self.assertRaisesRegex(ContractError,"differs from admitted"):
            bound.observation(calendar[0],other["risk_input_ref"])
        self.assertNotEqual(BoundMarketRiskInput(other).observation(calendar[0],other["risk_input_ref"]),observation)

    def test_formal_stream_and_saved_native_journal_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,_=equity_fixture(root)
            plain=configured(manifest,stock_strategy_spec(top_k=2))
            self.assertNotIn("risk_input",plain);self.assertEqual(validate_manifest(plain),plain)
            request=configured(manifest,stock_strategy_spec(top_k=2,market_gate=GATE))
            risk=dependency(request["scope"]["calendar"])
            risk["view"]["records"][119]["close"]=20.0
            risk["native_ref"]=Document.from_dict(risk["view"]).identity
            request=bind_risk(request,risk)
            self.assertEqual(request["market_input"],plain["market_input"])
            self.assertEqual(request["prediction_input"],plain["prediction_input"])
            limits={**LIMITS,"max_result_part_bytes":500_000,"max_result_buffer_bytes":500_000}
            wire,projection,_=execute(root/"out",request,limits=limits)
            decisions=projection.rows["decisions"]
            self.assertEqual(wire["status"],"COMPLETE")
            self.assertEqual([d["strategy_targets"]["next_state"]["market_mode"] for d in decisions],["risk_on","exit_pending"])
            # Saved validation never reruns the strategy and uses the decision watermark.
            with patch.object(ConfiguredStockStrategy,"plan",side_effect=AssertionError("planner replay")):
                validate_saved_strategy_capacity(wire,projection.rows)
            bad=deepcopy(projection.rows)
            second=bad["decisions"][1]
            for trace in (second["trace"],second["strategy_targets"]["trace"]):
                next(t for t in trace if t["reason"]=="MARKET_GATE")["ownership"]["positions"]={}
            with self.assertRaisesRegex(ContractError,"decision account version"):
                validate_saved_strategy_capacity(wire,bad)
            altered=deepcopy(wire);altered["request_manifest"]=bind_risk(request,dependency(request["scope"]["calendar"]))
            with self.assertRaisesRegex(ContractError,"original native dependency"):
                validate_saved_strategy_capacity(altered,projection.rows)

    def test_pending_based_registered_right_is_not_flat(self):
        ledger=seeded();event=action("s")
        state=ledger.equity_entitlements[event["economic_event_id"]]
        state.update(record_quantity=0,record_pending_quantity=10)
        self.assertEqual(registered_market_risk_rights(ledger,[event]),["s"])
        state["ex_applied"]=True
        self.assertEqual(registered_market_risk_rights(ledger,[event]),[])
        data=inputs();security=data["universe"][0];calendar=[data["feature_session"],data["trade_session"]]
        risk=dependency(calendar);risk["view"]["records"][-1]["close"]=20.0
        risk["native_ref"]=Document.from_dict(risk["view"]).identity
        request=bind_risk(bare_request(calendar),risk);request["portfolio_policy"]["top_k"]=1
        bound=BoundMarketRiskInput(request);spec=request["portfolio_policy"]["strategy_spec"]
        prior=initial_strategy_state(spec);prior.update(check_index=1,market_mode="exit_pending")
        targets=ConfiguredStockStrategy(spec).plan(inputs=gated_inputs(bound=bound,risk_on=True,state=prior,
            registered=[security]),state=prior).to_dict()
        self.assertEqual(targets["target_weights"],{});self.assertEqual(targets["force_exit_security_ids"],[security])
        decision=dict(feature_session=calendar[0],risk_input_ref=bound.reference,expected_account_version=1,
            strategy_targets=targets)
        row=dict(sequence=1,security_id=security,source_event_id=event["economic_event_id"],reason="EQUITY_REGISTER",
            quantity_delta=0,sellable_delta=0,pending_share_delta=0,record_quantity=0,record_pending_quantity=10)
        wire=dict(request_manifest=request,committed_sequence=1,
            account_events=dict(equity_facts=dict(actions=[action(security)])))
        rows=dict(position_ledger=[row],decisions=[decision])
        validate_saved_strategy_capacity(wire,rows)
        # Flat positions/pending do not erase a journaled, not-yet-EX registration.
        next(t for t in targets["trace"] if t["reason"]=="MARKET_GATE")["ownership"]["registered_stock_right_security_ids"]=[]
        with self.assertRaisesRegex(ContractError,"decision account version"):
            validate_saved_strategy_capacity(wire,rows)


if __name__=="__main__": unittest.main()
