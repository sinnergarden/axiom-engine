"""Synthetic v6 ledger and immutable native/PIT admission boundaries."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_rules import listed, support_ref
from axiom_engine.runtime import (BacktestRequest, BacktestRun, run_backtest, stock_prediction_schedule,
    stock_market_from_batches, stock_daily_open_profile_v2, csi300_stock_portfolio_policy,
    save_backtest_run, load_backtest_run, stock_dividend_scope)
from axiom_engine.runtime import (evaluate_backtest, long_history_evaluation_spec, analysis_evaluation_spec,
    evaluate_saved_analysis, build_fill_display, save_backtest_evaluation, load_backtest_evaluation,
    save_fill_display, load_fill_display)
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.backtest import _simulate, _fee_components
from axiom_engine.runtime.evaluation import _verify_run
from axiom_engine.runtime.stock_evidence import native_ref
from test_csi300 import rules_for, fees_for, DAYS, BOARD_IDS, REF
from test_stock_schedule import fold, seal
import test_stocks as legacy


def full_request(*, rules=None, k=3, cash=50000000, mutate_native=None, mutate_frame=None):
    rules = rules or rules_for()
    calendar, universe = rules["calendar"], rules["universe"]
    identities = {r["security_id"]: r for r in rules["identity_input"]["rows"]}
    # Reuse only the legacy synthetic proof shape, then replace every source/scope/count.
    proof = legacy.request().to_dict()["admission_evidence"]
    with patch.object(legacy, "DAYS", calendar), patch.object(legacy, "SECURITIES", universe):
        batches = legacy.native_inputs(state="normal_trading")
        membership = legacy.batch("universe_membership", ["is_member"],
            [dict(session=d, security_id=s, is_member=listed(identities[s], d)) for d in calendar for s in universe], {})
        with patch.object(legacy, "SECURITIES", universe[:6]):
            saved_fold = fold(calendar, calendar[1:])
    if len(universe) > 6:
        frame = saved_fold["prediction_frame"]
        templates = {r["session"]:r for r in frame["rows"]}
        frame.update(universe=universe,rows=[{**templates[d],"security_id":s,"score":-i}
            for d in calendar[:-1] for i,s in enumerate(universe)])
    membership["context"]["query"].update(purpose="decision_facts", universe_id="csi300")
    for batch in batches[:4]:
        for row in batch["records"]:
            if listed(identities[row["security_id"]], row["session"]): continue
            for name in set(row) - {"session", "security_id"}:
                row[name] = "not_listed" if name == "market_state" else None
            for field in batch["field_meta"].values():
                entry = next(m for m in field["by_key"] if m["session"] == row["session"] and m["security_id"] == row["security_id"])
                entry.update(missing_reason="not_listed", usable_from=None)
    for row in saved_fold["prediction_frame"]["rows"]:
        if not listed(identities[row["security_id"]], row["session"]):
            row.update(member=False, valid=False, score=None, invalid_reason="NOT_MEMBER")
    if mutate_native: mutate_native(batches, membership)
    if mutate_frame: mutate_frame(saved_fold["prediction_frame"])
    saved_fold["prediction_frame"] = seal(saved_fold["prediction_frame"], "signal_run_ref")
    schedule = stock_prediction_schedule(folds=[saved_fold], calendar=calendar).to_dict()
    market = stock_market_from_batches(batches=batches, universe=universe, calendar=calendar,
        execution_rules=rules, membership_batch=membership).to_dict()
    for name in ("admission_ref", "signal_run_ref", "feature_ref", "model_ref", "listing_identity_equal_all_83"):
        proof.pop(name, None)
    n, count = len(universe), len(universe)*len(calendar)
    proof.update(contract_version="stock_snapshot_pair_admission_v2", prediction_schedule_ref=schedule["schedule_ref"],
        prediction_refs=[{"fold_ref": saved_fold["fold_ref"], **{name: saved_fold["prediction_frame"][name] for name in
            ("fold_spec_ref", "signal_run_ref", "feature_ref", "model_ref")}}],
        listing_identity_checks=dict(checked_rows=n, paired_rows=n, mismatches=[]),
        scope=dict(execution_security_ids=universe, initial_sessions=calendar, membership_security_ids=universe,
                   warmup_sessions=["2023-12-28"]),
        prediction_membership=dict(compared=n*(len(calendar)-1), saved_prediction_rows=n*(len(calendar)-1),
                                   model_mismatch=[], execution_mismatch=[]),
        previous_close_basis_checks=dict(all_available_at_feature_knowledge_cutoff=False,
            all_available_before_decision=False, paired_rows_per_root=count, checked_rows_both_roots=2*count,
            equal_rows=count, listing_identity_checked_rows=count, listing_identity_mismatches=[]))
    for section, size in (("primary_comparison", count), ("warmup_comparison", n)):
        for comparison in proof[section].values():
            comparison.update(left_rows=size, right_rows=size)
            for field in comparison["fields"].values():
                field["counts"] = {name: size for name in field["counts"]}
    native = batches + [membership]
    for side in ("execution", "model"):
        for kind, index in (("states",0), ("market",1), ("factor",3), ("membership",6), ("warmup-market",1), ("warmup-factor",3)):
            batch = native[index]; query = deepcopy(batch["context"]["query"])
            if kind.startswith("warmup-"):
                query.update(sessions=["2023-12-28"], cutoff_by_session={"2023-12-28":"2023-12-28T20:30:00+08:00"})
            proof["batches"][side+"-"+kind] = dict(wire_ref=native_ref(batch), file_digest=REF,
                snapshot_id=proof[side]["snapshot"], reader_version=batch["context"]["reader_version"], query=query)
    proof = seal(proof, "admission_ref")
    reference = Document.from_dict(rules).identity
    return BacktestRequest.from_dict(dict(contract_version="backtest_request_v6", account_id=f"synthetic-full-top{k}",
        start_session=calendar[1], end_session=calendar[-1], prediction_schedule=schedule, market_replay=market,
        initial_account=dict(cash_minor=cash, positions={}),
        profile=stock_daily_open_profile_v2(execution_rules=rules, fee_schedule=fees_for(through=calendar[-1])),
        prediction_universe=universe, execution_universe=universe, supported_universe_ref=support_ref(universe),
        portfolio_policy=csi300_stock_portfolio_policy(top_k=k, execution_universe=universe, execution_rules=rules),
        stock_execution_rules_ref=reference, admission_ref=proof["admission_ref"], admission_evidence=proof,
        stock_action_policy="observed_implemented_only"))


def simulation(security, quantity, *, cash=1000000000, volume=1000000, held=0, sellable=0, side="BUY", day=DAYS[1]):
    rules = rules_for(); profile = stock_daily_open_profile_v2(execution_rules=rules, fee_schedule=fees_for())
    ledger = AccountLedger(cash_minor=cash, calendar=DAYS, settlement_sessions=1,
        positions={security:dict(quantity=sellable,sellable_quantity=sellable,cost_minor=sellable*1000)} if sellable else {})
    if held > sellable:
        quantity_unsettled = held - sellable
        ledger.apply_fill(dict(fill_id="seed-unsettled",order_id="seed-order",session=day,security_id=security,
            side="BUY",quantity=quantity_unsettled,price="10",gross_minor=quantity_unsettled*1000,
            fee_minor=0,cash_delta_minor=-quantity_unsettled*1000))
    row = dict(market_state="normal_trading", state_reason=None, execution_evidence_cutoff=day+"T20:30:00+08:00",
        field_available_at={}, source_refs=[REF], open="10", close="10", volume_shares=volume,
        limit_up="11", limit_down="9", _stock_listed=True, _stock_factor_valid=True,
        _stock_factor_missing_reason=None, _stock_factor_source_ref=REF)
    intent = dict(security_id=security, side=side, quantity=quantity, intent_id="synthetic", valid_until=day, expected_account_version=0)
    order = _simulate(intent,row,ledger,profile,day,"synthetic-run",0,stock_market={"action_blocks":[]})
    return order, ledger


class FullRuntimeTests(unittest.TestCase):
    def test_one_account_top_k_identity_and_exact_saved_v6_loader(self):
        request = full_request(k=3); original = request.payload
        three = run_backtest(request)
        four = run_backtest(full_request(k=4))
        a,b = three.to_dict(),four.to_dict()
        self.assertEqual(request.payload,original)
        self.assertEqual(a["signal_ref"],b["signal_ref"])
        self.assertNotEqual(a["run_id"],b["run_id"])
        self.assertNotEqual(a["fills"],b["fills"])
        self.assertEqual((a["contract_version"],a["core_version"],a["runtime_version"]),
                         ("backtest_run_v6","axiom.stock_portfolio/3","axiom.backtest/6"))
        self.assertEqual(a["nav"][-1]["nav_minor"],50000000-a["metrics"]["total_fees_minor"])
        self.assertEqual(stock_dividend_scope(three).to_dict()["coverage"],"observed_records_only")
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"run.json"; save_backtest_run(three,path)
            with patch("axiom_engine.runtime.backtest.AccountLedger",side_effect=AssertionError("no account replay")):
                self.assertEqual(load_backtest_run(path).payload,three.payload)
                self.assertEqual(_verify_run(three),a)

    def test_submission_cash_then_one_share_volume_partial_all_four_boards(self):
        star=BOARD_IDS["SSE_STAR"]
        # 199-share buying power does not fund the legally submitted minimum 200.
        order,ledger=simulation(star,200,cash=199500,volume=1990)
        self.assertEqual((order["submitted_quantity"],order["unsubmitted_quantity"],len(ledger.fills)),(0,200,0))
        order,ledger=simulation(star,200,cash=200502,volume=1990)
        self.assertEqual((order["requested_quantity"],order["submitted_quantity"],order["filled_quantity"],order["unfilled_quantity"]),(200,200,199,1))
        self.assertEqual(ledger.fills[0]["commission_minor"],500)
        self.assertEqual(ledger.fills[0]["transfer_fee_minor"],2)
        self.assertEqual(ledger.cash,1000)
        order,ledger=simulation(star,200,volume=0)
        self.assertEqual((order["submitted_quantity"],order["filled_quantity"],order["unfilled_quantity"],len(ledger.fills)),(200,0,200,0))
        for board,security in BOARD_IDS.items():
            minimum=200 if board=="SSE_STAR" else 100
            order,ledger=simulation(security,minimum,volume=(minimum-1)*10)
            self.assertEqual((order["submitted_quantity"],order["filled_quantity"]),(minimum,minimum-1))

    def test_maximum_sellable_tail_and_no_t_plus_one_bypass(self):
        maxima={"SSE_MAIN":1000000,"SZSE_MAIN":1000000,"SZSE_CHINEXT":150000,"SSE_STAR":50000}
        for board,security in BOARD_IDS.items():
            order,_=simulation(security,2000000,cash=3000000000,volume=30000000)
            self.assertEqual(order["submitted_quantity"],maxima[board])
        star=BOARD_IDS["SSE_STAR"]
        order,_=simulation(star,199,held=199,sellable=199,side="SELL")
        self.assertEqual(order["filled_quantity"],199)
        order,ledger=simulation(star,500,held=500,sellable=100,side="SELL")
        self.assertEqual((order["submitted_quantity"],len(ledger.fills)),(0,1))
        for security in BOARD_IDS.values():
            order,ledger=simulation(security,200,held=200,sellable=0,side="SELL")
            self.assertEqual((order["submitted_quantity"],len(ledger.fills)),(0,1))

    def test_actual_fee_intervals_and_zero_fill_commission(self):
        profile=stock_daily_open_profile_v2(execution_rules=rules_for(),fee_schedule=fees_for())
        expected={"2022-04-28":(1000,20),"2022-04-29":(1000,10),"2023-08-27":(1000,10),"2023-08-28":(500,10)}
        for day,(stamp,transfer) in expected.items():
            self.assertEqual(_fee_components(Decimal("10"),1000,"SELL",profile,day),(1000000,500,stamp,transfer))
            self.assertEqual(_fee_components(Decimal("10"),0,"SELL",profile,day),(0,0,0,0))

    def test_prelisting_nonmember_nulls_do_not_block_other_stocks(self):
        boards={BOARD_IDS["SSE_MAIN"]:"SSE_MAIN",BOARD_IDS["SZSE_MAIN"]:"SZSE_MAIN",
                "cnstock.688200.SH.20240103":"SSE_STAR"}
        rules=rules_for(boards)
        result=run_backtest(full_request(rules=rules,k=2)).to_dict()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(result["decisions"][0]["selected_security_ids"],rules["universe"][:2])
        absent=next(r for r in result["plan"]["market_replay"]["rows"] if r["session"]==DAYS[0] and r["security_id"].startswith("cnstock.688"))
        self.assertEqual((absent["market_state"],absent["close"]),("not_listed",None))
        self.assertEqual(result["lifecycle_admission"]["pre_listing_null"],2)

    def test_nonmember_factor_gap_is_local_but_held_gap_stops_account(self):
        security=BOARD_IDS["SZSE_MAIN"]
        def missing(batches,membership):
            for row in batches[3]["records"]:
                if row["security_id"]==security and row["session"]==DAYS[2]:row["factor"]=None
            for meta in batches[3]["field_meta"]["factor"]["by_key"]:
                if meta["security_id"]==security and meta["session"]==DAYS[2]:meta.update(missing_reason="source_gap",usable_from=None)
        held=run_backtest(full_request(k=4,mutate_native=missing)).to_dict()
        self.assertEqual((held["status"],held["stopped"]["session"],held["stopped"]["reason"]),
                         ("BLOCKED",DAYS[2],"HELD_MISSING_STOCK_LIFECYCLE_CAPABILITY"))
        def nonmember(batches,membership):
            missing(batches,membership)
            for row in membership["records"]:
                if row["security_id"]==security:row["is_member"]=False
        def frame(wire):
            for row in wire["rows"]:
                if row["security_id"]==security:row.update(member=False,valid=False,score=None,invalid_reason="NOT_MEMBER")
        local=run_backtest(full_request(k=3,mutate_native=nonmember,mutate_frame=frame)).to_dict()
        self.assertEqual(local["status"],"COMPLETE")
        self.assertEqual(local["lifecycle_admission"]["listed_nonmember_gap"],1)
        self.assertEqual(held["lifecycle_admission"]["held_gap"],1)

    def test_300_original_members_one_invalid_continues_in_the_same_runtime(self):
        securities=[f"cnstock.600{i:03}.SH.20000101" for i in range(300)]
        rules=rules_for({s:"SSE_MAIN" for s in securities})
        def invalid(frame):
            for row in frame["rows"]:
                if row["security_id"]==securities[0]:row.update(valid=False,score=None,invalid_reason="FEATURE_MISSING")
        request=full_request(rules=rules,k=5,mutate_frame=invalid)
        result=run_backtest(request).to_dict()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(result["decisions"][0]["selected_security_ids"],securities[1:6])
        self.assertEqual(result["decisions"][0]["trace"][0]["excluded_invalid_member_count"],1)
        self.assertTrue(result["fills"])

    def test_saved_v6_analysis_and_display_without_account_execution(self):
        from test_fill_display import files
        from test_analysis_evaluation import RF
        run=run_backtest(full_request())
        benchmark=legacy.benchmark()
        with patch("axiom_engine.runtime.backtest.AccountLedger",side_effect=AssertionError("account replay")):
            base=evaluate_backtest(run,benchmark=benchmark,spec=long_history_evaluation_spec(),dividend_scope=stock_dividend_scope(run))
            report=evaluate_saved_analysis(run,base,benchmarks={"CSI300":benchmark,"SSE_COMPOSITE":None,"NASDAQ100":None},
                                           spec=analysis_evaluation_spec(risk_free=RF))
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                display=build_fill_display(run,display=files(root/"data",run))
                self.assertEqual(display.to_dict()["status"],"COMPLETE")
                save_backtest_evaluation(report,root/"eval.json");save_fill_display(display,root/"display.json")
                self.assertEqual(load_backtest_evaluation(root/"eval.json").payload,report.payload)
                self.assertEqual(load_fill_display(root/"display.json").payload,display.payload)
        self.assertEqual(report.to_dict()["input_run_ref"]["run_id"],run.to_dict()["run_id"])

    def test_held_missing_close_saves_stale_reason_without_filling_a_price(self):
        security=BOARD_IDS["SZSE_MAIN"]
        def gap(batches,membership):
            for row in batches[1]["records"]:
                if row["security_id"]==security and row["session"]==DAYS[2]:row["close"]=None
            for meta in batches[1]["field_meta"]["close"]["by_key"]:
                if meta["security_id"]==security and meta["session"]==DAYS[2]:meta.update(missing_reason="source_gap",usable_from=None)
        result=run_backtest(full_request(k=4,mutate_native=gap)).to_dict()
        position=next(p for p in result["positions"] if p["session"]==DAYS[2] and p["security_id"]==security)
        self.assertEqual((position["is_stale"],position["mark_session"],position["stale_reason"]),(True,DAYS[1],"source_gap"))
        self.assertEqual(result["status"],"COMPLETE")

    def test_source_membership_mismatch_and_proof_damage_fail_before_ledger(self):
        for mutation in ("member_ref","member_flag","rule_ref","null_pair_count","missing_native_key"):
            plan=full_request().to_dict()
            if mutation=="member_ref":plan["market_replay"]["membership_ref"]=REF
            if mutation=="member_flag":
                plan["prediction_schedule"]["folds"][0]["prediction_frame"]["rows"][0]["member"]=False
                from test_stock_schedule import reseal_schedule
                reseal_schedule(plan)
            if mutation=="rule_ref":plan["stock_execution_rules_ref"]=REF
            if mutation=="null_pair_count":
                proof=plan["admission_evidence"];proof["primary_comparison"]["factor"]["fields"]["factor"]["counts"]["missing_reason_equal"]-=1
                plan["admission_evidence"]=seal(proof,"admission_ref");plan["admission_ref"]=plan["admission_evidence"]["admission_ref"]
            if mutation=="missing_native_key":plan["market_replay"]["source_evidence"][6]["batch"]["records"].pop()
            with self.subTest(mutation=mutation),patch("axiom_engine.runtime.backtest.AccountLedger",side_effect=AssertionError("ledger started")):
                with self.assertRaises(ContractError):run_backtest(BacktestRequest.from_dict(plan))

    def test_loader_rejects_resealed_submission_fee_and_rule_forgery(self):
        original=run_backtest(full_request()).to_dict()
        for mutation in ("submitted","fee","rule","fee_period"):
            wire=deepcopy(original)
            if mutation=="submitted":wire["orders"][0]["submitted_quantity"]-=1
            if mutation=="fee":wire["fills"][0]["transfer_fee_minor"]+=1
            if mutation=="rule":wire["fills"][0]["stock_execution_rules_ref"]=REF
            if mutation=="fee_period":wire["fills"][0]["fee_interval_effective_from"]="2022-04-29"
            wire=seal(wire,"content_digest")
            with self.subTest(mutation=mutation),self.assertRaises(ContractError):_verify_run(BacktestRun.from_dict(wire))


if __name__=="__main__":unittest.main()
