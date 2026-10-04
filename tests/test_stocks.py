"""Hand-reconciled narrow stock contracts; no Data/Research execution."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from axiom_engine.core import StockPredictionFrame, plan_stock_portfolio
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_portfolio import BUDGET_BASIS
from axiom_engine.runtime import (BacktestRequest, run_backtest, stock_daily_open_profile,
    stock_market_from_batches, stock_dividend_scope, save_backtest_run, load_backtest_run,
    long_history_evaluation_spec, evaluate_backtest, BenchmarkSeries,
    save_backtest_evaluation, load_backtest_evaluation)
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.backtest import _simulate
from axiom_engine.runtime.stock_inputs import ELIGIBILITY_ID, support_ref, supported_universe
from axiom_engine.runtime.stock_inputs import validate_stock_request
from axiom_engine.runtime.stock_market import STOCK_EVENT_FIELDS
from axiom_engine.runtime.stock_evidence import native_batches, native_ref
from axiom_engine.runtime.evaluation_inputs import benchmark_rows

REF = "sha256:" + "a" * 64
SNAPSHOT = "s_" + "b" * 64
DAYS = ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-08"]
SECURITIES = [f"cnstock.000{100+i}.SZ.20000101" for i in range(6)]


def predictions(scores=None):
    scores = scores or [-1, -2, -3, -4, -5, -6]
    return {"contract_version": "stock_prediction_run_v1", "signal_run_ref": REF,
        "signal_stage": "prediction_raw", "score_semantics": "forward_5_session_cs_zscore_prediction", "score_unit": "dimensionless",
        "feature_ref": REF, "model_ref": REF, "limitations": ["synthetic golden"], "universe": SECURITIES,
        "rows": [{"security_id": s, "session": d, "knowledge_cutoff": d + "T20:30:00+08:00",
                  "available_at": d + "T12:00:00Z", "score": scores[i], "valid": True,
                  "invalid_reason": None, "member": True, "source_refs": [REF, "c" * 64]}
                 for d in DAYS for i, s in enumerate(SECURITIES)]}


def batch(domain, names, records, units, *, time_field=None):
    query = {"purpose": "market_replay", "pit_policy": "best_effort_vendor_v1", "fields": list(names),
             "symbols": SECURITIES, "sessions": DAYS, "cutoff_by_session": {d:d+"T20:30:00+08:00" for d in DAYS},
             "price_basis": "unadjusted", "adjustment_anchor": None, "universe_id": None, "policy_by_session": None}
    if time_field:
        query = {k:query[k] for k in ('purpose','pit_policy','fields','symbols')}
        query.update(time_field=time_field, start=DAYS[0], end=DAYS[-1], cutoff=DAYS[-1]+"T20:30:00+08:00", filters={})
    meta = {}
    for name in names:
        keyed = []
        for row in records:
            keys = {k:row[k] for k in (("security_id", "report_period", "announcement_date", "process_status") if time_field else ("security_id", "session"))}
            when = DAYS[0] if time_field else row["session"]
            keys.update(usable_from=when+"T12:00:00Z", status="value", missing_reason=None)
            if name == "market_state":
                keys["missing_reason"] = "status_source_missing" if row[name] == "unknown_status" else None
            if name in ("up_limit", "down_limit"):
                keys["usable_from"] = when+"T02:00:00Z"
            keyed.append(keys)
        meta[name] = {"unit": units.get(name), "by_key": keyed}
    return {"context": {"contract_version": "data_batch_v1", "domain": domain, "snapshot_id": SNAPSHOT,
            "reader_version": "synthetic_stock_reader/1", "query": query, "limitations": ["synthetic golden"]},
            "records": records, "field_meta": meta}


def native_inputs(events=None, state="unknown_status"):
    keys = [{"session":d,"security_id":s} for d in DAYS for s in SECURITIES]
    states = batch("market_daily", ["market_state"], [{**k,"market_state":state} for k in keys], {})
    states['context']['query']['fields']=['close']
    states['context']['domain']='market_state_diagnostics'
    prices = batch("market_daily", ["open","high","low","close","volume_shares","amount_cny"],
                   [{**k,"open":10.0,"high":10.0,"low":10.0,"close":10.0,"volume_shares":1000000,"amount_cny":10000000.0} for k in keys],
                   {"open":"CNY/share","high":"CNY/share","low":"CNY/share","close":"CNY/share","volume_shares":"shares","amount_cny":"CNY"})
    limits = batch("price_limits", ["up_limit","down_limit"], [{**k,"up_limit":11.0,"down_limit":9.0} for k in keys],
                   {"up_limit":"CNY/share","down_limit":"CNY/share"})
    factors = batch("adjustment_factors", ["factor"], [{**k,"factor":1.0} for k in keys], {"factor":"dimensionless"})
    actions = [batch("corporate_actions", STOCK_EVENT_FIELDS, deepcopy(events or []),
        {"cash_dividend_before_tax_per_share":"CNY/share","bonus_shares_per_share":"shares/share","capital_transfer_shares_per_share":"shares/share"},time_field=time_field)
        for time_field in ("ex_date","record_date")]
    return [states,prices,limits,factors,*actions]


def cash_event(status="实施", issue=None):
    return {"security_id":SECURITIES[0],"report_period":"2023-06-30","announcement_date":"2023-12-28",
        "process_status":status,"implementation_announcement_date":"2023-12-28","record_date":"2024-01-02","ex_date":"2024-01-03",
        "cash_dividend_before_tax_per_share":0.1,"bonus_shares_per_share":0.0,"capital_transfer_shares_per_share":0.0,
        "source_issue":issue,"source_candidate_count":2 if issue else 1,"candidate_economic_dates":None}


def request(*, inputs=None, frame=None, strict=False):
    frame = frame or predictions()
    universe = supported_universe(frame["universe"])
    market = stock_market_from_batches(batches=inputs or native_inputs(), universe=universe, calendar=DAYS).to_dict()
    proof = {"contract_version":"stock_snapshot_pair_admission_v1","signal_run_ref":frame["signal_run_ref"],
        "feature_ref":frame["feature_ref"],"model_ref":frame["model_ref"],"scope":{"execution_security_ids":universe,"initial_sessions":DAYS,
        "membership_security_ids":frame['universe'],'warmup_sessions':['2023-12-28']},
        "model":{"snapshot":"s_model"},"execution":{"snapshot":SNAPSHOT},"status":"NUMERIC_POLICY_IDENTITY_PAIR_PASS"}
    count=len(universe)*len(DAYS)
    proof.update(listing_identity_equal_all_83=True,
        prediction_membership={'compared':len(frame['rows']),'saved_prediction_rows':len(frame['rows']),'model_mismatch':[],'execution_mismatch':[]},
        previous_close_basis_checks={**{k:True for k in ('all_available_at_feature_knowledge_cutoff','all_available_before_decision','equal_all_83_23','listing_suffix_and_SZSE_identity_all_83')},
        'paired_rows_per_root':count,'checked_rows_both_roots':count*2},
        previous_close_basis_artifact={k:REF for k in ('basis_ref','canonical_content_digest','file_digest')},batches={})
    def paired(n,names):
        return {'left_rows':n,'right_rows':n,'missing_keys_left':[],'missing_keys_right':[],
            'fields':{name:{'counts':{k:n for k in ('values_equal','usable_from_equal','missing_reason_equal','availability_basis_equal',
                'raw_batch_id_equal','revision_id_equal','revision_sequence_equal','first_observed_at_equal','source_available_at_equal','evidence_ref_equal')},
                'units_equal':True,'model_unit':None,'execution_unit':None,'difference_samples':[]} for name in names}}
    names=['open','high','low','close','volume_shares','amount_cny']
    proof['primary_comparison']={'market':paired(count,names),'factor':paired(count,['factor']),
                                'membership':paired(len(frame['universe'])*len(DAYS),['is_member'])}
    proof['warmup_comparison']={'market':paired(len(universe),names),'factor':paired(len(universe),['factor'])}
    native=[x['batch'] for x in market['source_evidence']]
    for side in ('execution','model'):
        for kind,index in (('states',0),('market',1),('factor',3),('membership',0),('warmup-market',1),('warmup-factor',3)):
            b=native[index]
            q=deepcopy(b['context']['query'])
            if kind=='membership':q.update(purpose='decision_facts',fields=['is_member'],symbols=frame['universe'],universe_id='csi300')
            if kind.startswith('warmup-'):q.update(sessions=['2023-12-28'],cutoff_by_session={'2023-12-28':'2023-12-28T20:30:00+08:00'})
            proof['batches'][side+'-'+kind]={'wire_ref':market['source_evidence'][index]['reference'],'file_digest':REF,
                'snapshot_id':proof[side]['snapshot'],'reader_version':b['context']['reader_version'],'query':q}
    proof["admission_ref"] = Document.from_dict(proof).identity
    return BacktestRequest.from_dict({"contract_version":"backtest_request_v3","account_id":"synthetic-stock",
        "start_session":DAYS[1],"end_session":DAYS[-1],"signal_frame":frame,"market_replay":market,
        "initial_account":{"cash_minor":1000000,"positions":{}},"profile":stock_daily_open_profile(unknown_status_policy="block" if strict else "stock_daily_observed"),
        "prediction_universe":frame["universe"],"execution_universe":universe,"supported_universe_ref":support_ref(universe),
        "portfolio_policy":{"eligibility_id":ELIGIBILITY_ID,"top_k":5,"rebalance":"weekly_first_trading_session","budget_basis":BUDGET_BASIS},
        "admission_ref":proof["admission_ref"],"admission_evidence":proof,"stock_action_policy":"observed_implemented_only"})


def context():
    return {"trade_session":DAYS[1],"feature_session":DAYS[0],"decision_time":DAYS[1]+"T00:55:00Z",
        "knowledge_cutoff":DAYS[0]+"T20:30:00+08:00","reference_prices":{s:{"price":"10","session":DAYS[0],"available_at":DAYS[0]+"T12:00:00Z","source_refs":[REF]} for s in SECURITIES},
        "lot_size":100,"commission_rate":"0.0003","minimum_commission_minor":500,"slippage_bps":"0","account_state_version":0,
        "supported_security_ids":SECURITIES,"supported_universe_ref":support_ref(SECURITIES)}


def benchmark():
    records=[{"security_id":"000300.SH","session":d,"close":100.0} for d in DAYS]
    b=batch("benchmark_daily",["close"],records,{"close":"index points"})
    b["context"]["query"]["symbols"]=["000300.SH"]
    ref=Document.from_dict(b).identity
    return BenchmarkSeries.from_dict({"contract_version":"benchmark_series_v1","security_id":"000300.SH",
        "series_kind":"price_index_excluding_dividends","unit":"index points","calendar":DAYS,"rows":benchmark_rows(b,DAYS),
        "source_refs":[ref],"source_evidence":[{"reference":ref,"batch":b}],"limitations":["synthetic golden"]})


class StockTests(unittest.TestCase):
    def plan(self, frame=None, account=None, ctx=None):
        return plan_stock_portfolio(StockPredictionFrame.from_dict(frame or predictions()),
            account=account or {"cash_minor":1000000,"positions":{},"version":0},context=ctx or context()).to_dict()

    def test_negative_scores_and_ties_use_exact_top_five(self):
        p=self.plan();self.assertEqual(p["selected_security_ids"],SECURITIES[:5])
        self.assertEqual(set(p["targets"].values()),{200})
        self.assertEqual(self.plan(predictions([-1]*6))["selected_security_ids"],SECURITIES[:5])

    def test_invalid_member_no_decision_nonmember_exclusion(self):
        p=predictions();r=p["rows"][0];r.update(valid=False,score=None,invalid_reason="MISSING")
        result=self.plan(p);self.assertEqual(result["status"],"NO_DECISION");self.assertEqual(result["intents"],[])
        r["member"]=False;self.assertEqual(self.plan(p)["selected_security_ids"],SECURITIES[1:])
        p["rows"][1]["member"]=False;self.assertEqual(self.plan(p)["status"],"NO_DECISION")

    def test_clock_account_and_coverage_fail_before_mutation(self):
        p=predictions();p["rows"][0]["available_at"]=DAYS[0]+"T12:45:00Z"
        with self.assertRaises(ValueError):self.plan(p)
        p=predictions();p["rows"].pop()
        with self.assertRaises(ValueError):self.plan(p)
        c=context();c["account_state_version"]=1
        with self.assertRaises(ValueError):self.plan(ctx=c)

    def test_required_group_and_fixed_instant_rejected_before_ledger(self):
        from unittest.mock import patch
        w=request().to_dict();w['signal_frame']['rows']=[r for r in w['signal_frame']['rows'] if r['session']!=DAYS[2]]
        with patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('ledger must not start')):
            with self.assertRaisesRegex(ValueError,'previous-session'):run_backtest(BacktestRequest.from_dict(w))
        p=predictions()
        for r in p['rows']:r.update(knowledge_cutoff=r['session']+'T23:30:00+08:00',available_at=r['session']+'T23:00:00+08:00')
        c=context();c['knowledge_cutoff']=DAYS[0]+'T23:30:00+08:00'
        with self.assertRaisesRegex(ValueError,'clock'):self.plan(p,ctx=c)
        p=predictions()
        for r in p['rows']:r['knowledge_cutoff']=r['session']+'T12:30:00Z'
        self.assertEqual(self.plan(p)['status'],'DECISION_COMPLETE')

    def test_pair_identity_cannot_replace_full_admission_closure(self):
        for mutation in ('membership','basis','comparison','missing_count','batch'):
            w=request().to_dict();proof=w['admission_evidence']
            if mutation=='membership':proof['scope']['membership_security_ids']=SECURITIES[:5]
            if mutation=='basis':proof['previous_close_basis_checks']['all_available_before_decision']=False
            if mutation=='comparison':proof['primary_comparison']['market']['fields']['close']['counts']['values_equal']-=1
            if mutation=='missing_count':proof['primary_comparison']['market']['fields']['close']['counts'].pop('values_equal')
            if mutation=='batch':proof['batches']['execution-factor']['wire_ref']=REF
            proof.pop('admission_ref');proof['admission_ref']=Document.from_dict(proof).identity;w['admission_ref']=proof['admission_ref']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_stock_request(w)

    def test_action_business_key_conflict_cannot_create_two_receivables(self):
        inputs=native_inputs([cash_event()]);inputs[-1]['records'][0]['cash_dividend_before_tax_per_share']=0.2
        with self.assertRaisesRegex(ValueError,'conflicting native'):request(inputs=inputs)

    def test_native_basis_range_and_field_visibility_are_fenced(self):
        for mutation in ('adjusted','range','future_cutoff','factor_unavailable','cash_unavailable'):
            inputs=native_inputs([cash_event()])
            if mutation=='adjusted':inputs[1]['context']['query']['price_basis']='adjusted'
            if mutation=='range':inputs[-1]['context']['query']['start']=DAYS[-1]
            if mutation=='future_cutoff':inputs[-1]['context']['query']['cutoff']='2030-01-01T20:30:00+08:00'
            if mutation=='factor_unavailable':inputs[3]['field_meta']['factor']['by_key'][0]['missing_reason']='not_visible_at_cutoff'
            if mutation=='cash_unavailable':inputs[-1]['field_meta']['cash_dividend_before_tax_per_share']['by_key'][0]['status']='unavailable'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):request(inputs=inputs)

    def test_cash_ex_alone_does_not_explain_factor_magnitude(self):
        inputs=native_inputs([cash_event()])
        for row in inputs[3]['records']:
            if row['security_id']==SECURITIES[0] and row['session']>=DAYS[2]:row['factor']=100.0
        w=run_backtest(request(inputs=inputs)).to_dict()
        self.assertEqual(w['status'],'BLOCKED');self.assertEqual(w['stopped']['session'],DAYS[2])

    def test_dropped_member_sell_and_budget_excludes_receivables(self):
        p=predictions();p["rows"][5].update(member=False,valid=False,score=None,invalid_reason="NONMEMBER")
        a={"cash_minor":0,"version":0,"positions":{SECURITIES[5]:{"quantity":1000,"sellable_quantity":900}}}
        result=self.plan(p,a);self.assertEqual(result["intents"][0]["quantity"],900)
        self.assertEqual(result["trace"][-1]["reference_budget_minor"],1000000)

    def test_observed_vs_strict_and_saved_only_readers(self):
        observed=run_backtest(request());strict=run_backtest(request(strict=True))
        w=observed.to_dict();self.assertEqual(w["contract_version"],"backtest_run_v3")
        self.assertTrue(w["fills"]);self.assertEqual(strict.to_dict()["fills"],[])
        self.assertTrue(all(f["market_state"]=="unknown_status" and f["execution_admission"]=="STOCK_OBSERVED_DAILY_ASSUMPTION" for f in w["fills"]))
        self.assertEqual(w["orders"][0]["field_available_at"]["limit_up"],DAYS[1]+"T02:00:00Z")
        report=evaluate_backtest(observed,spec=long_history_evaluation_spec(),benchmark=benchmark(),dividend_scope=stock_dividend_scope(observed))
        self.assertEqual(report.to_dict()["period_metrics"]["account"]["cagr_status"],"INSUFFICIENT_SPAN")
        with tempfile.TemporaryDirectory() as d:
            rp,ep=Path(d)/'run.json',Path(d)/'evaluation.json';save_backtest_run(observed,rp);save_backtest_evaluation(report,ep)
            self.assertEqual(load_backtest_run(rp).payload,observed.payload)
            self.assertEqual(load_backtest_evaluation(ep).payload,report.payload)

    def test_nonimplemented_conflict_diagnostics_do_not_remove_top_five(self):
        event=cash_event("股东大会通过","ambiguous_action_identity_or_revision")
        for name in ('cash_dividend_before_tax_per_share','bonus_shares_per_share','capital_transfer_shares_per_share','record_date','ex_date'):event[name]=None
        req=request(inputs=native_inputs([event]));w=req.to_dict()
        self.assertEqual(w['market_replay']['action_blocks'],[]);self.assertEqual(w['market_replay']['cash_dividends'],[])
        self.assertEqual(run_backtest(req).to_dict()['decisions'][0]['selected_security_ids'],SECURITIES[:5])
        self.assertEqual(w['market_replay']['action_diagnostics'][0]['native_record'],event)

    def test_implemented_uncertainty_blocks_candidate_and_held_quantity_change_stops(self):
        ambiguous=cash_event(issue="ambiguous_action_identity_or_revision")
        ambiguous['ex_date']=DAYS[1]
        req=request(inputs=native_inputs([ambiguous]))
        w=run_backtest(req).to_dict();self.assertEqual(w['orders'][0]['reason'],'UNSUPPORTED_STOCK_ACTION')
        inputs=native_inputs()
        for row in inputs[3]['records']:
            if row['security_id']==SECURITIES[0] and row['session']==DAYS[-1]:row['factor']=2.0
        stopped=run_backtest(request(inputs=inputs));w=stopped.to_dict()
        self.assertEqual(w['status'],'BLOCKED');self.assertEqual(w['stopped']['session'],DAYS[-1]);self.assertIsNone(w['metrics']['total_return'])
        with self.assertRaises(ValueError):evaluate_backtest(stopped,spec=long_history_evaluation_spec(),benchmark=benchmark())

    def test_native_projection_tampering_is_rejected(self):
        w=request().to_dict();w['market_replay']['rows'][0]['close']='10.01'
        with self.assertRaises(ValueError):run_backtest(BacktestRequest.from_dict(w))

    def test_large_native_coverage_is_bundle_local_shared_and_hash_checked(self):
        inputs=native_inputs([cash_event()]);coverage={'supplier_scope':'x'*(1024*1024)}
        for i in (2,4,5):inputs[i]['context']['coverage']=coverage
        req=request(inputs=inputs);market=req.to_dict()['market_replay']
        self.assertEqual(len(market['coverage_bundle']),1)
        self.assertEqual(native_batches(market['source_evidence'],market['coverage_bundle']),inputs)
        self.assertEqual(market['source_refs'],[native_ref(b) for b in inputs])
        self.assertLess(len(req.payload),200000)
        inline=req.to_dict();inline_market=inline['market_replay'];inline_market.pop('coverage_bundle')
        inline_market['source_evidence']=[{'reference':native_ref(b),'batch':b} for b in inputs]
        validate_stock_request(inline)
        run=run_backtest(req);scope=stock_dividend_scope(run).to_dict()
        self.assertEqual(scope['coverage_bundle'],market['coverage_bundle'])
        report=evaluate_backtest(run,spec=long_history_evaluation_spec(),benchmark=benchmark(),dividend_scope=stock_dividend_scope(run))
        with tempfile.TemporaryDirectory() as d:
            rp,ep=Path(d)/'run.json',Path(d)/'evaluation.json'
            save_backtest_run(run,rp);save_backtest_evaluation(report,ep)
            self.assertEqual(load_backtest_run(rp).payload,run.payload)
            self.assertEqual(load_backtest_evaluation(ep).payload,report.payload)
            old_run=run_backtest(BacktestRequest.from_dict(inline));old_path=Path(d)/'old-inline.json'
            save_backtest_run(old_run,old_path);self.assertEqual(load_backtest_run(old_path).payload,old_run.payload)
        broken=deepcopy(market);broken['coverage_bundle'][0]['payload']='AA=='
        with self.assertRaises(ValueError):native_batches(broken['source_evidence'],broken['coverage_bundle'])
        broken=deepcopy(market);broken['coverage_bundle']=[]
        with self.assertRaises(ValueError):native_batches(broken['source_evidence'],broken['coverage_bundle'])
        import base64,gzip,hashlib
        broken=deepcopy(market);item=broken['coverage_bundle'][0]
        compressed=gzip.compress(b'x'*(item['uncompressed_bytes']*2),mtime=0)
        item.update(payload=base64.b64encode(compressed).decode(),compressed_digest='sha256:'+hashlib.sha256(compressed).hexdigest())
        with self.assertRaises(ValueError):native_batches(broken['source_evidence'],broken['coverage_bundle'])

    def test_cash_record_ex_unknown_pay_and_budget(self):
        run=run_backtest(request(inputs=native_inputs([cash_event()])));w=run.to_dict()
        ex=[x for x in w['cash_ledger'] if x['reason']=='DIVIDEND_EX']
        self.assertEqual(ex[0]['receivable_delta_minor'],2000)
        self.assertEqual(w['final_account']['receivable_minor'],2000)
        self.assertEqual(w['decisions'][-1]['trace'][-1]['reference_budget_minor'],w['nav'][-2]['cash_minor']+w['nav'][-2]['market_value_minor'])
        report=evaluate_backtest(run,spec=long_history_evaluation_spec(),benchmark=benchmark(),dividend_scope=stock_dividend_scope(run)).to_dict()
        self.assertEqual(report['episode_metrics']['payment_unknown_count'],1)
        self.assertEqual(report['episodes'][0]['dividends'][0]['payment_status'],'UNKNOWN')

    def test_fee_cash_t_plus_one_golden(self):
        profile=stock_daily_open_profile(unknown_status_policy='stock_daily_observed')
        row=request().to_dict()['market_replay']['rows'][6]
        ledger=AccountLedger(cash_minor=1000000,calendar=DAYS,settlement_sessions=1)
        ledger.advance(DAYS[1]);intent={'security_id':SECURITIES[0],'side':'BUY','quantity':100}
        _simulate(intent,row,ledger,profile,DAYS[1],REF,0,stock_market={'action_blocks':[]})
        self.assertEqual(ledger.cash,899499);self.assertEqual(ledger.fills[0]['fee_minor'],501)
        sell={**intent,'side':'SELL'}
        self.assertEqual(_simulate(sell,row,ledger,profile,DAYS[1],REF,1,stock_market={'action_blocks':[]})['filled_quantity'],0)
        ledger.advance(DAYS[2]);_simulate(sell,row,ledger,profile,DAYS[2],REF,2,stock_market={'action_blocks':[]})
        self.assertEqual(ledger.cash,998948);self.assertEqual(ledger.fills[-1]['stamp_tax_minor'],50)
        self.assertEqual(ledger.fills[-1]['transfer_fee_minor'],1)

    def test_suspension_tick_capacity_odd_exit_and_maximum_order(self):
        profile=stock_daily_open_profile(unknown_status_policy='stock_daily_observed')
        row=request().to_dict()['market_replay']['rows'][6];ledger=AccountLedger(cash_minor=2000000000,calendar=DAYS,settlement_sessions=1)
        intent={'security_id':SECURITIES[0],'side':'BUY','quantity':2000000}
        r=deepcopy(row);r['market_state']='suspended'
        self.assertEqual(_simulate(intent,r,ledger,profile,DAYS[1],REF,0,stock_market={'action_blocks':[]})['reason'],'NOT_TRADING')
        r=deepcopy(row);r['open']='10.001'
        self.assertEqual(_simulate(intent,r,ledger,profile,DAYS[1],REF,1,stock_market={'action_blocks':[]})['reason'],'PRICE_TICK')
        r=deepcopy(row);r['volume_shares']=30000000
        self.assertEqual(_simulate(intent,r,ledger,profile,DAYS[1],REF,2,stock_market={'action_blocks':[]})['filled_quantity'],1000000)
        odd=AccountLedger(cash_minor=1000,calendar=DAYS,settlement_sessions=1,positions={SECURITIES[0]:{'quantity':150,'sellable_quantity':150,'cost_minor':1000}})
        sell={'security_id':SECURITIES[0],'side':'SELL','quantity':150}
        r['volume_shares']=1490
        self.assertEqual(_simulate(sell,r,odd,profile,DAYS[1],REF,3,stock_market={'action_blocks':[]})['filled_quantity'],100)
        sell['quantity']=50;r['volume_shares']=500
        self.assertEqual(_simulate(sell,r,odd,profile,DAYS[2],REF,4,stock_market={'action_blocks':[]})['filled_quantity'],50)

    def test_verified_payment_is_idempotent_transfer(self):
        ledger=AccountLedger(cash_minor=1000,calendar=DAYS,settlement_sessions=1)
        action={'contract_version':'stock_cash_action_v1','event_id':REF,'record_session':DAYS[1],'ex_session':DAYS[2],
                'pay_session':DAYS[3],'cash_before_tax_per_share':'0.1'}
        ledger.dividend(action,'EX',100);self.assertEqual(ledger.cash,1000)
        ledger.dividend(action,'PAY',100);self.assertEqual(ledger.cash,2000)
        ledger.dividend(action,'PAY',100);self.assertEqual(ledger.cash,2000)
        with self.assertRaises(ValueError):ledger.dividend(action,'PAY',200)


if __name__=='__main__':unittest.main()
