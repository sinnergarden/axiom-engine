"""Synthetic rights transitions and saved v8 accounting; no provider calls."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document, canonical
from axiom_engine.runtime import BacktestRequest, equity_profile, equity_request
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.stock_equity import advance, gaps, pending, register, validate_facts
from test_stock_stream_inputs import source_fixture, artifact_file
from test_stock_stream import execute

REF = 'sha256:' + 'c' * 64
DAYS = ['2023-12-29', '2024-01-02', '2024-01-03', '2024-01-08']


def action(security, **changes):
    result = dict(economic_event_id=REF, revision_ref=REF, identity_status='RESOLVED',
        security_id=security, report_period='2023-06-30', record_date=DAYS[1],
        ex_date=DAYS[2], payment_date='2024-01-06', stock_listing_date='2024-01-06',
        cash_dividend_before_tax_per_share='0.05', stock_distribution_shares_per_share='0.1',
        bonus_shares_per_share=None, capital_transfer_shares_per_share=None,
        available_at='2023-12-28T09:30:00+08:00', first_observed_at='2026-10-09T02:27:35.621825Z',
        raw_batch_ids=['b_synthetic'], source_refs=[REF], aliases=[dict(security_id=security,
        report_period='2023-06-30', announcement_date='2023-12-20', process_status='实施')])
    result.update(changes)
    return result


def fixture(root, *, cash='0.05', stock='0.1', change=None, plan=None):
    manifest, old = source_fixture(root,plan=plan)
    scope = manifest['scope']; security=scope['execution_universe'][0]
    a=action(security,cash_dividend_before_tax_per_share=cash,stock_distribution_shares_per_share=stock)
    if change: change(a)
    facts=dict(contract_version='stock_action_facts_v1',producer='axiom-data',
        snapshot_id='s_synthetic_new_execution_supplement',observed_at='2026-10-09T02:27:35.621825Z',
        availability_basis='declared_vendor_assumption',universe=scope['execution_universe'],
        calendar=scope['calendar'],parent_native_refs=old['market_replay']['source_refs'][4:6],
        supplemental_view_refs=[REF],identity_policy_ref=REF,actions=[a],limitations=['Synthetic resolved facts only.'])
    profile=equity_profile(execution_rules=old['profile']['stock_execution_rules'],
        fee_schedule=old['profile']['stock_fee_schedule'])
    pi=dict(artifact=artifact_file(root,'equity-profile',profile),profile_ref=Document.from_dict(profile).identity,
        **{k:profile[k] for k in ('stock_execution_rules_ref','stock_fee_schedule_ref')})
    request=equity_request(BacktestRequest.from_dict(manifest),profile_input=pi,
        action_facts_artifact=artifact_file(root,'equity-actions',facts)).to_dict()
    return request,manifest,a


def seeded(quantity=100):
    ledger=AccountLedger(cash_minor=10000,calendar=DAYS,settlement_sessions=1,
        positions={'s':dict(quantity=quantity,sellable_quantity=quantity,cost_minor=100000)})
    ledger.advance(DAYS[1]); register(ledger,action('s'),DAYS[1])
    return ledger


class EquityTests(unittest.TestCase):
    def test_cash_receivable_pending_shares_and_weekend_transfer_do_not_add_income_twice(self):
        ledger=seeded(); a=action('s'); ledger.advance(DAYS[2])
        self.assertEqual(gaps(ledger,[a],DAYS[2],'09:30:00+08:00'),[])
        advance(ledger,[a],DAYS[2]); advance(ledger,[a],DAYS[2])
        self.assertEqual((ledger.cash,sum(ledger.receivables.values()),pending(ledger)),(10000,500,{'s':10}))
        self.assertEqual(ledger.account()['positions']['s']['quantity'],100)
        ledger.advance(DAYS[3]); advance(ledger,[a],DAYS[3]); advance(ledger,[a],DAYS[3])
        self.assertEqual((ledger.cash,ledger.receivables,pending(ledger)),(10500,{},{}))
        self.assertEqual(ledger.positions['s'],dict(quantity=110,sellable_quantity=110,cost_minor=100000))
        self.assertEqual([e['reason'] for e in ledger.cash_ledger],['DIVIDEND_EX','DIVIDEND_PAY'])
        self.assertEqual(sum(e['cash_delta_minor']+e['receivable_delta_minor'] for e in ledger.cash_ledger),500)

    def test_fractional_unknown_and_unheld_facts_are_account_specific(self):
        for changes,reason in [({'stock_distribution_shares_per_share':None},'UNKNOWN_SHARE_ENTITLEMENT'),
                ({'cash_dividend_before_tax_per_share':None},'UNKNOWN_CASH_ENTITLEMENT'),
                ({'payment_date':None},'MISSING_PAY_DATE'),({'stock_listing_date':None},'MISSING_LIST_DATE'),
                ({'stock_distribution_shares_per_share':'0.001'},'UNSUPPORTED_FRACTIONAL_ENTITLEMENT'),
                ({'identity_status':'UNRESOLVED'},'UNRESOLVED_ACTION_IDENTITY')]:
            with self.subTest(reason=reason):
                a=action('s',**changes); ledger=seeded()
                self.assertEqual(gaps(ledger,[a],DAYS[2],'09:30:00+08:00')[0]['reason'],reason)
                other=AccountLedger(cash_minor=1,calendar=DAYS,settlement_sessions=1)
                self.assertEqual(gaps(other,[a],DAYS[2],'09:30:00+08:00'),[])

    def test_public_source_and_saved_projection_bind_new_input_without_predicting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,old,a=fixture(root)
            self.assertEqual(manifest['prediction_input'],old['prediction_input'])
            self.assertEqual(manifest['market_input'],old['market_input'])
            wire,view,_=execute(root/'out',manifest)
            self.assertEqual(wire['status'],'COMPLETE')
            self.assertEqual(wire['contract_version'],'backtest_run_v8')
            positions=view.rows['positions']
            before=next(p for p in positions if p['session']==DAYS[1] and p['security_id']==a['security_id'])
            ex=next(p for p in positions if p['session']==DAYS[2] and p['security_id']==a['security_id'])
            self.assertEqual(ex['pending_share_quantity'],before['quantity']//10)
            self.assertEqual(ex['market_value_minor'],int(Decimal(ex['mark_price'])*(ex['quantity']+ex['pending_share_quantity'])*100))

    def test_saved_partial_gap_replays_and_zero_total_does_not_require_listing(self):
        for total,expected in [(None,'BLOCKED'),('0','COMPLETE')]:
            with self.subTest(total=total),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); manifest,_,_=fixture(root,stock=total,change=lambda a:a.update(stock_listing_date=None))
                wire,view,_=execute(root/'out',manifest)
                self.assertEqual(wire['status'],expected)
                if total is None:
                    self.assertEqual(wire['stopped']['reason'],'HELD_EQUITY_FACT_GAP')
                    self.assertEqual(len(view.rows['nav']),1)

    def test_data_identity_alias_assignment_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,_,_=fixture(root)
            facts=Document(Path(manifest['action_facts_artifact']['manifest_uri']).read_text()).to_dict()
            duplicate=deepcopy(facts['actions'][0]); duplicate['economic_event_id']='sha256:'+'a'*64
            facts['actions'].append(duplicate)
            with self.assertRaisesRegex(ContractError,'alias assigned more than once'):
                validate_facts(facts,universe=facts['universe'],calendar=facts['calendar'],parent_refs=facts['parent_native_refs'])

    def test_record_sale_new_buy_and_listing_keep_old_origin_without_free_buy(self):
        from test_csi300 import rules_for
        from test_csi300_runtime import full_request
        from axiom_engine.runtime.evaluation import _verify_run
        from axiom_engine.runtime.episode_evaluation import evaluate_episodes
        days=['2023-12-29','2024-01-02','2024-01-03','2024-01-08','2024-01-09',
            '2024-01-10','2024-01-15','2024-01-16','2024-01-17','2024-01-22']
        rules=rules_for(days=days); security=rules['universe'][0]
        def scores(frame):
            for row in frame['rows']:
                if row['session'] in ('2024-01-03','2024-01-17') and row['security_id']==security:
                    row['score']=-20
        plan=full_request(rules=rules,k=1,mutate_frame=scores).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,_,a=fixture(root,plan=plan,
                change=lambda a:a.update(ex_date='2024-01-09',payment_date='2024-01-09',stock_listing_date='2024-01-22'))
            wire,view,_=execute(root/'out',manifest)
            self.assertEqual(wire['status'],'COMPLETE')
            episodes,metrics=evaluate_episodes(_verify_run(view),None,view.input_run_ref)
            own=[e for e in episodes if e['security_id']==security]
            self.assertEqual(len(own),2)
            original,new=own
            self.assertEqual(original['entry_session'],'2024-01-02')
            self.assertEqual(new['entry_session'],'2024-01-15')
            self.assertEqual(original['status'],'CLOSED')
            self.assertEqual(original['exit_session'],'2024-01-22')
            self.assertGreater(original['dividend_income_minor'],0)
            self.assertEqual(new['dividend_income_minor'],0)
            self.assertEqual(len(new['rights']),0)
            buys=[f for f in view.rows['fills'] if f['security_id']==security and f['side']=='BUY']
            self.assertEqual(len(buys),2)
            self.assertEqual(sum(e['buy_cost_minor'] for e in own),-sum(f['cash_delta_minor'] for f in buys))
            sells=[f for f in view.rows['fills'] if f['security_id']==security and f['side']=='SELL']
            self.assertEqual(sum(e['sell_proceeds_minor'] for e in own),sum(f['cash_delta_minor'] for f in sells))
            self.assertEqual(sum(e['fees_minor'] for e in own),sum(f['fee_minor'] for f in buys+sells))
            self.assertEqual(metrics['attribution_policy'],'record_origin_fifo_v1')

    def test_overlapping_unlisted_rights_need_resolved_registered_share_basis(self):
        ledger=seeded(); first=action('s',stock_listing_date='2024-01-09')
        ledger.advance(DAYS[2]); advance(ledger,[first],DAYS[2])
        second=action('s',economic_event_id='sha256:'+'a'*64,record_date=DAYS[2],ex_date=DAYS[3])
        register(ledger,second,DAYS[2]); ledger.advance(DAYS[3])
        self.assertEqual(gaps(ledger,[second],DAYS[3],'09:30:00+08:00')[0]['reason'],
            'UNRESOLVED_REGISTERED_SHARE_BASIS')

    def test_market_owner_uses_same_rights_path_and_saved_evaluation_never_reopens_source(self):
        from test_stock_market_owner import market_for,bind
        from test_stock_owned_inputs import run_owned
        from axiom_engine.runtime.evaluation import _verify_run
        from axiom_engine.runtime.episode_evaluation import evaluate_episodes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,_,_=fixture(root)
            with market_for(manifest) as market:
                with bind(market,manifest) as inputs:
                    wire,view=run_owned(root/'out',manifest,inputs)
            # Parent native files and predictions are no longer readable.
            for entry in manifest['market_input']['native_inputs']:
                Path(entry['artifact']['manifest_uri']).unlink(missing_ok=True)
            for entry in manifest['prediction_input']['frames']:
                for name in ('fold_spec_artifact','model_metadata_artifact','prediction_artifact'):
                    Path(entry[name]['manifest_uri']).unlink(missing_ok=True)
            Path(manifest['action_facts_artifact']['manifest_uri']).unlink()
            from axiom_engine.runtime.stock_stream_projection import load_stock_backtest_projection
            from test_stock_stream_inputs import LIMITS
            view=load_stock_backtest_projection(root/'out'/'run.json',artifact_reader=lambda r:Path(r['manifest_uri']),limits=LIMITS)
            episodes,_=evaluate_episodes(_verify_run(view),None,view.input_run_ref)
            self.assertTrue(any(e['pending_share_quantity']==0 and e['rights'] for e in episodes))
            import test_stocks as legacy
            from test_analysis_evaluation import RF
            from axiom_engine.runtime import (stock_dividend_scope,evaluate_backtest,long_history_evaluation_spec,
                evaluate_saved_analysis,analysis_evaluation_spec)
            with patch.object(legacy,'DAYS',manifest['scope']['calendar']): benchmark=legacy.benchmark()
            base=evaluate_backtest(view,benchmark=benchmark,spec=long_history_evaluation_spec(),
                dividend_scope=stock_dividend_scope(view))
            report=evaluate_saved_analysis(view,base,benchmarks=dict(CSI300=benchmark,SSE_COMPOSITE=None,NASDAQ100=None),
                spec=analysis_evaluation_spec(risk_free=RF))
            self.assertEqual(report.to_dict()['input_run_ref'],view.input_run_ref)

    def test_resolved_total_with_null_components_does_not_require_factor_rounding(self):
        from test_csi300_runtime import full_request
        def factors(batches,membership):
            for row in batches[3]['records']:
                if row['session']>=DAYS[2]: row['factor']=1.234567
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,_,_=fixture(root,plan=full_request(mutate_native=factors).to_dict())
            wire,view,_=execute(root/'out',manifest)
            self.assertEqual(wire['status'],'COMPLETE')
            self.assertEqual(wire['source_audit']['counts']['action_blocks'],0)
            self.assertGreater(wire['source_audit']['counts']['action_diagnostics'],0)

    def test_source_relative_identity_new_snapshot_and_view_binding_are_explicit(self):
        from axiom_engine.runtime.stock_stream_inputs import StockInputSource
        from axiom_engine.runtime.stock_stream import audit_stock_backtest_source
        from test_stock_stream_inputs import LIMITS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,old,a=fixture(root,change=lambda a:a.update(economic_event_id='tushare:distribution:synthetic'))
            wire,view,_=execute(root/'out',manifest)
            self.assertIn(a['economic_event_id'],wire['final_account']['equity_entitlements'])
            self.assertEqual(view.account_events['equity_facts']['actions'][0]['bonus_shares_per_share'],None)
            facts=Document(Path(manifest['action_facts_artifact']['manifest_uri']).read_text()).to_dict()
            facts['snapshot_id']=old['market_input']['execution_snapshot_id']
            manifest=equity_request(BacktestRequest.from_dict(old),profile_input=manifest['profile_input'],
                action_facts_artifact=artifact_file(root,'wrong-snapshot',facts))
            with self.assertRaisesRegex(ContractError,'new execution Snapshot'):
                audit_stock_backtest_source(manifest,source=StockInputSource(),block_sessions=2,limits=LIMITS)

    def test_saved_rights_and_cost_corruption_fail_without_original_input(self):
        from axiom_engine.runtime.stock_equity import verify_saved
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,_,_=fixture(root)
            wire,view,_=execute(root/'out',manifest)
            for field,session in [('pending_share_quantity',DAYS[2]),('cost_minor',DAYS[3])]:
                with self.subTest(field=field):
                    rows=deepcopy(view.rows)
                    p=next(p for p in rows['positions'] if p['session']==session)
                    p[field]+=1
                    with self.assertRaisesRegex(ContractError,'position/rights/cost mismatch'):
                        verify_saved(wire,rows,view.profile)

    def test_all_payments_precede_all_listings_and_t1_remains_separate(self):
        ledger=seeded(); first=action('s'); second=action('s',economic_event_id='tushare:second')
        register(ledger,second,DAYS[1]); ledger.advance(DAYS[2]); advance(ledger,[first,second],DAYS[2])
        ledger.advance(DAYS[3]); advance(ledger,[first,second],DAYS[3])
        pay=[r['sequence'] for r in ledger.cash_ledger if r['reason']=='DIVIDEND_PAY']
        listings=[r['sequence'] for r in ledger.position_ledger if r['reason']=='EQUITY_LISTING']
        self.assertLess(max(pay),min(listings))
        self.assertEqual(ledger.positions['s']['quantity'],120)

    def test_post_ex_new_buyer_does_not_inherit_missing_old_record_facts(self):
        from test_csi300 import rules_for
        from test_csi300_runtime import full_request
        days=DAYS+['2024-01-09']; rules=rules_for(days=days); security=rules['universe'][0]
        def scores(frame):
            for row in frame['rows']:
                if row['security_id']==security and row['session']<'2024-01-03': row['score']=-20
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=full_request(rules=rules,k=1,mutate_frame=scores).to_dict()
            manifest,_,_=fixture(root,plan=plan,stock=None,cash=None,
                change=lambda a:a.update(record_date=None,payment_date=None,stock_listing_date=None))
            wire,view,_=execute(root/'out',manifest)
            self.assertEqual(wire['status'],'COMPLETE')
            self.assertTrue(any(f['security_id']==security and f['side']=='BUY' and f['session']=='2024-01-08'
                for f in view.rows['fills']))
            self.assertEqual(wire['final_account']['equity_entitlements'],{})


if __name__=='__main__': unittest.main()
