"""Integer replay allocation and saved accounting; no real source/account run."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError
from axiom_engine.runtime import (settle_stock_share_allocation,
    evaluate_backtest, long_history_evaluation_spec, stock_dividend_scope,
    evaluate_saved_analysis, analysis_evaluation_spec, save_backtest_evaluation,
    load_backtest_evaluation, load_stock_backtest_projection)
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.equity_episodes import _settled_grants
from axiom_engine.runtime.fill_display import _contract_unit
from axiom_engine.runtime.stock_equity import FLOOR_POLICY, SIMULATION, advance, gaps, pending, register
from test_stock_equity import action, fixture, DAYS, REF
from test_stock_stream import execute, map_run_ids
from test_stock_stream_inputs import LIMITS
from axiom_engine.runtime.stock_stream_inputs import StockInputSource
from test_stock_stream_projection import reseal_run
from test_analysis_evaluation import RF
import test_stocks as legacy


class ShareAllocationTests(unittest.TestCase):
    def test_announced_ratio_floor_and_confirmed_integer_credits_are_distinct(self):
        args = dict(registered_quantity=1000, shares_per_share='0.2965786')
        floor = settle_stock_share_allocation(**args, policy=FLOOR_POLICY)
        self.assertEqual(floor['allocated_grant'], 296)
        self.assertEqual(Decimal(floor['theoretical_grant']), Decimal('296.5786'))
        self.assertEqual(Decimal(floor['fractional_remainder']), Decimal('0.5786'))
        self.assertEqual(floor['possible_additional_share'], 1)
        for actual in (296, 297):
            credit = settle_stock_share_allocation(**args, policy='confirmed_integer_credit',
                actual_quantity=actual, evidence_refs=[REF])
            self.assertEqual(credit['allocated_grant'], actual)
            self.assertEqual(credit['evidence_refs'], [REF])
            self.assertEqual(credit['possible_additional_share'], 0)
        with self.assertRaises(ContractError): settle_stock_share_allocation(**args)

    def test_invalid_credit_or_implied_fallback_is_rejected(self):
        args = dict(registered_quantity=1000, shares_per_share='0.2965786', policy='confirmed_integer_credit')
        for extra in ({}, dict(actual_quantity=297), dict(actual_quantity=True,evidence_refs=[REF]),
                      dict(actual_quantity=-1,evidence_refs=[REF]), dict(actual_quantity=297,evidence_refs=[{}]),
                      dict(actual_quantity=297,evidence_refs=[REF,REF])):
            with self.subTest(extra=extra), self.assertRaises(ContractError):
                settle_stock_share_allocation(**args, **extra)
        with self.assertRaises(ContractError):
            settle_stock_share_allocation(registered_quantity=True, shares_per_share='0.5', policy=FLOOR_POLICY)
        with self.assertRaises(ContractError):
            settle_stock_share_allocation(registered_quantity=100, shares_per_share='0.5',
                policy=FLOOR_POLICY, actual_quantity=50)

    def test_zero_tail_zero_registration_and_integer_grants_have_no_uncertainty(self):
        for quantity,rate,expected in ((1000,'0.4',400),(0,'0.2965786',0),(100,'0',0)):
            strict = settle_stock_share_allocation(registered_quantity=quantity, shares_per_share=rate)
            floor = settle_stock_share_allocation(registered_quantity=quantity, shares_per_share=rate,policy=FLOOR_POLICY)
            self.assertEqual(strict['allocated_grant'], expected)
            self.assertEqual(floor['allocated_grant'], expected)
            self.assertEqual(Decimal(floor['fractional_remainder']), 0)
            self.assertEqual(floor['possible_additional_share'], 0)

    def test_integer_pending_listing_cash_and_next_registration(self):
        ledger = AccountLedger(cash_minor=0, calendar=DAYS, settlement_sessions=1,
            positions={'s':dict(quantity=101,sellable_quantity=101,cost_minor=101000)})
        a = action('s',cash_dividend_before_tax_per_share='1',stock_distribution_shares_per_share='0.5')
        ledger.advance(DAYS[1]); register(ledger,a,DAYS[1]); ledger.advance(DAYS[2])
        self.assertEqual(gaps(ledger,[a],DAYS[2],'09:30:00+08:00',share_allocation_policy=FLOOR_POLICY), [])
        advance(ledger,[a],DAYS[2],share_allocation_policy=FLOOR_POLICY)
        self.assertEqual(pending(ledger), {'s':50})
        self.assertEqual(ledger.positions['s']['quantity'], 101)
        self.assertEqual(sum(ledger.receivables.values()), 10100)
        before = deepcopy(ledger.position_ledger)
        advance(ledger,[a],DAYS[2],share_allocation_policy=FLOOR_POLICY)
        self.assertEqual(ledger.position_ledger,before)
        ledger.advance(DAYS[3]); advance(ledger,[a],DAYS[3],share_allocation_policy=FLOOR_POLICY)
        self.assertEqual(ledger.positions['s'],dict(quantity=151,sellable_quantity=151,cost_minor=101000))
        self.assertEqual(ledger.cash,10100); self.assertEqual(ledger.receivables,{})
        self.assertEqual(ledger.fills,[]); self.assertEqual(pending(ledger),{})
        later=action('s',economic_event_id='second',record_date=DAYS[3],ex_date='2024-01-09')
        register(ledger,later,DAYS[3])
        self.assertEqual(ledger.equity_entitlements['second']['record_quantity'],151)

    def test_episode_attribution_uses_one_account_grant_not_per_episode_floor(self):
        owners = [('old',Fraction(1)),('new',Fraction(2))]
        allocated = settle_stock_share_allocation(registered_quantity=3,shares_per_share='0.4',policy=FLOOR_POLICY)
        grants = _settled_grants(owners,allocated['allocated_grant'])
        self.assertEqual(grants,[Fraction(1,3),Fraction(2,3)])
        self.assertEqual(sum(grants),1)
        self.assertNotEqual(sum(int(q*Fraction('0.4')) for _,q in owners),sum(grants))

    def test_floor_does_not_relax_unknown_cash_identity_or_share_basis(self):
        for changes,reason in (({'identity_status':'UNRESOLVED'},'UNRESOLVED_ACTION_IDENTITY'),
                ({'cash_dividend_before_tax_per_share':None},'UNKNOWN_CASH_ENTITLEMENT'),
                ({'stock_distribution_shares_per_share':None},'UNKNOWN_SHARE_ENTITLEMENT')):
            ledger=AccountLedger(cash_minor=0,calendar=DAYS,settlement_sessions=1,
                positions={'s':dict(quantity=1000,sellable_quantity=1000,cost_minor=1)})
            a=action('s',**changes);register(ledger,a,DAYS[1])
            self.assertEqual(gaps(ledger,[a],DAYS[2],'09:30:00+08:00',share_allocation_policy=FLOOR_POLICY)[0]['reason'],reason)

    def test_public_saved_floor_and_v2_v3_report_disclose_tail_without_source_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest,_,a=fixture(root,stock='0.2965786',allocation_policy=FLOOR_POLICY)
            wire,view,_=execute(root/'out',manifest)
            self.assertEqual(wire['status'],'COMPLETE')
            self.assertEqual(view.profile['contract_version'],'stock_daily_open_profile_v4')
            state=wire['final_account']['equity_entitlements'][a['economic_event_id']]
            self.assertEqual(state['record_quantity'],16600)
            self.assertEqual(state['share_allocation']['allocated_grant'],4923)
            self.assertEqual(Decimal(state['share_allocation']['fractional_remainder']),Decimal('0.20476'))
            self.assertTrue(all(type(p['quantity']) is int and type(p['sellable_quantity']) is int
                and type(p['pending_share_quantity']) is int for p in view.rows['positions']))
            with patch.object(legacy,'DAYS',manifest['scope']['calendar']): benchmark=legacy.benchmark()
            with patch.object(StockInputSource,'audit',side_effect=AssertionError('source reopened')):
                base=evaluate_backtest(view,benchmark=benchmark,spec=long_history_evaluation_spec(),
                    dividend_scope=stock_dividend_scope(view))
                report=evaluate_saved_analysis(view,base,benchmarks=dict(CSI300=benchmark,SSE_COMPOSITE=None,NASDAQ100=None),
                    spec=analysis_evaluation_spec(risk_free=RF))
            summary=base.to_dict()['episode_metrics']['share_allocation_summary']
            self.assertEqual(summary['rounded_event_count'],1)
            self.assertEqual(summary['by_security'][a['security_id']]['maximum_additional_shares'],1)
            self.assertEqual(Decimal(summary['by_security'][a['security_id']]['theoretical_fraction_sum']),Decimal('0.20476'))
            self.assertEqual(summary,report.to_dict()['episode_metrics']['share_allocation_summary'])
            for name,value in (('v2',base),('v3',report)):
                target=root/(name+'.json');save_backtest_evaluation(value,target)
                self.assertEqual(load_backtest_evaluation(target).payload,value.payload)
            self.assertEqual(_contract_unit(dict(backtest_contract_version='backtest_run_v8',
                profile_contract_version='stock_daily_open_profile_v4',quantity_unit='shares',price_unit='CNY/share')),'CNY/share')

    def test_rehashed_saved_allocation_cannot_create_a_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,a=fixture(root,stock='0.2965786',allocation_policy=FLOOR_POLICY)
            wire,_,_=execute(root/'out',manifest)
            wire['final_account']['equity_entitlements'][a['economic_event_id']]['share_allocation']['allocated_grant']+=1
            target=root/'tampered.json';reseal_run(target,wire)
            with self.assertRaisesRegex(ContractError,'saved final equity account mismatch'):
                load_stock_backtest_projection(target,artifact_reader=lambda r:Path(r['manifest_uri']),limits=LIMITS)

    def test_strict_factory_is_unchanged_and_zero_tail_economic_rows_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results=[]
            for name,policy in (('strict','reject_non_integer'),('floor',FLOOR_POLICY)):
                path=root/name;path.mkdir()
                manifest,_,a=fixture(path,stock='0.1',allocation_policy=policy)
                wire,view,_=execute(path/'out',manifest);results.append((wire,view))
                if name=='strict':
                    self.assertEqual(view.profile['equity_simulation'],SIMULATION)
                    self.assertEqual(view.profile['contract_version'],'stock_daily_open_profile_v3')
            (old,ov),(new,nv)=results
            for kind in ('nav','positions','fills','cash_ledger','position_ledger','orders','decisions'):
                actual=map_run_ids(deepcopy(nv.rows[kind]),new['run_id'],old['run_id'])
                if kind=='position_ledger':
                    for row in actual:row.pop('share_allocation',None)
                self.assertEqual(actual,ov.rows[kind],kind)
            self.assertEqual(new['metrics'],old['metrics'])


if __name__=='__main__':unittest.main()
