"""Small synthetic member-carry regressions; no Data, fit or long replay."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from axiom_engine.core import (ContractError, StockPredictionFrame, initial_strategy_state,
                               plan_stock_portfolio, stock_strategy_spec)
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_portfolio import _plan_admitted_stock_portfolio
from axiom_engine.core.stock_rules import validate_execution_rules
from axiom_engine.runtime import StockInputSource, run_backtest, stock_portfolio_policy
from axiom_engine.runtime.stock_inputs import validate_stock_request
from axiom_engine.runtime.stock_stream_contracts import read_budget
from axiom_engine._implementation import IMPLEMENTATION_REF
from test_csi300 import BOARD_IDS, DAYS, core_context, frame_for
from test_csi300_runtime import full_request
from test_stock_native_lifecycle import event_rules, terminal
from test_stock_stream_inputs import LIMITS, source_fixture
from test_stock_stream import execute
from test_stock_equity import fixture as equity_fixture
from test_stock_strategy import overlay


RETIRED = BOARD_IDS['SSE_MAIN']


def carry(batches, membership):
    terminal(batches, membership)
    for row in membership['records']:
        if row['security_id'] == RETIRED:
            row['is_member'] = True


def carry_frame(frame):
    for row in frame['rows']:
        if row['security_id'] == RETIRED:
            row.update(member=True, valid=True, score=100, invalid_reason=None)


def context_for(rules, k, feature=DAYS[2], trade=DAYS[3]):
    context = core_context(rules, k)
    context.update(feature_session=feature, trade_session=trade,
                   knowledge_cutoff=feature+'T20:30:00+08:00', decision_time=trade+'T08:55:00+08:00')
    for quote in context['reference_prices'].values():
        quote.update(session=feature, available_at=feature+'T12:00:00Z')
    context['reference_prices'].pop(RETIRED)
    return context


class MemberLifecycleTests(unittest.TestCase):
    def test_carried_member_and_null_delisted_facts_admit_exactly_in_both_adapters(self):
        plan = full_request(rules=event_rules(), k=1, mutate_native=carry, mutate_frame=carry_frame).to_dict()
        before = deepcopy(plan)
        with patch('axiom_engine.runtime.backtest.AccountLedger', side_effect=AssertionError('account started')):
            eager = validate_stock_request(plan)[5]
            row = eager[DAYS[2], RETIRED]
            self.assertEqual((row['_stock_member'], row['_stock_listed'], row['market_state']), (True, False, 'delisted'))
            self.assertEqual([row[name] for name in ('open', 'close', 'volume_shares')], [None, None, None])
            self.assertEqual(row['_stock_close_missing_reason'], 'source_gap')
            for size in (1, 2):
                with self.subTest(block_sessions=size), TemporaryDirectory() as tmp:
                    manifest = source_fixture(Path(tmp), plan)[0]
                    source = StockInputSource()
                    try:
                        audit = source.audit(manifest, block_sessions=size, read_budget=read_budget(LIMITS),
                                             limits=LIMITS, implementation_ref=IMPLEMENTATION_REF)
                        streamed = {}
                        for block in source.iter_blocks(manifest, block_sessions=size, read_budget=read_budget(LIMITS)):
                            streamed.update(deepcopy(block.market_rows))
                        self.assertEqual(streamed, eager)
                        self.assertEqual(audit.receipt['counts']['market_rows'], 16)
                    finally:
                        source._close_private_views()
        self.assertEqual(plan, before)
        def observed_after_end(batches, membership):
            carry(batches, membership)
            for native in batches[1]['records']:
                if (native['session'], native['security_id']) == (DAYS[2], RETIRED):
                    native['open'] = 10.0
            for meta in batches[1]['field_meta']['open']['by_key']:
                if (meta['session'], meta['security_id']) == (DAYS[2], RETIRED):
                    meta.update(missing_reason=None, usable_from=DAYS[2]+'T12:00:00Z')
        with self.assertRaisesRegex(ContractError, 'unlisted stock carries observed market facts'):
            full_request(rules=event_rules(), k=1, mutate_native=observed_after_end, mutate_frame=carry_frame)

    def test_retired_nonheld_member_without_price_is_excluded_before_top_k_and_sizing(self):
        rules = event_rules(); wire = frame_for(rules, scores={RETIRED:100})
        frame = StockPredictionFrame.from_dict(wire)
        account = dict(cash_minor=50000000, positions={}, version=0)
        before = deepcopy(wire)
        for k in (1, 4):
            with self.subTest(k=k):
                context = context_for(rules, k)
                result = plan_stock_portfolio(frame, account=account, context=context, top_k=k).to_dict()
                private = _plan_admitted_stock_portfolio(wire,
                    {(r['session'], r['security_id']):r for r in wire['rows']}, account=account,
                    context=context, top_k=k, rules_index=validate_execution_rules(rules)).to_dict()
                self.assertEqual(result, private)
                excluded = result['trace'][0]['excluded_members'][0]
                self.assertEqual((excluded['security_id'], excluded['feature_lifecycle'], excluded['trade_lifecycle']),
                                 (RETIRED, 'delisted', 'delisted'))
                identity = next(r for r in rules['identity_input']['rows'] if r['security_id']==RETIRED)
                self.assertEqual(excluded['source_refs'], identity['source_refs'])
                self.assertEqual(result['trace'][0]['pit_member_count'], 4)
                self.assertNotIn(RETIRED, result['selected_security_ids'])
                self.assertFalse(any(i['security_id']==RETIRED for i in result['intents']))
                if k == 4:
                    self.assertEqual(result['status'], 'NO_DECISION')
                    self.assertEqual(result['trace'][-1], dict(reason='INSUFFICIENT_ELIGIBLE_MEMBERS', count=3, top_k=4))
                else:
                    self.assertEqual(result['status'], 'DECISION_COMPLETE')
        self.assertEqual(wire, before)

    def test_listing_end_on_intended_trade_day_excludes_prior_active_member(self):
        rules = event_rules(); frame = StockPredictionFrame.from_dict(frame_for(rules, scores={RETIRED:100}))
        result = plan_stock_portfolio(frame, account=dict(cash_minor=50000000, positions={}, version=0),
            context=context_for(rules, 1, DAYS[1], DAYS[2]), top_k=1).to_dict()
        self.assertEqual((result['trace'][0]['excluded_members'][0]['feature_lifecycle'],
                          result['trace'][0]['excluded_members'][0]['trade_lifecycle']), ('active', 'delisted'))
        self.assertNotIn(RETIRED, result['selected_security_ids'])

    def test_configured_strategies_use_separate_mask_with_original_candidates(self):
        rules = event_rules(); wire = frame_for(rules, scores={RETIRED:100})
        for spec in (stock_strategy_spec(top_k=1),
                     stock_strategy_spec(selection='rank_buffer', top_k=1, rank_buffer={'keep_rank':2}),
                     stock_strategy_spec(top_k=1, exit_overlay=overlay())):
            context = context_for(rules, 1)
            context.update(portfolio_policy=stock_portfolio_policy(strategy_spec=spec, universe_id='csi300',
                execution_universe=rules['universe'], execution_rules=rules), strategy_state=initial_strategy_state(),
                pending_share_quantities={}, reserved_security_ids=[], risk_observations={})
            with self.subTest(spec=spec), patch(
                    'axiom_engine.core.stock_strategy.ConfiguredStockStrategy.plan', autospec=True,
                    wraps=None) as observed:
                from axiom_engine.core.stock_strategy import ConfiguredStockStrategy
                observed.side_effect = lambda instance, **kwargs: instance._plan(kwargs['inputs'], kwargs['state'])
                result = plan_stock_portfolio(StockPredictionFrame.from_dict(wire),
                    account=dict(cash_minor=50000000, positions={}, version=0), context=context, top_k=1).to_dict()
                inputs = observed.call_args.kwargs['inputs']
                original = next(r for r in inputs['candidates'] if r['security_id']==RETIRED)
                self.assertEqual((original['member'], original['valid'], original['score']), (True, True, 100))
                self.assertNotIn(RETIRED, inputs['constraints']['eligible_security_ids'])
                self.assertNotIn(RETIRED, result['selected_security_ids'])

    def test_unknown_identity_lifecycle_is_rejected_without_false_or_active_fallback(self):
        rules = event_rules(); frame = StockPredictionFrame.from_dict(frame_for(rules))
        damaged = deepcopy(rules); damaged['identity_input']['rows'][0]['listing_date'] = None
        damaged['identity_input_ref'] = Document.from_dict(damaged['identity_input']).identity
        context = context_for(rules, 1)
        context.update(stock_execution_rules=damaged, stock_execution_rules_ref=Document.from_dict(damaged).identity)
        with self.assertRaises(ContractError):
            plan_stock_portfolio(frame, account=dict(cash_minor=50000000, positions={}, version=0), context=context, top_k=1)

    def test_retired_holding_still_stops_before_nav_and_is_not_discarded(self):
        plan = full_request(rules=event_rules(), k=1, mutate_native=carry, mutate_frame=carry_frame)
        eager = run_backtest(plan).to_dict()
        self.assertEqual(eager['stopped']['reason'], 'HELD_MISSING_STOCK_LIFECYCLE_CAPABILITY')
        self.assertEqual(eager['stopped']['session'], DAYS[2])
        self.assertGreater(eager['final_account']['positions'][RETIRED]['quantity'], 0)
        self.assertEqual(len(eager['nav']), 1)
        with TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = source_fixture(root, plan.to_dict())[0]
            wire, view, _ = execute(root/'out', manifest, block_sessions=1)
            self.assertEqual(wire['stopped']['reason'], eager['stopped']['reason'])
            self.assertEqual(wire['final_account'], eager['final_account'])
            self.assertEqual(len(view.rows['nav']), 1)

    def test_carried_retired_holding_missing_equity_keeps_equity_stop(self):
        plan = full_request(rules=event_rules(), k=1, mutate_native=carry, mutate_frame=carry_frame).to_dict()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            def missing_equity(a):
                a.update(security_id=RETIRED, stock_listing_date=None)
                for alias in a['aliases']:
                    alias['security_id'] = RETIRED
            manifest, _, _ = equity_fixture(root, plan=plan, stock=None, change=missing_equity)
            wire, view, _ = execute(root/'out', manifest, block_sessions=1)
            self.assertEqual(wire['stopped']['reason'], 'HELD_EQUITY_FACT_GAP')
            self.assertEqual(wire['stopped']['session'], DAYS[2])
            self.assertGreater(wire['final_account']['positions'][RETIRED]['quantity'], 0)
            self.assertEqual(len(view.rows['nav']), 1)


if __name__ == '__main__':
    unittest.main()
