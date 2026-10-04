from copy import deepcopy
from decimal import Decimal, getcontext, setcontext, ROUND_DOWN, Inexact
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import BacktestRequest, BacktestRun, run_backtest, save_backtest_run, load_backtest_run
from axiom_engine.runtime.accounting import AccountLedger
from axiom_engine.runtime.episode_evaluation import evaluate_episodes
from axiom_engine.runtime.unit_splits import EVENT_FIELDS, UNIT_SPLIT_POLICY
from test_backtest import etf_fixture, DAYS, REF


def bind_event(wire, event):
    available = '2026-05-27T01:30:00Z'
    batch = {'records': [deepcopy(event)], 'field_meta': {name: {'by_key': [
        {'security_id': event['security_id'], 'event_id': event['event_id'],
         'revision_id': event['revision_id'], 'usable_from': available,
         'status': 'value' if event[name] is not None else 'source_missing'}]} for name in EVENT_FIELDS},
        'context': {'contract_version': 'data_batch_v1', 'domain': 'fund_share_conversions',
                    'snapshot_id': 's_synthetic', 'query': {'purpose': 'market_replay',
                    'pit_policy': 'best_effort_vendor_v1', 'time_field': 'effective_date',
                    'cutoff': event['record_date'] + 'T12:30:00Z'}}}
    ref = Document.from_dict(batch).identity
    market = wire['market_replay']
    market['source_refs'] = [REF, ref]
    market['source_evidence'] = [{'reference': REF, 'context': {'snapshot_id': 's_synthetic', 'query': {'purpose':'market_replay'}}},
                                 {'reference': ref, 'batch': batch}]
    market['unit_splits'] = [{'event': deepcopy(event), 'available_at': available, 'source_refs': [ref]}]
    return wire


def split_fixture(n=5, d=1, rounding='not_stated', quantity=100):
    wire = etf_fixture()
    security = wire['signal_frame']['universe'][0]
    wire.update(contract_version='backtest_request_v2', unit_split_policy=UNIT_SPLIT_POLICY, end_session=DAYS[4])
    wire['market_replay']['contract_version'] = 'market_replay_v2'
    wire['initial_account']['positions'] = {} if not quantity else {security: {
        'quantity': quantity, 'sellable_quantity': quantity, 'cost_minor': 100000}}
    for signal in wire['signal_frame']['rows']:
        if signal['session'] == DAYS[0]:
            signal.update(score=None, valid=False, invalid_reason='synthetic warmup gap')
    for row in wire['market_replay']['rows']:
        if row['security_id'] == security and row['session'] >= DAYS[4]:
            row['open'] = row['close'] = str(10 * d / n)
    event = dict.fromkeys(EVENT_FIELDS)
    event.update(security_id=security, event_id='issuer:synthetic:unit-split', event_type='unit_split',
        announcement_date='2026-05-26', announcement_precision='day', process_status='planned',
        record_date=DAYS[2], effective_date=DAYS[3], effective_phase='not_stated',
        new_price_basis_session=DAYS[4], new_price_basis_basis='issuer_announced_next_business_resume_after_conversion',
        ratio_numerator=n, ratio_denominator=d, quantity_rounding=rounding,
        quantity_rounding_scope=None if rounding == 'not_stated' else 'registered_holder_units',
        document_refs='["synthetic-plan"]', extraction_version='reviewed_fund_share_conversions_v1',
        revision_id='synthetic-r3', revision_sequence=3, first_observed_at='2026-10-04T00:00:00+00:00', raw_batch_id='synthetic-raw')
    return bind_event(wire, event)


def run(wire):
    return run_backtest(BacktestRequest.from_dict(wire)).to_dict()


class UnitSplitTests(unittest.TestCase):
    def test_five_for_one_preserves_cash_cost_nav_and_settled_sellability(self):
        result = run(split_fixture())
        application = result['unit_split_applications'][0]
        self.assertEqual(result['contract_version'], 'backtest_run_v2')
        self.assertEqual(result['runtime_version'], 'axiom.backtest/2')
        self.assertEqual((application['before_quantity'], application['after_quantity'], application['after_sellable_quantity']), (100, 500, 500))
        self.assertEqual(application['normalized_quote']['price'], '2')
        self.assertEqual(application['cost_minor'], 100000)
        self.assertEqual(application['rounding_value_minor'], 0)
        self.assertEqual(result['cash_ledger'], [])
        self.assertEqual(result['fills'], [])
        self.assertEqual({p['nav_minor'] for p in result['nav']}, {1100000})
        split_position = next(p for p in result['positions'] if p['session'] == DAYS[3])
        self.assertEqual(split_position['mark_basis_event_id'], application['event_id'])
        self.assertIsNone(result['positions'][-1]['mark_basis_event_id'])
        self.assertLess(application['sequence'], split_position['committed_sequence'])
        episodes, _ = evaluate_episodes(result, None, {'run': result['run_id']})
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]['final_quantity'], 500)

    def test_holder_total_ceiling_and_value_difference_no_cash_compensation(self):
        wire = split_fixture(114539, 100000, 'ceiling_to_whole_fund_unit')
        security = wire['signal_frame']['universe'][0]
        for row in wire['market_replay']['rows']:
            row['limit_up'] = '1000'
            if row['security_id'] == security:
                row['open'] = row['close'] = '114.539' if row['session'] < DAYS[4] else '100'
        application = run(wire)['unit_split_applications'][0]
        self.assertEqual(application['after_quantity'], 115)
        self.assertEqual(application['rounding_extra_fraction'], {'numerator': 46100, 'denominator': 100000})
        self.assertEqual(Decimal(application['normalized_quote']['price']), Decimal('100'))
        self.assertEqual(application['rounding_value_minor'], 4610)

    def test_zero_entitlement_normalizes_candidate_without_creating_position(self):
        result = run(split_fixture(quantity=0))
        application = result['unit_split_applications'][0]
        self.assertEqual(application['status'], 'NO_ENTITLEMENT')
        self.assertEqual(application['normalized_quote']['price'], '2')
        self.assertEqual(result['final_account']['positions'], {})
        self.assertEqual(result['position_ledger'][0]['quantity_delta'], 0)

    def test_core_next_week_receives_bridge_for_zero_held_candidate(self):
        wire = split_fixture(quantity=0)
        wire['end_session'] = DAYS[-1]
        event = deepcopy(wire['market_replay']['unit_splits'][0]['event'])
        event.update(record_date=DAYS[-2], effective_date=DAYS[-2], effective_phase='end_of_day', new_price_basis_session=DAYS[-1])
        bind_event(wire, event)
        security = event['security_id']
        for row in wire['signal_frame']['rows']:
            if row['session'] == DAYS[-2]:
                row['score'] = 0.1 if row['security_id'] == security else -0.1
        for row in wire['market_replay']['rows']:
            if row['security_id'] == security:
                row['open'] = row['close'] = '10' if row['session'] < DAYS[-1] else '2'
        result = run(wire)
        quote = result['decisions'][-1]['reference_prices'][security]
        self.assertEqual((quote['price'], quote['session'], quote['available_at']), ('2', DAYS[-2], DAYS[-2] + 'T12:30:00Z'))
        self.assertEqual(result['fills'][0]['price'], '2')
        self.assertEqual(result['fills'][0]['quantity'], 4900)

    def test_disclosed_full_session_halt_blocks_observed_etf_profile(self):
        wire = split_fixture()
        event = deepcopy(wire['market_replay']['unit_splits'][0]['event'])
        event.update(suspension_start=DAYS[-1], suspension_end=DAYS[-1], suspension_scope='full_session', resume_session='2026-06-09')
        bind_event(wire, event)
        wire['end_session'] = DAYS[-1]
        result = run(wire)
        self.assertTrue(result['orders'])
        self.assertTrue(all(o['reason'] == 'ANNOUNCED_SUSPENSION' for o in result['orders'] if o['security_id'] == event['security_id']))
        order = next(o for o in result['orders'] if o['security_id'] == event['security_id'])
        self.assertEqual(order['market_state'], 'unknown_status')
        self.assertEqual(order['announced_suspension_event_ids'], [event['event_id']])

    def test_atomic_duplicate_conflict_and_fractional_undisclosed_rounding(self):
        wire = split_fixture()
        item = wire['market_replay']['unit_splits'][0]
        security = item['event']['security_id']
        ledger = AccountLedger(cash_minor=1000, calendar=DAYS, settlement_sessions=1, positions=wire['initial_account']['positions'])
        registration = {'sequence': 0, 'position': deepcopy(ledger.positions[security])}
        quote = {'price': '10', 'session': DAYS[2], 'available_at': DAYS[2]+'T12:00:00Z', 'source_refs': [REF]}
        ledger.unit_split(item, registration, quote)
        snapshot = deepcopy(ledger.__dict__)
        self.assertIsNone(ledger.unit_split(item, registration, quote))
        changed = deepcopy(item); changed['event']['revision_id'] = 'later-result'
        with self.assertRaisesRegex(ContractError, 'conflicting'):
            ledger.unit_split(changed, registration, quote)
        self.assertEqual(snapshot, ledger.__dict__)
        with self.assertRaisesRegex(ContractError, 'fractional units'):
            run(split_fixture(3, 2, quantity=1))

    def test_unsettled_registration_and_changed_entitlement_reject(self):
        wire = split_fixture()
        for signal in wire['signal_frame']['rows']:
            if signal['session'] == DAYS[0]:
                signal.update(valid=True, invalid_reason=None, score=0.1 if signal['security_id'] == wire['signal_frame']['universe'][0] else -0.1)
        event = deepcopy(wire['market_replay']['unit_splits'][0]['event']); event['record_date'] = DAYS[1]
        bind_event(wire, event)
        with self.assertRaisesRegex(ContractError, 'fully settled'):
            run(wire)
        item = split_fixture()['market_replay']['unit_splits'][0]
        security = item['event']['security_id']
        ledger = AccountLedger(cash_minor=1, calendar=DAYS, settlement_sessions=1, positions={security: {'quantity': 100, 'sellable_quantity': 100, 'cost_minor': 1000}})
        reg = {'sequence': 0, 'position': deepcopy(ledger.positions[security])}
        ledger.positions[security]['quantity'] = ledger.positions[security]['sellable_quantity'] = 101
        before = deepcopy(ledger.__dict__)
        with self.assertRaisesRegex(ContractError, 'unchanged'):
            ledger.unit_split(item, reg, {'price':'10','session':DAYS[2],'available_at':DAYS[2]+'T12:00:00Z','source_refs':[REF]})
        self.assertEqual(before, ledger.__dict__)

    def test_future_revision_and_proof_mutation_cannot_unlock(self):
        wire = split_fixture()
        wire['market_replay']['unit_splits'][0]['available_at'] = DAYS[3]+'T12:30:00Z'
        with self.assertRaisesRegex(ContractError, 'unavailable at registration'):
            run(wire)

    def test_consolidation_unbound_quote_and_cash_record_collision_reject(self):
        with self.assertRaisesRegex(ContractError, 'consolidation'):
            run(split_fixture(1,2))
        wire = split_fixture()
        event = wire['market_replay']['unit_splits'][0]['event']
        for row in wire['market_replay']['rows']:
            if row['session']==event['effective_date'] and row['security_id']==event['security_id']:
                row['source_refs']=['sha256:'+'a'*64]
        with self.assertRaisesRegex(ContractError, 'unbound v2 market row'):
            run(wire)
        wire=split_fixture()
        wire['market_replay']['cash_dividends']=[{'event_id':'cash','security_id':event['security_id'],
            'record_session':event['effective_date'],'ex_session':DAYS[4],'pay_session':DAYS[4],
            'cash_per_unit':'1','source_refs':[REF]}]
        with self.assertRaisesRegex(ContractError, 'cash record and unit EOD collision'):
            run(wire)

    def test_other_security_pending_does_not_block_zero_entitlement(self):
        wire=split_fixture(quantity=0)
        event=deepcopy(wire['market_replay']['unit_splits'][0]['event'])
        event.update(record_date=DAYS[1],effective_date=DAYS[1],new_price_basis_session=DAYS[2])
        bind_event(wire,event)
        other=wire['signal_frame']['universe'][1]
        for signal in wire['signal_frame']['rows']:
            if signal['session']==DAYS[0]:
                signal.update(valid=True,invalid_reason=None,score=0.1 if signal['security_id']==other else -0.1)
        result=run(wire)
        self.assertEqual(result['fills'][0]['security_id'],other)
        self.assertEqual(result['unit_split_applications'][0]['status'],'NO_ENTITLEMENT')

    def test_partial_odd_lot_reduction_is_not_full_liquidation(self):
        wire=split_fixture(114539,100000,'ceiling_to_whole_fund_unit')
        wire['initial_account']['cash_minor']=0
        wire['end_session']=DAYS[-1]
        security=wire['signal_frame']['universe'][0]
        for row in wire['market_replay']['rows']:
            row['limit_up']='1000'
            if row['security_id']==security:row['open']=row['close']='114.539' if row['session']<DAYS[4] else '100'
        for signal in wire['signal_frame']['rows']:
            if signal['session']==DAYS[-2]:signal['score']=0.1 if signal['security_id']==security else -0.1
        result=run(wire)
        self.assertEqual(result['orders'][-1]['quantity'],15)
        self.assertEqual(result['orders'][-1]['filled_quantity'],0)
        self.assertEqual(result['final_account']['positions'][security]['quantity'],115)

    def test_odd_lot_full_exit_and_capacity_limited_tail(self):
        for volume, sold, tail in [('1000000', 115, 0), ('1000', 100, 15)]:
            with self.subTest(volume=volume):
                wire = split_fixture(114539, 100000, 'ceiling_to_whole_fund_unit')
                wire['end_session'] = DAYS[-1]
                security = wire['signal_frame']['universe'][0]
                for row in wire['market_replay']['rows']:
                    row['limit_up'] = '1000'
                    if row['security_id'] == security:
                        row['open'] = row['close'] = '114.539' if row['session'] < DAYS[4] else '100'
                        if row['session'] == DAYS[-1]:
                            row['volume_units'] = volume
                for signal in wire['signal_frame']['rows']:
                    if signal['session'] == DAYS[-2]:
                        signal['score'] = -0.1
                result = run(wire)
                self.assertEqual(result['fills'][0]['quantity'], sold)
                self.assertEqual(result['final_account']['positions'][security]['quantity'], tail)
                episodes, _ = evaluate_episodes(result, None, {'run': result['run_id']})
                self.assertEqual(len(episodes), 1)
                self.assertEqual(episodes[0]['final_quantity'], tail)
                self.assertEqual(episodes[0]['status'], 'OPEN' if tail else 'CLOSED')

    def test_buy_split_sell_is_one_closed_episode_with_original_cost(self):
        wire = split_fixture(quantity=0)
        wire['end_session'] = DAYS[-1]
        wire['profile']['commission_rate'] = '0'
        security = wire['signal_frame']['universe'][0]
        for signal in wire['signal_frame']['rows']:
            if signal['session'] == DAYS[0]:
                signal.update(valid=True, invalid_reason=None, score=0.1 if signal['security_id'] == security else -0.1)
            if signal['session'] == DAYS[-2]:
                signal['score'] = -0.1
        result = run(wire)
        episodes, metrics = evaluate_episodes(result, None, {'run': result['run_id']})
        self.assertEqual([(f['side'], f['quantity']) for f in result['fills']], [('BUY', 1000), ('SELL', 5000)])
        self.assertEqual(len(episodes), 1)
        self.assertEqual((episodes[0]['status'], episodes[0]['buy_cost_minor'], episodes[0]['net_pnl_minor']), ('CLOSED', 1000000, 0))
        self.assertEqual(metrics['eligible_closed_count'], 1)
        wire = split_fixture()
        wire['market_replay']['unit_splits'][0]['event']['ratio_numerator'] = 7
        with self.assertRaisesRegex(ContractError, 'differs from public selected'):
            run(wire)

    def test_save_load_repetition_decimal_context_and_v1_load(self):
        wire = split_fixture()
        original = getcontext().copy()
        try:
            first = run(wire)
            getcontext().prec = 7; getcontext().rounding = ROUND_DOWN; getcontext().traps[Inexact] = True
            self.assertEqual(first, run(wire))
        finally:
            setcontext(original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'v2.json'
            result = BacktestRun.from_dict(first)
            save_backtest_run(result, path)
            with patch('axiom_engine.runtime.backtest.run_backtest', side_effect=AssertionError('no account replay')):
                self.assertEqual(load_backtest_run(path), result)
            old = Path(__file__).parents[1]/'.artifacts/etf-daily-v1/etf_rotation.daily.backtest.json'
            if old.exists():
                self.assertEqual(load_backtest_run(old).to_dict()['contract_version'], 'backtest_run_v1')

    def test_public_saved_only_v2_evaluation_keeps_contract_and_input_ref(self):
        from axiom_engine.runtime import evaluate_backtest, daily_evaluation_spec, long_history_evaluation_spec
        from test_evaluation import benchmark
        result = BacktestRun.from_dict(run(split_fixture()))
        with patch('axiom_engine.runtime.backtest.run_backtest', side_effect=AssertionError('no replay')):
            for spec, version in [(daily_evaluation_spec(), 'evaluation_report_v1'),
                                  (long_history_evaluation_spec(), 'evaluation_report_v2')]:
                report = evaluate_backtest(result, spec=spec, benchmark=benchmark(result)).to_dict()
                self.assertEqual(report['contract_version'], version)
                self.assertEqual(report['input_run_ref']['content_digest'], result.to_dict()['content_digest'])
                self.assertEqual(report['episodes'][0]['final_quantity'], 500)


if __name__ == '__main__':
    unittest.main()
