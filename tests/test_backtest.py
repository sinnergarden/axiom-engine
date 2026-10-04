from copy import deepcopy
from decimal import getcontext, setcontext, Inexact, ROUND_DOWN
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, SignalFrame, plan_rotation
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, run_backtest, save_backtest_run,
                                  load_backtest_run)
from axiom_engine.runtime.accounting import AccountLedger

REF = 'sha256:' + '1' * 64
DAYS = ['2026-05-29', '2026-06-01', '2026-06-02', '2026-06-03',
        '2026-06-04', '2026-06-05', '2026-06-08']


def fixture():
    signals, market = [], []
    for day in DAYS:
        for security in ('A', 'B'):
            signals.append({'security_id': security, 'session': day,
                'knowledge_cutoff': day + 'T12:30:00Z', 'available_at': day + 'T12:00:00Z',
                'score': 0.1 if security == ('A' if day < '2026-06-05' else 'B') else -0.1,
                'valid': True, 'invalid_reason': None, 'source_refs': [REF]})
            market.append({'security_id': security, 'session': day,
                'open': '10' if security == 'A' else '20',
                'close': '10' if security == 'A' else '20', 'volume_units': '1000000',
                'limit_up': '100', 'limit_down': '1',
                'market_state': 'normal_trading', 'state_reason': None,
                'close_available_at': day + 'T12:00:00Z', 'source_refs': [REF]})
    return {'contract_version': 'backtest_request_v1', 'account_id': 'synthetic-account',
        'start_session': DAYS[1], 'end_session': DAYS[-1],
        'signal_frame': {'contract_version': 'signal_frame_v1', 'signal_run_ref': REF,
            'signal_stage': 'final', 'score_semantics': 'momentum_20d', 'universe': ['A', 'B'], 'rows': signals},
        'market_replay': {'contract_version': 'market_replay_v1', 'price_basis': 'unadjusted',
            'calendar': DAYS, 'universe': ['A', 'B'], 'rows': market, 'cash_dividends': [],
            'source_refs': [REF], 'source_evidence': [], 'limitations': ['Synthetic prices; not historical returns.']},
        'initial_account': {'cash_minor': 1000000, 'positions': {}},
        'profile': {'contract_version': 'daily_open_profile_v1', 'lot_size': 100,
            'settlement_sessions': 1, 'commission_rate': '0.0003', 'minimum_commission_minor': 0,
            'tax_rate': '0', 'slippage_bps': '0', 'participation_rate': '0.1',
            'decision_time_utc': '00:55:00Z', 'execution': 'open', 'approximation': 'daily_volume_proxy',
            'unknown_status_policy': 'block',
            'limitation': 'Synthetic conservative T+1, 100-unit lot; not real ETF rules.'}}


def run(wire):
    return run_backtest(BacktestRequest.from_dict(wire)).to_dict()


class AccountingGoldenTests(unittest.TestCase):
    def test_design_golden_cash_positions_t_plus_one_and_duplicate(self):
        ledger = AccountLedger(cash_minor=1000000, calendar=DAYS, settlement_sessions=1)
        buy = {'fill_id': 'buy', 'order_id': 'o1', 'session': DAYS[1], 'security_id': 'A',
               'side': 'BUY', 'quantity': 100, 'price': '10', 'gross_minor': 100000,
               'fee_minor': 500, 'cash_delta_minor': -100500}
        ledger.apply_fill(buy)
        self.assertEqual((ledger.cash, ledger.positions['A']),
                         (899500, {'quantity': 100, 'sellable_quantity': 0, 'cost_minor': 100500}))
        self.assertEqual(ledger.cash + 100 * 10 * 100, 999500)
        self.assertFalse(ledger.apply_fill(buy))
        self.assertEqual(ledger.sequence, 1)
        sell = {'fill_id': 'sell', 'order_id': 'o2', 'session': DAYS[1], 'security_id': 'A',
                'side': 'SELL', 'quantity': 40, 'price': '11', 'gross_minor': 44000,
                'fee_minor': 500, 'cash_delta_minor': 43500}
        with self.assertRaisesRegex(ContractError, r'T\+1'):
            ledger.apply_fill(sell)
        ledger.advance(DAYS[2])
        with self.assertRaisesRegex(ContractError, 'account clock'):
            ledger.apply_fill(sell)
        sell['session'] = DAYS[2]
        ledger.apply_fill(sell)
        self.assertEqual((ledger.cash, ledger.positions['A']['quantity']), (943000, 60))
        self.assertEqual(ledger.cash + 60 * 11 * 100, 1009000)
        bad = {**buy, 'fee_minor': 600}
        with self.assertRaisesRegex(ContractError, 'conflicting'):
            ledger.apply_fill(bad)

    def test_dividend_receivable_then_cash_not_double_income(self):
        ledger = AccountLedger(cash_minor=0, calendar=DAYS, settlement_sessions=1,
            positions={'A': {'quantity': 100, 'sellable_quantity': 100, 'cost_minor': 100000}})
        action = {'event_id': 'd', 'ex_session': DAYS[2], 'pay_session': DAYS[3], 'cash_per_unit': '1'}
        ledger.dividend(action, 'EX', 100)
        self.assertEqual((ledger.cash, sum(ledger.receivables.values())), (0, 10000))
        ledger.dividend(action, 'EX', 100)
        self.assertEqual(ledger.sequence, 1)
        ledger.dividend(action, 'PAY', 100)
        self.assertEqual((ledger.cash, sum(ledger.receivables.values())), (10000, 0))
        ledger.dividend(action, 'PAY', 100)
        self.assertEqual(ledger.sequence, 2)


class BacktestTests(unittest.TestCase):
    def test_weekly_previous_signal_rounding_fees_and_reconciliation(self):
        result = run(fixture())
        self.assertEqual([d['feature_session'] for d in result['decisions']], [DAYS[0], DAYS[-2]])
        self.assertEqual([(f['side'], f['security_id'], f['quantity']) for f in result['fills']],
                         [('BUY', 'A', 900), ('SELL', 'A', 900), ('BUY', 'B', 400)])
        self.assertEqual(result['metrics']['total_fees_minor'], 780)
        self.assertEqual(result['nav'][-1]['nav_minor'], 999220)
        self.assertEqual(result['final_account']['cash_minor'], 199220)
        self.assertEqual(result['nav'][0]['nav_minor'], 999730)
        self.assertEqual(result['positions'][0]['sellable_quantity'], 0)
        self.assertEqual(result['positions'][1]['sellable_quantity'], 900)
        for point in result['nav']:
            held = [p for p in result['positions'] if p['session'] == point['session']]
            self.assertTrue(all(p['committed_sequence'] == point['committed_sequence'] for p in held))
            self.assertEqual(point['market_value_minor'], sum(p['market_value_minor'] for p in held))
            self.assertEqual(point['nav_minor'], point['cash_minor'] + point['market_value_minor'] + point['receivable_minor'])
        self.assertEqual(result['final_account']['cash_minor'], 1000000 + sum(e['cash_delta_minor'] for e in result['cash_ledger']))
        self.assertEqual(result['nav'][-1]['committed_sequence'], result['committed_sequence'])

    def test_sell_blocked_does_not_manufacture_cash_for_buy(self):
        wire = fixture()
        for row in wire['market_replay']['rows']:
            if row['session'] == DAYS[-1] and row['security_id'] == 'A':
                row['limit_down'] = '10'
        result = run(wire)
        self.assertEqual(result['orders'][1]['reason'], 'PRICE_LIMIT')
        self.assertEqual(result['orders'][2]['reason'], 'INSUFFICIENT_CASH')
        self.assertEqual(len(result['fills']), 1)
        self.assertEqual(result['final_account']['positions']['A']['quantity'], 900)

    def test_unknown_state_blocks_and_cannot_opt_into_trading(self):
        wire = fixture()
        for row in wire['market_replay']['rows']:
            row.update(market_state='unknown_status', state_reason='status_source_missing')
        blocked = run(wire)
        self.assertEqual(blocked['fills'], [])
        self.assertTrue(all(o['reason'] == 'UNKNOWN_MARKET_STATUS' for o in blocked['orders']))
        wire['profile']['unknown_status_policy'] = 'observed_daily_proxy'
        with self.assertRaisesRegex(ContractError, 'must block'):
            run(wire)

    def test_trade_day_cutoff_rejected_and_sell_fees_reject_without_failure(self):
        wire = fixture()
        wire['signal_frame']['rows'][0].update(knowledge_cutoff=DAYS[1] + 'T00:50:00Z',
                                             available_at=DAYS[1] + 'T00:45:00Z')
        with self.assertRaisesRegex(ContractError, 'feature session'):
            run(wire)
        wire = fixture()
        wire['initial_account'] = {'cash_minor': 0, 'positions': {
            'A': {'quantity': 1, 'sellable_quantity': 1, 'cost_minor': 1000}}}
        wire['profile']['minimum_commission_minor'] = 1500
        for row in wire['signal_frame']['rows']:
            row['score'] = -1
        result = run(wire)
        self.assertEqual(result['orders'][0]['reason'], 'INSUFFICIENT_CASH_FOR_FEES')
        self.assertEqual(result['fills'], [])

    def test_contradictory_price_limits_and_impossible_prices_rejected(self):
        wire = fixture()
        wire['market_replay']['rows'][2]['open'] = '0.5'
        with self.assertRaisesRegex(ContractError, 'outside declared limits'):
            run(wire)
        wire = fixture()
        wire['market_replay']['rows'][2]['limit_down'] = '101'
        with self.assertRaisesRegex(ContractError, 'contradictory'):
            run(wire)

    def test_volume_partial_fill_single_minimum_fee_and_slippage_not_double_charged(self):
        wire = fixture()
        wire['end_session'] = DAYS[1]
        wire['profile'].update(minimum_commission_minor=500, slippage_bps='100')
        for row in wire['market_replay']['rows']:
            if row['session'] == DAYS[1] and row['security_id'] == 'A':
                row['volume_units'] = '1500'
        result = run(wire)
        fill = result['fills'][0]
        self.assertEqual((fill['quantity'], fill['gross_minor'], fill['fee_minor'], fill['slippage_minor']), (100, 101000, 500, 1000))
        self.assertEqual(result['final_account']['cash_minor'], 898500)
        self.assertEqual(result['orders'][0]['status'], 'PARTIAL_EXPIRED')

    def test_no_positive_cash_and_invalid_signal_keeps_positions(self):
        wire = fixture()
        for row in wire['signal_frame']['rows']:
            if row['session'] == DAYS[-2]:
                row['score'] = -1
        result = run(wire)
        self.assertEqual(result['decisions'][-1]['selected_security_id'], None)
        self.assertEqual(result['final_account']['positions']['A']['quantity'], 0)
        row = wire['signal_frame']['rows'][-4]
        row.update(score=None, valid=False, invalid_reason='MISSING_CLOSE')
        result = run(wire)
        self.assertEqual(result['decisions'][-1]['status'], 'NO_DECISION')
        self.assertEqual(result['final_account']['positions']['A']['quantity'], 900)

    def test_missing_signals_future_signal_and_future_quote_rejected(self):
        wire = fixture()
        wire['signal_frame']['rows'].pop(0)
        with self.assertRaisesRegex(ContractError, 'coverage|missing'):
            run(wire)
        wire = fixture()
        wire['signal_frame']['rows'][0]['available_at'] = DAYS[1] + 'T02:00:00Z'
        with self.assertRaisesRegex(ContractError, 'cutoff'):
            run(wire)
        wire = fixture()
        wire['market_replay']['rows'][0]['close_available_at'] = DAYS[1] + 'T02:00:00Z'
        with self.assertRaisesRegex(ContractError, 'future reference'):
            run(wire)

    def test_tie_break_is_canonical_id_and_core_never_receives_open_volume(self):
        wire = fixture()
        for row in wire['signal_frame']['rows']:
            row['score'] = 0.1
        with patch('axiom_engine.runtime.backtest.plan_rotation', wraps=plan_rotation) as planner:
            result = run(wire)
        self.assertTrue(all(d['selected_security_id'] == 'A' for d in result['decisions']))
        for call in planner.call_args_list:
            context = call.kwargs['context']
            self.assertNotIn('open', json.dumps(context))
            self.assertNotIn('volume', json.dumps(context))

    def test_cash_dividend_ex_and_pay_and_stale_valuation(self):
        wire = fixture()
        wire['end_session'] = DAYS[4]
        wire['profile']['commission_rate'] = '0'
        wire['market_replay']['cash_dividends'] = [{'event_id': 'd', 'security_id': 'A',
            'record_session': DAYS[1], 'ex_session': DAYS[2], 'pay_session': DAYS[3],
            'cash_per_unit': '1', 'source_refs': [REF]}]
        for row in wire['market_replay']['rows']:
            if row['security_id'] == 'A' and row['session'] >= DAYS[2]:
                row['open'] = row['close'] = '9'
            if row['security_id'] == 'A' and row['session'] == DAYS[4]:
                row['close'] = None
        result = run(wire)
        self.assertEqual([p['nav_minor'] for p in result['nav']], [1000000] * 4)
        self.assertEqual([p['receivable_minor'] for p in result['nav']], [0, 100000, 0, 0])
        self.assertTrue(result['positions'][-1]['is_stale'])
        self.assertEqual(result['positions'][-1]['stale_sessions'], 1)

    def test_input_mutation_replay_identity_decimal_context_and_offline(self):
        wire = fixture()
        request = BacktestRequest.from_dict(wire)
        wire['profile']['commission_rate'] = '1'
        original_context = getcontext().copy()
        try:
            getcontext().prec = 7
            getcontext().rounding = ROUND_DOWN
            getcontext().traps[Inexact] = True
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                first = run_backtest(request)
            setcontext(original_context.copy())
            second = run_backtest(request)
        finally:
            setcontext(original_context)
        self.assertEqual(first.payload, second.payload)
        changed = request.to_dict()
        changed['profile']['commission_rate'] = '0.001'
        self.assertNotEqual(first.to_dict()['run_id'], run(changed)['run_id'])

    def test_save_load_integrity_without_recompute_and_conflict(self):
        result = run_backtest(BacktestRequest.from_dict(fixture()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.json'
            save_backtest_run(result, path)
            save_backtest_run(result, path)
            with patch('axiom_engine.runtime.backtest.run_backtest', side_effect=AssertionError('no computation')):
                self.assertEqual(load_backtest_run(path).payload, result.payload)
            bad = json.loads(path.read_text())
            bad['nav'][-1]['nav_minor'] += 1
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ContractError, 'digest'):
                load_backtest_run(path)
            with self.assertRaisesRegex(ContractError, 'conflicting'):
                save_backtest_run(result, path)


if __name__ == '__main__':
    unittest.main()
