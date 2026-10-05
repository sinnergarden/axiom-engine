"""Hand-checkable P10 evidence; scenarios use the existing owner ledger."""
from copy import deepcopy
from decimal import Context, Decimal, Inexact, ROUND_DOWN, getcontext, setcontext, localcontext
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRun, BenchmarkSeries, DividendScope,
    daily_evaluation_spec, evaluate_backtest, load_backtest_evaluation,
    save_backtest_evaluation, read_csi300_benchmark, read_dividend_scope)
from axiom_engine.runtime.accounting import AccountLedger
from test_backtest import DAYS, REF, fixture


def trade(day, side, security, quantity, price, fee=0):
    return dict(session=day, side=side, security_id=security, quantity=quantity, price=price, fee_minor=fee)


def native_action(a):
    return dict(security_id=a['security_id'], announcement_date=a['record_session'], process_status='实施',
        record_date=a['record_session'], ex_date=a['ex_session'], pay_date=a['pay_session'],
        cash_dividend_per_unit=float(a['cash_per_unit']))


def saved(trades=(), *, actions=(), days=DAYS, initial=None, start=None):
    """Create synthetic saved observations with the same existing AccountLedger."""
    plan = fixture()
    actions = [dict(a, event_id=Document.from_dict(native_action(a)).identity,
                    cash_per_unit=str(native_action(a)['cash_dividend_per_unit'])) for a in actions]
    plan.update(start_session=start or days[1], end_session=days[-1])
    plan['market_replay'].update(calendar=days, cash_dividends=list(actions))
    plan['market_replay']['source_evidence'] = [dict(reference=REF, context=dict(snapshot_id='s_synthetic', query=dict(purpose='market_replay')))]
    plan['initial_account']['positions'] = initial or {}
    plan['profile']['settlement_sessions'] = 0
    ledger = AccountLedger(cash_minor=1000000, calendar=days, settlement_sessions=0, positions=initial)
    initial_nav = 1000000 + sum(p['quantity'] * (1000 if s == 'A' else 2000) for s, p in (initial or {}).items())
    entitlements, nav, positions = {}, [], []
    for day in days:
        if day < plan['start_session']:
            continue
        ledger.advance(day)
        for phase, key in [('EX', 'ex_session'), ('PAY', 'pay_session')]:
            for action in actions:
                if action[key] == day and action['event_id'] in entitlements:
                    ledger.dividend(action, phase, entitlements[action['event_id']])
        for raw in trades:
            if raw['session'] == day:
                f = {**raw, 'fill_id': 'f' + str(len(ledger.fills)), 'order_id': 'o' + str(len(ledger.fills))}
                f['gross_minor'] = int(Decimal(f['price']) * f['quantity'] * 100)
                f['cash_delta_minor'] = -f['gross_minor'] - f['fee_minor'] if f['side'] == 'BUY' else f['gross_minor'] - f['fee_minor']
                ledger.apply_fill(f)
        for action in actions:
            if action['record_session'] == day:
                entitlements[action['event_id']] = ledger.positions.get(action['security_id'], {}).get('quantity', 0)
        ledger.sequence += 1
        value = 0
        for security, position in ledger.positions.items():
            if position['quantity']:
                amount = position['quantity'] * (1000 if security == 'A' else 2000)
                value += amount
                positions.append(dict(session=day, security_id=security, **position, market_value_minor=amount,
                    committed_sequence=ledger.sequence))
        receivable = sum(ledger.receivables.values())
        wealth = ledger.cash + value + receivable
        nav.append(dict(session=day, cash_minor=ledger.cash, market_value_minor=value, receivable_minor=receivable,
            nav_minor=wealth, nav_index=str(Decimal(wealth) / initial_nav), committed_sequence=ledger.sequence))
    wire = dict(contract_version='backtest_run_v1', status='COMPLETE', account_id='synthetic-account', plan=plan,
        core_version='axiom.rotation/1', runtime_version='axiom.backtest/1.1', implementation_ref=IMPLEMENTATION_REF,
        signal_ref=REF, market_ref=Document.from_dict(plan['market_replay']).identity,
        profile_ref=Document.from_dict(plan['profile']).identity, committed_sequence=ledger.sequence,
        initial_nav_minor=initial_nav, final_account=dict(cash_minor=ledger.cash, receivable_minor=sum(ledger.receivables.values()),
            positions=ledger.positions, committed_sequence=ledger.sequence),
        fills=ledger.fills, cash_ledger=ledger.cash_ledger, position_ledger=ledger.position_ledger,
        nav=nav, positions=positions, limitations=['Synthetic event scenario, not historical returns.'])
    wire['run_id'] = Document.from_dict(dict(request=plan, core=wire['core_version'], runtime=wire['runtime_version'],
        implementation_ref=wire['implementation_ref'])).identity
    return rehash(wire)


def rehash(wire):
    wire = deepcopy(wire)
    wire.pop('content_digest', None)
    wire['content_digest'] = Document.from_dict(wire).identity
    return BacktestRun.from_dict(wire)


def benchmark(run):
    wire = run.to_dict()
    calendar = wire['plan']['market_replay']['calendar']
    anchor = calendar[calendar.index(wire['plan']['start_session']) - 1]
    days = [anchor, *[r['session'] for r in wire['nav']]]
    batch = dict(records=[dict(session=d, security_id='000300.SH', close=100+i) for i,d in enumerate(days)],
        field_meta=dict(close=dict(unit='index points', by_key=[dict(session=d, security_id='000300.SH',
            usable_from=d+'T12:00:00Z', missing_reason=None) for d in days])),
        context=dict(contract_version='data_batch_v1', domain='benchmark_daily', snapshot_id='s_synthetic', reader_version='synthetic_reader',
            query=dict(fields=['close'], symbols=['000300.SH'], sessions=days, price_basis='unadjusted', purpose='market_replay',
                pit_policy='best_effort_vendor_v1', cutoff_by_session={d:d+'T20:30:00+08:00' for d in days})))
    ref = Document.from_dict(batch).identity
    return BenchmarkSeries.from_dict(dict(contract_version='benchmark_series_v1', security_id='000300.SH',
        series_kind='price_index_excluding_dividends', unit='index points', calendar=days,
        rows=[dict(session=d, close=str(100 + i), available_at=d + 'T12:00:00Z', valid=True,
            missing_reason=None, source_refs=[ref]) for i, d in enumerate(days)],
        source_refs=[ref], source_evidence=[dict(reference=ref, batch=batch)], limitations=['Synthetic index prices.']))


def missing_benchmark(run, index):
    b = benchmark(run).to_dict()
    batch = b['source_evidence'][0]['batch']
    batch['records'][index]['close'] = None
    batch['field_meta']['close']['by_key'][index].update(usable_from=None, missing_reason='source_missing')
    ref = Document.from_dict(batch).identity
    b['source_refs'] = [ref]
    b['source_evidence'][0]['reference'] = ref
    for row in b['rows']:
        row['source_refs'] = [ref]
    b['rows'][index].update(close=None, available_at=None, valid=False, missing_reason='source_missing')
    return BenchmarkSeries.from_dict(b)


def scope(run, extra=()):
    w = run.to_dict()
    raw = [native_action(a) for a in [*w['plan']['market_replay']['cash_dividends'], *extra]]
    names = ['cash_dividend_per_unit','record_date','pay_date','ex_date']
    meta = [dict(security_id=r['security_id'], announcement_date=r['announcement_date'], process_status=r['process_status'],
        status='value', missing_reason=None, usable_from=r['record_date']+'T12:00:00Z') for r in raw]
    batch = dict(records=raw, field_meta={n:dict(unit='CNY/fund unit' if n=='cash_dividend_per_unit' else None, by_key=meta) for n in names},
        context=dict(contract_version='data_batch_v1', domain='corporate_actions', snapshot_id='s_synthetic', reader_version='synthetic_reader',
            logical_key=['security_id','announcement_date','process_status'], query=dict(fields=names, symbols=w['plan']['market_replay']['universe'],
                start=w['plan']['start_session'], end=w['plan']['end_session'], cutoff=w['plan']['end_session']+'T20:30:00+08:00',
                purpose='market_replay', pit_policy='best_effort_vendor_v1', time_field='record_date', filters={})))
    ref = Document.from_dict(batch).identity
    actions = [dict(event_id=Document.from_dict(r).identity,security_id=r['security_id'],record_session=r['record_date'],
        ex_session=r['ex_date'],pay_session=r['pay_date'],cash_per_unit=str(r['cash_dividend_per_unit']),
        available_at=r['record_date']+'T12:00:00Z',source_refs=[ref]) for r in raw]
    return DividendScope.from_dict(dict(contract_version='dividend_scope_v1', coverage='observed_records_only',
        start_session=w['plan']['start_session'], end_session=w['plan']['end_session'],
        knowledge_cutoff=w['plan']['end_session'] + 'T12:30:00Z', universe=w['plan']['market_replay']['universe'],
        actions=sorted(actions,key=lambda a:a['event_id']), source_refs=[ref],
        source_evidence=[dict(reference=ref,batch=batch)], limitations=['Synthetic known record-date scope.']))


def action(ex=DAYS[3], pay=DAYS[4]):
    return dict(event_id=REF, security_id='A', record_session=DAYS[1], ex_session=ex, pay_session=pay,
        cash_per_unit='1', source_refs=[REF])


def evaluate(run, *, bench=None, dividend_scope=None):
    return evaluate_backtest(run, benchmark=bench or benchmark(run), spec=daily_evaluation_spec(),
        dividend_scope=dividend_scope).to_dict()


class EvaluationTests(unittest.TestCase):
    def test_additions_reductions_fees_one_episode_and_denominator(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 100, '10', 500), trade(DAYS[2], 'BUY', 'A', 50, '12', 100),
            trade(DAYS[3], 'SELL', 'A', 40, '13', 100), trade(DAYS[4], 'BUY', 'A', 10, '11', 100),
            trade(DAYS[5], 'SELL', 'A', 120, '14', 200)])
        w = evaluate(run, dividend_scope=scope(run))
        e, = w['episodes']
        self.assertEqual((e['buy_cost_minor'], e['sell_proceeds_minor'], e['fees_minor'], e['net_pnl_minor']), (171700, 219700, 1000, 48000))
        self.assertEqual(len(e['fill_refs']), 5)
        with localcontext(Context(prec=40)):
            self.assertEqual(Decimal(e['net_return']), Decimal(48000) / Decimal(171700))
        self.assertEqual(w['series'][0]['drawdown'], '-0.0005')
        self.assertEqual(w['episode_metrics']['win_rate'], '1')
        self.assertEqual(w['pnl_distribution']['status'], 'INSUFFICIENT_SAMPLE')

    def test_record_ownership_survives_close_reentry_ex_pay(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 100, '10'), trade(DAYS[2], 'SELL', 'A', 100, '11'),
            trade(DAYS[2], 'BUY', 'A', 200, '10')], actions=[action()])
        before = run.payload
        with patch.object(AccountLedger, 'apply_fill', side_effect=AssertionError('account replay')):
            w = evaluate(run, dividend_scope=scope(run))
        first, second = w['episodes']
        self.assertEqual((first['dividend_income_minor'], first['net_pnl_minor'], first['receivable_minor']), (10000, 20000, 0))
        self.assertEqual(first['dividends'][0]['entitlement_quantity'], 100)
        self.assertEqual(second['dividend_income_minor'], 0)
        self.assertEqual((second['status'], second['marked_pnl_minor']), ('OPEN', 0))
        self.assertEqual(run.payload, before)

    def test_recognized_unpaid_receivable_is_economic_income(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 100, '10'), trade(DAYS[2], 'SELL', 'A', 100, '11')],
            actions=[action(pay='2026-06-09')])
        e, = evaluate(run, dividend_scope=scope(run))['episodes']
        self.assertEqual((e['net_pnl_minor'], e['receivable_minor'], e['statistics_eligible']), (20000, 10000, True))

    def test_scope_only_known_future_ex_classifies_pending_without_changing_nav(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 100, '10'), trade(DAYS[2], 'SELL', 'A', 100, '11')])
        pending = action(ex='2026-06-09', pay='2026-06-10')
        absent = evaluate(run)
        present = evaluate(run, dividend_scope=scope(run, extra=[pending]))
        self.assertEqual(absent['episode_metrics']['dividend_scope_status'], 'COVERAGE_UNKNOWN')
        self.assertIsNone(absent['episodes'][0]['pending_dividend_minor'])
        self.assertEqual(present['episodes'][0]['pending_dividend_minor'], 10000)
        self.assertFalse(present['episodes'][0]['statistics_eligible'])
        self.assertEqual(present['episode_metrics']['income_pending_count'], 1)
        self.assertEqual(absent['series'], present['series'])
        self.assertNotEqual(absent['evaluation_ref'], present['evaluation_ref'])

    def test_future_announcement_and_in_period_correction_are_rejected(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 100, '10')])
        s = scope(run, extra=[action(ex='2026-06-09', pay='2026-06-10')]).to_dict()
        s['actions'][0]['available_at'] = '2026-06-09T12:00:00Z'
        with self.assertRaisesRegex(ContractError, 'later announcement'):
            evaluate(run, dividend_scope=DividendScope.from_dict(s))
        with self.assertRaisesRegex(ContractError, 'new in-period dividend'):
            evaluate(run, dividend_scope=scope(run, extra=[action()]))

    def test_seed_and_open_excluded_tie_is_in_denominator(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 1, '10'), trade(DAYS[2], 'SELL', 'A', 1, '10'),
            trade(DAYS[2], 'SELL', 'B', 10, '21'), trade(DAYS[3], 'BUY', 'A', 1, '10')],
            initial={'B': dict(quantity=10, sellable_quantity=10, cost_minor=20000)})
        w = evaluate(run)
        m = w['episode_metrics']
        self.assertEqual((m['closed_count'], m['eligible_closed_count'], m['open_count'], m['left_censored_count']), (2, 1, 1, 1))
        self.assertEqual((m['tie_count'], m['win_rate'], m['mean_net_pnl_minor']), (1, '0', '0'))
        self.assertTrue(any(e['left_censored'] and not e['statistics_eligible'] for e in w['episodes']))

    def test_month_boundary_initial_peak_and_partial(self):
        days = ['2026-05-29', '2026-06-01', '2026-06-30', '2026-07-01', '2026-07-31', '2026-08-03', '2026-08-31']
        run = saved([trade(days[1], 'BUY', 'A', 100, '10', 1000)], days=days)
        w = evaluate(run, dividend_scope=scope(run))
        self.assertEqual([m['status'] for m in w['monthly_returns']], ['COMPLETE'] * 3)
        self.assertEqual(w['monthly_returns'][0]['return'], '-0.001')
        self.assertEqual(w['monthly_returns'][0]['boundary_session'], days[0])
        self.assertEqual(w['series'][0]['drawdown'], '-0.001')
        partial = saved(days=days, start=days[2])
        p = evaluate(partial)['monthly_returns'][0]
        self.assertEqual(p['status'], 'PARTIAL')
        self.assertIsNone(p['return'])
        self.assertEqual(p['observed_return'], '0')
        truncated = saved(days=days[:-1])
        self.assertEqual(evaluate(truncated)['monthly_returns'][-1]['status'], 'PARTIAL')

    def test_missing_benchmark_anchor_and_gap_never_filled(self):
        run = saved()
        w = evaluate(run, bench=missing_benchmark(run,0))['benchmark']
        self.assertIsNone(w['total_return'])
        self.assertTrue(all(r['nav_index'] is None for r in w['series']))
        w = evaluate(run, bench=missing_benchmark(run,2))['benchmark']
        self.assertIsNone(w['max_drawdown'])
        self.assertIsNone(w['series'][2]['drawdown'])
        self.assertIsNone(w['series'][2]['daily_return'])
        self.assertIsNotNone(w['total_return'])

    def test_fixed_bins_boundaries_and_counts(self):
        values = [-100001, -100000, -50000, -10000, -1, 0, 9999, 10000, 50000, 100000]
        trades = []
        for pnl in values:
            trades += [trade(DAYS[1], 'BUY', 'A', 1, '2000'), trade(DAYS[1], 'SELL', 'A', 1, str(Decimal(200000 + pnl) / 100))]
        run = saved(trades)
        d = evaluate(run)['pnl_distribution']
        self.assertEqual((d['status'], d['included_episode_count']), ('AVAILABLE', 10))
        self.assertEqual([b['count'] for b in d['bins']], [1, 1, 1, 2, 2, 1, 1, 1])
        self.assertEqual(sum(b['count'] for b in d['bins']), 10)

    def test_global_decimal_context_and_repeat_output(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 3, '10'), trade(DAYS[2], 'SELL', 'A', 3, '11')])
        expected = evaluate_backtest(run, benchmark=benchmark(run), spec=daily_evaluation_spec()).payload
        prior = getcontext().copy()
        try:
            getcontext().prec = 6
            getcontext().rounding = ROUND_DOWN
            getcontext().traps[Inexact] = True
            actual = evaluate_backtest(run, benchmark=benchmark(run), spec=daily_evaluation_spec()).payload
            self.assertEqual(actual, expected)
        finally:
            setcontext(prior)

    def test_saved_reader_hash_refs_no_business_recomputation_or_write(self):
        run = saved()
        report = evaluate_backtest(run, benchmark=benchmark(run), spec=daily_evaluation_spec())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'evaluation.json'
            save_backtest_evaluation(report, p)
            before = (p.read_bytes(), p.stat().st_mtime_ns)
            with (patch('axiom_engine.runtime.evaluation.evaluate_backtest', side_effect=AssertionError('recompute')),
                    patch('axiom_engine.runtime.backtest.run_backtest', side_effect=AssertionError('replay'))):
                self.assertEqual(load_backtest_evaluation(p).payload, report.payload)
            self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before)
            save_backtest_evaluation(report, p)
            self.assertEqual(p.stat().st_mtime_ns, before[1])
            other = evaluate_backtest(run, benchmark=benchmark(run), spec=daily_evaluation_spec(), dividend_scope=scope(run))
            with self.assertRaisesRegex(ContractError, 'conflicting'):
                save_backtest_evaluation(other, p)
            wire = json.loads(p.read_text())
            wire['series'][0]['nav_minor'] += 1
            p.write_text(json.dumps(wire))
            with self.assertRaisesRegex(ContractError, 'content digest'):
                load_backtest_evaluation(p)

    def test_watermark_external_flow_and_benchmark_keys_rejected(self):
        run = saved([trade(DAYS[1], 'BUY', 'A', 1, '10')])
        w = run.to_dict()
        w['positions'][0]['committed_sequence'] += 1
        with self.assertRaisesRegex(ContractError, 'watermarks'):
            evaluate(rehash(w))
        w = run.to_dict()
        w['cash_ledger'].append(dict(reason='DEPOSIT', source_event_id='x'))
        with self.assertRaisesRegex(ContractError, 'external cash flows'):
            evaluate(rehash(w))
        b = benchmark(run).to_dict()
        b['rows'].pop()
        with self.assertRaisesRegex(ContractError, 'key coverage'):
            evaluate(run, bench=BenchmarkSeries.from_dict(b))

    def test_native_projection_deletion_amount_change_and_missing_proof_rejected(self):
        run = saved([trade(DAYS[1],'BUY','A',100,'10'),trade(DAYS[2],'SELL','A',100,'11')])
        original = scope(run, extra=[action(ex='2026-06-09',pay='2026-06-10')]).to_dict()
        deleted = deepcopy(original)
        deleted['actions'] = []
        with self.assertRaisesRegex(ContractError, 'projection'):
            evaluate(run, dividend_scope=DividendScope.from_dict(deleted))
        changed = deepcopy(original)
        changed['actions'][0]['cash_per_unit'] = '999'
        with self.assertRaisesRegex(ContractError, 'projection'):
            evaluate(run, dividend_scope=DividendScope.from_dict(changed))
        missing = deepcopy(original)
        missing['source_evidence'] = []
        with self.assertRaisesRegex(ContractError, 'Snapshot mismatch|native DataBatch'):
            evaluate(run, dividend_scope=DividendScope.from_dict(missing))
        b = benchmark(run).to_dict()
        b['rows'][-1]['close'] = '999999'
        with self.assertRaisesRegex(ContractError, 'projection'):
            evaluate(run, bench=BenchmarkSeries.from_dict(b))

    def test_native_query_scope_cannot_claim_record_date_or_snapshot_coverage(self):
        run = saved()
        original = scope(run).to_dict()
        for key,value in [('time_field','ex_date'),('cutoff','2026-06-09T20:30:00+08:00'),('symbols',['A'])]:
            s=deepcopy(original)
            batch=s['source_evidence'][0]['batch']
            batch['context']['query'][key]=value
            ref=Document.from_dict(batch).identity
            s['source_refs']=[ref]
            s['source_evidence'][0]['reference']=ref
            with self.assertRaisesRegex(ContractError,'query scope'):
                evaluate(run,dividend_scope=DividendScope.from_dict(s))
        b=benchmark(run).to_dict()
        batch=b['source_evidence'][0]['batch']
        batch['context']['snapshot_id']='s_other'
        ref=Document.from_dict(batch).identity
        b['source_refs']=[ref]
        b['source_evidence'][0]['reference']=ref
        for row in b['rows']: row['source_refs']=[ref]
        with self.assertRaisesRegex(ContractError,'Snapshot mismatch'):
            evaluate(run,bench=BenchmarkSeries.from_dict(b))


class AdapterTests(unittest.TestCase):
    def test_public_benchmark_mapping_unit_missing_and_frozen_batch(self):
        class Query:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
        batch = dict(records=[dict(session=DAYS[0], security_id='000300.SH', close=100),
                dict(session=DAYS[1], security_id='000300.SH', close=None)],
            field_meta=dict(close=dict(unit='index points', by_key=[dict(session=d, security_id='000300.SH',
                usable_from=d+'T12:00:00Z' if i == 0 else None, missing_reason=None if i == 0 else 'not_visible_at_cutoff') for i,d in enumerate(DAYS[:2])])),
            context=dict(contract_version='data_batch_v1', domain='benchmark_daily', snapshot_id='s_test', reader_version='synthetic_reader',
                query=dict(purpose='market_replay', pit_policy='best_effort_vendor_v1',fields=['close'],symbols=['000300.SH'],
                    sessions=DAYS[:2],price_basis='unadjusted',cutoff_by_session={d:d+'T20:30:00+08:00' for d in DAYS[:2]})))
        class Data:
            def read_market(self, *, snapshot, query):
                self.query = query
                return SimpleNamespace(to_json=lambda: batch)
        data = Data()
        with patch.dict(sys.modules, {'axiom_data': SimpleNamespace(QuerySpec=Query)}):
            b = read_csi300_benchmark(data, snapshot='s_test', sessions=DAYS[:2]).to_dict()
            self.assertEqual(data.query.args[:3], ('benchmark_daily', ('close',), ('000300.SH',)))
            self.assertEqual(b['rows'][1]['close'], None)
            self.assertEqual(b['rows'][1]['missing_reason'], 'not_visible_at_cutoff')
            self.assertEqual(b['source_evidence'][0]['batch'], batch)
            batch['field_meta']['close']['unit'] = 'CNY'
            with self.assertRaisesRegex(ContractError, 'index points'):
                read_csi300_benchmark(data, snapshot='s_test', sessions=DAYS[:2])


if __name__ == '__main__':
    unittest.main()
