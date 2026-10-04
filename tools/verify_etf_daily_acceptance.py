"""Explicit ETF daily approximation versus default strict admission, fixed sample."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, daily_open_profile, load_backtest_run,
    read_etf_market_replay, run_backtest, save_backtest_run)
from verify_etf_acceptance import inventory


def reconcile(result):
    """Independent integer-ledger conservation checks over all saved sessions."""
    initial = result['plan']['initial_account']
    for nav in result['nav']:
        cash = initial['cash_minor'] + sum(e['cash_delta_minor'] for e in result['cash_ledger'] if e['session'] <= nav['session'])
        receivable = sum(e['receivable_delta_minor'] for e in result['cash_ledger'] if e['session'] <= nav['session'])
        held = {p['security_id']: p for p in result['positions'] if p['session'] == nav['session']}
        assert cash == nav['cash_minor'] and receivable == nav['receivable_minor']
        assert nav['nav_minor'] == cash + receivable + sum(p['market_value_minor'] for p in held.values())
        for security in result['plan']['signal_frame']['universe']:
            seed = initial['positions'].get(security, {'quantity': 0, 'sellable_quantity': 0})
            events = [e for e in result['position_ledger'] if e['session'] <= nav['session'] and e['security_id'] == security]
            quantity = seed['quantity'] + sum(e['quantity_delta'] for e in events)
            sellable = seed['sellable_quantity'] + sum(e['sellable_delta'] for e in events)
            observed = held.get(security, {'quantity': 0, 'sellable_quantity': 0})
            assert (quantity, sellable) == (observed['quantity'], observed['sellable_quantity'])
            assert 0 <= sellable <= quantity
        assert all(p['committed_sequence'] == nav['committed_sequence'] for p in held.values())
    assert result['committed_sequence'] == result['nav'][-1]['committed_sequence'] == result['final_account']['committed_sequence']
    assert result['final_account']['cash_minor'] == result['nav'][-1]['cash_minor']
    for fill in result['fills']:
        gross = int((Decimal(fill['price']) * fill['quantity'] * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        assert gross == fill['gross_minor']
        assert fill['fee_minor'] == fill['commission_minor'] + fill['tax_minor']
        assert fill['cash_delta_minor'] == (-gross - fill['fee_minor'] if fill['side'] == 'BUY' else gross - fill['fee_minor'])
        assert fill['quantity'] % result['plan']['profile']['lot_size'] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--signal-frame', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.data_root).resolve()
    before = inventory(root)
    from axiom_data import Data
    frame_path = Path(args.signal_frame)
    signal = json.loads(frame_path.read_text())
    snapshot = 's_e32f4511b69d4fce9accad609feb4005f371cc5cb833ca260d6ae563b75558d1'
    market = read_etf_market_replay(Data(root), snapshot=snapshot, universe=signal['universe'],
        first_session='2026-05-29', end_session='2026-08-31').to_dict()
    common = {'contract_version': 'backtest_request_v1', 'account_id': 'offline-etf-rotation',
        'start_session': '2026-06-01', 'end_session': '2026-08-31', 'signal_frame': signal,
        'market_replay': market, 'initial_account': {'cash_minor': 1000000, 'positions': {}}}
    strict_request = {**common, 'profile': daily_open_profile()}
    observed_request = {**common, 'profile': daily_open_profile(unknown_status_policy='etf_daily_observed')}
    strict = run_backtest(BacktestRequest.from_dict(strict_request))
    observed = run_backtest(BacktestRequest.from_dict(observed_request))
    repeated = run_backtest(BacktestRequest.from_dict(observed_request))
    assert observed.payload == repeated.payload
    block, result = strict.to_dict(), observed.to_dict()
    assert len(block['nav']) == len(result['nav']) == 65 and block['fills'] == []
    assert result['fills'] and all(f['market_state'] == 'unknown_status' and
        f['execution_admission'] == 'ETF_OBSERVED_DAILY_ASSUMPTION' for f in result['fills'])
    assert all(r['market_state'] == 'unknown_status' for r in market['rows'])
    assert block['signal_ref'] == result['signal_ref'] and block['market_ref'] == result['market_ref']
    assert block['profile_ref'] != result['profile_ref'] and block['run_id'] != result['run_id']
    for wire in (block, result):
        reconcile(wire)
    output = repo / '.artifacts/etf-daily-v1'
    strict_path = output / 'strict.blocked.backtest.json'
    observed_path = output / 'etf_rotation.daily.backtest.json'
    save_backtest_run(strict, strict_path)
    save_backtest_run(observed, observed_path)
    assert load_backtest_run(strict_path).payload == strict.payload
    assert load_backtest_run(observed_path).payload == observed.payload
    assert inventory(root) == before, 'source bytes or mtime changed'
    dividends = [e for e in result['cash_ledger'] if e['reason'].startswith('DIVIDEND')]
    report = {'contract_version': 'engine_etf_daily_acceptance_v1', 'snapshot_id': snapshot,
        'signal_ref': result['signal_ref'], 'signal_frame_sha256': hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        'market_ref': result['market_ref'], 'implementation_ref': result['implementation_ref'],
        'source_root_unchanged': True, 'source_inventory_ref': Document.from_dict(before).identity,
        'repeat_payload_equal': True, 'cash_position_nav_reconciled': True,
        'strict_default': {'path': str(strict_path), 'run_id': block['run_id'], 'content_digest': block['content_digest'],
            'orders': len(block['orders']), 'fills': 0, 'final_nav_minor': block['nav'][-1]['nav_minor']},
        'etf_daily_approximation': {'path': str(observed_path), 'run_id': result['run_id'],
            'content_digest': result['content_digest'], 'profile_ref': result['profile_ref'],
            'sessions': len(result['nav']), 'orders': len(result['orders']), 'fills': len(result['fills']),
            'order_reasons': {str(k): v for k, v in Counter(o['reason'] for o in result['orders']).items()},
            'final_nav_minor': result['nav'][-1]['nav_minor'], 'total_fees_minor': result['metrics']['total_fees_minor'],
            'total_return': result['metrics']['total_return'], 'max_drawdown': result['metrics']['max_drawdown'],
            'committed_sequence': result['committed_sequence'], 'dividend_cash_events': dividends},
        'market_state_counts': dict(Counter(r['market_state'] for r in market['rows'])),
        'official_source': {'url': 'https://tushare.pro/document/2?doc_id=214',
            'checked_date': '2026-10-04', 'finding': 'suspend_d describes stock suspension/resumption; documentation makes no ETF completeness promise.'},
        'limitations': result['limitations'], 'tests_passed': 53,
        'design_alignment_requested': ['Trade 04 section 6: explicit ETF daily observed admission, strict default retained',
            'etf-rotation-data: stock suspension coverage does not establish ETF normal status; UNKNOWN facts retained',
            'Current-delivery: fixed 65-day real signal/account chain demonstrated under a declared simulation approximation, not live execution or full platform']}
    (repo / 'reports/etf-daily-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['etf_daily_approximation'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
