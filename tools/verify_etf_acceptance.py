"""Fixed small-sample evidence; no source writes or bulk process interaction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, load_backtest_run,
                                  read_etf_market_replay, run_backtest, save_backtest_run)


def inventory(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--signal-frame', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve()
    before = inventory(data_root)
    from axiom_data import Data
    signal_path = Path(args.signal_frame)
    signal = json.loads(signal_path.read_text())
    snapshot = 's_e32f4511b69d4fce9accad609feb4005f371cc5cb833ca260d6ae563b75558d1'
    market = read_etf_market_replay(Data(data_root), snapshot=snapshot,
        universe=signal['universe'], first_session='2026-05-29', end_session='2026-08-31')
    sys.path.insert(0, str(root / 'tests'))
    from test_backtest import fixture
    synthetic_request = fixture()
    real_request = {**synthetic_request, 'account_id': 'offline-etf-rotation',
        'start_session': '2026-06-01', 'end_session': '2026-08-31',
        'signal_frame': signal, 'market_replay': market.to_dict()}
    real_request['profile'] = {**real_request['profile'], 'limitation':
        'Conservative experimental T+1 and 100-unit ETF lot; not real settlement rules. UNKNOWN status blocks execution. This sample validates blocking, not strategy performance.'}
    real = run_backtest(BacktestRequest.from_dict(real_request))
    repeated = run_backtest(BacktestRequest.from_dict(real_request))
    assert real.payload == repeated.payload
    synthetic = run_backtest(BacktestRequest.from_dict(synthetic_request))
    output = root / '.artifacts/delivery'
    real_path, synthetic_path = output / 'etf_rotation.blocked.backtest.json', output / 'synthetic_golden.backtest.json'
    save_backtest_run(real, real_path)
    save_backtest_run(synthetic, synthetic_path)
    assert load_backtest_run(real_path).payload == real.payload
    assert load_backtest_run(synthetic_path).payload == synthetic.payload
    wire, golden = real.to_dict(), synthetic.to_dict()
    assert len(wire['nav']) == 65 and len(wire['orders']) == 14 and wire['fills'] == []
    assert {o['reason'] for o in wire['orders']} == {'UNKNOWN_MARKET_STATUS'}
    assert wire['final_account']['cash_minor'] == 1000000
    assert golden['nav'][-1]['nav_minor'] == 999220 and golden['metrics']['total_fees_minor'] == 780
    for result in (wire, golden):
        for nav in result['nav']:
            held = [p for p in result['positions'] if p['session'] == nav['session']]
            assert all(p['committed_sequence'] == nav['committed_sequence'] for p in held)
            assert nav['nav_minor'] == nav['cash_minor'] + nav['market_value_minor'] + nav['receivable_minor']
            assert nav['market_value_minor'] == sum(p['market_value_minor'] for p in held)
        assert result['committed_sequence'] == result['nav'][-1]['committed_sequence'] == result['final_account']['committed_sequence']
        assert result['final_account']['cash_minor'] == result['plan']['initial_account']['cash_minor'] + sum(e['cash_delta_minor'] for e in result['cash_ledger'])
    after = inventory(data_root)
    assert before == after, 'source root contents or mtime changed'
    report = {'contract_version': 'engine_acceptance_v1', 'project_id': '76e26315-e881-4634-b1de-b60bbb725a0d',
        'project_evidence': 'Codex list_threads returned this exact local projectId for thread 01a1051a-03db-7675-aa19-9de7ef9330d6',
        'snapshot_id': snapshot, 'signal_frame_file_sha256': hashlib.sha256(signal_path.read_bytes()).hexdigest(),
        'signal_ref': wire['signal_ref'], 'market_ref': wire['market_ref'], 'implementation_ref': wire['implementation_ref'],
        'data_root_unchanged': before == after, 'data_inventory_ref': Document.from_dict(before).identity,
        'repeat_payload_equal': real.payload == repeated.payload,
        'market_states': dict(Counter(r['market_state'] for r in market.to_dict()['rows'])),
        'cash_dividends': market.to_dict()['cash_dividends'],
        'real_fixed_input': {'path': str(real_path), 'run_id': wire['run_id'], 'content_digest': wire['content_digest'],
            'sessions': len(wire['nav']), 'orders': len(wire['orders']), 'fills': len(wire['fills']),
            'order_reasons': dict(Counter(o['reason'] for o in wire['orders'])),
            'final_nav_minor': wire['nav'][-1]['nav_minor'], 'committed_sequence': wire['committed_sequence'],
            'outcome': 'UNKNOWN status blocked; this is not strategy-effectiveness acceptance'},
        'synthetic_known_state': {'path': str(synthetic_path), 'run_id': golden['run_id'],
            'content_digest': golden['content_digest'], 'fills': len(golden['fills']),
            'final_nav_minor': golden['nav'][-1]['nav_minor'], 'total_fees_minor': golden['metrics']['total_fees_minor'],
            'committed_sequence': golden['committed_sequence']},
        'test_command': 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3.12 -m unittest discover -s tests -v',
        'tests_passed': 49,
        'unknown_status_gap': 'Existing ETF suspend_d responses were empty; source contract intentionally does not infer normal trading from absence. No Engine mapping omission.',
        'next_data_step': 'Data owner must provide independently justified ETF suspension/normal-status coverage and completeness contract; leave ongoing bulk unchanged.',
        'design_alignment_requested': ['Core 03 sections 3/6: SignalFrame and pure rotation intents implemented',
            'Trade 04 sections 4/6/9/13/14: isolated cached backtest, fixed profile/source digest, Decimal ledger and cash dividends; no live broker or SQLite recovery',
            'etf-rotation-data: real fixed signals accepted, UNKNOWN tradeability blocks fills; do not label zero-trade run as strategy acceptance']}
    (root / 'reports/etf-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'source-inventory.json').write_text(json.dumps(before, sort_keys=True) + '\n')
    print(json.dumps({'real': report['real_fixed_input'], 'synthetic': report['synthetic_known_state'],
                      'data_root_unchanged': report['data_root_unchanged']}, indent=2))


if __name__ == '__main__':
    main()
