"""Offline public API example; writes only the explicit output path."""
import argparse
import json
from pathlib import Path

from axiom_engine.runtime import (BacktestRequest, read_etf_market_replay,
                                  run_backtest, save_backtest_run, daily_open_profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--signal-frame', required=True)
    parser.add_argument('--first-session', required=True, help='strict preceding trading session')
    parser.add_argument('--start-session', required=True)
    parser.add_argument('--end-session', required=True)
    parser.add_argument('--cash-minor', type=int, required=True, help='initial cash in CNY fen')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from axiom_data import Data
    signals = json.loads(Path(args.signal_frame).read_text())
    market = read_etf_market_replay(Data(args.data_root), snapshot=args.snapshot,
        universe=signals['universe'], first_session=args.first_session, end_session=args.end_session)
    request = BacktestRequest.from_dict({'contract_version': 'backtest_request_v1',
        'account_id': 'offline-etf-rotation', 'start_session': args.start_session,
        'end_session': args.end_session, 'signal_frame': signals, 'market_replay': market.to_dict(),
        'initial_account': {'cash_minor': args.cash_minor, 'positions': {}},
        'profile': daily_open_profile(unknown_status_policy='etf_daily_observed')})
    result = run_backtest(request)
    save_backtest_run(result, args.output)
    wire = result.to_dict()
    print(json.dumps({'run_id': wire['run_id'], 'status': wire['status'],
        'sessions': len(wire['nav']), 'orders': len(wire['orders']), 'fills': len(wire['fills']),
        'final_nav_minor': wire['nav'][-1]['nav_minor'], 'content_digest': wire['content_digest']}, indent=2))


if __name__ == '__main__':
    main()
