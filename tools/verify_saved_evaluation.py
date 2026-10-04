"""Bounded P10 acceptance with private outputs and unchanged input inventories."""
import argparse
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from axiom_engine.runtime import (load_backtest_run, load_backtest_evaluation, save_backtest_evaluation,
    read_csi300_benchmark, read_dividend_scope, daily_evaluation_spec, evaluate_backtest)


def inventory(path):
    paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    return {str(p): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        mtime_ns=p.stat().st_mtime_ns, size=p.stat().st_size) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data-root', 'snapshot', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    from axiom_data import Data
    run_path, root, out = Path(args.run), Path(args.data_root), Path(args.output_dir)
    before = dict(run=inventory(run_path), data=inventory(root))
    saved = load_backtest_run(run_path)
    run = saved.to_dict()
    plan = run['plan']
    calendar = plan['market_replay']['calendar']
    anchor = calendar[calendar.index(plan['start_session']) - 1]
    data = Data(root)
    benchmark = read_csi300_benchmark(data, snapshot=args.snapshot,
        sessions=[anchor, *[r['session'] for r in run['nav']]])
    scope = read_dividend_scope(data, snapshot=args.snapshot, universe=plan['signal_frame']['universe'],
        start_session=plan['start_session'], end_session=plan['end_session'])
    with patch('axiom_engine.runtime.backtest.run_backtest', side_effect=AssertionError('account replay')):
        report = evaluate_backtest(saved, benchmark=benchmark, spec=daily_evaluation_spec(), dividend_scope=scope)
        repeated = evaluate_backtest(saved, benchmark=benchmark, spec=daily_evaluation_spec(), dividend_scope=scope)
    assert report.payload == repeated.payload
    wire = report.to_dict()
    assert wire['input_run_ref'] == {k: run[k] for k in ('run_id', 'content_digest', 'committed_sequence')}
    assert [(r['session'], r['nav_minor'], r['nav_index'], r['committed_sequence']) for r in wire['series']] == [
        (r['session'], r['nav_minor'], r['nav_index'], r['committed_sequence']) for r in run['nav']]
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        peak, previous = run['initial_nav_minor'], run['initial_nav_minor']
        for source, row in zip(run['nav'], wire['series']):
            peak = max(peak, source['nav_minor'])
            assert row['peak_nav_minor'] == peak and Decimal(row['drawdown']) == Decimal(source['nav_minor']) / peak - 1
        for month in wire['monthly_returns']:
            dates = [r for r in run['nav'] if r['session'].startswith(month['month'])]
            if month['status'] == 'COMPLETE':
                assert Decimal(month['return']) == Decimal(dates[-1]['nav_minor']) / previous - 1
                previous = dates[-1]['nav_minor']
        if not plan['initial_account']['positions']:
            marked = sum((e['net_pnl_minor'] if e['status'] == 'CLOSED' else e['marked_pnl_minor']) or 0 for e in wire['episodes'])
            assert marked == run['nav'][-1]['nav_minor'] - run['initial_nav_minor']
        assert sum(e['fees_minor'] for e in wire['episodes']) == sum(f['fee_minor'] for f in run['fills'])
        assert sum(e['dividend_income_minor'] for e in wire['episodes']) == sum(
            e['receivable_delta_minor'] for e in run['cash_ledger'] if e['reason'] == 'DIVIDEND_EX')
        eligible = [e for e in wire['episodes'] if e['statistics_eligible']]
        metrics = wire['episode_metrics']
        assert metrics['win_count'] + metrics['loss_count'] + metrics['tie_count'] == len(eligible)
        if eligible:
            assert Decimal(metrics['win_rate']) == Decimal(sum(e['net_pnl_minor'] > 0 for e in eligible)) / len(eligible)
    path = out / 'evaluation.json'
    save_backtest_evaluation(report, path)
    output_before = inventory(path)
    with patch('axiom_engine.runtime.evaluation.evaluate_backtest', side_effect=AssertionError('reader computes')):
        assert load_backtest_evaluation(path).payload == report.payload
    assert inventory(path) == output_before
    after = dict(run=inventory(run_path), data=inventory(root))
    assert before == after
    evidence = dict(status='PASS', contract_version=wire['contract_version'], evaluation_ref=wire['evaluation_ref'],
        content_digest=wire['content_digest'], implementation_ref=wire['implementation_ref'],
        input_run_ref=wire['input_run_ref'], benchmark_ref=wire['benchmark_ref'], dividend_scope_ref=wire['dividend_scope_ref'],
        sessions=len(wire['series']), monthly_returns=wire['monthly_returns'], episode_metrics=wire['episode_metrics'],
        output_path=str(path.resolve()), checks=['no_account_replay', 'repeat_bytes_equal', 'saved_NAV_watermarks_preserved',
            'daily_drawdown_initial_peak', 'monthly_previous_boundary', 'episode_currency_PnL_reconciles',
            'fees_and_EX_income_reconcile', 'reader_no_recompute', 'run_and_source_hash_mtime_size_unchanged'],
        original_inventories=before, evaluation_inventory=output_before)
    (out / 'acceptance.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: evidence[k] for k in ('status', 'evaluation_ref', 'content_digest', 'sessions', 'episode_metrics', 'output_path')}))


if __name__ == '__main__':
    main()
