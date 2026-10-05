"""V3 saved-only acceptance; never collect Data or execute an account."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter
from unittest.mock import patch

from axiom_engine.runtime import (BenchmarkSeries, analysis_evaluation_spec,
    evaluate_saved_analysis, load_backtest_run, load_backtest_evaluation, save_backtest_evaluation)


def inventory(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    stat=path.stat()
    return dict(sha256=digest.hexdigest(),mtime_ns=stat.st_mtime_ns,size=stat.st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','base-evaluation','output-dir','annual-risk-free-rate','risk-free-source'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run_path,base_path,output=Path(args.run),Path(args.base_evaluation),Path(args.output_dir)
    before={str(p):inventory(p) for p in (run_path,base_path)}
    start=perf_counter()
    run=load_backtest_run(run_path);base=load_backtest_evaluation(base_path)
    load_seconds=perf_counter()-start
    old=base.to_dict()
    assert old['contract_version']=='evaluation_report_v2'
    spec=analysis_evaluation_spec(risk_free=dict(currency='CNY',annual_effective_rate=args.annual_risk_free_rate,source=args.risk_free_source))
    benchmarks=dict(CSI300=BenchmarkSeries.from_dict(old['benchmark_input']),SSE_COMPOSITE=None,NASDAQ100=None)
    start=perf_counter()
    with patch('axiom_engine.runtime.backtest.run_backtest',side_effect=AssertionError('account replay')), \
         patch('axiom_engine.runtime.accounting.AccountLedger.apply_fill',side_effect=AssertionError('fill execution')), \
         patch('axiom_engine.runtime.evaluation.evaluate_backtest',side_effect=AssertionError('old evaluator')):
        report=evaluate_saved_analysis(run,base,benchmarks=benchmarks,spec=spec)
    evaluate_seconds=perf_counter()-start
    wire=report.to_dict()
    for key in ('input_run_ref','signal_ref','market_ref','profile_ref','series','monthly_returns','episodes',
                'episode_metrics','pnl_distribution','benchmark','period_metrics'):
        assert wire[key]==old[key],key
    assert wire['base_evaluation']==old
    assert wire['drawdown_interval']['drawdown']==old['period_metrics']['account']['max_drawdown']
    path=output/'evaluation.json';save_backtest_evaluation(report,path)
    with patch('axiom_engine.runtime.analysis_evaluation._risk',side_effect=AssertionError('reader computes')), \
         patch('axiom_engine.runtime.analysis_evaluation._drawdown',side_effect=AssertionError('reader computes')):
        assert load_backtest_evaluation(path).payload==report.payload
    assert before=={str(p):inventory(p) for p in (run_path,base_path)}
    evidence=dict(status='PASS',source_kind='saved_real_account_readonly',input_run_ref=wire['input_run_ref'],
        base_evaluation_ref=wire['base_evaluation_ref'],evaluation_ref=wire['evaluation_ref'],content_digest=wire['content_digest'],
        implementation_ref=wire['implementation_ref'],risk_metrics=wire['risk_metrics'],drawdown_interval=wire['drawdown_interval'],
        sessions=len(wire['series']),eligible_episodes=wire['return_distribution']['included_episode_count'],
        distribution_status=wire['return_distribution']['status'],distribution_bins=wire['return_distribution']['bins'],
        benchmark_status={k:v['status'] for k,v in wire['benchmark_comparisons'].items()},
        execution_summary=wire['execution_summary'],load_seconds=load_seconds,evaluate_seconds=evaluate_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        original_inputs=before,output_path=str(path.resolve()),output_inventory=inventory(path),
        checks=['saved_run_and_v2_hash_mtime_size_unchanged','v2_facts_preserved','new_v3_identity','no_Data_calls',
                'no_account_or_fill_execution','no_old_evaluator','v3_reader_no_business_compute','missing_benchmark_inputs_explicit'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'acceptance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('status','sessions','eligible_episodes','risk_metrics','load_seconds','evaluate_seconds','peak_rss_bytes')},ensure_ascii=False))


if __name__=='__main__':main()
