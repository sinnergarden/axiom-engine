"""One account load; display then saved-v2 analysis, releasing full Data payloads."""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import resource
from time import perf_counter
from unittest.mock import patch

from axiom_engine.runtime import (load_backtest_run,load_backtest_evaluation,read_review_display,
    build_fill_display,save_fill_display,load_fill_display,read_sse_benchmark,BenchmarkSeries,
    evaluate_saved_analysis,analysis_evaluation_spec,save_backtest_evaluation)
from verify_saved_fill_display import inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('account','run','base-evaluation','display-handoff','sse-handoff','output-dir',
            'annual-risk-free-rate','risk-free-source'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();account=args.account
    display_handoff_path,sse_handoff_path=Path(args.display_handoff),Path(args.sse_handoff)
    display_handoff=json.loads(display_handoff_path.read_text())
    receipt=json.loads(sse_handoff_path.read_text())
    fixed_display=display_handoff['production_artifacts'][account]
    directory=Path(fixed_display['directory']);sse_path=Path(receipt['runs'][account]['native_batch_ref']['path'])
    run_path,base_path,output=Path(args.run),Path(args.base_evaluation),Path(args.output_dir)
    paths=[run_path,base_path,display_handoff_path,sse_handoff_path,sse_path,directory/'manifest.json',
        *[directory/name for name in fixed_display['files']]]
    before={str(p):inventory(p) for p in paths}
    start=perf_counter();run=load_backtest_run(run_path);run_load_seconds=perf_counter()-start
    run_identity=run.identity
    with ExitStack() as stack:
        for target in ('axiom_engine.runtime.backtest.run_backtest','axiom_engine.runtime.accounting.AccountLedger.apply_fill',
                'axiom_engine.runtime.evaluation.evaluate_backtest','axiom_data.api.Data','axiom_data.adjust_prices',
                'axiom_data.review_display.project_review_display'):
            stack.enter_context(patch(target,side_effect=AssertionError('forbidden account/query/transform: '+target)))
        start=perf_counter();display=read_review_display(directory,manifest_sha256=fixed_display['manifest_file_ref']['sha256'])
        display_load_seconds=perf_counter()-start
        start=perf_counter();mapped=build_fill_display(run,display=display);display_build_seconds=perf_counter()-start
        wire=mapped.to_dict();reasons={}
        for c in wire['coordinates']:
            reason=c['reason'] or 'AVAILABLE';reasons[reason]=reasons.get(reason,0)+1
        display_path=output/'fill-display.json';save_fill_display(mapped,display_path)
        with patch('axiom_engine.runtime.fill_display._coordinate',side_effect=AssertionError('reader computes')):
            assert load_fill_display(display_path).payload==mapped.payload
        display_evidence={k:wire[k] for k in ('input_run_ref','display_ref','display_result_ref','consumed_input_ref',
            'content_digest','implementation_ref','status')}
        display_evidence.update(fills=len(wire['fills']),coordinate_status=reasons,output_path=str(display_path.resolve()),
            output_inventory=inventory(display_path),load_seconds=display_load_seconds,build_seconds=display_build_seconds)
        del display,mapped,wire
        gc.collect()  # No full Data OHLCV is retained with the base evaluation.
        start=perf_counter();base=load_backtest_evaluation(base_path);base_load_seconds=perf_counter()-start
        old=base.to_dict();bench=read_sse_benchmark(sse_path,receipt=receipt,run_key=account)
        spec=analysis_evaluation_spec(risk_free=dict(currency='CNY',annual_effective_rate=args.annual_risk_free_rate,
            source=args.risk_free_source))
        start=perf_counter();analysis=evaluate_saved_analysis(run,base,benchmarks=dict(
            CSI300=BenchmarkSeries.from_dict(old['benchmark_input']),SSE_COMPOSITE=bench,NASDAQ100=None),spec=spec)
        analysis_seconds=perf_counter()-start
    wire=analysis.to_dict()
    for key in ('series','monthly_returns','episodes','pnl_distribution','benchmark','period_metrics'):
        assert wire[key]==old[key],key
    assert wire['base_evaluation']==old and run.identity==run_identity
    analysis_path=output/'evaluation.json';save_backtest_evaluation(analysis,analysis_path)
    with patch('axiom_engine.runtime.analysis_benchmarks._project_native',side_effect=AssertionError('reader computes')), \
         patch('axiom_engine.runtime.analysis_evaluation._risk',side_effect=AssertionError('reader computes')):
        assert load_backtest_evaluation(analysis_path).payload==analysis.payload
    for key in ('CSI300','SSE_COMPOSITE'):
        c=wire['benchmark_comparisons'][key]
        assert all('benchmark_cumulative_return' in p for p in [*c['native_series'],*c['series']])
    analysis_evidence={k:wire[k] for k in ('input_run_ref','base_evaluation_ref','evaluation_ref','content_digest',
        'implementation_ref','benchmark_refs','risk_metrics','drawdown_interval','status')}
    analysis_evidence.update(benchmark_status={k:v['status'] for k,v in wire['benchmark_comparisons'].items()},
        sessions=len(wire['series']),output_path=str(analysis_path.resolve()),output_inventory=inventory(analysis_path),
        base_load_seconds=base_load_seconds,evaluate_seconds=analysis_seconds)
    assert before=={str(p):inventory(p) for p in paths}
    evidence=dict(status='PASS',source_kind='saved_real_account_and_fixed_Data_inputs',run_load_seconds=run_load_seconds,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,display=display_evidence,
        analysis=analysis_evidence,original_inputs=before,
        checks=['one_run_load','release_Data_before_base_load','native_account_and_base_values_unchanged',
            'original_SHA_mtime_size_unchanged','zero_Data_queries_transforms_account_or_old_evaluator_calls',
            'loaders_no_business_compute','SSE_new_observation_current_cutoff','benchmark_percentages_saved','Nasdaq_missing_explicit'])
    (output/'acceptance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',account=account,display_fills=display_evidence['fills'],coordinate_status=reasons,
        benchmark_status=analysis_evidence['benchmark_status'],run_load_seconds=run_load_seconds,
        display_load_seconds=display_load_seconds,display_build_seconds=display_build_seconds,
        base_load_seconds=base_load_seconds,analysis_seconds=analysis_seconds,
        peak_process_rss_bytes=evidence['peak_process_rss_bytes'])))


if __name__=='__main__':main()
