"""Saved SSE byte/clock/account admission and percentage projections."""
from copy import deepcopy
from decimal import Decimal, Inexact, getcontext, setcontext
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError
from axiom_engine.runtime import (retrospective_sse_benchmark,read_sse_benchmark,evaluate_saved_analysis,
    analysis_evaluation_spec,load_backtest_evaluation,save_backtest_evaluation)
from test_analysis_evaluation import wealth,analysis,RF,rehash_report


def native(base, values=None):
    days=[base['period_metrics']['window']['anchor_session'],*[p['session'] for p in base['series']]]
    cutoff='2026-10-05T06:34:05+00:00';observed='2026-10-05T06:34:04+00:00'
    values=values or [100.0+10*i for i in range(len(days))]
    context=dict(contract_version='data_batch_v1',snapshot_id='s_new_observation',domain='benchmark_daily',
        contract_id='synthetic.benchmark/1',source_profile_id='synthetic.index_daily/1',reader_version='synthetic_reader/1',
        query=dict(fields=['close'],symbols=['000001.SH'],sessions=days,purpose='historical_exploration',
            pit_policy='operational_pit_v1',cutoff_by_session={d:cutoff for d in days}))
    batch=dict(context=context,records=[dict(security_id='000001.SH',session=d,close=v) for d,v in zip(days,values)],
        field_meta={'close':dict(dtype='float64',unit='index points',by_key=[dict(security_id='000001.SH',session=d,
            revision_id='synthetic:'+d,raw_batch_id='raw',usable_from=observed if v is not None else None,
            first_observed_at=observed,missing_reason=None if v is not None else 'source_missing') for d,v in zip(days,values)])})
    receipt=dict(status='SYNTHETIC_PUBLIC_NATIVE_BATCH',snapshot_id=context['snapshot_id'],source='synthetic index daily',
        identity='000001.SH',unit='index points',series_kind='price_index',source_receipt=observed,cutoff=cutoff,
        runs={'test':dict(source_run_ref=base['input_run_ref'],start_session=base['series'][0]['session'],
            end_session=days[-1],comparison_anchor_session=days[0],sessions=len(days),native_batch_ref={},
            context={k:context[k] for k in ('domain','snapshot_id','contract_id','reader_version')})})
    return batch,receipt


def admitted(batch, receipt):
    raw=json.dumps(batch,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    receipt=deepcopy(receipt);receipt['runs']['test']['native_batch_ref']=dict(path='/synthetic/native.json',
        sha256=sha256(raw.encode()).hexdigest(),bytes=len(raw.encode()))
    return retrospective_sse_benchmark(native_batch_text=raw,receipt=receipt,run_key='test')


class RetrospectiveBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.account=wealth([1100000,1200000,1150000]);self.old,self.base=analysis(self.account)
        self.base_wire=self.base.to_dict();self.batch,self.receipt=native(self.base_wire)

    def report(self, bench):
        from axiom_engine.runtime import BenchmarkSeries
        return evaluate_saved_analysis(self.account,self.base,benchmarks=dict(
            CSI300=BenchmarkSeries.from_dict(self.base_wire['benchmark_input']),SSE_COMPOSITE=bench,NASDAQ100=None),
            spec=analysis_evaluation_spec(risk_free=RF))

    def test_new_observation_and_original_receipt_clock_stay_retrospective(self):
        bench=admitted(self.batch,self.receipt);report=self.report(bench).to_dict()
        sse=report['benchmark_comparisons']['SSE_COMPOSITE']
        self.assertEqual(sse['status'],'COMPLETE');self.assertEqual(report['status'],'PARTIAL')
        self.assertEqual(sse['observation_snapshot_id'],'s_new_observation')
        self.assertEqual(sse['observation_cutoff'],self.receipt['cutoff'])
        self.assertEqual(sse['series'][0]['available_at'],'2026-10-05T06:34:04+00:00')
        self.assertEqual(sse['series'][0]['benchmark_cumulative_return'],'0.1')
        self.assertEqual(sse['native_series'][0]['benchmark_cumulative_return'],'0')
        self.assertEqual(report['benchmark_input'],self.base_wire['benchmark_input'])
        self.assertEqual(report['base_evaluation'],self.base_wire)
        for key in ('series','monthly_returns','episodes','period_metrics'):self.assertEqual(report[key],self.base_wire[key])
        self.assertNotEqual(report['evaluation_ref'],self.old.to_dict()['evaluation_ref'])

    def test_file_hash_owner_unit_purpose_and_current_clock_admission(self):
        bench=admitted(self.batch,self.receipt).to_dict()
        with self.assertRaises(ContractError):retrospective_sse_benchmark(
            native_batch_text=bench['native_batch_text']+' ',receipt=bench['receipt'],run_key='test')
        changes=[lambda b:b['field_meta']['close'].update(unit='CNY'),
            lambda b:b['context']['query'].update(purpose='market_replay'),
            lambda b:b['context']['query'].update(pit_policy='best_effort_vendor_v1'),
            lambda b:b['field_meta']['close']['by_key'][0].update(usable_from='2026-10-06T00:00:00Z'),
            lambda b:b['records'].append(deepcopy(b['records'][0]))]
        for change in changes:
            value=deepcopy(self.batch);change(value)
            with self.assertRaises(ContractError):admitted(value,self.receipt)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.json';path.write_text(bench['native_batch_text'])
            self.assertEqual(read_sse_benchmark(path,receipt=bench['receipt'],run_key='test').identity,
                admitted(self.batch,self.receipt).identity)

    def test_fixed_account_triple_and_native_calendar_binding(self):
        receipt=deepcopy(self.receipt);receipt['runs']['test']['source_run_ref']['run_id']='sha256:'+'f'*64
        with self.assertRaises(ContractError):self.report(admitted(self.batch,receipt))
        batch,receipt=native(self.base_wire);day=batch['context']['query']['sessions'].pop(1)
        batch['context']['query']['cutoff_by_session'].pop(day)
        batch['records']=[r for r in batch['records'] if r['session']!=day]
        batch['field_meta']['close']['by_key']=[r for r in batch['field_meta']['close']['by_key'] if r['session']!=day]
        receipt['runs']['test']['sessions']-=1
        with self.assertRaises(ContractError):self.report(admitted(batch,receipt))

    def test_missing_native_price_and_anchor_have_saved_null_percentages(self):
        for values in ([100,None,120,130],[None,110,120,130]):
            batch,receipt=native(self.base_wire,values)
            sse=self.report(admitted(batch,receipt)).to_dict()['benchmark_comparisons']['SSE_COMPOSITE']
            self.assertEqual(sse['status'],'PARTIAL')
            for point in [*sse['native_series'],*sse['series']]:
                self.assertEqual(point['benchmark_cumulative_return'] is None,point['normalized_index'] is None)
            if values[0] is None:self.assertTrue(all(p['benchmark_cumulative_return'] is None for p in sse['series']))

    def test_reader_no_business_compute_and_legacy_no_percentage_compatibility(self):
        report=self.report(admitted(self.batch,self.receipt))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'new.json';save_backtest_evaluation(report,path)
            prior=getcontext().copy()
            try:
                getcontext().prec=1;getcontext().traps[Inexact]=True
                with patch('axiom_engine.runtime.analysis_benchmarks._project_native',side_effect=AssertionError('compute')), \
                     patch('axiom_engine.runtime.analysis_evaluation._risk',side_effect=AssertionError('compute')):
                    self.assertEqual(load_backtest_evaluation(path).payload,report.payload)
            finally:setcontext(prior)
            wire=self.old.to_dict();comparison=wire['benchmark_comparisons']['CSI300']
            for point in [*comparison['native_series'],*comparison['series']]:point.pop('benchmark_cumulative_return')
            legacy=rehash_report(wire);path=Path(temp)/'old.json';save_backtest_evaluation(legacy,path)
            self.assertEqual(load_backtest_evaluation(path).payload,legacy.payload)
            bad=report.to_dict();bad['benchmark_comparisons']['SSE_COMPOSITE']['observation_cutoff']='2020-01-01T00:00:00Z'
            with self.assertRaises(ContractError):save_backtest_evaluation(rehash_report(bad),Path(temp)/'bad.json')


if __name__=='__main__':unittest.main()
