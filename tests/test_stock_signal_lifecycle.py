"""Sensitive bounded-admission regressions from the aa06475 scale review."""
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.core import stock_signal, ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.runtime import (BacktestRequest, StockInputSource, admit_stock_inputs,
                                  load_stock_backtest_projection)
from axiom_engine.runtime import stock_market_owner
from axiom_engine.runtime.stock_stream_contracts import prediction_inventory, read_budget
from test_stock_signal_inputs import v3_request, derived_request
from test_stock_market_owner import market_for, bind, seal_request
from test_stock_owned_inputs import run_owned
from test_stock_stream_inputs import artifact_file, source_fixture, LIMITS
from test_stock_stream import execute
from test_stock_stream_projection import mutate_part
from test_csi300_runtime import full_request
from test_csi300 import BOARD_IDS, rules_for


def split_raw(root, plan):
    old=plan['prediction_input']['frames'][0]
    spec=Document(Path(old['fold_spec_artifact']['manifest_uri']).read_text()).to_dict()
    wire=Document(Path(old['prediction_artifact']['manifest_uri']).read_text()).to_dict()
    days=sorted({r['session'] for r in wire['rows']});calendar=plan['scope']['calendar']
    following=dict(zip(calendar[:-1],calendar[1:]));frames=[]
    for i,selected in enumerate((days[:1],days[1:])):
        part_spec=deepcopy(spec)
        part_spec['oos_trade_sessions']=[following[d] for d in selected]
        part_spec['inference_cutoff_by_session']={d:spec['inference_cutoff_by_session'][d] for d in selected}
        part=deepcopy(wire);part['rows']=[r for r in part['rows'] if r['session'] in selected]
        part['fold_spec_ref']=Document.from_dict(part_spec).identity
        part.pop('signal_run_ref');part['signal_run_ref']=Document.from_dict(part).identity
        item=deepcopy(old)
        item.update(fold_ref=Document.from_dict({'synthetic_fold':i,'spec':part_spec}).identity,
            fold_spec_ref=part['fold_spec_ref'],signal_run_ref=part['signal_run_ref'],
            fold_spec_artifact=artifact_file(root,'split-spec-'+str(i),part_spec),
            prediction_artifact=artifact_file(root,'split-prediction-'+str(i),part))
        frames.append(item)
    plan['prediction_input']['frames']=frames
    return seal_request(plan)


def longer_request(root, sessions):
    days=[];current=date(2023,12,29)
    while len(days)<sessions+1:
        if current.weekday()<5:days.append(current.isoformat())
        current+=timedelta(days=1)
    boards={security:board for board,security in BOARD_IDS.items()}
    boards.update({'cnstock.000101.SZ.20000101':'SZSE_MAIN','cnstock.000102.SZ.20000101':'SZSE_MAIN'})
    plan=source_fixture(root,full_request(rules=rules_for(boards=boards,days=days)).to_dict())[0]
    return v3_request(root,base_plan=plan)


class StockSignalLifecycleTests(unittest.TestCase):
    def test_multiple_folds_and_deduplicated_derived_parents_at_exact_actual_row_limit(self):
        for derived in (False,True):
            with self.subTest(derived=derived),TemporaryDirectory() as tmp:
                root=Path(tmp);raw=split_raw(root,v3_request(root))
                plan=derived_request(root,raw) if derived else raw
                grid=(len(plan['scope']['calendar'])-1)*len(plan['scope']['prediction_universe'])
                actual=grid*(2 if derived else 1);limits={**LIMITS,'max_prediction_rows':actual}
                self.assertIsNone(prediction_inventory(plan['prediction_input'],plan['scope'])[1])
                source=StockInputSource()
                try:
                    audit=source.audit(plan,block_sessions=2,read_budget=read_budget(limits),limits=limits,implementation_ref=IMPLEMENTATION_REF)
                    self.assertEqual(audit.receipt['counts']['prediction_rows'],actual)
                finally:source._close_private_views()
                with market_for(raw) as market,bind(market,plan,limits=limits) as inputs:
                    self.assertEqual(inputs.inventory(plan)['declared_rows']['prediction_rows'],actual)
                    owner,projection=run_owned(root/'owner',plan,inputs,limits)
                with admit_stock_inputs(BacktestRequest.from_dict(plan),source=StockInputSource(),block_sessions=2,
                        limits=limits,max_owned_bytes=2_000_000) as inputs:
                    combined,combined_projection=run_owned(root/'combined',plan,inputs,limits)
                ordinary,ordinary_projection,_=execute(root/'ordinary',plan,limits=limits)
                self.assertEqual(owner['final_account'],combined['final_account'])
                self.assertEqual(owner['final_account'],ordinary['final_account'])
                self.assertEqual(projection.rows,combined_projection.rows)
                self.assertEqual(projection.rows,ordinary_projection.rows)
                with market_for(raw) as market,self.assertRaises(ContractError):
                    bind(market,plan,limits={**limits,'max_prediction_rows':actual-1})

    def test_complete_context_once_ref_only_private_days_and_linear_storage_all_entries(self):
        for entry in ('source','market_bind','combined'):
            sizes=[]
            for count in (3,12):
                with self.subTest(entry=entry,sessions=count),TemporaryDirectory() as tmp:
                    root=Path(tmp);raw=longer_request(root,count);plan=derived_request(root,raw)
                    calls=Counter();writes=[];full=[];actual_context=stock_signal._context
                    actual_write=stock_market_owner._Store.write
                    def context(value,universe,days):
                        calls[len(value['reference_members'])]+=1
                        return actual_context(value,universe,days)
                    def write(store,value,memory):
                        if value.get('contract_version')=='stock_signal_metadata_v1':full.append(value)
                        for signal in value.get('signals',[]):
                            self.assertNotIn('header',signal);self.assertIn('signal_ref',signal)
                            writes.append(signal['session'])
                        for header in value.get('signal_headers',{}).values():
                            self.assertNotIn('context',header);self.assertNotIn('signal_plan',header)
                        return actual_write(store,value,memory)
                    with patch.object(stock_signal,'_context',context),patch.object(stock_market_owner._Store,'write',write):
                        if entry=='source':
                            execute(root/'out',plan)
                        elif entry=='market_bind':
                            with market_for(raw) as market,bind(market,plan) as inputs:
                                before=dict(calls)
                                run_owned(root/'out',plan,inputs)
                                self.assertEqual(dict(calls),before)
                                sizes.append(inputs.statistics['owned_bytes'])
                        else:
                            with admit_stock_inputs(BacktestRequest.from_dict(plan),source=StockInputSource(),block_sessions=2,
                                    limits=LIMITS,max_owned_bytes=2_000_000) as inputs:
                                before=dict(calls)
                                run_owned(root/'out',plan,inputs)
                                self.assertEqual(dict(calls),before)
                                sizes.append(inputs.statistics['owned_bytes'])
                    self.assertEqual(calls,{count:1})
                    if entry!='source':
                        self.assertEqual(len(full),1);self.assertEqual(len(writes),count)
                        self.assertEqual(len(full[0]['header']['context']['reference_members']),count)
            if sizes:self.assertLess(sizes[1],4.5*sizes[0])

    def test_saved_raw_target_ref_tamper_rejects_without_opening_original_parents(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);plan=v3_request(root)
            with market_for(plan) as market,bind(market,plan) as inputs:
                wire,_=run_owned(root/'out',plan,inputs)
            target=wire['source_audit']['prediction_targets'][plan['prediction_input']['frames'][0]['signal_run_ref']]
            self.assertEqual(target,Document(Path(plan['prediction_input']['frames'][0]['prediction_artifact']['manifest_uri']).read_text()).to_dict()['label_spec_ref'])
            for frame in plan['prediction_input']['frames']:
                for name in ('fold_spec_artifact','model_metadata_artifact','prediction_artifact'):Path(frame[name]['manifest_uri']).unlink()
            path=root/'out'/'run.json'
            index=next(i for i,p in enumerate(wire['result_parts']) if p['row_counts']['decisions'])
            mutate_part(path,wire,index,lambda part:part['rows']['decisions'][0]['prediction_clock'].update(label_spec_ref='sha256:'+'4'*64))
            with self.assertRaisesRegex(ContractError,'admitted LabelSpec'):
                load_stock_backtest_projection(path,artifact_reader=lambda ref:ref['manifest_uri'],limits=LIMITS)
