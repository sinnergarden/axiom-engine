"""Saved v3/derived integration through the sole stock owner and account loop."""
from copy import deepcopy
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

from axiom_engine.core import StockPredictionFrame, execute_signal_plan, ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_signal import _ref
from axiom_engine.runtime import load_stock_backtest_projection
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_stock_signal_plan import label, plan_for, PARAMS
from test_stock_owned_inputs import request_for, run_owned
from test_stock_market_owner import market_for, bind, seal_request
from test_stock_stream_inputs import artifact_file, LIMITS
from test_stock_stream import execute


def v3_request(root, *, h=17):
    plan=request_for(root);plan['prediction_input']['contract_version']='stock_prediction_input_refs_v2'
    for i,item in enumerate(plan['prediction_input']['frames']):
        item['kind']='raw'
        model=Document(Path(item['model_metadata_artifact']['manifest_uri']).read_text()).to_dict()
        spec=label(h);spec['calendar_ref']=_ref(plan['scope']['calendar'])
        old_model=model['model_ref']
        model.update(contract_version='stock_model_release_v3',label_spec=spec,label_spec_ref=_ref(spec),
            label_normalization=dict(operator='cs_zscore',operator_version='1',params=deepcopy(PARAMS)),
            target_semantics='configured_normalized_label_prediction',ordered_features=['a','b','c'])
        model['parameters']['objective']='regression'
        model.pop('model_ref');model['model_ref']=_ref(model)
        wire=Document(Path(item['prediction_artifact']['manifest_uri']).read_text()).to_dict()
        wire.update(contract_version='stock_prediction_run_v3',signal_stage='raw_prediction',label_spec=spec,
            label_spec_ref=_ref(spec),label_normalization=deepcopy(model['label_normalization']),
            score_semantics=model['target_semantics'],model_ref=model['model_ref'])
        for row in wire['rows']:
            row['source_refs']=[model['model_ref'] if r==old_model else r for r in row['source_refs']]
        wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
        item.update(model_ref=model['model_ref'],signal_run_ref=wire['signal_run_ref'],
            model_metadata_artifact=artifact_file(root,'v3-model-'+str(i),model),
            prediction_artifact=artifact_file(root,'v3-prediction-'+str(i),wire))
    return seal_request(plan)


def derived_request(root, original):
    plan=deepcopy(original);plan['account_id']+='-derived'
    frames=[]
    for i,parent in enumerate(original['prediction_input']['frames']):
        wire=Document(Path(parent['prediction_artifact']['manifest_uri']).read_text()).to_dict()
        days=sorted({r['session'] for r in wire['rows']});members={};membership_ref=None
        for entry in plan['market_input']['native_inputs']:
            if entry['role']!='execution':continue
            batch=Document(Path(entry['artifact']['manifest_uri']).read_text()).to_dict()
            if batch['context']['domain']!='universe_membership':continue
            membership_ref=entry['native_ref']
            facts={(r['session'],r['security_id']):r for r in batch['field_meta']['is_member']['by_key']}
            for day in days:
                members[day]=[dict(security_id=r['security_id'],member=r['is_member'],
                    available_at=facts[day,r['security_id']]['usable_from'],source_refs=[membership_ref])
                    for r in batch['records'] if r['session']==day]
        context=dict(calendar_ref=wire['label_spec']['calendar_ref'],reference_universe='frozen_members',
            reference_universe_ref=membership_ref,reference_members=members,
            cutoff_by_session={d:next(r['knowledge_cutoff'] for r in wire['rows'] if r['session']==d) for d in days},
            clock_basis='declared_simulation')
        signal=execute_signal_plan(plan_for({'prediction':wire}),{'prediction':StockPredictionFrame.from_dict(wire)},context).to_dict()
        frames.append(dict(kind='derived',**{k:signal[k] for k in ('signal_run_ref','signal_plan_ref','score_ref','implementation_ref','signal_stage')},
            signal_artifact=artifact_file(root,'derived-'+str(i),signal),parent_inputs={'prediction':deepcopy(parent)}))
    plan['prediction_input']['frames']=frames
    return seal_request(plan)


class StockSignalInputTests(unittest.TestCase):
    def test_raw_horizon_and_feature_width_use_existing_account_and_saved_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=v3_request(root,h=17)
            with market_for(plan) as market,bind(market,plan) as inputs:
                with patch('axiom_engine.runtime.stock_stream_inputs._CanonicalIndex',side_effect=AssertionError('source reopened')):
                    run,projection=run_owned(root/'out',plan,inputs)
            self.assertTrue(projection.rows['decisions'])
            clock=projection.rows['decisions'][0]['prediction_clock']
            self.assertEqual(clock['kind'],'raw');self.assertEqual(clock['signal_stage'],'raw_prediction')
            self.assertEqual(clock['model_ref'],plan['prediction_input']['frames'][0]['model_ref'])

    def test_derived_same_account_math_and_top3_top5_keep_true_signal_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=v3_request(root);plan=derived_request(root,raw)
            with market_for(raw) as market,bind(market,plan) as inputs:
                with patch('axiom_engine.core.stock_signal._execute_signal_plan_admitted',side_effect=AssertionError('Signal plan executed during account')):
                    a,projection=run_owned(root/'three',plan,inputs)
                five=deepcopy(plan);five['account_id']+='-five';five['portfolio_policy']['top_k']=5;seal_request(five)
                b,_=run_owned(root/'five',five,inputs)
                self.assertNotEqual(a['final_account'],b['final_account'])
                self.assertEqual(a['signal_ref'],b['signal_ref'])
            clock=projection.rows['decisions'][0]['prediction_clock']
            self.assertEqual(clock['kind'],'derived');self.assertNotIn('model_ref',clock)
            self.assertEqual(clock['parent_signal_refs'],{'prediction':raw['prediction_input']['frames'][0]['signal_run_ref']})
            self.assertEqual(a['source_audit']['counts']['prediction_rows'],
                2*(len(plan['scope']['calendar'])-1)*len(plan['scope']['prediction_universe']))
            # Ordinary source path uses the same admission and saved projection.
            shutil.rmtree(root/'three')
            expected,expected_projection,_=execute(root/'three',plan)
            self.assertEqual(a,expected);self.assertEqual(projection.rows,expected_projection.rows)

    def test_saved_derived_trace_refs_are_checked_without_reopening_signal_parents(self):
        from test_stock_stream_projection import mutate_part
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=v3_request(root);plan=derived_request(root,raw)
            with market_for(raw) as market,bind(market,plan) as inputs:
                wire,projection=run_owned(root/'out',plan,inputs)
            # Saved loading can work after every original Signal parent is gone.
            for item in plan['prediction_input']['frames']:
                Path(item['signal_artifact']['manifest_uri']).unlink()
                for parent in item['parent_inputs'].values():
                    for name in ('fold_spec_artifact','model_metadata_artifact','prediction_artifact'):
                        Path(parent[name]['manifest_uri']).unlink()
            path=root/'out'/'run.json'
            self.assertEqual(load_stock_backtest_projection(path,artifact_reader=lambda ref:ref['manifest_uri'],limits=LIMITS).rows,projection.rows)
            index=next(i for i,item in enumerate(wire['result_parts']) if 'decisions' in item['row_counts'] and item['row_counts']['decisions'])
            mutate_part(path,wire,index,lambda part:part['rows']['decisions'][0]['prediction_clock'].update(score_ref='sha256:'+'4'*64))
            with self.assertRaisesRegex(ContractError,'Derived parent metadata'):
                load_stock_backtest_projection(path,artifact_reader=lambda ref:ref['manifest_uri'],limits=LIMITS)

    def test_derived_and_parent_artifact_locations_do_not_change_logical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=v3_request(root);plan=derived_request(root,raw)
            original=plan['prediction_input']['prediction_ref']
            for item in plan['prediction_input']['frames']:
                item['signal_artifact']['manifest_uri']='/relocated/derived.json'
                for parent in item['parent_inputs'].values():
                    for name in ('fold_spec_artifact','model_metadata_artifact','prediction_artifact'):
                        parent[name]['manifest_uri']='/relocated/'+name+'.json'
            self.assertEqual(logical_ref(plan['prediction_input'],'prediction_ref'),original)

    def test_resigned_derived_score_and_reference_fabrication_reject_before_account(self):
        for mutation in ('score','membership_clock','membership_ref'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=v3_request(root);plan=derived_request(root,raw)
                item=plan['prediction_input']['frames'][0]
                wire=Document(Path(item['signal_artifact']['manifest_uri']).read_text()).to_dict()
                if mutation=='score': wire['rows'][0]['score']+=0.01
                else:
                    context=wire['context'];day=next(iter(context['reference_members']))
                    if mutation=='membership_clock':context['reference_members'][day][0]['available_at']=day+'T19:00:00+08:00'
                    else:context['reference_members'][day][0]['source_refs']=['sha256:'+'4'*64]
                    parent=item['parent_inputs']['prediction'];prediction=Document(Path(parent['prediction_artifact']['manifest_uri']).read_text()).to_dict()
                    wire=execute_signal_plan(wire['signal_plan'],{'prediction':StockPredictionFrame.from_dict(prediction)},context).to_dict()
                wire['score_ref']=_ref(wire['rows']);wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
                item.update({k:wire[k] for k in ('signal_run_ref','signal_plan_ref','score_ref','implementation_ref','signal_stage')});item['signal_artifact']=artifact_file(root,'tampered-derived',wire);seal_request(plan)
                with market_for(raw) as market,patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('account started')):
                    with self.assertRaises(ContractError):bind(market,plan)

    def test_parent_rows_and_artifacts_are_counted_in_resource_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=v3_request(root);plan=derived_request(root,raw)
            with market_for(raw) as market:
                count=(len(plan['scope']['calendar'])-1)*len(plan['scope']['prediction_universe'])
                with self.assertRaises(ContractError):bind(market,plan,limits={**LIMITS,'max_prediction_rows':count})

    def test_model_target_spec_and_normalization_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=v3_request(root);item=plan['prediction_input']['frames'][0]
            wire=Document(Path(item['prediction_artifact']['manifest_uri']).read_text()).to_dict()
            wire['label_normalization']={'operator':'identity','operator_version':'1','params':{}}
            wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
            item['signal_run_ref']=wire['signal_run_ref'];item['prediction_artifact']=artifact_file(root,'wrong-normalization',wire);seal_request(plan)
            with market_for(plan) as market,self.assertRaises(ContractError):bind(market,plan)
