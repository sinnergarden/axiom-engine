"""Target-family and exact shared-math SignalPlan boundaries, entirely synthetic."""
from copy import deepcopy
import unittest

from axiom_engine.core import (ContractError, StockPredictionFrame, execute_signal_plan,
                              validate_label_spec, validate_stock_predictions)
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_signal import _ref, validate_derived_signal, signal_plan_ref
from hashlib import sha256
import json
from test_stock_prediction_v2 import staged_predictions

PARAMS=dict(group='session',unknown_group='reject',missing='skip',ddof=0,epsilon=1e-12,
            constant='missing',clip=None,excluded='missing')


def label(h=7):
    return dict(label_id='configurable-label',semantic_version='1',horizon_sessions=h,start_session_offset=1,
        end_session_offset=h,start_price='open',end_price='close',calendar_ref='sha256:'+'1'*64,
        price_basis='common_anchor_adjusted_v1',adjustment_anchor='2024-01-31',formula='close(f+h) / open(f+1) - 1',
        normalization='none',costs='none',corporate_action_policy='factor_ratio_no_separate_cashflow',
        availability='max_endpoint_price_factor_anchor_usable_from',maturity_rule='all_outcome_dependencies_strictly_before_fit_cutoff',
        missing_policy='invalid_null_preserve_grid')


def raw(h=7):
    wire=staged_predictions(); spec=label(h)
    wire.update(contract_version='stock_prediction_run_v3',signal_stage='raw_prediction',label_spec=spec,
        label_spec_ref=_ref(spec),label_normalization={'operator':'cs_zscore','operator_version':'1','params':deepcopy(PARAMS)},
        score_semantics='configured_normalized_label_prediction')
    wire.pop('signal_run_ref');wire['signal_run_ref']=Document.from_dict(wire).identity
    return wire


def typed(kind,**value): return dict(contract_type=kind,contract_version='1',metadata={},**value)


def typed_label(spec):
    h=spec['horizon_sessions']
    result=typed('LabelSpec',name=spec['label_id'],key=['security_id','session'],
        horizon_sessions=h,feature_session='actual_feature_session',formula=spec['formula'],
        return_start_rule='next_session_open',return_end_rule='horizon_session_close',
        return_start_offset_sessions=1,return_end_offset_sessions=h,price_basis=spec['price_basis'],
        benchmark_semantics='absolute_return',corporate_action_semantics=spec['corporate_action_policy'],
        normalization_policy='none',missing_delisting_policy=spec['missing_policy'],
        maturity=typed('MaturitySpec',rule=spec['maturity_rule'],lag_sessions=h,
            calendar_policy='actual_exchange_sessions',availability_rule=spec['availability']))
    result['contract_version']='2'
    return result


def plan_for(inputs):
    nodes=[]; declared=[]
    for alias,wire in inputs.items():
        declared.append(typed('SignalInput',alias=alias,label=deepcopy(wire['label_spec']),source_stage='raw_prediction',score_semantics=wire['score_semantics']))
        nodes.append(typed('SignalNode',name=alias+'_z',op='daily_zscore',inputs=[alias],input_stages=['raw_prediction'],
            output_stage='daily_zscore',weights=[],reference_universe='frozen_members',missing_policy='skip',parameters=deepcopy(PARAMS)))
    output=nodes[-1]['name']
    if len(inputs)>1:
        output='combined'; nodes.append(typed('SignalNode',name=output,op='weighted_combine',inputs=[a+'_z' for a in inputs],
            input_stages=['daily_zscore']*len(inputs),output_stage='final',weights=[1/len(inputs)]*len(inputs),
            reference_universe='frozen_members',missing_policy='propagate',parameters={}))
    return typed('SignalPlanSpec',name='configurable plan',key=['security_id','session'],inputs=declared,nodes=nodes,output=output,
        join_policy='inner_on_security_session',score_semantics='standardized_rank_score',available_time_semantics='max_parent_and_reference_dependency')


def context_for(wire):
    days=sorted({r['session'] for r in wire['rows']});first={r['session']:r for r in wire['rows']}
    return dict(calendar_ref=wire['label_spec']['calendar_ref'],reference_universe='frozen_members',reference_universe_ref='sha256:'+'2'*64,
        reference_members={day:[dict(security_id=s,member=True,available_at=day+'T20:30:00+08:00',source_refs=['sha256:'+'3'*64])
             for s in wire['universe']] for day in days},cutoff_by_session={d:first[d]['knowledge_cutoff'] for d in days},clock_basis='declared_simulation')


def execute(inputs,plan=None,context=None):
    return execute_signal_plan(plan or plan_for(inputs),{a:StockPredictionFrame.from_dict(v) for a,v in inputs.items()},
        context or context_for(next(iter(inputs.values())))).to_dict()


class StockSignalPlanTests(unittest.TestCase):
    def test_original_research_typed_plan_semantic_identity_survives_neutral_save(self):
        wire=raw(23);p=plan_for({'prediction':wire});spec=wire['label_spec']
        p['inputs'][0]['label']=typed_label(spec)
        p['metadata']={'description':'annotation outside identity'}
        p['inputs'][0]['label']['metadata']={'source':'local annotation'}
        # Independently implement the published Research semantic projection.
        def semantic(v):
            if isinstance(v,list):return [semantic(x) for x in v]
            if not isinstance(v,dict):return v
            return {k:semantic(x) for k,x in v.items() if not (v.get('contract_type') and k=='metadata')}
        reference='sha256:'+sha256(json.dumps(semantic(p),sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        out=execute({'prediction':wire},p)
        self.assertEqual(out['signal_plan_ref'],reference)
        self.assertEqual(signal_plan_ref(out['signal_plan']),reference)
        self.assertNotIn('contract_type',out['signal_plan'])
        p['inputs'][0]['label']['maturity']['lag_sessions']=22
        with self.assertRaises(ContractError):execute({'prediction':wire},p)

    def test_typed_v2_does_not_reinterpret_v1_or_expand_other_versions_and_maturity(self):
        wire=raw(3);p=plan_for({'prediction':wire});p['inputs'][0]['label']=typed_label(wire['label_spec'])
        p['inputs'][0]['label']['formula']='close(f+3) / open(f+1) - 1'
        self.assertTrue(execute({'prediction':wire},p)['rows'])
        old=deepcopy(p);declared=old['inputs'][0]['label'];declared['contract_version']='1'
        with self.assertRaisesRegex(ContractError,'v1 endpoint-distance mismatch'):execute({'prediction':wire},old)
        declared['return_end_offset_sessions']=4;declared['maturity']['lag_sessions']=4
        with self.assertRaisesRegex(ContractError,'v1 endpoint-distance target cannot map'):execute({'prediction':wire},old)
        for path in ((),('inputs',0),('nodes',0),('inputs',0,'label','maturity')):
            changed=deepcopy(p);value=changed
            for key in path:value=value[key]
            value['contract_version']='2'
            with self.subTest(version_path=path),self.assertRaisesRegex(ContractError,'Unsupported original'):
                execute({'prediction':wire},changed)
        changed=deepcopy(p);changed['inputs'][0]['label']['maturity']['rule']='all_outcome_dependencies_at_or_before_fit_cutoff'
        with self.assertRaisesRegex(ContractError,'maturity differs'):execute({'prediction':wire},changed)

    def test_outer_join_retains_missing_parent_days_and_inner_join_explicitly_intersects(self):
        a,b=raw(3),raw(17)
        # Split the raw grids over real session keys, preserving parent identities.
        from datetime import date,timedelta
        day=a['rows'][0]['session'];next_day=(date.fromisoformat(day)+timedelta(days=1)).isoformat()
        for wire in (a,b):
            wire['rows']=[r for r in wire['rows'] if r['session']==day]
            wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
        extension=[]
        for row in b['rows']:
            new=deepcopy(row);new['session']=next_day
            for name in ('knowledge_cutoff','available_at','feature_knowledge_cutoff','feature_available_at'):
                if new[name] is not None:new[name]=new[name].replace(day,next_day)
            extension.append(new)
        b['rows']+=extension;b.pop('signal_run_ref');b['signal_run_ref']=_ref(b)
        inputs={'a':a,'b':b};p=plan_for(inputs);ctx=context_for(b)
        p['join_policy']='outer_on_security_session'
        out=execute(inputs,p,ctx)
        missing=[r for r in out['rows'] if r['session']==next_day]
        self.assertEqual(len(missing),len(a['universe']));self.assertTrue(all(not r['valid'] and r['score'] is None for r in missing))
        p['join_policy']='inner_on_security_session';ctx=context_for(a)
        self.assertEqual({r['session'] for r in execute(inputs,p,ctx)['rows']},{day})
    def test_non_five_horizon_v3_and_old_v2_unchanged(self):
        old=staged_predictions();before=deepcopy(old)
        self.assertEqual(validate_stock_predictions(StockPredictionFrame.from_dict(old))[0],before)
        for h in (1,3,17,180):
            wire=raw(h);self.assertEqual(validate_stock_predictions(StockPredictionFrame.from_dict(wire))[0],wire)
        for h in (True,False,0,-1,1.0):
            with self.subTest(h=h),self.assertRaises(ContractError):validate_label_spec(label(h))

    def test_explicit_target_options_and_refs_reject(self):
        for key,value in [('end_session_offset',8),('start_price','close'),('price_basis','guessed'),('adjustment_anchor',None),
                          ('maturity_rule','ignore'),('calendar_ref',None),('normalization','implicit_zscore')]:
            wire=raw();wire['label_spec'][key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):validate_stock_predictions(StockPredictionFrame.from_dict(wire))

    def test_label_normalization_does_not_consume_raw_score_stage(self):
        wire=raw();out=execute({'prediction':wire})
        self.assertEqual(out['signal_stage'],'daily_zscore'); self.assertNotIn('model_ref',out)
        self.assertEqual(out['parent_signal_refs'],{'prediction':wire['signal_run_ref']})
        self.assertEqual(out['score_ref'],_ref(out['rows']))
        unsigned=dict(out);reference=unsigned.pop('signal_run_ref');self.assertEqual(reference,_ref(unsigned))
        validate_derived_signal(StockPredictionFrame.from_dict(out))

    def test_multiple_horizons_blend_exact_key_alignment_and_true_parents(self):
        a,b=raw(3),raw(23)
        for r in b['rows']: r['score']=-r['score'] if r['score'] is not None else None
        b.pop('signal_run_ref');b['signal_run_ref']=_ref(b)
        inputs={'short':a,'long':b};plan=plan_for(inputs)
        out=execute(inputs,plan)
        self.assertEqual(out['signal_stage'],'final')
        self.assertTrue(all(r['score']==0.0 for r in out['rows']))
        self.assertEqual(out['parent_signal_refs'],{'short':a['signal_run_ref'],'long':b['signal_run_ref']})
        changed=deepcopy(b);changed['rows'].reverse();changed.pop('signal_run_ref');changed['signal_run_ref']=_ref(changed)
        reordered=execute({'short':a,'long':changed},plan)
        self.assertEqual([r['score'] for r in out['rows']],[r['score'] for r in reordered['rows']])

    def test_reference_clock_excluded_rows_and_microseconds_are_dependencies(self):
        wire=raw();ctx=context_for(wire);day=next(iter(ctx['reference_members']))
        ctx['reference_members'][day][-1].update(member=False,available_at=day+'T20:59:59.999999+08:00')
        next(r for r in wire['rows'] if r['session']==day and r['security_id']==wire['universe'][-1])['member']=False
        wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
        out=execute({'prediction':wire},context=ctx)
        self.assertEqual(out['rows'][0]['available_at'],day+'T13:00:00Z')
        ctx['reference_members'][day][-1]['available_at']=day+'T21:00:00.000001+08:00'
        with self.assertRaisesRegex(ContractError,'Unavailable Signal reference'): execute({'prediction':wire},context=ctx)

    def test_missing_reference_constant_and_stage_policy_are_explicit(self):
        wire=raw();row=wire['rows'][0];row.update(valid=False,score=None,invalid_reason='FEATURE_MISSING')
        wire.pop('signal_run_ref');wire['signal_run_ref']=_ref(wire)
        out=execute({'prediction':wire});self.assertIsNone(out['rows'][0]['score'])
        p=plan_for({'prediction':wire});p['nodes'][0]['parameters']['missing']='propagate';p['nodes'][0]['missing_policy']='propagate'
        out=execute({'prediction':wire},p)
        self.assertTrue(all(r['score'] is None for r in out['rows'] if r['session']==row['session']))
        p['nodes'][0]['input_stages']=['daily_zscore']
        with self.assertRaisesRegex(ContractError,'stage/parent'):execute({'prediction':wire},p)

    def test_weights_units_parent_tamper_and_complete_label_match(self):
        a,b=raw(3),raw(12);inputs={'a':a,'b':b}
        p=plan_for(inputs);p['nodes'][-1]['weights']=[0.4,0.5]
        with self.assertRaises(ContractError):execute(inputs,p)
        p=plan_for(inputs);p['inputs'][0]['label']['start_price']='close'
        with self.assertRaises(ContractError):execute(inputs,p)
        b['score_unit']='percent'
        with self.assertRaises(ContractError):execute(inputs)
        b=raw(12);b['rows'][0]['score']+=1
        with self.assertRaisesRegex(ContractError,'identity mismatch'):execute({'a':a,'b':b})
