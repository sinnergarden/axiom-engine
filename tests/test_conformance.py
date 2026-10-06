"""Independent golden examples for frozen R0 capabilities, through the public ABI."""
import ast
import copy
from dataclasses import FrozenInstanceError
from datetime import date, timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import (ABI, SEMANTICS, ContractError, FactBatch, FeaturePlan,
    ExecutionContext, execute_feature_plan, required_history, unresolved, validate_plan)

REF = 'sha256:' + 'a' * 64
SOURCES = [dict(id='facts', data_ref=REF, view_ref=REF, revision_policy='fixed-data-projection-v1',
                qualification='synthetic', availability_basis='synthetic-explicit-dependency-max')]
FIXTURES = Path(__file__).parent / 'fixtures'


def col(name, unit='dimensionless', dtype='float64', stage='base', missing='preserve'):
    return dict(name=name, dtype=dtype, unit=unit, stage=stage, missing=missing)


def node(name, op, inputs, params=None, unit='dimensionless', dtype='float64', stage='base'):
    return dict(name=name, op=op, version='1', inputs=inputs, params=params or {},
                column=col(name, unit, dtype, stage))


def rolling(name, src='x', window=2, min_periods=2, reduction='mean', inclusive=True, ddof=0, missing='skip'):
    return node(name, 'rolling', [src], dict(window=window, min_periods=min_periods,
        reduction=reduction, inclusive_current=inclusive, ddof=ddof, missing=missing, ties='average'))


def cs(name, op='cs_rank', src='x', **kwargs):
    p = dict(group='session', unknown_group='missing', missing='skip', excluded='missing')
    if op == 'cs_rank': p.update(ties='average')
    elif op == 'cs_zscore': p.update(ddof=0, epsilon=1e-12, constant='zero', clip=None)
    else: p.update(lower=.01, upper=.99, interpolation='linear')
    p.update(kwargs)
    return node(name, op, [src], p, stage='cross_sectional')


def setup(values, nodes, outputs=None, members=None, industries=None, full=False, schema=None):
    """values is security -> ordered daily value vectors; no Core helper computes expectations."""
    count = len(next(iter(values.values())))
    sessions = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(count)]
    columns = schema or [col('x', stage='fact')]
    rows, refs, keys = [], [], []
    for security, vectors in values.items():
        for i, vector in enumerate(vectors):
            vs = vector if isinstance(vector, list) else [vector]
            key = [security, sessions[i]]; keys.append(key)
            rows.append(dict(security_id=security, session=sessions[i], values=vs,
                availability=[sessions[i]+'T12:00:00Z']*len(vs), sources=[['facts'] for _ in vs],
                missing_reasons=['NOT_REPORTED' if v is None else None for v in vs]))
            refs.append(dict(security_id=security, session=sessions[i], member=True if members is None else members[security][i],
                industry='industry1' if industries is None else industries[security][i],
                available_at=sessions[i]+'T00:00:00Z', source='facts'))
    selected = outputs or [nodes[-1]['name']]
    nmap = {n['name']: n for n in nodes}
    p = dict(abi=ABI, semantics=SEMANTICS, recipe_ref=REF, calendar_ref=REF, reference_ref=REF,
        reference_members={s:{r['security_id']:r['industry'] for r in refs if r['session']==s and r['member']} for s in sessions},
        input_schema=columns, event_schema={}, sources=copy.deepcopy(SOURCES), observation_domain='sessions',
        history_policy='full' if full else 'partial', nodes=nodes,
        outputs=[dict(node=n, column=copy.deepcopy(nmap[n]['column'])) for n in selected], obligations=[])
    f = dict(abi=ABI, calendar_ref=REF, schema=columns, sources=copy.deepcopy(SOURCES), rows=rows, event_schema={}, events=[])
    c = dict(abi=ABI, calendar_ref=REF, reference_ref=REF, sessions=sessions,
        cutoffs={s:s+'T23:00:00Z' for s in sessions}, history_keys=keys, output_keys=keys[:], reference=refs)
    return p, f, c


def run(p, f, c):
    return execute_feature_plan(FeaturePlan.from_dict(p), FactBatch.from_dict(f), ExecutionContext.from_dict(c)).to_dict()


def values(result, index=0):
    return [r['values'][index] for r in result['rows']]


class Conformance(unittest.TestCase):
    def assertValues(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            if e is None: self.assertIsNone(a)
            else:
                self.assertIsNotNone(a)
                self.assertTrue(math.isclose(a, e, rel_tol=1e-13, abs_tol=1e-14), (a, e))

    def test_r0_declarative_golden_cases(self):
        cases = json.loads((FIXTURES/'r0_cases.json').read_text())['cases']
        fixture = cases['grouped_shift_and_finite_rolling']
        p,f,c = setup({'A':fixture['input']['ordered_observations']},
            [node('lag','shift',['x'],{'periods':1}), rolling('mean')], ['lag','mean'])
        out=run(p,f,c)
        self.assertEqual(values(out,0), fixture['expected']['shift_1'])
        self.assertEqual(values(out,1), fixture['expected']['rolling_mean'])
        fixture=cases['explicit_reference_universe_group_transform']
        v=fixture['input']['same_session_values']; m=fixture['input']['member_mask']
        p,f,c=setup({str(i):[x] for i,x in enumerate(v)},[cs('rank')],members={str(i):[x] for i,x in enumerate(m)})
        self.assertValues(values(run(p,f,c)),fixture['expected']['percentile_rank'])
        fixture=cases['cross_sectional_quantile_winsorization']
        p,f,c=setup({'A':[0],'B':[100],'C':[10000]},[cs('winsor','cs_winsorize')],members={'A':[True],'B':[True],'C':[False]})
        self.assertValues(values(run(p,f,c)),fixture['expected']['winsorized'])
        p,f,c=setup({'A':[1],'B':[3]},[cs('z','cs_zscore')])
        self.assertValues(values(run(p,f,c)),cases['parameterized_zscore_semantics']['expected']['signal_zscore'])

    def test_projection_shuffle_duplicates_schema(self):
        p,f,c=setup({'B':[10,20],'A':[1,2]},[node('lag','shift',['x'],{'periods':1}),node('base','identity',['x'])],['base','lag'])
        out=run(p,f,c)
        f['rows'].reverse(); c['history_keys'].reverse(); c['output_keys'].reverse(); c['reference'].reverse()
        self.assertEqual(run(p,f,c)['rows'],out['rows'])
        self.assertEqual([x['name'] for x in out['schema']],['base','lag'])
        bad=copy.deepcopy(f);bad['rows'].append(bad['rows'][0])
        with self.assertRaisesRegex(ContractError,'Duplicate'):run(p,bad,c)
        for edit in ('column','dtype','unit','stage'):
            bad=copy.deepcopy(f)
            if edit=='column':bad['rows'][0]['values']=[]
            else:bad['schema'][0][edit]='wrong'
            with self.assertRaises(ContractError):run(p,bad,c)
        bad=copy.deepcopy(p);bad['nodes'][0]['column']['dtype']='float32'
        with self.assertRaises(ContractError):run(bad,f,c)

    def test_membership_reentry_does_not_compress_history(self):
        p,f,c=setup({'A':[1,10,100,1000]},[node('lag','shift',['x'],{'periods':2})],members={'A':[True,False,False,True]},full=True)
        c['output_keys']=[['A',c['sessions'][-1]]]
        self.assertValues(values(run(p,f,c)),[10])
        f['rows'].pop(1)
        with self.assertRaisesRegex(ContractError,'MISSING_HISTORY_ROW'):run(p,f,c)

    def test_sessions_vs_observations_and_missing_cells(self):
        p,f,c=setup({'A':[1,None,3]},[node('lag','shift',['x'],{'periods':1})])
        self.assertValues(values(run(p,f,c)),[None,1,None])
        f['rows'].pop(1);c['history_keys'].pop(1);c['output_keys'].pop(1);c['reference'].pop(1)
        with self.assertRaisesRegex(ContractError,'MISSING_SESSION'):run(p,f,c)
        p['observation_domain']='observations'
        self.assertValues(values(run(p,f,c)),[None,1])

    def test_756_lag_full_history_and_nested_closure(self):
        p,f,c=setup({'A':list(range(757))},[node('lag','shift',['x'],{'periods':756})],full=True)
        c['output_keys']=[['A',c['sessions'][-1]]]
        self.assertEqual(required_history(FeaturePlan.from_dict(p)),{'lag':756})
        self.assertValues(values(run(p,f,c)),[0])
        c['output_keys']=[['A',c['sessions'][-2]]]
        with self.assertRaisesRegex(ContractError,'INSUFFICIENT_HISTORY'):run(p,f,c)
        nodes=[rolling('high',window=120,min_periods=40,reduction='max'),
               node('ratio','divide',['high','x'],{'zero':'missing'}),
               node('one','constant',[],{'value':1}),node('dd','sub',['ratio','one']),
               rolling('pullback','dd',120,40,'max')]
        data=[100]*119+[200]+[100]*119
        p,f,c=setup({'A':data},nodes,full=True);c['output_keys']=[['A',c['sessions'][-1]]]
        self.assertEqual(required_history(FeaturePlan.from_dict(p)),{'pullback':238})
        self.assertValues(values(run(p,f,c)),[1])
        p,f,c=setup({'A':[1]*119},[rolling('a',window=60,min_periods=20),rolling('b','a',60,20)],full=True)
        self.assertEqual(required_history(FeaturePlan.from_dict(p)),{'b':118})
        c['output_keys']=[['A',c['sessions'][-1]]];self.assertValues(values(run(p,f,c)),[1])

    def test_window_endpoints_reductions_ddof_missing(self):
        p,f,c=setup({'A':[1,2,9]},[rolling('mean',inclusive=False)])
        self.assertValues(values(run(p,f,c)),[None,None,1.5])
        expected={'mean':2,'sum':6,'min':1,'max':3,'median':2,'rank':1,'std':math.sqrt(2/3)}
        for reduction, e in expected.items():
            p,f,c=setup({'A':[1,2,3]},[rolling('r',window=3,min_periods=3,reduction=reduction)])
            self.assertValues(values(run(p,f,c)),[None,None,e])
        p,f,c=setup({'A':[1,3]},[rolling('std',reduction='std',ddof=1)])
        self.assertValues(values(run(p,f,c)),[None,math.sqrt(2)])
        p,f,c=setup({'A':[1,None,3]},[rolling('mean',window=3,min_periods=2)])
        self.assertValues(values(run(p,f,c)),[None,None,2])
        p['nodes'][0]['params']['missing']='propagate'
        self.assertValues(values(run(p,f,c)),[None,None,None])
        p,f,c=setup({'A':[1,1,3]},[rolling('r',window=3,min_periods=2,reduction='rank')])
        self.assertValues(values(run(p,f,c)),[None,.75,1])

    def test_arithmetic_boolean_and_legal_missing(self):
        nodes=[node('zero','constant',[],{'value':0}),node('d','divide',['x','zero'],{'zero':'missing'}),
               node('flag','gt',['x','zero'],{'missing':'false'},dtype='bool'),
               node('flagf','to_float',['flag']),node('filled','fill',['x'],{'value':.5})]
        p,f,c=setup({'A':[None,2]},nodes,['d','flagf','filled'])
        self.assertValues(values(run(p,f,c),0),[None,None])
        self.assertValues(values(run(p,f,c),1),[0,1])
        self.assertValues(values(run(p,f,c),2),[.5,2])
        p['nodes'][1]['params']['zero']='reject'
        with self.assertRaisesRegex(ContractError,'ZERO_DENOMINATOR'):run(p,f,c)
        p,f,c=setup({'A':[None,2]},[node('z','constant',[],{'value':0}),node('flag','gt',['x','z'],{'missing':'preserve'},dtype='bool'),
            node('choose','where',['flag','x','z'],{'missing':'false'})])
        self.assertValues(values(run(p,f,c)),[0,2])
        p['nodes'][1]['params']['missing']='reject'
        with self.assertRaisesRegex(ContractError,'MISSING_BOOLEAN'):run(p,f,c)
        p,f,c=setup({'A':[1,None,3]},[node('ret','pct_change',['x'],{'periods':1,'fill_method':'none','zero':'missing'})])
        self.assertValues(values(run(p,f,c)),[None,None,None])
        p['nodes'][0]['params']['fill_method']='legacy_default'
        with self.assertRaises(ContractError):run(p,f,c)

    def test_single_output_full_reference_and_industry(self):
        p,f,c=setup({'A':[1],'B':[1],'C':[3],'Z':[10000]},[cs('rank')],members={'A':[True],'B':[True],'C':[True],'Z':[False]})
        c['output_keys']=[['A',c['sessions'][0]]]
        self.assertValues(values(run(p,f,c)),[.5])
        f['rows'][-1]['values']=[-1e20]
        self.assertValues(values(run(p,f,c)),[.5])
        missing_f=copy.deepcopy(f);missing_c=copy.deepcopy(c)
        missing_f['rows']=[r for r in f['rows'] if r['security_id']!='B']
        for field in ('history_keys','output_keys'):
            missing_c[field]=[k for k in c[field] if k[0]!='B']
        missing_c['reference']=[r for r in c['reference'] if r['security_id']!='B']
        with self.assertRaisesRegex(ContractError,'INCOMPLETE_REFERENCE'):run(p,missing_f,missing_c)
        f['rows'].pop(1)
        with self.assertRaises(ContractError):run(p,f,c)
        p,f,c=setup({'A':[1],'B':[3],'C':[8]},[cs('r',group='industry')],industries={'A':['x'],'B':['x'],'C':[None]})
        self.assertValues(values(run(p,f,c)),[.5,1,None])
        p['nodes'][0]['params']['unknown_group']='reject'
        with self.assertRaisesRegex(ContractError,'UNKNOWN_INDUSTRY'):run(p,f,c)

    def test_cs_constants_prefill_scale_clip_and_empty(self):
        p,f,c=setup({'A':[5],'B':[5]},[cs('z','cs_zscore')])
        self.assertValues(values(run(p,f,c)),[0,0])
        p['nodes'][0]['params']['constant']='missing'
        self.assertValues(values(run(p,f,c)),[None,None])
        p['nodes'][0]['params']['constant']='reject'
        with self.assertRaisesRegex(ContractError,'UNDEFINED_CS_SCALE'):run(p,f,c)
        for policy, expected in [('skip',[None,0]),('fill_zero',[-1,1]),('propagate',[None,None])]:
            p,f,c=setup({'A':[None],'B':[2]},[cs('z','cs_zscore',missing=policy)])
            self.assertValues(values(run(p,f,c)),expected)
        p['nodes'][0]['params']['missing']='reject'
        with self.assertRaises(ContractError):run(p,f,c)
        p,f,c=setup({'A':[1],'B':[3]},[cs('z','cs_zscore',ddof=1,clip=[-.5,.5])])
        self.assertValues(values(run(p,f,c)),[-.5,.5])
        p,f,c=setup({'A':[1],'B':[2]},[cs('z','cs_zscore',group='industry',ddof=1,excluded='zero_if_undefined')],members={'A':[True],'B':[False]})
        self.assertValues(values(run(p,f,c)),[0,0])
        p,f,c=setup({'A':[1]},[cs('rank')],members={'A':[False]})
        self.assertValues(values(run(p,f,c)),[None])
        p,f,c=setup({'A':[1e-17],'B':[3e-17]},[cs('w','cs_winsorize')])
        for actual, expected in zip(values(run(p,f,c)), [1.02e-17,2.98e-17]):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-14, abs_tol=0))

    def test_event_strict_exact_future_and_availability(self):
        p,f,c=setup({'A':[0,0,0]},[node('event','asof',[],{'stream':'income','field':'value','match':'exact_date','report_policy':'nondecreasing'})])
        columns=[col('value',stage='fact')];p['event_schema']={'income':columns};f['event_schema']=copy.deepcopy(p['event_schema'])
        def event(eid,day,value,available):
            return dict(stream='income',security_id='A',event_id=eid,event_session=day,report_period='2019-12-31',values=[value],availability=[available],sources=[['facts']],missing_reasons=[None])
        day=c['sessions'][1]
        f['events']=[event('one',day,10,day+'T12:00:00Z')]
        self.assertValues(values(run(p,f,c)),[None,10,10])
        p['nodes'][0]['params']['match']='strict_before'
        self.assertValues(values(run(p,f,c)),[None,None,10])
        f['events'].append(event('future','2021-01-01',999,'2021-01-01T00:00:00Z'))
        self.assertValues(values(run(p,f,c)),[None,None,10])
        f['events'][0]['availability'][0]=c['sessions'][2]+'T12:00:00Z'
        self.assertValues(values(run(p,f,c)),[None,None,None])
        p['nodes'][0]['params']['match']='exact_date'
        self.assertValues(values(run(p,f,c)),[None,None,10])
        regressed=copy.deepcopy(f);regressed['events'][-1]['report_period']='2018-12-31'
        with self.assertRaisesRegex(ContractError,'REPORT_PERIOD_REGRESSION'):run(p,regressed,c)
        f['events'].append(copy.deepcopy(f['events'][0]));f['events'][-1]['event_id']='ambiguous'
        with self.assertRaisesRegex(ContractError,'Ambiguous'):run(p,f,c)

    def test_late_fact_not_authorized_by_final_cutoff_or_fill(self):
        p,f,c=setup({'A':[1,2,3]},[node('fill','fill',['x'],{'value':0}),rolling('mean','fill',2,1)],['fill','mean'])
        f['rows'][0]['availability'][0]=c['sessions'][2]+'T12:00:00Z'
        out=run(p,f,c)
        self.assertValues(values(out,0),[None,2,3])
        self.assertValues(values(out,1),[None,None,2.5])
        self.assertIn('UNAVAILABLE_AT_SESSION_CUTOFF',out['rows'][0]['reasons'][0])
        self.assertEqual(out['source_bindings'],p['sources'])
        bad=copy.deepcopy(f);bad['sources'][0]['qualification']='verified'
        with self.assertRaisesRegex(ContractError,'qualification'):run(p,bad,c)

    def test_full_history_includes_past_reference_members(self):
        nodes=[node('lag','shift',['x'],{'periods':1}),cs('rank',src='lag'),
               node('previous_rank','shift',['rank'],{'periods':1},stage='cross_sectional')]
        p,f,c=setup({'A':[1,2,3],'B':[2,4,6]},nodes,members={'A':[True]*3,'B':[False,True,False]},full=True)
        first=c['sessions'][0]
        f['rows']=[r for r in f['rows'] if not (r['security_id']=='B' and r['session']==first)]
        c['history_keys']=[k for k in c['history_keys'] if k!=['B',first]]
        c['reference']=[r for r in c['reference'] if not (r['security_id']=='B' and r['session']==first)]
        c['output_keys']=[['A',c['sessions'][-1]]]
        with self.assertRaisesRegex(ContractError,'INSUFFICIENT_HISTORY'):run(p,f,c)
        p['history_policy']='partial'
        self.assertValues(values(run(p,f,c)),[1])

    def test_remaining_primitives_and_rejections(self):
        nodes=[node('zero','constant',[],{'value':0}),node('abs','abs',['x']),
               node('clipped','clip',['x'],{'lower':0,'upper':3}),
               node('log','log1p',['clipped'],{'domain':'missing'}),
               node('flag','gt',['x','zero'],{'missing':'false'},dtype='bool'),
               node('inverse','not',['flag'],{'missing':'preserve'},dtype='bool'),
               node('either','or',['flag','inverse'],{'missing':'preserve'},dtype='bool'),
               node('both','and',['flag','inverse'],{'missing':'preserve'},dtype='bool'),
               node('times','mul',['x','abs']),node('sum','add',['x','abs']),
               node('missing','is_missing',['x'],dtype='bool')]
        p,f,c=setup({'A':[-2,None,4]},nodes,['abs','log','either','both','times','sum','missing'])
        out=run(p,f,c)
        for i,e in enumerate([[2,None,4],[0,None,math.log(4)],[True,True,True],[False,False,False],[-4,None,16],[0,None,8],[False,True,False]]):
            self.assertValues(values(out,i),e)
        schema=[col('d1',unit='calendar_date',dtype='date',stage='fact'),col('d2',unit='calendar_date',dtype='date',stage='fact')]
        p,f,c=setup({'A':[['2020-03-01','2020-02-28'],['2020-02-28','2020-03-01']]},[node('age','calendar_age',['d1','d2'],unit='days')],schema=schema)
        self.assertValues(values(run(p,f,c)),[2,0])
        p,f,c=setup({'A':[-2,3]},[node('log','log1p',['x'],{'domain':'missing'})])
        self.assertValues(values(run(p,f,c)),[None,math.log(4)])
        p['nodes'][0]['params']['domain']='reject'
        with self.assertRaisesRegex(ContractError,'LOG_DOMAIN'):run(p,f,c)
        p,f,c=setup({'A':[1]},[rolling('m')])
        del p['nodes'][0]['params']['min_periods']
        with self.assertRaises(ContractError):run(p,f,c)
        p,f,c=setup({'A':[1]},[node('x2','identity',['x'])])
        c['cutoffs']={}
        with self.assertRaises(ContractError):run(p,f,c)
        p,f,c=setup({'A':[1]},[node('x2','identity',['x'])])
        c['reference'][0]['available_at']='2021-01-01T00:00:00Z'
        with self.assertRaises(ContractError):run(p,f,c)
        p,f,c=setup({'A':[1]},[node('x2','identity',['x'])])
        p['nodes'][0]['column']['stage']='cross_sectional'
        with self.assertRaises(ContractError):run(p,f,c)

    def test_batch_and_single_session_same_entrypoint(self):
        p,f,c=setup({'A':[1,2,3,4],'B':[4,2,1,5]},[rolling('mean'),cs('r',src='mean')])
        batch=run(p,f,c)
        for s in c['sessions']:
            daily_f=copy.deepcopy(f);daily_c=copy.deepcopy(c)
            daily_f['rows']=[r for r in f['rows'] if r['session']<=s]
            daily_c['sessions']=[d for d in c['sessions'] if d<=s]
            daily_c['cutoffs']={d:t for d,t in c['cutoffs'].items() if d<=s}
            daily_c['history_keys']=[k for k in c['history_keys'] if k[1]<=s]
            daily_c['reference']=[r for r in c['reference'] if r['session']<=s]
            daily_c['output_keys']=[k for k in c['output_keys'] if k[1]==s]
            daily=run(p,daily_f,daily_c)
            self.assertEqual(daily['rows'],[r for r in batch['rows'] if r['session']==s])
            self.assertEqual(daily['schema'],batch['schema'])

    def test_unknown_transport_admission_and_versions(self):
        unknowns=json.loads((FIXTURES/'r0_unknowns.json').read_text())
        self.assertEqual(len(unknowns),24)
        p,f,c=setup({'A':[1]},[node('base','identity',['x'])])
        for unknown in unknowns:
            d=copy.deepcopy(p);d['obligations']=[{'nested':[unknown]}]
            frozen=FeaturePlan.from_dict(d)
            self.assertEqual(unresolved(FeaturePlan(frozen.payload)),(unknown,))
            validate_plan(frozen)
            with self.assertRaisesRegex(ContractError,'UNRESOLVED'):execute_feature_plan(frozen,FactBatch.from_dict(f),ExecutionContext.from_dict(c))
        d=copy.deepcopy(p);d['nodes'][0]['params']={'nested':{'metadata':[unknowns[0]]}}
        with self.assertRaisesRegex(ContractError,'UNRESOLVED'):run(d,f,c)
        for edit in ('abi','op','version','policy'):
            d=copy.deepcopy(p)
            if edit=='abi':d['abi']='future'
            elif edit=='policy':d['nodes'][0]['params']={'allow_unknown':True}
            else:d['nodes'][0][edit]='future'
            with self.assertRaises(ContractError):run(d,f,c)
        with self.assertRaises(ContractError):FeaturePlan('{"a":1,"a":2}')
        with self.assertRaises(ContractError):FeaturePlan.from_dict({'nested':{'contract_type':'Plugin','value':'x'}})
        with self.assertRaises(ContractError):FactBatch.from_dict({'value':float('nan')})
        with self.assertRaises(ContractError):FactBatch.from_dict({'value':float('inf')})

    def test_freezing_identity_and_no_mutable_plugin_or_io(self):
        p,f,c=setup({'A':[1,3]},[rolling('mean')])
        frozen=FeaturePlan.from_dict(p);original=frozen.identity
        p['nodes'][0]['params']['window']=90
        self.assertEqual(frozen.identity,original)
        changed=frozen.to_dict();changed['nodes'][0]['params']['window']=3
        self.assertNotEqual(FeaturePlan.from_dict(changed).identity,original)
        with self.assertRaises(FrozenInstanceError):frozen.payload='x'
        facts=FactBatch.from_dict(f);ctx=ExecutionContext.from_dict(c)
        expected=execute_feature_plan(frozen,facts,ctx)
        with patch('builtins.open',side_effect=AssertionError('I/O')), patch('socket.socket',side_effect=AssertionError('network')):
            self.assertEqual(execute_feature_plan(frozen,facts,ctx),expected)
        package_root=Path(__file__).parents[1]/'src'
        source=package_root/'axiom_engine'/'core'
        allowed={'dataclasses','datetime','decimal','hashlib','json','math','re','statistics'}
        for path in source.rglob('*.py'):
            tree=ast.parse(path.read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):
                    for alias in n.names:self.assertIn(alias.name.split('.')[0],allowed | ({'struct','sys'} if path.name=='cs_batch.py' else set()))
                if isinstance(n,ast.ImportFrom) and n.level==0:self.assertIn(n.module.split('.')[0],allowed)
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                    self.assertNotIn(n.func.id,{'eval','exec','open','__import__'})
        # An unimportable/malicious plugin is rejected as data; no loader exists.
        bad=frozen.to_dict();bad['nodes'][0]['op']='plugin';bad['nodes'][0]['params']={'module':'axiom_research.evil'}
        with self.assertRaises(ContractError):run(bad,f,c)
        code='''import json,sys\nfrom axiom_engine.core import *\np,f,c=json.loads(sys.stdin.read())\nprint(execute_feature_plan(FeaturePlan.from_dict(p),FactBatch.from_dict(f),ExecutionContext.from_dict(c)).payload)'''
        import os
        env={**os.environ,'PYTHONPATH':str(package_root.resolve())}
        with tempfile.TemporaryDirectory() as cwd:
            Path(cwd,'latest').write_text('unrelated mutable pointer')
            r=subprocess.run([sys.executable,'-c',code],input=json.dumps([frozen.to_dict(),f,c]),text=True,capture_output=True,cwd=cwd,env=env,check=True)
        self.assertEqual(json.loads(r.stdout),expected.to_dict())


if __name__=='__main__':unittest.main()
