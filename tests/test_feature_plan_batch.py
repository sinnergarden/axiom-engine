"""Original-Frame equivalence, real helper reuse, and bounded failure controls."""
import copy
import hashlib
import math
import unittest
from unittest.mock import patch

from axiom_engine.core import (ContractError, ExecutionContext, FactBatch, FeaturePlan,
                               execute_feature_plan, execute_feature_plan_batch)
from axiom_engine.core import execution, feature_batch
from test_conformance import cs, node, rolling, setup


def docs(p,f,c):
    return FeaturePlan.from_dict(p), FactBatch.from_dict(f), ExecutionContext.from_dict(c)


def requests_for(base, *, history, view_count, provenance=True, transform=None):
    requests=[]
    for index in range(view_count):
        p,f,c=copy.deepcopy(base)
        days=c['sessions'][index:index+history]; day=days[-1]
        p['reference_members']={d:p['reference_members'][d] for d in days}
        c.update(sessions=days,cutoffs={d:day+'T23:00:00Z' for d in days})
        c['history_keys']=[k for k in c['history_keys'] if k[1] in days]
        c['output_keys']=[[s,day] for s in sorted({k[0] for k in c['history_keys']})]
        c['reference']=[r for r in c['reference'] if r['session'] in days]
        f['rows']=[r for r in f['rows'] if r['session'] in days]
        if provenance:
            source='original-view-'+str(index)
            p['sources'][0].update(id=source,view_ref='sha256:'+hashlib.sha256(source.encode()).hexdigest())
            f['sources']=copy.deepcopy(p['sources'])
            for row in f['rows']:
                row.update(availability=[day+'T12:00:00Z']*len(row['values']),sources=[[source] for _ in row['values']])
            for row in c['reference']:row.update(available_at=day+'T08:00:00Z',source=source)
        if transform:transform(index,p,f,c)
        requests.append(docs(p,f,c))
    return tuple(requests)


def rolling_views(*, width=4, window=64, tail=4, count=2, transform=None):
    total=window+tail+count-2
    data={f'S{i:03d}':[(j*0.01234567+i*0.056789+1)/(1+j%3) for j in range(total)] for i in range(width)}
    nodes=[rolling('std',window=window,min_periods=window,reduction='std'),
           rolling('mean','std',tail,tail),cs('z','cs_zscore',src='mean')]
    return requests_for(setup(data,nodes),history=window+tail-1,view_count=count,transform=transform)


class FeaturePlanBatch(unittest.TestCase):
    def assertExact(self,requests,budget):
        expected=tuple(execute_feature_plan(*r) for r in requests)
        result=execute_feature_plan_batch(requests,reuse_budget_bytes=budget)
        self.assertIs(type(result['frames']),tuple)
        self.assertEqual(set(result),{'frames','stats'})
        self.assertEqual(tuple(f.payload for f in result['frames']),tuple(f.payload for f in expected))
        self.assertEqual(tuple(f.identity for f in result['frames']),tuple(f.identity for f in expected))
        self.assertEqual(result['stats']['views'],len(requests))
        self.assertEqual(result['stats']['retained_reuse_bytes'],0)
        self.assertLessEqual(result['stats']['peak_reuse_bytes'],budget)
        for count in result['stats']['helpers'].values():
            self.assertEqual(count['requested'],count['computed']+count['reused'])
        return result

    def test_original_anchor_provenance_and_real_rolling_helper_reuse(self):
        requests=rolling_views()
        before=tuple(tuple(d.payload for d in r) for r in requests)
        self.assertExact(requests,0)
        with patch.object(execution,'_std',wraps=execution._std) as calls:
            result=execute_feature_plan_batch(requests,reuse_budget_bytes=2**20)
        counts=result['stats']['helpers']
        self.assertEqual(counts['rolling.std']['requested'],32)
        self.assertEqual(counts['rolling.std']['computed'],20)
        self.assertEqual(counts['rolling.std']['reused'],12)
        self.assertEqual(calls.call_count,sum(c['computed'] for c in counts.values()))
        self.assertEqual(before,tuple(tuple(d.payload for d in r) for r in requests))
        self.assertEqual(tuple(f.payload for f in result['frames']),tuple(execute_feature_plan(*r).payload for r in requests))
        self.assertNotEqual(requests[0][1].to_dict()['sources'],requests[1][1].to_dict()['sources'])
        self.assertNotEqual(result['frames'][0].to_dict()['rows'][0]['availability'],result['frames'][1].to_dict()['rows'][0]['availability'])

    def test_cs_scale_reuses_complete_ordered_cohort_with_new_clocks_sources(self):
        # Shift needs two historical CS groups per view; the middle group is
        # shared numerically, while its clocks/source bindings remain per-view.
        data={f'S{i:03d}':[i*0.01234567+j*0.7654321 for j in range(3)] for i in range(70)}
        nodes=[cs('z','cs_zscore'),node('lag','shift',['z'],{'periods':1},stage='cross_sectional'),
               node('out','add',['z','lag'],stage='cross_sectional')]
        requests=requests_for(setup(data,nodes),history=2,view_count=2)
        with patch.object(execution,'_cs_zscore_scale',wraps=execution._cs_zscore_scale) as calls:
            result=execute_feature_plan_batch(requests,reuse_budget_bytes=2**20)
        self.assertEqual(result['stats']['helpers']['cs_zscore.scale']['reused'],1)
        self.assertEqual(calls.call_count,3)
        self.assertEqual(tuple(f.payload for f in result['frames']),tuple(execute_feature_plan(*r).payload for r in requests))
        def member_change(index,p,f,c):
            if index:
                p['reference_members'][c['sessions'][0]].pop('S000')
                next(r for r in c['reference'] if r['session']==c['sessions'][0] and r['security_id']=='S000')['member']=False
        changed=requests_for(setup(data,nodes),history=2,view_count=2,transform=member_change)
        self.assertEqual(self.assertExact(changed,2**20)['stats']['helpers']['cs_zscore.scale']['reused'],0)
        def reorder(index,p,f,c):
            if index:c['reference'].reverse()
        changed=requests_for(setup(data,nodes),history=2,view_count=2,transform=reorder)
        self.assertEqual(self.assertExact(changed,2**20)['stats']['helpers']['cs_zscore.scale']['reused'],0)

    def test_anchor_values_revision_nulls_and_future_visibility_do_not_reuse(self):
        def anchor(index,p,f,c):
            if index:
                for row in f['rows']:row['values'][0]*=1.25
        def revision(index,p,f,c):
            if index:
                for row in f['rows']:
                    if row['session']==c['sessions'][30]:row['values'][0]+=0.25
        for transform in (anchor,revision):
            with self.subTest(transform=transform.__name__):
                result=self.assertExact(rolling_views(transform=transform),2**20)
                self.assertEqual(result['stats']['helpers']['rolling.std']['reused'],0)
        def null(index,p,f,c):
            if index:
                for row in f['rows']:
                    if row['session']==c['sessions'][30]:row.update(values=[None],missing_reasons=['RESTORED_LATER'])
        self.assertExact(rolling_views(transform=null),2**20)
        data={f'S{i:03d}':[float(j+i) for j in range(66)] for i in range(3)}
        nodes=[rolling('std',window=64,min_periods=1,reduction='std'),rolling('mean','std',2,1)]
        def future(index,p,f,c):
            target=base[2]['sessions'][31]; available=base[2]['sessions'][-1]+'T12:00:00Z'
            for row in f['rows']:
                if row['session']==target:row['availability']=[available]
        base=setup(data,nodes)
        result=self.assertExact(requests_for(base,history=65,view_count=2,transform=future),2**20)
        self.assertEqual(result['stats']['helpers']['rolling.std']['reused'],0)
        self.assertIn('UNAVAILABLE_AT_SESSION_CUTOFF',result['frames'][0].to_dict()['rows'][0]['reasons'][0])

    def test_signed_zero_dependencies_and_no_hit_disable(self):
        def scale(index,p,f,c):
            if index:
                for row in f['rows']:row['values'][0]*=1.25
        result=self.assertExact(rolling_views(count=3,transform=scale),2**20)
        count=result['stats']['helpers']['rolling.std']
        self.assertEqual(count['reused'],0)
        self.assertGreater(count['disabled'],0)
        self.assertEqual(result['stats']['disabled_ops']['rolling.std'],'no_hits')
        base=setup({'A':[float(i) for i in range(66)]},
                   [rolling('std',window=64,min_periods=64,reduction='std'),rolling('mean','std',2,2)])
        for row in base[1]['rows']:
            if row['session']==base[2]['sessions'][31]:row['values']=[-0.0]
            if row['session']==base[2]['sessions'][32]:row['values']=[0.0]
        def swap(index,p,f,c):
            if index:
                for row in f['rows']:
                    if row['session']==base[2]['sessions'][31]:row['values']=[0.0]
                    if row['session']==base[2]['sessions'][32]:row['values']=[-0.0]
        result=self.assertExact(requests_for(base,history=65,view_count=2,transform=swap),2**20)
        self.assertEqual(result['stats']['helpers']['rolling.std']['reused'],0)

    def test_zero_tiny_and_eviction_budgets_and_singleton_shadow(self):
        requests=rolling_views(width=6,count=3)
        for budget in (0,1,64,128,250000,2**20):
            with self.subTest(budget=budget):
                result=self.assertExact(requests,budget)
                if budget==0:
                    self.assertEqual(result['stats']['key_attempts'],0)
                    self.assertEqual(result['stats']['key_build_ns'],0)
                if budget<1024:self.assertEqual(result['stats']['keys_built'],0)
        result=self.assertExact(requests,250000)
        self.assertGreater(result['stats']['evictions'],0)
        for request in requests:
            self.assertEqual(self.assertExact((request,),2**20)['frames'][0].payload,execute_feature_plan(*request).payload)

    def test_key_budget_before_large_unicode_key_build(self):
        name='證券😀'*300
        base=setup({name:[float(i) for i in range(65)]},[rolling('std',window=64,min_periods=64,reduction='std')])
        requests=requests_for(base,history=64,view_count=2)
        result=execute_feature_plan_batch(requests,reuse_budget_bytes=100000)
        self.assertEqual(result['stats']['keys_built'],0)
        self.assertGreater(result['stats']['budget_fallbacks'],0)
        self.assertEqual(tuple(f.payload for f in result['frames']),tuple(execute_feature_plan(*r).payload for r in requests))

    def test_no_global_cache_no_cell_results_and_error_cleanup(self):
        captured=[]
        original=feature_batch._NumericReuse
        class Observed(original):
            def __init__(self,budget):super().__init__(budget);captured.append(self)
            def clear(self):
                if self.entries:
                    for key,(value,_) in self.entries.items():
                        self_outer.assertTrue(all(type(x) in (str,int,tuple) for x in key))
                        self_outer.assertTrue(type(value) in (float,int,bool,type(None),tuple))
                        self_outer.assertNotIn('original-view-',repr(self.intern))
                super().clear()
        self_outer=self
        requests=rolling_views()
        with patch.object(feature_batch,'_NumericReuse',Observed):
            self.assertExact(requests,2**20);self.assertExact(requests,2**20)
        self.assertEqual([c.stats['helpers']['rolling.std']['reused'] for c in captured],[12,12])
        bad=requests_for(setup({f'S{i:03d}':[1e308 if i%2 else 9e307] for i in range(70)},
                              [cs('z','cs_zscore',constant='missing')]),history=1,view_count=1)
        with patch.object(feature_batch,'_NumericReuse',Observed),self.assertRaises(OverflowError):
            execute_feature_plan_batch(bad,reuse_budget_bytes=2**20)
        self.assertIsNone(captured[-1].entries)
        self.assertEqual(captured[-1].entry_bytes,0)
        self.assertEqual(captured[-1].stats['retained_reuse_bytes'],0)

    def test_exact_other_operator_policies_signed_zero_and_epsilon(self):
        samples=[[-0.0,0.0,1e-320,1e-200,-1e-12,1e-12],[-1e150,1e150,1,1,3,None]]
        for values in samples:
            for op in ('cs_rank','cs_zscore','cs_winsorize'):
                nodes=[rolling('r',window=3,min_periods=1,reduction='mean'),cs('c',op,src='r'),
                       node('filled','fill',['c'],{'value':0.0},stage='cross_sectional')]
                base=setup({'A':values,'B':[v if v is None else v*2 for v in values]},nodes)
                requests=requests_for(base,history=5,view_count=2)
                with self.subTest(values=values,op=op):self.assertExact(requests,2**20)

    def test_six_158_and_300_columns_use_same_path_without_legacy_loop(self):
        for columns in (6,158,300):
            nodes=[node('feature'+str(i),'abs',['x']) for i in range(columns)]
            requests=requests_for(setup({'A':[1,2],'B':[3,4]},nodes,outputs=[n['name'] for n in nodes]),history=1,view_count=2)
            expected=tuple(execute_feature_plan(*r).payload for r in requests)
            with patch.object(execution,'execute_feature_plan',side_effect=AssertionError('legacy public loop')):
                result=execute_feature_plan_batch(requests,reuse_budget_bytes=2**20)
            self.assertEqual(tuple(f.payload for f in result['frames']),expected)
            self.assertEqual(len(result['frames'][0].to_dict()['schema']),columns)

    def test_request_profile_and_budget_admission(self):
        requests=rolling_views()
        for budget in (True,False,-1,1.0,None):
            with self.subTest(budget=budget),self.assertRaisesRegex(ContractError,'REUSE_BUDGET'):
                execute_feature_plan_batch(requests,reuse_budget_bytes=budget)
        for value in ([],(),iter(requests),(requests[0][:2],)):
            with self.assertRaises(ContractError):execute_feature_plan_batch(value,reuse_budget_bytes=0)
        with self.assertRaisesRegex(ContractError,'SESSION_ORDER'):
            execute_feature_plan_batch(requests[::-1],reuse_budget_bytes=0)
        p,f,c=[d.to_dict() for d in requests[1]];p['nodes'][0]['params']['ddof']=1
        with self.assertRaisesRegex(ContractError,'SEMANTIC_UNIVERSE'):
            execute_feature_plan_batch((requests[0],docs(p,f,c)),reuse_budget_bytes=0)
        p,f,c=[d.to_dict() for d in requests[0]];c['output_keys'].pop()
        with self.assertRaisesRegex(ContractError,'COMPLETE_GRID'):
            execute_feature_plan_batch((docs(p,f,c),),reuse_budget_bytes=0)
        p,f,c=[d.to_dict() for d in requests[0]];f['rows'].append(copy.deepcopy(f['rows'][0]))
        for function,args in ((execute_feature_plan,docs(p,f,c)),(execute_feature_plan_batch,((docs(p,f,c),),))):
            with self.assertRaisesRegex(ContractError,'Duplicate fact'):
                function(*args,**({'reuse_budget_bytes':0} if function is execute_feature_plan_batch else {}))
