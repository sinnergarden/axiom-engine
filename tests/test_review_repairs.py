"""Controls that distinguish dependency scoping/stable std from relaxed policies."""
import copy
import math
import unittest

from axiom_engine.core import ContractError
from test_review_probes import fixture, node, rank_params, run


def shift(missing='preserve', source='x', name='lag', stage='base'):
    return node(name, 'shift', [source], {'periods': 1}, stage, missing)


def rolling(source='x', reduction='mean', ddof=0, inclusive=True):
    return node('roll', 'rolling', [source], dict(window=2, min_periods=2,
        inclusive_current=inclusive, reduction=reduction, missing='skip', ddof=ddof,
        ties='average'), missing='reject')


def zscore(ddof=0, epsilon=0, constant='zero', clip=None, missing='skip'):
    return node('z', 'cs_zscore', ['x'], dict(group='session', unknown_group='missing',
        missing=missing, ddof=ddof, epsilon=epsilon, constant=constant, clip=clip,
        excluded='missing'), stage='cross_sectional')


def numbers(result):
    return [row['values'][0] for row in result['rows']]


class RepairControls(unittest.TestCase):
    def test_needed_lag_null_and_history_row_still_reject(self):
        p,f,c=fixture({'A':[None,2]},[shift('reject')])
        with self.assertRaisesRegex(ContractError,'Missing value in required column'):
            run(p,f,c)
        p,f,c=fixture({'A':[1,2]},[shift('reject')])
        f['rows'].pop(0)
        with self.assertRaisesRegex(ContractError,'MISSING_HISTORY_ROW'):run(p,f,c)
        # Removing the declared history as well still fails full lag admission.
        c['history_keys'].pop(0); c['reference'].pop(0)
        with self.assertRaisesRegex(ContractError,'INSUFFICIENT_HISTORY'):run(p,f,c)

    def test_intermediate_reject_cannot_be_hidden_by_final_fill(self):
        p,f,c=fixture({'A':[None,2]},[shift('reject'),
            node('filled','fill',['lag'],{'value':0})])
        with self.assertRaisesRegex(ContractError,'Missing value in required column'):
            run(p,f,c)

    def test_subset_retains_reference_values_and_reject_policy(self):
        p,f,c=fixture({'A':[1,2],'B':[3,4]},[shift('reject'),
            node('rank','cs_rank',['lag'],rank_params(),stage='cross_sectional')])
        c['output_keys']=[c['output_keys'][0]]
        self.assertEqual(numbers(run(p,f,c)),[.5])
        p['nodes'][0]['column']['missing']='preserve'
        f['rows'][2]['values']=[None]; f['rows'][2]['missing_reasons']=['NOT_REPORTED']
        with self.assertRaisesRegex(ContractError,'MISSING_REFERENCE_VALUE'):run(p,f,c)
        f['rows']=[r for r in f['rows'] if r['security_id']=='A']
        c['history_keys']=[k for k in c['history_keys'] if k[0]=='A']
        c['reference']=[r for r in c['reference'] if r['security_id']=='A']
        with self.assertRaisesRegex(ContractError,'INCOMPLETE_REFERENCE'):run(p,f,c)

    def test_rolling_then_shift_keeps_the_intermediate_window(self):
        p,f,c=fixture({'A':[2,4,100]},[rolling(),shift('reject','roll','lag')])
        self.assertEqual(numbers(run(p,f,c)),[3])
        # The needed intermediate is day 2, not the final day 3.
        f['rows'][0]['values']=[None];f['rows'][0]['missing_reasons']=['NOT_REPORTED']
        with self.assertRaisesRegex(ContractError,'Missing value in required column'):
            run(p,f,c)
        p,f,c=fixture({'A':[2,4,100]},[rolling(inclusive=False)])
        self.assertEqual(numbers(run(p,f,c)),[3])

    def test_historical_cs_dependencies_expand_before_shift(self):
        p,f,c=fixture({'A':[1,2,100],'B':[3,4,200]},[shift('reject'),
            node('rank','cs_rank',['lag'],rank_params(),stage='cross_sectional'),
            shift('reject','rank','previous_rank',stage='cross_sectional')])
        c['output_keys']=[c['output_keys'][0]]
        self.assertEqual(numbers(run(p,f,c)),[.5])
        # B leaves on the output date but is still a necessary past contributor.
        last=c['sessions'][-1]
        p['reference_members'][last].pop('B')
        next(r for r in c['reference'] if r['security_id']=='B' and r['session']==last)['member']=False
        self.assertEqual(numbers(run(p,f,c)),[.5])
        f['rows'][3]['values']=[None];f['rows'][3]['missing_reasons']=['NOT_REPORTED']
        with self.assertRaisesRegex(ContractError,'Missing value in required column'):
            run(p,f,c)

    def test_scoped_elementwise_rejection_and_partial_warmup(self):
        p,f,c=fixture({'A':[-2,3]},[node('log','log1p',['x'],{'domain':'reject'})])
        self.assertEqual(numbers(run(p,f,c)),[math.log(4)])
        c['output_keys']=[['A',c['sessions'][0]]]
        with self.assertRaisesRegex(ContractError,'LOG_DOMAIN'):run(p,f,c)
        p,f,c=fixture({'A':[1,2]},[shift('reject')]);p['history_policy']='partial'
        self.assertEqual(numbers(run(p,f,c)),[1])
        c['output_keys']=[['A',c['sessions'][0]]]
        with self.assertRaisesRegex(ContractError,'Missing value in required column'):run(p,f,c)
        p['nodes'][0]['column']['missing']='preserve';p['outputs'][0]['column']['missing']='preserve'
        self.assertEqual(numbers(run(p,f,c)),[None])
        # Empty upstream scope at partial warmup must not force evaluation.
        p,f,c=fixture({'A':[1]},[node('base','identity',['x'],{}),shift(source='base')])
        p['history_policy']='partial'
        self.assertEqual(numbers(run(p,f,c)),[None])

    def test_industry_scope_preserves_full_group_not_unrelated_group(self):
        p,f,c=fixture({'A':[1,2],'B':[3,4],'C':[None,8]},[shift(),
            node('rank','cs_rank',['lag'],rank_params(),stage='cross_sectional')])
        p['nodes'][1]['params']['group']='industry'
        for day in c['sessions']:p['reference_members'][day]['C']='OTHER'
        for r in c['reference']:
            if r['security_id']=='C':r['industry']='OTHER'
        c['output_keys']=[c['output_keys'][0]]
        self.assertEqual(numbers(run(p,f,c)),[.5])
        p['nodes'][1]['params']['group']='session'
        with self.assertRaisesRegex(ContractError,'MISSING_REFERENCE_VALUE'):run(p,f,c)

    def test_scaled_std_and_zscore_both_ddof(self):
        # Zero absolute tolerance prevents both underflow and broad near-zero passes.
        for scale in (1e-200,1.0,1e150):
            for ddof in (0,1):
                with self.subTest(scale=scale,ddof=ddof):
                    p,f,c=fixture({'A':[scale,3*scale]},[rolling(reduction='std',ddof=ddof)])
                    actual=numbers(run(p,f,c))[0]
                    self.assertTrue(math.isclose(actual,scale*math.sqrt(2 if ddof else 1),rel_tol=1e-13,abs_tol=0),actual)
                    p,f,c=fixture({'A':[scale],'B':[3*scale]},[zscore(ddof=ddof)])
                    expected=1/math.sqrt(2) if ddof else 1
                    for a,e in zip(numbers(run(p,f,c)),[-expected,expected]):
                        self.assertTrue(math.isclose(a,e,rel_tol=1e-13,abs_tol=0),(a,e))

    def test_tiny_constants_epsilon_clip_and_missing_policies(self):
        for policy,expected in [('zero',[0,0]),('missing',[None,None])]:
            p,f,c=fixture({'A':[1e-200],'B':[1e-200]},[zscore(constant=policy)])
            self.assertEqual(numbers(run(p,f,c)),expected)
        p['nodes'][0]['params']['constant']='reject'
        with self.assertRaisesRegex(ContractError,'UNDEFINED_CS_SCALE'):run(p,f,c)
        p,f,c=fixture({'A':[1e-200],'B':[3e-200]},[zscore(epsilon=2e-200)])
        self.assertEqual(numbers(run(p,f,c)),[0,0])
        p['nodes'][0]['params']['epsilon']=0;p['nodes'][0]['params']['clip']=[-.25,.25]
        self.assertEqual(numbers(run(p,f,c)),[-.25,.25])
        for policy,expected in [('skip',[None,0]),('propagate',[None,None]),('fill_zero',[-1,1])]:
            p,f,c=fixture({'A':[None],'B':[2e-200]},[zscore(missing=policy)])
            self.assertEqual(numbers(run(p,f,c)),expected)
        p['nodes'][0]['params']['missing']='reject'
        with self.assertRaisesRegex(ContractError,'MISSING_REFERENCE_VALUE'):run(p,f,c)

    def test_reject_pipeline_and_tiny_std_batch_daily_equivalence(self):
        for scale in (1.0,1e-200):
            params=zscore()['params'];params['missing']='reject'
            nodes=[shift('reject'),node('z','cs_zscore',['lag'],params,stage='cross_sectional')]
            p,f,c=fixture({'A':[scale,2*scale,3*scale],'B':[3*scale,4*scale,5*scale]},nodes)
            c['output_keys']=[k for k in c['history_keys'] if k[1]!=c['sessions'][0]]
            batch=run(p,f,c)
            for day in c['sessions'][1:]:
                daily_f=copy.deepcopy(f);daily_c=copy.deepcopy(c)
                daily_f['rows']=[r for r in f['rows'] if r['session']<=day]
                daily_c['sessions']=[d for d in c['sessions'] if d<=day]
                daily_c['cutoffs']={d:t for d,t in c['cutoffs'].items() if d<=day}
                daily_c['history_keys']=[k for k in c['history_keys'] if k[1]<=day]
                daily_c['reference']=[r for r in c['reference'] if r['session']<=day]
                daily_c['output_keys']=[k for k in c['output_keys'] if k[1]==day]
                actual=run(p,daily_f,daily_c)
                self.assertEqual(actual['rows'],[r for r in batch['rows'] if r['session']==day])
                self.assertEqual(actual['schema'],batch['schema'])
