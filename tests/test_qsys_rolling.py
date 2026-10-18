"""Independent QSYS windows through the existing public scalar/packed APIs."""
import copy
import unittest
from unittest.mock import patch

from axiom_engine.core import (ContractError, FeaturePlan, FactBatch, ExecutionContext,
    execute_feature_plan, execute_feature_plan_batch, compile_feature_columns, execute_feature_columns, required_history, validate_plan)
from axiom_engine.core.operator_semantics import OPERATOR_SEMANTIC_REVISIONS
from axiom_engine.core import execution
from test_conformance import setup, node, col
from test_feature_columns import packed, utc


def window(name, op, inputs=None, **params):
    q = dict(window=4, min_periods=4, inclusive_current=True, missing='skip')
    q.update(params)
    if op in ('rolling_corr', 'rolling_regression'): q.setdefault('zero', 'missing')
    if op == 'rolling_quantile': q.setdefault('interpolation', 'linear')
    if op == 'rolling_idx': q.update(index_base=1)
    return node(name, op, inputs or ['x'], q)


class QsysRolling(unittest.TestCase):
    def compare(self, p, f, c):
        plan = FeaturePlan.from_dict(p)
        scalar = execute_feature_plan(plan, FactBatch.from_dict(f), ExecutionContext.from_dict(c)).to_dict()
        days = sorted({k[1] for k in c['output_keys']})
        inputs = packed(p,f,c)
        source_sets = sorted({tuple(sources) for row in f['rows'] for sources in row['sources']})
        inputs['source_sets'] = [list(s) for s in source_sets]
        for row in f['rows']:
            d = inputs['sessions'].index(row['session']); s = inputs['security_ids'].index(row['security_id'])
            for k, schema in enumerate(p['input_schema']):
                inputs['source_codes'][schema['name']][d,s] = source_sets.index(tuple(row['sources'][k]))
        inputs['reference_source_codes'][:] = source_sets.index(('facts',))
        columns = execute_feature_columns(compile_feature_columns(plan), inputs=inputs,
            output_sessions=days, max_working_bytes=2**22)
        for row in scalar['rows']:
            d = columns['sessions'].index(row['session']); s = columns['security_ids'].index(row['security_id'])
            for k, value in enumerate(row['values']):
                self.assertEqual(bool(columns['validity'][d,s,k]), row['valid'][k])
                if value is not None: self.assertAlmostEqual(float(columns['values'][d,s,k]), value, places=13)
                reason = columns['reason_dictionary'][int(columns['reason_codes'][d,s,k])]
                self.assertEqual([] if reason is None else [reason], row['reasons'][k])
                self.assertEqual(columns['source_sets'][int(columns['source_codes'][d,s,k])], row['sources'][k])
                self.assertEqual(int(columns['available_at_utc_us'][d,s,k]), utc(row['availability'][k]) if row['availability'][k] else -(2**63))
        return scalar['rows']

    def test_independent_golden_and_shared_regression(self):
        nodes = [window(o, 'rolling_regression', output=o) for o in ('slope','rsquare','resi')]
        nodes += [window('q', 'rolling_quantile', quantile=.25),
                  window('imax', 'rolling_idx', extreme='max', ties='first'),
                  window('imin', 'rolling_idx', extreme='min', ties='last'),
                  window('corr', 'rolling_corr', ['x','y'])]
        p,f,c = setup({'A':[[1,8],[2,6],[4,2],[5,0]]}, nodes, [n['name'] for n in nodes],
                      schema=[col('x',stage='fact'),col('y',stage='fact')])
        c['output_keys'] = [c['history_keys'][-1]]
        # x=[1,2,4,5], t=[1,2,3,4]: cov numerator=7, ss(t)=5,
        # ss(x)=10; slope=7/5, R²=49/50, residual=5-5.1.
        with patch.object(execution, '_regression', wraps=execution._regression) as regression:
            rows = self.compare(p,f,c)
            self.assertEqual(regression.call_count, 4)  # scalar shares; packed uses no unbudgeted tuple cache
        for actual, expected in zip(rows[0]['values'], [1.4,.98,-.1,1.75,4,1,-1]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(required_history(FeaturePlan.from_dict(p)), {n['name']:3 for n in nodes})
        self.assertEqual(OPERATOR_SEMANTIC_REVISIONS['rolling'], 'r0_rolling_v1')
        self.assertIn('rolling_regression', OPERATOR_SEMANTIC_REVISIONS)

    def test_missing_positions_pairwise_ties_and_zero_variance(self):
        nodes = [window('slope','rolling_regression', output='slope', min_periods=2),
                 window('resi','rolling_regression', output='resi', min_periods=2),
                 window('first','rolling_idx', extreme='max', ties='first', min_periods=2),
                 window('last','rolling_idx', extreme='max', ties='last', min_periods=2),
                 window('corr','rolling_corr',['x','y'], min_periods=2),
                 window('blocked','rolling_quantile',quantile=.5, min_periods=2, missing='propagate')]
        p,f,c = setup({'A':[[1,1],[None,9],[5,5],[None,None]],
                      'B':[[2,1],[2,2],[2,3],[2,4]]}, nodes, [n['name'] for n in nodes],
                      schema=[col('x',stage='fact'),col('y',stage='fact')])
        c['output_keys'] = [[s,c['sessions'][-1]] for s in ('A','B')]
        rows = self.compare(p,f,c)
        self.assertEqual(rows[0]['values'][:4], [2,None,3,3])
        self.assertAlmostEqual(rows[0]['values'][4], 1)
        self.assertIsNone(rows[0]['values'][5])
        self.assertEqual(rows[1]['values'], [0,0,1,4,None,2])
        p['nodes'][4]['params']['zero'] = 'reject'
        with self.assertRaisesRegex(ContractError,'ZERO_VARIANCE'): self.compare(p,f,c)

    def test_exclusive_empty_partial_history_and_nested_history(self):
        nodes = [node('lag','shift',['x'],dict(periods=2)),
                 window('q','rolling_quantile',['lag'],quantile=1,inclusive_current=False)]
        p,f,c = setup({'A':[1,2,3,4,5,6,7]},nodes)
        self.assertEqual(required_history(FeaturePlan.from_dict(p)), {'q':6})
        rows = self.compare(p,f,c)
        self.assertEqual([r['values'][0] for r in rows], [None]*6+[4])
        p['history_policy'] = 'full'
        with self.assertRaisesRegex(ContractError,'INSUFFICIENT_HISTORY'): self.compare(p,f,c)
        c['output_keys'] = [c['history_keys'][-1]]
        self.assertEqual(self.compare(p,f,c)[0]['values'],[4])

    def test_unavailable_dependency_and_lineage_union(self):
        n = window('corr','rolling_corr',['x','y'])
        p,f,c = setup({'A':[[1,1],[2,2],[3,3],[4,4]]},[n],
                      schema=[col('x',stage='fact'),col('y',stage='fact')])
        c['output_keys'] = [c['history_keys'][-1]]
        second = dict(p['sources'][0], id='other')
        p['sources'].append(second); f['sources'].append(second)
        for row in f['rows']: row['sources'][1] = ['other']
        f['rows'][1]['availability'][1] = c['sessions'][2]+'T00:00:00Z'
        rows = self.compare(p,f,c)
        self.assertEqual(rows[0]['values'], [None])
        self.assertEqual(rows[0]['reasons'], [['UNAVAILABLE_AT_SESSION_CUTOFF']])
        self.assertEqual(rows[0]['sources'], [['facts','other']])
        self.assertEqual(rows[0]['availability'], [c['sessions'][-1]+'T12:00:00Z'])

    def test_regression_degenerate_and_window_endpoints(self):
        nodes = [window(o,'rolling_regression',output=o,min_periods=1) for o in ('slope','rsquare','resi')]
        nodes += [window('low','rolling_quantile',quantile=0,min_periods=1),
                  window('high','rolling_quantile',quantile=1,min_periods=1)]
        p,f,c = setup({'A':[7,7,7,7]},nodes,[n['name'] for n in nodes])
        rows = self.compare(p,f,c)
        self.assertEqual(rows[0]['values'],[None,None,None,7,7])
        self.assertEqual(rows[-1]['values'],[0,None,0,7,7])
        p['nodes'][1]['params']['zero']='reject'
        c['output_keys']=[c['history_keys'][-1]]
        with self.assertRaisesRegex(ContractError,'ZERO_VARIANCE'): self.compare(p,f,c)

    def test_existing_batch_entrypoint_keeps_view_provenance(self):
        from test_feature_plan_batch import requests_for
        nodes = [window(o,'rolling_regression',output=o) for o in ('slope','rsquare','resi')]
        nodes += [window('q','rolling_quantile',quantile=.8),
                  window('idx','rolling_idx',extreme='max',ties='first'),
                  window('corr','rolling_corr',['x','x'])]
        requests = requests_for(setup({'A':[1,2,4,5,8]},nodes,[n['name'] for n in nodes]),
                                history=4,view_count=2)
        actual = execute_feature_plan_batch(requests,reuse_budget_bytes=2**20)['frames']
        expected = tuple(execute_feature_plan(*r) for r in requests)
        self.assertEqual(tuple(f.payload for f in actual),tuple(f.payload for f in expected))
        self.assertNotEqual(actual[0].to_dict()['rows'][0]['sources'],actual[1].to_dict()['rows'][0]['sources'])

    def test_packed_window_workspace_rejected_before_numeric_execution(self):
        nodes = [window('slope','rolling_regression',output='slope',window=256,min_periods=256)]
        p,f,c = setup({'A':list(range(256))},nodes)
        c['output_keys'] = [c['history_keys'][-1]]
        plan = FeaturePlan.from_dict(p); compiled = compile_feature_columns(plan)
        def run(limit):
            return execute_feature_columns(compiled,inputs=packed(p,f,c),
                output_sessions=[c['sessions'][-1]],max_working_bytes=limit)
        with patch.object(execution,'_regression_cache',side_effect=AssertionError('packed cache forbidden')):
            result = run(2**22)
        self.assertAlmostEqual(float(result['values'][0,0,0]),1.)
        peak = result['stats']['peak_working_bytes']
        self.assertGreater(peak,256*1024)  # Python and gather workspace is actually charged.
        with patch.object(execution,'_window_value',side_effect=AssertionError('numeric execution before budget admission')):
            with self.assertRaisesRegex(ContractError,'COLUMN_WORKING_BYTES_EXCEEDED'):
                run(peak-1)
        accepted = run(peak)
        self.assertEqual(accepted['values'].tobytes(),result['values'].tobytes())
        self.assertEqual(accepted['validity'].tobytes(),result['validity'].tobytes())
        self.compare(p,f,c)

    def test_admission_rejects_implicit_policies(self):
        n = window('q','rolling_quantile',quantile=.5)
        p,_,_ = setup({'A':[1,2,3,4]},[n])
        for field, value in [('quantile',2),('interpolation','nearest'),('min_periods',5),('missing','fill')]:
            bad=copy.deepcopy(p); bad['nodes'][0]['params'][field]=value
            with self.assertRaises(ContractError): validate_plan(FeaturePlan.from_dict(bad),execution=True)


if __name__ == '__main__': unittest.main()
