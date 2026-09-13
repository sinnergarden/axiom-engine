"""Independent PR1 probes. Execute with fixed archived Engine src on PYTHONPATH.

Only synthetic in-memory inputs; no Data/Research imports or business execution.
Observed defect checks deliberately assert the documented expected behavior.
"""
import copy
import json
import math
import unittest

from axiom_engine.core import (
    ABI, SEMANTICS, FeaturePlan, FactBatch, ExecutionContext,
    execute_feature_plan, required_history,
)

REF = 'sha256:' + '1' * 64


def column(name, stage='base', missing='preserve'):
    return dict(name=name, dtype='float64', unit='dimensionless',
                stage=stage, missing=missing)


def node(name, op, inputs, params, stage='base', missing='preserve'):
    return dict(name=name, op=op, inputs=inputs, params=params, version='1',
                column=column(name, stage, missing))


def fixture(data, nodes):
    days = ['2020-01-02', '2020-01-03', '2020-01-06'][:len(next(iter(data.values()))) ]
    sources = [dict(id='s', data_ref=REF, view_ref=REF,
                    revision_policy='fixed', qualification='synthetic',
                    availability_basis='explicit')]
    schema = [column('x', 'fact')]
    keys = [[sec, day] for sec in data for day in days]
    rows = [dict(security_id=sec, session=day, values=[data[sec][i]],
                 availability=[day+'T07:00:00Z'], sources=[['s']],
                 missing_reasons=[None if data[sec][i] is not None else 'NOT_REPORTED'])
            for sec in data for i, day in enumerate(days)]
    refs = [dict(security_id=sec, session=day, member=True, industry='I',
                 available_at=day+'T00:00:00Z', source='s') for sec, day in keys]
    plan = dict(abi=ABI, semantics=SEMANTICS, recipe_ref=REF,
                calendar_ref=REF, reference_ref=REF,
                reference_members={day: {sec: 'I' for sec in data} for day in days},
                input_schema=schema, event_schema={}, sources=sources,
                observation_domain='sessions', history_policy='full',
                nodes=nodes, outputs=[dict(node=nodes[-1]['name'],
                                         column=copy.deepcopy(nodes[-1]['column']))],
                obligations=[])
    facts = dict(abi=ABI, calendar_ref=REF, schema=schema, sources=sources,
                 rows=rows, event_schema={}, events=[])
    context = dict(abi=ABI, calendar_ref=REF, reference_ref=REF, sessions=days,
                   cutoffs={d: d+'T08:00:00Z' for d in days}, history_keys=keys,
                   output_keys=[[sec, days[-1]] for sec in data], reference=refs)
    return plan, facts, context


def run(p, f, c):
    return execute_feature_plan(FeaturePlan.from_dict(p), FactBatch.from_dict(f),
                                ExecutionContext.from_dict(c)).to_dict()


def rank_params(missing='reject'):
    return dict(group='session', unknown_group='missing', missing=missing,
                ties='average', excluded='missing')


class ReviewProbes(unittest.TestCase):
    def test_f1_shift_required_column_with_complete_history(self):
        p, f, c = fixture({'A': [1, 2]},
                          [node('lag', 'shift', ['x'], {'periods': 1}, missing='reject')])
        self.assertEqual(required_history(FeaturePlan.from_dict(p)), {'lag': 1})
        result = run(p, f, c)
        self.assertEqual(result['rows'][0]['values'], [1.0])
        self.assertEqual(result['rows'][0]['valid'], [True])

    def test_f1_cross_section_reject_sees_only_needed_lag_cells(self):
        p, f, c = fixture({'A': [1, 2], 'B': [3, 4]}, [
            node('lag', 'shift', ['x'], {'periods': 1}),
            node('rank', 'cs_rank', ['lag'], rank_params(), stage='cross_sectional')])
        self.assertEqual(required_history(FeaturePlan.from_dict(p)), {'rank': 1})
        result = run(p, f, c)
        self.assertEqual([r['values'] for r in result['rows']], [[0.5], [1.0]])

    def test_control_same_cross_section_plan_with_skip(self):
        p, f, c = fixture({'A': [1, 2], 'B': [3, 4]}, [
            node('lag', 'shift', ['x'], {'periods': 1}),
            node('rank', 'cs_rank', ['lag'], rank_params('skip'), stage='cross_sectional')])
        self.assertEqual([r['values'] for r in run(p, f, c)['rows']], [[0.5], [1.0]])

    def test_numeric_small_std_is_not_zero(self):
        p, f, c = fixture({'A': [1e-200, 3e-200]}, [node('std', 'rolling', ['x'],
            dict(window=2, min_periods=2, inclusive_current=True, reduction='std',
                 missing='skip', ddof=0, ties='average'))])
        actual = run(p, f, c)['rows'][0]['values'][0]
        self.assertTrue(math.isclose(actual, 1e-200, rel_tol=1e-13, abs_tol=0), actual)

    def test_numeric_small_cs_scale_is_not_constant(self):
        p, f, c = fixture({'A': [1e-200], 'B': [3e-200]}, [node('z', 'cs_zscore', ['x'],
            dict(group='session', unknown_group='missing', missing='skip', ddof=0,
                 epsilon=0, constant='zero', clip=None, excluded='missing'),
            stage='cross_sectional')])
        self.assertEqual([r['values'] for r in run(p, f, c)['rows']], [[-1.0], [1.0]])

    def test_control_late_fact_cannot_be_filled_valid(self):
        p, f, c = fixture({'A': [1, 2]}, [node('filled', 'fill', ['x'], {'value': 0})])
        c['output_keys'] = [['A', c['sessions'][0]]]
        f['rows'][0]['availability'][0] = '2020-01-03T07:00:00Z'
        row = run(p, f, c)['rows'][0]
        self.assertEqual(row['values'], [None])
        self.assertEqual(row['valid'], [False])
        self.assertIn('UNAVAILABLE_AT_SESSION_CUTOFF', row['reasons'][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
