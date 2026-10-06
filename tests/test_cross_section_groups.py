"""Exact frozen-a18 frame/error controls, using only synthetic public ABI inputs.

The golden identities cover the entire canonical FeatureFrame, including input
identities, availability, source bindings, validity and missing/invalid reasons.
No Git or baseline executor is needed to run these tests.
"""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from axiom_engine.core import ExecutionContext, FactBatch, FeaturePlan
from axiom_engine.core import execution
from test_conformance import cs, node, rolling, setup

BASE = 'a18d38ff0708139b681dd76831013cebd98429cd'
GOLDEN = Path(__file__).parent / 'fixtures' / 'core_cs_a18_outcomes.json'


def documents(triple):
    p, f, c = triple
    return FeaturePlan.from_dict(p), FactBatch.from_dict(f), ExecutionContext.from_dict(c)


def outcome(triple):
    try:
        frame = execution.execute_feature_plan(*documents(triple))
        return {'status': 'OK', 'identity': frame.identity}
    except Exception as error:
        return {'status': 'ERROR', 'error_type': type(error).__name__, 'message': str(error)}


def synthetic_cases():
    members = {'A': [True, False, True], 'B': [True]*3, 'C': [True]*3, 'Z': [False]*3}
    industries = {'A': ['I', 'J', 'I'], 'B': ['I']*3, 'C': [None]*3, 'Z': ['I', None, 'I']}
    for op in ('cs_rank', 'cs_zscore', 'cs_winsorize'):
        for group in ('session', 'industry'):
            for missing in ('skip', 'fill_zero', 'propagate', 'reject'):
                for unknown in ('missing', 'reject'):
                    triple = setup({'A': [1, None, 3], 'B': [1, None, 5], 'C': [8, 9, 10], 'Z': [1e10]*3},
                                   [cs('result', op, group=group, missing=missing, unknown_group=unknown)],
                                   members=members, industries=industries)
                    triple[2]['reference'].reverse()
                    yield f'policies/{op}/{group}/{missing}/{unknown}', triple
    samples = {'empty': [None, None], 'singleton': [None, 2], 'constant': [5, 5],
               'signed_zero': [-0.0, 0.0], 'tiny': [1e-320, 3e-320],
               'small': [1e-200, 3e-200], 'large': [-1e308, 1e308],
               'sum_overflow': [1e308, 9e307], 'cancellation': [1e16, 1, -1e16],
               'ties': [3, 1, 1, 5, 3]}
    for label, vals in samples.items():
        for op in ('cs_rank', 'cs_zscore', 'cs_winsorize'):
            for ddof in (0, 1):
                kwargs = {'ddof': ddof, 'epsilon': 0} if op == 'cs_zscore' else {}
                triple = setup({str(i): [v] for i, v in enumerate(vals)}, [cs('result', op, **kwargs)])
                triple[2]['reference'].reverse()
                yield f'numeric/{label}/{op}/{ddof}', triple
    for constant in ('zero', 'missing', 'reject'):
        for excluded in ('missing', 'zero_if_undefined'):
            for missing in ('skip', 'fill_zero', 'propagate', 'reject'):
                triple = setup({'A': [None], 'B': [5], 'N': [99], 'U': [8]},
                               [cs('z', 'cs_zscore', group='industry', constant=constant,
                                   excluded=excluded, missing=missing, ddof=1, clip=[-.25, .25])],
                               members={s: [s != 'N'] for s in ('A', 'B', 'N', 'U')},
                               industries={'A': ['I'], 'B': ['I'], 'N': ['I'], 'U': [None]})
                yield f'undefined/{constant}/{excluded}/{missing}', triple
    # Summing the eligible values would overflow; excluded-only output must
    # retain the original lazy arithmetic behavior and classification clock.
    for excluded in ('missing', 'zero_if_undefined'):
        triple = setup({'A': [1e308], 'B': [9e307], 'N': [99]},
                       [cs('z', 'cs_zscore', epsilon=0, excluded=excluded)],
                       members={'A': [True], 'B': [True], 'N': [False]})
        triple[2]['output_keys'] = [['N', triple[2]['sessions'][0]]]
        yield 'excluded-only-overflow/' + excluded, triple
    for group in ('session', 'industry'):
        triple = setup({'A': [1, 2], 'B': [3, None], 'N': [99, 100], 'U': [5, 6]},
                       [cs('z', 'cs_zscore', group=group)],
                       members={s: [s != 'N']*2 for s in ('A', 'B', 'N', 'U')},
                       industries={'A': ['I']*2, 'B': ['I']*2, 'N': ['I']*2, 'U': [None]*2})
        p, f, c = triple
        for source in ('classification', 'excluded_fact', 'late_fact'):
            p['sources'].append(dict(p['sources'][0], id=source))
        f['sources'] = copy.deepcopy(p['sources'])
        for ref in c['reference']:
            if ref['security_id'] in ('N', 'U'):
                ref.update(source='classification', available_at=ref['session']+'T22:00:00Z')
        for row in f['rows']:
            if row['security_id'] == 'N': row['sources'] = [['excluded_fact']]
            if row['security_id'] == 'B':
                row.update(sources=[['late_fact']], availability=[row['session']+'T23:30:00Z'])
        for subset in ('all', 'eligible', 'excluded', 'unknown'):
            selected = copy.deepcopy(triple)
            if subset != 'all':
                s = {'eligible': 'A', 'excluded': 'N', 'unknown': 'U'}[subset]
                selected[2]['output_keys'] = [k for k in selected[2]['output_keys'] if k[0] == s]
            yield f'provenance/{group}/{subset}', selected
    for group in ('session', 'industry'):
        nodes = [rolling('mean', window=2, min_periods=1), cs('rank', src='mean', group=group),
                 node('lag', 'shift', ['rank'], {'periods': 1}, stage='cross_sectional'),
                 cs('z', 'cs_zscore', src='lag', group=group)]
        triple = setup({'A': [1, 2, 3], 'B': [3, 4, 5], 'N': [9, 9, 9]}, nodes,
                       outputs=['z', 'rank'], members={'A': [True, False, True], 'B': [True]*3, 'N': [False]*3},
                       industries={'A': ['I', 'J', 'I'], 'B': ['I']*3, 'N': ['I']*3})
        triple[2]['reference'].reverse()
        for full in (False, True):
            for scope in ('all', 'last', 'single'):
                selected = copy.deepcopy(triple)
                selected[0]['history_policy'] = 'full' if full else 'partial'
                if scope != 'all':
                    selected[2]['output_keys'] = [k for k in selected[2]['output_keys'] if k[1] == selected[2]['sessions'][-1]]
                    if scope == 'single': selected[2]['output_keys'] = selected[2]['output_keys'][:1]
                yield f'nested/{group}/{full}/{scope}', selected
        early = copy.deepcopy(triple)
        early[2]['cutoffs'] = {d: d+'T11:00:00Z' for d in early[2]['sessions']}
        for label, selected in (('original', triple), ('early', early), ('original-again', triple)):
            yield f'cutoff/{group}/{label}', selected
    for first, second in (('session', 'industry'), ('industry', 'session')):
        nodes = [cs('rank', group=first), rolling('roll', 'rank', window=2, min_periods=1),
                 cs('z', 'cs_zscore', src='roll', group=second)]
        nodes[1]['column']['stage'] = 'cross_sectional'
        triple = setup({'A': [1, 2, 3], 'B': [3, 4, 5], 'C': [8, None, 9]}, nodes,
                       outputs=['rank', 'z'], industries={'A': ['I']*3, 'B': ['I', 'J', 'I'], 'C': ['J']*3})
        triple[2]['reference'].reverse()
        yield f'mixed-groups/{first}/{second}', triple
    base = setup({'A': [1, 2], 'B': [3, 4]}, [cs('z', 'cs_zscore')])
    mutations = {
        'duplicate-fact': lambda p, f, c: f['rows'].append(copy.deepcopy(f['rows'][0])),
        'missing-fact': lambda p, f, c: f['rows'].pop(),
        'undeclared-fact': lambda p, f, c: f['rows'][0].update(security_id='outside'),
        'nonfinite-fact': lambda p, f, c: f['rows'][0].update(values=[float('inf')]),
        'misaligned-cell': lambda p, f, c: f['rows'][0].update(values=[]),
        'unbound-source': lambda p, f, c: f['rows'][0].update(sources=[['outside']]),
        'duplicate-source': lambda p, f, c: f['rows'][0].update(sources=[['facts', 'facts']]),
        'invalid-clock': lambda p, f, c: f['rows'][0].update(availability=['invalid']),
        'duplicate-history': lambda p, f, c: c['history_keys'].append(c['history_keys'][0]),
        'duplicate-output': lambda p, f, c: c['output_keys'].append(c['output_keys'][0]),
        'outside-output': lambda p, f, c: c['output_keys'].append(['outside', c['sessions'][0]]),
        'empty-reference': lambda p, f, c: c.update(reference=[]),
        'duplicate-reference': lambda p, f, c: c['reference'].append(c['reference'][0]),
        'incomplete-reference': lambda p, f, c: c['reference'][0].update(member=False),
        'unavailable-reference': lambda p, f, c: c['reference'][0].update(available_at='2021-01-01T00:00:00Z'),
        'nonbool-member': lambda p, f, c: c['reference'][0].update(member=1),
        'bad-group': lambda p, f, c: p['nodes'][0]['params'].update(group='unsupported'),
        'bad-input': lambda p, f, c: p['nodes'][0].update(inputs=['undeclared']),
        'cell-before-undeclared': lambda p, f, c: f['rows'][0].update(security_id='outside', values=[]),
        'reference-before-event': lambda p, f, c: (c['reference'][0].update(member=1), f['events'].append({})),
    }
    for name, mutate in mutations.items():
        triple = copy.deepcopy(base)
        mutate(*triple)
        yield 'invalid/' + name, triple


class CrossSectionGroupControls(unittest.TestCase):
    def test_exact_a18_frame_and_error_outcomes(self):
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(golden['baseline_commit'], BASE)
        cases = dict(synthetic_cases())
        self.assertEqual(set(cases), set(golden['outcomes']))
        for name, triple in cases.items():
            with self.subTest(case=name):
                self.assertEqual(outcome(triple), golden['outcomes'][name])

    def test_batch_daily_rows_and_sources(self):
        triple = setup({'A': [1, 2, 3], 'B': [3, 4, 5]},
                       [node('lag', 'shift', ['x'], {'periods': 1}), cs('z', 'cs_zscore', src='lag')])
        batch = execution.execute_feature_plan(*documents(triple)).to_dict()
        for day in triple[2]['sessions']:
            daily = copy.deepcopy(triple)
            daily[2]['output_keys'] = [k for k in daily[2]['output_keys'] if k[1] == day]
            for prefix_only in (False, True):
                selected = copy.deepcopy(daily)
                if prefix_only:
                    selected[1]['rows'] = [r for r in selected[1]['rows'] if r['session'] <= day]
                    selected[2]['sessions'] = [s for s in selected[2]['sessions'] if s <= day]
                    selected[2]['cutoffs'] = {s: t for s, t in selected[2]['cutoffs'].items() if s <= day}
                    selected[2]['history_keys'] = [k for k in selected[2]['history_keys'] if k[1] <= day]
                    selected[2]['reference'] = [r for r in selected[2]['reference'] if r['session'] <= day]
                actual = execution.execute_feature_plan(*documents(selected)).to_dict()
                self.assertEqual(actual['rows'], [r for r in batch['rows'] if r['session'] == day])

    def test_source_clock_and_excluded_own_dependency(self):
        triple = dict(synthetic_cases())['provenance/industry/all']
        result = execution.execute_feature_plan(*documents(triple)).to_dict()
        rows = {r['security_id']: r for r in result['rows'] if r['session'] == triple[2]['sessions'][0]}
        self.assertEqual(rows['A']['sources'], [['classification', 'facts', 'late_fact']])
        self.assertEqual(rows['N']['sources'], [['classification', 'excluded_fact', 'facts', 'late_fact']])
        self.assertEqual(rows['A']['availability'], ['2020-01-01T23:30:00Z'])
        self.assertEqual(rows['N']['reasons'], [['UNAVAILABLE_AT_SESSION_CUTOFF']])
        self.assertEqual(rows['U']['sources'], [['classification', 'facts']])

    def test_group_mean_uses_original_order_and_only_when_needed(self):
        triple = setup({'A': [1e16], 'B': [1], 'C': [-1e16], 'N': [99]},
                       [cs('z', 'cs_zscore', epsilon=0)],
                       members={'A': [True], 'B': [True], 'C': [True], 'N': [False]})
        triple[2]['reference'].reverse()
        original = execution.math.fsum
        seen = []
        def observed(vals):
            seen.append(tuple(vals))
            return original(vals)
        with patch.object(execution.math, 'fsum', observed):
            execution.execute_feature_plan(*documents(triple))
        self.assertEqual(seen, [(-1e16, 1.0, 1e16)])
        triple[2]['output_keys'] = [['N', triple[2]['sessions'][0]]]
        with patch.object(execution.math, 'fsum', side_effect=AssertionError('excluded mean evaluated')):
            execution.execute_feature_plan(*documents(triple))
