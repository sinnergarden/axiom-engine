"""Synthetic differential/timing probe against the original frozen Core source.

Run from this checkout with PYTHONPATH=src:tests. No Data, Research, model, or
account execution. Timings exclude synthetic input construction and input-document
encoding; public execute includes output encoding and consumes the same bound inputs.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import types
from contextlib import ExitStack
from unittest.mock import patch

from axiom_engine.core import ExecutionContext, FactBatch, FeaturePlan
from test_conformance import cs, node, rolling, setup

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'efd36664ed9530b0eb9c46cd5ee154301a15c4c1'


def baseline_module():
    source = subprocess.check_output(
        ['git', 'show', BASELINE + ':src/axiom_engine/core/execution.py'], cwd=ROOT)
    name = 'axiom_engine.core._synthetic_baseline_execution'
    module = types.ModuleType(name)
    module.__package__ = 'axiom_engine.core'
    sys.modules[name] = module
    exec(compile(source, BASELINE + ':execution.py', 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def documents(triple):
    p, f, c = triple
    return FeaturePlan.from_dict(p), FactBatch.from_dict(f), ExecutionContext.from_dict(c)


def outcome(module, triple):
    try:
        report = module.execute_feature_plan(*documents(triple))
        return ('OK', report.payload, report.identity)
    except Exception as error:
        return ('ERROR', type(error).__module__ + '.' + type(error).__name__, str(error))


def equivalent(old, new, name, triple):
    expected, actual = outcome(old, triple), outcome(new, triple)
    assert expected == actual, name + ': baseline/candidate differ'
    return dict(case=name, status=expected[0],
                error=expected[2] if expected[0] == 'ERROR' else None,
                payload_identity_equal=expected[0] == 'OK')


def differential_cases(old, new):
    checks = []
    members = {'A': [True, False, True], 'B': [True, True, True], 'C': [True]*3, 'Z': [False]*3}
    industries = {'A': ['I', 'J', 'I'], 'B': ['I']*3, 'C': [None]*3, 'Z': ['I', None, 'I']}
    for op in ('cs_rank', 'cs_zscore', 'cs_winsorize'):
        for group in ('session', 'industry'):
            for missing in ('skip', 'fill_zero', 'propagate', 'reject'):
                for unknown in ('missing', 'reject'):
                    triple = setup({'A': [1, None, 3], 'B': [3, None, 5], 'C': [8, 9, 10], 'Z': [1e10]*3},
                                   [cs('result', op, group=group, missing=missing, unknown_group=unknown)],
                                   members=members, industries=industries)
                    # Deliberately preserve a non-sorted reference insertion order.
                    triple[2]['reference'].reverse()
                    check = equivalent(old, new, f'{op}/{group}/{missing}/{unknown}', triple)
                    if unknown == 'missing':
                        assert (check['status'] == 'ERROR') == (missing == 'reject')
                        if missing == 'reject': assert check['error'] == 'MISSING_REFERENCE_VALUE'
                    checks.append(check)
    for excluded in ('missing', 'zero_if_undefined'):
        triple = setup({'A': [5], 'B': [5], 'C': [7], 'Z': [1e10]},
                       [cs('result', 'cs_zscore', group='industry', excluded=excluded)],
                       members={s: [s != 'Z'] for s in ('A', 'B', 'C', 'Z')},
                       industries={'A': ['I'], 'B': ['I'], 'C': [None], 'Z': ['I']})
        checks.append(equivalent(old, new, 'constant/excluded/' + excluded, triple))
    for group in ('session', 'industry'):
        triple = setup({'A': [1], 'B': [3], 'N': [99], 'U': [5]}, [cs('result', group=group)],
                       members={s: [s != 'N'] for s in ('A', 'B', 'N', 'U')},
                       industries={'A': ['I'], 'B': ['I'], 'N': ['I'], 'U': [None]})
        p, f, c = triple
        p['sources'].append(dict(p['sources'][0], id='classification'))
        f['sources'] = copy.deepcopy(p['sources'])
        for ref in c['reference']:
            if ref['security_id'] in ('N', 'U'):
                ref.update(source='classification', available_at=ref['session']+'T22:00:00Z')
        checks.append(equivalent(old, new, 'excluded-and-null-classification-provenance/' + group, triple))
        rows = json.loads(outcome(new, triple)[1])['rows']
        assert all(row['sources'] == [['classification', 'facts']] for row in rows)
        assert all(row['availability'] == [row['session']+'T22:00:00Z'] for row in rows)
    base = setup({'A': [1, 2, 3], 'B': [3, 4, 5]},
                 [node('lag', 'shift', ['x'], {'periods': 1}), cs('rank', src='lag')])
    early, late = copy.deepcopy(base), copy.deepcopy(base)
    early[2]['cutoffs'] = {d: d + 'T11:00:00Z' for d in early[2]['sessions']}
    for label, triple in (('cutoff/original', base), ('cutoff/early', early), ('cutoff/original-again', late)):
        checks.append(equivalent(old, new, label, triple))
    subset = copy.deepcopy(base)
    subset[2]['output_keys'] = [subset[2]['output_keys'][-1]]
    checks.append(equivalent(old, new, 'single-output/full-reference-closure', subset))

    def mutation(label, mutate):
        triple = copy.deepcopy(base)
        mutate(*triple)
        result = equivalent(old, new, label, triple)
        assert result['status'] == 'ERROR', label + ' must reject'
        checks.append(result)

    mutation('duplicate-fact', lambda p,f,c: f['rows'].append(copy.deepcopy(f['rows'][0])))
    mutation('missing-fact', lambda p,f,c: f['rows'].pop())
    mutation('undeclared-fact', lambda p,f,c: f['rows'][0].update(security_id='undeclared'))
    mutation('duplicate-history-key', lambda p,f,c: c['history_keys'].append(c['history_keys'][0][:]))
    mutation('duplicate-output-key', lambda p,f,c: c['output_keys'].append(c['output_keys'][0][:]))
    mutation('output-outside-history', lambda p,f,c: c['output_keys'].append(['outside',c['sessions'][0]]))
    mutation('unknown-session', lambda p,f,c: c['history_keys'][0].__setitem__(1,'2021-01-01'))
    mutation('misaligned-column', lambda p,f,c: f['rows'][0].update(values=[]))
    mutation('unbound-cell-source', lambda p,f,c: f['rows'][0].update(sources=[['unbound']]))
    mutation('duplicate-cell-source', lambda p,f,c: f['rows'][0].update(sources=[['facts','facts']]))
    mutation('present-missing-reason', lambda p,f,c: f['rows'][0].update(missing_reasons=['unexpected']))
    mutation('invalid-availability', lambda p,f,c: f['rows'][0].update(availability=['invalid']))
    mutation('nonmonotonic-cutoffs', lambda p,f,c: c['cutoffs'].update({c['sessions'][0]:'2021-01-01T23:00:00Z'}))
    mutation('duplicate-reference', lambda p,f,c: c['reference'].append(copy.deepcopy(c['reference'][0])))
    mutation('missing-reference', lambda p,f,c: c['reference'].pop())
    mutation('undeclared-reference', lambda p,f,c: c['reference'][0].update(security_id='outside'))
    mutation('nonbool-member', lambda p,f,c: c['reference'][0].update(member=1))
    mutation('empty-industry', lambda p,f,c: c['reference'][0].update(industry=' '))
    mutation('unavailable-reference', lambda p,f,c: c['reference'][0].update(available_at='2021-01-01T00:00:00Z'))
    mutation('unbound-reference', lambda p,f,c: c['reference'][0].update(source='unbound'))
    mutation('incomplete-plan-members', lambda p,f,c: c['reference'][0].update(member=False))
    mutation('invalid-event-after-reference', lambda p,f,c: f['events'].append({}))
    # Two simultaneous errors prove the original validation order is retained.
    mutation('cell-validation-before-undeclared-fact', lambda p,f,c: f['rows'][0].update(security_id='outside',values=[]))
    mutation('reference-validation-before-event', lambda p,f,c: (c['reference'][0].update(member=1),f['events'].append({})))
    return checks


def timed(module, inputs):
    stages = {}
    with ExitStack() as stack:
        for name in ('validate_plan', '_inputs', '_reference_index', '_required_cells', '_cross_section'):
            if not hasattr(module, name):
                continue
            original = getattr(module, name)
            def measured(*args, _original=original, _name=name, **kwargs):
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    stages[_name] = stages.get(_name, 0) + time.perf_counter() - started
            stack.enter_context(patch.object(module, name, measured))
        started = time.perf_counter()
        result = module.execute_feature_plan(*inputs)
        total = time.perf_counter() - started
    return result, dict(total_seconds=total, stages_seconds=stages,
                        remaining_operator_output_seconds=total-sum(stages.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert args.repeats > 0 and not args.output.exists()
    old, baseline_sha = baseline_module()
    new = importlib.import_module('axiom_engine.core.execution')
    checks = differential_cases(old, new)
    nodes = [node('lag', 'shift', ['x'], {'periods': 1}), rolling('mean', window=20, min_periods=2),
             cs('rank'), cs('z', 'cs_zscore'), cs('industry_rank', group='industry'),
             cs('industry_winsor', 'cs_winsorize', group='industry')]
    data = {f'S{i:03}': [float((i*17+j*3)%97) + j/10 for j in range(21)] for i in range(389)}
    members = {s: [int(s[1:])%13 != 0 or j%3 == 0 for j in range(21)] for s in data}
    industries = {s: [None if int(s[1:])%17 == 0 else 'I'+str(int(s[1:])%7) for j in range(21)] for s in data}
    large = setup(data, nodes, outputs=[n['name'] for n in nodes], members=members, industries=industries)
    large[2]['reference'].reverse()
    timings = {}
    for scope in ('latest_session', 'all_sessions'):
        triple = copy.deepcopy(large)
        if scope == 'latest_session':
            triple[2]['output_keys'] = [k for k in triple[2]['output_keys'] if k[1] == triple[2]['sessions'][-1]]
        inputs = documents(triple)
        measurements = {'baseline': [], 'candidate': []}
        for repetition in range(args.repeats):
            reports = {}
            order = ('baseline','candidate') if repetition%2 == 0 else ('candidate','baseline')
            for label in order:
                reports[label], record = timed(old if label == 'baseline' else new, inputs)
                measurements[label].append(record)
                print(json.dumps(dict(scope=scope, repetition=repetition, executor=label, **record)), flush=True)
            assert reports['baseline'].payload == reports['candidate'].payload
            assert reports['baseline'].identity == reports['candidate'].identity
        report = reports['candidate']
        timings[scope] = dict(fact_rows=8169, securities=389, history_sessions=21, features=6,
                             output_rows=len(triple[2]['output_keys']), payload_identity_equal=True,
                             feature_identity=report.identity, payload_bytes=len(report.payload.encode()),
                             measurements=measurements,
                             median_seconds={k:statistics.median(x['total_seconds'] for x in v) for k,v in measurements.items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = dict(status='PASS', qualification='pure_synthetic', baseline_commit=BASELINE,
                  baseline_execution_file_sha256=baseline_sha,
                  candidate_execution_file_sha256=hashlib.sha256((ROOT/'src/axiom_engine/core/execution.py').read_bytes()).hexdigest(),
                  python=sys.version, platform=platform.platform(),
                  peak_process_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                  timing_scope='Public execute only, same bound documents; staged wrappers; alternating order; no real-data performance claim.',
                  differential_checks=checks, timings=timings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print('PASS', str(args.output), len(checks), 'differential cases', flush=True)


if __name__ == '__main__':
    main()
