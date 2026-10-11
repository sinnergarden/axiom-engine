"""One composite synthetic comparison with the original public Core entrypoint."""
import copy
from datetime import datetime, timezone
import unittest
import numpy as np

from axiom_engine.core import (ContractError, FeaturePlan, FactBatch, ExecutionContext,
    compile_feature_columns, execute_feature_columns, execute_feature_plan)
from test_conformance import setup, node, rolling, cs, REF


def utc(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()) * 1000000


def packed(p, f, c):
    days, securities = c['sessions'], sorted({r['security_id'] for r in f['rows']})
    shape = len(days), len(securities)
    rows = {(r['session'], r['security_id']): r for r in f['rows']}
    refs = {(r['session'], r['security_id']): r for r in c['reference']}
    result = dict(contract_version='feature_column_input_v1', calendar_ref=p['calendar_ref'],
        reference_ref=p['reference_ref'], schema=p['input_schema'], source_bindings=p['sources'],
        sessions=days, security_ids=securities,
        cutoffs_utc_us=np.array([utc(c['cutoffs'][d]) for d in days], dtype='int64'),
        values={}, validity={}, reason_codes={}, reason_dictionary=[None, 'NOT_REPORTED'],
        available_at_utc_us={}, source_codes={}, source_sets=[['facts']],
        reference_member=np.array([[refs[d,s]['member'] for s in securities] for d in days], dtype='bool'),
        reference_industry=np.array([[refs[d,s]['industry'] for s in securities] for d in days], dtype='object'),
        reference_available_at_utc_us=np.array([[utc(refs[d,s]['available_at']) for s in securities] for d in days], dtype='int64'),
        reference_source_codes=np.zeros(shape, dtype='int32'), input_refs={'prepared_scope_ref': REF})
    for i, schema in enumerate(p['input_schema']):
        name = schema['name']
        result['values'][name] = np.array([[rows[d,s]['values'][i] or 0 for s in securities] for d in days], dtype=schema['dtype'])
        result['validity'][name] = np.array([[rows[d,s]['values'][i] is not None for s in securities] for d in days], dtype='bool')
        result['reason_codes'][name] = (~result['validity'][name]).astype('int32')
        result['available_at_utc_us'][name] = np.array([[utc(rows[d,s]['availability'][i]) for s in securities] for d in days], dtype='int64')
        result['source_codes'][name] = np.zeros(shape, dtype='int32')
    return result


class FeatureColumns(unittest.TestCase):
    def test_composite_original_math_clocks_missing_cohort_anchor_and_batch(self):
        nodes = [node('lag', 'shift', ['x'], {'periods': 1}),
                 rolling('mean', 'lag', 3, 2), rolling('std', 'mean', 2, 1, 'std', ddof=1),
                 cs('z', 'cs_zscore', 'mean'), cs('rank', src='mean'),
                 node('out', 'add', ['z', 'rank'], stage='cross_sectional')]
        p,f,c = setup({'A':[1,None,3,4,5,6], 'B':[1,2,3,4,5,8], 'C':[9,8,7,6,5,4]},
                      nodes, ['lag','mean','std','z','rank','out'])
        # Explicit stopped/missing observation stays on the full calendar.
        f['rows'][2]['availability'][0] = c['sessions'][3] + 'T00:00:00Z'
        c['output_keys'] = [[s,d] for s in ('A','B','C') for d in c['sessions'][2:]]
        compiled = compile_feature_columns(FeaturePlan.from_dict(p))
        for anchor in (1.0, 1.25):
            view = copy.deepcopy(f)
            for row in view['rows']:
                if row['values'][0] is not None: row['values'][0] *= anchor
            inputs = packed(p, view, c)
            result = execute_feature_columns(compiled, inputs=inputs,
                output_sessions=c['sessions'][2:], max_working_bytes=2**20)
            old = execute_feature_plan(FeaturePlan.from_dict(p), FactBatch.from_dict(view), ExecutionContext.from_dict(c)).to_dict()
            for row in old['rows']:
                d = result['sessions'].index(row['session']); s = result['security_ids'].index(row['security_id'])
                for k, value in enumerate(row['values']):
                    self.assertEqual(bool(result['validity'][d,s,k]), row['valid'][k])
                    if value is not None: self.assertEqual(float(result['values'][d,s,k]), value)
                    reason = result['reason_dictionary'][int(result['reason_codes'][d,s,k])]
                    self.assertEqual([] if reason is None else [reason], row['reasons'][k])
                    self.assertEqual(sorted(result['source_sets'][int(result['source_codes'][d,s,k])]), row['sources'][k])
                    actual = int(result['available_at_utc_us'][d,s,k])
                    self.assertEqual(actual, utc(row['availability'][k]) if row['availability'][k] else -(2**63))
            # Exact bytes are immutable; caller mutations cannot change saved result.
            identity = result['result_ref']; inputs['values']['x'][:] = 999
            self.assertEqual(result['result_ref'], identity)
            with self.assertRaises(ValueError): result['values'].setflags(write=True)
        with self.assertRaisesRegex(ContractError, 'WORKING_BYTES'):
            execute_feature_columns(compiled, inputs=packed(p,f,c), output_sessions=c['sessions'][2:], max_working_bytes=1)


if __name__ == '__main__': unittest.main()
