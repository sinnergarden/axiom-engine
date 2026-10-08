"""Exact 2e84262 admission controls and bounded call-local work counters."""
import ast
import copy
import gc
import inspect
import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from axiom_engine.core import (ContractError, ExecutionContext, FactBatch,
    FeatureFrame, FeaturePlan, execute_feature_plan, execute_feature_plan_batch,
    validate_plan, unresolved)
from axiom_engine.core import contracts, execution
from axiom_engine.core.contracts import Document
import core_inputs_2e84262 as original
from test_conformance import col, cs, node, setup


def outcome(function, *args):
    try:
        return ('ok', function(*args))
    except Exception as error:
        return ('error', type(error), str(error))


def documents(p, f, c):
    return FeaturePlan.from_dict(p), FactBatch.from_dict(f), ExecutionContext.from_dict(c)


def frame_outcome(request, *, old=False, batch=False):
    def run():
        if batch:
            frames = execute_feature_plan_batch((request,), reuse_budget_bytes=0)['frames']
        else:
            frames = (execute_feature_plan(*request),)
        return tuple((frame.payload, frame.identity) for frame in frames)
    if old:
        with patch.object(execution, '_inputs', original.original_inputs):
            return outcome(run)
    return outcome(run)


class DocumentAdmission(unittest.TestCase):
    def test_payload_hash_and_isolation_match_original_object_entry(self):
        value = {'unicode': '证券🧭', 'nested': ({'values': [-0.0, 1, 1.0, True,
            None, sys.float_info.max, math.ulp(0.0)]},)}
        for cls in (Document, FactBatch, FeaturePlan, ExecutionContext, FeatureFrame):
            with self.subTest(cls=cls.__name__):
                old = cls(contracts.canonical(value))
                new = cls.from_dict(value)
                self.assertIs(type(new), cls)
                self.assertEqual((new.payload, new.identity), (old.payload, old.identity))
                first, second = new.to_dict(), new.to_dict()
                first['nested'][0]['values'][0] = 999
                self.assertEqual(second, old.to_dict())
                changed = copy.deepcopy(value); saved = cls.from_dict(changed)
                changed['nested'][0]['values'].append(999)
                self.assertEqual(saved.payload, old.payload)
                with self.assertRaises(FrozenInstanceError): new.payload = '{}'

    def test_object_errors_match_original_and_unknown_transport_is_unchanged(self):
        unknown = dict(contract_type='Unknown', contract_version='1', metadata={},
            unknown_id='pending', reason='missing source', required_evidence='fixed fact')
        invalid_unknowns = [dict(unknown, contract_type='Other'),
            dict(unknown, contract_version='2'), dict(unknown, metadata=[]),
            dict(unknown, reason=''), {k:v for k,v in unknown.items() if k!='unknown_id'}]
        inputs = [None, [], (), 1, 'text', {1:'key'}, {'x':float('inf')},
            {'x':float('nan')}, {'x':object()}, {'x':2**2400},
            *({'nested':item} for item in invalid_unknowns)]
        for value in inputs:
            with self.subTest(value=repr(value)):
                self.assertEqual(outcome(Document.from_dict, value),
                    outcome(lambda v:Document(contracts.canonical(v)), value))
        for value in ({'nested':unknown}, {'a':unknown, 'b':dict(unknown, reason='different')}):
            new = Document.from_dict(value); old = Document(contracts.canonical(value))
            self.assertEqual(new.payload, old.payload)
            self.assertEqual(outcome(unresolved, new), outcome(unresolved, old))

    def test_string_constructor_keeps_duplicate_key_unknown_and_finite_checks(self):
        cases = [('[]', 'Contract must be an object'),
            ('{"x":1,"x":2}', 'Duplicate JSON key: x'),
            ('{"nested":{"x":1,"x":2}}', 'Duplicate JSON key: x'),
            ('{"x":NaN}', 'Expected finite number'),
            ('{"x":Infinity}', 'Expected finite number'),
            ('{"contract_type":"Other"}', 'Unsupported reserved contract_type')]
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaises(ContractError) as raised:
                Document(payload)
            self.assertEqual(str(raised.exception), message)
        with self.assertRaisesRegex(ContractError, 'Use from_dict'): Document({})
        payload = ' { "z": 1.0, "a": [false, null, -0.0] } '
        self.assertEqual(Document(payload).payload, '{"a":[false,null,-0.0],"z":1.0}')

    def test_custom_constructor_hooks_retain_original_path(self):
        calls = []
        class Hooks(Document):
            def __new__(cls, payload):
                calls.append('new'); return super().__new__(cls)
            def __init__(self, payload):
                calls.append('init'); super().__init__(payload)
            def __post_init__(self):
                calls.append('post'); super().__post_init__()
        class Meta(type):
            def __call__(cls, *args, **kwargs):
                calls.append('meta'); return super().__call__(*args, **kwargs)
        class MetaclassDocument(Document, metaclass=Meta): pass
        class WrappedList(Document):
            def __post_init__(self):
                object.__setattr__(self, 'payload', contracts.canonical({'wrapped':contracts.json.loads(self.payload)}))
                super().__post_init__()
        class CustomRead(Document):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                if name == 'payload': return contracts.canonical({'read':contracts.json.loads(value)})
                return value
        class PayloadDescriptor(Document):
            @property
            def payload(self):
                return contracts.canonical({'descriptor':contracts.json.loads(self._payload)})
            @payload.setter
            def payload(self, value): self.__dict__['_payload'] = value
        for cls, value in ((Hooks, {'x':1}), (MetaclassDocument, {'x':1}), (WrappedList, [1]),
                (CustomRead, {'x':1}), (PayloadDescriptor, {'x':1})):
            calls.clear(); old = cls(contracts.canonical(value)); before = calls[:]
            calls.clear(); new = cls.from_dict(value)
            self.assertEqual((new.payload, calls), (old.payload, before))
        class Lifecycle(Document):
            def __del__(self): calls.append('del')
        failures = []
        for make in (lambda:Lifecycle(contracts.canonical([])), lambda:Lifecycle.from_dict([])):
            calls.clear(); failures.append(outcome(make)); gc.collect()
            self.assertEqual(calls, ['del'])
        self.assertEqual(failures[0], failures[1])

    def test_object_entry_canonical_once_without_parse_string_entry_still_parses(self):
        canonical, loads = contracts.canonical, contracts.json.loads
        with patch.object(contracts, 'canonical', wraps=canonical) as encodes, \
                patch.object(contracts.json, 'loads', wraps=loads) as parses:
            new = FactBatch.from_dict({'tuple':(1, 2), 'zero':-0.0})
            self.assertEqual((encodes.call_count, parses.call_count), (1, 0))
        with patch.object(contracts, 'canonical', wraps=canonical) as encodes, \
                patch.object(contracts.json, 'loads', wraps=loads) as parses:
            old = FactBatch(contracts.canonical({'tuple':(1, 2), 'zero':-0.0}))
            self.assertEqual((encodes.call_count, parses.call_count), (2, 1))
        self.assertEqual((new.payload, new.identity), (old.payload, old.identity))


class InputAdmission(unittest.TestCase):
    def base(self):
        return setup({'A':[-0.0, None, 3], 'B':[1, 2, 3]},
            [node('raw', 'identity', ['x']), cs('z', 'cs_zscore')], ['raw', 'z'],
            members={'A':[True,False,True], 'B':[False,False,True]})

    def assertOriginal(self, p, f, c):
        request = documents(p, f, c)
        actual = frame_outcome(request); expected = frame_outcome(request, old=True)
        self.assertEqual(actual, expected)
        if actual[0] == 'ok':
            p_wire = validate_plan(request[0], execution=True)
            self.assertEqual(execution._inputs(p_wire, *request[1:]),
                original.original_inputs(p_wire, *request[1:]))
        return actual

    def test_valid_orders_members_missing_extremes_and_industry_are_exact(self):
        for reverse in (False, True):
            p, f, c = self.base()
            if reverse:
                for rows in (f['rows'], c['reference'], c['history_keys'], c['output_keys']): rows.reverse()
            self.assertEqual(self.assertOriginal(p, f, c)[0], 'ok')
        for value in (sys.float_info.max, math.ulp(0.0), -0.0):
            p, f, c = setup({'A':[value, value]}, [node('raw','identity',['x'])])
            self.assertEqual(self.assertOriginal(p, f, c)[0], 'ok')
        p, f, c = setup({'A':[1, 2], 'B':[3, 4]}, [cs('z','cs_zscore',group='industry')],
            industries={'A':[None,'one'], 'B':['two','one']})
        self.assertEqual(self.assertOriginal(p, f, c)[0], 'ok')

    def test_cached_clock_never_authorizes_future_cells_or_later_reference(self):
        p, f, c = setup({'A':[1, 2]}, [node('raw','identity',['x'])])
        first, last = c['sessions']; available = last+'T12:00:00Z'
        for row in f['rows']: row['availability'] = [available]
        result = self.assertOriginal(p, f, c)
        self.assertEqual(result[0], 'ok')
        frame = execute_feature_plan(*documents(p, f, c)).to_dict()
        self.assertEqual(frame['rows'][0]['reasons'], [['UNAVAILABLE_AT_SESSION_CUTOFF']])
        self.assertEqual(frame['rows'][1]['values'], [2.0])
        c['reference'][0]['available_at'] = available
        self.assertEqual(self.assertOriginal(p, f, c)[1:],
            (ContractError, 'Unavailable/unbound reference'))

    def test_input_failures_preserve_original_error_and_check_order(self):
        edits = [
            lambda p,f,c:c['sessions'].reverse(),
            lambda p,f,c:c['sessions'].append(c['sessions'][0]),
            lambda p,f,c:c['cutoffs'].update({c['sessions'][0]:c['sessions'][-1]+'T23:00:00Z'}),
            lambda p,f,c:c['cutoffs'].update({c['sessions'][0]:'invalid'}),
            lambda p,f,c:c['history_keys'].append(c['history_keys'][0]),
            lambda p,f,c:c['history_keys'][0].__setitem__(1,'2030-01-01'),
            lambda p,f,c:f['rows'].append(copy.deepcopy(f['rows'][0])),
            lambda p,f,c:f['rows'].pop(),
            lambda p,f,c:f['rows'][1].update(sources=[['unbound']]),
            lambda p,f,c:f['rows'][1].update(sources=[['facts','facts']]),
            lambda p,f,c:f['rows'][0].update(values=[True],availability=['bad'],sources=[['unbound']]),
            lambda p,f,c:f['rows'][0].update(availability=['bad'],sources=[['unbound']]),
            lambda p,f,c:f['rows'][0].update(availability=[None]),
            lambda p,f,c:f['rows'][0].update(availability=[[]]),
            lambda p,f,c:f['rows'][0].update(missing_reasons=['present reason']),
            lambda p,f,c:c['reference'].append(copy.deepcopy(c['reference'][0])),
            lambda p,f,c:c['reference'].pop(),
            lambda p,f,c:c['reference'][1].update(member=1),
            lambda p,f,c:c['reference'][0].update(source='unbound'),
            lambda p,f,c:c['reference'][0].update(available_at='bad'),
            lambda p,f,c:p['reference_members'][c['sessions'][0]].pop('A'),
        ]
        for index, edit in enumerate(edits):
            p, f, c = self.base(); edit(p, f, c)
            with self.subTest(edit=index): self.assertEqual(self.assertOriginal(p,f,c)[0], 'error')
        # The same internal calendar gap must still fail for session observations.
        p, f, c = setup({'A':[1,2,3]}, [node('raw','identity',['x'])])
        for rows in (f['rows'],c['reference'],c['history_keys'],c['output_keys']): rows.pop(1)
        self.assertEqual(self.assertOriginal(p,f,c)[1:],
            (ContractError,'MISSING_SESSION: history cannot compress calendar gaps'))

    def test_event_admission_matches_original_for_clocks_sources_and_errors(self):
        p,f,c = setup({'A':[0,0,0]}, [node('event','asof',[],dict(stream='income',
            field='value',match='exact_date',report_policy='nondecreasing'))])
        p['event_schema'] = {'income':[col('value',stage='fact')]}
        f['event_schema'] = copy.deepcopy(p['event_schema'])
        day = c['sessions'][1]
        f['events'] = [dict(stream='income',security_id='A',event_id='one',event_session=day,
            report_period='2019-12-31',values=[10],availability=[day+'T12:00:00Z'],
            sources=[['facts']],missing_reasons=[None]),
            dict(stream='income',security_id='A',event_id='future',event_session='2021-01-01',
            report_period='2020-12-31',values=[999],availability=['2021-01-01T00:00:00Z'],
            sources=[['facts']],missing_reasons=[None])]
        self.assertEqual(self.assertOriginal(p,f,c)[0], 'ok')
        late = copy.deepcopy(f)
        late['events'][0]['availability'] = [c['sessions'][2]+'T12:00:00Z']
        self.assertEqual(self.assertOriginal(p,late,c)[0], 'ok')
        edits = [lambda f:f['events'].append(copy.deepcopy(f['events'][0])),
            lambda f:f['events'][1].update(event_session=day),
            lambda f:f['events'][0].update(stream='other'),
            lambda f:f['events'][0].update(event_id=''),
            lambda f:f['events'][0].update(report_period='bad'),
            lambda f:f['events'][0].update(availability=[None]),
            lambda f:f['events'][0].update(availability=['bad']),
            lambda f:f['events'][0].update(sources=[['unbound']]),
            lambda f:f['events'][0].update(values=[None],missing_reasons=[None])]
        for index, edit in enumerate(edits):
            changed = copy.deepcopy(f); edit(changed)
            with self.subTest(edit=index): self.assertEqual(self.assertOriginal(p,changed,c)[0], 'error')

    def test_single_batch_and_successive_source_cutoff_changes_use_same_path(self):
        p, f, c = setup({'A':[1,2], 'B':[3,4]}, [cs('z','cs_zscore')])
        c['output_keys'] = [[s,c['sessions'][-1]] for s in ('A','B')]
        for index in range(2):
            if index:
                source = 'revision-at-later-cutoff'
                p['sources'][0].update(id=source, qualification='observed')
                f['sources'] = copy.deepcopy(p['sources'])
                for row in f['rows']: row.update(sources=[[source]], values=[row['values'][0]+0.25])
                for row in c['reference']: row['source'] = source
                c['cutoffs'] = {d:c['sessions'][-1]+'T23:00:00Z' for d in c['sessions']}
            request = documents(p,f,c)
            self.assertEqual(frame_outcome(request), frame_outcome(request,old=True))
            self.assertEqual(frame_outcome(request,batch=True), frame_outcome(request,old=True,batch=True))
            self.assertEqual(frame_outcome(request), frame_outcome(request,batch=True))

    def test_timestamp_cache_is_bounded_and_cleared_between_admissions(self):
        history, fields = 150, 5
        p, f, c = setup({s:[[float(j)]*fields for j in range(history)] for s in ('A','B')},
            [node('raw','identity',['x0'])], schema=[col('x'+str(i),stage='fact') for i in range(fields)])
        request = documents(p,f,c); plan = validate_plan(request[0], execution=True)
        timestamp, once = execution.timestamp, execution._timestamp_once
        sizes = []; calls = [0]
        def counted(value):
            calls[0] += 1; return timestamp(value)
        def bounded(value, checked):
            result = once(value, checked); sizes.append(len(checked)); return result
        counts = []
        with patch.object(execution, 'timestamp', counted), patch.object(execution, '_timestamp_once', bounded):
            for _ in range(2):
                before = calls[0]; execution._inputs(plan,*request[1:]); counts.append(calls[0]-before)
        self.assertEqual(max(sizes), 128)
        # Even a calendar larger than the memo must leave repeated field clocks
        # in each row reusable: 150 cutoff + 300 row + 300 reference parses.
        self.assertEqual(counts, [5*history,5*history])

    def test_389_and_622_history_work_counts_are_linear_and_exact(self):
        history, fields = 21, 5
        for width in (389,622):
            p,f,c = setup({f'S{i:03d}':[[float(j)]*fields for j in range(history)] for i in range(width)},
                [node('raw','identity',['x0'])], schema=[col('x'+str(i),stage='fact') for i in range(fields)])
            request = documents(p,f,c); plan = validate_plan(request[0],execution=True)
            counts = {'old_timestamp':0,'new_timestamp':0,'old_reference_visits':0,'new_reference_visits':0}
            original_timestamp = contracts.timestamp
            def timestamp(kind):
                def check(value):
                    counts[kind] += 1; return original_timestamp(value)
                return check
            # Wrap only the frozen oracle's reference iterator, without tracing
            # all scalar admission calls (or changing its predicate/order).
            class CountItems(ast.NodeTransformer):
                def visit_Call(self, item):
                    self.generic_visit(item)
                    if (isinstance(item.func, ast.Attribute) and item.func.attr=='items' and
                            isinstance(item.func.value, ast.Name) and item.func.value.id=='refs'):
                        return ast.copy_location(ast.Call(func=ast.Name(id='_count_items',ctx=ast.Load()),
                            args=[item.func.value],keywords=[]),item)
                    return item
            def count_items(refs):
                for item in refs.items():
                    counts['old_reference_visits'] += 1; yield item
            namespace = dict(vars(original), timestamp=timestamp('old_timestamp'), _count_items=count_items)
            tree = CountItems().visit(ast.parse(inspect.getsource(original.original_inputs)))
            exec(compile(ast.fix_missing_locations(tree), '<original reference iterator>', 'exec'), namespace)
            class CountRefs(dict):
                def items(self):
                    for item in super().items():
                        counts['new_reference_visits'] += 1; yield item
            members = execution._reference_members_by_day
            with patch.object(original,'timestamp',timestamp('old_timestamp')):
                expected = namespace['original_inputs'](plan,*request[1:])
            with patch.object(execution,'timestamp',timestamp('new_timestamp')), \
                    patch.object(execution,'_reference_members_by_day',lambda refs:members(CountRefs(refs))):
                actual = execution._inputs(plan,*request[1:])
            with self.subTest(width=width):
                self.assertEqual(actual,expected)
                self.assertEqual(counts['old_reference_visits'],width*history*history)
                self.assertEqual(counts['new_reference_visits'],width*history)
                self.assertEqual(counts['old_timestamp'],history+width*history*(fields+1))
                self.assertEqual(counts['new_timestamp'],3*history)


if __name__ == '__main__': unittest.main()
