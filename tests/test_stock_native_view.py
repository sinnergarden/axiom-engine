"""Synthetic physical carriers; never invoke Data producers or suppliers.

Integration cases run when the separately pinned Data development dependency
provides open_native_view. Engine itself has no import/dependency on Data.
"""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document, canonical
from axiom_engine.runtime import (BacktestRequest, StockInputSource, StockMarketSpec,
    StockNativeViewSource, admit_stock_market_inputs, bind_stock_prediction_inputs)
from axiom_engine.runtime.stock_market import (EVENT_KEY, STOCK_EVENT_FIELDS,
    _StockActionStream, _project_stock_actions)
from axiom_engine.runtime.stock_stream_inputs import _DecodedMemory
from test_stock_owned_inputs import request_for, run_owned
from test_stock_market_owner import seal_request, new_basis
from test_stock_stream import execute
from test_stock_stream_inputs import LIMITS, rebind_native
import test_stocks as legacy

try:
    from axiom_data import open_native_view
except ImportError:
    open_native_view = None

ACTIVE = 16_000_000
ADAPTER_LIMITS = {**LIMITS, 'max_block_bytes': ACTIVE}
DATA_LIMITS = dict(max_part_bytes=1_000_000, max_working_bytes=32_000_000,
    max_saved_bytes=2_000_000, max_rows_per_block=7)


def synthetic_request(root, *, actions=False):
    plan = request_for(root)
    prices = Document((root/'native-1.json').read_text()).to_dict()
    close = {(r['security_id'], r['session']): r['close'] for r in prices['records']}
    names = {i['artifact']['artifact_id'] for i in plan['market_input']['native_inputs']}
    for name in sorted(names):
        def normalize(value):
            value['context']['coverage'] = {'synthetic': True, 'gap_count': 0}
            if value['context']['domain'] == 'market_state_diagnostics':
                for row in value['records']:
                    row['close'] = close[row['security_id'], row['session']]
                value['field_meta']['close'] = deepcopy(prices['field_meta']['close'])
            if value['context']['domain'] == 'corporate_actions':
                value['context']['logical_key'] = list(EVENT_KEY)
                if actions:
                    events = []
                    for security in value['context']['query']['symbols'][:2]:
                        events.append({**legacy.cash_event(), 'security_id': security})
                        events.append({**legacy.cash_event('预案'), 'security_id': security,
                                       'report_period': '2023-09-30'})
                    with patch.object(legacy, 'DAYS', plan['scope']['calendar']), \
                            patch.object(legacy, 'SECURITIES', plan['scope']['execution_universe']):
                        action = legacy.batch('corporate_actions', STOCK_EVENT_FIELDS, events,
                            {'cash_dividend_before_tax_per_share': 'CNY/share'},
                            time_field=value['context']['query']['time_field'])
                    value.update(action)
                    value['context'].update(coverage={'synthetic': True, 'gap_count': 0}, logical_key=list(EVENT_KEY))
        rebind_native(plan, name, normalize)
    return seal_request(plan)


def carrier(root, plan, *, part_rows=7, roles=None):
    """Write tiny test JSON parts directly, respecting the public v1 shape."""
    root.mkdir(); files = {}; batches = []; refs = set()
    def save(value, kind):
        raw = canonical(value).encode(); digest = 'sha256:'+sha256(raw).hexdigest()
        uri = kind+'/'+digest[7:]+'.json'
        if uri not in files:
            (root/kind).mkdir(exist_ok=True); (root/uri).write_bytes(raw)
            files[uri] = {'sha256': digest, 'bytes': len(raw), 'kind': kind}
        return {'uri': uri, 'sha256': digest, 'bytes': len(raw)}
    for entry in plan['market_input']['native_inputs']:
        if (roles is not None and entry['role'] not in roles) or entry['native_ref'] in refs:
            continue
        refs.add(entry['native_ref'])
        wire = Document(Path(entry['artifact']['manifest_uri']).read_text()).to_dict()
        context = deepcopy(wire['context']); coverage = save(context.pop('coverage'), 'coverage')
        event = context['domain'] == 'corporate_actions'
        method = ('events' if event else 'states' if 'market_state' in wire['field_meta'] else
                  'members' if context['domain'] == 'universe_membership' else 'read_market')
        blocks = []
        for ordinal in range(0, len(wire['records']), part_rows):
            rows = wire['records'][ordinal:ordinal+part_rows]
            blocks.append({'ordinal':ordinal, 'rows':len(rows), 'records':save(rows, 'array'),
                'by_key':{n:save(h['by_key'][ordinal:ordinal+part_rows], 'array') for n,h in wire['field_meta'].items()},
                'sessions':list(dict.fromkeys(r['session'] for r in rows)) if not event else []})
        batches.append({'context':context, 'coverage':coverage,
            'field_meta':{n:{k:v for k,v in h.items() if k != 'by_key'} for n,h in wire['field_meta'].items()},
            'row_key':list(EVENT_KEY) if event else ['security_id','session'],
            'row_count':len(wire['records']), 'blocks':blocks, 'method':method, 'native_ref':entry['native_ref']})
    manifest = {'contract_version':'data_native_view_v1', 'logical_contract_version':'data_batch_v1',
        'exporter_version':'native_json_parts_v1', 'snapshot_id':batches[0]['context']['snapshot_id'],
        'batches':batches, 'files':files, 'statistics':{'synthetic':True}}
    raw = canonical(manifest).encode(); (root/'manifest.json').write_bytes(raw)
    return 'sha256:'+sha256(raw).hexdigest(), manifest


def market(plan, source, limits=ADAPTER_LIMITS, block_sessions=2):
    return admit_stock_market_inputs(StockMarketSpec.from_request(BacktestRequest.from_dict(plan)),
        source=source, limits=limits, block_sessions=block_sessions, max_market_bytes=2_000_000)


@unittest.skipUnless(open_native_view, 'requires separately pinned Data native_view development dependency')
class NativeCarrierTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, 'fork'), 'requires fork ownership counterexample')
    def test_inherited_scope_exit_rejects_before_cleanup_and_parent_remains_usable(self):
        class ForbiddenLock:
            def acquire(self, **kwargs): raise AssertionError('child touched lock')
            def release(self): raise AssertionError('child released lock')
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root)
            ref, _ = carrier(root/'carrier', plan)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            scope = source.execution_scope(); scope.__enter__()
            read_fd, write_fd = os.pipe(); child = os.fork()
            if child == 0:
                os.close(read_fd); source._lease = ForbiddenLock(); answers = []
                for action in (source.close, lambda:source.execution_scope().__enter__(),
                        lambda:scope.__exit__(None,None,None), lambda:source.statistics):
                    try: action(); answers.append('ACCEPTED')
                    except Exception as exc: answers.append(str(exc))
                os.write(write_fd, json.dumps(answers).encode()); os.close(write_fd); os._exit(0)
            os.close(write_fd)
            try:
                answers = json.loads(os.read(read_fd, 4096)); _, status = os.waitpid(child, 0)
                self.assertEqual(status, 0)
                self.assertTrue(all('creating process' in answer for answer in answers), answers)
            finally:
                os.close(read_fd); scope.__exit__(None,None,None)
            owner = stack.enter_context(market(plan, source))
            self.assertEqual(owner.statistics['source_admissions'], 1)

    def test_new_native_basis_provenance_checks_and_reuse_after_handles_close(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); first = synthetic_request(root)
            good = new_basis(root/'basis', first)
            expected, projection, _ = execute(root/'oracle', good, limits=ADAPTER_LIMITS)
            changes = {
                'unit': lambda v:v['field_meta']['high'].update(unit='USD/share') if 'high' in v['field_meta'] else None,
                'source': lambda v:v['field_meta']['close']['by_key'][0].update(raw_batch_id='different') if 'close' in v['field_meta'] else None,
                'pit': lambda v:v['field_meta']['close']['by_key'][0].update(usable_from='2099-01-01T00:00:00Z') if 'close' in v['field_meta'] else None,
                'query': lambda v:v['context']['query'].update(price_basis='adjusted') if 'close' in v['field_meta'] else None}
            bad = [new_basis(root/name, first, change) for name,change in changes.items()]
            ref, _ = carrier(root/'market-carrier', first)
            handle = stack.enter_context(open_native_view(root/'market-carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            owner = stack.enter_context(market(first, source))
            handle.close()
            for n,plan in enumerate(bad+[good]):
                basis_ref, _ = carrier(root/('basis-carrier-'+str(n)), plan, roles={'prediction_basis'})
                with open_native_view(root/('basis-carrier-'+str(n)), manifest_sha256=basis_ref, limits=DATA_LIMITS) as basis_handle, \
                        StockNativeViewSource(native_views=[basis_handle], max_active_bytes=ACTIVE) as basis_source:
                    if n < len(bad):
                        with self.assertRaises(ContractError):
                            bind_stock_prediction_inputs(owner, BacktestRequest.from_dict(plan), source=basis_source,
                                limits=ADAPTER_LIMITS, max_signal_bytes=2_000_000)
                        self.assertEqual(owner.statistics['basis_proofs'], 0)
                        self.assertEqual(basis_source._indexes, {})
                    else:
                        inputs = stack.enter_context(bind_stock_prediction_inputs(owner, BacktestRequest.from_dict(plan),
                            source=basis_source, limits=ADAPTER_LIMITS, max_signal_bytes=2_000_000))
            for entry in good['market_input']['native_inputs']:
                Path(entry['artifact']['manifest_uri']).unlink(missing_ok=True)
            actual, actual_projection = run_owned(root/'actual', good, inputs, ADAPTER_LIMITS)
            self.assertEqual(actual_projection.rows, projection.rows)
            self.assertEqual(actual['source_audit']['market_ref'], expected['source_audit']['market_ref'])
            self.assertEqual(actual['source_audit']['prediction_ref'], expected['source_audit']['prediction_ref'])
            with bind_stock_prediction_inputs(owner, BacktestRequest.from_dict(good), source=StockInputSource(),
                    limits=ADAPTER_LIMITS, max_signal_bytes=2_000_000):
                self.assertEqual(owner.statistics['basis_reuses'], 1)

    def test_business_exact_and_only_physical_inventory_changes(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root, actions=True)
            expected, projection, _ = execute(root/'old', plan, limits=ADAPTER_LIMITS)
            ref, physical = carrier(root/'carrier', plan, part_rows=1)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            cold = deepcopy(handle.statistics); starts = []; original = handle.iter_blocks
            def observed(**kwargs):
                starts.append(kwargs); yield from original(**kwargs)
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            inv = source.inventory(StockMarketSpec.from_request(BacktestRequest.from_dict(plan)))
            inv['files'][0]['file_bytes'] = 0
            self.assertNotEqual(source.inventory(StockMarketSpec.from_request(BacktestRequest.from_dict(plan)))['files'][0]['file_bytes'], 0)
            for path in {i['artifact']['manifest_uri'] for i in plan['market_input']['native_inputs']}:
                Path(path).unlink()  # Old full-native JSON must never be reopened.
            with patch.object(handle, 'iter_blocks', side_effect=observed), patch.object(handle, '_unchanged', wraps=handle._unchanged) as scans:
                owner = stack.enter_context(market(plan, source))
            self.assertEqual(len(starts), len(physical['batches']))
            self.assertEqual(scans.call_count, 2*len(starts))
            self.assertEqual(handle.statistics['logical_replays'], cold['logical_replays'])
            self.assertEqual(handle.statistics['coverage_validations'], cold['coverage_validations'])
            self.assertLessEqual(owner.statistics['source_decoded_bytes_peak'], ACTIVE)
            inputs = stack.enter_context(bind_stock_prediction_inputs(owner, BacktestRequest.from_dict(plan),
                source=StockInputSource(), limits=ADAPTER_LIMITS, max_signal_bytes=2_000_000))
            selected = deepcopy(handle.statistics)
            handle.close()
            actual, actual_projection = run_owned(root/'new', plan, inputs, ADAPTER_LIMITS)
            self.assertEqual(actual_projection.rows, projection.rows)
            self.assertEqual(handle.statistics, selected)
            old_audit, new_audit = deepcopy(expected['source_audit']), deepcopy(actual['source_audit'])
            differences = {n:(old_audit['counts'].pop(n), new_audit['counts'].pop(n)) for n in ('input_bytes','input_files')}
            self.assertTrue(any(a != b for a,b in differences.values()))
            self.assertEqual(new_audit, old_audit)
            self.evidence = {'physical_inventory_differences':differences, 'business_audit_exact':True,
                'business_groups_exact':sorted(actual_projection.rows), 'native_streams':len(starts),
                'stream_inventory_checks':scans.call_count, 'cold_logical_replays':cold['logical_replays'],
                'warm_logical_replays_added':handle.statistics['logical_replays']-cold['logical_replays'],
                'warm_coverage_validations_added':handle.statistics['coverage_validations']-cold['coverage_validations'],
                'market_source_decoded_peak':owner.statistics['source_decoded_bytes_peak'], 'max_active_bytes':ACTIVE,
                'selected_reads':selected.get('selected_reads',0), 'selected_bytes':selected.get('selected_bytes',0)}
            def output_locations(value, directory):
                if type(value) is dict:
                    return {k:(v.replace(str(directory), '<saved-output>') if k == 'manifest_uri' and type(v) is str
                               else output_locations(v, directory)) for k,v in value.items()}
                if type(value) is list:
                    return [output_locations(v, directory) for v in value]
                return value
            expected = output_locations(expected, root/'old')
            actual = output_locations(actual, root/'new')
            for n in ('content_digest','source_audit_ref','source_audit'):
                expected.pop(n, None); actual.pop(n, None)
            self.assertEqual(actual, expected)

    def test_constructor_and_next_block_budget_fail_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root)
            ref, _ = carrier(root/'carrier', plan)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            with patch.object(type(handle), 'manifest', new_callable=__import__('unittest').mock.PropertyMock) as header:
                with self.assertRaisesRegex(ContractError, 'before cache allocation'):
                    StockNativeViewSource(native_views=[handle], max_active_bytes=1)
                header.assert_not_called()
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            from axiom_engine.runtime.stock_native_view import _NativeIndex
            entry = next(i for i in plan['market_input']['native_inputs'] if i['artifact']['artifact_id'] == 'native-1')
            delivery, batch = source._native_batches[entry['native_ref']]
            source._memory = _DecodedMemory(ACTIVE)
            with source.execution_scope():
                index = _NativeIndex(source, entry['artifact'], delivery, batch, plan)
                source._indexes['probe'] = index; source._attach()
                source._memory.reserve_global('occupy', source._memory.available-1)
                before = deepcopy(handle.statistics)
                with self.assertRaisesRegex(ContractError, 'before'):
                    index._load()
                self.assertEqual(handle.statistics, before)
                self.assertIsNone(index._pending)
                self.assertEqual(source._memory.temporary_bytes, 0)
            self.assertIsNotNone(handle.manifest)  # Source never owns caller close.

    def test_failed_growth_closes_streams_and_source_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root)
            ref, _ = carrier(root/'carrier', plan)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            from axiom_engine.runtime import stock_native_view as adapter
            original = adapter.deepcopy
            def fail_window(value):
                if type(value) is tuple:
                    raise MemoryError('synthetic returned-window failure')
                return original(value)
            with patch.object(adapter, 'deepcopy', side_effect=fail_window):
                with self.assertRaisesRegex(MemoryError, 'returned-window'):
                    market(plan, source)
            self.assertEqual(source._indexes, {}); self.assertIsNone(source._memory)
            self.assertIsNone(source._thread)
            owner = stack.enter_context(market(plan, source))
            self.assertEqual(owner.statistics['source_admissions'], 1)

    def test_missing_ref_and_returned_parent_or_metadata_cannot_fallback(self):
        for damage, message in ((lambda b:b.update(native_ref='sha256:'+'f'*64), 'parent/context/ordinals'),
                (lambda b:b['context']['query'].update(pit_policy='invalid'), 'parent/context/ordinals'),
                (lambda b:b['field_meta']['open'].update(unit='USD/share'), 'by_key alignment'),
                (lambda b:b['ordinals'].reverse(), 'parent/context/ordinals')):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp); plan = synthetic_request(root)
                ref, _ = carrier(root/'carrier', plan)
                handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
                source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
                original = handle.iter_blocks
                def corrupt(**kwargs):
                    for block in original(**kwargs):
                        if 'open' in block['field_meta']: damage(block)
                        yield block
                with patch.object(handle, 'iter_blocks', side_effect=corrupt):
                    with self.assertRaisesRegex(ContractError, message): market(plan, source)
                self.assertIsNone(source._memory); self.assertEqual(source._indexes, {})
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root)
            ref, _ = carrier(root/'carrier', plan)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            entry = plan['market_input']['native_inputs'][0]
            entry['native_ref'] = entry['artifact']['content_digest'] = 'sha256:'+'f'*64
            seal_request(plan)
            with self.assertRaisesRegex(ContractError, 'missing logical ref'): market(plan, source)

    def test_process_exclusive_scope_and_closed_borrow_checks_release_lease(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); plan = synthetic_request(root)
            ref, _ = carrier(root/'carrier', plan)
            handle = stack.enter_context(open_native_view(root/'carrier', manifest_sha256=ref, limits=DATA_LIMITS))
            source = stack.enter_context(StockNativeViewSource(native_views=[handle], max_active_bytes=ACTIVE))
            with patch('axiom_engine.runtime.stock_native_view.os.getpid', return_value=source._pid+1):
                with self.assertRaisesRegex(ContractError, 'creating process'): source.execution_scope().__enter__()
            with source.execution_scope():
                with self.assertRaisesRegex(ContractError, 'sequential admission'): source.execution_scope().__enter__()
                with self.assertRaisesRegex(ContractError, 'during admission'): source.close()
            with self.assertRaisesRegex(Exception, 'native view closed'):
                with source.execution_scope(): handle.close()
            self.assertIsNone(source._thread)
            self.assertFalse(source._lease.locked())


class StreamedActionsTests(unittest.TestCase):
    def views(self, events):
        return [legacy.batch('corporate_actions', STOCK_EVENT_FIELDS, deepcopy(events),
                    {'cash_dividend_before_tax_per_share':'CNY/share'}, time_field=t)
                for t in ('ex_date','record_date')]

    def stream(self, batch, closed):
        def blocks():
            try:
                for n in range(len(batch['records'])):
                    yield {'context':batch['context'], 'records':batch['records'][n:n+1],
                        'field_meta':{name:{**h,'by_key':h['by_key'][n:n+1]} for name,h in batch['field_meta'].items()}}
            finally: closed.append(True)
        return _StockActionStream(context=batch['context'], field_meta=batch['field_meta'],
            row_count=len(batch['records']), blocks=blocks)

    def project(self, batches, **kwargs):
        return _project_stock_actions(batches=batches, universe=legacy.SECURITIES,
            source_refs=['a','b','c','d','ex-original-ref','record-original-ref'], **kwargs)

    def test_exact_cash_refs_and_diagnostic_order_across_blocks(self):
        events = [legacy.cash_event(), {**legacy.cash_event('预案'),'report_period':'2023-09-30'}]
        views = self.views(events); expected = self.project(views); closed = []
        memory = _DecodedMemory(100_000)
        with memory.stage() as reservation:
            actual = self.project([self.stream(v, closed) for v in views], reservation=reservation)
            self.assertEqual(actual, expected)
        self.assertEqual(closed, [True, True]); self.assertEqual(memory.temporary_bytes, 0)
        self.assertEqual(actual[0][0]['source_refs'], ['ex-original-ref','record-original-ref'])

    def test_global_duplicate_and_cross_query_conflict_rejected(self):
        for duplicate in (True, False):
            views = self.views([legacy.cash_event()]); closed = []
            if duplicate:
                views = self.views([legacy.cash_event(), legacy.cash_event()])
            else:
                views[1]['records'][0]['cash_dividend_before_tax_per_share'] = 0.2
            memory = _DecodedMemory(100_000)
            with memory.stage() as reservation, self.assertRaisesRegex(ContractError, 'duplicate native|conflicting native'):
                self.project([self.stream(v, closed) for v in views], reservation=reservation)
            self.assertTrue(closed); self.assertEqual(memory.temporary_bytes, 0)

    def test_budget_rejection_finally_closes_borrowed_event_iterator(self):
        closed = []; views = self.views([legacy.cash_event()]); memory = _DecodedMemory(100_000)
        with memory.stage() as reservation, self.assertRaisesRegex(ContractError, 'budget exceeded|exceeds budget'):
            self.project([self.stream(v, closed) for v in views], max_event_bytes=100, reservation=reservation)
        self.assertEqual(closed, [True]); self.assertEqual(memory.temporary_bytes, 0)


if __name__ == '__main__':
    unittest.main()
