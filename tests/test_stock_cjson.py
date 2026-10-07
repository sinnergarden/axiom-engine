"""Opt-in one-file helper: original canonical gates, exact accounts, failure cleanup.

The helper tests require read-only process RSS visibility. No Data/ML input runs.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, canonical
from axiom_engine.runtime import StockInputSource
from axiom_engine.runtime.stock_stream_contracts import logical_ref, read_budget
from axiom_engine.runtime.stock_stream_inputs import _CanonicalIndex, _DecodedMemory
from axiom_engine.runtime import stock_cjson as fast
from test_stock_owned_inputs import request_for, imported, run_owned
from test_stock_stream_inputs import LIMITS, rebind_native


OPTIONS = dict(max_file_bytes=2_000_000, max_tree_rss_bytes=2*1024**3,
               max_spool_bytes=2_000_000, max_seconds=10)


def source(**changes):
    options = dict(file_parse_mode='cjson', max_file_parse_bytes=OPTIONS['max_file_bytes'],
        max_file_parse_rss_bytes=OPTIONS['max_tree_rss_bytes'],
        max_file_parse_spool_bytes=OPTIONS['max_spool_bytes'], max_file_parse_seconds=10)
    options.update(changes)
    return StockInputSource(**options)


def mixed_coverage(wire, count):
    wire['context'].setdefault('coverage', {})['synthetic_mixed'] = [
        dict(status='complete', repeated='same-value', ordinal=i,
             unique_ref='sha256:'+hashlib.sha256(str(i).encode()).hexdigest(),
             value=(i+1)/100003, label='合成-'+str(i)) for i in range(count)]


def replace_bytes(manifest, name, raw):
    entry = next(item for item in manifest['market_input']['native_inputs']
                 if item['artifact']['artifact_id'] == name)
    path = Path(entry['artifact']['manifest_uri']); path.write_bytes(raw)
    reference = 'sha256:'+hashlib.sha256(raw[:-1] if raw.endswith(b'\n') else raw).hexdigest()
    for item in manifest['market_input']['native_inputs']:
        if item['artifact']['manifest_uri'] == str(path):
            item['artifact']['content_digest'] = item['native_ref'] = reference
    manifest['market_input']['market_ref'] = logical_ref(manifest['market_input'], 'market_ref')
    manifest['request_ref'] = logical_ref(manifest, 'request_ref')
    return entry['artifact']


class StockCjsonConfigurationTests(unittest.TestCase):
    def test_defaults_and_all_explicit_positive_integer_budgets(self):
        self.assertEqual(StockInputSource()._file_parse_mode, 'stream')
        self.assertEqual(StockInputSource()._scalar_cache_bytes, 0)
        with self.assertRaises(ContractError):
            StockInputSource(file_parse_mode='cjson')
        for name in ('max_file_parse_bytes', 'max_file_parse_rss_bytes',
                     'max_file_parse_spool_bytes', 'max_file_parse_seconds'):
            for value in (True, 0, -1, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ContractError):
                    source(**{name: value})
        with self.assertRaises(ContractError):
            source(file_parse_mode='other')


class StockCjsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            fast._processes()
        except (OSError, subprocess.SubprocessError, ContractError):
            raise unittest.SkipTest('CJSON requires permitted read-only process RSS monitoring')

    def index(self, artifact, manifest, events=None, **changes):
        return fast.cjson_native_index(artifact, read_budget(LIMITS), row_limit=1000,
            native_scope=(manifest, LIMITS), memory=_DecodedMemory(LIMITS['max_block_bytes']),
            options={**OPTIONS, **changes}, spool_used=0, events=[] if events is None else events)

    def test_full_cold_admission_and_top3_top5_complete_saved_output_match_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            rebind_native(manifest, 'native-4', lambda wire: mixed_coverage(wire, 250))
            old_source = StockInputSource(); new_source = source()
            with imported(manifest, old_source) as old, imported(manifest, new_source) as new:
                self.assertEqual(new.statistics['source_operations']['cjson_files'], 9)
                self.assertEqual({event['phase'] for event in new_source._file_parse_events}, {'complete'})
                for k in (3, 5):
                    candidate = deepcopy(manifest); candidate['account_id'] += '-top'+str(k)
                    candidate['portfolio_policy']['top_k'] = k
                    candidate['request_ref'] = logical_ref(candidate, 'request_ref')
                    output = root/('top'+str(k))
                    expected, ep = run_owned(output, candidate, old)
                    shutil.rmtree(output)
                    actual, ap = run_owned(output, candidate, new)
                    self.assertEqual(actual, expected)
                    self.assertEqual(ap.rows, ep.rows)
                self.assertEqual(new.statistics['source_admissions'], 1)
                self.assertEqual(new.statistics['account_bindings'], 2)
            self.assertEqual(new_source._indexes, {})

    def test_headers_groups_rows_original_identity_utf8_lf_and_no_native_scanner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = request_for(root)
            rebind_native(manifest, 'native-4', lambda wire: mixed_coverage(wire, 10))
            entry = next(item for item in manifest['market_input']['native_inputs']
                         if item['artifact']['artifact_id'] == 'native-4')
            path = Path(entry['artifact']['manifest_uri'])
            artifact = replace_bytes(manifest, 'native-4', path.read_bytes()+b'\n')
            old = _CanonicalIndex(artifact, read_budget(LIMITS), row_limit=1000,
                native_scope=(manifest, LIMITS))
            with patch.object(_CanonicalIndex, '__init__', side_effect=AssertionError('old scanner reused')):
                new = self.index(artifact, manifest)
            try:
                self.assertEqual(new.content_digest, old.content_digest)
                self.assertEqual(new.file_digest, old.file_digest)
                self.assertEqual(new.header(), old.header())
                self.assertEqual(new.metadata_header(), old.metadata_header())
                self.assertEqual(new.group_hashes, old.group_hashes)
                for key, day in old.groups:
                    with old._memory.stage() as a, new._memory.stage() as b:
                        self.assertEqual(new.rows(key, [day], reservation=b), old.rows(key, [day], reservation=a))
                self.assertEqual(new.statistics['cjson_compare_bytes'], path.stat().st_size)
                for action in (new.whole, lambda: new.span_digest(('context',)),
                               lambda: new.unsigned_digest('content_digest')):
                    with self.assertRaises(ContractError):
                        action()
            finally:
                new.close()
            with self.assertRaisesRegex(ContractError, 'closed'):
                new._open_data()

    def test_full_bytes_counterexamples_reject_in_both_paths(self):
        invalid_unknown = dict(contract_type='Unknown', contract_version='1', metadata={},
            unknown_id='x', reason=' ', required_evidence='x')
        nested = 0
        for _ in range(128):
            nested = [nested]
        bad = [b'{"a":1,"a":1}', b'{"b":1,"a":2}', b'{"a": 1}', b'{"a":1.00}',
               b'{"a":-0}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}',
               b'{"a":"\\u4e2d"}', b'{"a":"\\ud800"}', b'{"a":"\xff"}',
               b'{}\n\n', b'{} ', b'{}x',
               json.dumps({'a': invalid_unknown}, sort_keys=True, separators=(',', ':')).encode(),
               json.dumps({'a': {'contract_type': 'other'}}, sort_keys=True, separators=(',', ':')).encode(),
               json.dumps({'a': nested}, separators=(',', ':')).encode()]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = request_for(Path(tmp))
            for raw in bad:
                with self.subTest(raw=raw[:80]):
                    artifact = replace_bytes(manifest, 'native-4', raw)
                    with self.assertRaises((ContractError, UnicodeError, ValueError)):
                        _CanonicalIndex(artifact, read_budget(LIMITS), row_limit=1000,
                            native_scope=(manifest, LIMITS))
                    events = []
                    with self.assertRaisesRegex(ContractError, 'no stream retry'):
                        self.index(artifact, manifest, events)
                    self.assertEqual(events[-1]['phase'], 'failed')

    def test_original_native_and_business_gates_and_all_spools_release_on_failure(self):
        changes = [('native-4', lambda w: w['context']['query'].update(symbols=[])),
            ('native-1', lambda w: w['records'].append(w['records'][0])),
            ('native-1', lambda w: w['records'][0].update(session='1999-01-01')),
            ('native-1', lambda w: w['field_meta'].update(undeclared={'by_key': [w['records'][0]]})),
            ('warmup-1', lambda w: w['context']['query'].update(sessions=['2024-01-01'])),
            ('native-1', lambda w: w['field_meta']['close'].update(unit='wrong'))]
        for name, change in changes:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                manifest = request_for(Path(tmp)); rebind_native(manifest, name, change)
                with self.assertRaises(ContractError):
                    with imported(manifest):
                        pass
                candidate = source(); opened = []; original = tempfile.TemporaryFile
                def track(*args, **kwargs):
                    value = original(*args, **kwargs); opened.append(value); return value
                with patch.object(fast.tempfile, 'TemporaryFile', side_effect=track):
                    with self.assertRaises(ContractError):
                        imported(manifest, candidate)
                self.assertTrue(opened)
                self.assertTrue(all(value.closed for value in opened))

    def test_preflight_stream_choices_are_before_any_helper_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = request_for(Path(tmp)); artifact = manifest['market_input']['native_inputs'][0]['artifact']
            for changes in ({'max_file_bytes': 1}, {'max_tree_rss_bytes': 1}):
                events = []
                monitor = fast._processes()
                with patch.object(fast, '_processes', return_value=monitor), \
                        patch.object(fast.subprocess, 'Popen', side_effect=AssertionError('helper launched')):
                    self.assertIsNone(self.index(artifact, manifest, events, **changes))
                self.assertEqual(events[-1]['phase'], 'preflight')
            events = []
            with patch.object(fast, '_processes', side_effect=OSError('monitor unavailable')):
                self.assertIsNone(self.index(artifact, manifest, events))
            self.assertEqual(events[-1]['backend'], 'stream')
            fallback = source(max_file_parse_bytes=1)
            with imported(manifest, fallback) as handle:
                self.assertEqual(handle.statistics['source_operations']['cjson_files'], 0)
            self.assertEqual({e['phase'] for e in fallback._file_parse_events}, {'preflight'})

    def test_started_memoryerror_timeout_rss_and_spool_stops_never_retry_and_close(self):
        real_popen = subprocess.Popen; real_tree = fast._tree_rss
        with tempfile.TemporaryDirectory() as tmp:
            manifest = request_for(Path(tmp)); artifact = manifest['market_input']['native_inputs'][0]['artifact']
            for mode in ('memoryerror', 'timeout', 'rss', 'spool'):
                events = []; opened = []; original = tempfile.TemporaryFile
                def track(*args, **kwargs):
                    value = original(*args, **kwargs); opened.append(value); return value
                def launch(command, **kwargs):
                    # Replace only the fixture's helper process; monitor and cleanup are real.
                    if command[2:4] == ['-m', 'axiom_engine.runtime.stock_cjson_worker'] and mode != 'spool':
                        script = ('import os,sys,time; os.read(int(sys.argv[1]),1); '
                            + ('sys.stderr.write("MemoryError synthetic worker failure"); sys.exit(1)'
                               if mode == 'memoryerror' else 'time.sleep(3)'))
                        command = [command[0], '-B', '-c', script, command[-1]]
                    return real_popen(command, **kwargs)
                count = [0]
                def rss(values, root):
                    count[0] += 1
                    return 3*1024**3 if mode == 'rss' and count[0] >= 3 else real_tree(values, root)
                changes = {'max_seconds': 1} if mode == 'timeout' else {'max_spool_bytes': 1} if mode == 'spool' else {}
                with self.subTest(mode=mode), patch.object(fast.tempfile, 'TemporaryFile', side_effect=track), \
                        patch.object(fast.subprocess, 'Popen', side_effect=launch), \
                        patch.object(fast, '_tree_rss', side_effect=rss), \
                        patch.object(_CanonicalIndex, '__init__', side_effect=AssertionError('stream retry')):
                    with self.assertRaisesRegex(ContractError, 'helper failed|wall-time stop|RSS stop'):
                        self.index(artifact, manifest, events, **changes)
                self.assertEqual(events[-1]['phase'], 'failed')
                self.assertTrue(all(value.closed for value in opened))


if __name__ == '__main__':
    unittest.main()
