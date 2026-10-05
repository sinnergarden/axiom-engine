"""Complete native identity and corruption checks for single-load byte reuse."""
import base64
from copy import deepcopy
import gzip
import hashlib
import json
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import canonical
from axiom_engine.runtime import stock_evidence as evidence


def batch(coverage, label='one'):
    return {'z': {'label': label}, 'context': {'z': 'last', 'coverage': coverage,
            'query': {'sessions': ['2024-01-02'], 'knowledge_cutoff': '2024-01-02T12:30:00Z'},
            'reader_version': label, 'a': 'first'},
            'records': [{'price': 1.25, 'security_id': label}],
            'field_meta': {'price': {'unit': 'CNY/share', 'available_at': '2024-01-02T12:00:00Z'}},
            'a': [True, None, 0.0]}


def raw_evidence(raw):
    ref = 'sha256:' + hashlib.sha256(raw).hexdigest()
    compressed = gzip.compress(raw, mtime=0)
    return [{'reference': 'sha256:' + '0' * 64, 'batch': {'context': {'coverage_ref': ref}}}], [{
        'reference': ref, 'encoding': evidence.ENCODING, 'uncompressed_bytes': len(raw),
        'compressed_digest': 'sha256:' + hashlib.sha256(compressed).hexdigest(),
        'payload': base64.b64encode(compressed).decode()}]


class StockEvidenceTests(unittest.TestCase):
    def setUp(self):
        # Exercise the identical codec with small adversarial fixtures.
        self.minimum = patch.object(evidence, 'MIN_COVERAGE_BYTES', 1)
        self.minimum.start()
        self.addCleanup(self.minimum.stop)

    def test_streaming_identity_matches_canonical_unicode_numbers_and_order(self):
        coverage = {'marker': 'coverage', '\U00010000': 'astral', '\ue000': 'bmp',
                    'é': 'e\u0301', 'e\u0301': 'é', '中文': ['😀', '\n\t"\\'],
                    'numbers': [0, 0.0, -0.0, 1e-7, 1e20, 1e30, 1.25, -12]}
        originals = [batch(coverage, label) for label in ('first', 'second')]
        originals[0]['\U00010000'] = {'\ue000': 1, '\U00010000': 2}
        originals[0]['\ue000'] = 'before astral in Python order'
        originals[1] = dict(reversed(list(originals[1].items())))
        originals[1]['context'] = dict(reversed(list(originals[1]['context'].items())))
        originals.append(batch({'other': [False, -0.0]}, 'third'))
        expected = [evidence.native_ref(value) for value in originals]
        entries, bundle = evidence.pack_stock_batches(originals)
        self.assertEqual(len(bundle), 2)
        loaded = evidence.native_batches(entries, bundle)
        self.assertEqual(loaded, originals)
        self.assertEqual([entry['reference'] for entry in entries], expected)
        self.assertEqual([evidence.native_ref(value) for value in loaded], expected)

    def test_coverage_verified_once_and_every_complete_batch_hashed(self):
        coverage = {'marker': 'unique_coverage', 'numbers': [-0.0, 1e-7]}
        entries, bundle = evidence.pack_stock_batches([batch(coverage, str(i)) for i in range(3)])
        checked = []
        def observe(value):
            if type(value) is dict and value.get('marker') == 'unique_coverage':
                checked.append(value)
            return canonical(value)
        with patch.object(evidence, 'canonical', side_effect=observe), patch.object(
                evidence, '_native_ref_with_verified_coverage',
                wraps=evidence._native_ref_with_verified_coverage) as whole:
            loaded = evidence.native_batches(entries, bundle)
        self.assertEqual(len(checked), 1)
        self.assertEqual(whole.call_count, 3)
        self.assertIs(loaded[0]['context']['coverage'], loaded[1]['context']['coverage'])

    def test_every_batch_unique_field_remains_bound(self):
        entries, bundle = evidence.pack_stock_batches([batch({'marker': 'same'}, label) for label in ('a', 'b')])
        changes = [lambda value: value['context']['query']['sessions'].append('2024-01-03'),
                   lambda value: value['context'].__setitem__('reader_version', 'changed'),
                   lambda value: value['records'][0].__setitem__('price', 2),
                   lambda value: value['field_meta']['price'].__setitem__('available_at', '2024-01-03T12:00:00Z'),
                   lambda value: value.__setitem__('z', {'label': 'changed'})]
        for change in changes:
            with self.subTest(change=change):
                broken = deepcopy(entries)
                change(broken[1]['batch'])
                with self.assertRaisesRegex(ValueError, 'native DataBatch hash mismatch'):
                    evidence.native_batches(broken, bundle)

    def test_repeated_load_has_no_cross_call_trust_or_mutation(self):
        entries, bundle = evidence.pack_stock_batches([batch({'items': [1, 2]})])
        before = deepcopy((entries, bundle))
        first = evidence.native_batches(entries, bundle)
        first[0]['context']['coverage']['items'].append(3)
        second = evidence.native_batches(entries, bundle)
        self.assertEqual(second[0]['context']['coverage']['items'], [1, 2])
        self.assertEqual((entries, bundle), before)
        broken = deepcopy(bundle)
        broken[0]['payload'] = 'AA=='
        with self.assertRaises(ValueError):
            evidence.native_batches(entries, broken)

    def test_old_inline_native_identity_unchanged(self):
        originals = [batch({'old_inline': True}), batch({'other': 0}, 'second')]
        entries = [{'reference': evidence.native_ref(value), 'batch': value} for value in originals]
        self.assertEqual(evidence.native_batches(entries), originals)
        broken = deepcopy(entries)
        broken[0]['batch']['context']['coverage']['old_inline'] = False
        with self.assertRaisesRegex(ValueError, 'native DataBatch hash mismatch'):
            evidence.native_batches(broken)

    def test_unknown_shape_keeps_coverage_key_in_context_validation(self):
        unknown = {'contract_type': 'Unknown', 'contract_version': '1', 'metadata': {},
                   'unknown_id': 'u', 'reason': 'missing', 'required_evidence': 'source'}
        originals = [batch({'known': unknown})]
        entries, bundle = evidence.pack_stock_batches(originals)
        self.assertEqual(evidence.native_batches(entries, bundle), originals)
        coverage_ref = entries[0]['batch']['context']['coverage_ref']
        entries[0]['batch']['context'] = {**unknown, 'coverage_ref': coverage_ref}
        with self.assertRaisesRegex(ValueError, 'Expected exact fields'):
            evidence.native_batches(entries, bundle)
        entries[0]['batch'] = {**unknown, 'context': {'coverage_ref': coverage_ref}}
        with self.assertRaisesRegex(ValueError, 'Expected exact fields'):
            evidence.native_batches(entries, bundle)

    def test_nonfinite_and_invalid_unknown_in_coverage_rejected(self):
        invalid = [{'x': float('nan')}, {'x': float('inf')}, {'x': -float('inf')},
                   {'contract_type': 'Other'}, {'contract_type': 'Unknown', 'contract_version': '1'}]
        for value in invalid:
            with self.subTest(value=value):
                entries, bundle = raw_evidence(json.dumps(value, separators=(',', ':')).encode())
                with self.assertRaises(ValueError):
                    evidence.native_batches(entries, bundle)

    def test_nonfinite_and_invalid_unknown_outside_coverage_rejected(self):
        entries, bundle = evidence.pack_stock_batches([batch({'ok': True})])
        for value in (float('nan'), float('inf'), -float('inf'), {'contract_type': 'Other'}):
            with self.subTest(value=value):
                broken = deepcopy(entries)
                broken[0]['batch']['records'][0]['price'] = value
                with self.assertRaisesRegex(ValueError, 'finite|reserved'):
                    evidence.native_batches(broken, bundle)

    def test_duplicate_or_noncanonical_coverage_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"z":1,"a":2}', b'{"a": 1}',
                    b'{"a":1.00}', b'{"a":"\\u00e9"}'):
            with self.subTest(raw=raw):
                entries, bundle = raw_evidence(raw)
                with self.assertRaisesRegex(ValueError, 'compressed coverage|noncanonical'):
                    evidence.native_batches(entries, bundle)

    def test_compressed_hash_length_and_reference_rejected(self):
        entries, bundle = evidence.pack_stock_batches([batch({'ok': [1, 2]})])
        for key, replacement in (('compressed_digest', 'sha256:' + '0' * 64),
                                 ('payload', 'AA=='), ('uncompressed_bytes', 1)):
            with self.subTest(key=key):
                broken = deepcopy(bundle)
                broken[0][key] = replacement
                with self.assertRaises(ValueError):
                    evidence.native_batches(entries, broken)
        broken = deepcopy(entries)
        broken[0]['reference'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(ValueError, 'native DataBatch hash mismatch'):
            evidence.native_batches(broken, bundle)

    def test_missing_duplicate_and_unused_closure_rejected(self):
        entries, bundle = evidence.pack_stock_batches([batch({'ok': True})])
        for sources, table in ((entries, []), (entries, bundle + bundle),
                               (entries + entries, bundle), ([], bundle)):
            with self.subTest(sources=sources, table=table):
                with self.assertRaises(ValueError):
                    evidence.native_batches(sources, table)


if __name__ == '__main__':
    unittest.main()
