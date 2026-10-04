"""Bundle-local deduplication of large stock source coverage, preserving native refs."""
import base64
import gzip
import hashlib
import io
import json

from ..core.contracts import canonical, digest, fields, require, _pairs

ENCODING = 'gzip_base64_json_v1'
MIN_COVERAGE_BYTES = 1024 * 1024


def native_ref(value):
    return 'sha256:' + hashlib.sha256(canonical(value).encode()).hexdigest()


def pack_stock_batches(batches, references=None):
    entries, bundle = [], {}
    for index, native in enumerate(batches):
        ref = native_ref(native) if references is None else references[index]
        batch = native
        coverage = native['context'].get('coverage')
        if coverage is not None:
            raw = canonical(coverage).encode()
            if len(raw) >= MIN_COVERAGE_BYTES:
                coverage_ref = 'sha256:' + hashlib.sha256(raw).hexdigest()
                if coverage_ref not in bundle:
                    compressed = gzip.compress(raw, mtime=0)
                    bundle[coverage_ref] = {'reference': coverage_ref, 'encoding': ENCODING,
                        'uncompressed_bytes': len(raw), 'compressed_digest': 'sha256:' + hashlib.sha256(compressed).hexdigest(),
                        'payload': base64.b64encode(compressed).decode('ascii')}
                context = {key: value for key, value in native['context'].items() if key != 'coverage'}
                context['coverage_ref'] = coverage_ref
                batch = {**native, 'context': context}
        entries.append({'reference': ref, 'batch': batch})
    return entries, [bundle[ref] for ref in sorted(bundle)]


def native_batches(entries, bundle=()):
    """Resolve only the saved local table; verify complete reconstructed native hashes."""
    require(type(entries) is list and type(bundle) in (list, tuple), 'stock evidence table required')
    referenced = set()
    for entry in entries:
        fields(entry, 'reference batch')
        context = entry['batch']['context']
        if 'coverage_ref' in context:
            referenced.add(context['coverage_ref'])
    require({item['reference'] for item in bundle} == referenced, 'unused or missing stock coverage closure')
    table = {}
    for item in bundle:
        fields(item, 'reference encoding uncompressed_bytes compressed_digest payload')
        digest(item['reference']); digest(item['compressed_digest'])
        require(item['reference'] not in table and item['encoding'] == ENCODING and
                type(item['uncompressed_bytes']) is int and item['uncompressed_bytes'] >= MIN_COVERAGE_BYTES,
                'invalid stock coverage entry')
        try:
            compressed = base64.b64decode(item['payload'], validate=True)
            require('sha256:' + hashlib.sha256(compressed).hexdigest() == item['compressed_digest'], 'stock compressed coverage hash mismatch')
            with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as stream:
                raw = stream.read(item['uncompressed_bytes'] + 1)
            require(len(raw) == item['uncompressed_bytes'] and
                    'sha256:' + hashlib.sha256(raw).hexdigest() == item['reference'], 'stock coverage content hash mismatch')
            value = json.loads(raw, object_pairs_hook=_pairs)
        except (ValueError, TypeError, OSError, EOFError) as exc:
            raise ValueError('invalid stock compressed coverage') from exc
        require(type(value) is dict and native_ref(value) == item['reference'], 'noncanonical stock coverage')
        table[item['reference']] = value
    batches, used, refs = [], set(), set()
    for entry in entries:
        fields(entry, 'reference batch'); digest(entry['reference'])
        require(entry['reference'] not in refs, 'duplicate stock native evidence ref')
        refs.add(entry['reference'])
        batch = entry['batch']; context = batch['context']
        if 'coverage_ref' in context:
            ref = context['coverage_ref']
            require(ref in table and 'coverage' not in context, 'unbound stock coverage reference')
            used.add(ref)
            context = {key: value for key, value in context.items() if key != 'coverage_ref'}
            batch = {**batch, 'context': {**context, 'coverage': table[ref]}}
        require(native_ref(batch) == entry['reference'], 'stock native DataBatch hash mismatch')
        batches.append(batch)
    require(used == set(table), 'unused or missing stock coverage closure')
    return batches


def scoped_bundle(entries, bundle):
    refs = {entry['batch']['context'].get('coverage_ref') for entry in entries}
    return [item for item in bundle if item['reference'] in refs]
