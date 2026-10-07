"""Private one-file C JSON worker. No account, model or supplier calls."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

from ..core.contracts import ContractError, _walk, require, session
from .stock_stream_inputs import _CanonicalIndex, _encoded_size
from .stock_cjson import FOOTER


PARAMS = dict(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _pairs(pairs):
    result = {}; previous = None
    for name, value in pairs:
        require(previous is None or previous < name, 'Noncanonical or duplicate stock JSON key')
        result[name] = value; previous = name
    return result


def _constant(_):
    raise ContractError('Nonfinite stock JSON constant')


def _depth(value, depth=0):
    require(depth <= 128, 'Stock JSON nesting limit exceeded')
    if type(value) is dict:
        for child in value.values():
            _depth(child, depth+1)
    elif type(value) is list:
        for child in value:
            _depth(child, depth+1)


def _mark(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _fd_mark(stream):
    stat = os.fstat(stream.fileno())
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _children(value):
    # Match original depth-three metadata spans, including list wildcards.
    # Scalars and empty containers have no child spans in the stream oracle.
    if type(value) is dict:
        return value.items()
    if type(value) is list:
        return (('*', child) for child in value)
    return ()


class _Gate(_CanonicalIndex):
    def header(self):
        return self.wire['context']


def run(spec, store):
    from urllib.parse import unquote, urlparse
    path = Path(unquote(urlparse(spec['artifact']['manifest_uri']).path)).resolve()
    observed = _mark(path)
    require(observed[2] <= spec['max_file_bytes'], 'CJSON file cap exceeded before read')
    require(observed == spec['stat'], 'CJSON original file changed before read')
    stats = dict(read_calls=2, scan_read_bytes=0, read_seconds=0.0, hash_seconds=0.0,
        content_hash_bytes=0, file_hash_bytes=0, cjson_parse_seconds=0.0,
        cjson_validation_seconds=0.0, cjson_compare_seconds=0.0, cjson_compare_bytes=0,
        cjson_spool_seconds=0.0)
    with path.open('rb', buffering=0) as original:
        observed = _fd_mark(original)
        require(observed[2] <= spec['max_file_bytes'], 'CJSON fd file cap exceeded before read')
        require(observed == spec['stat'], 'CJSON original fd changed before read')
        reading = perf_counter(); raw = original.read(spec['stat'][2])
        require(len(raw) == spec['stat'][2], 'CJSON original size changed while reading')
        require(original.read(1) == b'' and _fd_mark(original) == _mark(path) == spec['stat'],
                'CJSON original file changed during read')
        stats['scan_read_bytes'] = len(raw); stats['read_seconds'] = perf_counter()-reading
        require(raw.startswith(b'{'), 'Stock artifact must be a canonical JSON object')
        lf = raw.endswith(b'\n'); body_size = len(raw)-int(lf)
        hashing = perf_counter()
        file_digest = 'sha256:'+hashlib.sha256(raw).hexdigest()
        content_digest = 'sha256:'+hashlib.sha256(memoryview(raw)[:body_size]).hexdigest()
        stats.update(hash_seconds=perf_counter()-hashing, content_hash_bytes=body_size, file_hash_bytes=len(raw))
        require(content_digest == spec['artifact']['content_digest'], 'Stock artifact content digest mismatch')
        parsing = perf_counter(); text = raw.decode('utf-8'); raw = None
        wire = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        text = None
        stats['cjson_parse_seconds'] = perf_counter()-parsing
        validating = perf_counter()
        require(type(wire) is dict, 'Stock artifact must be a JSON object')
        _depth(wire)
        for _ in _walk(wire):
            pass
        stats['cjson_validation_seconds'] = perf_counter()-validating
        comparing = perf_counter(); canonical_text = json.dumps(wire, **PARAMS)
        original.seek(0); compared = 0
        for offset in range(0, len(canonical_text), 1024*1024):
            encoded = canonical_text[offset:offset+1024*1024].encode('utf-8')
            require(original.read(len(encoded)) == encoded, 'Noncanonical stock JSON bytes')
            compared += len(encoded)
        require(compared == body_size and original.read(1) == (b'\n' if lf else b'') and
                original.read(1) == b'', 'Trailing or noncanonical stock artifact bytes')
        canonical_text = encoded = None
        require(_fd_mark(original) == _mark(path) == spec['stat'],
                'CJSON original fd/file changed during comparison')
        stats.update(cjson_compare_seconds=perf_counter()-comparing, cjson_compare_bytes=compared+int(lf))
    spooling = perf_counter()
    gate = _Gate.__new__(_Gate)
    gate.wire = wire; gate._native_scope = spec['native_scope']
    gate._row_limit = spec['row_limit']; gate._index_limit = spec['row_limit']
    gate._metadata_fields = None; gate._array_counts = {}; gate._indexed_rows = 0
    gate._install_native_limits()
    spans = []; groups = {}; hashes = {}
    maximum = spec['max_spool_bytes']
    def write(value):
        # Check encoded size/quota before C encoding or writing a selected view.
        size = _encoded_size(value)
        require(size*3 <= spec['read_budget']['max_decoded_bytes'], 'CJSON selected view exceeds decoded budget')
        require(store.tell()+size+FOOTER.size <= maximum, 'CJSON private spool exceeds budget before write')
        raw = json.dumps(value, **PARAMS).encode('utf-8')
        require(len(raw) == size, 'CJSON selected view canonical size mismatch')
        start = store.tell(); store.write(raw)
        return start, len(raw), raw
    def header(key, value):
        start, size, _ = write(value)
        spans.append([list(key), start, size])
    def rows(key, values):
        if type(values) is not list:
            return
        for row in values:
            gate._row_guard(key)
            require(type(row) is dict, 'Stock artifact row must be an object')
            day = row.get('session')
            if day is not None:
                session(day)
            start, size, raw = write(row)
            group = key, day
            groups.setdefault(group, []).extend((start, size))
            hashes.setdefault(group, hashlib.sha256()).update(raw)
            gate._array_counts[key] = gate._array_counts.get(key, 0)+1
            gate._indexed_rows += 1
    # Coverage participates in whole-file validation, but never in retained views.
    for name, value in wire['context'].items():
        if name != 'coverage':
            header(('context', name), value)
    for key in (('records',), ('rows',)):
        rows(key, wire.get(key[0]))
    field_meta = wire.get('field_meta', {})
    for name, field in _children(field_meta):
        for key, value in _children(field):
            if key == 'by_key':
                rows(('field_meta', name, key), value)
            else:
                header(('field_meta', name, key), value)
    stats['cjson_spool_seconds'] = perf_counter()-spooling
    metadata = dict(content_digest=content_digest, file_digest=file_digest,
        file_bytes=spec['stat'][2], stat=spec['stat'], spans=spans,
        groups=[[list(key), day, values, 'sha256:'+hashes[key,day].hexdigest()]
                for (key, day), values in groups.items()],
        array_counts=[[list(key), count] for key, count in gate._array_counts.items()],
        indexed_rows=gate._indexed_rows, statistics=stats)
    # Retire the full graph before metadata publication; no graph IPC or pickle.
    gate.wire = wire = field_meta = field = value = None
    offset, length, raw = write(metadata)
    raw = None
    require(store.tell()+FOOTER.size <= maximum, 'CJSON private footer exceeds budget')
    native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    native_peak *= 1 if sys.platform == 'darwin' else 1024
    store.write(FOOTER.pack(offset, length, native_peak)); store.flush()
    require(_mark(path) == spec['stat'], 'CJSON original file changed after spool')


if __name__ == '__main__':
    try:
        spec_fd, store_fd, gate_fd = map(int, sys.argv[1:])
        require(os.read(gate_fd, 1) == b'1', 'CJSON monitored start denied')
        os.close(gate_fd)
        with os.fdopen(spec_fd, 'rb') as parameters, os.fdopen(store_fd, 'w+b', buffering=0) as store:
            run(json.load(parameters), store)
    except BaseException as error:
        sys.stderr.write(json.dumps(dict(type=type(error).__name__, message=str(error)[:1024]))+'\n')
        sys.exit(1)
