"""Explicit opt-in, monitored C JSON import for one original native file.

Only preflight can select stream. A started helper failure ends admission.
The helper owns one full graph; the owner keeps bounded private row views.
"""
from array import array
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

from ..core.contracts import ContractError, require
from .stock_stream_inputs import _CanonicalIndex, _encoded_size, _path


FOOTER = struct.Struct('<QQQ')
MIB = 1048576


def _processes():
    output = subprocess.check_output(['/bin/ps', '-Ao', 'pid=,ppid=,rss='],
                                     text=True, timeout=5)
    values = {}
    for row in output.splitlines():
        pid, parent, rss = row.split()
        values[int(pid)] = (int(parent), int(rss)*1024)
    require(os.getpid() in values and values[os.getpid()][1] > 0, 'CJSON RSS monitor unavailable')
    return values


def _tree_rss(values, root):
    selected = {root}
    while True:
        added = {pid for pid, (parent, _) in values.items() if parent in selected}-selected
        if not added:
            return sum(values[pid][1] for pid in selected if pid in values)
        selected |= added


def _kill(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait(timeout=2)
            return
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)


class _CjsonIndex(_CanonicalIndex):
    def whole(self, budget=None):
        raise ContractError('Native CJSON views cannot decode a whole original parent')

    def span_digest(self, path):
        raise ContractError('Native CJSON view offsets are private, not original parent offsets')

    def unsigned_digest(self, excluded):
        raise ContractError('Native CJSON views cannot create a new parent identity')

    def _open_data(self):
        require(self._store is not None, 'Private CJSON view is closed')
        return nullcontext(self._store)

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    def __del__(self):
        store = getattr(self, '_store', None)
        if store is not None:
            store.close()


def cjson_native_index(artifact, budget, *, row_limit, native_scope, memory,
                       options, spool_used, events):
    """None means a preflight stream choice, never a failed helper retry."""
    started = perf_counter()
    path = _path(artifact)
    stat = path.stat()
    fixed_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    size = fixed_stat[2]
    reason = None
    processes = None
    try:
        from json.scanner import c_make_scanner
        from json.encoder import c_make_encoder
        if os.name != 'posix' or c_make_scanner is None or c_make_encoder is None:
            reason = 'CJSON platform/C parser unavailable'
        elif size > options['max_file_bytes']:
            reason = 'CJSON physical file exceeds explicit budget'
        else:
            processes = _processes()
            parent_rss = _tree_rss(processes, os.getpid())
            estimate = 34*size + 64*MIB + max(512*MIB, parent_rss) + 256*MIB
            if estimate > options['max_tree_rss_bytes']:
                reason = 'CJSON planned peak exceeds explicit process-tree RSS budget'
    except (ImportError, OSError, ValueError, subprocess.SubprocessError, ContractError):
        reason = 'CJSON preflight RSS monitor unavailable'
    if reason is not None:
        events.append(dict(path=str(path), backend='stream', phase='preflight', reason=reason))
        return None
    require(spool_used < options['max_spool_bytes'], 'CJSON source private spool budget exhausted')
    index = _CjsonIndex.__new__(_CjsonIndex)
    index._store = None
    event = dict(path=str(path), backend='cjson', phase='starting', planned_peak_bytes=estimate)
    events.append(event)
    child = None
    gate_read = gate_write = None
    peak = 0
    try:
        index._initialize(artifact, budget, row_limit=row_limit, native_scope=native_scope,
                          memory=memory, fixed_stat=fixed_stat)
        index._store = tempfile.TemporaryFile(mode='w+b')
        with tempfile.TemporaryFile(mode='w+b') as specification, tempfile.TemporaryFile(mode='w+b') as errors:
            spec = dict(artifact=artifact, native_scope=native_scope, row_limit=row_limit,
                read_budget=budget, stat=list(fixed_stat), max_file_bytes=options['max_file_bytes'],
                max_spool_bytes=options['max_spool_bytes']-spool_used)
            with memory.stage() as scratch:
                scratch.reserve(3*_encoded_size(spec))
                encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False,
                    allow_nan=False, separators=(',', ':')).encode()
                specification.write(encoded)
                specification.seek(0)
                spec = encoded = None
            gate_read, gate_write = os.pipe()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
            launch = perf_counter()
            child = subprocess.Popen([sys.executable, '-B', '-m',
                'axiom_engine.runtime.stock_cjson_worker', str(specification.fileno()),
                str(index._store.fileno()), str(gate_read)], env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                pass_fds=(specification.fileno(), index._store.fileno(), gate_read), start_new_session=True)
            os.close(gate_read); gate_read = None
            # The child waits before reading/parsing. Monitoring must succeed first.
            initial = _processes()
            require(child.pid in initial and child.poll() is None, 'CJSON helper exited before monitored start')
            peak = _tree_rss(initial, os.getpid())
            require(peak <= options['max_tree_rss_bytes'], 'CJSON process-tree RSS stop before parse')
            os.write(gate_write, b'1'); os.close(gate_write); gate_write = None
            event['phase'] = 'running'
            while child.poll() is None:
                require(perf_counter()-launch <= options['max_seconds'], 'CJSON helper wall-time stop')
                current = _processes()
                require(child.pid in current or child.poll() is not None, 'CJSON helper RSS disappeared')
                peak = max(peak, _tree_rss(current, os.getpid()))
                require(peak <= options['max_tree_rss_bytes'], 'CJSON process-tree RSS stop')
                require(os.fstat(index._store.fileno()).st_size <= options['max_spool_bytes']-spool_used,
                        'CJSON private spool stop')
                sleep(0.02)
            helper_seconds = perf_counter()-launch
            require(helper_seconds <= options['max_seconds'], 'CJSON helper wall-time stop')
            if child.returncode != 0:
                errors.seek(0)
                message = errors.read(8192).decode('utf-8', errors='replace')
                raise ContractError('CJSON helper failed; no stream retry: '+message.strip())
        index.unchanged()
        used = os.fstat(index._store.fileno()).st_size
        require(FOOTER.size <= used <= options['max_spool_bytes']-spool_used, 'CJSON private spool size mismatch')
        index._store.seek(-FOOTER.size, os.SEEK_END)
        offset, length, native_peak = FOOTER.unpack(index._store.read(FOOTER.size))
        require(offset+length+FOOTER.size == used, 'CJSON private metadata bounds mismatch')
        # Native child peak closes short sampling gaps; summed peak is conservative.
        peak = max(peak, native_peak + _tree_rss(_processes(), os.getpid()))
        require(peak <= options['max_tree_rss_bytes'], 'CJSON native peak process-tree RSS stop')
        consuming = perf_counter()
        with memory.stage() as scratch:
            scratch.reserve(3*length)
            index._store.seek(offset)
            raw = bytearray()
            while len(raw) < length:
                count = min(length-len(raw), budget['max_read_bytes'], 65536)
                chunk = index._store.read(count)
                require(len(chunk) == count, 'CJSON private metadata truncated')
                raw.extend(chunk)
                chunk = None
            require(len(raw) == length, 'CJSON private metadata truncated')
            meta = json.loads(raw)
            raw = None
            require(meta['content_digest'] == artifact['content_digest'] and
                    meta['file_bytes'] == size and meta['stat'] == list(index._stat),
                    'CJSON original identity metadata mismatch')
            entries = meta['indexed_rows']
            memory.reserve_global((str(path), 'cjson_index'),
                length+16*entries+64*(len(meta['spans'])+len(meta['groups'])))
            index.spans = {tuple(key): (start, size) for key, start, size in meta['spans']}
            index.groups = {(tuple(key), day): array('Q', spans) for key, day, spans, _ in meta['groups']}
            index.group_hashes = {(tuple(key), day): digest for key, day, _, digest in meta['groups']}
            index._array_counts = {tuple(key): count for key, count in meta['array_counts']}
            index._indexed_rows = entries
            index.content_digest = meta['content_digest']
            index.file_digest = meta['file_digest']
            index.statistics.update(meta['statistics'])
            index.statistics.update(cjson_files=1, cjson_helper_seconds=helper_seconds,
                cjson_spool_bytes=used, cjson_peak_tree_rss_bytes=peak)
            meta = None
        index.statistics['cjson_metadata_consume_seconds'] = perf_counter()-consuming
        index.statistics['scan_seconds'] = perf_counter()-started
        index._install_native_limits()
        index.unchanged()
        event.update(phase='complete', peak_tree_rss_bytes=peak, spool_bytes=used)
        return index
    except BaseException as exc:
        if child is not None:
            _kill(child)
        index.close()
        event.update(phase='failed', reason=str(exc)[:1024], peak_tree_rss_bytes=peak)
        raise
    finally:
        for descriptor in (gate_read, gate_write):
            if descriptor is not None:
                os.close(descriptor)
