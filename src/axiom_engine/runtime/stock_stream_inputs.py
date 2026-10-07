"""Bounded local views of fixed stock inputs for the sole stock Runtime.

This adapter reads canonical JSON artifacts.  It never publishes a sliced
DataBatch/PredictionFrame: byte spans remain bound to the original parent.
No supplier, Research loader, training or account execution belongs here.
"""
from array import array
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Iterator
from urllib.parse import unquote, urlparse

from ..core.contracts import (ContractError, Document, canonical, digest, fields,
                              integer, require, session, _pairs)


# C regex searches the existing bounded buffer by position; no suffix slice.
_STRING_BOUNDARY = re.compile(rb'["\\]')
_SCALAR_BOUNDARY = re.compile(rb'[,\]}]')


def _hash(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _wire(value):
    return value.to_dict() if isinstance(value, Document) else value


def _budget(value):
    fields(value, "max_read_bytes max_decoded_bytes")
    for limit in value.values():
        integer(limit, 1)
    return value


class _DecodedMemory:
    """Canonical byte-volume accounting, separate from RSS and compact indexes."""
    def __init__(self, limit, *, scalar_cache_bytes=0):
        self.limit = limit
        self.globals = {}
        self.global_bytes = 0
        self.temporary_bytes = 0
        self.peak_bytes = 0
        self.scalar_cache = _ScalarCache(self, scalar_cache_bytes)

    @property
    def available(self):
        return self.limit - self.global_bytes - self.temporary_bytes

    def reserve_global(self, key, size):
        if key in self.globals:
            require(self.globals[key] == size, "Conflicting decoded cache reservation")
            return
        self.make_room(size)
        require(size <= self.available, "Stock cumulative decoded budget exceeded before cache allocation")
        self.globals[key] = size; self.global_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.global_bytes + self.temporary_bytes)

    def replace_global(self, key, size):
        old = self.globals.get(key, 0)
        self.make_room(size-old)
        require(size-old <= self.available, "Stock cumulative decoded budget exceeded before state growth")
        self.globals[key] = size; self.global_bytes += size-old
        self.peak_bytes = max(self.peak_bytes, self.global_bytes + self.temporary_bytes)

    def release_global(self, key):
        self.global_bytes -= self.globals.pop(key, 0)

    def stage(self):
        return _DecodedStage(self)

    def make_room(self, size):
        # Optional scalar reuse never steals space from required input growth.
        self.scalar_cache.make_room(size)


class _ScalarCache:
    """Bounded immutable literals, each admitted by the original scalar path.

    This is only an optimization within this source admission. Exact raw bytes
    are the keys; no document, row, ordering or Unknown validation is skipped.
    """
    def __init__(self, memory, limit):
        self.memory = memory
        self.limit = limit
        self.entries = OrderedDict()
        self.used = 0
        self.evictions = 0

    def _evict(self):
        raw, (_, size) = self.entries.popitem(last=False)
        self.used -= size
        self.memory.release_global(("scalar_literal", raw))
        self.evictions += 1

    def make_room(self, size):
        while self.entries and size > self.memory.available:
            self._evict()

    def get(self, raw):
        entry = self.entries.get(raw)
        if entry is None:
            return False, None
        self.entries.move_to_end(raw)
        return True, entry[0]

    def put(self, raw, value):
        # Bound encoded key/value volume plus conservative per-entry overhead.
        size = 2*len(raw)+128
        if size > self.limit:
            return
        while self.entries and self.used+size > self.limit:
            self._evict()
        self.make_room(size)
        if size > self.memory.available:
            return
        self.memory.reserve_global(("scalar_literal", raw), size)
        self.entries[raw] = value, size
        self.used += size


class _DecodedStage:
    def __init__(self, memory):
        self.memory = memory
        self.used = 0

    def reserve(self, size):
        self.memory.make_room(size)
        require(size <= self.memory.available, "Stock cumulative decoded budget exceeded before temporary growth")
        self.memory.temporary_bytes += size; self.used += size
        self.memory.peak_bytes = max(self.memory.peak_bytes,
                                     self.memory.global_bytes + self.memory.temporary_bytes)

    def release(self, size):
        require(0 <= size <= self.used, "Invalid decoded reservation release")
        self.memory.temporary_bytes -= size; self.used -= size

    def close(self):
        self.release(self.used)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _object_size(parts):
    """Known canonical size of an object before its values are decoded."""
    return 2 + sum(_encoded_size(name) + 1 + size for name, size in parts) + max(0, len(parts)-1)


def _encoded_size(value):
    from .stock_stream_outputs import _json_pieces
    return sum(len(piece) for piece in _json_pieces(value, piece_chars=64))


def _path(artifact):
    fields(artifact, "artifact_type artifact_id contract_version manifest_uri content_digest")
    for name in ("artifact_type", "artifact_id", "contract_version", "manifest_uri"):
        require(type(artifact[name]) is str and bool(artifact[name]), "Invalid stock artifact envelope")
    digest(artifact["content_digest"])
    uri = urlparse(artifact["manifest_uri"])
    require(uri.scheme in ("", "file") and not uri.netloc and uri.fragment in ("", "definition/fold_spec") and not uri.query,
            "Stock source requires an explicit local canonical artifact")
    path = Path(unquote(uri.path)).resolve()
    require(path.is_file(), "Missing fixed stock artifact: " + str(path))
    require(not uri.fragment or path.name == "fold.json", "Saved fold selector requires the original fold.json")
    return path


@dataclass(frozen=True)
class StockInputBlock:
    sessions: tuple
    market_rows: dict
    signals: dict
    bindings: tuple
    decoded_bytes: int = 0
    _reservation: object = None


@dataclass(frozen=True)
class StockSourceAudit:
    """Invocation-local admission; receipt contains no large source graph."""
    receipt: dict
    globals: dict
    lifecycle: dict
    manifest: dict = None
    segment_index: dict = None


class _CanonicalIndex:
    """One pass canonical-byte validation and compact row-span indexing.

    Large coverage trees are scanned, not decoded.  Date-keyed arrays retain
    only uint64 offsets/lengths plus one content hash per date.  A single JSON
    scalar/row/header must fit the decoded budget; unsupported oversized
    values fail before their byte buffer grows.  No decompression is accepted.
    """
    def _initialize(self, artifact, budget, *, row_limit=None, native_scope=None, memory=None, descriptor_path=None):
        self.path = Path(descriptor_path).resolve() if descriptor_path is not None else _path(artifact)
        self.artifact = artifact
        fragment = urlparse(artifact["manifest_uri"]).fragment if artifact is not None else ""
        self.selector = tuple(fragment.split("/")) if fragment else None
        self.budget = _budget(budget)
        self._memory = memory or _DecodedMemory(budget["max_decoded_bytes"])
        self._values = {}
        self._object_headers = {}
        self._whole_value = None
        self.size = self.path.stat().st_size
        self.statistics = {"file_bytes": self.size, "scan_seconds": 0.0,
            "read_calls": 0, "scan_read_bytes": 0, "read_seconds": 0.0, "hash_seconds": 0.0,
            "content_hash_bytes": 0, "file_hash_bytes": 0,
            "scalar_count": 0, "scalar_batches": 0, "scalar_bytes": 0,
            "scalar_cache_hits": 0, "scalar_decode_count": 0, "scalar_canonical_count": 0,
            "scalar_decode_seconds": 0.0, "scalar_canonical_seconds": 0.0,
            "row_decode_count": 0, "row_decode_seconds": 0.0,
            "reread_bytes": 0, "reread_calls": 0,
            "local_decode_count": 0, "local_decode_seconds": 0.0,
            "span_identity_count": 0, "span_identity_seconds": 0.0,
            "unsigned_identity_count": 0, "unsigned_identity_seconds": 0.0,
            "cjson_files": 0, "cjson_helper_seconds": 0.0, "cjson_parse_seconds": 0.0,
            "cjson_validation_seconds": 0.0, "cjson_compare_seconds": 0.0,
            "cjson_spool_seconds": 0.0, "cjson_compare_bytes": 0,
            "cjson_spool_bytes": 0, "cjson_metadata_consume_seconds": 0.0,
            "cjson_peak_tree_rss_bytes": 0}
        stat = self.path.stat()
        self._stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        self._header_hashes = {}
        self._context_header = None
        self._metadata_header = None
        self.spans = {}
        self.groups = {}
        self.group_hashes = {}
        self._group_hashers = {}
        self._hash = hashlib.sha256()
        self._file_hash = hashlib.sha256()
        self._terminal_lf = False
        self._position = 0
        self._buffer = b""
        self._cursor = 0
        self._capture = None
        self._capture_stage = None
        if row_limit is not None:
            integer(row_limit, 1)
        self._row_limit = row_limit
        self._index_limit = row_limit
        self._native_scope = native_scope
        self._metadata_fields = None
        self._array_counts = {}
        self._indexed_rows = 0

    def __init__(self, artifact, budget, *, row_limit=None, native_scope=None, memory=None, descriptor_path=None):
        self._initialize(artifact, budget, row_limit=row_limit, native_scope=native_scope,
                         memory=memory, descriptor_path=descriptor_path)
        scan_started = perf_counter()
        with self._memory.stage() as self._scan_stage:
            with self.path.open("rb", buffering=0) as self._stream:
                require(self._peek() == 123, "Stock artifact must be a canonical JSON object")
                self._value(())
                self._canonical_size = self._position
                if self._peek() == 10:
                    self._take()
                require(self._peek() is None, "Trailing or noncanonical stock artifact bytes")
            self._buffer = b""
        self.statistics["scan_seconds"] = perf_counter() - scan_started
        self.unchanged()
        self.content_digest = "sha256:" + self._hash.hexdigest()
        self.file_digest = "sha256:" + self._file_hash.hexdigest()
        if self.selector:
            require(self.span_digest(self.selector) == artifact["content_digest"], "Saved fold child content digest mismatch")
        elif descriptor_path is None:
            require(self.content_digest == artifact["content_digest"], "Stock artifact content digest mismatch")
        self.group_hashes = {key: "sha256:" + value.hexdigest()
                             for key, value in self._group_hashers.items()}
        self._group_hashers.clear()
        self._buffer = b""

    def unchanged(self):
        stat = self.path.stat()
        require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == self._stat,
                "Fixed stock artifact changed after admission")

    def _open_data(self):
        return self.path.open("rb", buffering=0)

    def _peek(self):
        if self._cursor == len(self._buffer):
            self._scan_stage.release(len(self._buffer))
            self._buffer = b""
            if self._stream.tell() == self.size:
                return None
            # Bound each allocation before read, including at EOF.
            self._memory.make_room(min(65536, self.budget["max_read_bytes"]))
            count = min(65536, self.budget["max_read_bytes"], self._memory.available)
            require(count > 0, "Stock cumulative decoded budget exceeded before scan buffer allocation")
            self._scan_stage.reserve(count)
            reading = perf_counter()
            self._buffer = self._stream.read(count)
            self.statistics["read_seconds"] += perf_counter() - reading
            self.statistics["read_calls"] += 1
            self.statistics["scan_read_bytes"] += len(self._buffer)
            self._scan_stage.release(count-len(self._buffer))
            hashing = perf_counter()
            self._file_hash.update(self._buffer)
            # A single terminal LF is delivery formatting.  The original
            # canonical identity excludes it; the actual file hash retains it.
            last = self._stream.tell() == self.size and self._buffer.endswith(b"\n")
            self._terminal_lf = self._terminal_lf or last
            self._hash.update(memoryview(self._buffer)[:-1] if last else self._buffer)
            self.statistics["hash_seconds"] += perf_counter() - hashing
            self.statistics["file_hash_bytes"] += len(self._buffer)
            self.statistics["content_hash_bytes"] += len(self._buffer) - int(last)
            self._cursor = 0
        return self._buffer[self._cursor] if self._buffer else None

    def _take(self):
        value = self._peek()
        require(value is not None, "Truncated stock artifact")
        if self._capture is not None:
            self._capture_stage.reserve(1)
            self._capture.append(value)
        self._cursor += 1
        self._position += 1
        return value

    def _expect(self, byte):
        require(self._take() == byte, "Noncanonical or malformed stock JSON")

    def _scalar(self):
        stage = self._memory.stage()
        raw = bytearray()
        self.statistics["scalar_count"] += 1
        def take(count):
            # Reserve both growing destinations before copying or advancing.
            # A view of this read chunk is retired before the next _peek.
            require(0 < count <= len(self._buffer)-self._cursor, "Invalid bounded scalar span")
            stage.reserve(count)
            if self._capture is not None:
                self._capture_stage.reserve(count)
            with memoryview(self._buffer) as whole:
                with whole[self._cursor:self._cursor+count] as piece:
                    raw.extend(piece)
                    if self._capture is not None:
                        self._capture.extend(piece)
            self._cursor += count; self._position += count
            self.statistics["scalar_batches"] += 1
            self.statistics["scalar_bytes"] += count
        try:
            if self._peek() == 34:
                take(1); escape = False
                while True:
                    byte = self._peek()
                    require(byte is not None, "Truncated stock JSON string")
                    if escape:
                        take(1); escape = False
                        continue
                    boundary = _STRING_BOUNDARY.search(self._buffer, self._cursor)
                    if boundary is None:
                        take(len(self._buffer)-self._cursor)
                        continue
                    count = boundary.start()-self._cursor+1
                    closing = self._buffer[boundary.start()] == 34
                    take(count)
                    if closing:
                        break
                    escape = True
            else:
                while self._peek() not in (None, 44, 93, 125):
                    boundary = _SCALAR_BOUNDARY.search(self._buffer, self._cursor)
                    count = (boundary.start() if boundary is not None else len(self._buffer))-self._cursor
                    take(count)
            size = len(raw)
            cache = self._memory.scalar_cache
            key = None
            if cache.limit and size <= 256:
                self._memory.make_room(4*size)
            if cache.limit and size <= 256 and 4*size <= self._memory.available:
                stage.reserve(size)
                key = bytes(raw)
                found, value = cache.get(key)
                if found:
                    self.statistics["scalar_cache_hits"] += 1
                    raw = key = None
                    stage.release(size)
                    return value, stage
            # Raw, decoded scalar, canonical text and encoded comparison can
            # coexist. Reserve that growth before invoking either decoder.
            stage.reserve(3*size)
            try:
                decoding = perf_counter()
                value = json.loads(raw, object_pairs_hook=_pairs)
                self.statistics["scalar_decode_count"] += 1
                self.statistics["scalar_decode_seconds"] += perf_counter()-decoding
            except (ValueError, UnicodeError) as exc:
                raise ContractError("Malformed stock JSON scalar") from exc
            encoding = perf_counter()
            require(canonical(value).encode() == raw, "Noncanonical stock JSON scalar")
            self.statistics["scalar_canonical_count"] += 1
            self.statistics["scalar_canonical_seconds"] += perf_counter()-encoding
            raw = None
            stage.release(3*size)
            if key is not None:
                cache.put(key, value)
                key = None
                stage.release(size)
            return value, stage
        except Exception:
            stage.close()
            raise

    @staticmethod
    def _rows(path):
        return path in (("records",), ("rows",)) or (
            len(path) == 3 and path[0] == "field_meta" and path[2] == "by_key")

    def _install_native_limits(self):
        """Original context precedes row arrays in canonical native JSON."""
        if self._native_scope is None:
            return
        manifest, limits = self._native_scope
        context = self.header(); query = context.get("query", {})
        domain = context.get("domain")
        names = query.get("fields")
        universe = manifest["scope"]["execution_universe"]
        require(type(names) is list and bool(names) and all(type(name) is str for name in names),
                "Original native fields required before compact indexing")
        require(query.get("symbols") == universe, "Original native symbols required before compact indexing")
        if domain == "corporate_actions":
            from .stock_market import STOCK_EVENT_FIELDS
            require(names == list(STOCK_EVENT_FIELDS), "Original action fields required before compact indexing")
            row_limit = limits["max_market_rows"]
        else:
            require(domain in _KINDS, "Unsupported original native domain before compact indexing")
            days = query.get("sessions")
            require(type(days) is list and bool(days) and days == sorted(set(days)),
                    "Original native sessions required before compact indexing")
            for day in days:
                session(day)
            calendar = manifest["scope"]["calendar"]
            warmup = manifest["market_input"]["warmup_sessions"]
            query_rows = len(days) * len(universe)
            if domain == "market_state_diagnostics":
                require(set(calendar) <= set(days), "Original state session coverage required before compact indexing")
                # States can retain the original Reader's extra query sessions.
                extra = len(set(days) - set(calendar)) * len(universe)
                row_limit = min(query_rows, limits["max_market_rows"] + extra)
                names = [*names, "market_state"]
            elif days == warmup:
                row_limit = query_rows
            else:
                require(days == calendar, "Original daily scope required before compact indexing")
                row_limit = min(query_rows, limits["max_market_rows"])
        self._row_limit = row_limit
        self._metadata_fields = set(names)
        # One records array plus at most the declared/derived metadata arrays.
        self._index_limit = row_limit * (1 + len(self._metadata_fields))

    def _row_guard(self, path):
        if len(path) == 3 and self._metadata_fields is not None:
            require(path[1] in self._metadata_fields, "Undeclared native metadata array before compact indexing")
        require(self._row_limit is None or self._array_counts.get(path, 0) < self._row_limit,
                "Stock row budget exceeded before compact index growth: " + ".".join(path))
        require(self._index_limit is None or self._indexed_rows < self._index_limit,
                "Stock total span budget exceeded before compact index growth")

    def _value(self, path, depth=0):
        require(depth <= 128, "Stock JSON nesting limit exceeded")
        start = self._position
        scalar = None
        scalar_stage = None
        if self._peek() == 123:
            self._take(); last = None; count = 0; reserved = None; reserved_names = True; has_reserved = False
            reserved_values = {}; metadata_object = False
            name_stage = reserved_stage = None
            try:
                if self._peek() != 125:
                    while True:
                        require(self._peek() == 34, "Stock JSON object key required")
                        name, next_name_stage = self._scalar()
                        try:
                            require(type(name) is str and (last is None or last < name),
                                    "Noncanonical or duplicate stock JSON key")
                        except Exception:
                            next_name_stage.close()
                            raise
                        last = name
                        if name_stage is not None:
                            name_stage.close()
                        name_stage = next_name_stage
                        count += 1
                        reserved_names = reserved_names and name in {
                            "contract_type", "contract_version", "metadata", "unknown_id", "reason", "required_evidence"}
                        self._expect(58)
                        if name == "contract_type":
                            has_reserved = True
                            reserved, reserved_stage = self._scalar()
                        else:
                            if name == "metadata":
                                metadata_object = self._peek() == 123
                            value, value_stage = self._value(path + (name,), depth + 1)
                            if name == "contract_version":
                                reserved_values[name] = value == "1"
                            elif name in ("unknown_id", "reason", "required_evidence"):
                                reserved_values[name] = type(value) is str and bool(value) and not value.isspace()
                            value = None
                            if value_stage is not None:
                                value_stage.close()
                        if self._peek() != 44:
                            break
                        self._take()
                self._expect(125)
                if has_reserved:
                    require(reserved == "Unknown" and count == 6 and reserved_names,
                        "Unsupported reserved contract_type in stock source")
                    require(reserved_values.get("contract_version") is True and metadata_object and
                            all(reserved_values.get(name) is True for name in ("unknown_id", "reason", "required_evidence")),
                            "Invalid reserved Unknown in stock source")
            finally:
                name = last = reserved = None
                if name_stage is not None:
                    name_stage.close()
                if reserved_stage is not None:
                    reserved_stage.close()
        elif self._peek() == 91:
            self._take()
            if self._peek() != 93:
                while True:
                    if self._rows(path):
                        require(self._capture is None, "Nested stock row arrays are unsupported")
                        # Refuse before decoding the next row or allocating its span.
                        self._row_guard(path)
                        offset = self._position
                        with self._memory.stage() as capture_stage:
                            try:
                                self._capture_stage = capture_stage
                                self._capture = bytearray()
                                self._value(path + ("*",), depth + 1)
                                raw = self._capture
                                self._capture = None; self._capture_stage = None
                                # loads(bytearray) also keeps its decoded input
                                # text alive until the result graph returns.
                                capture_stage.reserve(2*len(raw))
                                try:
                                    decoding = perf_counter()
                                    row = json.loads(raw, object_pairs_hook=_pairs)
                                    self.statistics["row_decode_count"] += 1
                                    self.statistics["row_decode_seconds"] += perf_counter()-decoding
                                except (ValueError, UnicodeError) as exc:
                                    raise ContractError("Malformed stock JSON row") from exc
                                require(type(row) is dict, "Stock artifact row must be an object")
                                # Action rows have no session and stay in a small global group.
                                day = row.get("session")
                                if day is not None:
                                    session(day)
                                key = (path, day)
                                self._row_guard(path)
                                spans = self.groups.get(key)
                                if spans is None:
                                    spans = self.groups[key] = array("Q")
                                spans.extend((offset, len(raw)))
                                self._array_counts[path] = self._array_counts.get(path, 0) + 1
                                self._indexed_rows += 1
                                self._group_hashers.setdefault(key, hashlib.sha256()).update(raw)
                                raw = row = None
                            finally:
                                self._capture = None; self._capture_stage = None
                    else:
                        value, value_stage = self._value(path + ("*",), depth + 1)
                        value = None
                        if value_stage is not None:
                            value_stage.close()
                    if self._peek() != 44:
                        break
                    self._take()
            self._expect(93)
        else:
            scalar, scalar_stage = self._scalar()
        # Keep only top/header boundaries, never per-coverage-object dictionaries.
        if len(path) == 1 or (len(path) == 2 and path[0] == "context") or (
                len(path) == 3 and path[0] == "field_meta" and path[2] != "by_key") or (
                self.selector is not None and path in (self.selector, ("definition", "fold_spec_ref"))):
            self.spans[path] = (start, self._position - start)
        if path == ("context",):
            self._install_native_limits()
        return scalar, scalar_stage

    def _read_span(self, stream, start, length, budget, retained=0):
        require(retained + length <= budget["max_decoded_bytes"],
                "Stock decoded budget exceeded before span allocation")
        stream.seek(start)
        raw = bytearray()
        remaining = length
        while remaining:
            size = min(remaining, budget["max_read_bytes"], 65536)
            require(len(raw) + size + retained <= budget["max_decoded_bytes"],
                    "Stock decoded budget exceeded before span growth")
            chunk = stream.read(size)
            self.statistics["reread_calls"] += 1
            self.statistics["reread_bytes"] += len(chunk)
            require(len(chunk) == size, "Fixed stock artifact truncated after admission")
            raw.extend(chunk)
            chunk = None
            remaining -= size
        return raw

    def _decode_value(self, path, budget=None):
        budget = self.budget if budget is None else _budget(budget)
        require(path in self.spans, "Missing stock artifact header: " + ".".join(path))
        self.unchanged()
        with self._memory.stage() as stage:
            stage.reserve(2*self.spans[path][1])
            with self._open_data() as stream:
                raw = self._read_span(stream, *self.spans[path], budget)
            self.unchanged()
            reference = _hash(raw)
            require(self._header_hashes.setdefault(path, reference) == reference,
                    "Stock artifact header changed after admission")
            try:
                decoding = perf_counter()
                value = json.loads(raw, object_pairs_hook=_pairs)
                self.statistics["local_decode_count"] += 1
                self.statistics["local_decode_seconds"] += perf_counter()-decoding
            except (ValueError, UnicodeError) as exc:
                raise ContractError("Stock artifact header changed after admission") from exc
            raw = None
            return value

    def value(self, path, budget=None):
        if len(path) == 2 and path[0] == "context":
            return self.header()[path[1]]
        if len(path) == 3 and path[0] == "field_meta":
            return self.metadata_header()[path[1]][path[2]]
        if path not in self._values:
            self._memory.reserve_global((str(self.path), path), self.spans[path][1])
            self._values[path] = self._decode_value(path, budget)
        return self._values[path]

    def header(self):
        # Native coverage is hash-bound by the original full parent scan.
        if self._context_header is None:
            selected = [key for key in self.spans if len(key) == 2 and key[0] == "context" and key[-1] != "coverage"]
            size = _object_size([(key[-1], self.spans[key][1]) for key in selected])
            self._memory.reserve_global((str(self.path), "context_header"), size)
            self._context_header = {key[-1]: self._decode_value(key) for key in selected}
        return self._context_header

    def metadata_header(self):
        if self._metadata_header is None:
            by_name = {}
            for key in self.spans:
                if len(key) == 3 and key[0] == "field_meta":
                    by_name.setdefault(key[1], []).append(key)
            size = _object_size([(name, _object_size([(key[2], self.spans[key][1]) for key in keys]))
                                 for name, keys in by_name.items()])
            self._memory.reserve_global((str(self.path), "metadata_header"), size)
            self._metadata_header = {name: {key[2]: self._decode_value(key) for key in keys}
                                     for name, keys in by_name.items()}
        return self._metadata_header

    def rows(self, path, days, budget=None, *, reservation=None):
        budget = self.budget if budget is None else _budget(budget)
        require(isinstance(reservation, _DecodedStage) and reservation.memory is self._memory,
                "Original stock rows require a shared decoded reservation")
        result, retained, bindings = [], 0, []
        self.unchanged()
        with self._open_data() as stream:
            for day in days:
                spans = self.groups.get((path, day), ())
                hashed = hashlib.sha256()
                for i in range(0, len(spans), 2):
                    if reservation is not None:
                        reservation.reserve(spans[i+1])
                    with self._memory.stage() as scratch:
                        scratch.reserve(2*spans[i+1])
                        raw = self._read_span(stream, spans[i], spans[i+1], budget, retained)
                        retained += len(raw)
                        hashed.update(raw)
                        try:
                            decoding = perf_counter()
                            result.append(json.loads(raw, object_pairs_hook=_pairs))
                            self.statistics["local_decode_count"] += 1
                            self.statistics["local_decode_seconds"] += perf_counter()-decoding
                        except (ValueError, UnicodeError) as exc:
                            raise ContractError("Stock artifact segment changed after admission") from exc
                        raw = None
                reference = "sha256:" + hashed.hexdigest()
                expected = self.group_hashes.get((path, day), _hash(b""))
                require(reference == expected, "Stock artifact segment hash mismatch after admission")
                binding_size = _object_size([(name, _encoded_size(value)) for name, value in (
                    ("parent_ref", self.content_digest), ("array_path", list(path)), ("session", day),
                    ("rows", len(spans)//2), ("segment_digest", reference), ("file_digest", self.file_digest))])
                if reservation is not None:
                    reservation.reserve(binding_size)
                retained += binding_size
                bindings.append({"parent_ref": self.content_digest, "array_path": list(path),
                                 "session": day, "rows": len(spans)//2, "segment_digest": reference,
                                 "file_digest": self.file_digest})
        self.unchanged()
        return result, retained, bindings

    def batch_view(self, days, budget=None, *, reservation=None):
        """Original header plus selected values; explicitly not an identity object."""
        budget = self.budget if budget is None else _budget(budget)
        retained, bindings = 0, []
        rows, used, refs = self.rows(("records",), days, budget, reservation=reservation)
        retained += used; bindings += refs
        metadata = {name: dict(value) for name, value in self.metadata_header().items()}
        names = set(metadata) | {path[1] for path, _ in self.groups if len(path) == 3 and path[0] == "field_meta"}
        for name in sorted(names):
            remaining = {**budget, "max_decoded_bytes": budget["max_decoded_bytes"] - retained}
            require(remaining["max_decoded_bytes"] > 0, "Stock decoded block budget exceeded")
            values, used, refs = self.rows(("field_meta", name, "by_key"), days, remaining, reservation=reservation)
            metadata.setdefault(name, {})["by_key"] = values
            retained += used; bindings += refs
        overhead = 64 + sum(_encoded_size(name) + 32 for name in names) + len(rows) + sum(
            len(field["by_key"]) for field in metadata.values())
        if reservation is not None:
            reservation.reserve(overhead)
        retained += overhead
        return {"context": self.header(), "records": rows, "field_meta": metadata}, retained, bindings

    def whole(self, budget=None):
        """Only small profile/fold/model documents may use this operation."""
        require(self.selector is None, "A selected fold child is not its full parent document")
        budget = self.budget if budget is None else _budget(budget)
        if self._whole_value is not None:
            return self._whole_value
        self._memory.reserve_global((str(self.path), "whole"), self._canonical_size)
        self.unchanged()
        with self._memory.stage() as stage:
            stage.reserve(2*self._canonical_size)
            with self.path.open("rb", buffering=0) as stream:
                raw = self._read_span(stream, 0, self._canonical_size, budget)
            self.unchanged()
            require(_hash(raw) == self.content_digest, "Small stock artifact replaced after admission")
            decoding = perf_counter()
            self._whole_value = json.loads(raw, object_pairs_hook=_pairs)
            self.statistics["local_decode_count"] += 1
            self.statistics["local_decode_seconds"] += perf_counter()-decoding
            raw = None
        return self._whole_value

    def span_digest(self, path):
        """Hash an original selected value without decoding its parent graph."""
        require(path in self.spans, "Missing original saved fold selector")
        self.statistics["span_identity_count"] += 1
        started = perf_counter()
        self.unchanged()
        hashed = hashlib.sha256()
        with self.path.open("rb", buffering=0) as stream:
            start, length = self.spans[path]; stream.seek(start)
            while length:
                with self._memory.stage() as scratch:
                    count = min(length, self.budget["max_read_bytes"], 65536, self._memory.available)
                    require(count > 0, "Stock cumulative decoded budget exceeded before hash buffer allocation")
                    scratch.reserve(count)
                    raw = stream.read(count)
                    self.statistics["reread_calls"] += 1
                    self.statistics["reread_bytes"] += len(raw)
                    require(len(raw) == count, "Fixed stock artifact truncated during selected identity admission")
                    hashed.update(raw); length -= count; raw = None
        self.unchanged()
        self.statistics["span_identity_seconds"] += perf_counter() - started
        return "sha256:" + hashed.hexdigest()

    def unsigned_digest(self, excluded):
        """Original canonical object identity with its self-reference removed."""
        hashed = hashlib.sha256(); hashed.update(b"{")
        self.statistics["unsigned_identity_count"] += 1
        started = perf_counter()
        self.unchanged()
        names = sorted(path[0] for path in self.spans if len(path) == 1 and path[0] != excluded)
        with self.path.open("rb", buffering=0) as stream:
            for index, name in enumerate(names):
                if index:
                    hashed.update(b",")
                hashed.update(canonical(name).encode()); hashed.update(b":")
                start, length = self.spans[(name,)]
                stream.seek(start)
                while length:
                    with self._memory.stage() as scratch:
                        count = min(length, self.budget["max_read_bytes"], 65536, self._memory.available)
                        require(count > 0, "Stock cumulative decoded budget exceeded before hash buffer allocation")
                        scratch.reserve(count)
                        raw = stream.read(count)
                        self.statistics["reread_calls"] += 1
                        self.statistics["reread_bytes"] += len(raw)
                        require(len(raw) == count, "Fixed stock artifact truncated during identity admission")
                        hashed.update(raw); length -= count; raw = None
        hashed.update(b"}")
        self.unchanged()
        self.statistics["unsigned_identity_seconds"] += perf_counter() - started
        return "sha256:" + hashed.hexdigest()

    def object_header(self, excluded=()):
        excluded = tuple(excluded)
        if excluded not in self._object_headers:
            selected = [path for path in self.spans if len(path) == 1 and path[0] not in excluded]
            size = _object_size([(path[0], self.spans[path][1]) for path in selected])
            self._memory.reserve_global((str(self.path), "object_header", excluded), size)
            self._object_headers[excluded] = {path[0]: self._decode_value(path) for path in selected}
        return self._object_headers[excluded]


class StockInputSource:
    """Explicit local source. Inventory never opens a source payload."""
    def __init__(self, *, scalar_cache_bytes=0, file_parse_mode="stream",
                 max_file_parse_bytes=None, max_file_parse_rss_bytes=None,
                 max_file_parse_spool_bytes=None, max_file_parse_seconds=120):
        integer(scalar_cache_bytes)
        require(file_parse_mode in ("stream", "cjson"), "Unsupported stock file parse mode")
        integer(max_file_parse_seconds, 1)
        for value in (max_file_parse_bytes, max_file_parse_rss_bytes, max_file_parse_spool_bytes):
            if value is not None:
                integer(value, 1)
        require(file_parse_mode != "cjson" or all(value is not None for value in
            (max_file_parse_bytes, max_file_parse_rss_bytes, max_file_parse_spool_bytes)),
            "CJSON requires explicit file, process-tree RSS and private spool budgets")
        self._scalar_cache_bytes = scalar_cache_bytes
        self._file_parse_mode = file_parse_mode
        self._file_parse_options = dict(max_file_bytes=max_file_parse_bytes,
            max_tree_rss_bytes=max_file_parse_rss_bytes, max_spool_bytes=max_file_parse_spool_bytes,
            max_seconds=max_file_parse_seconds)
        self._file_parse_events = []
        self._cjson_spool_bytes = 0
        self._indexes = {}
        self._prepared = None
        self._inventory_sizes = {}
        self._audited = False
        self._scan_row_limits = {}
        self._scan_native_scopes = {}
        self._prediction_paths = set()
        self._prediction_remaining = 0
        self._memory = None

    def execution_scope(self):
        """Original sources retain their existing invocation-local behavior."""
        if self._file_parse_mode == "cjson":
            return self._private_scope()
        return nullcontext()

    @contextmanager
    def _private_scope(self):
        try:
            yield
        finally:
            self._close_private_views()
            self._prepared = None
            self._audited = False

    def _close_private_views(self):
        for index in self._indexes.values():
            if hasattr(index, "close"):
                index.close()

    @property
    def statistics(self):
        """Read-only operation counters; never a persisted admission capability."""
        files = [{"path": str(index.path), **index.statistics} for index in self._indexes.values()]
        return {"files": files, "source_operations": {
                    name: (max(item[name] for item in files) if name == "cjson_peak_tree_rss_bytes"
                           else sum(item[name] for item in files))
                    for name in (files[0].keys()-{"path"})} if files else {},
                "decoded_bytes_peak": 0 if self._memory is None else self._memory.peak_bytes,
                "scalar_cache_evictions": 0 if self._memory is None else self._memory.scalar_cache.evictions,
                "file_parse_events": [dict(event) for event in self._file_parse_events],
                "cjson_peak_tree_rss_bytes": max((item["cjson_peak_tree_rss_bytes"] for item in files), default=0)}

    @staticmethod
    def _artifacts(manifest):
        yield manifest["profile_input"]["artifact"]
        for entry in manifest["market_input"]["native_inputs"]:
            yield entry["artifact"]
        for frame in manifest["prediction_input"]["frames"]:
            for key in ("fold_spec_artifact", "model_metadata_artifact", "prediction_artifact"):
                yield frame[key]

    def inventory(self, manifest):
        manifest = _wire(manifest)
        files = {}
        for artifact in self._artifacts(manifest):
            path = _path(artifact)
            key = str(path)
            require(key not in files or files[key]["artifact"]["content_digest"] == artifact["content_digest"],
                    "Conflicting stock artifact references to one file")
            files[key] = {"artifact": artifact, "file_bytes": path.stat().st_size}
            if urlparse(artifact["manifest_uri"]).fragment:
                descriptor = path.with_name("manifest.json")
                require(descriptor.is_file(), "Saved fold selector requires its original sibling manifest")
                files.setdefault(str(descriptor), {"manifest_uri": str(descriptor), "file_bytes": descriptor.stat().st_size})
        return {"files": list(files.values()), "input_bytes": sum(f["file_bytes"] for f in files.values()),
                "folds": len(manifest["prediction_input"]["frames"]),
                "declared_rows": {"market_rows": len(manifest["scope"]["calendar"]) * len(manifest["scope"]["execution_universe"]),
                                  "prediction_rows": (len(manifest["scope"]["calendar"])-1) * len(manifest["scope"]["prediction_universe"])},
                "declared_scope": manifest["scope"]}

    def _index(self, artifact, budget):
        key = (str(_path(artifact)), artifact["content_digest"])
        if key not in self._indexes:
            if key[0] in self._inventory_sizes:
                require(Path(key[0]).stat().st_size == self._inventory_sizes[key[0]],
                        "Stock source file size changed after inventory")
            row_limit = self._scan_row_limits.get(key[0])
            if key[0] in self._prediction_paths:
                row_limit = min(row_limit, self._prediction_remaining)
                require(row_limit > 0, "Stock prediction row budget exhausted before compact index growth")
            native_scope = self._scan_native_scopes.get(key[0])
            index = None
            if self._file_parse_mode == "cjson" and native_scope is not None:
                from .stock_cjson import cjson_native_index
                index = cjson_native_index(artifact, budget, row_limit=row_limit,
                    native_scope=native_scope, memory=self._memory, options=self._file_parse_options,
                    spool_used=self._cjson_spool_bytes, events=self._file_parse_events)
                if index is not None:
                    self._cjson_spool_bytes += index.statistics["cjson_spool_bytes"]
            self._indexes[key] = index if index is not None else _CanonicalIndex(artifact, budget,
                row_limit=row_limit, native_scope=native_scope, memory=self._memory)
            if key[0] in self._prediction_paths:
                self._prediction_remaining -= self._indexes[key]._array_counts.get(("rows",), 0)
        return self._indexes[key]

    def _fold_spec(self, item, budget):
        index = self._index(item["fold_spec_artifact"], budget)
        if index.selector is None:
            return index, index.whole(), None
        # Only the original small descriptor is decoded.  Its training parents
        # are neither opened nor passed to Research's recursive saved loader.
        path = index.path.with_name("manifest.json"); key = (str(path), "fold_descriptor")
        if key not in self._indexes:
            require(path.stat().st_size == self._inventory_sizes[str(path)], "Saved fold descriptor changed after inventory")
            self._indexes[key] = _CanonicalIndex(None, budget, memory=self._memory, descriptor_path=path)
        descriptor = self._indexes[key].whole()
        fields(descriptor, "contract_version fold_ref files")
        names = {"fold.json", "feature-slice.json", "label-slice.json", "dataset.json", "model.json",
                 "predictions.json", "signal-evidence.json", "booster.txt"}
        require(descriptor["contract_version"] in ("stock_ml_fold_manifest_v1", "stock_ml_fold_manifest_v2") and type(descriptor["files"]) is dict and
                set(descriptor["files"]) == names, "Original saved fold manifest required")
        for reference in descriptor["files"].values():
            digest(reference)
        spec = index.value(index.selector)
        header = index.object_header(("definition",))
        fields(header, "contract_version content_digest definition_ref status feature_ref label_ref dataset_ref model_ref signal_run_ref evidence_ref fold_ref engine_admission limitations")
        refs = {name: header[name] for name in ("feature_ref", "label_ref", "dataset_ref", "model_ref", "signal_run_ref", "evidence_ref")}
        require((descriptor["contract_version"], header["contract_version"], spec.get("contract_version")) in {
                    ("stock_ml_fold_manifest_v1", "stock_ml_fold_v1", "stock_ml_fold_spec_v1"),
                    ("stock_ml_fold_manifest_v1", "stock_ml_fold_v2", "stock_ml_fold_spec_v2"),
                    ("stock_ml_fold_manifest_v2", "stock_ml_fold_v3", "stock_ml_fold_spec_v1"),
                    ("stock_ml_fold_manifest_v2", "stock_ml_fold_v3", "stock_ml_fold_spec_v2")} and header["status"] == "COMPLETE" and
                header["content_digest"] == index.unsigned_digest("content_digest") and
                header["definition_ref"] == index.span_digest(("definition",)) and
                header["fold_ref"] == Document.from_dict({"definition_ref": header["definition_ref"], **refs}).identity ==
                descriptor["fold_ref"] == item["fold_ref"] and
                index.file_digest == descriptor["files"]["fold.json"], "Original saved fold wrapper identity mismatch")
        require(all(item[name] == header[name] for name in ("feature_ref", "model_ref", "signal_run_ref")),
                "Original saved fold wrapper stage linkage mismatch")
        require(index.value(("definition", "fold_spec_ref")) == item["fold_spec_ref"], "Original saved fold child reference mismatch")
        binding = {"parent_ref": index.content_digest, "parent_content_digest": header["content_digest"],
                   "fold_ref": header["fold_ref"], "file_digest": index.file_digest,
                   "selector": list(index.selector), "child_ref": item["fold_spec_ref"],
                   "manifest_ref": self._indexes[key].content_digest, "manifest_file_digest": self._indexes[key].file_digest}
        self._memory.reserve_global((str(index.path), "fold_binding"), _encoded_size(binding))
        return index, spec, (binding, descriptor)

    def audit(self, manifest, *, block_sessions, read_budget, limits, implementation_ref):
        self._prepared = None; self._audited = False
        try:
            audit = _audit(self, _wire(manifest), block_sessions, _budget(read_budget), limits, implementation_ref)
            self._audited = True
            return audit
        except (KeyError, IndexError, TypeError, OSError) as exc:
            self._close_private_views()
            self._prepared = None
            raise ContractError("Malformed or unreadable fixed stock source: " + str(exc)) from exc
        except BaseException:
            self._close_private_views()
            self._prepared = None
            self._audited = False
            raise

    def iter_blocks(self, manifest, *, block_sessions, read_budget) -> Iterator[StockInputBlock]:
        from .stock_stream_contracts import validate_manifest
        integer(block_sessions, 1); read_budget = _budget(read_budget)
        manifest = validate_manifest(manifest)
        require(self._audited and self._prepared is not None and self._prepared["request_ref"] == manifest["request_ref"],
                "Stock source needs this invocation's complete first-pass audit")
        require(self._memory.global_bytes <= read_budget["max_decoded_bytes"], "Stock cumulative decoded cache exceeds execution budget")
        self._memory.limit = read_budget["max_decoded_bytes"]
        for index in self._indexes.values():
            index.unchanged()
        calendar = manifest["scope"]["calendar"]
        for offset in range(0, len(calendar), block_sessions):
            days = tuple(calendar[offset:offset+block_sessions])
            block = self._block(manifest, days, read_budget)
            try:
                yield block
            finally:
                reservation = block._reservation
                block = None
                reservation.close()
        for index in self._indexes.values():
            index.unchanged()

    def _native(self, role, kind, days, budget, *, reservation=None):
        """Read original sections once each; multiple query files may share a role."""
        selected = {}
        for day in days:
            entry = self._prepared["native_by_day"].get((role, kind, day))
            require(entry is not None, "Missing fixed stock native source: " + role + "/" + kind + "/" + str(day))
            selected.setdefault(entry, []).append(day)
        require(len(selected) == 1, "Stock block crosses native parent scope; reduce block_sessions")
        index, selected_days = next(iter(selected.items()))
        return index.batch_view(selected_days, budget, reservation=reservation)

    def _block(self, manifest, days, budget):
        from .stock_market import _project_stock_rows
        stage = self._memory.stage()
        try:
            batches = []; bindings = []; native_used = 0
            for kind in ("states", "market", "limits", "factor", "membership"):
                batch, used, refs = self._native("execution", kind, days, budget, reservation=stage)
                native_used += used; batches.append(batch); bindings += refs
            prepared = self._prepared
            # Bound both the original projection and expanded-row construction
            # while original native rows are still live.  This is a conservative
            # byte bound, not a numeric or execution-policy approximation.
            upper = 2*native_used + 1024*len(days)*len(manifest["scope"]["execution_universe"])
            stage.reserve(upper)
            rows = _project_stock_rows(batches=batches[:4], universe=manifest["scope"]["execution_universe"],
                calendar=list(days), identities=prepared["rules_index"][0], source_refs=prepared["source_refs"],
                membership_batch=batches[4], expanded=True)
            row_bytes = sum(_encoded_size(row) for row in rows)
            require(row_bytes <= upper, "Stock projection exceeds reserved decoded growth bound")
            stage.release(upper); stage.reserve(row_bytes)
            del batches, batch
            # Byte bindings remain owned by this block; the native row views do not.
            binding_bytes = sum(_encoded_size(binding) for binding in bindings)
            stage.release(native_used - binding_bytes)
            from .stock_inputs import _validate_stock_market_row
            universe_set, days_set = set(manifest["scope"]["execution_universe"]), set(days)
            for row in rows:
                _validate_stock_market_row({key: value for key, value in row.items() if not key.startswith("_stock_")},
                    universe=universe_set, calendar=days_set, refs=prepared["source_refs"], full=True)
            signals = {}
            parent_bindings = set()
            for day in days:
                if day not in prepared["trade_map"]:
                    continue
                frame, feature = prepared["trade_map"][day]
                frame["spec_index"].unchanged(); frame["model_index"].unchanged()
                binding = frame["parent_binding"]
                if binding is not None and binding["parent_ref"] not in parent_bindings:
                    parent_bindings.add(binding["parent_ref"])
                    require(frame["spec_index"].span_digest(frame["spec_index"].selector) == binding["child_ref"],
                            "Original saved fold child changed after admission")
                    bindings.append(binding)
                values, used, refs = frame["index"].rows(("rows",), [feature], budget, reservation=stage)
                bindings += refs
                indexed = {(row["session"], row["security_id"]): row for row in values}
                require(len(indexed) == len(values), "Duplicate original saved prediction key")
                signals[day] = (frame["header"], indexed)
            return StockInputBlock(days, {(r["session"], r["security_id"]): r for r in rows},
                                   signals, tuple(bindings), stage.used, stage)
        except Exception:
            stage.close()
            raise


LocalStockInputSource = StockInputSource


_PAIR_META = ("usable_from", "missing_reason", "availability_basis", "raw_batch_id", "revision_id",
              "revision_sequence", "first_observed_at", "source_available_at", "evidence_ref")
_KINDS = {"market_state_diagnostics": "states", "market_daily": "market", "price_limits": "limits",
          "adjustment_factors": "factor", "universe_membership": "membership"}


def _native_header(index, entry, manifest):
    from .stock_market import _native_query, STOCK_EVENT_FIELDS
    context = index.header(); query = context.get("query", {})
    role = entry["role"]
    require(role in ("execution", "prediction_basis"), "Unsupported stock native input role")
    require(context.get("contract_version") == "data_batch_v1" and
            context.get("snapshot_id") == manifest["market_input"][role.replace("prediction_basis", "model") + "_snapshot_id"] and
            bool(context.get("reader_version")), "Stock native Snapshot/Reader binding mismatch")
    require(index.content_digest == entry["native_ref"], "Stock native original identity mismatch")
    expected_units = {"market_daily": {"open": "CNY/share", "close": "CNY/share", "volume_shares": "shares"},
        "price_limits": {"up_limit": "CNY/share", "down_limit": "CNY/share"},
        "adjustment_factors": {"factor": "dimensionless"},
        "corporate_actions": {"cash_dividend_before_tax_per_share": "CNY/share"}}
    domain = context.get("domain")
    for name, unit in expected_units.get(domain, {}).items():
        require(index.value(("field_meta", name, "unit")) == unit, "Unexpected original stock native unit")
    if domain == "corporate_actions":
        require(role == "execution" and query.get("purpose") == "market_replay" and
                query.get("time_field") in ("ex_date", "record_date"),
                "Unsupported stock action source role")
        kind = "actions-ex" if query["time_field"] == "ex_date" else "actions-record"
        days = [None]
        _native_query({"context": context}, domain=domain, names=STOCK_EVENT_FIELDS,
            universe=manifest["scope"]["execution_universe"], calendar=manifest["scope"]["calendar"],
            time_field=query["time_field"])
    else:
        require(domain in _KINDS, "Unsupported fixed stock native domain")
        kind = _KINDS[domain]; days = query.get("sessions", [])
        require(type(days) is list and bool(days) and days == sorted(set(days)), "Ordered native stock sessions required")
        calendar = manifest["scope"]["calendar"]
        warmup = manifest["market_input"]["warmup_sessions"]
        if kind == "membership":
            require(query.get("fields") == ["is_member"] and query.get("symbols") == manifest["scope"]["prediction_universe"] and
                    query.get("purpose") == "decision_facts" and query.get("universe_id") == "csi300" and
                    query.get("pit_policy") == "best_effort_vendor_v1" and days == calendar and
                    set(query.get("cutoff_by_session", {})) == set(days), "Original PIT stock membership query required")
            from ..core.stock_portfolio import instant
            require(all(instant(query["cutoff_by_session"][d]) == instant(d + "T20:30:00+08:00") for d in days),
                    "Stock membership cutoff changed")
        else:
            require(query.get("purpose") == "market_replay", "Stock native purpose mismatch")
            names = {"states": ("close",), "market": ("open", "close", "volume_shares"),
                     "limits": ("up_limit", "down_limit"), "factor": ("factor",)}[kind]
            requested = calendar if kind == "states" else days
            require(kind == "states" or days in (calendar, warmup), "Unexpected stock native query interval")
            _native_query({"context": context}, domain="market_daily" if kind == "states" else domain,
                names=names, universe=manifest["scope"]["execution_universe"], calendar=requested, states=kind == "states")
    expected_days = set(days)
    require(all(day in expected_days for path, day in index.groups if path in (("records",),) or
                (len(path) == 3 and path[0] == "field_meta")), "Native stock row/metadata lies outside its original query")
    return kind, days, context


def _grid(batch, days, universe):
    records, meta = {}, {}
    expected = {(day, security) for day in days for security in universe}
    for row in batch["records"]:
        key = (row["session"], row["security_id"])
        require(key not in records, "Duplicate native stock row key")
        records[key] = row
    require(set(records) == expected, "Incomplete original native stock grid")
    for name, field in batch["field_meta"].items():
        indexed = {}
        for row in field["by_key"]:
            key = (row["session"], row["security_id"])
            require(key not in indexed, "Duplicate native stock metadata key")
            indexed[key] = row
        require(set(indexed) == expected, "Incomplete original native stock metadata grid")
        meta[name] = indexed
    return records, meta


def _pair(left, right, days, universe, names):
    lrows, lmeta = _grid(left, days, universe); rrows, rmeta = _grid(right, days, universe)
    require(set(names) <= set(lmeta) and set(names) <= set(rmeta), "Missing paired stock fields")
    for name in names:
        require(left["field_meta"][name]["unit"] == right["field_meta"][name]["unit"], "Stock paired units differ")
        for key in lrows:
            require(lrows[key][name] == rrows[key][name] and
                    all(lmeta[name][key].get(field) == rmeta[name][key].get(field) for field in _PAIR_META),
                    "Stock native dual-source value/provenance pairing difference")


def _warmup_visibility(batch, days, universe, names):
    from .stock_market import _visible
    records, metadata = _grid(batch, days, universe)
    for key, row in records.items():
        for name in names:
            fact = metadata[name][key]
            if row[name] is None:
                require(bool(fact.get("missing_reason")), "Null warmup fact requires original missing reason")
            else:
                try:
                    visible = _visible(fact, batch["context"]["query"]["cutoff_by_session"][key[0]])
                except (ValueError, TypeError) as exc:
                    raise ContractError("Invalid original stock warmup availability timestamp") from exc
                require(visible, "Unavailable original stock warmup fact")


def _frames(source, manifest, budget):
    from .stock_schedule import _verify_model
    from ..core.stock_portfolio import instant
    calendar = manifest["scope"]["calendar"]
    previous = {d: calendar[i-1] for i, d in enumerate(calendar) if i}
    trade_map, frames, seen = {}, [], set()
    last = None
    for item in manifest["prediction_input"]["frames"]:
        spec_index, spec, parent = source._fold_spec(item, budget)
        model_index = source._index(item["model_metadata_artifact"], budget)
        model = model_index.whole()
        fields(spec, "contract_version training_window fit_session fit_cutoff simulated_model_available_at oos_trade_sessions inference_cutoff_by_session evaluation_cutoff")
        trades = spec["oos_trade_sessions"]
        require(type(trades) is list and bool(trades) and trades == sorted(set(trades)) and
                all(day in previous for day in trades), "Original fold OOS scope required before compact prediction indexing")
        prediction_path = str(_path(item["prediction_artifact"]))
        source._scan_row_limits[prediction_path] = min(source._scan_row_limits[prediction_path],
            len(trades)*len(manifest["scope"]["prediction_universe"]))
        frame_index = source._index(item["prediction_artifact"], budget)
        if parent is not None:
            _, descriptor = parent
            require(model_index.path == spec_index.path.with_name("model.json") and
                    frame_index.path == spec_index.path.with_name("predictions.json") and
                    model_index.file_digest == descriptor["files"]["model.json"] and
                    frame_index.file_digest == descriptor["files"]["predictions.json"], "Original saved fold stage bytes mismatch")
        header = frame_index.object_header(("rows",))
        require(spec["contract_version"] in ("stock_ml_fold_spec_v1", "stock_ml_fold_spec_v2"), "Saved fold spec required")
        with source._memory.stage() as proof_stage:
            proof_stage.reserve(3*(_encoded_size(spec)+_encoded_size(model)))
            _verify_model(model)
            spec_identity = Document.from_dict(spec).identity
        fields(header, "contract_version signal_run_ref signal_stage score_semantics score_unit feature_ref model_ref limitations universe fold_spec_ref clock_basis")
        require(header["contract_version"] == "stock_prediction_run_v2" and
                frame_index.unsigned_digest("signal_run_ref") == header["signal_run_ref"], "Original saved v2 signal identity mismatch")
        require(header["universe"] == manifest["scope"]["prediction_universe"], "Saved fold prediction union mismatch")
        require(spec_identity == item["fold_spec_ref"] == header["fold_spec_ref"] and
                model["model_ref"] == item["model_ref"] == header["model_ref"] and
                model["feature_ref"] == item["feature_ref"] == header["feature_ref"] and
                item["signal_run_ref"] == header["signal_run_ref"] and
                model["target_semantics"] == header["score_semantics"], "Original saved fold/model/prediction linkage mismatch")
        require(item["fold_ref"] not in seen, "Duplicate saved fold ref")
        digest(item["fold_ref"]); seen.add(item["fold_ref"])
        trades = spec["oos_trade_sessions"]
        require(type(trades) is list and bool(trades) and trades == sorted(set(trades)) and
                all(d in previous for d in trades) and (last is None or last < trades[0]), "Overlapping or unordered stock OOS folds")
        last = trades[-1]; features = [previous[d] for d in trades]
        require(set(spec["inference_cutoff_by_session"]) == set(features) and
                {d for path, d in frame_index.groups if path == ("rows",)} == set(features), "Stock fold predecessor coverage mismatch")
        require(spec["fit_session"] in calendar and
                instant(model["fit_cutoff"]) == instant(spec["fit_cutoff"]) and
                instant(model["simulated_available_at"]) == instant(spec["simulated_model_available_at"]) ==
                instant(spec["fit_session"] + "T20:45:00+08:00") and
                instant(spec["fit_cutoff"]) < instant(model["simulated_available_at"]), "Fold model clock conflict")
        frame = {"index": frame_index, "header": header, "spec": spec, "model": model, "item": item,
                 "spec_index": spec_index, "model_index": model_index, "parent_binding": parent[0] if parent else None}
        frames.append(frame)
        for trade, feature in zip(trades, features):
            require(instant(spec["inference_cutoff_by_session"][feature]) == instant(feature+"T21:00:00+08:00") and
                    instant(model["simulated_available_at"]) < instant(feature+"T21:00:00+08:00") <=
                    instant(trade+"T08:55:00+08:00"), "Fold inference/decision clock conflict")
            require(trade not in trade_map, "Duplicate OOS trade date")
            trade_map[trade] = (frame, feature)
    required = calendar[calendar.index(manifest["scope"]["start_session"]):calendar.index(manifest["scope"]["end_session"])+1]
    require(all(d in trade_map for d in required), "Missing required stock OOS trade date")
    return frames, trade_map


def _audit(source, manifest, block_sessions, budget, limits, implementation_ref):
    from .stock_stream_contracts import validate_manifest, validate_limits
    from .stock_market import _project_stock_actions, _visible
    from .profiles import stock_daily_open_profile_v2
    from .stock_rules import csi300_stock_portfolio_policy
    from ..core.stock_rules import validate_execution_rules, support_ref, listed, LIFECYCLE_POLICY
    from ..core.stock_portfolio import StockPredictionFrame, validate_stock_predictions, instant
    from ..core.portfolio import decimal
    from .stock_inputs import validate_cash_action
    integer(block_sessions, 1); digest(implementation_ref)
    manifest = validate_manifest(manifest); validate_limits(limits)
    inventory = source.inventory(manifest)
    require(inventory["input_bytes"] <= limits["max_input_bytes"] and inventory["folds"] <= limits["max_folds"],
            "Stock source inventory exceeds input/fold budget")
    require(all(inventory["declared_rows"][name] <= limits["max_" + name] for name in ("market_rows", "prediction_rows")),
            "Stock source inventory exceeds declared row budget")
    # An audit is never a cache exemption across invocations.
    source._close_private_views()
    source._indexes = {}; source._prepared = None
    source._file_parse_events = []; source._cjson_spool_bytes = 0
    source._memory = _DecodedMemory(budget["max_decoded_bytes"], scalar_cache_bytes=source._scalar_cache_bytes)
    source._memory.reserve_global(("request", manifest["request_ref"]), _encoded_size(manifest))
    source._inventory_sizes = {str(_path(item["artifact"])) if "artifact" in item else item["manifest_uri"]: item["file_bytes"]
                               for item in inventory["files"]}
    source._scan_row_limits = {}
    source._scan_native_scopes = {}
    for item in manifest["market_input"]["native_inputs"]:
        key = str(_path(item["artifact"]))
        source._scan_row_limits[key] = limits["max_market_rows"]
        source._scan_native_scopes[key] = (manifest, limits)
    for frame in manifest["prediction_input"]["frames"]:
        source._scan_row_limits[str(_path(frame["prediction_artifact"]))] = limits["max_prediction_rows"]
    source._prediction_paths = {str(_path(frame["prediction_artifact"])) for frame in manifest["prediction_input"]["frames"]}
    source._prediction_remaining = limits["max_prediction_rows"]
    scope = manifest["scope"]; calendar = scope["calendar"]; universe = scope["execution_universe"]
    require(scope["prediction_universe"] == universe, "Full-stock prediction/execution union mismatch")
    require(bool(manifest["market_input"]["warmup_sessions"]) and
            manifest["market_input"]["warmup_sessions"][-1] < calendar[0], "Explicit preceding stock warmup required")
    profile_index = source._index(manifest["profile_input"]["artifact"], budget)
    profile = profile_index.whole()
    require(profile_index.content_digest == manifest["profile_input"]["profile_ref"], "Stock profile identity mismatch")
    rules = profile["stock_execution_rules"]; rules_index = validate_execution_rules(rules)
    require(profile == stock_daily_open_profile_v2(execution_rules=rules, fee_schedule=profile["stock_fee_schedule"],
        unknown_status_policy=profile.get("unknown_status_policy")) and rules["calendar"] == calendar and
        rules["universe"] == universe and scope["supported_universe_ref"] == support_ref(universe), "Stock profile/rules scope mismatch")
    for name in ("stock_execution_rules_ref", "stock_fee_schedule_ref"):
        require(manifest["profile_input"][name] == profile[name], "Stock profile nested reference mismatch")
    require(manifest["portfolio_policy"] == csi300_stock_portfolio_policy(top_k=manifest["portfolio_policy"]["top_k"],
        execution_universe=universe, execution_rules=rules), "Stock portfolio policy mismatch")
    native_by_day, contexts = {}, []
    for item in manifest["market_input"]["native_inputs"]:
        index = source._index(item["artifact"], budget)
        kind, days, context = _native_header(index, item, manifest)
        contexts.append(context)
        for day in days:
            key = (item["role"], kind, day)
            require(key not in native_by_day, "Duplicate/overlapping stock native query scope")
            native_by_day[key] = index
    source_refs = [native_by_day["execution", kind, calendar[0] if kind not in ("actions-ex", "actions-record") else None].content_digest
                   for kind in ("states", "market", "limits", "factor", "actions-ex", "actions-record", "membership")]
    frames, trade_map = _frames(source, manifest, budget)
    prepared = {"request_ref": manifest["request_ref"], "native_by_day": native_by_day,
                "rules_index": rules_index, "source_refs": source_refs, "trade_map": trade_map}
    source._prepared = prepared
    try:
        with source._memory.stage() as stage:
            actions = []
            for kind in ("actions-ex", "actions-record"):
                view, _, action_bindings = source._native("execution", kind, [None], budget, reservation=stage)
                actions.append(view)
            margin = 10 + 2*sum(len(view["records"]) for view in actions)
            require(source._memory.available > margin, "Stock decoded event cache budget exceeded")
            cash, diagnostics, blocks = _project_stock_actions(batches=actions, universe=universe,
                source_refs=source_refs, max_event_bytes=source._memory.available-margin)
            source._memory.reserve_global(("projected", "events"), _encoded_size([cash, diagnostics, blocks]))
            del actions, view, action_bindings
        for action in cash:
            validate_cash_action(action, universe, source_refs)
            require(action["record_session"] in calendar or action["record_session"] < calendar[0], "Missing stock record calendar")
            for name in ("ex_session", "pay_session"):
                require(action[name] is None or action[name] > scope["end_session"] or action[name] < calendar[0] or action[name] in calendar,
                        "Stock cash phase calendar mismatch")
        market_rows = prediction_rows = 0
        lifecycle = {"pre_listing_null": 0, "listed_nonmember_gap": 0, "member_gap": 0, "held_gap": 0}
        previous_factor = {}
        factor_blocks = []
        for offset in range(0, len(calendar), block_sessions):
            days = tuple(calendar[offset:offset+block_sessions])
            # Pair original price/factor/member facts and provenance, not synthetic PASS flags.
            for kind, names in (("market", ("open", "high", "low", "close", "volume_shares", "amount_cny")),
                                ("factor", ("factor",)), ("membership", ("is_member",))):
                with source._memory.stage() as stage:
                    left, _, left_bindings = source._native("prediction_basis", kind, days, budget, reservation=stage)
                    right, _, right_bindings = source._native("execution", kind, days, budget, reservation=stage)
                    _pair(left, right, days, universe, names)
                    del left, right, left_bindings, right_bindings
            with source._memory.stage() as stage:
                factor, _, factor_bindings = source._native("execution", "factor", days, budget, reservation=stage)
                frows, fmeta = _grid(factor, days, universe)
                require(factor["field_meta"]["factor"]["unit"] == "dimensionless", "Unexpected stock factor unit")
                for day in days:
                    for security in universe:
                        key = day, security; value = frows[key]["factor"]; metadata = fmeta["factor"][key]
                        if value is None:
                            require(bool(metadata.get("missing_reason")), "Null stock factor requires native missing reason")
                            source._memory.replace_global(("previous_factor", security), _encoded_size({security: None}))
                            previous_factor[security] = None
                        else:
                            require(listed(rules_index[0][security], day) and _visible(metadata, day+"T20:30:00+08:00") and
                                    decimal(str(value), minimum=0) > 0, "Unavailable/invalid original stock factor")
                            old = previous_factor.get(security)
                            if old is not None and old != value:
                                size = _object_size([(name, _encoded_size(item)) for name, item in (
                                    ("security_id", security), ("effective_session", day), ("available_at", metadata["usable_from"]),
                                    ("reason", "UNEXPLAINED_FACTOR_CHANGE"), ("source_refs", [source_refs[3]]))])
                                source._memory.reserve_global(("factor_block", security, day), size)
                                item = {"security_id": security, "effective_session": day,
                                        "available_at": metadata["usable_from"], "reason": "UNEXPLAINED_FACTOR_CHANGE", "source_refs": [source_refs[3]]}
                                factor_blocks.append(item)
                            source._memory.replace_global(("previous_factor", security), _encoded_size({security: value}))
                            previous_factor[security] = value
                del factor, frows, fmeta, factor_bindings, metadata
            block = source._block(manifest, days, budget)
            try:
                market_rows += len(block.market_rows)
                require(market_rows <= limits["max_market_rows"], "Stock market row budget exceeded")
                for row in block.market_rows.values():
                    if row["market_state"] == "not_listed":
                        lifecycle["pre_listing_null"] += 1
                    elif row["_stock_listed"] and (not row["_stock_factor_valid"] or
                            any(row[name] is None for name in ("open", "close", "volume_shares", "limit_up", "limit_down"))):
                        lifecycle["member_gap" if row["_stock_member"] else "listed_nonmember_gap"] += 1
                row = None
                for trade, (header, indexed) in block.signals.items():
                    feature = trade_map[trade][1]
                    with source._memory.stage() as stage:
                        validation_bytes = 3*(_encoded_size(header) + sum(_encoded_size(row) for row in indexed.values()) + 32)
                        stage.reserve(validation_bytes)
                        # Reading view only; the original parent identity was checked above.
                        validated_wire = checked = membership = members = member_metadata = member_bindings = None
                        try:
                            validated_wire, checked = validate_stock_predictions(StockPredictionFrame.from_dict({**header, "rows": list(indexed.values())}))
                            require(len(checked) == len(universe), "Incomplete original saved prediction date group")
                            membership, _, member_bindings = source._native("execution", "membership", [feature], budget, reservation=stage)
                            members, member_metadata = _grid(membership, [feature], universe)
                            for key, row in checked.items():
                                require(row["member"] == members[key]["is_member"] and
                                        instant(row["feature_knowledge_cutoff"]) == instant(feature+"T20:30:00+08:00") and
                                        instant(row["knowledge_cutoff"]) == instant(row["available_at"]) == instant(feature+"T21:00:00+08:00") and
                                        instant(row["simulated_model_available_at"]) == instant(trade_map[trade][0]["model"]["simulated_available_at"]),
                                        "Saved prediction membership/fixed clock mismatch")
                            prediction_rows += len(checked)
                            require(prediction_rows <= limits["max_prediction_rows"], "Stock prediction row budget exceeded")
                        finally:
                            validated_wire = checked = membership = members = member_metadata = member_bindings = row = None
            finally:
                row = header = indexed = None
                reservation = block._reservation
                block = None
                reservation.close()
        warmup = manifest["market_input"]["warmup_sessions"]
        for day in warmup:
            for kind, names in (("market", ("open", "high", "low", "close", "volume_shares", "amount_cny")), ("factor", ("factor",))):
                with source._memory.stage() as stage:
                    left, _, left_bindings = source._native("prediction_basis", kind, [day], budget, reservation=stage)
                    right, _, right_bindings = source._native("execution", kind, [day], budget, reservation=stage)
                    _pair(left, right, [day], universe, names)
                    _warmup_visibility(left, [day], universe, names)
                    _warmup_visibility(right, [day], universe, names)
                    del left, right, left_bindings, right_bindings
        # Original v6 emits factors security-major, even when delivery is date-major.
        blocks.extend(sorted(factor_blocks, key=lambda item: (item["security_id"], item["effective_session"])))
        del previous_factor
        for security in universe:
            source._memory.release_global(("previous_factor", security))
        limitations = sorted({x for c in contexts for x in c.get("limitations", [])}) + [
            "OBSERVED IMPLEMENTED ACTIONS ONLY: nonimplemented uncertainty remains diagnostic; no complete action-history claim.",
            "Stock cash source lacks PAY dates; gross-before-tax receivables remain pending until verified native payment.",
            "Native UNKNOWN and actual field availability are retained; daily open/volume/limits are retrospective execution evidence."]
        market_header = {"contract_version": "market_replay_v4", "price_basis": "unadjusted", "calendar": calendar,
                         "universe": universe, "cash_dividends": cash, "action_diagnostics": diagnostics,
                         "action_blocks": blocks, "source_refs": source_refs, "limitations": limitations,
                         "stock_execution_rules_ref": profile["stock_execution_rules_ref"], "membership_ref": source_refs[6],
                         "lifecycle_policy": LIFECYCLE_POLICY}
        source._memory.reserve_global(("projected", "market_header"), _encoded_size({key: value for key, value in market_header.items()
            if key not in ("cash_dividends", "action_diagnostics", "action_blocks")}))
        receipt = {"contract_version": "stock_input_audit_v1", "request_ref": manifest["request_ref"],
                   "market_ref": manifest["market_input"]["market_ref"], "prediction_ref": manifest["prediction_input"]["prediction_ref"],
                   "profile_ref": manifest["profile_input"]["profile_ref"], "implementation_ref": implementation_ref,
                   "counts": {"input_bytes": inventory["input_bytes"], "input_files": len(inventory["files"]),
                              "folds": len(frames), "prediction_rows": prediction_rows, "market_rows": market_rows,
                              "cash_actions": len(cash), "action_diagnostics": len(diagnostics), "action_blocks": len(blocks)},
                   "limitations": limitations}
        source._memory.reserve_global(("projected", "audit_receipt"), _encoded_size(receipt))
        return StockSourceAudit(receipt=receipt, globals={"profile": profile, "market_header": market_header,
            "calendar": calendar, "rules_index": rules_index, "profile_bytes": profile_index.size,
            "signal_limitations": sorted({value for frame in frames for value in frame["header"]["limitations"]})},
            lifecycle=lifecycle, manifest=manifest,
            segment_index={index.content_digest: index for index in source._indexes.values()})
    except Exception:
        source._prepared = None
        raise
