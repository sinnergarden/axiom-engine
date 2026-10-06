"""Bounded immutable result parts produced by the existing stock Runtime.

The sink serializes original rows. It does not calculate fees, execute orders,
advance an account or decide which ledger history can be drained.
"""
from hashlib import sha256
import json
from pathlib import Path

from ..core.contracts import ContractError, _walk, digest, fields, integer, require, session


RESULT_KINDS = ("nav", "positions", "decisions", "orders", "fills", "cash_ledger",
                "position_ledger", "session_phases")
RESULT_PART_VERSION = "stock_backtest_result_part_v1"
PHASES = ("SESSION_COMMITTED", "STOPPED_BEFORE_NAV")


def _budget(value):
    fields(value, "max_part_bytes max_buffer_bytes max_total_bytes")
    for item in value.values():
        integer(item, 1)
    return value


def _json_pieces(value, *, piece_chars=512):
    """Canonical UTF-8 in bounded scalar pieces, never a whole string encoding."""
    if type(value) is str:
        yield b'"'
        for start in range(0, len(value), piece_chars):
            # Each source slice is bounded before escaping or UTF-8 encoding.
            yield json.dumps(value[start:start + piece_chars], ensure_ascii=False)[1:-1].encode("utf-8")
        yield b'"'
    elif type(value) is dict:
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield from _json_pieces(key, piece_chars=piece_chars)
            yield b":"
            yield from _json_pieces(value[key], piece_chars=piece_chars)
        yield b"}"
    elif type(value) in (list, tuple):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _json_pieces(item, piece_chars=piece_chars)
        yield b"]"
    else:
        yield json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def canonical_size(value, maximum, *, scratch_bytes=None):
    """Measure without a growing row buffer, rejecting an oversized scalar."""
    integer(maximum, 1)
    try:
        for _ in _walk(value):
            pass
        piece_chars = max(1, min(512, (scratch_bytes or maximum) // 6))
        size = 0
        for piece in _json_pieces(value, piece_chars=piece_chars):
            size += len(piece)
            require(size <= maximum, "result row exceeds output buffer/part budget")
        return size
    except (UnicodeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("result row cannot be encoded as canonical JSON") from exc


def bounded_canonical_bytes(value, maximum, *, scratch_bytes=None):
    """Check encoded size before allocating the growing row buffer."""
    canonical_size(value, maximum, scratch_bytes=scratch_bytes)
    try:
        piece_chars = max(1, min(512, (scratch_bytes or maximum) // 6))
        buffer = bytearray()
        for piece in _json_pieces(value, piece_chars=piece_chars):
            require(len(buffer) + len(piece) <= maximum,
                    "result row exceeds output buffer/part budget")
            buffer.extend(piece)
        return buffer
    except (UnicodeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("result row cannot be encoded as canonical JSON") from exc


class StockResultSink:
    """Exclusive-create local parts, yielding receipts only after hash verification.

    Every append seals its session's residual part. Large sessions may produce
    several parts. The automatic phase marker describes each sealed part; it is
    not an account event, and must not be counted as an extra ledger sequence.
    """
    def __init__(self, root, *, run_id):
        digest(run_id)
        self.root = Path(root)
        self.run_id = run_id
        self._rows = {kind: [] for kind in RESULT_KINDS}
        self._encoded_bytes = 0
        self._total_bytes = 0
        self._index = 0
        self._previous = None
        self._active = None
        self._last_session = None
        self._last_sequence = 0
        self._failed = False
        self._finished = False
        self._busy = False
        self._scratch_bytes = 16384
        # Runtime supplies canonical bytes of its undrained original rows.
        # The callback is invocation-local and cannot authorize a drain.
        self._external_pending_bytes = lambda: 0
        self._base_size = sum(len(piece) for piece in self._pieces("sha256:" + "0" * 64))

    @property
    def buffered_bytes(self):
        return self._encoded_bytes

    @property
    def total_bytes(self):
        return self._total_bytes

    def _pieces(self, content_digest=None):
        yield b"{"
        if content_digest is not None:
            yield b'"content_digest":'
            yield from _json_pieces(content_digest, piece_chars=max(1, self._scratch_bytes // 6))
            yield b","
        yield b'"contract_version":"stock_backtest_result_part_v1","part_index":'
        yield str(self._index).encode("ascii")
        yield b',"rows":{'
        for index, kind in enumerate(sorted(RESULT_KINDS)):
            if index:
                yield b","
            yield from _json_pieces(kind, piece_chars=max(1, self._scratch_bytes // 6))
            yield b":["
            for row_index, row in enumerate(self._rows[kind]):
                if row_index:
                    yield b","
                yield row
            yield b"]"
        yield b'},"run_id":'
        yield from _json_pieces(self.run_id, piece_chars=max(1, self._scratch_bytes // 6))
        yield b"}"

    def _size(self):
        # Digest syntax and width are fixed, so this is exact before allocation.
        return (self._base_size + len(str(self._index)) - 1 + self._encoded_bytes +
                sum(max(0, len(rows) - 1) for rows in self._rows.values()))

    def _capacity(self, budget):
        external = self._external_pending_bytes()
        integer(external)
        capacity = budget["max_buffer_bytes"] - external - self._scratch_bytes
        require(capacity > 0, "unified result buffer budget exceeded")
        return capacity

    def _start(self, day, phase, committed_sequence, budget):
        self._scratch_bytes = max(8, min(16384, budget["max_buffer_bytes"] // 8))
        capacity = self._capacity(budget)
        marker = {"session": day, "phase": phase, "committed_sequence": committed_sequence}
        size = canonical_size(marker, capacity, scratch_bytes=self._scratch_bytes)
        require(self._encoded_bytes + size <= capacity, "result buffer budget exceeded")
        part_size = self._size() + size + bool(self._rows["session_phases"])
        require(part_size <= budget["max_part_bytes"], "result part budget exceeded")
        require(self._total_bytes + part_size <= budget["max_total_bytes"], "total result budget exceeded")
        encoded = bounded_canonical_bytes(marker, capacity - self._encoded_bytes, scratch_bytes=self._scratch_bytes)
        self._active = (day, phase, committed_sequence)
        self._rows["session_phases"].append(encoded)
        self._encoded_bytes += len(encoded)
        self._check_budget(budget)

    def _check_budget(self, budget):
        require(self._encoded_bytes <= self._capacity(budget), "unified result buffer budget exceeded")
        size = self._size()
        require(size <= budget["max_part_bytes"], "result part budget exceeded")
        require(self._total_bytes + size <= budget["max_total_bytes"], "total result budget exceeded")

    def _seal(self, budget):
        require(self._active is not None, "no active result part")
        self._check_budget(budget)
        unsigned_hash = sha256()
        for piece in self._pieces():
            unsigned_hash.update(piece)
        content_digest = "sha256:" + unsigned_hash.hexdigest()
        expected_hash = sha256()
        for piece in self._pieces(content_digest):
            expected_hash.update(piece)
        expected_digest = "sha256:" + expected_hash.hexdigest()
        expected_size = self._size()
        path = self.root / f"part-{self._index:08d}.json"
        created = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                created = True
                written = 0
                for piece in self._pieces(content_digest):
                    require(written + len(piece) <= budget["max_part_bytes"] and
                            self._total_bytes + written + len(piece) <= budget["max_total_bytes"],
                            "result write budget exceeded")
                    require(output.write(piece) == len(piece), "incomplete result part write")
                    written += len(piece)
            observed = sha256()
            observed_size = 0
            read_size = min(self._scratch_bytes, budget["max_part_bytes"])
            with path.open("rb") as saved:
                while True:
                    piece = saved.read(read_size)
                    if not piece:
                        break
                    observed_size += len(piece)
                    require(observed_size <= expected_size, "result part changed during verification")
                    observed.update(piece)
                    del piece
            require(written == observed_size == expected_size and
                    "sha256:" + observed.hexdigest() == expected_digest,
                    "result part byte/content verification failed")
        except Exception:
            self._failed = True
            # A failed file is never a commit receipt. Preserve buffered rows so
            # the caller cannot mistake the failure for permission to drain.
            if created:
                path.unlink(missing_ok=True)
            raise
        day, _, sequence = self._active
        receipt = {"artifact": {"artifact_type": "stock_backtest_result_part",
            "artifact_id": f"{self.run_id}:part:{self._index}",
            "contract_version": RESULT_PART_VERSION, "manifest_uri": str(path.resolve()),
            "content_digest": expected_digest}, "part_index": self._index,
            "start_session": day, "end_session": day,
            "first_committed_sequence": sequence, "last_committed_sequence": sequence,
            "previous_part_digest": self._previous,
            "row_counts": {kind: len(self._rows[kind]) for kind in RESULT_KINDS}}
        self._total_bytes += expected_size
        self._previous = expected_digest
        self._index += 1
        self._rows = {kind: [] for kind in RESULT_KINDS}
        self._encoded_bytes = 0
        self._active = None
        return receipt

    def append(self, *, session: str, phase: str, rows, committed_sequence: int,
               write_budget: dict):
        budget = _budget(write_budget)
        require(not self._failed and not self._finished and not self._busy,
                "result sink unavailable or append already active")
        globals()["session"](session)
        require(phase in PHASES, "unsupported result session phase")
        integer(committed_sequence)
        require(self._last_session is None or session >= self._last_session,
                "result session cannot go backwards")
        require(committed_sequence >= self._last_sequence, "result sequence cannot go backwards")
        self._busy = True
        try:
            self._start(session, phase, committed_sequence, budget)
            for kind, row in rows:
                require(kind in RESULT_KINDS and type(row) is dict, "unknown result row group")
                if kind == "session_phases":
                    require(row == {"session": session, "phase": phase,
                                    "committed_sequence": committed_sequence},
                            "result phase row differs from append")
                    del row
                    continue
                row_session = row.get("session", row.get("trade_session"))
                require(row_session == session, "result row outside append session")
                row_sequence = row.get("sequence", row.get("committed_sequence"))
                if row_sequence is not None:
                    integer(row_sequence)
                    require(row_sequence <= committed_sequence, "result row exceeds committed sequence")
                capacity = self._capacity(budget)
                remaining = capacity - self._encoded_bytes
                # First check a row independently. Existing buffered rows may be
                # sealed before encoding this next row into the growing buffer.
                row_size = canonical_size(row, min(capacity, budget["max_part_bytes"]),
                                          scratch_bytes=self._scratch_bytes)
                candidate_size = self._size() + row_size + bool(self._rows[kind])
                if row_size > remaining or candidate_size > budget["max_part_bytes"]:
                    require(sum(len(self._rows[k]) for k in RESULT_KINDS[:-1]) > 0,
                            "single result row exceeds part/buffer budget")
                    yield self._seal(budget)
                    self._start(session, phase, committed_sequence, budget)
                    capacity = self._capacity(budget)
                require(self._encoded_bytes + row_size <= capacity and
                        self._size() + row_size + bool(self._rows[kind]) <= budget["max_part_bytes"],
                        "single result row exceeds part/buffer budget")
                require(self._total_bytes + self._size() + row_size + bool(self._rows[kind]) <=
                        budget["max_total_bytes"], "total result budget exceeded")
                encoded = bounded_canonical_bytes(row, capacity - self._encoded_bytes,
                                                  scratch_bytes=self._scratch_bytes)
                require(self._encoded_bytes + len(encoded) <= self._capacity(budget),
                        "unified result buffer budget changed before row addition")
                self._rows[kind].append(encoded)
                self._encoded_bytes += len(encoded)
                # Ownership moves to the pending part. In particular, the
                # append generator must not retain an old encoded row after
                # _seal clears that part and reports its drain receipt.
                del encoded, row
            yield self._seal(budget)
            self._last_session = session
            self._last_sequence = committed_sequence
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False

    def finish(self, *, write_budget: dict):
        budget = _budget(write_budget)
        require(not self._failed and not self._finished and not self._busy, "result sink unavailable")
        residual = [self._seal(budget)] if self._active is not None else []
        self._finished = True
        return residual
