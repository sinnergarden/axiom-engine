"""Process-local, bounded ownership of fully admitted stock execution inputs.

The private temporary file is not an artifact or a persisted admission flag.
It has no published path and no writer after import. Each account receives
fresh decoded blocks, while original canonical sources are consumed once.
"""
from array import array
from contextlib import closing, contextmanager
import json
import tempfile
from threading import Lock
from time import perf_counter

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import canonical, integer, require
from ..core.stock_portfolio import validate_top_k
from .stock_stream_contracts import identity_view, read_budget, validate_limits, validate_manifest
from .stock_stream_inputs import (StockInputSource, StockInputBlock, StockSourceAudit,
                                  _DecodedMemory, _budget, _encoded_size)
from .stock_stream_outputs import _json_pieces


_IMPORT = object()


def _input_key(plan):
    # Every other manifest field remains bound, including non-TopK policy,
    # profile/fees/rules, clock/action policy, scope, all original refs and
    # their exact ArtifactRef envelopes. Locations retain their existing
    # delivery-only semantics; this owner never opens a replacement path.
    value = identity_view(plan)
    for name in ("request_ref", "account_id", "initial_account"):
        value.pop(name)
    value["portfolio_policy"].pop("top_k")
    return canonical(value).encode("utf-8")


def _block_wire(block):
    return {"sessions": list(block.sessions), "market_rows": list(block.market_rows.values()),
            "signals": [{"session": day, "header": header, "rows": list(rows.values())}
                        for day, (header, rows) in block.signals.items()],
            "bindings": list(block.bindings)}


class AdmittedStockInputs(StockInputSource):
    """Opaque sequential-account source; construct only with admit_stock_inputs.

    close()/context-manager exit release the private store. No account can
    mutate a later account's inputs through its decoded globals or blocks.
    """
    def __init__(self, token, store, offsets, key, block_sessions, inventory, statistics):
        require(token is _IMPORT, "Use admit_stock_inputs; admission flags are unsupported")
        super().__init__()
        self.__store = store
        self.__offsets = offsets
        self.__key = key
        self.__block_sessions = block_sessions
        self.__inventory = inventory
        self.__statistics = statistics
        self.__lock = Lock()
        self.__closed = False

    @property
    def statistics(self):
        return json.loads(canonical(self.__statistics))

    def inventory(self, manifest):
        self.__check(manifest)
        return json.loads(self.__inventory)

    def __check(self, manifest):
        require(not self.__closed, "Admitted stock inputs are closed")
        plan = validate_manifest(manifest)
        validate_top_k(plan["portfolio_policy"]["top_k"], plan["scope"]["execution_universe"])
        require(_input_key(plan) == self.__key, "Stock inputs differ from the complete admitted capability")
        return plan

    @contextmanager
    def execution_scope(self):
        require(self.__lock.acquire(blocking=False), "Admitted stock inputs support sequential accounts only")
        try:
            require(not self.__closed, "Admitted stock inputs are closed")
            yield
        finally:
            self._prepared = None
            self._memory = None
            self.__lock.release()

    def __read(self, number, budget):
        start, length = self.__offsets[2*number:2*number+2]
        stage = self._memory.stage()
        try:
            # Reserve decoded graph and both decoder input buffers before read.
            stage.reserve(3*length)
            raw = bytearray()
            self.__store.seek(start)
            remaining = length
            while remaining:
                chunk = self.__store.read(min(remaining, budget["max_read_bytes"], 65536))
                require(bool(chunk), "Private stock input store truncated")
                raw.extend(chunk); remaining -= len(chunk)
            value = json.loads(raw)
            raw = chunk = None
            stage.release(2*length)
            self.__statistics["owned_read_bytes"] += length
            self.__statistics["owned_record_decodes"] += 1
            return value, stage
        except Exception:
            stage.close()
            raise

    def audit(self, manifest, *, block_sessions, read_budget, limits, implementation_ref):
        plan = self.__check(manifest)
        integer(block_sessions, 1); budget = _budget(read_budget); limits = validate_limits(limits)
        require(block_sessions == self.__block_sessions, "Admitted stock block_sessions must remain fixed")
        inventory = json.loads(self.__inventory)
        require(inventory["input_bytes"] <= limits["max_input_bytes"] and inventory["folds"] <= limits["max_folds"],
                "Admitted stock inventory exceeds this account's limits")
        require(all(inventory["declared_rows"][name] <= limits["max_"+name]
                    for name in ("market_rows", "prediction_rows")), "Admitted stock declared row limit exceeded")
        self._memory = _DecodedMemory(budget["max_decoded_bytes"])
        self._memory.reserve_global("owned_binding", len(self.__key)+len(self.__inventory)+len(self.__offsets)*8)
        self._memory.reserve_global("owned_request", _encoded_size(plan))
        meta, stage = self.__read(0, budget)
        size = stage.used
        stage.close()
        self._memory.reserve_global("owned_globals", size)
        receipt = meta["receipt"]
        require(receipt["implementation_ref"] == implementation_ref, "Admitted stock implementation differs")
        require(all(receipt["counts"][name] <= limits["max_"+name]
                    for name in ("market_rows", "prediction_rows")), "Admitted stock actual row limit exceeded")
        receipt["request_ref"] = plan["request_ref"]
        self._prepared = {"request_ref": plan["request_ref"]}
        self.__statistics["account_bindings"] += 1
        return StockSourceAudit(receipt=receipt, globals=meta["globals"], lifecycle=meta["lifecycle"], manifest=plan)

    def iter_blocks(self, manifest, *, block_sessions, read_budget):
        plan = self.__check(manifest); budget = _budget(read_budget)
        require(block_sessions == self.__block_sessions and self._prepared is not None and
                self._prepared["request_ref"] == plan["request_ref"], "Stock owner needs this account's binding")
        require(self._memory.limit == budget["max_decoded_bytes"], "Stock owner decoded budget changed")
        calendar = plan["scope"]["calendar"]
        for number in range(1, len(self.__offsets)//2):
            value, stage = self.__read(number, budget)
            block = None
            try:
                days = tuple(value["sessions"])
                offset = (number-1)*block_sessions
                require(days == tuple(calendar[offset:offset+block_sessions]), "Private stock block calendar mismatch")
                count = len(value["market_rows"]) + sum(len(item["rows"]) for item in value["signals"])
                stage.reserve(64*count)
                rows = {(row["session"], row["security_id"]): row for row in value["market_rows"]}
                signals = {item["session"]: (item["header"],
                    {(row["session"], row["security_id"]): row for row in item["rows"]}) for item in value["signals"]}
                block = StockInputBlock(days, rows, signals, tuple(value["bindings"]), stage.used, stage)
                value = rows = signals = None
                yield block
            finally:
                value = rows = signals = block = None
                stage.close()

    def close(self):
        require(self.__lock.acquire(blocking=False), "Cannot close stock inputs during account execution")
        try:
            if not self.__closed:
                self.__closed = True
                self.__store.close()
                self.__offsets = array("Q")
                self.__key = self.__inventory = b""
                self._memory = self._prepared = None
        finally:
            self.__lock.release()

    def __enter__(self):
        require(not self.__closed, "Admitted stock inputs are closed")
        return self

    def __exit__(self, *_):
        self.close()


def admit_stock_inputs(manifest, *, source, block_sessions: int, limits: dict,
                       max_owned_bytes: int) -> AdmittedStockInputs:
    """Fully admit original bytes, then capture bounded owned execution blocks.

    The original source path remains available as an exact account oracle.
    No saved accounts, Data access, Research loaders or model calls occur.
    """
    from .stock_stream import _admit
    require(not isinstance(source, AdmittedStockInputs), "Import original stock sources, not an admission marker")
    integer(max_owned_bytes, 1)
    started = perf_counter()
    plan, audit, limits = _admit(manifest, source, block_sessions, limits)
    audited = perf_counter()
    inventory = canonical(source.inventory(plan)).encode("utf-8")
    key = _input_key(plan)
    store = tempfile.TemporaryFile(mode="w+b")
    offsets = array("Q")
    binding_bytes = len(key)+len(inventory)+16*(1+(len(plan["scope"]["calendar"])+block_sessions-1)//block_sessions)
    base_bytes = binding_bytes+_encoded_size(plan)
    peak_owned_decode = 0
    # Compact offsets are charged before growth; raw history never stays live.
    offset_stage = source._memory.stage()
    def write(value):
        offset_stage.reserve(16)
        start = store.tell()
        scratch_bytes = min(1024, limits["max_read_bytes"])
        with source._memory.stage() as scratch:
            scratch.reserve(scratch_bytes)
            for piece in _json_pieces(value, piece_chars=max(1, scratch_bytes//8)):
                require(store.tell()+len(piece) <= max_owned_bytes, "Owned stock input store exceeds max_owned_bytes before write")
                store.write(piece)
        offsets.extend((start, store.tell()-start))
    try:
        write({"receipt": audit.receipt, "globals": audit.globals, "lifecycle": audit.lifecycle})
        meta_bytes = offsets[1]
        peak_owned_decode = base_bytes+3*meta_bytes
        require(peak_owned_decode <= limits["max_block_bytes"], "Owned stock globals exceed decoded budget before account")
        with closing(source.iter_blocks(plan, block_sessions=block_sessions, read_budget=read_budget(limits))) as blocks:
            while True:
                try:
                    block = next(blocks)
                except StopIteration:
                    break
                try:
                    write(_block_wire(block))
                    count = len(block.market_rows)+sum(len(rows) for _, rows in block.signals.values())
                    peak_owned_decode = max(peak_owned_decode, base_bytes+meta_bytes+3*offsets[-1]+64*count)
                    require(peak_owned_decode <= limits["max_block_bytes"], "Owned stock block exceeds decoded budget before account")
                finally:
                    block = None
        store.flush()
        statistics = {"source_admissions": 1, "source_audit_seconds": audited-started,
                      "capture_seconds": perf_counter()-audited, "owned_bytes": store.tell(),
                      "max_owned_decoded_bytes": peak_owned_decode,
                      "owned_blocks": len(offsets)//2-1, "owned_read_bytes": 0,
                      "owned_record_decodes": 0, "account_bindings": 0,
                      "source_operations": {name: sum(index.statistics[name] for index in source._indexes.values())
                                            for name in next(iter(source._indexes.values())).statistics}}
        return AdmittedStockInputs(_IMPORT, store, offsets, key, block_sessions, inventory, statistics)
    except Exception:
        store.close()
        raise
    finally:
        offset_stage.close()
        # The caller can still run the original source, which admits afresh.
        # Its full-history span index is unnecessary after ownership transfer.
        source._indexes = {}; source._prepared = None; source._audited = False
        source._memory = None; source._scan_native_scopes = {}; source._scan_row_limits = {}
