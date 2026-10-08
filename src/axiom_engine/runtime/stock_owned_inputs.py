"""Process-local, bounded ownership of fully admitted stock execution inputs.

The private temporary file is not an artifact or a persisted admission flag.
It has no published path and no writer after import. Each account receives
fresh decoded blocks, while original canonical sources are consumed once.
"""
from array import array
from contextlib import contextmanager
import json
import os
from threading import Lock

from ..core.contracts import canonical, integer, require
from ..core.stock_portfolio import validate_top_k
from .stock_stream_contracts import identity_view, read_budget, validate_limits, validate_manifest
from .stock_stream_inputs import (StockInputSource, StockInputBlock, StockSourceAudit,
                                  _DecodedMemory, _budget, _encoded_size)


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


class AdmittedStockInputs(StockInputSource):
    """Opaque sequential-account source; construct only with admit_stock_inputs.

    close()/context-manager exit release the private store. No account can
    mutate a later account's inputs through its decoded globals or blocks.
    """
    def __init__(self, token, store, offsets, key, block_sessions, inventory, statistics, *, market=None):
        require(token is _IMPORT, "Use admit_stock_inputs; admission flags are unsupported")
        super().__init__()
        self.__owner_pid = os.getpid()
        self.__store = store
        self.__offsets = offsets
        self.__key = key
        self.__block_sessions = block_sessions
        self.__inventory = inventory
        self.__statistics = statistics
        self.__lock = Lock()
        self.__closed = False
        self.__market = market
        self.__owns_market = False
        self.__market_final_stats = None

    @property
    def statistics(self):
        self.__owner()
        value = json.loads(canonical(self.__statistics))
        if self.__owns_market:
            shared = self.__market_final_stats or self.__market.statistics
            left, right = shared["source_operations"], value["source_operations"]
            value["source_operations"] = {name: (max(left.get(name, 0), right.get(name, 0))
                if name == "cjson_peak_tree_rss_bytes" else left.get(name, 0)+right.get(name, 0))
                for name in left.keys() | right.keys()}
            for name in ("source_audit_seconds", "capture_seconds", "owned_bytes", "source_scalar_cache_evictions",
                         "owned_read_bytes", "owned_record_decodes"):
                value[name] += shared[name]
            for name in ("source_decoded_bytes_peak", "max_owned_decoded_bytes"):
                value[name] = max(value[name], shared["source_decoded_bytes_peak"])
        return value

    def inventory(self, manifest):
        self.__check(manifest)
        return json.loads(self.__inventory)

    def __check(self, manifest):
        self.__owner()
        require(not self.__closed, "Admitted stock inputs are closed")
        plan = validate_manifest(manifest)
        validate_top_k(plan["portfolio_policy"]["top_k"], plan["scope"]["execution_universe"])
        require(_input_key(plan) == self.__key, "Stock inputs differ from the complete admitted capability")
        return plan

    def __owner(self):
        require(os.getpid() == self.__owner_pid, "Admitted stock inputs belong to their creating process")

    @contextmanager
    def execution_scope(self):
        self.__owner()
        require(self.__lock.acquire(blocking=False), "Admitted stock inputs support sequential accounts only")
        try:
            require(not self.__closed, "Admitted stock inputs are closed")
            if self.__market is None:
                yield
            else:
                with self.__market._scope():
                    yield
        finally:
            self.__owner()
            self._prepared = None
            self._memory = None
            self.__lock.release()

    def __read(self, number, budget):
        self.__owner()
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
        if self.__market is not None:
            self.__market._assert_lease()
        integer(block_sessions, 1); budget = _budget(read_budget); limits = validate_limits(limits)
        require(block_sessions == self.__block_sessions, "Admitted stock block_sessions must remain fixed")
        inventory = json.loads(self.__inventory)
        require(inventory["input_bytes"] <= limits["max_input_bytes"] and inventory["folds"] <= limits["max_folds"],
                "Admitted stock inventory exceeds this account's limits")
        require(all(inventory["declared_rows"][name] <= limits["max_"+name]
                    for name in ("market_rows", "prediction_rows")), "Admitted stock declared row limit exceeded")
        self._memory = _DecodedMemory(budget["max_decoded_bytes"])
        # Immutable history/key/index volume belongs to each owner's lifetime,
        # not to a fresh account's temporary decoded budget.
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
        self._prepared = {'request_ref':plan['request_ref'],'signal_headers':meta.get('signal_headers',{})}
        self.__statistics["account_bindings"] += 1
        if self.__market is None:
            globals_value, lifecycle = meta["globals"], meta["lifecycle"]
        else:
            market_meta, market_stage = self.__market._globals(self._memory, budget)
            market_size = market_stage.used
            market_stage.close()
            self._memory.reserve_global("market_globals", market_size)
            globals_value = market_meta["globals"]
            globals_value["signal_limitations"] = meta["signal_limitations"]
            globals_value["market_header"]["limitations"] = receipt["limitations"]
            lifecycle = market_meta["lifecycle"]
        return StockSourceAudit(receipt=receipt, globals=globals_value, lifecycle=lifecycle, manifest=plan)

    def iter_blocks(self, manifest, *, block_sessions, read_budget):
        self.__owner()
        return self.__blocks(manifest, block_sessions=block_sessions, read_budget=read_budget)

    def __blocks(self, manifest, *, block_sessions, read_budget):
        plan = self.__check(manifest); budget = _budget(read_budget)
        if self.__market is not None:
            self.__market._assert_lease()
        require(block_sessions == self.__block_sessions and self._prepared is not None and
                self._prepared["request_ref"] == plan["request_ref"], "Stock owner needs this account's binding")
        require(self._memory.limit == budget["max_decoded_bytes"], "Stock owner decoded budget changed")
        calendar = plan["scope"]["calendar"]
        for number in range(1, len(self.__offsets)//2):
            value, stage = self.__read(number, budget)
            block = None
            market_stage = None
            try:
                days = tuple(value["sessions"])
                offset = (number-1)*block_sessions
                require(days == tuple(calendar[offset:offset+block_sessions]), "Private stock block calendar mismatch")
                if self.__market is not None:
                    market_value, market_stage = self.__market._market_block(number-1, self._memory, budget)
                    value["market_rows"] = market_value["market_rows"]
                    stage.reserve(64*(len(market_value["bindings"])+len(value["bindings"]))+512)
                    value["bindings"] = market_value["bindings"] + value["bindings"]
                count = len(value["market_rows"]) + sum(len(item["rows"]) for item in value["signals"])
                stage.reserve(64*count)
                rows = {(row["session"], row["security_id"]): row for row in value["market_rows"]}
                signals = {item["session"]: (item['header'] if 'header' in item else self._prepared['signal_headers'][item['signal_ref']],
                    {(row["session"], row["security_id"]): row for row in item["rows"]}) for item in value["signals"]}
                if market_stage is not None:
                    stage.used += market_stage.used
                    market_stage.used = 0
                block = StockInputBlock(days, rows, signals, tuple(value["bindings"]), stage.used, stage)
                value = rows = signals = None
                market_value = None
                yield block
            finally:
                self.__owner()
                value = rows = signals = block = market_value = None
                if market_stage is not None:
                    market_stage.close()
                stage.close()

    def close(self):
        self.__owner()
        require(self.__lock.acquire(blocking=False), "Cannot close stock inputs during account execution")
        try:
            if not self.__closed:
                if self.__market is not None:
                    with self.__market._scope():
                        self.__market._release()
                    if self.__owns_market:
                        self.__market_final_stats = self.__market.statistics
                self.__closed = True
                self.__store.close()
                self.__offsets = array("Q")
                self.__key = self.__inventory = b""
                self._memory = self._prepared = None
                if self.__owns_market:
                    self.__market.close()
        finally:
            self.__lock.release()

    def __enter__(self):
        self.__owner()
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
    require(not isinstance(source, AdmittedStockInputs), "Import original stock sources, not an admission marker")
    require(isinstance(source, StockInputSource), "StockInputSource required")
    from .stock_market_owner import StockMarketSpec, admit_stock_market_inputs, bind_stock_prediction_inputs
    integer(max_owned_bytes, 1)
    market = admit_stock_market_inputs(StockMarketSpec.from_request(manifest), source=source,
        block_sessions=block_sessions, limits=limits, max_market_bytes=max_owned_bytes)
    market_parse_events = source._file_parse_events
    market._combined_limit = max_owned_bytes
    try:
        remaining = max_owned_bytes-market.statistics["owned_bytes"]
        require(remaining > 0, "Owned stock inputs exceed max_owned_bytes before Signal binding")
        inputs = bind_stock_prediction_inputs(market, manifest, source=source,
            limits=limits, max_signal_bytes=remaining)
        source._file_parse_events = market_parse_events + source._file_parse_events
        require(market.statistics["owned_bytes"]+inputs.statistics["owned_bytes"] <= max_owned_bytes,
                "Owned stock inputs exceed max_owned_bytes after pairing proof")
        inputs._AdmittedStockInputs__owns_market = True
        return inputs
    except BaseException:
        if market.statistics["signal_borrows"]:
            inputs.close()
        market.close()
        raise
