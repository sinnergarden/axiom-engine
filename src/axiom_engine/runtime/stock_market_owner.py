"""Private market ownership and separate original saved-prediction admission.

No persisted trust marker, Data writer or second account executor is introduced.
"""
from array import array
from contextlib import contextmanager
import json
import os
import tempfile
from threading import Lock, get_ident
from time import perf_counter
from urllib.parse import urlparse

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import ContractError, Document, canonical, digest, fields, integer, require, session, text
from .backtest import BacktestRequest
from .stock_stream_contracts import (artifact_identity, identity_view, read_budget, validate_artifact_ref,
                                     validate_limits, validate_manifest, _validate_scope)
from .stock_stream_inputs import (StockInputSource, _DecodedMemory, _audit, _encoded_size,
    _frames, _grid, _native_header, _pair, _path, _warmup_visibility)
from .stock_stream_outputs import _json_pieces
from .stock_schedule import CLOCK_POLICY
from .stock_inputs import ACTION_POLICY

_TOKEN = object()
_PAIRS = (("market", ("open", "high", "low", "close", "volume_shares", "amount_cny")),
          ("factor", ("factor",)), ("membership", ("is_member",)))


class StockMarketSpec(Document):
    """Execution-only immutable spec; Signal/model/account fields are absent."""
    def __post_init__(self):
        super().__post_init__()
        value = self.to_dict()
        equity = value.get("stock_action_policy") == "registered_equity_v1"
        fields(value, "scope profile_input market_input stock_action_policy clock_policy" + (" action_facts_artifact" if equity else ""))
        if equity: validate_artifact_ref(value["action_facts_artifact"])
        scope = value["scope"]; _validate_scope(scope)
        require(scope["prediction_universe"] == scope["execution_universe"], "Full-stock prediction/execution union mismatch")
        require(value["stock_action_policy"] in (ACTION_POLICY, "registered_equity_v1") and value["clock_policy"] == CLOCK_POLICY,
                "stock action/clock policy mismatch")
        profile = value["profile_input"]
        fields(profile, "artifact profile_ref stock_execution_rules_ref stock_fee_schedule_ref")
        validate_artifact_ref(profile["artifact"])
        for name in ("profile_ref", "stock_execution_rules_ref", "stock_fee_schedule_ref"):
            digest(profile[name])
        market = value["market_input"]
        fields(market, "contract_version execution_snapshot_id warmup_sessions price_basis projection_version native_inputs")
        require(market["contract_version"] == "stock_market_input_refs_v1" and market["price_basis"] == "unadjusted" and
                market["projection_version"] == "market_replay_v4", "stock market reference contract mismatch")
        text(market["execution_snapshot_id"])
        warmup = market["warmup_sessions"]
        require(type(warmup) is list and bool(warmup) and warmup == sorted(set(warmup)), "Explicit preceding stock warmup required")
        for day in warmup:
            session(day); require(day < scope["calendar"][0], "Explicit preceding stock warmup required")
        require(type(market["native_inputs"]) is list and bool(market["native_inputs"]), "native input references required")
        for item in market["native_inputs"]:
            fields(item, "role artifact native_ref")
            require(item["role"] == "execution", "Market owner accepts only execution sources")
            validate_artifact_ref(item["artifact"]); digest(item["native_ref"])

    @classmethod
    def from_request(cls, request):
        require(isinstance(request, BacktestRequest), "BacktestRequest required")
        plan = validate_manifest(request)
        market = plan["market_input"]
        return cls.from_dict({**({"action_facts_artifact": plan["action_facts_artifact"]} if "action_facts_artifact" in plan else {}), "scope": plan["scope"], "profile_input": plan["profile_input"],
            "market_input": {**{name: market[name] for name in
                ("contract_version", "execution_snapshot_id", "warmup_sessions", "price_basis", "projection_version")},
                "native_inputs": [item for item in market["native_inputs"] if item["role"] == "execution"]},
            "stock_action_policy": plan["stock_action_policy"], "clock_policy": plan["clock_policy"]})


def _market_key(spec):
    return canonical(identity_view(spec.to_dict())).encode()


def _basis_key(plan):
    market = plan["market_input"]
    return canonical(identity_view({"market_input": {"model_snapshot_id": market["model_snapshot_id"],
        "native_inputs": [item for item in market["native_inputs"] if item["role"] == "prediction_basis"]}})).encode()


class _Store:
    """Quota covers private bytes plus retained key/index volume, before growth."""
    def __init__(self, maximum, fixed, label):
        integer(maximum, 1)
        require(fixed <= maximum, label+" exceeds quota before allocation")
        self.maximum, self.fixed, self.label = maximum, fixed, label
        self.offsets = array("Q")
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.bytes = 0
        self.write_seconds = 0

    @property
    def used(self):
        return self.fixed + len(self.offsets)*8 + self.bytes

    def charge(self, size):
        require(self.used+size <= self.maximum, self.label+" exceeds quota before index growth")
        self.fixed += size

    def write(self, value, memory):
        started = perf_counter()
        require(self.used+16 <= self.maximum, self.label+" exceeds quota before index growth")
        memory.replace_global(("private_offsets", id(self)), (len(self.offsets)+2)*8)
        start = self.file.seek(0, 2)
        with memory.stage() as scratch:
            scratch.reserve(2048)
            for piece in _json_pieces(value, piece_chars=64):
                require(self.fixed+len(self.offsets)*8+16+self.file.tell()+len(piece) <= self.maximum,
                        self.label+" exceeds quota before write")
                self.file.write(piece)
        self.offsets.extend((start, self.file.tell()-start))
        self.bytes = self.file.tell()
        self.write_seconds += perf_counter()-started
        return len(self.offsets)//2-1

    def write_metadata(self,reference,header,memory):
        """Retain a full original header once, outside account day offsets."""
        number=self.write({'contract_version':'stock_signal_metadata_v1','signal_ref':reference,'header':header},memory)
        start,length=self.offsets[2*number:2*number+2]
        del self.offsets[2*number:2*number+2]
        self.charge(16+len(reference))
        memory.replace_global(('private_offsets',id(self)),len(self.offsets)*8)
        return {'signal_ref':reference,'offset':start,'length':length}

    def read(self, number, memory, budget):
        start, length = self.offsets[2*number:2*number+2]
        stage = memory.stage()
        try:
            stage.reserve(3*length)
            self.file.seek(start); raw = bytearray(); remaining = length
            while remaining:
                chunk = self.file.read(min(remaining, budget["max_read_bytes"], 65536))
                require(bool(chunk), "Private stock input store truncated")
                raw.extend(chunk); remaining -= len(chunk)
            value = json.loads(raw); raw = chunk = None
            stage.release(2*length)
            return value, stage
        except BaseException:
            stage.close(); raise

    def close(self):
        self.file.close(); self.offsets = array("Q")


def _release_source(source):
    source._close_private_views()
    source._indexes = {}; source._prepared = None; source._audited = False
    source._memory = None; source._scan_native_scopes = {}; source._scan_row_limits = {}
    source._inventory_sizes = {}; source._prediction_paths = set()


class AdmittedStockMarket:
    """Process-local immutable market store with exclusive leases and borrows."""
    def __init__(self, token, store, key, calendar, block_sessions, native, blocks, meta, inventory, statistics):
        require(token is _TOKEN, "Use admit_stock_market_inputs")
        self._pid = os.getpid(); self._closed = False; self._lock = Lock(); self._borrows = 0
        self._active_thread = None
        self._combined_limit = None  # Only the legacy combined factory sets this.
        self._store, self._key = store, key
        self._calendar, self._block_sessions = tuple(calendar), block_sessions
        self._native, self._blocks, self._meta = native, blocks, meta
        self._inventory = inventory; self._proofs = {}; self._statistics = statistics

    def _check(self):
        require(os.getpid() == self._pid, "Admitted stock market belongs to its creating process")
        require(not self._closed, "Admitted stock market is closed")

    @contextmanager
    def _scope(self):
        self._check()
        require(self._lock.acquire(blocking=False), "Admitted stock market supports sequential binding/accounts only")
        try:
            self._check(); self._active_thread = get_ident(); yield
        finally:
            self._check(); self._active_thread = None; self._lock.release()

    def _assert_lease(self):
        self._check()
        require(self._active_thread == get_ident(), "Stock market reads require this thread's exclusive scope")

    @property
    def statistics(self):
        self._check()
        return {**json.loads(canonical(self._statistics)), "owned_bytes": self._store.used,
                "signal_borrows": self._borrows, "basis_proofs": len(self._proofs)}

    def _matches(self, request):
        self._check()
        require(_market_key(StockMarketSpec.from_request(request)) == self._key,
                "Execution market differs from the admitted market capability")

    def _read(self, number, memory, budget):
        self._assert_lease()
        value, stage = self._store.read(number, memory, budget)
        self._statistics["owned_read_bytes"] += self._store.offsets[2*number+1]
        self._statistics["owned_record_decodes"] += 1
        return value, stage

    def _globals(self, memory, budget):
        return self._read(self._meta, memory, budget)

    def _market_block(self, number, memory, budget):
        return self._read(self._blocks[number], memory, budget)

    def _native_key(self, kind, days):
        self._check()
        selected = next((key for key in self._native if key[0] == kind and set(days) <= set(key[1])), None)
        require(selected is not None, "Missing admitted original market view")
        return selected

    def _native_view(self, kind, days, memory, budget, reservation):
        days = tuple(days)
        selected = self._native_key(kind, days)
        value, stage = self._read(self._native[selected], memory, budget)
        try:
            view = value["view"]
            # Select by original session keys; retain the original full Query/context.
            wanted = set(days)
            stage.reserve(64*(len(view["records"])+sum(len(f["by_key"]) for f in view["field_meta"].values())))
            view["records"] = [row for row in view["records"] if row["session"] in wanted]
            for field in view["field_meta"].values():
                field["by_key"] = [row for row in field["by_key"] if row["session"] in wanted]
            require(reservation.memory is stage.memory, "Market view requires one decoded budget")
            reservation.used += stage.used
            stage.used = 0
            return view
        finally:
            stage.close()

    def _retain(self):
        self._check(); self._borrows += 1

    def _release(self):
        self._check(); require(self._borrows > 0, "Invalid stock market borrow release"); self._borrows -= 1

    def close(self):
        require(os.getpid() == self._pid, "Admitted stock market belongs to its creating process")
        require(self._lock.acquire(blocking=False), "Cannot close market during binding/account execution")
        try:
            require(self._borrows == 0, "Cannot close market with Signal borrows")
            if not self._closed:
                self._closed = True; self._store.close()
                self._native.clear(); self._blocks.clear(); self._proofs.clear()
                self._key = self._inventory = b""
        finally:
            self._lock.release()

    def __enter__(self):
        self._check(); return self

    def __exit__(self, *_):
        self.close()


def admit_stock_market_inputs(spec, *, source, block_sessions, limits, max_market_bytes):
    from .stock_rules import csi300_stock_portfolio_policy
    require(isinstance(spec, StockMarketSpec), "StockMarketSpec required")
    from .stock_owned_inputs import AdmittedStockInputs
    require(isinstance(source, StockInputSource) and not isinstance(source, AdmittedStockInputs), "Original StockInputSource required")
    integer(block_sessions, 1); integer(max_market_bytes, 1); limits = validate_limits(limits)
    require(len(spec.payload.encode()) <= limits["max_block_bytes"], "Market spec exceeds decoded budget")
    plan = spec.to_dict(); plan["request_ref"] = spec.identity
    inventory = source.inventory(plan); inventory_wire = canonical(inventory).encode(); key = _market_key(spec)
    store = _Store(max_market_bytes, len(key)+len(inventory_wire), "max_market_bytes")
    native, blocks = {}, []
    started = perf_counter()
    def capture(kind, selected, value, stage):
        # Retained map/array wrappers are charged before insertion, independently
        # of the source's transient decoder and compact source index.
        cost = _encoded_size(selected)+64
        store.charge(cost)
        source._memory.reserve_global(("private_map", kind, selected), cost)
        with source._memory.stage() as wrappers:
            count = len(value.market_rows)+len(value.bindings) if kind == "block" else len(value["bindings"])
            wrappers.reserve(64*count+512)
            if kind == "block":
                value = {"sessions": list(selected), "market_rows": list(value.market_rows.values()),
                         "bindings": list(value.bindings)}
            number = store.write(value, source._memory)
        if kind == "native":
            native[selected] = number
        else:
            blocks.append(number)
    try:
        with source.execution_scope():
            audit = _audit(source, plan, block_sessions, read_budget(limits), limits, IMPLEMENTATION_REF,
                           market_only=True, capture=capture)
            descriptors = []
            for item in plan["market_input"]["native_inputs"]:
                index = source._index(item["artifact"], read_budget(limits))
                kind, days, context = _native_header(index, item, plan)
                descriptors.append({"entry": item, "kind": kind, "days": days, "context": context})
            for index in source._indexes.values():
                index.unchanged()
            policy = csi300_stock_portfolio_policy(top_k=1, execution_universe=plan["scope"]["execution_universe"],
                                                  execution_rules=audit.globals["profile"]["stock_execution_rules"])
            meta = store.write({"globals": audit.globals, "lifecycle": audit.lifecycle, "policy": policy,
                "counts": audit.receipt["counts"], "descriptors": descriptors,
                "implementation_ref": IMPLEMENTATION_REF}, source._memory)
            stats = {"source_admissions": 1, "source_audit_seconds": perf_counter()-started-store.write_seconds,
                "capture_seconds": store.write_seconds,
                "source_operations": source.statistics["source_operations"],
                "source_decoded_bytes_peak": source.statistics["decoded_bytes_peak"],
                "source_scalar_cache_evictions": source.statistics["scalar_cache_evictions"],
                "owned_read_bytes": 0, "owned_record_decodes": 0,
                "basis_pairings": 0, "basis_reuses": 0, "signal_bindings": 0}
            return AdmittedStockMarket(_TOKEN, store, key, plan["scope"]["calendar"], block_sessions,
                                       native, blocks, meta, inventory_wire, stats)
    except (KeyError, IndexError, TypeError, OSError) as exc:
        store.close(); raise ContractError("Malformed or unreadable fixed stock market: "+str(exc)) from exc
    except BaseException:
        store.close(); raise
    finally:
        _release_source(source)


def _binding_inventory(market, plan, source, proof):
    """Stat only new original Signal/basis paths; frozen market sizes are reused."""
    inventory = json.loads(market._inventory)
    known = {canonical(artifact_identity(f["artifact"])): f["file_bytes"] for f in inventory["files"] if "artifact" in f}
    if proof is not None:
        known.update(proof["sizes"])
    native = {canonical(artifact_identity(i["artifact"])): i for i in inventory.get("native_artifacts", [])}
    if proof is not None:
        native.update({canonical(artifact_identity(i["artifact"])): i for i in proof.get("native_artifacts", [])})
    files = {f["manifest_uri"]: f for f in inventory["files"] if "native_view_ref" in f}
    if proof is not None:
        files.update({f["manifest_uri"]: f for f in proof.get("native_files", [])})
    for artifact in source._artifacts(plan):
        logical = canonical(artifact_identity(artifact))
        if logical in native:
            continue
        delivery = getattr(source, "native_inventory", lambda _: None)(artifact)
        if delivery is not None:
            physical, binding = delivery
            native[logical] = binding
            for item in physical:
                files.setdefault(item["manifest_uri"], item)
            continue
        if hasattr(source, "native_inventory"):
            require(artifact not in [i["artifact"] for i in plan["market_input"]["native_inputs"]], "Native handle missing logical ref")
        path = _path(artifact, must_exist=logical not in known)
        size = known[logical] if logical in known else path.stat().st_size
        key = str(path)
        require(key not in files or files[key]["artifact"]["content_digest"] == artifact["content_digest"],
                "Conflicting stock artifact references to one file")
        files[key] = {"artifact": artifact, "file_bytes": size}
        if urlparse(artifact["manifest_uri"]).fragment:
            descriptor = path.with_name("manifest.json")
            files.setdefault(str(descriptor), {"manifest_uri": str(descriptor), "file_bytes": descriptor.stat().st_size})
    from .stock_stream_contracts import prediction_inventory
    folds,prediction_rows=prediction_inventory(plan['prediction_input'],plan['scope'])
    value = {"files": list(files.values()), "input_bytes": sum(f["file_bytes"] for f in files.values()),
        "folds": folds, "declared_rows": {
            "market_rows": len(plan["scope"]["calendar"])*len(plan["scope"]["execution_universe"]),
            "prediction_rows": prediction_rows},
        "declared_scope": plan["scope"]}
    if native:
        value["native_artifacts"] = list(native.values())
    return value


def bind_stock_prediction_inputs(market, request, *, source, limits, max_signal_bytes):
    require(isinstance(market, AdmittedStockMarket), "AdmittedStockMarket required")
    market._matches(request)
    from .stock_owned_inputs import AdmittedStockInputs
    require(isinstance(source, StockInputSource) and not isinstance(source, AdmittedStockInputs), "Original StockInputSource required")
    integer(max_signal_bytes, 1); limits = validate_limits(limits)
    plan = validate_manifest(request)
    require(len(request.payload.encode()) <= limits["max_block_bytes"], "stock request manifest exceeds decoded budget")
    with market._scope(), source.execution_scope():
        try:
            _release_source(source)
            source._file_parse_events = []; source._cjson_spool_bytes = 0
            return _bind(market, plan, source, limits, max_signal_bytes)
        except (KeyError, IndexError, TypeError, OSError) as exc:
            raise ContractError("Malformed or unreadable fixed stock prediction: "+str(exc)) from exc
        finally:
            _release_source(source)


def _bind(market, plan, source, limits, maximum):
    from ..core.stock_portfolio import StockPredictionFrame, validate_stock_predictions, validate_top_k, instant
    from .stock_owned_inputs import AdmittedStockInputs, _IMPORT, _input_key
    started = perf_counter()
    budget = read_budget(limits); scope = plan["scope"]; calendar = scope["calendar"]; universe = scope["execution_universe"]
    key = _basis_key(plan); proof = market._proofs.get(key)
    inventory = _binding_inventory(market, plan, source, proof)
    require(inventory["input_bytes"] <= limits["max_input_bytes"] and inventory["folds"] <= limits["max_folds"],
            "Stock source inventory exceeds input/fold budget")
    from .stock_stream_contracts import check_inventory_rows,resolve_prediction_inventory
    check_inventory_rows(inventory,limits)
    source._memory = _DecodedMemory(budget["max_decoded_bytes"], scalar_cache_bytes=source._scalar_cache_bytes)
    source._memory.reserve_global("request", _encoded_size(plan))
    source._inventory_sizes = {str(_path(f["artifact"], must_exist=False)) if "artifact" in f else f["manifest_uri"]: f["file_bytes"] for f in inventory["files"]}
    source._scan_row_limits = {}; source._scan_native_scopes = {}; source._prediction_paths = set()
    source._prediction_remaining = limits["max_prediction_rows"]
    from .stock_stream_contracts import score_artifacts
    for artifact in score_artifacts(plan['prediction_input']):
        path=str(_path(artifact))
        source._scan_row_limits[path]=limits['max_prediction_rows'];source._prediction_paths.add(path)
    meta, meta_stage = market._globals(source._memory, budget)
    store = None
    member_key = member_stage = member_view = members = member_metadata = None
    def membership_for(feature):
        nonlocal member_key, member_stage, member_view, members, member_metadata
        selected = market._native_key("membership", [feature])
        if selected != member_key:
            # Retire every borrower before releasing the previous block's charge.
            member_view = members = member_metadata = None
            if member_stage is not None:
                member_stage.close()
            member_stage = source._memory.stage()
            member_view = market._native_view("membership", selected[1], source._memory, budget, member_stage)
            member_stage.reserve(64*(len(member_view["records"])+sum(len(f["by_key"]) for f in member_view["field_meta"].values())))
            members, member_metadata = _grid(member_view, selected[1], universe)
            member_key = selected
    try:
        require(meta["implementation_ref"] == IMPLEMENTATION_REF, "Admitted stock implementation differs")
        validate_top_k(plan["portfolio_policy"]["top_k"], universe)
        require(plan["portfolio_policy"] == {**meta["policy"], "top_k": plan["portfolio_policy"]["top_k"]},
                "Stock portfolio policy mismatch")
        native_by_day, contexts = {}, []
        execution = {canonical(artifact_identity(d["entry"]["artifact"])): d for d in meta["descriptors"]}
        if proof is None:
            for entry in plan["market_input"]["native_inputs"]:
                if entry["role"] != "prediction_basis":
                    continue
                known = execution.get(canonical(artifact_identity(entry["artifact"])))
                if known is not None:
                    require(entry["native_ref"] == known["entry"]["native_ref"] and
                            plan["market_input"]["model_snapshot_id"] == known["context"]["snapshot_id"] and
                            known["kind"] not in ("actions-ex", "actions-record"), "Stock native Snapshot/Reader binding mismatch")
                    kind, days, context = known["kind"], known["days"], known["context"]
                    index = None
                else:
                    path = str(source._location(entry["artifact"]))
                    source._scan_row_limits[path] = limits["max_market_rows"]; source._scan_native_scopes[path] = (plan, limits)
                    index = source._index(entry["artifact"], budget)
                    kind, days, context = _native_header(index, entry, plan)
                contexts.append(context)
                for day in days:
                    require((kind, day) not in native_by_day, "Duplicate/overlapping stock native query scope")
                    native_by_day[kind, day] = index
            for days in [tuple(calendar[o:o+market._block_sessions]) for o in range(0, len(calendar), market._block_sessions)] + [(d,) for d in plan["market_input"]["warmup_sessions"]]:
                warmup = days[0] not in calendar
                for kind, names in (_PAIRS[:2] if warmup else _PAIRS):
                    require(all((kind, d) in native_by_day for d in days), "Missing fixed stock prediction basis")
                    selected = {native_by_day[kind, d] for d in days}
                    require(len(selected) == 1, "Stock block crosses native parent scope; reduce block_sessions")
                    index = next(iter(selected))
                    with source._memory.stage() as stage:
                        right = market._native_view(kind, days, source._memory, budget, stage)
                        if index is None:
                            left = right
                        else:
                            left, _, refs = index.batch_view(days, budget, reservation=stage)
                        _pair(left, right, days, universe, names)
                        if warmup:
                            _warmup_visibility(left, days, universe, names)
                        left = right = refs = None
            sizes = {canonical(artifact_identity(f["artifact"])): f["file_bytes"] for f in inventory["files"]
                     if "artifact" in f and f["artifact"] in [i["artifact"] for i in plan["market_input"]["native_inputs"] if i["role"] == "prediction_basis"]}
            proof = {"contexts": contexts, "sizes": sizes}
            selected_artifacts = {canonical(artifact_identity(i["artifact"])) for i in plan["market_input"]["native_inputs"] if i["role"] == "prediction_basis"}
            selected_native = [i for i in inventory.get("native_artifacts", []) if canonical(artifact_identity(i["artifact"])) in selected_artifacts]
            if selected_native:
                proof["native_artifacts"] = selected_native
                refs = {i["native_view_ref"] for i in selected_native}
                proof["native_files"] = [f for f in inventory["files"] if f.get("native_view_ref") in refs]
        frames, trade_map = _frames(source, plan, budget)
        resolve_prediction_inventory(inventory,source,limits)
        modern=plan['prediction_input']['contract_version']=='stock_prediction_input_refs_v2'
        binding_key = _input_key(plan); inventory_wire = canonical(inventory).encode()
        proof_cost = 0 if key in market._proofs else len(key)+_encoded_size(proof)+64
        require(market._store.used+proof_cost <= market._store.maximum,
                "max_market_bytes exceeds quota before pairing proof retention")
        if market._combined_limit is not None:
            maximum = min(maximum, market._combined_limit-market._store.used-proof_cost)
            require(maximum > 0, "Owned stock inputs exceed max_owned_bytes before Signal capture")
        store = _Store(maximum, len(binding_key)+len(inventory_wire), "max_signal_bytes")
        # Record zero is completed last; reserve its offset slot now.
        require(store.used+16 <= store.maximum, "max_signal_bytes exceeds quota before index growth")
        source._memory.reserve_global(("private_offsets", id(store)), 16)
        store.offsets.extend((0, 0))
        headers={};metadata_locations=[]
        if modern:
            for frame in frames:
                reference=frame['header']['signal_run_ref'];headers[reference]=frame['header']
                if frame.get('kind')=='derived':
                    metadata_locations.append(store.write_metadata(reference,frame['original_header'],source._memory))
        prediction_rows = 0
        for offset in range(0, len(calendar), market._block_sessions):
            days = tuple(calendar[offset:offset+market._block_sessions]); signals, bindings, parents = [], [], set()
            with source._memory.stage() as stage:
                for trade in days:
                    if trade not in trade_map:
                        continue
                    frame, feature = trade_map[trade]
                    if frame.get('kind')=='derived':
                        from .stock_signal_inputs import read_derived
                        rows,refs=read_derived(source,frame,feature,budget,stage)
                    else:
                        frame["spec_index"].unchanged(); frame["model_index"].unchanged()
                        parent = frame["parent_binding"]
                        if parent is not None and parent["parent_ref"] not in parents:
                            parents.add(parent["parent_ref"])
                            require(frame["spec_index"].span_digest(frame["spec_index"].selector) == parent["child_ref"],
                                    "Original saved fold child changed after admission")
                            bindings.append(parent)
                        rows, used, refs = frame["index"].rows(("rows",), [feature], budget, reservation=stage)
                    bindings += refs
                    with source._memory.stage() as validation:
                        validation.reserve(3*(_encoded_size(frame["header"])+sum(_encoded_size(r) for r in rows)+32))
                        from .stock_signal_inputs import check_day_rows
                        checked=check_day_rows(frame,feature,rows)
                        require(len(checked) == len(universe), "Incomplete original saved prediction date group")
                        membership_for(feature)
                        from .stock_signal_inputs import validate_day
                        validate_day(frame,feature,checked,members,member_metadata,
                            next(d['entry']['native_ref'] for d in meta['descriptors'] if d['kind']=='membership' and feature in d['days']))
                        prediction_rows += len(checked)
                        require(prediction_rows <= limits["max_prediction_rows"], "Stock prediction row budget exceeded")
                        wire = checked = row = None
                    stage.reserve(64*len(rows)+512)
                    signals.append({'session':trade,'rows':rows,**({'signal_ref':frame['header']['signal_run_ref']} if modern else {'header':frame['header']})})
                    wire = checked = row = refs = None
                stage.reserve(512)
                store.write({"sessions": list(days), "signals": signals, "bindings": bindings}, source._memory)
                signals = bindings = rows = None
        contexts = [d["context"] for d in meta["descriptors"]] + proof["contexts"]
        limitations = sorted({x for c in contexts for x in c.get("limitations", [])}) + meta["globals"]["market_header"]["limitations"][-3:]
        counts = {**meta["counts"], "input_bytes": inventory["input_bytes"], "input_files": len(inventory["files"]),
                  "folds": inventory['folds'], "prediction_rows":
                      limits['max_prediction_rows']-source._prediction_remaining if plan['prediction_input']['contract_version']=='stock_prediction_input_refs_v2' else prediction_rows}
        receipt = {"contract_version": "stock_input_audit_v1", "request_ref": plan["request_ref"],
            "market_ref": plan["market_input"]["market_ref"], "prediction_ref": plan["prediction_input"]["prediction_ref"],
            "profile_ref": plan["profile_input"]["profile_ref"], "implementation_ref": IMPLEMENTATION_REF,
            "counts": counts, "limitations": limitations}
        if modern:
            from .stock_signal_inputs import prediction_targets
            receipt.update(contract_version='stock_input_audit_v2',prediction_targets=prediction_targets(frames))
        number = store.write({"receipt": receipt, "signal_limitations": sorted({v for f in frames for v in f["header"]["limitations"]}),
            **({'signal_headers':headers,'signal_metadata_locations':metadata_locations} if modern else {})}, source._memory)
        # Place metadata at logical slot zero without moving private bytes.
        store.offsets[:2] = store.offsets[2*number:2*number+2]; del store.offsets[2*number:2*number+2]
        for index in source._indexes.values():
            index.unchanged()
        stats = {"source_admissions": 1, "source_audit_seconds": perf_counter()-started-store.write_seconds,
            "capture_seconds": store.write_seconds,
            "owned_bytes": store.used, "max_owned_decoded_bytes": source._memory.peak_bytes,
            "owned_blocks": len(store.offsets)//2-1, "owned_read_bytes": 0, "owned_record_decodes": 0,
            "account_bindings": 0, "source_operations": source.statistics["source_operations"],
            "source_decoded_bytes_peak": source.statistics["decoded_bytes_peak"],
            "source_scalar_cache_evictions": source.statistics["scalar_cache_evictions"]}
        if modern:
            stats.update(signal_metadata_records=len(metadata_locations),signal_metadata_bytes=sum(i['length'] for i in metadata_locations))
        result = AdmittedStockInputs(_IMPORT, store.file, store.offsets, binding_key, market._block_sessions,
                                     inventory_wire, stats, market=market)
        if key not in market._proofs:
            old_fixed = market._store.fixed
            try:
                market._store.charge(proof_cost)
                market._proofs[key] = proof
            except BaseException:
                market._store.fixed = old_fixed
                raise
            market._statistics["basis_pairings"] += 1
        else:
            market._statistics["basis_reuses"] += 1
        market._retain(); market._statistics["signal_bindings"] += 1
        store = None
        return result
    finally:
        member_view = members = member_metadata = None
        if member_stage is not None:
            member_stage.close()
        meta_stage.close()
        if store is not None:
            store.close()
