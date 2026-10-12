"""Standard immutable artifacts backing the existing stock input owners.

Only completed public market/prediction admission can save a snapshot. Loading
checks its entire local content and required scope before exposing an owner;
original Data/Research ancestors are not reopened or re-signed.
"""
from array import array
from bisect import bisect_right
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import ContractError, Document, canonical, digest, fields, integer, require
from .stock_input_store import InputStore, INPUT_SEMANTICS, measured_size
from .stock_stream_contracts import validate_artifact_ref, validate_limits, read_budget
from .stock_stream_inputs import _DecodedMemory, _path

MARKET_VERSION = "stock_prepared_market_v1"
PREDICTION_VERSION = "stock_prepared_predictions_v1"


def _checked_load(function):
    """Malformed local snapshots fail through the public contract boundary."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ContractError:
            raise
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as error:
            raise ContractError("Malformed prepared input snapshot: " + type(error).__name__) from error
    return checked


def _mark(path):
    s = Path(path).stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


class _PartitionFile:
    """A read-only logical file; one bounded physical part is open at a time."""
    def __init__(self, root, parts, marks):
        self.root,self.parts,self.marks = root,parts,marks
        self.starts = []; total = 0
        for part in parts: self.starts.append(total); total += part["bytes"]
        self.size = total; self.position = 0; self.closed = False
        self.current = None; self.stream = None

    def seek(self, offset, whence=0):
        require(not self.closed, "Prepared input store is closed")
        value = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        require(0 <= value <= self.size, "Prepared input seek outside snapshot")
        self.position = value; return value

    def tell(self): return self.position

    def read(self, size):
        require(not self.closed and type(size) is int and size >= 0, "Invalid prepared input read")
        output = bytearray(); remaining = min(size,self.size-self.position)
        while remaining:
            i = bisect_right(self.starts,self.position)-1; part = self.parts[i]
            path = self.root/part["uri"]
            require(_mark(path) == self.marks[part["uri"]], "Prepared input part changed after loading")
            if self.current != i:
                if self.stream is not None: self.stream.close()
                self.stream = path.open("rb"); self.current = i
            self.stream.seek(self.position-self.starts[i])
            count = min(remaining,part["bytes"]-(self.position-self.starts[i]))
            chunk = self.stream.read(count)
            require(len(chunk) == count and _mark(path) == self.marks[part["uri"]],
                    "Prepared input part changed during read")
            output.extend(chunk); remaining -= count; self.position += count
        return output

    def close(self):
        if self.stream is not None: self.stream.close(); self.stream = None
        self.closed = True


class _LoadedStore(InputStore):
    def write(self, *_): raise ContractError("Prepared input snapshots are immutable")
    def charge(self, size):
        # Invocation-local retained pairing evidence may grow, never disk bytes.
        super().charge(size)


def _save(store, destination, metadata, limits):
    limits = validate_limits(limits); destination = Path(destination).absolute()
    require(not destination.exists(), "Prepared input destination already exists")
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stock-input-",dir=destination.parent))
    try:
        (stage/"parts").mkdir(); store.file.seek(0)
        if hasattr(store.file,"flush"): store.file.flush()
        parts = []; remaining = store.bytes; number = 0
        part_limit = min(limits["max_result_part_bytes"],limits["max_block_bytes"])
        while remaining:
            uri = f"parts/{number:06d}.bin"; length = min(remaining,part_limit)
            h = hashlib.sha256(); pending = length
            with (stage/uri).open("xb") as target:
                while pending:
                    raw = store.file.read(min(pending,limits["max_read_bytes"],65536))
                    require(bool(raw), "Prepared input store truncated while saving")
                    target.write(raw); h.update(raw); pending -= len(raw)
                target.flush(); os.fsync(target.fileno())
            parts.append(dict(uri=uri,bytes=length,sha256="sha256:"+h.hexdigest()))
            remaining -= length; number += 1
        manifest = {**metadata,"input_semantics":INPUT_SEMANTICS,"producer_implementation_ref":IMPLEMENTATION_REF,
            "store":dict(bytes=store.bytes,fixed_bytes=store.fixed,records=[list(x) for x in store.records],
                         offsets=list(store.offsets),parts=parts)}
        raw = canonical(manifest).encode()
        require(len(raw) <= limits["max_block_bytes"] and len(raw)+store.bytes <= limits["max_input_bytes"],
                "Prepared input artifact exceeds manifest/input budget")
        with (stage/"manifest.json").open("xb") as target:
            target.write(raw); target.flush(); os.fsync(target.fileno())
        # Only the fully written snapshot becomes discoverable. Never replace
        # an old immutable artifact or publish a partial validation receipt.
        require(not destination.exists(), "Prepared input destination appeared during save")
        stage.rename(destination)
        identity = Document.from_dict({"contract_version":metadata["contract_version"],
            "input_semantics":INPUT_SEMANTICS,"input_key":metadata["input_key"]}).identity
        return dict(artifact_type="StockMarketInputs" if metadata["contract_version"] == MARKET_VERSION else "StockPredictionInputs",
            artifact_id=identity,contract_version=metadata["contract_version"],
            manifest_uri=str(destination/"manifest.json"),content_digest="sha256:"+hashlib.sha256(raw).hexdigest())
    finally:
        if stage.exists(): shutil.rmtree(stage)


def _load(ref, *, version, limits, maximum):
    started = perf_counter()
    validate_artifact_ref(ref); limits = validate_limits(limits); integer(maximum,1)
    require(ref["contract_version"] == version and ref["artifact_type"] ==
            ("StockMarketInputs" if version == MARKET_VERSION else "StockPredictionInputs"), "Prepared input artifact type mismatch")
    path = _path(ref); root = path.parent
    require(path.name == "manifest.json" and not path.is_symlink() and not root.is_symlink(), "Prepared input manifest location mismatch")
    mark = _mark(path); require(0 < mark[2] <= limits["max_block_bytes"], "Prepared input manifest exceeds decoded budget")
    raw = bytearray()
    with path.open("rb") as source:
        while len(raw) <= mark[2]:
            chunk = source.read(min(limits["max_read_bytes"],65536,mark[2]+1-len(raw)))
            if not chunk: break
            raw.extend(chunk)
    require(len(raw) == mark[2] and _mark(path) == mark and
            "sha256:"+hashlib.sha256(raw).hexdigest() == ref["content_digest"], "Prepared input manifest content mismatch")
    manifest = json.loads(raw)
    require(canonical(manifest).encode() == raw, "Prepared input manifest must be canonical JSON")
    raw = None
    fields(manifest,"contract_version input_semantics producer_implementation_ref input_key block_sessions calendar inventory store owner")
    require(manifest["contract_version"] == version and manifest["input_semantics"] == INPUT_SEMANTICS,
            "Prepared input semantic version mismatch")
    digest(manifest["producer_implementation_ref"])
    require(Document.from_dict({"contract_version":version,"input_semantics":INPUT_SEMANTICS,
            "input_key":manifest["input_key"]}).identity == ref["artifact_id"], "Prepared input identity mismatch")
    integer(manifest["block_sessions"],1)
    state = manifest["store"]; fields(state,"bytes fixed_bytes records offsets parts")
    integer(state["bytes"],1); integer(state["fixed_bytes"])
    require(state["bytes"]+mark[2] <= limits["max_input_bytes"] and
            state["bytes"]+state["fixed_bytes"]+8*len(state["offsets"]) <= maximum,
            "Prepared input snapshot exceeds owned/input budget")
    parts, marks = state["parts"], {}; total = 0
    require(type(parts) is list and bool(parts), "Prepared input part inventory required")
    for i,part in enumerate(parts):
        fields(part,"uri bytes sha256"); integer(part["bytes"],1); digest(part["sha256"])
        require(part["uri"] == f"parts/{i:06d}.bin" and part["bytes"] <= min(limits["max_result_part_bytes"],limits["max_block_bytes"]),
                "Prepared input part ordering/budget mismatch")
        target = root/part["uri"]; require(not target.is_symlink(), "Prepared input part symlink refused")
        observed = _mark(target); require(observed[2] == part["bytes"], "Prepared input part size mismatch")
        h = hashlib.sha256()
        with target.open("rb") as source:
            while chunk := source.read(min(limits["max_read_bytes"],65536)): h.update(chunk)
        require(_mark(target) == observed and "sha256:"+h.hexdigest() == part["sha256"], "Prepared input part digest mismatch")
        marks[part["uri"]] = observed; total += part["bytes"]
    require(total == state["bytes"] and _mark(path) == mark, "Prepared input inventory changed during loading")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == {"manifest.json",*(p["uri"] for p in parts)}, "Prepared input file closure mismatch")
    records, spans, position = state["records"], {}, 0
    require(type(records) is list and bool(records), "Prepared input record index required")
    for record in records:
        require(type(record) is list and len(record) == 3, "Prepared input record descriptor mismatch")
        start,length,expanded = record
        for value in record: integer(value)
        require(start == position and length > 0 and expanded > 0 and
                length <= limits["max_block_bytes"] and expanded <= limits["max_block_bytes"],
                "Prepared input record scope/budget mismatch")
        spans[start] = length,expanded; position += length
    require(position == total and type(state["offsets"]) is list and len(state["offsets"]) % 2 == 0,
            "Prepared input record coverage mismatch")
    for start,length in zip(state["offsets"][::2],state["offsets"][1::2]):
        require(type(start) is int and type(length) is int and start in spans and spans[start][0] == length,
                "Prepared input offset outside record index")
    store = _LoadedStore.__new__(_LoadedStore)
    store.maximum,store.fixed,store.label = maximum,state["fixed_bytes"],"Prepared input quota"
    store.file = _PartitionFile(root,parts,marks); store.offsets = array("Q",state["offsets"])
    store.bytes = total; store.write_seconds = 0; store.records = [tuple(r) for r in records]
    store.expanded = {start:expanded for start,_,expanded in records}
    store.lengths = {start:length for start,length,_ in records}
    try:
        # Parse each bounded record once and validate the expansion schema
        # without allocating rows from an untrusted expanded-size claim.
        for start,length,expanded in records:
            require(3*length <= limits["max_block_bytes"], "Prepared input decoder exceeds budget")
            store.file.seek(start); encoded = bytearray(); remaining = length
            while remaining:
                chunk = store.file.read(min(remaining,limits["max_read_bytes"],65536))
                require(bool(chunk), "Prepared input record truncated")
                encoded.extend(chunk); remaining -= len(chunk)
            value = json.loads(encoded)
            require(canonical(value).encode() == encoded, "Prepared input record must be canonical JSON")
            encoded = chunk = None
            require(measured_size(value,limits["max_block_bytes"]) == expanded,
                    "Prepared input decoded-size descriptor mismatch")
            value = None
        store.prepared_statistics = dict(prepared_content_bytes=total+mark[2],prepared_record_validations=len(records),
            prepared_codec_bytes_peak=max(3*r[1] for r in records),prepared_content_seconds=perf_counter()-started)
        return manifest,store
    except BaseException: store.close(); raise


def save_stock_market_inputs(market, destination, *, limits):
    from .stock_market_owner import AdmittedStockMarket
    require(isinstance(market,AdmittedStockMarket), "Completed AdmittedStockMarket required")
    with market._scope():
        metadata = dict(contract_version=MARKET_VERSION,input_key=json.loads(market._key),
            block_sessions=market._block_sessions,calendar=list(market._calendar),inventory=json.loads(market._inventory),
            owner=dict(native=[[kind,list(days),number] for (kind,days),number in market._native.items()],
                       blocks=market._blocks,meta=market._meta))
        return _save(market._store,destination,metadata,limits)


@_checked_load
def load_stock_market_inputs(ref, spec, *, limits, max_market_bytes):
    from .stock_market_owner import AdmittedStockMarket, StockMarketSpec, _TOKEN, _market_key
    require(isinstance(spec,StockMarketSpec), "StockMarketSpec required")
    manifest,store = _load(ref,version=MARKET_VERSION,limits=limits,maximum=max_market_bytes)
    try:
        key = canonical(manifest["input_key"]).encode()
        require(key == _market_key(spec) and manifest["calendar"] == spec.to_dict()["scope"]["calendar"],
                "Prepared market scope/dependency mismatch")
        owner = manifest["owner"]; fields(owner,"native blocks meta")
        native = {(kind,tuple(days)):number for kind,days,number in owner["native"]}
        require(len(native) == len(owner["native"]), "Duplicate prepared native window")
        for number in [owner["meta"],*owner["blocks"],*native.values()]:
            require(type(number) is int and 0 <= number < len(store.offsets)//2, "Prepared market locator outside store")
        stats = dict(source_admissions=0,source_audit_seconds=0,capture_seconds=0,source_operations={},
            source_decoded_bytes_peak=0,source_scalar_cache_evictions=0,owned_read_bytes=0,owned_record_decodes=0,
            basis_pairings=0,basis_reuses=0,signal_bindings=0,prepared_loads=1)
        stats.update(store.prepared_statistics)
        result = AdmittedStockMarket(_TOKEN,store,key,manifest["calendar"],manifest["block_sessions"],native,
            owner["blocks"],owner["meta"],canonical(manifest["inventory"]).encode(),stats)
        # A process-local pairing capability is never portable. A new Signal
        # bind must pair its basis again (or establish exact immutable-source
        # equality); a reopened prediction uses its checked saved projection.
        _validate_market_snapshot(result,spec,limits)
        return result
    except BaseException: store.close(); raise


def _validate_market_snapshot(market, spec, limits):
    from .stock_stream_contracts import artifact_identity
    from .stock_stream_inputs import _grid, _native_header
    from .stock_inputs import _validate_stock_market_row
    from .stock_market import _stock_lifecycle_active, _membership_rows, _visible
    from .stock_rules import validate_fee_schedule
    from ..core.stock_rules import validate_execution_rules
    from ..core.portfolio import decimal
    wire = spec.to_dict(); scope = wire["scope"]; universe = scope["execution_universe"]
    calendar = scope["calendar"]; memory = _DecodedMemory(limits["max_block_bytes"]); budget = read_budget(limits)
    universe_set = set(universe); calendar_set = set(calendar)
    admitted_days = calendar_set | set(wire["market_input"]["warmup_sessions"])
    with market._scope():
        meta,stage = market._globals(memory,budget)
        try:
            require(meta["input_semantics"] == INPUT_SEMANTICS, "Prepared market semantic metadata mismatch")
            globals_value = meta["globals"]; profile = globals_value["profile"]; rules = profile["stock_execution_rules"]
            require(Document.from_dict(profile).identity == wire["profile_input"]["profile_ref"] and
                    Document.from_dict(rules).identity == wire["profile_input"]["stock_execution_rules_ref"] and
                    Document.from_dict(profile["stock_fee_schedule"]).identity == wire["profile_input"]["stock_fee_schedule_ref"] and
                    rules["calendar"] == calendar and rules["universe"] == universe, "Prepared profile/rules identity mismatch")
            validated_rules = validate_execution_rules(rules)
            identities,_ = validated_rules; validate_fee_schedule(profile["stock_fee_schedule"])
            require(globals_value["rules_index"] == list(validated_rules), "Prepared rule index differs from original rules")
            from .profiles import stock_daily_open_profile_v2
            from .stock_equity import validate_equity_profile, validate_facts
            if wire["stock_action_policy"] == "registered_equity_v1":
                validate_equity_profile(profile)
            else:
                require(profile == stock_daily_open_profile_v2(execution_rules=rules,
                    fee_schedule=profile["stock_fee_schedule"],unknown_status_policy=profile["unknown_status_policy"]),
                    "Prepared execution profile differs")
            refs = globals_value["market_header"]["source_refs"]
            if wire["stock_action_policy"] == "registered_equity_v1":
                facts = globals_value["market_header"]["equity_facts"]
                validate_facts(facts,universe=universe,calendar=calendar,parent_refs=refs[4:6])
                require(Document.from_dict(facts).identity == wire["action_facts_artifact"]["content_digest"],
                        "Prepared equity facts differ from their original ref")
            global_bytes = stage.used; stage.close(); memory.reserve_global("prepared_globals",global_bytes)
            chunks = [tuple(calendar[o:o+market._block_sessions]) for o in range(0,len(calendar),market._block_sessions)]
            require(len(market._blocks) == len(chunks), "Prepared market calendar coverage mismatch")
            ended = set(); coverage = set(); descriptors = meta["descriptors"]
            previous_factor = {}; factor_blocks = []
            memory.reserve_global("prepared_factor_state",64*len(universe))
            expected_descriptors = wire["market_input"]["native_inputs"]
            def logical_entry(entry):
                fields(entry,"role native_ref artifact")
                return dict(role=entry["role"],native_ref=entry["native_ref"],artifact=artifact_identity(entry["artifact"]))
            require([logical_entry(d["entry"]) for d in descriptors] == [logical_entry(d) for d in expected_descriptors],
                    "Prepared original source descriptors differ")
            lifecycle_owners = {}; contexts = {}
            from .stock_input_store import encoded_size
            memory.reserve_global("prepared_descriptor_indexes",512*len(descriptors)+
                128*sum(len(d["days"]) for d in descriptors)+sum(encoded_size(d["context"]) for d in descriptors))
            for descriptor in descriptors:
                with memory.stage() as encoding:
                    encoding.reserve(2*encoded_size(descriptor["context"]))
                    context_key = descriptor["kind"],canonical(descriptor["context"]).encode()
                require(context_key not in contexts, "Duplicate prepared native parent/context")
                contexts[context_key] = descriptor
                if descriptor["kind"] == "states":
                    for day in descriptor["days"]:
                        require(day not in lifecycle_owners, "Overlapping prepared lifecycle owners")
                        lifecycle_owners[day] = descriptor["entry"]["native_ref"]
            require(calendar_set <= lifecycle_owners.keys(), "Prepared lifecycle owner coverage mismatch")
            from ..core.stock_portfolio import instant
            for number,days in enumerate(chunks):
                day_set = set(days)
                value,current = market._market_block(number,memory,budget)
                try:
                    require(value["sessions"] == list(days) and len(value["market_rows"]) == len(days)*len(universe),
                            "Prepared market partition shape mismatch")
                    keys = [(r["session"],r["security_id"]) for r in value["market_rows"]]
                    require(keys == [(day,s) for day in days for s in universe], "Prepared market row order/coverage mismatch")
                    indexed = dict(zip(keys,value["market_rows"]))
                    current.reserve(64*len(indexed))
                    for row in value["market_rows"]:
                        _validate_stock_market_row({k:v for k,v in row.items() if not k.startswith("_stock_")},
                            universe=universe_set,calendar=day_set,refs=refs,full=True)
                        active = _stock_lifecycle_active(identities[row["security_id"]],row["session"],
                            row["market_state"],row["_stock_lifecycle_meta"])
                        require(type(row["_stock_listed"]) is bool and row["_stock_listed"] == active and
                                (not active or row["security_id"] not in ended) and type(row["_stock_member"]) is bool and
                                row["_stock_lifecycle_policy"] == "best_effort_vendor_v1" and
                                instant(row["_stock_lifecycle_cutoff"]) == instant(row["execution_evidence_cutoff"]) and
                                type(row["_stock_factor_valid"]) is bool and row["_stock_factor_source_ref"] == refs[3],
                                "Prepared market lifecycle/clock mismatch")
                        require(lifecycle_owners[row["session"]] == row["_stock_lifecycle_owner_ref"],
                                "Prepared lifecycle differs from its original owner")
                        if row["market_state"] == "delisted": ended.add(row["security_id"])
                        require(active or all(row[n] is None for n in ("open","close","volume_shares")),
                                "Prepared unlisted security carries observations")
                    # Compare retained projections with the original admitted
                    # price/member/factor facts, one bounded view at a time.
                    for kind in ("market","factor","membership"):
                        native_stage = memory.stage()
                        try:
                            native = market._native_view(kind,days,memory,budget,native_stage)
                            native_stage.reserve(64*(len(native["records"])+sum(len(f["by_key"]) for f in native["field_meta"].values())))
                            original,metadata = _grid(native,days,universe)
                            for key,row in indexed.items():
                                fact = original[key]
                                if kind == "market":
                                    require(all(row[name] == (None if fact[name] is None else str(fact[name])) for name in ("open","close")) and
                                            row["volume_shares"] == fact["volume_shares"] and
                                            row["_stock_close_missing_reason"] == metadata["close"][key].get("missing_reason") and
                                            all(row["field_available_at"][name] == metadata[name][key].get("usable_from")
                                                for name in ("open","close","volume_shares")), "Prepared prices differ from original facts")
                                elif kind == "factor":
                                    require(row["_stock_factor_valid"] == (fact["factor"] is not None) and
                                            row["_stock_factor_missing_reason"] == metadata["factor"][key].get("missing_reason"),
                                            "Prepared factor qualification differs from original facts")
                                    factor = fact["factor"]; factor_meta = metadata["factor"][key]
                                    if factor is None:
                                        require(bool(factor_meta.get("missing_reason")), "Prepared null factor has no original reason")
                                        previous_factor[key[1]] = None
                                    else:
                                        require(decimal(str(factor),minimum=0) > 0 and
                                                _visible(factor_meta,key[0]+"T20:30:00+08:00"),
                                                "Prepared original factor is unavailable/invalid")
                                        old = previous_factor.get(key[1])
                                        if row["_stock_listed"] and old is not None and old != factor:
                                            memory.replace_global("prepared_factor_blocks",512*(len(factor_blocks)+1))
                                            factor_blocks.append(dict(security_id=key[1],effective_session=key[0],
                                                available_at=factor_meta["usable_from"],reason="UNEXPLAINED_FACTOR_CHANGE",source_refs=[refs[3]]))
                                        previous_factor[key[1]] = factor if row["_stock_listed"] else None
                                    if key[0] < identities[key[1]]["listing_date"]:
                                        require(fact["factor"] is None and row["limit_up"] is None and row["limit_down"] is None,
                                                "Prepared pre-listing diagnostics carry factor/limit facts")
                                else:
                                    require(row["_stock_member"] == fact["is_member"], "Prepared membership differs from original facts")
                        finally:
                            native = original = metadata = None; native_stage.close()
                finally: value = indexed = None; current.close()
            for (kind,days),locator in market._native.items():
                value,current = market._read(locator,memory,budget)
                try:
                    fields(value,"view bindings"); view = value["view"]; rows,metadata = _grid(view,days,universe)
                    require(kind in ("market","factor","membership") and
                            set(days) <= admitted_days, "Prepared native view scope mismatch")
                    require(not (coverage & {(kind,d) for d in days}), "Overlapping prepared native partitions")
                    coverage.update((kind,d) for d in days)
                    selected = contexts.get((kind,canonical(view["context"]).encode()))
                    require(selected is not None and set(days) <= set(selected["days"]), "Prepared native parent/context mismatch")
                    class Index:
                        content_digest = selected["entry"]["native_ref"]
                        groups = [(('records',),d) for d in days]
                        def header(self): return view["context"]
                        def value(self,path): return view["field_meta"][path[1]][path[2]]
                    _native_header(Index(),selected["entry"],wire)
                    if kind == "membership": _membership_rows(view,universe=universe,calendar=list(days),identities=identities)
                    else:
                        for name,entries in metadata.items():
                            for key,fact in entries.items():
                                if rows[key][name] is not None:
                                    require(_visible(fact,key[0]+"T20:30:00+08:00"), "Prepared native fact exceeds PIT cutoff")
                finally: value = None; current.close()
            require(coverage == {(k,d) for k in ("market","factor","membership") for d in calendar} |
                    {(k,d) for k in ("market","factor") for d in wire["market_input"]["warmup_sessions"]},
                    "Prepared native full coverage mismatch")
            equity = wire["stock_action_policy"] == "registered_equity_v1"
            target = globals_value["market_header"]["action_diagnostics" if equity else "action_blocks"]
            observed = [b for b in target if b.get("reason") == "UNEXPLAINED_FACTOR_CHANGE"]
            expected = factor_blocks if equity else sorted(factor_blocks,key=lambda b:(b["security_id"],b["effective_session"]))
            require(observed == expected, "Prepared factor continuity restrictions differ from original facts")
            market._statistics["prepared_decoded_bytes_peak"] = memory.peak_bytes
        finally: stage.close()


def save_stock_prediction_inputs(inputs, destination, *, limits):
    from .stock_owned_inputs import AdmittedStockInputs
    require(isinstance(inputs,AdmittedStockInputs), "Completed AdmittedStockInputs required")
    with inputs.execution_scope():
        market = inputs._AdmittedStockInputs__market
        require(market is not None, "Prepared predictions need their admitted market owner")
        store = inputs._AdmittedStockInputs__projection_store
        metadata = dict(contract_version=PREDICTION_VERSION,input_key=json.loads(inputs._AdmittedStockInputs__projection_key),
            block_sessions=inputs._AdmittedStockInputs__block_sessions,calendar=list(market._calendar),
            inventory=json.loads(inputs._AdmittedStockInputs__inventory),
            owner=dict(market_key=json.loads(market._key)))
        return _save(store,destination,metadata,limits)


@_checked_load
def load_stock_prediction_inputs(market, ref, request, *, limits, max_signal_bytes):
    from .stock_market_owner import AdmittedStockMarket
    from .stock_owned_inputs import AdmittedStockInputs, _IMPORT, _input_key, _projection_key
    from .stock_stream_contracts import validate_manifest
    require(isinstance(market,AdmittedStockMarket), "AdmittedStockMarket required")
    plan = validate_manifest(request); market._matches(request)
    manifest,store = _load(ref,version=PREDICTION_VERSION,limits=limits,maximum=max_signal_bytes)
    try:
        key = canonical(manifest["input_key"]).encode(); fields(manifest["owner"],"market_key")
        require(key == _projection_key(plan) and canonical(manifest["owner"]["market_key"]).encode() == market._key and
                manifest["block_sessions"] == market._block_sessions and manifest["calendar"] == list(market._calendar),
                "Prepared prediction scope/dependency mismatch")
        stats = dict(source_admissions=0,source_audit_seconds=0,capture_seconds=0,owned_bytes=store.used,
            max_owned_decoded_bytes=0,owned_blocks=len(store.offsets)//2-1,owned_read_bytes=0,owned_record_decodes=0,
            account_bindings=0,source_operations={},source_decoded_bytes_peak=0,source_scalar_cache_evictions=0,prepared_loads=1)
        stats.update(store.prepared_statistics)
        with market._scope():
            market._retain()
            result = AdmittedStockInputs(_IMPORT,store.file,store.offsets,_input_key(plan),market._block_sessions,
                canonical(manifest["inventory"]).encode(),stats,market=market,projection_store=store,projection_key=key)
        # Existing audit checks per-account policy, target refs and globals;
        # the entire saved prediction grid is checked before ledger allocation.
        _validate_prediction_snapshot(result,request,limits)
        with market._scope():
            if market._cache_prediction(key,store,canonical(manifest["inventory"]).encode(),stats,max_signal_bytes):
                result._AdmittedStockInputs__cache_key = key
        return result
    except BaseException:
        if "result" in locals(): result.close()
        else: store.close()
        raise


def _validate_prediction_snapshot(inputs, request, limits):
    from ..core.stock_portfolio import StockPredictionFrame, validate_stock_predictions
    from .stock_stream_contracts import validate_manifest, validate_source_audit
    from .stock_stream import _admit
    from .stock_stream_contracts import prediction_bindings
    from .stock_schedule import _verify_model
    from ..core.stock_portfolio import instant
    plan = validate_manifest(request); count = 0
    with inputs.execution_scope():
        _,audit,_ = _admit(request,inputs,inputs._AdmittedStockInputs__block_sessions,limits)
        validate_source_audit(audit.receipt,plan)
        lineage = inputs._prepared["raw_lineage"]
        bindings = {b["signal_run_ref"]:b for b in prediction_bindings(plan["prediction_input"])}
        require(set(lineage) == set(bindings), "Prepared model/fold lineage coverage mismatch")
        for reference,source in lineage.items():
            fields(source,"binding header spec model"); binding = bindings[reference]
            from .stock_stream_contracts import identity_view
            require(identity_view({"prediction_input":{"contract_version":"stock_prediction_input_refs_v2",
                "frames":[source["binding"]]}}) == identity_view({"prediction_input":{"contract_version":"stock_prediction_input_refs_v2",
                "frames":[binding]}}), "Prepared original raw binding differs")
            model,spec,header = source["model"],source["spec"],source["header"]
            _verify_model(model)
            require(Document.from_dict(spec).identity == binding["fold_spec_ref"] == header["fold_spec_ref"] and
                    model["model_ref"] == binding["model_ref"] == header["model_ref"] and
                    model["feature_ref"] == binding["feature_ref"] == header["feature_ref"] and
                    header["signal_run_ref"] == reference and spec["fit_session"] in plan["scope"]["calendar"] and
                    instant(model["fit_cutoff"]) == instant(spec["fit_cutoff"]) and
                    instant(model["simulated_available_at"]) == instant(spec["simulated_model_available_at"]),
                    "Prepared original model/fold/header linkage differs")
        output_bindings = {b["signal_run_ref"]:b for b in plan["prediction_input"]["frames"]}
        expected_trades = set(plan["scope"]["calendar"][1:]); observed_trades = set()
        collected = {}; byte_counts = {}; budget = inputs._memory
        orders = inputs._prepared["signal_orders"]
        require(set(orders) == set(output_bindings), "Prepared original row-order coverage mismatch")
        securities = {s:i for i,s in enumerate(plan["scope"]["prediction_universe"])}
        dates = {d:i for i,d in enumerate(plan["scope"]["calendar"])}
        full_headers = {}
        for location in inputs._prepared["signal_metadata_locations"]:
            value,metadata_stage = inputs._AdmittedStockInputs__projection_store._read_span(
                location["offset"],location["length"],inputs._memory,read_budget(limits))
            fields(value,"contract_version signal_ref header")
            require(value["contract_version"] == "stock_signal_metadata_v1" and value["signal_ref"] == location["signal_ref"] and
                    value["header"]["signal_run_ref"] == location["signal_ref"], "Prepared Derived original metadata mismatch")
            full_headers[location["signal_ref"]] = value["header"]
            charge = metadata_stage.used; metadata_stage.close()
            inputs._memory.reserve_global(("prepared_derived_header",location["signal_ref"]),charge)
        market = inputs._AdmittedStockInputs__market
        member_cache = member_stage = member_key = None
        def members_for(feature):
            nonlocal member_cache,member_stage,member_key
            from .stock_stream_inputs import _grid
            key = market._native_key("membership",[feature])
            if key != member_key:
                member_cache = None
                if member_stage is not None: member_stage.close()
                member_stage = inputs._memory.stage()
                view = market._native_view("membership",key[1],inputs._memory,read_budget(limits),member_stage)
                member_stage.reserve(64*(len(view["records"])+sum(len(f["by_key"]) for f in view["field_meta"].values())))
                member_cache = _grid(view,key[1],plan["scope"]["prediction_universe"])[0]
                member_key = key
            return member_cache
        def prediction_blocks():
            from .stock_stream_inputs import StockInputBlock
            calendar = plan["scope"]["calendar"]; width = inputs._AdmittedStockInputs__block_sessions
            for number in range(1,len(inputs._AdmittedStockInputs__offsets)//2):
                value,stage = inputs._AdmittedStockInputs__read(number,read_budget(limits))
                try:
                    days = tuple(value["sessions"])
                    require(days == tuple(calendar[(number-1)*width:number*width]), "Prepared prediction partition calendar mismatch")
                    signals = {item["session"]:(item.get("header",inputs._prepared["signal_headers"].get(item.get("signal_ref"))),
                        {(r["session"],r["security_id"]):r for r in item["rows"]}) for item in value["signals"]}
                    yield StockInputBlock(days,{},signals,tuple(value["bindings"]),stage.used,stage)
                finally: value = signals = None; stage.close()
        for block in prediction_blocks():
            for day,(header,rows) in block.signals.items():
                full = full_headers.get(header["signal_run_ref"],header)
                validated,indexed = validate_stock_predictions(StockPredictionFrame.from_dict({**full,"rows":list(rows.values())}))
                require(len(indexed) == len(plan["scope"]["prediction_universe"]), "Prepared prediction grid incomplete")
                feature = plan["scope"]["calendar"][dates[day]-1]
                require({key[0] for key in indexed} == {feature}, "Prepared prediction predecessor mismatch")
                reference = header["signal_run_ref"]
                require(reference in output_bindings and day not in observed_trades and day in expected_trades,
                        "Prepared output Signal identity/date mismatch")
                observed_trades.add(day)
                membership = members_for(feature)
                for key,row in indexed.items():
                    require(row["member"] == membership[key]["is_member"] and
                            instant(row["feature_knowledge_cutoff"]) == instant(feature+"T20:30:00+08:00") and
                            instant(row["knowledge_cutoff"]) == instant(feature+"T21:00:00+08:00"),
                            "Prepared prediction membership/clock differs from its original basis")
                if output_bindings[reference].get("kind") != "derived":
                    require(header == lineage[reference]["header"], "Prepared raw header differs from original lineage")
                    model = lineage[reference]["model"]; spec = lineage[reference]["spec"]
                    require(day in spec["oos_trade_sessions"] and
                            instant(spec["inference_cutoff_by_session"][feature]) == instant(feature+"T21:00:00+08:00"),
                            "Prepared fold inference scope differs")
                    for row in indexed.values():
                        require(instant(row["simulated_model_available_at"]) == instant(model["simulated_available_at"]),
                                "Prepared prediction/model release clock differs")
                current = collected.setdefault(reference,{})
                for row in indexed.values():
                    ordinal = dates[feature]*len(securities)+securities[row["security_id"]]
                    require(ordinal not in current, "Duplicate prepared original prediction ordinal")
                    size = len(canonical(row).encode())+128
                    byte_counts[reference] = byte_counts.get(reference,0)+size
                    require(3*byte_counts[reference] <= limits["max_block_bytes"], "Prepared original frame proof exceeds decoded budget")
                    budget.replace_global(("prepared_frame",reference),3*byte_counts[reference])
                    current[ordinal] = row
                # Original fold scopes are disjoint and complete; release each
                # proof as soon as its final original row has arrived.
                if len(current) == len(orders[reference]):
                    require(set(current) == set(orders[reference]) and len(set(orders[reference])) == len(current),
                            "Prepared original row order/coverage differs")
                    original = {**full,"rows":[current[i] for i in orders[reference]]}
                    if output_bindings[reference].get("kind") != "derived":
                        original.pop("signal_run_ref")
                        require(Document.from_dict(original).identity == reference, "Prepared raw values differ from original Signal ref")
                    else:
                        from ..core.stock_signal import _ref
                        require(_ref(original["rows"]) == header["score_ref"],
                                "Prepared Derived scores differ from original score ref")
                        original.pop("signal_run_ref")
                        require(Document.from_dict(original).identity == reference,
                                "Prepared Derived metadata differs from original Signal ref")
                    original = None; current.clear(); collected.pop(reference); byte_counts.pop(reference)
                    budget.release_global(("prepared_frame",reference))
                count += len(indexed)
        parent_rows = sum(len(source["spec"]["oos_trade_sessions"])*len(securities)
            for reference,source in lineage.items() if reference not in output_bindings)
        require(count+parent_rows == audit.receipt["counts"]["prediction_rows"] and observed_trades == expected_trades and not collected,
                "Prepared prediction full coverage mismatch")
        member_cache = None
        if member_stage is not None: member_stage.close()
        inputs._AdmittedStockInputs__statistics["max_owned_decoded_bytes"] = budget.peak_bytes
