"""Consumer adapter for explicitly admitted Data native JSON part handles.

No Data writer/import or additional account executor. Caller owns the handles;
use them sequentially and close them after market/basis admission. Delivery
manifest hashes are separate from complete original DataBatch native refs.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from threading import Lock, get_ident

from ..core.contracts import integer, require
from .stock_market import _StockActionStream
from .stock_stream_inputs import StockInputSource, _DecodedMemory, _encoded_size, _path
from .stock_stream_outputs import _json_pieces


def _digest(value):
    h = hashlib.sha256()
    for piece in _json_pieces(value):
        h.update(piece)
    return 'sha256:'+h.hexdigest()


class StockNativeViewSource(StockInputSource):
    """Original Signal loader plus borrowed, admitted native Data handles.

    max_active_bytes must equal the account input decoded limit. Reservations
    cover the borrowed handle graph, one copied manifest, pending Data blocks,
    current windows and returned copies together. No persisted PASS flag.

    Use this Source with admit_stock_market_inputs, or with
    bind_stock_prediction_inputs for a previously unadmitted model basis.
    The normal saved Signal loader is inherited unchanged. After admission,
    accounts use the private market/Signal owners, and callers may close both
    this Source and their Data handles. Keep handles alive during admission.
    """
    def __init__(self, *, native_views, max_active_bytes, **options):
        super().__init__(**options)
        integer(max_active_bytes, 1)
        require(type(native_views) in (tuple, list) and bool(native_views), 'Admitted native handles required')
        self._pid = os.getpid(); self._closed = False; self._lease = Lock(); self._thread = None
        self._maximum = max_active_bytes; self._baseline = _DecodedMemory(max_active_bytes)
        self._native_batches = {}; self._deliveries = []
        for number, handle in enumerate(native_views):
            size = handle.statistics['manifest_bytes']; integer(size, 1)
            # Charge existing handle and copied descriptor graph before deepcopy.
            self._baseline.reserve_global(('handle', number), 64*size+1024)
            manifest = handle.manifest
            require(manifest['contract_version'] == 'data_native_view_v1' and
                    manifest['logical_contract_version'] == 'data_batch_v1' and
                    manifest['exporter_version'] == 'native_json_parts_v1', 'Unsupported native delivery')
            root = Path(handle.root).resolve(); ref = _digest(manifest)
            # Include two consumer inventory copies and the handle stat/key
            # maps, besides our retained wrappers, before any of them grow.
            self._baseline.reserve_global(('delivery_wrappers', number), 1536*(len(manifest['files'])+1)+16384)
            files = [{'manifest_uri': str(root/'manifest.json'), 'file_bytes': size, 'native_view_ref': ref}]
            files += [{'manifest_uri': str(root/uri), 'file_bytes': value['bytes'], 'native_view_ref': ref}
                      for uri, value in manifest['files'].items()]
            delivery = {'handle': handle, 'manifest': manifest, 'ref': ref, 'files': files,
                        'initial_statistics': deepcopy(handle.statistics)}
            self._deliveries.append(delivery)
            for batch in manifest['batches']:
                require(batch['native_ref'] not in self._native_batches, 'Duplicate native handle logical ref')
                self._native_batches[batch['native_ref']] = delivery, batch

    def _check(self):
        require(os.getpid() == self._pid, 'Native stock source belongs to its creating process')
        require(not self._closed, 'Native stock source is closed')

    def _attach(self):
        self._check()
        require(self._thread == get_ident(), 'Native stock reads require exclusive source scope')
        require(self._memory is not None and self._memory.limit == self._maximum,
                'Native active budget must equal max_block_bytes')
        self._memory.reserve_global(('native_handle_baseline', id(self)), self._baseline.global_bytes)

    @contextmanager
    def execution_scope(self):
        self._check(); require(self._lease.acquire(blocking=False), 'Native stock source supports sequential admission only')
        try:
            self._check(); self._thread = get_ident()
            yield
            self._check()
            # Public close/mutation guard once at admission boundary, not per
            # window. Caller must keep borrowed handles alive for this scope.
            self._close_private_views()
            for number, delivery in enumerate(self._deliveries):
                with self._baseline.stage() as stage:
                    stage.reserve(32*delivery['initial_statistics']['manifest_bytes']+1024)
                    require(_digest(delivery['handle'].manifest) == delivery['ref'], 'Borrowed native manifest changed')
        finally:
            # An inherited scope must reject before touching locks/cursors.
            self._check()
            try:
                self._close_private_views()
            finally:
                self._thread = None; self._lease.release()

    def native_inventory(self, artifact):
        self._check()
        pair = self._native_batches.get(artifact['content_digest'])
        if pair is None:
            return None
        delivery, batch = pair
        self._baseline.reserve_global(('artifact_wrapper', _digest(artifact)), 6*_encoded_size(artifact)+512)
        return deepcopy(delivery['files']), {'artifact': deepcopy(artifact), 'native_ref': batch['native_ref'],
            'native_view_ref': delivery['ref']}

    def _location(self, artifact):
        self._check()
        require(artifact['content_digest'] in self._native_batches, 'Native handle missing logical ref')
        return _path(artifact, must_exist=False)

    def inventory(self, manifest):
        self._check()
        plan = manifest.to_dict() if hasattr(manifest, 'to_dict') else manifest
        files = {}; native = []
        native_refs = {i['artifact']['content_digest'] for i in plan['market_input']['native_inputs']}
        for artifact in self._artifacts(plan):
            delivery = self.native_inventory(artifact)
            if delivery is not None:
                physical, binding = delivery; native.append(binding)
                for item in physical:
                    files.setdefault(item['manifest_uri'], item)
            else:
                require(artifact['content_digest'] not in native_refs, 'Native handle missing logical ref')
                path = _path(artifact); files[str(path)] = {'artifact': artifact, 'file_bytes': path.stat().st_size}
                if '#' in artifact['manifest_uri']:
                    descriptor = path.with_name('manifest.json')
                    files.setdefault(str(descriptor), {'manifest_uri': str(descriptor), 'file_bytes': descriptor.stat().st_size})
        return {'files': list(files.values()), 'native_artifacts': native,
            'input_bytes': sum(f['file_bytes'] for f in files.values()), 'folds': len(plan.get('prediction_input', {}).get('frames', [])),
            'declared_rows': {'market_rows': len(plan['scope']['calendar'])*len(plan['scope']['execution_universe']),
                'prediction_rows': (len(plan['scope']['calendar'])-1)*len(plan['scope']['prediction_universe']) if plan.get('prediction_input', {}).get('frames') else 0},
            'declared_scope': deepcopy(plan['scope'])}

    def _index(self, artifact, budget):
        self._attach()
        if artifact['content_digest'] not in self._native_batches:
            require(str(_path(artifact, must_exist=False)) not in self._scan_native_scopes, 'Native handle missing logical ref')
            return super()._index(artifact, budget)
        key = ('native_view', artifact['content_digest'])
        if key not in self._indexes:
            delivery, batch = self._native_batches[artifact['content_digest']]
            location = str(self._location(artifact))
            plan, _ = self._scan_native_scopes[location]
            self._indexes[key] = _NativeIndex(self, artifact, delivery, batch, plan)
        return self._indexes[key]

    @property
    def statistics(self):
        self._check()
        value = super().statistics
        value['native_views'] = [{'manifest_ref': d['ref'], 'cold_statistics': deepcopy(d['initial_statistics']),
            'current_statistics': deepcopy(d['handle'].statistics)} for d in self._deliveries]
        value['source_operations']['native_selected_reads'] = sum(d['handle'].statistics.get('selected_reads', 0)-d['initial_statistics'].get('selected_reads', 0) for d in self._deliveries)
        value['source_operations']['native_selected_bytes'] = sum(d['handle'].statistics.get('selected_bytes', 0)-d['initial_statistics'].get('selected_bytes', 0) for d in self._deliveries)
        return value

    def close(self):
        self._check(); require(self._lease.acquire(blocking=False), 'Cannot close native stock source during admission')
        try:
            self._close_private_views(); self._indexes.clear()
            self._native_batches.clear(); self._deliveries.clear(); self._baseline = None
            self._closed = True
        finally:
            self._lease.release()

    def __enter__(self):
        self._check(); return self

    def __exit__(self, *_):
        self.close()


class _NativeIndex:
    """One parent stream and bounded pending/current windows; never full JSON."""
    def __init__(self, source, artifact, delivery, batch, plan):
        self.source, self.delivery, self.batch = source, delivery, batch
        self.path = source._location(artifact); self.content_digest = batch['native_ref']
        self._event = batch['method'] == 'events'
        days = [None] if self._event else batch['context']['query']['sessions']
        source._memory.reserve_global(('native_group_index', id(self)), 64*len(days)*(1+len(batch['field_meta']))+512)
        self.groups = [(path, day) for day in days for path in [('records',)]+[('field_meta', n, 'by_key') for n in batch['field_meta']]]
        self.statistics = {k: 0 for k in ('file_bytes','scan_seconds','read_calls','scan_read_bytes','read_seconds','hash_seconds',
            'content_hash_bytes','file_hash_bytes','scalar_count','scalar_batches','scalar_bytes','scalar_cache_hits','scalar_decode_count',
            'scalar_canonical_count','scalar_decode_seconds','scalar_canonical_seconds','row_decode_count','row_decode_seconds',
            'reread_bytes','reread_calls','local_decode_count','local_decode_seconds','span_identity_count','span_identity_seconds',
            'unsigned_identity_count','unsigned_identity_seconds','cjson_files','cjson_helper_seconds','cjson_parse_seconds',
            'cjson_validation_seconds','cjson_compare_seconds','cjson_spool_seconds','cjson_compare_bytes','cjson_spool_bytes',
            'cjson_metadata_consume_seconds','cjson_peak_tree_rss_bytes')}
        wanted = None if self._event else set(days)&set(plan['scope']['calendar']+plan['market_input']['warmup_sessions'])
        self._wanted = wanted
        self._descriptors = [b for b in batch['blocks'] if wanted is None or wanted.intersection(b['sessions'])]
        self._iterator = delivery['handle'].iter_blocks(native_ref=self.content_digest, sessions=None if wanted is None else sorted(wanted))
        self._number = 0; self._pending = self._pending_stage = None; self._position = 0
        self._cache = self._cache_stage = self._cache_days = None; self._ended = False

    def header(self):
        return self.batch['context']

    def value(self, path):
        require(len(path) == 3 and path[0] == 'field_meta', 'Unsupported native descriptor value')
        return self.batch['field_meta'][path[1]][path[2]]

    def unchanged(self):
        self.source._attach()
        # Data's one stream verifies its inventory at entry/exit and selected
        # files on each yield. No per-window whole-directory scan is added.

    def _drop_pending(self):
        self._pending = None
        if self._pending_stage is not None:
            self._pending_stage.close(); self._pending_stage = None

    def _load(self):
        self.source._attach()
        if self._number == len(self._descriptors):
            require(next(self._iterator, None) is None, 'Native parent stream has extra blocks')
            self._drop_pending(); self._ended = True; return False
        desc = self._descriptors[self._number]
        unique = {r['uri']: r for r in [desc['records'], *desc['by_key'].values()]}
        stage = self.source._memory.stage()
        try:
            stage.reserve(32*sum(r['bytes'] for r in unique.values())+3*_encoded_size(self.batch['context'])+64*desc['rows']+512)
            value = next(self._iterator, None)
            q = self.batch['context']['query']
            expected = [i for i in range(desc['ordinal'], desc['ordinal']+desc['rows'])
                        if self._wanted is None or q['sessions'][i//len(q['symbols'])] in self._wanted]
            require(value is not None and value['native_ref'] == self.content_digest and
                    value['ordinal'] == desc['ordinal'] and value['context'] == self.batch['context'] and
                    value['ordinals'] == expected,
                    'Native returned parent/context/ordinals differ from admission')
            require(len(value['records']) == len(expected) and set(value['field_meta']) == set(self.batch['field_meta']),
                    'Native returned records/metadata shape differs')
            for name, field in value['field_meta'].items():
                require({k:v for k,v in field.items() if k != 'by_key'} == self.batch['field_meta'][name] and
                    len(field['by_key']) == len(value['records']) and
                    [tuple(r[k] for k in self.batch['row_key']) for r in field['by_key']] ==
                    [tuple(r[k] for k in self.batch['row_key']) for r in value['records']], 'Native returned by_key alignment differs')
            # The borrowed Data generator may retain its last yield until it
            # advances. Keep that charge while decoding the replacement.
            self._drop_pending()
            self._pending, self._pending_stage = value, stage; self._position = 0; self._number += 1
            return True
        except BaseException:
            close = getattr(self._iterator, 'close', None)
            if close is not None: close()
            value = None; self._drop_pending(); stage.close(); raise

    def _events(self):
        try:
            while self._load():
                yield self._pending
        finally:
            self.source._check()
            close = getattr(self._iterator, 'close', None)
            if close is not None: close()
            self._drop_pending()

    def batch_view(self, days, budget, *, reservation):
        self.unchanged()
        require(reservation.memory is self.source._memory, 'One native decoded budget required')
        if self._event:
            require(days == [None] or tuple(days) == (None,), 'Native events cannot be date-filtered')
            return _StockActionStream(context=self.header(), field_meta=self.batch['field_meta'],
                row_count=self.batch['row_count'], blocks=self._events), 0, []
        days = tuple(days)
        if self._cache_days != days:
            require(self._cache_days is None or self._cache_days[-1] < days[0], 'Native window cannot go backwards')
            self._cache = None
            if self._cache_stage is not None: self._cache_stage.close()
            self._cache_stage = self.source._memory.stage(); stage = self._cache_stage
            stage.reserve(3*_encoded_size([self.header(), self.batch['field_meta']])+1024)
            window = {'context':deepcopy(self.header()), 'records':[],
                'field_meta':{n:{**deepcopy(h),'by_key':[]} for n,h in self.batch['field_meta'].items()}}
            ordinals = []; parts = {}; wanted = set(days)
            while True:
                if self._pending is None or self._position == len(self._pending['records']):
                    if not self._load(): break
                row = self._pending['records'][self._position]
                if row['session'] > days[-1]: break
                if row['session'] in wanted:
                    facts = {n:h['by_key'][self._position] for n,h in self._pending['field_meta'].items()}
                    stage.reserve(3*(_encoded_size(row)+sum(_encoded_size(r) for r in facts.values()))+128*(1+len(facts)))
                    window['records'].append(row)
                    for n,fact in facts.items(): window['field_meta'][n]['by_key'].append(fact)
                    ordinals.append(self._pending['ordinals'][self._position])
                    for part in self._pending['bindings']: parts.setdefault(part['uri'], part)
                self._position += 1
                row = facts = fact = part = None
            refs = [{'parent_ref':self.content_digest, 'native_view_ref':self.delivery['ref'],
                'ordinals':ordinals, 'parts':list(parts.values())}]
            stage.reserve(3*_encoded_size(refs)+512)
            self._cache = window, refs; self._cache_days = days
        used = 3*_encoded_size(self._cache)+512
        reservation.reserve(used)
        view, bindings = deepcopy(self._cache)
        return view, used, bindings

    def close(self):
        self.source._check()
        close = getattr(self._iterator, 'close', None)
        if close is not None: close()
        self._cache = None
        if self._cache_stage is not None: self._cache_stage.close(); self._cache_stage = None
        self._drop_pending()
