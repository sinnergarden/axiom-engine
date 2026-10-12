"""Incremental immutable Feature tiles, scoped coverage and bounded projections.

The source adapter owns actual fact selection, per-output anchor/cutoff and
dependency receipts. This module calls the original Core and never substitutes
Feature arithmetic. Sample rows do not change the physical/reference universe.
"""
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path

from ..core import compile_feature_columns, execute_feature_columns
from .codec import digest, seal, require, is_ref, file_digest
from .identity import feature_definitions, select_feature_plan
from .storage import write_buffer, write_part
from .store import VerifiedMatrixStore
from .coverage import (VIEW_V1, VIEW_VERSION, coverage_summary, partition_descriptor,
                       read_index, write_date, publish_index, compact_view)

ARRAYS = {'values': 'float64_le', 'validity': 'bool_u8', 'reason_codes': 'int32_le',
          'available_at_utc_us': 'int64_le', 'source_codes': 'int32_le'}
PARTITION_VERSION = 'feature_library_partition_v1'


def _ordered(values, label):
    require(type(values) in (list, tuple) and bool(values) and
            all(type(v) is str and v for v in values) and list(values) == sorted(set(values)),
            'ordered unique ' + label + ' required')
    return list(values)


@dataclass(frozen=True)
class FeatureRequest:
    """Physical Feature grid plus an independent sample-row projection.

    security_ids is the prepared physical universe; it includes the complete
    reference cohort whenever the plan has CS. sample_universe only selects
    returned rows. A source adapter must reject unproved PIT/member profiles.
    """
    plan: object
    output_sessions: tuple
    security_ids: tuple
    input_semantics: dict
    sample_universe: tuple | None = None

    def admitted(self):
        days = _ordered(self.output_sessions, 'output sessions')
        securities = _ordered(self.security_ids, 'physical securities')
        sample = securities if self.sample_universe is None else _ordered(self.sample_universe, 'sample securities')
        require(set(sample) <= set(securities), 'sample must project physical securities')
        definitions = feature_definitions(self.plan, input_semantics=self.input_semantics)
        p = self.plan.to_dict()
        require(set(days) <= set(p['reference_members']), 'output dates absent from frozen calendar')
        if any(d['reference_dependent'] for d in definitions.values()):
            require(all(set(p['reference_members'][day]) <= set(securities) for day in p['reference_members']),
                    'physical grid must contain the complete CS reference cohort')
        return days, securities, sample, definitions


def _core_ref(value):
    return 'sha256:' + sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _capture_result(result, *, day, columns, securities, maximum_bytes):
    """Capture exact Core output bytes before validating its immutable receipt."""
    import numpy as np
    header = deepcopy({k: v for k, v in result.items() if k not in set(ARRAYS) | {'stats'}})
    ref = header.pop('result_ref')
    require(header.get('contract_version') == 'feature_column_result_v1' and
            _core_ref(header) == ref, 'Core column result receipt mismatch')
    require(header['sessions'] == [day] and header['security_ids'] == securities and
            [c['name'] for c in header['schema']] == columns, 'Core output axes/schema mismatch')
    shape = (1, len(securities), len(columns)); captured = {}
    require(sum(result[name].nbytes for name in ARRAYS) <= maximum_bytes,
            'Feature output capture exceeds working budget')
    for name in ARRAYS:
        array = result[name]
        raw = array.tobytes(order='C')
        descriptor = {'shape': list(shape), 'dtype': array.dtype.str,
                      'sha256': sha256(raw).hexdigest()}
        require(array.shape == shape and header['output_buffers'][name] == descriptor,
                'Core output buffer differs from exact receipt')
        captured[name] = np.frombuffer(raw, dtype=array.dtype).reshape(shape)
    require(np.isfinite(captured['values']).all(), 'finite Core values required')
    require(np.all((captured['reason_codes'] >= 0) &
                   (captured['reason_codes'] < len(header['reason_dictionary']))), 'Core reason code range')
    require(np.all((captured['source_codes'] >= 0) &
                   (captured['source_codes'] < len(header['source_sets']))), 'Core source code range')
    header['result_ref'] = ref
    return captured, header


def _write_codes(root, array):
    """Exact constant/sparse/dense codes; no expanded per-cell JSON metadata."""
    import numpy as np
    flat = array.reshape(-1)
    if np.all(flat == flat[0]):
        return {'encoding': 'constant', 'value': int(flat[0]), 'shape': list(array.shape)}
    code_dtype = 'uint8' if int(flat.max()) <= 255 else 'int32_le'
    value = flat.astype('u1' if code_dtype == 'uint8' else '<i4', copy=False)
    indices = np.flatnonzero(flat).astype('<u8')
    if indices.nbytes + indices.size * value.dtype.itemsize < value.nbytes:
        return {'encoding': 'sparse_zero', 'shape': list(array.shape),
            'indices': write_buffer(root, indices, dtype='uint64_le', shape=[indices.size]),
            'values': write_buffer(root, value[indices], dtype=code_dtype, shape=[indices.size])}
    return {'encoding': 'dense', 'buffer': write_buffer(root, value, dtype=code_dtype, shape=list(array.shape))}


def _read_codes(store, descriptor, shape):
    import numpy as np
    encoding = descriptor.get('encoding')
    if encoding == 'dense':
        value = store.buffer(descriptor['buffer'])
        require(value.shape == shape, 'code buffer shape mismatch')
        return value
    require(descriptor['shape'] == list(shape), 'code shape mismatch')
    if encoding == 'constant':
        require(type(descriptor['value']) is int and descriptor['value'] >= 0, 'nonnegative code required')
        return np.broadcast_to(np.asarray(descriptor['value'], dtype='int32'), shape)
    require(encoding == 'sparse_zero', 'unsupported code encoding')
    indices, codes = store.buffer(descriptor['indices']), store.buffer(descriptor['values'])
    require(indices.ndim == codes.ndim == 1 and indices.size == codes.size and
            (not indices.size or int(indices[-1]) < int(np.prod(shape))) and
            np.all(indices[1:] > indices[:-1]), 'ordered sparse code positions required')
    value = np.zeros(shape, dtype='int32'); value.reshape(-1)[indices] = codes
    value.flags.writeable = False
    return value


def _column_sources(header, codes, columns):
    import numpy as np
    bindings={b['id']:b for b in header['source_bindings']}
    return {column:[bindings[source] for source in sorted({source
        for code in np.unique(codes[:,:,j]) for source in header['source_sets'][int(code)]})]
        for j,column in enumerate(columns)}


def _partition_arrays(store, part):
    require(part['contract_version'] == PARTITION_VERSION, 'unsupported Feature partition')
    shape = (len(part['sessions']), len(part['security_ids']), len(part['columns']))
    require(set(part['buffers']) == set(ARRAYS), 'complete Feature buffers required')
    require(int(math.prod(shape)) * 50 <= store.maximum_matrix_bytes,
            'Feature partition exceeds matrix budget')
    arrays = {}
    for name in ARRAYS:
        arrays[name] = (_read_codes(store, part['buffers'][name], shape) if name in ('reason_codes', 'source_codes')
                        else store.buffer(part['buffers'][name]))
        require(arrays[name].shape == shape, 'Feature partition buffer shape mismatch')
    header = part['core_header']; binding = {k:v for k,v in header.items() if k != 'result_ref'}
    projection=part.get('core_output_projection')
    output_buffers=header['output_buffers'] if projection is None else projection['output_buffers']
    indices=list(range(len(part['columns']))) if projection is None else projection['column_indices']
    require(_core_ref(binding) == header['result_ref'] and
            header['sessions'] == part['sessions'] and header['security_ids'] == part['security_ids'] and
            [header['schema'][i] for i in indices] == part['schema'] and
            [c['name'] for c in part['schema']] == part['columns'],
            'partition Core receipt/axes mismatch')
    if projection is not None:
        require(projection['contract_version']=='feature_output_projection_v1' and
            projection['original_result_ref']==header['result_ref'] and
            projection['projection_ref']==digest({k:v for k,v in projection.items() if k!='projection_ref'}),
            'exact Core output projection receipt required')
    require(store.np.all((arrays['reason_codes'] >= 0) & (arrays['reason_codes'] < len(header['reason_dictionary']))) and
            store.np.all((arrays['source_codes'] >= 0) & (arrays['source_codes'] < len(header['source_sets']))),
            'partition dictionary range')
    require(part['column_source_bindings']==_column_sources(header,arrays['source_codes'],part['columns']),
            'partition column source projection mismatch')
    types = {'values':'float64', 'validity':'bool', 'available_at_utc_us':'int64',
             'reason_codes':'int32', 'source_codes':'int32'}
    for name, array in arrays.items():
        original = array.astype(types[name], copy=False)
        require(output_buffers[name] == {'shape':list(shape), 'dtype':original.dtype.str,
            'sha256':sha256(original.tobytes(order='C')).hexdigest()}, 'saved Feature bytes differ from Core receipt')
    return arrays


def _partition_buffer_paths(part):
    for descriptor in part['buffers'].values():
        if 'path' in descriptor: yield descriptor['path']
        elif descriptor['encoding']=='dense': yield descriptor['buffer']['path']
        elif descriptor['encoding']=='sparse_zero':
            yield descriptor['indices']['path']; yield descriptor['values']['path']


class FeatureLibrary:
    """One local cache root; immutable records are authoritative, index rebuildable."""
    def __init__(self, root, *, maximum_source_bytes=8*1024**3, maximum_working_bytes=512*1024**2,
                 column_block=16):
        self.root = Path(root).resolve()
        self.limits = {'maximum_source_bytes': maximum_source_bytes, 'maximum_matrix_bytes': maximum_working_bytes}
        require(type(column_block) is int and 0<column_block<=128, 'bounded positive column block required')
        self.column_block=column_block
        self._session=None

    @contextmanager
    def execution_session(self):
        """Bounded same-process admission epoch; fresh sessions hash actual files."""
        require(self._session is None, 'Feature execution session already active')
        with VerifiedMatrixStore(**self.limits) as store:
            self._session={'store':store,'arrays':{}}
            try: yield self
            finally:
                self._session['arrays'].clear(); self._session=None

    def _index(self):
        partitions, shards = read_index(self.root, min(64*1024**2,self.limits['maximum_matrix_bytes']//8))
        return {'partitions':partitions, 'shards':shards}

    def _publish_index(self, shards):
        publish_index(self.root, shards)

    def _recover(self, partitions, store, validated):
        known = {p['partition_ref'] for p in partitions}; recovered = []
        for path in sorted((self.root/'partitions').glob('*.json')):
            if not is_ref('sha256:' + path.stem): continue
            if 'sha256:' + path.stem in known: continue
            require(path.stat().st_size<=store.maximum_parent_bytes, 'matrix parent byte budget exceeded')
            descriptor = {'path': str(path), 'file_digest': file_digest(path), 'partition_ref': 'sha256:' + path.stem}
            part = store.read_json(descriptor, 'partition_ref')
            validated[part['partition_ref']]=_partition_arrays(store, part)
            partitions.append(coverage_summary(descriptor, part)); known.add(descriptor['partition_ref'])
            recovered.extend(part['sessions'])
        return recovered

    @staticmethod
    def _summary(descriptor, part):
        return {**descriptor, 'sessions': part['sessions'], 'security_axis_ref': digest(part['security_ids']),
            'columns': part['columns'], 'semantic_refs': part['semantic_refs'],
            'dependency_refs': part['dependency_refs'],'source_bindings':part['column_source_bindings']}

    @staticmethod
    def _coverage_lookup(parts):
        lookup = {}
        for part in parts:
            require(len(part['sessions']) == 1, 'one output day per Feature partition required')
            for column in part['columns']:
                key=(part['sessions'][0],part['security_axis_ref'],column,
                     part['semantic_refs'][column],part['dependency_refs'][column])
                lookup[key]=part
        return lookup

    def plan_coverage(self, request, source_adapter, *, _partitions=None):
        """Read metadata and source dependency receipts; never run Core or write."""
        days, securities, sample, definitions = request.admitted()
        security_axis_ref=digest(securities)
        dependencies = source_adapter.dependency_refs(definitions, days, securities)
        require(set(dependencies) == set(days), 'complete daily dependency bindings required')
        parts = self._index()['partitions'] if _partitions is None else _partitions
        lookup=self._coverage_lookup(parts); hits, gaps = [], []
        for day in days:
            require(set(dependencies[day]) == set(definitions) and
                    all(is_ref(ref) for ref in dependencies[day].values()), 'exact column dependency bindings required')
            for column, definition in definitions.items():
                match=lookup.get((day,security_axis_ref,column,definition['semantic_ref'],dependencies[day][column]))
                if match is not None:
                    hits.append({'session':day, 'column':column, 'partition':match})
                else: gaps.append({'session':day, 'column':column})
        return {'hits':hits, 'gaps':gaps, 'dependencies':dependencies, 'definitions':definitions,
            'sessions':days, 'security_ids':securities, 'sample_universe':sample}

    def ensure_features(self, request, source_adapter, *, progress=None):
        """Compute gaps, seal each date, then publish one compact coverage root.

        Adapter.prepare(plan, day, physical_security_ids, dependency_refs) must
        return the original daily anchored Core input, with feature_dependency_ref
        in input_refs. A failed build keeps sealed tiles for one recovery pass.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root/'.write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = self._index(); partitions = index['partitions']; shards = index['shards']
            # An index is a hint, never admission of its existing bytes.
            owner=VerifiedMatrixStore(**self.limits) if self._session is None else self._session['store']
            with (owner if self._session is None else nullcontext(owner)) as admitted:
                validated={} if self._session is None else self._session['arrays']
                baseline=deepcopy(admitted.metrics)
                admitted.check()
                recovered = self._recover(partitions,admitted,validated)
                coverage = self.plan_coverage(request, source_adapter,_partitions=partitions)
                hits={h['partition']['partition_ref']:h['partition'] for h in coverage['hits']}
                for summary in hits.values():
                    descriptor={k:summary[k] for k in ('path','file_digest','partition_ref')}
                    part=admitted.read_json(descriptor,'partition_ref')
                    require(coverage_summary(descriptor,part)==summary, 'coverage differs from admitted partition')
                    if part['partition_ref'] not in validated:
                        validated[part['partition_ref']]=_partition_arrays(admitted,part)
                admitted.check()
                admission_metrics={k:v-baseline[k] for k,v in admitted.metrics.items()}
            stats = {'core_calls':0, 'computed_output_cells':0, 'hit_columns':len(coverage['hits']),
                     'gap_columns':len(coverage['gaps']), 'recovered_partitions':len(recovered),
                     'coverage_date_writes':0, 'coverage_root_writes':0,
                     'maximum_execution_source_ids':0,
                     'admission_file_hash_calls':admission_metrics['file_hash_calls'],
                     'admission_hash_bytes':admission_metrics['hash_bytes']}
            compiled = {}; by_day={day:[] for day in coverage['sessions']}
            parts_by_day={}
            for part in partitions: parts_by_day.setdefault(part['sessions'][0],[]).append(part)
            dirty=set(recovered) | (set(parts_by_day)-set(shards))
            for day in sorted(dirty):
                shards[day]=write_date(self.root,day,parts_by_day[day])
                stats['coverage_date_writes']+=1
            final_hits={(h['session'],h['column']):h for h in coverage['hits']}
            for gap in coverage['gaps']: by_day[gap['session']].append(gap['column'])
            for day in coverage['sessions']:
                columns = by_day[day]
                if not columns: continue
                plan = select_feature_plan(request.plan, columns)
                bindings = {c:coverage['dependencies'][day][c] for c in columns}
                if hasattr(source_adapter,'execution_plan'):
                    actual=source_adapter.execution_plan(plan,day,coverage['security_ids'],bindings)
                    before=plan.to_dict(); after=actual.to_dict()
                    require({k:v for k,v in before.items() if k!='sources'}==
                            {k:v for k,v in after.items() if k!='sources'},
                            'execution source projection changed Feature semantics')
                    plan=actual
                stats['maximum_execution_source_ids']=max(stats['maximum_execution_source_ids'],len(plan.to_dict()['sources']))
                inputs = source_adapter.prepare(plan, day, coverage['security_ids'], bindings)
                require(inputs['input_refs'].get('feature_dependency_ref') == digest(bindings),
                        'prepared inputs do not bind planned dependencies')
                key = plan.identity
                if key not in compiled: compiled[key] = compile_feature_columns(plan)
                result = execute_feature_columns(compiled[key], inputs=inputs, output_sessions=[day],
                    max_working_bytes=self.limits['maximum_matrix_bytes'])
                arrays, header = _capture_result(result, day=day, columns=columns,
                    securities=coverage['security_ids'], maximum_bytes=self.limits['maximum_matrix_bytes'])
                stats['core_calls'] += 1; stats['computed_output_cells'] += len(coverage['security_ids'])*len(columns)
                for start in range(0,len(columns),self.column_block):
                    stop=min(start+self.column_block,len(columns)); block_columns=columns[start:stop]
                    block={name:value[:,:,start:stop] for name,value in arrays.items()}; buffers={}; block_digests={}
                    for name,dtype in ARRAYS.items():
                        array=block[name]
                        block_digests[name]={'shape':list(array.shape),'dtype':array.dtype.str,
                            'sha256':sha256(array.tobytes(order='C')).hexdigest()}
                        if name in ('reason_codes','source_codes'): buffers[name]=_write_codes(self.root,array)
                        else:
                            value=array.view('u1') if dtype=='bool_u8' else array
                            buffers[name]=write_buffer(self.root,value.reshape(-1),dtype=dtype,shape=list(value.shape))
                    projection=seal({'contract_version':'feature_output_projection_v1',
                        'original_result_ref':header['result_ref'],'column_indices':list(range(start,stop)),
                        'output_buffers':block_digests},'projection_ref')
                    part=seal({'contract_version':PARTITION_VERSION,'sessions':[day],
                        'security_ids':coverage['security_ids'],'columns':block_columns,
                        'schema':[coverage['definitions'][c]['schema'] for c in block_columns],
                        'semantic_refs':{c:coverage['definitions'][c]['semantic_ref'] for c in block_columns},
                        'dependency_refs':{c:bindings[c] for c in block_columns},'buffers':buffers,
                        'column_source_bindings':_column_sources(header,block['source_codes'],block_columns),
                        'core_header':header,'core_output_projection':projection},'partition_ref')
                    descriptor=write_part(self.root,part,'partition_ref')
                    records=self.root/'partitions'; records.mkdir(exist_ok=True)
                    record=records/(part['partition_ref'][7:]+'.json')
                    try: os.link(descriptor['path'],record)
                    except FileExistsError: require(file_digest(record)==descriptor['file_digest'],'Feature partition conflict')
                    descriptor['path']=str(record); summary=coverage_summary(descriptor,part); partitions.append(summary)
                    parts_by_day.setdefault(day,[]).append(summary)
                    for column in block_columns:
                        final_hits[day,column]={'session':day,'column':column,'partition':summary}
                    if progress is not None: progress({'stage':'partition_sealed','session':day,**deepcopy(stats)})
                shards[day]=write_date(self.root,day,parts_by_day[day])
                stats['coverage_date_writes']+=1
                del result, arrays, inputs
            require(len(final_hits)==len(coverage['sessions'])*len(coverage['definitions']),
                    'Feature coverage remains incomplete')
            if self._session is not None: self._session['store'].validate_boundary()
            view = compact_view(coverage, final_hits)
            descriptor = write_part(self.root, view, 'view_ref')
            self._publish_index(shards)
            stats['coverage_root_writes']=1
            if self._session is not None:
                stats['admission_file_hash_calls']=admitted.metrics['file_hash_calls']-baseline['file_hash_calls']
                stats['admission_hash_bytes']=admitted.metrics['hash_bytes']-baseline['hash_bytes']
            return descriptor, stats

    def open_feature_view(self, descriptor):
        """Open a fixed saved view. Reading never calls Data/Core or modifies index."""
        return FeatureView(descriptor,_session=self._session, **self.limits)


class FeatureView:
    def __init__(self, descriptor, _session=None, **limits):
        self._own_store=_session is None
        self._store = VerifiedMatrixStore(**limits) if _session is None else _session['store']
        self._view = self._store.read_json(descriptor, 'view_ref')
        require(self._view['contract_version'] in (VIEW_V1,VIEW_VERSION), 'unsupported Feature view')
        self.descriptor = deepcopy(descriptor)
        self._loaded = {} if _session is None else _session['arrays']
        self._parts={}; self._tiles={}; self._legacy=self._view['contract_version']==VIEW_V1
        if self._legacy:
            selections=({'session':t['session'],'columns':[t['column']], 'partition':t['partition']}
                        for t in self._view['tiles'])
        else:
            selections=self._view['blocks']
        for block in selections:
            for column in block['columns']:
                key=block['session'],column
                require(key not in self._tiles, 'duplicate Feature view cell selection')
                self._tiles[key]=block['partition']
        require(set(self._tiles)=={(d,c) for d in self._view['sessions'] for c in self._view['columns']} and
                set(self._view['definitions'])==set(self._view['columns']), 'incomplete Feature view selection')

    def _partition(self, day, column, *, store=None, parts=None):
        store=self._store if store is None else store
        parts=self._parts if parts is None else parts
        desc=self._tiles[day,column]; ref=desc['partition_ref']
        if ref not in parts:
            proof=partition_descriptor(desc)
            part=store.read_json(proof,'partition_ref')
            require(part['contract_version']==PARTITION_VERSION and part['sessions']==[day] and
                    part['security_ids']==self._view['security_ids'] and
                    set(part['semantic_refs'])==set(part['dependency_refs'])==set(part['columns']) and
                    all(is_ref(r) for r in part['semantic_refs'].values()) and
                    all(is_ref(r) for r in part['dependency_refs'].values()), 'view partition coverage mismatch')
            parts[ref]=part
        else: store.check_path(partition_descriptor(desc)['path'])
        part=parts[ref]
        require(part['sessions']==[day] and column in part['columns'] and
                part['semantic_refs'][column]==self._view['definitions'][column]['semantic_ref'],
                'view Feature semantics mismatch')
        if self._legacy:
            require(FeatureLibrary._summary(partition_descriptor(desc),part)==desc,
                    'coverage summary differs from sealed partition')
        return part

    def column_receipts(self, session, columns=None):
        """Read original per-column semantics, dependencies and sources, without Core.

        The compact selection only points to immutable partitions. Actual parent
        bytes and content refs are admitted before returning their receipts;
        source eligibility/PIT remains the source adapter's responsibility.
        """
        with self.operation():
            columns=list(self._view['columns'] if columns is None else columns)
            require(bool(columns) and len(columns)==len(set(columns)) and
                    all((session,c) in self._tiles for c in columns), 'Feature receipt selection outside view')
            cached=all(self._tiles[session,c]['partition_ref'] in self._parts for c in columns)
            # Planning years of dependencies must not accumulate every date's
            # metadata in one owner and repeatedly stat all previous parents. Reuse
            # projected parents, otherwise admit this day's receipts in a bounded
            # metadata epoch; original array admission still happens in project().
            owner=(nullcontext(self._store) if cached else VerifiedMatrixStore(
                maximum_source_bytes=self._store.maximum_source_bytes,
                maximum_matrix_bytes=self._store.maximum_matrix_bytes,
                maximum_parent_bytes=self._store.maximum_parent_bytes))
            result={}
            with owner as store, store.operation():
                parts=self._parts if cached else {}
                for column in columns:
                    part=self._partition(session,column,store=store,parts=parts)
                    result[column]={'semantic_ref':part['semantic_refs'][column],
                        'dependency_ref':part['dependency_refs'][column],
                        'source_bindings':deepcopy(part['column_source_bindings'][column]),
                        'partition':partition_descriptor(self._tiles[session,column])}
            return result

    @property
    def identity(self): return self._view['view_ref']

    @contextmanager
    def operation(self):
        """Keep an adapter's actual consumption and publication in one lease."""
        with self._store.operation():
            self._store.check_path(self.descriptor['path'])
            yield self

    def validate_boundary(self):
        """Explicit publication check; outside a lease checks the whole owner."""
        self._store.check_path(self.descriptor['path'])
        self._store.validate_boundary()

    def describe(self):
        """Stable reference, stages, physical/sample coverage and projection abilities."""
        with self.operation():
            return {'contract_version':'feature_view_capabilities_v1', 'view_ref':self.identity,
                **deepcopy({k:self._view[k] for k in ('sessions','security_ids','sample_universe','columns','definitions')}),
                'complete':True, 'supports_projection':True, 'missing_values_preserved':True}

    def project(self, keys=None, columns=None):
        """Return owned readonly packed arrays in requested key/column order."""
        import numpy as np
        with self.operation():
            view = self._view
            columns = list(view['columns'] if columns is None else columns)
            require(bool(columns) and len(columns) == len(set(columns)) and set(columns) <= set(view['columns']),
                    'ordered unique Feature columns required')
            keys = ([(s,d) for d in view['sessions'] for s in view['sample_universe']] if keys is None else list(keys))
            require(bool(keys) and len(set(keys)) == len(keys) and all(s in view['security_ids'] and d in view['sessions'] for s,d in keys),
                    'unique keys inside fixed Feature view required')
            shape = len(keys), len(columns)
            require(len(keys)*len(columns)*50 <= self._store.maximum_matrix_bytes, 'Feature projection budget exceeded')
            out = {name:np.zeros(shape, dtype='bool' if name == 'validity' else 'float64' if name == 'values'
                               else 'int64' if name == 'available_at_utc_us' else 'int32') for name in ARRAYS}
            reasons = [None]; source_sets = [[]]; bindings = {}; used = {}
            targets={}
            for i,(_,day) in enumerate(keys): targets.setdefault(day,[]).append(i)
            tiles=self._tiles
            for day,target in targets.items():
                for j, column in enumerate(columns):
                    desc = tiles[day,column]; ref = desc['partition_ref']
                    if ref not in used:
                        part = self._partition(day,column)
                        if ref not in self._loaded: self._loaded[ref] = _partition_arrays(self._store, part)
                        for path in _partition_buffer_paths(part): self._store.check_path(path)
                        arrays = self._loaded[ref]
                        reason_map=[]; source_map=[]
                        for reason in part['core_header']['reason_dictionary']:
                            if reason not in reasons: reasons.append(reason)
                            reason_map.append(reasons.index(reason))
                        for sources in part['core_header']['source_sets']:
                            if sources not in source_sets: source_sets.append(sources)
                            source_map.append(source_sets.index(sources))
                        for binding in part['core_header']['source_bindings']:
                            require(binding['id'] not in bindings or bindings[binding['id']] == binding, 'conflicting source ID')
                            bindings[binding['id']] = binding
                        used[ref]=(part,arrays,np.asarray(reason_map),np.asarray(source_map),
                            {s:i for i,s in enumerate(part['security_ids'])},
                            {c:i for i,c in enumerate(part['columns'])})
                    part,arrays,reason_map,source_map,axis,positions = used[ref]
                    require(part['semantic_refs'][column] == view['definitions'][column]['semantic_ref'], 'view Feature semantics mismatch')
                    col = positions[column]
                    rows = [axis[keys[i][0]] for i in target]
                    for name in ARRAYS:
                        values = arrays[name][0,rows,col]
                        if name == 'reason_codes': values = reason_map[values]
                        if name == 'source_codes': values = source_map[values]
                        out[name][target,j] = values
            selected=sorted(set(int(code) for code in np.unique(out['source_codes'])))
            remap=np.zeros(len(source_sets),dtype='int32')
            for code,old in enumerate(selected): remap[old]=code
            source_sets=[source_sets[old] for old in selected]
            out['source_codes']=remap[out['source_codes']]
            actual_sources={source for group in source_sets for source in group}
            for name, array in out.items(): out[name]=np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(shape)
            self.validate_boundary()
            return {**out, 'keys':keys, 'schema':[deepcopy(view['definitions'][c]['schema']) for c in columns],
                'reason_dictionary':reasons, 'source_sets':source_sets,
                'source_bindings':[bindings[source] for source in sorted(actual_sources)],
                'view_ref':self.identity, 'partition_refs':list(used), 'null_available_at_utc_us':-(2**63)}

    def close(self):
        self._tiles.clear(); self._parts.clear()
        if self._own_store:
            self._loaded.clear(); self._store.close()
        else: self._store.check()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
