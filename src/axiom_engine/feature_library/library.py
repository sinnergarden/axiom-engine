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
import time
from pathlib import Path

from ..core import compile_feature_columns, execute_feature_columns
from .codec import digest, seal, require, is_ref, file_digest
from .identity import feature_definitions, select_feature_plan, select_security_plan
from .storage import write_buffer, write_part
from .store import VerifiedMatrixStore
from .coverage import (VIEW_V1, VIEW_V2, VIEW_VERSION, coverage_summary, partition_descriptor,
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

    def admitted(self, definitions=None):
        days = _ordered(self.output_sessions, 'output sessions')
        securities = _ordered(self.security_ids, 'physical securities')
        sample = securities if self.sample_universe is None else _ordered(self.sample_universe, 'sample securities')
        require(set(sample) <= set(securities), 'sample must project physical securities')
        definitions = feature_definitions(self.plan, input_semantics=self.input_semantics) if definitions is None else definitions
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
        owner=array
        while isinstance(owner,np.ndarray) and owner.base is not None:owner=owner.base
        require(isinstance(owner,bytes) and not array.flags.writeable and array.shape==shape and
                header['output_buffers'][name]['shape']==list(shape) and
                header['output_buffers'][name]['dtype']==array.dtype.str,'immutable Core buffer required')
        captured[name]=array
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
    nonzero=int(np.count_nonzero(flat))
    if nonzero*(8+value.dtype.itemsize)<value.nbytes:
        indices = np.flatnonzero(flat).astype('<u8')
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
    header = part['core_header']
    if 'path' in header:header=store.read_json(header,'header_ref')['core_header']
    binding = {k:v for k,v in header.items() if k != 'result_ref'}
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
        checksum=(part['buffers'][name]['file_digest'][7:] if name not in ('reason_codes','source_codes') else
                  sha256(original.tobytes(order='C')).hexdigest())
        require(output_buffers[name] == {'shape':list(shape), 'dtype':original.dtype.str,
            'sha256':checksum}, 'saved Feature bytes differ from Core receipt')
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
                 column_block=None, recover=False):
        self.root = Path(root).resolve()
        self.limits = {'maximum_source_bytes': maximum_source_bytes, 'maximum_matrix_bytes': maximum_working_bytes}
        require(column_block is None or type(column_block) is int and column_block>0, 'positive optional column block required')
        self.column_block=column_block;self.recover=recover
        self._session=None
        self._definitions={};self._compiled={}
        self._coverage_cache=None;self._recovered_directory_mark=None
        self.coverage_metrics={'index_loads':0,'recovery_scans':0}

    @contextmanager
    def execution_session(self):
        """Bounded same-process admission epoch; fresh sessions hash actual files."""
        require(self._session is None, 'Feature execution session already active')
        with VerifiedMatrixStore(**self.limits) as store:
            self._session={'store':store,'arrays':{}}
            try: yield self
            finally:
                self._session['arrays'].clear(); self._session=None

    @staticmethod
    def _epoch(path):
        if not path.exists():return None
        st=path.stat()
        return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns

    def _index(self, sessions=None):
        wanted=None if sessions is None else set(sessions)
        mark=self._epoch(self.root/'index.json')
        if self._coverage_cache is None or self._coverage_cache[0]!=mark:
            self._coverage_cache=(mark,{'partitions':[],'shards':{},'loaded_days':set()})
            self._recovered_directory_mark=None
        value=self._coverage_cache[1]
        missing=None if wanted is None else wanted-value['loaded_days']
        if missing is None or missing:
            parts,shards=read_index(self.root,min(64*1024**2,self.limits['maximum_matrix_bytes']//8),missing)
            require(self._epoch(self.root/'index.json')==mark,'coverage changed during read')
            known={p['partition_ref'] for p in value['partitions']}
            value['partitions'].extend(p for p in parts if p['partition_ref'] not in known)
            value['shards']=shards;value['loaded_days'].update(shards if wanted is None else wanted)
            self.coverage_metrics['index_loads']+=1
        return {'partitions':[p for p in value['partitions'] if wanted is None or p['sessions'][0] in wanted],
                'shards':dict(value['shards'])}

    def _publish_index(self, shards):
        publish_index(self.root, shards)

    def _recover(self, partitions, store, validated):
        if not self.recover:return []
        directory=self.root/'partitions';mark=self._epoch(directory)
        if self._recovered_directory_mark==mark and self._coverage_cache is not None:return []
        self.coverage_metrics['recovery_scans']+=1
        known={p['partition_ref'] for p in partitions};recovered=[]
        for path in sorted(directory.glob('*.json')):
            if not is_ref('sha256:' + path.stem):continue
            if 'sha256:' + path.stem in known:continue
            require(path.stat().st_size<=store.maximum_parent_bytes,'matrix parent byte budget exceeded')
            descriptor={'path':str(path),'file_digest':file_digest(path),'partition_ref':'sha256:'+path.stem}
            part=store.read_json(descriptor,'partition_ref')
            validated[part['partition_ref']]=_partition_arrays(store,part)
            partitions.append(coverage_summary(descriptor,part));known.add(descriptor['partition_ref'])
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
        key=(request.plan.identity,digest(request.input_semantics))
        if key not in self._definitions:self._definitions[key]=feature_definitions(request.plan,input_semantics=request.input_semantics)
        days, securities, sample, definitions = request.admitted(self._definitions[key])
        security_axis_ref=digest(securities)
        dependencies = source_adapter.dependency_refs(definitions, days, securities)
        require(set(dependencies) == set(days), 'complete daily dependency bindings required')
        parts = self._index(days)['partitions'] if _partitions is None else _partitions
        lookup=self._coverage_lookup(parts); hits, gaps = [], []
        # Legacy immutable summaries have only an axis hash. Read the sealed
        # parent to recover its axis; never rewrite a partition to add an index.
        candidates={}; scoped={}; selected_axes={}
        independent={c:d for c,d in definitions.items() if not d['reference_dependent']}
        for candidate in parts:
            for column in candidate['columns']:
                candidates.setdefault((candidate['sessions'][0],column,candidate['semantic_refs'][column]),[]).append(candidate)
        for day in days:
            candidate_axes={}
            with VerifiedMatrixStore(**self.limits) as store:
                require(set(dependencies[day]) == set(definitions) and
                        all(is_ref(ref) for ref in dependencies[day].values()), 'exact column dependency bindings required')
                for column, definition in definitions.items():
                    match=lookup.get((day,security_axis_ref,column,definition['semantic_ref'],dependencies[day][column]))
                    if match is not None:
                        hits.append({'session':day,'column':column,'partition':match,'security_ids':securities})
                        continue
                    remaining=set(securities)
                    # Reference-dependent results use the complete physical grid.
                    # Only independent columns can be assembled from older blocks.
                    if not definition['reference_dependent']:
                        for candidate in reversed(candidates.get((day,column,definition['semantic_ref']),[])):
                            ref=candidate['partition_ref']
                            if ref not in candidate_axes:
                                desc=partition_descriptor(candidate)
                                part=store.read_json(desc,'partition_ref')
                                require(coverage_summary(desc,part)==candidate, 'coverage differs from admitted partition')
                                candidate_axes[ref]=_ordered(part['security_ids'],'partition securities')
                            axis=candidate_axes[ref]
                            selected=sorted(remaining.intersection(axis))
                            if not selected or not set(axis)<=set(securities): continue
                            key=candidate['security_axis_ref']
                            if key not in scoped:
                                scoped[key]=source_adapter.dependency_refs(independent,days,axis)
                            if scoped[key][day][column]!=candidate['dependency_refs'][column]: continue
                            selected=selected_axes.setdefault(tuple(selected),selected)
                            hits.append({'session':day,'column':column,'partition':candidate,'security_ids':selected})
                            remaining.difference_update(selected)
                            if not remaining: break
                    if remaining: gaps.append({'session':day,'column':column,'security_ids':sorted(remaining)})
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
            planning_started=time.monotonic()
            index = self._index(None if self.recover else request.output_sessions); partitions = index['partitions']; shards = index['shards']
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
            stats = {'planning_seconds':time.monotonic()-planning_started,'prepare_seconds':0.0,'compile_seconds':0.0,'publication_seconds':0.0,
                     'prepared_array_bytes':0,'output_array_bytes':0,'core_peak_working_bytes':0,
                     'core_calls':0,'core_seconds':0.0, 'computed_output_cells':0, 'hit_columns':len({(h['session'],h['column']) for h in coverage['hits']}),
                     'hit_security_blocks':len(coverage['hits']),
                     'gap_columns':len(coverage['gaps']), 'recovered_partitions':len(recovered),
                     'coverage_date_writes':0, 'coverage_root_writes':0,
                     'maximum_execution_source_ids':0,
                     'admission_file_hash_calls':admission_metrics['file_hash_calls'],
                     'admission_hash_bytes':admission_metrics['hash_bytes']}
            compiled = self._compiled; by_day={day:{} for day in coverage['sessions']}
            parts_by_day={}
            for part in partitions: parts_by_day.setdefault(part['sessions'][0],[]).append(part)
            dirty=set(recovered) | (set(parts_by_day)-set(shards))
            for day in sorted(dirty):
                shards[day]=write_date(self.root,day,parts_by_day[day])
                stats['coverage_date_writes']+=1
            final_hits={}
            for hit in coverage['hits']: final_hits.setdefault((hit['session'],hit['column']),[]).append(hit)
            for gap in coverage['gaps']:
                by_day[gap['session']].setdefault(tuple(gap['security_ids']),[]).append(gap['column'])
            jobs=[(day,list(axis),columns) for day,groups in by_day.items() for axis,columns in groups.items()]
            gap_scopes={}
            for day,axis,columns in jobs:
                scope=gap_scopes.setdefault(tuple(axis),{'days':set(),'columns':set()})
                scope['days'].add(day); scope['columns'].update(columns)
            gap_bindings={axis:(coverage['dependencies'] if list(axis)==coverage['security_ids'] else source_adapter.dependency_refs(
                {c:coverage['definitions'][c] for c in coverage['definitions'] if c in scope['columns']},
                sorted(scope['days']),list(axis))) for axis,scope in gap_scopes.items()}
            selected_plans={}; admitted_plans={}
            for day,securities,columns in jobs:
                selection_key=(tuple(columns),tuple(securities))
                if selection_key not in selected_plans:
                    selected=select_feature_plan(request.plan,columns)
                    if securities!=coverage['security_ids']:selected=select_security_plan(selected,securities)
                    selected_plans[selection_key]=(selected,selected.to_dict())
                plan,before=selected_plans[selection_key]
                bindings = {c:gap_bindings[tuple(securities)][day][c] for c in columns}
                if hasattr(source_adapter,'execution_plan'):
                    actual=source_adapter.execution_plan(plan,day,securities,bindings)
                    key=(selection_key,actual.identity)
                    if key not in admitted_plans:
                        after=actual.to_dict()
                        require({k:v for k,v in before.items() if k!='sources'}==
                                {k:v for k,v in after.items() if k!='sources'},
                                'execution source projection changed Feature semantics')
                        admitted_plans[key]=(actual,len(after['sources']))
                    plan,source_count=admitted_plans[key]
                else:source_count=len(before['sources'])
                stats['maximum_execution_source_ids']=max(stats['maximum_execution_source_ids'],source_count)
                prepare_started=time.monotonic()
                inputs = source_adapter.prepare(plan, day, securities, bindings)
                stats['prepare_seconds']+=time.monotonic()-prepare_started
                stats['prepared_array_bytes']+=sum(a.nbytes for name in ('values','validity','reason_codes','available_at_utc_us','source_codes') for a in inputs[name].values())
                require(inputs['input_refs'].get('feature_dependency_ref') == digest(bindings),
                        'prepared inputs do not bind planned dependencies')
                key = plan.identity
                compile_started=time.monotonic()
                if key not in compiled: compiled[key] = compile_feature_columns(plan)
                stats['compile_seconds']+=time.monotonic()-compile_started
                core_started=time.monotonic()
                result = execute_feature_columns(compiled[key], inputs=inputs, output_sessions=[day],
                    max_working_bytes=self.limits['maximum_matrix_bytes'])
                stats['core_seconds']+=time.monotonic()-core_started
                stats.setdefault('core_work',[]).append({'session':day,**{k:result['stats'][k] for k in ('shared_work','top_nodes')}})
                stats['core_peak_working_bytes']=max(stats['core_peak_working_bytes'],result['stats']['peak_working_bytes'])
                publication_started=time.monotonic()
                arrays, header = _capture_result(result, day=day, columns=columns,
                    securities=securities, maximum_bytes=self.limits['maximum_matrix_bytes'])
                stats['output_array_bytes']+=sum(a.nbytes for a in arrays.values())
                stats['core_calls'] += 1; stats['computed_output_cells'] += len(securities)*len(columns)
                header_descriptor=write_part(self.root,seal({'core_header':header},'header_ref'),'header_ref')
                width=self.column_block or len(columns)
                for start in range(0,len(columns),width):
                    stop=min(start+width,len(columns)); block_columns=columns[start:stop]
                    block={name:value[:,:,start:stop] for name,value in arrays.items()}; buffers={}; block_digests={}
                    for name,dtype in ARRAYS.items():
                        array=block[name]
                        block_digests[name]=(header['output_buffers'][name] if start==0 and stop==len(columns) else
                            {'shape':list(array.shape),'dtype':array.dtype.str,'sha256':sha256(array.tobytes(order='C')).hexdigest()})
                        if name in ('reason_codes','source_codes'): buffers[name]=_write_codes(self.root,array)
                        else:
                            value=array.view('u1') if dtype=='bool_u8' else array
                            buffers[name]=write_buffer(self.root,value.reshape(-1),dtype=dtype,shape=list(value.shape))
                    projection=seal({'contract_version':'feature_output_projection_v1',
                        'original_result_ref':header['result_ref'],'column_indices':list(range(start,stop)),
                        'output_buffers':block_digests},'projection_ref')
                    part=seal({'contract_version':PARTITION_VERSION,'sessions':[day],
                        'security_ids':securities,'columns':block_columns,
                        'schema':[coverage['definitions'][c]['schema'] for c in block_columns],
                        'semantic_refs':{c:coverage['definitions'][c]['semantic_ref'] for c in block_columns},
                        'dependency_refs':{c:bindings[c] for c in block_columns},'buffers':buffers,
                        'column_source_bindings':_column_sources(header,block['source_codes'],block_columns),
                        'core_header':header_descriptor,'core_output_projection':projection},'partition_ref')
                    descriptor=write_part(self.root,part,'partition_ref')
                    records=self.root/'partitions'; records.mkdir(exist_ok=True)
                    record=records/(part['partition_ref'][7:]+'.json')
                    try: os.link(descriptor['path'],record)
                    except FileExistsError: require(file_digest(record)==descriptor['file_digest'],'Feature partition conflict')
                    descriptor['path']=str(record); summary=coverage_summary(descriptor,part); partitions.append(summary)
                    parts_by_day.setdefault(day,[]).append(summary)
                    for column in block_columns:
                        final_hits.setdefault((day,column),[]).append({'session':day,'column':column,
                            'partition':summary,'security_ids':securities})
                    if progress is not None:
                        progress({'stage':'partition_sealed','session':day,
                            **{k:deepcopy(v) for k,v in stats.items() if k!='core_work'},
                            'core_work':deepcopy(stats.get('core_work',[])[-1:])})
                shards[day]=write_date(self.root,day,parts_by_day[day])
                stats['coverage_date_writes']+=1
                stats['publication_seconds']+=time.monotonic()-publication_started
                del result, arrays, inputs, block, array, value
            require(len(final_hits)==len(coverage['sessions'])*len(coverage['definitions']),
                    'Feature coverage remains incomplete')
            if self._session is not None: self._session['store'].validate_boundary()
            view = compact_view(coverage, final_hits)
            descriptor = write_part(self.root, view, 'view_ref')
            self._publish_index(shards)
            cached=self._coverage_cache[1]
            changed=set(coverage['sessions'])|set(recovered)
            retained=[p for p in cached['partitions'] if p['sessions'][0] not in changed]
            self._coverage_cache=(self._epoch(self.root/'index.json'),
                {'partitions':retained+list(partitions),'shards':dict(shards),'loaded_days':cached['loaded_days']|changed})
            self._recovered_directory_mark=self._epoch(self.root/'partitions')
            stats.update(self.coverage_metrics)
            stats['coverage_root_writes']=1
            if self._session is not None:
                stats['admission_file_hash_calls']=admitted.metrics['file_hash_calls']-baseline['file_hash_calls']
                stats['admission_hash_bytes']=admitted.metrics['hash_bytes']-baseline['hash_bytes']
            return descriptor, stats

    def open_feature_view(self, descriptor):
        """Open a fixed saved view. Reading never calls Data/Core or modifies index."""
        return FeatureView(descriptor,_session=self._session, **self.limits)

    def compose_views(self, views, *, output_sessions, security_ids, columns, sample_universe=None,
                      preserve_reference=False):
        """Save an exact, metadata-only selection of admitted saved views.

        Intersect each parent with the requested grid; overlapping parents or
        gaps fail, rather than choosing a source implicitly. Columns may come
        from different libraries; every parent must cover all requested dates. Original partition receipts remain
        unchanged. preserve_reference explicitly permits reading a subset of a
        frozen reference-dependent result, retaining its original full partition
        and definition. It never recomputes or relabels the original cohort.
        No coverage index, array buffer or Core computation is written here.
        Source/PIT admission remains the consuming adapter's responsibility.
        """
        days=_ordered(output_sessions,'output sessions')
        securities=_ordered(security_ids,'physical securities')
        sample=securities if sample_universe is None else _ordered(sample_universe,'sample securities')
        columns=list(columns)
        require(columns and len(columns)==len(set(columns)) and set(sample)<=set(securities),
                'unique columns and sample subset required')
        definitions={};hits={};parents=list(views)
        for view in parents:
            cap=view.describe()
            require(set(days)<=set(cap['sessions']),
                    'date stitching is not supported; each parent must cover all requested sessions')
            for day in days:
                names=[n for n in columns if n in cap['columns']]
                if not names:continue
                receipts=view.column_receipts(day,names)
                for name in names:
                    definition=cap['definitions'][name]
                    require(name not in definitions or definitions[name]==definition,
                            'composed Feature definitions differ')
                    definitions[name]=definition
                    for block in receipts[name]['blocks']:
                        selected=sorted(set(securities)&set(block['selected_security_ids']))
                        if not selected:continue
                        require(not definition['reference_dependent'] or preserve_reference is True or
                                selected==block['security_ids']==securities,
                                'reference-dependent Feature cannot change physical cohort')
                        previous=hits.setdefault((day,name),[])
                        require(not any(set(selected)&set(p['security_ids']) for p in previous),
                                'overlapping composed Feature cells')
                        previous.append({'partition':block['partition'],'security_ids':selected})
        require(set(definitions)==set(columns) and all(
            sum(len(b['security_ids']) for b in hits.get((day,name),[]))==len(securities)
            for day in days for name in columns),'incomplete composed Feature coverage')
        coverage={'sessions':days,'security_ids':securities,'sample_universe':sample,
                  'definitions':{n:definitions[n] for n in columns}}
        wire=compact_view(coverage,hits)
        for view in parents:view.validate_boundary()
        return write_part(self.root,wire,'view_ref')


class FeatureView:
    def __init__(self, descriptor, _session=None, **limits):
        self._own_store=_session is None
        self._store = VerifiedMatrixStore(**limits) if _session is None else _session['store']
        self._view = self._store.read_json(descriptor, 'view_ref')
        require(self._view['contract_version'] in (VIEW_V1,VIEW_V2,VIEW_VERSION), 'unsupported Feature view')
        self.descriptor = deepcopy(descriptor)
        self._loaded = {} if _session is None else _session['arrays']
        self._parts={}; self._tiles={}; self._legacy=self._view['contract_version']==VIEW_V1
        if self._legacy:
            selections=({'session':t['session'],'columns':[t['column']], 'partition':t['partition']}
                        for t in self._view['tiles'])
        else:
            selections=self._view['blocks']
        physical=_ordered(self._view['security_ids'],'view securities'); physical_set=frozenset(physical)
        self._security_set=physical_set;self._day_set=frozenset(self._view['sessions'])
        axes=self._view.get('security_axes',{})
        require(self._view['contract_version']==VIEW_VERSION or not axes,'legacy view cannot declare security blocks')
        require(all(digest(axis)==ref for ref,axis in axes.items()),'view security axis identity mismatch')
        axis_sets={ref:frozenset(_ordered(axis,'view block securities')) for ref,axis in axes.items()}
        for block in selections:
            axis_ref=block.get('security_axis_ref')
            require(self._view['contract_version']==VIEW_VERSION or axis_ref is None,
                    'legacy view cannot select security blocks')
            require(axis_ref is None or axis_ref in axes,'unknown view security axis')
            selected=physical if axis_ref is None else axes[axis_ref]
            selected_set=physical_set if axis_ref is None else axis_sets[axis_ref]
            require(selected_set<=physical_set,'Feature block outside view')
            selection={'partition':block['partition'],'security_ids':selected,'security_set':selected_set}
            for column in block['columns']:
                key=block['session'],column
                previous=self._tiles.setdefault(key,[])
                require(not any(selected_set & old['security_set'] for old in previous),
                        'duplicate Feature view cell selection')
                previous.append(selection)
        require(set(self._tiles)=={(d,c) for d in self._view['sessions'] for c in self._view['columns']} and
                set(self._view['definitions'])==set(self._view['columns']), 'incomplete Feature view selection')
        require(all(sum(len(b['security_ids']) for b in blocks)==len(physical)
                    for blocks in self._tiles.values()), 'incomplete Feature security coverage')

    def _partition(self, day, column, *, block=None, store=None, parts=None):
        store=self._store if store is None else store
        parts=self._parts if parts is None else parts
        block=self._tiles[day,column][0] if block is None else block
        desc=block['partition']; ref=desc['partition_ref']
        if ref not in parts:
            proof=partition_descriptor(desc)
            part=store.read_json(proof,'partition_ref')
            require(part['contract_version']==PARTITION_VERSION and part['sessions']==[day] and
                    set(block['security_ids'])<=set(part['security_ids']) and
                    set(part['semantic_refs'])==set(part['dependency_refs'])==set(part['columns']) and
                    all(is_ref(r) for r in part['semantic_refs'].values()) and
                    all(is_ref(r) for r in part['dependency_refs'].values()), 'view partition coverage mismatch')
            if 'path' in part['core_header']:
                part={**part,'core_header':store.read_json(part['core_header'],'header_ref')['core_header']}
            parts[ref]=part
        else: store.check_path(partition_descriptor(desc)['path'])
        part=parts[ref]
        require(self._view['contract_version']==VIEW_VERSION or part['security_ids']==self._view['security_ids'],
                'legacy view partition coverage mismatch')
        require(set(block['security_ids'])<=set(part['security_ids']) and
                part['sessions']==[day] and column in part['columns'] and
                part['semantic_refs'][column]==self._view['definitions'][column]['semantic_ref'],
                'view Feature semantics mismatch')
        if self._legacy:
            require(FeatureLibrary._summary(partition_descriptor(desc),part)==desc,
                    'coverage summary differs from sealed partition')
        return part

    def column_receipts(self, session, columns=None):
        """Read original per-column semantics, dependencies and sources, without Core.

        The compact selection only points to immutable partitions. Actual parent
        bytes and content refs are admitted before returning their receipts.
        ``blocks`` carries each original dependency and its full/selected axes.
        A single block retains the legacy ``partition`` and dependency ref; a
        multi-block dependency ref binds the composition, so adapters must check
        its block receipts against the source before deriving full-axis refs.
        Source eligibility/PIT remains the source adapter's responsibility.
        """
        with self.operation():
            columns=list(self._view['columns'] if columns is None else columns)
            require(bool(columns) and len(columns)==len(set(columns)) and
                    all((session,c) in self._tiles for c in columns), 'Feature receipt selection outside view')
            cached=all(b['partition']['partition_ref'] in self._parts for c in columns for b in self._tiles[session,c])
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
                    receipts=[]; bindings={}
                    for block in self._tiles[session,column]:
                        part=self._partition(session,column,block=block,store=store,parts=parts)
                        receipts.append({'semantic_ref':part['semantic_refs'][column],
                            'dependency_ref':part['dependency_refs'][column],
                            'security_ids':deepcopy(part['security_ids']),
                            'selected_security_ids':deepcopy(block['security_ids']),
                            'partition':partition_descriptor(block['partition'])})
                        for binding in part['column_source_bindings'][column]:
                            require(binding['id'] not in bindings or bindings[binding['id']]==binding,
                                    'conflicting source ID')
                            bindings[binding['id']]=binding
                    result[column]={'semantic_ref':receipts[0]['semantic_ref'],
                        'dependency_ref':receipts[0]['dependency_ref'] if len(receipts)==1 else digest(receipts),
                        'source_bindings':[bindings[k] for k in sorted(bindings)],
                        'blocks':receipts}
                    if len(receipts)==1: result[column]['partition']=receipts[0]['partition']
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
            require(bool(keys) and len(set(keys)) == len(keys) and all(s in self._security_set and d in self._day_set for s,d in keys),
                    'unique keys inside fixed Feature view required')
            shape = len(keys), len(columns)
            require(len(keys)*len(columns)*50 <= self._store.maximum_matrix_bytes, 'Feature projection budget exceeded')
            out = {name:np.zeros(shape, dtype='bool' if name == 'validity' else 'float64' if name == 'values'
                               else 'int64' if name == 'available_at_utc_us' else 'int32') for name in ARRAYS}
            reasons = [None]; source_sets = [[]]; bindings = {}; used = {}
            targets={}
            for i,(_,day) in enumerate(keys): targets.setdefault(day,[]).append(i)
            tiles=self._tiles;row_selections={}
            for day,target in targets.items():
                for j, column in enumerate(columns):
                    for selection in tiles[day,column]:
                        desc = selection['partition']; ref = desc['partition_ref']
                        block_target=[i for i in target if keys[i][0] in selection['security_set']]
                        if not block_target: continue
                        part = self._partition(day,column,block=selection)
                        if ref not in used:
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
                        row_key=(day,ref,tuple(block_target))
                        if row_key not in row_selections:row_selections[row_key]=[axis[keys[i][0]] for i in block_target]
                        rows = row_selections[row_key]
                        for name in ARRAYS:
                            values = arrays[name][0,rows,col]
                            if name == 'reason_codes': values = reason_map[values]
                            if name == 'source_codes': values = source_map[values]
                            out[name][block_target,j] = values
            selected=sorted(set(int(code) for code in np.unique(out['source_codes'])))
            remap=np.zeros(len(source_sets),dtype='int32')
            for code,old in enumerate(selected): remap[old]=code
            source_sets=[source_sets[old] for old in selected]
            out['source_codes']=remap[out['source_codes']]
            actual_sources={source for group in source_sets for source in group}
            for name, array in out.items(): out[name]=np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(shape)
            # The enclosing operation checks consumed paths at its outer return.
            # A borrowed projection must not rescan the growing parent operation.
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
