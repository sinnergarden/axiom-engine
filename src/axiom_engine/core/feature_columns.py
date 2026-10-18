"""Bounded packed transport for the existing Core FeaturePlan arithmetic.

No Data selection, account, provider, model or artifact reads happen here.
Each call is one explicitly prepared PIT/anchor view, with one immutable snapshot.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter_ns
from types import MappingProxyType

from .contracts import FeaturePlan, digest, fields, integer, require, session, text
from .plan import CS, WINDOW, required_history, validate_plan, _required_history
from . import execution as kernel
from .._implementation import IMPLEMENTATION_REF

_FIELDS = ('contract_version calendar_ref reference_ref schema source_bindings sessions '
           'security_ids cutoffs_utc_us values validity reason_codes reason_dictionary '
           'available_at_utc_us source_codes source_sets reference_member reference_industry '
           'reference_available_at_utc_us reference_source_codes input_refs')
_NONE_TIME = -(2**63)
_MIN_TIME = -62135596800000000  # 0001-01-01 UTC, microseconds since Unix epoch
_MAX_TIME = 253402300799000000  # 9999-12-31T23:59:59 UTC


def _ref(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _plain(v) for k, v in value.items()}
    if type(value) is tuple:
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class CompiledFeatureColumns:
    plan_ref: str
    compiled_ref: str
    _plan: object
    _history: object
    _last_use: object
    _sessions: tuple


def compile_feature_columns(plan: FeaturePlan):
    """Validate the original immutable DAG once; no expression translation."""
    p = validate_plan(plan, execution=True)
    require(p['observation_domain'] == 'sessions' and not p['event_schema'],
            'COLUMN_PROFILE: complete sessions domain without events required')
    require(all(c['dtype'] in ('float64', 'bool') for c in p['input_schema']),
            'COLUMN_PROFILE: only float64/bool inputs')
    require(len(p['sources']) <= 64, 'COLUMN_PROFILE: at most 64 bound source IDs')
    for n in p['nodes']:
        require(n['op'] not in ('asof', 'calendar_age') and
                n['column']['dtype'] in ('float64', 'bool'), 'COLUMN_PROFILE: unsupported node')
    last = {c['name']: -1 for c in p['input_schema']}
    for i, n in enumerate(p['nodes']):
        last[n['name']] = i
        for a in n['inputs']:
            last[a] = i
    identity = _ref({'contract_version': 'compiled_feature_columns_v1',
                     'plan_ref': plan.identity, 'implementation_ref': IMPLEMENTATION_REF,
                     'profile': 'sessions_float64_bool_sources64_v1'})
    return CompiledFeatureColumns(plan.identity, identity, _freeze(p),
                                  _freeze(_required_history(p)), _freeze(last),
                                  tuple(sorted(p['reference_members'])))


class _Budget:
    def __init__(self, limit):
        self.limit, self.live, self.peak = limit, 0, 0

    def add(self, count):
        require(self.live + count <= self.limit, 'COLUMN_WORKING_BYTES_EXCEEDED')
        self.live += count
        self.peak = max(self.peak, self.live)

    def drop(self, count):
        self.live -= count


def _snapshot(np, value, dtype, shape, name, budget, descriptors):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and
            value.shape == shape, 'COLUMN_SHAPE_DTYPE: ' + name)
    budget.add(value.nbytes)
    # bytes owns the admitted values: a caller cannot reopen a writable alias.
    raw = value.tobytes(order='C')
    descriptors[name] = {'shape': list(shape), 'dtype': np.dtype(dtype).str,
                         'sha256': hashlib.sha256(raw).hexdigest()}
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _time(np, value, cutoff=False):
    require(bool(np.all((value >= _MIN_TIME) & (value <= _MAX_TIME))), 'COLUMN_UTC_RANGE')
    seconds = value // 1000000
    if not cutoff:
        seconds = seconds + (value % 1000000 != 0)
    require(bool(np.all(seconds <= _MAX_TIME // 1000000)), 'COLUMN_UTC_RANGE')
    return seconds * 1000000


class _Column:
    def __init__(self, np, rows, width, dtype, budget):
        self.rows = tuple(sorted(rows))
        self.index = {r: i for i, r in enumerate(self.rows)}
        self.dtype = dtype
        shape = len(self.rows), width
        self.bytes = len(self.rows) * width * 30
        budget.add(self.bytes)
        self.value = np.zeros(shape, dtype='float64')
        self.valid = np.zeros(shape, dtype='bool')
        self.available = np.full(shape, _NONE_TIME, dtype='int64')
        self.source = np.zeros(shape, dtype='uint64')
        self.issue = np.zeros(shape, dtype='bool')
        self.reason = np.zeros(shape, dtype='int32')

    def at(self, row):
        return self.index[row]

    def numeric(self, row, security):
        i = self.index[row]
        if not self.valid[i, security]:
            return None
        v = self.value[i, security]
        return bool(v) if self.dtype == 'bool' else float(v)


def _needed(p, outputs):
    needed = {c['name']: set() for c in p['input_schema']}
    needed.update({n['name']: set() for n in p['nodes']})
    for o in p['outputs']:
        needed[o['node']].update(outputs)
    for n in reversed(p['nodes']):
        rows, op, q = needed[n['name']], n['op'], n['params']
        deps = set()
        for r in rows:
            if op in ('shift', 'pct_change'):
                previous = r - q['periods']
                require(previous >= 0 or p['history_policy'] != 'full', 'INSUFFICIENT_HISTORY')
                if previous >= 0:
                    deps.add(previous)
                if op == 'pct_change':
                    deps.add(r)
            elif op in ('rolling', 'rolling_industry') or op in WINDOW:
                end = r + int(q['inclusive_current'])
                start = end - q['window']
                require(start >= 0 or p['history_policy'] != 'full', 'INSUFFICIENT_HISTORY')
                require(end > 0 or op in WINDOW, 'COLUMN_PROFILE: empty exclusive rolling window')
                deps.update(range(max(0, start), end))
            else:
                deps.add(r)
        for a in n['inputs']:
            needed[a].update(deps)
    return needed


def _window_workspace_bytes(rows, securities, parents):
    """Conservative temporary reservation for the new window families only.

    Retained value/validity gathers use 9 bytes per parent cell, while the
    largest transient provenance gather uses 8 bytes per cell. Per row, 1 KiB
    covers indices and the one-security Python vectors, tuples, sorting and
    centered regression/correlation intermediates (including list replacement).
    Array/list headers and reduction vectors have separate fixed/width margins.
    This is an allocation bound, not a process RSS measurement.
    """
    return rows * ((9 * parents + 8) * securities + 1024) + 128 * securities + 4096


def _sorted_quantile(values, fraction):
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi]-values[lo])*(position-lo)


def _sliding_extrema(np, source, rows, window, inclusive, maximum, width):
    """One anchored input; strict eviction preserves first ties/signed zero."""
    from collections import deque
    result = {}
    queues = [deque() for _ in range(width)]
    cursor = max(0, rows[0]+int(inclusive)-window)
    for row in rows:
        end = row+int(inclusive); start = max(0,end-window)
        for day in range(max(cursor,start),end):
            pi = source.at(day)
            for security, queue in enumerate(queues):
                if source.valid[pi,security]:
                    value = float(source.value[pi,security])
                    while queue and (queue[-1][1] < value if maximum else queue[-1][1] > value):
                        queue.pop()
                    queue.append((day,value))
        cursor = end
        values = np.zeros(width,dtype='float64'); valid = np.zeros(width,dtype='bool')
        for security,queue in enumerate(queues):
            while queue and queue[0][0] < start: queue.popleft()
            if queue:values[security]=queue[0][1];valid[security]=True
        result[row] = values,valid
    return result


def _window_family(np, nodes, parents, needed, width, budget, reason_ids, counters):
    """Share array windows; convert only exact ordered scalar reductions to lists."""
    results={n['name']:_Column(np,needed[n['name']],width,n['column']['dtype'],budget) for n in nodes}
    rows=sorted(set().union(*(needed[n['name']] for n in nodes)))
    q0=nodes[0]['params'];window=q0['window'];inclusive=q0['inclusive_current']
    extremes={};extreme_bytes=0;counts_prefix=None
    for maximum in (False,True):
        targets=[n for n in nodes if n['op']=='rolling' and n['params']['reduction']==('max' if maximum else 'min') and len(needed[n['name']])>1]
        if targets:
            target_rows=sorted(set().union(*(needed[n['name']] for n in targets)))
            charge=(len(target_rows)+window)*width*96
            budget.add(charge);extreme_bytes+=charge
            extremes[maximum]=_sliding_extrema(np,parents[0],target_rows,window,inclusive,maximum,width)
    if extremes:
        prefix_bytes=(len(parents[0].rows)+1)*width*4
        budget.add(prefix_bytes);extreme_bytes+=prefix_bytes
        counts_prefix=np.zeros((len(parents[0].rows)+1,width),dtype='int32')
        np.cumsum(parents[0].valid,axis=0,dtype='int32',out=counts_prefix[1:])
    security_indices=np.arange(width)
    for r in rows:
        active=[n for n in nodes if r in needed[n['name']]]
        start=max(0,r+int(inclusive)-window);end=r+int(inclusive)
        workspace=_window_workspace_bytes(end-start,width,len(parents));budget.add(workspace)
        counters['window_gathers']+=1
        owner=results[active[0]['name']];owner_i=owner.at(r)
        indices=[]
        for parent in parents:
            ix=[parent.at(d) for d in range(start,end)];indices.append(ix)
            if ix:
                owner.available[owner_i]=np.maximum(owner.available[owner_i],np.max(parent.available[ix],axis=0))
                owner.source[owner_i]|=np.bitwise_or.reduce(parent.source[ix],axis=0)
                owner.issue[owner_i]|=np.any(parent.issue[ix],axis=0)
        # Extrema-only dates do not gather/box numeric windows at all.
        numeric=[n for n in active if not (n['op']=='rolling' and n['params']['reduction'] in ('min','max') and r in extremes.get(n['params']['reduction']=='max',{}))]
        windows=[]
        if numeric:
            counters['numeric_window_gathers']=counters.get('numeric_window_gathers',0)+1
            windows=[(parent.value[ix],parent.valid[ix]) for parent,ix in zip(parents,indices)]
            counts=np.count_nonzero(windows[0][1],axis=0)
        else:
            ix=indices[0]
            counts=counts_prefix[ix[-1]+1]-counts_prefix[ix[0]] if ix else np.zeros(width,dtype='int32')
        tasks={}
        for n in active:
            out=results[n['name']];oi=out.at(r);q=n['params'];op=n['op']
            if out is not owner:
                out.available[oi]=owner.available[owner_i];out.source[oi]=owner.source[owner_i];out.issue[oi]=owner.issue[owner_i]
            actual_counts=(np.count_nonzero(windows[0][1]&windows[1][1],axis=0) if op=='rolling_corr' else counts)
            ok=(actual_counts>=q['min_periods'])
            if q['missing']=='propagate':ok&=actual_counts==end-start
            out.reason[oi]=np.where(ok,reason_ids['MISSING'],reason_ids['WINDOW_MISSING'])
            if not np.any(ok):continue
            reduction=q.get('reduction')
            if op=='rolling' and reduction in ('min','max') and r in extremes.get(reduction=='max',{}):
                values,valid=extremes[reduction=='max'][r]
                out.value[oi]=values;out.valid[oi]=ok&valid
                continue
            if op=='rolling' and reduction in ('sum','mean'):kind='sum'
            elif op=='rolling_quantile' or op=='rolling' and reduction=='median':kind='sorted'
            elif op=='rolling_regression':kind='regression'
            elif op=='rolling_idx':kind=('extreme',q['extreme'])
            elif op=='rolling' and reduction in ('min','max'):kind=('extreme',reduction)
            elif op=='rolling':kind=(reduction,q.get('ddof'))
            else:kind='corr'
            tasks.setdefault(kind,[]).append((n,out,oi,ok))
        for kind,targets in tasks.items():
            values,valid=windows[0]
            eligible=np.logical_or.reduce([t[3] for t in targets])
            if isinstance(kind,tuple) and kind[0]=='extreme':
                maximum=kind[1]=='max'
                masked=np.where(valid,values,-np.inf if maximum else np.inf)
                arg=(np.argmax if maximum else np.argmin)(masked,axis=0)
                extreme=values[arg,security_indices]
                last=None
                for n,out,oi,ok in targets:
                    if n['op']=='rolling_idx':
                        chosen=arg
                        if n['params']['ties']=='last':
                            if last is None:last=len(masked)-1-(np.argmax if maximum else np.argmin)(masked[::-1],axis=0)
                            chosen=last
                        out.value[oi]=chosen+1
                    else:out.value[oi]=extreme
                    out.valid[oi]=ok
                del masked,arg,extreme,last
            elif isinstance(kind,tuple) and kind[0]=='rank':
                last=values[-1]
                less=np.count_nonzero((values<last)&valid,axis=0)
                equal=np.count_nonzero((values==last)&valid,axis=0)
                ranked=np.divide(less+(equal+1)/2,counts,out=np.zeros(width),where=counts!=0)
                for n,out,oi,ok in targets:out.value[oi]=ranked;out.valid[oi]=ok&valid[-1]
            else:
                for security in np.flatnonzero(eligible):
                    # Ordered fsum/statistics retain their original numerical algorithms.
                    present=values[:,security][valid[:,security]].tolist()
                    if kind=='sum':number=math.fsum(present);counters['window_fsums']+=1
                    elif kind=='sorted':ordered=sorted(present);counters['window_sorts']+=1
                    elif kind=='regression' or kind=='corr':
                        columns=[]
                        for vals,known in windows:
                            column=vals[:,security].tolist()
                            for missing in np.flatnonzero(~known[:,security]):column[missing]=None
                            columns.append(column)
                        if kind=='regression':regression=kernel._regression(tuple(columns[0]));counters['regression_solves']+=1
                    else:number=kernel._std(present,kind[1])
                    for n,out,oi,ok in targets:
                        if not ok[security]:continue
                        q=n['params'];value=None
                        if kind=='sum':value=number/int(counts[security]) if q['reduction']=='mean' else number
                        elif kind=='sorted':
                            if n['op']=='rolling_quantile':value=_sorted_quantile(ordered,q['quantile'])
                            else:
                                mid=len(ordered)//2;value=ordered[mid] if len(ordered)%2 else (ordered[mid-1]+ordered[mid])/2
                        elif kind=='regression':
                            value=regression[('slope','rsquare','resi').index(q['output'])]
                            if value is None and (q['output']=='rsquare' or len(present)<2):require(q['zero']=='missing','ZERO_VARIANCE')
                        elif kind=='corr':value,_=kernel._window_value('rolling_corr',columns,q,kernel._regression)
                        else:value=number
                        if value is not None:out.value[oi,security]=value;out.valid[oi,security]=True
        windows.clear();values=valid=None
        del windows,indices,tasks
        budget.drop(workspace)
    extremes.clear();counts_prefix=None;budget.drop(extreme_bytes)
    return results


_BATCH_ELEMENT={'constant','identity','is_missing','fill','gt','lt','eq','and','or','not',
                'divide','add','sub','mul','abs','to_float','clip','where'}


def _element_row(np, op, q, parents, row, out, oi, reason_ids):
    """Original scalar order over one security row; parent metadata is merged once."""
    values=[parent.value[parent.at(row)] for parent in parents]
    known=[parent.valid[parent.at(row)] for parent in parents]
    target=out.value[oi];valid=out.valid[oi]
    if op=='constant':
        if q['value'] is not None:target.fill(q['value']);valid.fill(True)
    elif op=='identity':
        target[:]=values[0];valid[:]=known[0]
        out.reason[oi]=parents[0].reason[parents[0].at(row)]
        return
    elif op=='is_missing':target[:]=~known[0];valid.fill(True)
    elif op=='fill':
        target[:]=values[0];valid[:]=known[0]
        if q['value'] is not None:
            np.copyto(target,q['value'],where=~known[0]);valid.fill(True)
    elif op=='where':
        require(q['missing']!='reject' or bool(np.all(known[0])),'MISSING_BOOLEAN')
        chosen=(values[0]!=0)&known[0]
        # Both branches already exist in the original DAG; select without arithmetic.
        target[:]=values[2];np.copyto(target,values[1],where=chosen)
        valid[:]=np.where(chosen,known[1],known[2])
        if q['missing']=='preserve':valid&=known[0]
    elif op in ('gt','lt','eq','and','or','not'):
        present=known[0].copy()
        for mask in known[1:]:present&=mask
        require(q['missing']!='reject' or bool(np.all(present)),'MISSING_BOOLEAN')
        if op in ('gt','lt','eq'):
            compare=np.greater if op=='gt' else np.less if op=='lt' else np.equal
            target[:]=compare(values[0],values[1])&present
        else:
            left=(values[0]!=0)&known[0]
            if op=='not':target[:]=~left
            elif op=='and':target[:]=left&((values[1]!=0)&known[1])
            else:target[:]=left|((values[1]!=0)&known[1])
        valid[:]=present if q['missing']=='preserve' else True
    else:
        valid[:]=known[0]
        for mask in known[1:]:valid&=mask
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            if op=='divide':
                zero=valid&(values[1]==0)
                require(q['zero']=='missing' or not bool(np.any(zero)),'ZERO_DENOMINATOR')
                valid&=~zero;np.divide(values[0],values[1],out=target,where=valid)
            elif op in ('add','sub','mul'):
                operation=np.add if op=='add' else np.subtract if op=='sub' else np.multiply
                operation(values[0],values[1],out=target,where=valid)
            elif op=='abs':np.absolute(values[0],out=target,where=valid)
            elif op=='to_float':np.copyto(target,values[0],where=valid)
            elif op=='clip':
                np.copyto(target,values[0],where=valid)
                # Python max/min keep their first argument on equal signed zeros.
                if q['lower'] is not None:np.copyto(target,q['lower'],where=valid&(target<q['lower']))
                if q['upper'] is not None:np.copyto(target,q['upper'],where=valid&(target>q['upper']))
    out.reason[oi]=np.where(valid,0,reason_ids['MISSING'])


def execute_feature_columns(compiled, *, inputs, output_sessions, max_working_bytes):
    """One admitted view, many output sessions; result has packed identity.

    Values are [output_session, security, feature], bool schema uses 0/1.
    Missing values use validity=False and canonical value=0. Time null uses
    int64 minimum; all other output times are UTC microseconds on whole seconds.
    Source/reason code zero is respectively the empty set/no missing reason.
    max_working_bytes bounds owned numeric arrays and reserves the new window
    families' temporary workspace; it does not bound interpreter/process RSS.
    """
    import numpy as np
    begin_ns = perf_counter_ns()
    require(type(compiled) is CompiledFeatureColumns, 'Expected compiled Feature columns')
    integer(max_working_bytes, 1)
    fields(inputs, _FIELDS + ' ' + ' '.join(k for k in ('reference_member_known', 'reference_pool_complete', 'field_rows') if k in inputs))
    require(inputs['contract_version'] == 'feature_column_input_v1', 'Column input version')
    p = compiled._plan
    for name in ('calendar_ref', 'reference_ref'):
        require(inputs[name] == p[name], 'Column ' + name + ' binding mismatch')
    require(inputs['schema'] == _plain(p['input_schema']) and
            inputs['source_bindings'] == _plain(p['sources']), 'Column schema/source binding mismatch')
    days, securities = inputs['sessions'], inputs['security_ids']
    require(type(days) is list and bool(days), 'Column sessions required')
    for day in days:
        session(day)
    require(days == sorted(set(days)), 'Column sessions must be unique/ordered')
    require(days[0] in p['reference_members'], 'MISSING_SESSION: unbound calendar start')
    start = compiled._sessions.index(days[0])
    require(tuple(days) == compiled._sessions[start:start + len(days)],
            'MISSING_SESSION: input must be a contiguous frozen-calendar slice')
    require(type(securities) is list and bool(securities), 'Column security IDs required')
    for security in securities:
        text(security)
    require(securities == sorted(set(securities)), 'Column security IDs must be unique/ordered')
    days, securities = tuple(days), tuple(securities)
    require(type(output_sessions) is list and bool(output_sessions) and
            output_sessions == sorted(set(output_sessions)) and set(output_sessions) <= set(days),
            'Column output sessions must be ordered subset')
    out_days = tuple(output_sessions)
    positions = {d: i for i, d in enumerate(days)}
    output_rows = tuple(positions[d] for d in out_days)
    needed = _needed(p, output_rows)
    last_use = compiled._last_use
    if any(not needed[n['name']] for n in p['nodes']):
        last_use = {name: -1 for name in needed}
        for i, n in enumerate(p['nodes']):
            if needed[n['name']]:
                last_use[n['name']] = i
                for a in n['inputs']:
                    last_use[a] = i
    metadata = json.loads(json.dumps({k: inputs[k] for k in
        ('contract_version', 'calendar_ref', 'reference_ref', 'schema', 'source_bindings',
         'sessions', 'security_ids', 'reason_dictionary', 'source_sets', 'input_refs')},
        allow_nan=False))
    # Bind the admitted axis snapshots, rather than mutable caller list aliases.
    metadata['sessions'], metadata['security_ids'] = list(days), list(securities)
    require(metadata['contract_version'] == 'feature_column_input_v1' and
            metadata['calendar_ref'] == p['calendar_ref'] and
            metadata['reference_ref'] == p['reference_ref'] and
            metadata['schema'] == _plain(p['input_schema']) and
            metadata['source_bindings'] == _plain(p['sources']), 'Column metadata snapshot mismatch')
    refs = metadata['input_refs']
    require(type(refs) is dict and bool(refs), 'Actual prepared input refs required')
    for name, value in refs.items():
        text(name); digest(value)
    reasons = metadata['reason_dictionary']
    require(type(reasons) is list and bool(reasons) and reasons[0] is None,
            'Reason dictionary: null at zero')
    for reason in reasons[1:]:
        text(reason)
    require(len(reasons) == len(set(reasons)), 'Duplicate reason dictionary entries')
    reasons = list(reasons)
    for reason in ('MISSING', 'WINDOW_MISSING', 'INSUFFICIENT_HISTORY',
                   'REFERENCE_MISSING', 'UNAVAILABLE_AT_SESSION_CUTOFF'):
        if reason not in reasons:
            reasons.append(reason)
    reason_ids = {r: i for i, r in enumerate(reasons)}
    bindings = tuple(s['id'] for s in p['sources'])
    source_ids = {s: i for i, s in enumerate(bindings)}
    sets = metadata['source_sets']
    require(type(sets) is list and bool(sets), 'Source sets required')
    masks = []
    for sources in sets:
        require(type(sources) is list and len(sources) == len(set(sources)) and
                all(type(s) is str and s in source_ids for s in sources), 'Unbound/duplicate source set')
        masks.append(sum(1 << source_ids[s] for s in sources))
    mask_table = np.array(masks, dtype='uint64')
    budget, descriptors = _Budget(max_working_bytes), {}
    shape = len(days), len(securities)
    caller_arrays=[value for field in ('values','validity','reason_codes','available_at_utc_us','source_codes') for value in inputs[field].values()]
    caller_arrays.extend(value for value in inputs.values() if isinstance(value,np.ndarray))
    caller_bytes=sum(value.nbytes for value in {id(a):a for a in caller_arrays}.values())
    budget.add(caller_bytes)
    # Reserve worst-case sequential vector/serialization scratch before allocation.
    # Caller arrays are charged above; Python object overhead is outside this array bound.
    output_cells = len(out_days) * len(securities) * len(p['outputs'])
    budget.add(max(math.prod(shape) * 96, output_cells * 64))
    def take(name, dtype, grid=shape):
        return _snapshot(np, inputs[name], dtype, grid, name, budget, descriptors)
    cutoff_raw = take('cutoffs_utc_us', 'int64', (len(days),))
    cutoff = _time(np, cutoff_raw, cutoff=True)
    require(bool(np.all(cutoff[1:] >= cutoff[:-1])), 'Cutoffs must be monotonic')
    member = take('reference_member', 'bool')
    member_known = (take('reference_member_known', 'bool') if 'reference_member_known' in inputs
                    else np.ones(shape, dtype=bool))
    pool_complete = (take('reference_pool_complete', 'bool') if 'reference_pool_complete' in inputs
                     else np.ones(shape, dtype=bool))
    require(bool(np.all(pool_complete == pool_complete[:, :1])), 'Reference pool completeness must be per session')
    require(not bool(np.any(member & ~member_known)), 'Unknown member cannot be positive')
    ref_available_raw = take('reference_available_at_utc_us', 'int64')
    ref_available = _time(np, ref_available_raw)
    require(bool(np.all(ref_available <= cutoff[:, None])), 'Unavailable reference')
    ref_codes = take('reference_source_codes', 'int32')
    require(bool(np.all((ref_codes >= 0) & (ref_codes < len(sets)))), 'Reference source code range')
    require(all(len(sets[int(c)]) == 1 for c in np.unique(ref_codes)), 'Reference source must be singleton')
    industry_raw = inputs['reference_industry']
    require(isinstance(industry_raw, np.ndarray) and industry_raw.shape == shape and
            industry_raw.dtype == np.dtype('O'), 'Reference industry object grid required')
    industry = tuple(tuple(r) for r in industry_raw)
    for r, day in enumerate(days):
        actual = {}
        for s, security in enumerate(securities):
            group = industry[r][s]
            require(group is None or type(group) is str and bool(group.strip()), 'Invalid reference industry')
            if member[r, s]:
                actual[security] = group
        require(day in p['reference_members'] and actual == dict(p['reference_members'][day]),
                'INCOMPLETE_REFERENCE: frozen member/industry mismatch')
    descriptors['reference_industry'] = {'shape': list(shape), 'digest': _ref(industry)}
    names = {c['name'] for c in p['input_schema']}
    for field in ('values', 'validity', 'reason_codes', 'available_at_utc_us', 'source_codes'):
        require(type(inputs[field]) is dict and set(inputs[field]) == names, 'Input columns mismatch: ' + field)
    field_rows=inputs.get('field_rows')
    if field_rows is not None:
        require(type(field_rows) is dict and set(field_rows)==names,'exact sparse field axes required')
    env = {}
    # Admission snapshots each source column once. Intermediates only cover closure rows.
    for schema in p['input_schema']:
        name = schema['name']
        selected=list(range(len(days))) if field_rows is None else field_rows[name]
        require(type(selected) is list and selected==sorted(set(selected)) and
                all(type(i) is int and 0<=i<len(days) for i in selected) and needed[name]<=set(selected),'sparse field rows omit dependencies')
        field_shape=(len(selected),len(securities))
        row_positions={r:i for i,r in enumerate(selected)}
        if field_rows is not None:descriptors['rows/'+name]=selected
        arrays = {}
        for field, dtype in (('values', schema['dtype']), ('validity', 'bool'),
                              ('reason_codes', 'int32'), ('available_at_utc_us', 'int64'),
                              ('source_codes', 'int32')):
            arrays[field] = _snapshot(np, inputs[field][name], dtype, field_shape,
                                      field + '/' + name, budget, descriptors)
        value, valid, reason, available, codes = (arrays[k] for k in
            ('values', 'validity', 'reason_codes', 'available_at_utc_us', 'source_codes'))
        require(bool(np.all(np.isfinite(value[valid]))), 'Nonfinite present input')
        require(bool(np.all((reason >= 0) & (reason < len(metadata['reason_dictionary'])))) and
                bool(np.all(reason[valid] == 0)) and bool(np.all(reason[~valid] != 0)),
                'Present/missing reason mismatch')
        require(schema['missing'] != 'reject' or bool(np.all(valid)), 'Missing rejected by input schema')
        require(bool(np.all((codes >= 0) & (codes < len(sets)))), 'Input source code range')
        # Packed Core outputs may have no source or clock (constants and
        # reference-excluded missing cells). Preserve that state when reused
        # as facts; a sourced fact still needs a real availability clock.
        clockless = available == _NONE_TIME
        require(bool(np.all(~clockless | (mask_table[codes] == 0))),
                'Clockless input carries source lineage')
        available_seconds = np.where(clockless, _NONE_TIME,
            _time(np, np.where(clockless, 0, available)))
        col = _Column(np, needed[name], len(securities), schema['dtype'], budget)
        if col.rows:
            ix = [row_positions[r] for r in col.rows]
            col.value[:] = value[ix]
            col.issue[:] = available_seconds[ix] > cutoff[list(col.rows), None]
            col.valid[:] = valid[ix] & ~col.issue
            col.available[:] = available_seconds[ix]
            col.source[:] = mask_table[codes[ix]]
            col.reason[:] = np.where(col.valid, 0, np.where(col.issue,
                reason_ids['UNAVAILABLE_AT_SESSION_CUTOFF'], reason[ix]))
            col.value[~col.valid] = 0
        env[name] = col
        # Snapshot bytes were already hashed; no duplicate source graph is retained.
        budget.drop(sum(a.nbytes for a in arrays.values()))
        del arrays, value, valid, reason, available, codes, available_seconds
    output_shape = len(out_days), len(securities), len(p['outputs'])
    output_bytes = math.prod(output_shape) * 29
    budget.add(output_bytes)
    values = np.zeros(output_shape, dtype='float64')
    validity = np.zeros(output_shape, dtype='bool')
    reason_codes = np.zeros(output_shape, dtype='int32')
    availability = np.full(output_shape, _NONE_TIME, dtype='int64')
    source_masks = np.zeros(output_shape, dtype='uint64')
    output_indices = {}
    for i, output in enumerate(p['outputs']):
        output_indices.setdefault(output['node'], []).append(i)
    admitted_ns = perf_counter_ns()

    def merge(out, oi, parents):
        for parent, pi in parents:
            out.available[oi] = np.maximum(out.available[oi], parent.available[pi])
            out.source[oi] |= parent.source[pi]
            out.issue[oi] |= parent.issue[pi]

    def put(out, oi, s, value, reason='MISSING'):
        if value is not None:
            require(type(value) in (int, float, bool) and math.isfinite(value), 'NONFINITE_RESULT')
            if out.dtype == 'bool':
                require(type(value) is bool, 'Boolean result dtype mismatch')
            out.value[oi, s] = value
            out.valid[oi, s] = True
        else:
            out.reason[oi, s] = reason_ids[reason]

    vector_workspace=len(securities)*128;budget.add(vector_workspace)
    reference_groups={}; industry_representatives={}; cohort_cache={}; cohort_charges={}; reference_charge=0
    families={}; family_keys={}; pending_windows={}; node_timings=[]
    counters=dict(window_gathers=0,regression_solves=0,window_sorts=0,window_fsums=0,batch_element_cells=0,scalar_element_cells=0)
    for node in p['nodes']:
        if needed[node['name']] and (node['op']=='rolling' or node['op'] in WINDOW):
            q=node['params'];key=(tuple(node['inputs']),q['window'],q['inclusive_current'])
            families.setdefault(key,[]).append(node);family_keys[node['name']]=key
    for node_index, n in enumerate(p['nodes']):
        name, op, q, args = n['name'], n['op'], n['params'], n['inputs']
        if not needed[name]:
            continue
        node_started=perf_counter_ns();family_names=[]
        parents = [env[a] for a in args]
        if name in family_keys:
            if name not in pending_windows:
                family_names=[node['name'] for node in families[family_keys[name]]]
                pending_windows.update(_window_family(np,families[family_keys[name]],parents,needed,len(securities),budget,reason_ids,counters))
            out=pending_windows.pop(name)
        else:out = _Column(np, needed[name], len(securities), n['column']['dtype'], budget)
        for oi, r in enumerate(out.rows):
            if op in ('shift', 'pct_change'):
                previous = r - q['periods']
                source = parents[0]
                if previous >= 0:
                    pi = source.at(previous)
                    merge(out, oi, [(source, pi)])
                if op == 'shift':
                    if previous < 0:
                        out.reason[oi] = reason_ids['INSUFFICIENT_HISTORY']
                    else:
                        out.value[oi], out.valid[oi], out.reason[oi] = source.value[pi], source.valid[pi], source.reason[pi]
                else:
                    merge(out, oi, [(source, source.at(r))])
                    out.reason[oi]=reason_ids['MISSING']
                    if previous>=0:
                        current=source.at(r);valid=source.valid[current]&source.valid[pi]
                        zero=valid&(source.value[pi]==0)
                        require(q['zero']=='missing' or not bool(np.any(zero)),'ZERO_DENOMINATOR')
                        valid&=~zero;out.valid[oi]=valid
                        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                            np.divide(source.value[current],source.value[pi],out=out.value[oi],where=valid)
                            np.subtract(out.value[oi],1,out=out.value[oi],where=valid)
            elif name in family_keys:
                pass  # Family outputs already computed from one anchored window traversal.
            elif op == 'rolling_industry':
                # One value per industry/session, then broadcast at TODAY's
                # membership. Never roll a stock's changing industry labels.
                source = parents[0]
                end = r + int(q['inclusive_current'])
                window = range(max(0, end-q['window']), end)
                groups = {g for g in industry[r] if g is not None}
                representatives = {}
                for d in window:
                    if d not in industry_representatives:
                        per_day = {}
                        for j, group in enumerate(industry[d]):
                            if member[d,j] and group is not None:per_day.setdefault(group,j)
                        charge=128+len(per_day)*128;budget.add(charge);reference_charge+=charge
                        industry_representatives[d]=per_day
                    representatives[d] = industry_representatives[d]
                for group in groups:
                    dependencies = [(source, source.at(d), representatives[d][group])
                                    for d in window if group in representatives[d]]
                    vals = [source.numeric(d,representatives[d][group]) if group in representatives[d] else None for d in window]
                    present = [v for v in vals if v is not None]
                    blocked = len(present)<q['min_periods'] or q['missing']=='propagate' and len(present)!=len(vals)
                    value = None if blocked else kernel.statistics.fmean(present)
                    for j,g in enumerate(industry[r]):
                        if g != group: continue
                        out.available[oi,j] = max([int(ref_available[r,j])] + [int(src.available[ix,k]) for src,ix,k in dependencies])
                        for src,ix,k in dependencies:
                            out.source[oi,j] |= src.source[ix,k]
                            out.issue[oi,j] |= src.issue[ix,k]
                        out.source[oi,j] |= masks[int(ref_codes[r,j])]
                        put(out,oi,j,value if member[r,j] else None,'REFERENCE_MISSING' if not member[r,j] else 'WINDOW_MISSING')
                for j,g in enumerate(industry[r]):
                    if g is None: put(out,oi,j,None,'REFERENCE_MISSING')
            elif op in CS:
                source = parents[0]
                pi = source.at(r)
                if not pool_complete[r, 0]:
                    # Physical facts remain usable; only reference-dependent
                    # nodes lack the proof of a complete cross section.
                    merge(out, oi, [(source, pi)])
                    ref_mask = 0
                    for code in ref_codes[r]: ref_mask |= masks[int(code)]
                    ref_time = int(np.max(ref_available[r]))
                    for s in range(len(securities)):
                        out.available[oi,s] = max(int(out.available[oi,s]), ref_time)
                        out.source[oi,s] |= ref_mask
                        put(out,oi,s,None,'REFERENCE_MISSING')
                    require(n['column']['missing'] != 'reject', 'Missing rejected by node schema')
                    continue
                group_key=(r,q['group'])
                if group_key not in reference_groups:
                    groups={}
                    for s in range(len(securities)):
                        group=industry[r][s] if q['group']=='industry' else ''
                        groups.setdefault(group,[]).append(s)
                    reference_groups[group_key]=groups
                groups=reference_groups[group_key]
                require(None not in groups or q['unknown_group']!='reject','UNKNOWN_INDUSTRY')
                for group, group_indices in groups.items():
                    cohort_key=(args[0],r,q['group'],group,q['missing'])
                    if cohort_key not in cohort_cache:
                        eligible = [s for s in group_indices if member[r, s] and group is not None]
                        eligible_set = frozenset(eligible)  # one compact cohort set, never per cell
                        raw = [source.numeric(r, s) for s in eligible]
                        absent = any(v is None for v in raw)
                        require(not absent or q['missing'] != 'reject', 'MISSING_REFERENCE_VALUE')
                        vals = [0.0 if v is None else v for v in raw] if q['missing'] == 'fill_zero' else [v for v in raw if v is not None]
                        blocked = absent and q['missing'] == 'propagate'
                        dep_time = max([int(source.available[pi, s]) for s in eligible] +
                                       [int(ref_available[r, s]) for s in group_indices], default=_NONE_TIME)
                        dep_source = 0
                        for s in eligible:
                            dep_source |= int(source.source[pi, s])
                        for s in group_indices:
                            dep_source |= masks[int(ref_codes[r, s])]
                        dep_issue = any(source.issue[pi, s] for s in eligible)
                        charge=512+len(group_indices)*64+len(eligible)*160+len(vals)*64
                        budget.add(charge);cohort_charges[cohort_key]=charge
                        cohort_cache[cohort_key]=(eligible_set,vals,blocked,dep_time,dep_source,dep_issue,{})
                    eligible_set,vals,blocked,dep_time,dep_source,dep_issue,shared=cohort_cache[cohort_key]
                    std, undefined = kernel._cs_zscore_scale(vals, q) if op == 'cs_zscore' else (None, True)
                    mean = kernel.statistics.fmean(vals) if op == 'cs_mean' and vals and not blocked else None
                    ranks = kernel._group_ranks(vals) if op == 'cs_rank' and vals and not blocked else {}
                    bounds=None
                    if op=='cs_winsorize' and vals and not blocked:
                        if 'sorted' not in shared:shared['sorted']=sorted(vals)
                        bounds=(_sorted_quantile(shared['sorted'],q['lower']),_sorted_quantile(shared['sorted'],q['upper']))
                    for s in group_indices:
                        out.available[oi, s] = max(dep_time, int(source.available[pi, s]))
                        out.source[oi, s] = dep_source | int(source.source[pi, s])
                        out.issue[oi, s] = dep_issue or source.issue[pi, s]
                        x = source.numeric(r, s)
                        if x is None and q['missing'] == 'fill_zero':
                            x = 0.0
                        value = None
                        if s in eligible_set and vals and not blocked and (x is not None or op == 'cs_mean'):
                            if op == 'cs_mean': value = mean
                            elif op == 'cs_rank': value = ranks[x]
                            elif op == 'cs_winsorize': value = min(max(x, bounds[0]), bounds[1])
                            else: value, mean = kernel._cs_zscore_value(x, vals, std, undefined, q, mean)
                        elif op == 'cs_zscore' and s in eligible_set and not blocked and undefined:
                            value, mean = kernel._cs_zscore_value(x, vals, std, undefined, q, mean)
                        if op == 'cs_zscore' and s not in eligible_set and q['excluded'] == 'zero_if_undefined' and source.numeric(r, s) is not None:
                            value = 0.0
                        if op == 'cs_zscore' and value is not None and q['clip'] is not None:
                            value = min(max(value, q['clip'][0]), q['clip'][1])
                        put(out, oi, s, value if member_known[r,s] else None, 'REFERENCE_MISSING')
            else:
                merge(out, oi, [(a, a.at(r)) for a in parents])
                if op in _BATCH_ELEMENT:
                    counters['batch_element_cells']+=len(securities)
                    _element_row(np,op,q,parents,r,out,oi,reason_ids)
                else:
                    # Transcendentals retain Python math's exact scalar rounding.
                    counters['scalar_element_cells']+=len(securities)
                    for security in range(len(securities)):
                        value=kernel._element_numeric(op,[a.numeric(r,security) for a in parents],q)
                        put(out,oi,security,value)
            require(bool(np.all(np.isfinite(out.value[oi,out.valid[oi]]))),'NONFINITE_RESULT')
            out.valid[oi, out.issue[oi]] = False
            out.reason[oi, out.issue[oi]] = reason_ids['UNAVAILABLE_AT_SESSION_CUTOFF']
            out.value[oi, ~out.valid[oi]] = 0
            out.reason[oi, out.valid[oi]] = 0
            require(not bool(np.any(out.valid[oi] & (out.available[oi] > cutoff[r]))), 'FUTURE_DEPENDENCY')
            require(n['column']['missing'] != 'reject' or bool(np.all(out.valid[oi])), 'Missing rejected by node schema')
        env[name] = out
        for index in output_indices.get(name, ()):
            ix = [out.at(r) for r in output_rows]
            values[:, :, index] = out.value[ix]
            validity[:, :, index] = out.valid[ix]
            reason_codes[:, :, index] = out.reason[ix]
            availability[:, :, index] = out.available[ix]
            source_masks[:, :, index] = out.source[ix]
        for a in set(args) | {name}:
            if last_use[a] == node_index and a in env:
                budget.drop(env[a].bytes)
                del env[a]
                for key in [key for key in cohort_cache if key[0]==a]:
                    del cohort_cache[key];budget.drop(cohort_charges.pop(key))
        if op in ('shift', 'pct_change', 'rolling_industry') or op in CS:
            del source
        node_timings.append(dict(node=name,op=op,elapsed_ns=perf_counter_ns()-node_started,needed_rows=len(out.rows),window=q.get('window'),securities=len(securities),family=family_names))
        del parents, out
    reference_groups.clear();industry_representatives.clear();budget.drop(reference_charge+vector_workspace)
    computed_ns = perf_counter_ns()
    unique, inverse = np.unique(source_masks, return_inverse=True)
    if unique[0] != 0:
        unique = np.concatenate((np.zeros(1, dtype='uint64'), unique))
        inverse += 1
    source_sets = [sorted(s for i, s in enumerate(bindings) if int(mask) & (1 << i)) for mask in unique]
    output_arrays = {'values': values, 'validity': validity, 'reason_codes': reason_codes,
                     'available_at_utc_us': availability,
                     'source_codes': inverse.astype('int32').reshape(output_shape)}
    result_descriptors = {}
    for name, value in output_arrays.items():
        # Immutable bytes are also the exact bytes bound in the result descriptor.
        output_arrays[name] = _snapshot(np, value, value.dtype, output_shape,
                                       name, budget, result_descriptors)
        budget.drop(value.nbytes)
    binding = {'contract_version': 'feature_column_result_v1', 'compiled_ref': compiled.compiled_ref,
               'plan_ref': compiled.plan_ref, 'implementation_ref': IMPLEMENTATION_REF,
               'calendar_ref': p['calendar_ref'], 'reference_ref': p['reference_ref'],
               'sessions': list(out_days), 'security_ids': list(securities),
               'schema': [_plain(o['column']) for o in p['outputs']],
               'source_bindings': metadata['source_bindings'], 'input_refs': metadata['input_refs'],
               'input_metadata': metadata, 'input_buffers': descriptors,
               'input_ref': _ref({'metadata': metadata, 'buffers': descriptors}),
               'reason_dictionary': reasons, 'source_sets': source_sets,
               'null_available_at_utc_us': _NONE_TIME, 'bool_values_encoding': '0_or_1',
               'output_buffers': result_descriptors}
    result_ref = _ref(binding)
    return {**binding, **output_arrays, 'result_ref': result_ref,
            'stats': {'peak_working_bytes': budget.peak, 'output_cells': math.prod(output_shape),
                      'input_snapshots': len(descriptors), 'view_count': 1,'caller_array_bytes':caller_bytes,
                      'shared_work':counters,'top_nodes':sorted(node_timings,key=lambda v:v['elapsed_ns'],reverse=True)[:12],
                      'admission_ns': admitted_ns - begin_ns,
                      'numeric_and_metadata_ns': computed_ns - admitted_ns,
                      'output_ns': perf_counter_ns() - computed_ns}}
