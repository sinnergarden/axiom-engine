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
from .plan import CS, required_history, validate_plan
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
                                  _freeze(required_history(plan)), _freeze(last),
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
            elif op == 'rolling':
                end = r + int(q['inclusive_current'])
                start = end - q['window']
                require(start >= 0 or p['history_policy'] != 'full', 'INSUFFICIENT_HISTORY')
                require(end > 0, 'COLUMN_PROFILE: empty exclusive rolling window')
                deps.update(range(max(0, start), end))
            else:
                deps.add(r)
        for a in n['inputs']:
            needed[a].update(deps)
    return needed


def execute_feature_columns(compiled, *, inputs, output_sessions, max_working_bytes):
    """One admitted view, many output sessions; result has packed identity.

    Values are [output_session, security, feature], bool schema uses 0/1.
    Missing values use validity=False and canonical value=0. Time null uses
    int64 minimum; all other output times are UTC microseconds on whole seconds.
    Source/reason code zero is respectively the empty set/no missing reason.
    max_working_bytes bounds owned numeric arrays, not interpreter/RSS overhead.
    """
    import numpy as np
    begin_ns = perf_counter_ns()
    require(type(compiled) is CompiledFeatureColumns, 'Expected compiled Feature columns')
    integer(max_working_bytes, 1)
    fields(inputs, _FIELDS)
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
        require(type(sources) is list and bool(sources) and len(sources) == len(set(sources)) and
                all(type(s) is str and s in source_ids for s in sources), 'Unbound/duplicate source set')
        masks.append(sum(1 << source_ids[s] for s in sources))
    mask_table = np.array(masks, dtype='uint64')
    budget, descriptors = _Budget(max_working_bytes), {}
    shape = len(days), len(securities)
    # Reserve worst-case sequential vector/serialization scratch before allocation.
    # Caller-owned buffers and Python object overhead are outside this array bound.
    output_cells = len(out_days) * len(securities) * len(p['outputs'])
    budget.add(max(math.prod(shape) * 96, output_cells * 64))
    def take(name, dtype, grid=shape):
        return _snapshot(np, inputs[name], dtype, grid, name, budget, descriptors)
    cutoff_raw = take('cutoffs_utc_us', 'int64', (len(days),))
    cutoff = _time(np, cutoff_raw, cutoff=True)
    require(bool(np.all(cutoff[1:] >= cutoff[:-1])), 'Cutoffs must be monotonic')
    member = take('reference_member', 'bool')
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
    env = {}
    # Admission snapshots each source column once. Intermediates only cover closure rows.
    for schema in p['input_schema']:
        name = schema['name']
        arrays = {}
        for field, dtype in (('values', schema['dtype']), ('validity', 'bool'),
                              ('reason_codes', 'int32'), ('available_at_utc_us', 'int64'),
                              ('source_codes', 'int32')):
            arrays[field] = _snapshot(np, inputs[field][name], dtype, shape,
                                      field + '/' + name, budget, descriptors)
        value, valid, reason, available, codes = (arrays[k] for k in
            ('values', 'validity', 'reason_codes', 'available_at_utc_us', 'source_codes'))
        require(bool(np.all(np.isfinite(value[valid]))), 'Nonfinite present input')
        require(bool(np.all((reason >= 0) & (reason < len(metadata['reason_dictionary'])))) and
                bool(np.all(reason[valid] == 0)) and bool(np.all(reason[~valid] != 0)),
                'Present/missing reason mismatch')
        require(schema['missing'] != 'reject' or bool(np.all(valid)), 'Missing rejected by input schema')
        require(bool(np.all((codes >= 0) & (codes < len(sets)))), 'Input source code range')
        available_seconds = _time(np, available)
        col = _Column(np, needed[name], len(securities), schema['dtype'], budget)
        if col.rows:
            ix = list(col.rows)
            col.value[:] = value[ix]
            col.issue[:] = available_seconds[ix] > cutoff[ix, None]
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

    for node_index, n in enumerate(p['nodes']):
        name, op, q, args = n['name'], n['op'], n['params'], n['inputs']
        if not needed[name]:
            continue
        out = _Column(np, needed[name], len(securities), n['column']['dtype'], budget)
        parents = [env[a] for a in args]
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
                    for s in range(len(securities)):
                        value = kernel._divide(source.numeric(r, s),
                            source.numeric(previous, s) if previous >= 0 else None, q['zero'])
                        put(out, oi, s, None if value is None else value - 1)
            elif op == 'rolling':
                end = r + int(q['inclusive_current'])
                rows = tuple(range(max(0, end - q['window']), end))
                source = parents[0]
                ix = [source.at(d) for d in rows]
                out.available[oi] = np.max(source.available[ix], axis=0)
                out.source[oi] = np.bitwise_or.reduce(source.source[ix], axis=0)
                out.issue[oi] = np.any(source.issue[ix], axis=0)
                window_values, window_valid = source.value[ix], source.valid[ix]
                for s in range(len(securities)):
                    present = window_values[:, s][window_valid[:, s]].tolist()
                    blocked = len(present) < q['min_periods'] or q['missing'] == 'propagate' and len(present) != len(rows)
                    value = None if blocked else kernel._rolling_value(present, source.numeric(rows[-1], s), q)
                    put(out, oi, s, value, 'WINDOW_MISSING' if blocked else 'MISSING')
                del window_values, window_valid
            elif op in CS:
                source = parents[0]
                pi = source.at(r)
                groups = {}
                for s in range(len(securities)):
                    group = industry[r][s] if q['group'] == 'industry' else ''
                    require(group is not None or q['unknown_group'] != 'reject', 'UNKNOWN_INDUSTRY')
                    groups.setdefault(group, []).append(s)
                for group, group_indices in groups.items():
                    eligible = [s for s in group_indices if member[r, s] and group is not None]
                    eligible_set = frozenset(eligible)  # one compact cohort set, never per cell
                    raw = [source.numeric(r, s) for s in eligible]
                    absent = any(v is None for v in raw)
                    require(not absent or q['missing'] != 'reject', 'MISSING_REFERENCE_VALUE')
                    vals = [0.0 if v is None else v for v in raw] if q['missing'] == 'fill_zero' else [v for v in raw if v is not None]
                    blocked = absent and q['missing'] == 'propagate'
                    std, undefined = kernel._cs_zscore_scale(vals, q) if op == 'cs_zscore' else (None, True)
                    dep_time = max([int(source.available[pi, s]) for s in eligible] +
                                   [int(ref_available[r, s]) for s in group_indices], default=_NONE_TIME)
                    dep_source = 0
                    for s in eligible:
                        dep_source |= int(source.source[pi, s])
                    for s in group_indices:
                        dep_source |= masks[int(ref_codes[r, s])]
                    dep_issue = any(source.issue[pi, s] for s in eligible)
                    mean = None
                    for s in group_indices:
                        out.available[oi, s] = max(dep_time, int(source.available[pi, s]))
                        out.source[oi, s] = dep_source | int(source.source[pi, s])
                        out.issue[oi, s] = dep_issue or source.issue[pi, s]
                        x = source.numeric(r, s)
                        if x is None and q['missing'] == 'fill_zero':
                            x = 0.0
                        value = None
                        if s in eligible_set and vals and not blocked and x is not None:
                            if op == 'cs_rank': value = kernel._rank(x, vals)
                            elif op == 'cs_winsorize': value = min(max(x, kernel._quantile(vals, q['lower'])), kernel._quantile(vals, q['upper']))
                            else: value, mean = kernel._cs_zscore_value(x, vals, std, undefined, q, mean)
                        elif op == 'cs_zscore' and s in eligible_set and not blocked and undefined:
                            value, mean = kernel._cs_zscore_value(x, vals, std, undefined, q, mean)
                        if op == 'cs_zscore' and s not in eligible_set and q['excluded'] == 'zero_if_undefined' and source.numeric(r, s) is not None:
                            value = 0.0
                        if op == 'cs_zscore' and value is not None and q['clip'] is not None:
                            value = min(max(value, q['clip'][0]), q['clip'][1])
                        put(out, oi, s, value, 'REFERENCE_MISSING')
            else:
                merge(out, oi, [(a, a.at(r)) for a in parents])
                for s in range(len(securities)):
                    value = kernel._element_numeric(op, [a.numeric(r, s) for a in parents], q)
                    reason = reasons[int(parents[0].reason[parents[0].at(r), s])] if op == 'identity' and value is None else 'MISSING'
                    put(out, oi, s, value, reason)
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
        if op in ('shift', 'pct_change', 'rolling') or op in CS:
            del source
        del parents, out
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
                      'input_snapshots': len(descriptors), 'view_count': 1,
                      'admission_ns': admitted_ns - begin_ns,
                      'numeric_and_metadata_ns': computed_ns - admitted_ns,
                      'output_ns': perf_counter_ns() - computed_ns}}
