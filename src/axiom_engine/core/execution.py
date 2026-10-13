"""The sole Feature execution path. No I/O, dynamic code, or mutable plugins."""
from dataclasses import dataclass
from datetime import date
import math
import statistics

from .contracts import (ABI, SEMANTICS, ContractError, ExecutionContext, FactBatch,
                        FeatureFrame, check_value, fields, require, session,
                        text, timestamp, unresolved)
from .plan import CS, WINDOW, validate_plan


@dataclass(frozen=True)
class _Cell:
    value: object
    available: str | int | None
    sources: tuple
    issues: tuple = ()
    reason: str | None = None


def _merge(value, deps, reason=None):
    deps = tuple(deps)
    available = max((d.available for d in deps if d.available is not None), default=None)
    sources = tuple(sorted({s for d in deps for s in d.sources}))
    issues = tuple(sorted({s for d in deps for s in d.issues}))
    if type(value) in (int, float) and not math.isfinite(value):
        raise ContractError('NONFINITE_RESULT: explicit input/arithmetic policy required')
    if issues:
        value = None
    return _Cell(value, available, sources, issues,
                 reason if value is None and reason else 'MISSING' if value is None else None)


def _key(obj):
    text(obj['security_id']); session(obj['session'])
    return obj['security_id'], obj['session']


def _key_list(values):
    require(type(values) is list, 'Keys must be an array')
    keys = []
    for x in values:
        require(type(x) is list and len(x) == 2, 'Key must be [security_id,session]')
        text(x[0]); session(x[1]); keys.append(tuple(x))
    require(len(keys) == len(set(keys)), 'Duplicate explicit keys')
    return keys


_INPUT_TIMESTAMP_CACHE_LIMIT = 128


def _timestamp_once(value, checked):
    # Only successfully admitted, exact strings enter this call-local memo.
    # Each is exactly 20 characters; capacity and borrowed-key storage are bounded.
    if type(value) is str and value in checked:
        return
    timestamp(value)
    if len(checked) == _INPUT_TIMESTAMP_CACHE_LIMIT:
        del checked[next(iter(checked))]
    checked[value] = None


def _cells(row, columns, source_ids, *, _timestamps=None):
    size = len(columns)
    for k in ('values', 'availability', 'sources', 'missing_reasons'):
        require(type(row[k]) is list and len(row[k]) == size, 'Missing/misaligned row columns')
    out = []
    for v, a, refs, reason, col in zip(row['values'], row['availability'], row['sources'], row['missing_reasons'], columns):
        check_value(v, col)
        if _timestamps is None: timestamp(a)
        else: _timestamp_once(a, _timestamps)
        require(type(refs) is list and bool(refs) and all(type(s) is str and s in source_ids for s in refs),
                'Unbound cell source')
        require(len(refs) == len(set(refs)), 'Duplicate cell source')
        if v is None:
            text(reason)
        else:
            require(reason is None, 'Present fact has missing reason')
        out.append(_Cell(float(v) if col['dtype'] == 'float64' and v is not None else v,
                         a, tuple(sorted(refs)), (), reason))
    return out


def _reference_members_by_day(refs):
    members = {}
    for key, row in refs.items():
        if row['member']: members.setdefault(key[1], {})[key[0]] = row['industry']
    return members


def _inputs(p, facts, context):
    require(type(facts) is FactBatch and type(context) is ExecutionContext, 'Expected explicit FactBatch/ExecutionContext')
    f, c = facts.to_dict(), context.to_dict()
    require(not unresolved(f) and not unresolved(c), 'UNRESOLVED input/context')
    fields(f, 'abi calendar_ref schema sources rows event_schema events')
    fields(c, 'abi calendar_ref reference_ref sessions cutoffs history_keys output_keys reference')
    require(f['abi'] == c['abi'] == ABI, 'Unsupported frame/context ABI')
    require(f['calendar_ref'] == c['calendar_ref'] == p['calendar_ref'], 'Calendar binding mismatch')
    require(c['reference_ref'] == p['reference_ref'], 'Reference binding mismatch')
    require(f['schema'] == p['input_schema'] and f['event_schema'] == p['event_schema'], 'Input schema mismatch')
    require(f['sources'] == p['sources'], 'Source/revision/qualification binding mismatch')
    sessions = c['sessions']
    require(type(sessions) is list and bool(sessions), 'Explicit calendar sessions required')
    for s in sessions: session(s)
    require(sessions == sorted(set(sessions)), 'Calendar sessions must be sorted and unique')
    session_positions = {s: i for i, s in enumerate(sessions)}
    checked_timestamps = {}
    require(type(c['cutoffs']) is dict and set(c['cutoffs']) == set(sessions), 'Every session needs its own cutoff')
    for t in c['cutoffs'].values(): _timestamp_once(t, checked_timestamps)
    require(list(c['cutoffs'][s] for s in sessions) == sorted(c['cutoffs'][s] for s in sessions), 'Cutoffs must be monotonic')
    history, outputs = _key_list(c['history_keys']), _key_list(c['output_keys'])
    history_set = set(history)
    require(bool(history) and bool(outputs) and set(outputs) <= history_set, 'Output/history coverage')
    require(all(k[1] in session_positions for k in history), 'Unknown session')
    history = sorted(history)
    source_ids = {s['id'] for s in p['sources']}
    require(type(f['rows']) is list, 'Rows must be array')
    rows = {}
    for r in f['rows']:
        fields(r, 'security_id session values availability sources missing_reasons')
        key = _key(r)
        require(key not in rows, 'Duplicate fact key')
        cells = _cells(r, f['schema'], source_ids, _timestamps=checked_timestamps)
        require(key in history_set, 'Undeclared fact key')
        cutoff = c['cutoffs'][key[1]]
        rows[key] = [_Cell(None, v.available, v.sources, ('UNAVAILABLE_AT_SESSION_CUTOFF',), 'UNAVAILABLE')
                     if v.available > cutoff else v for v in cells]
    require(set(rows) == history_set, 'MISSING_HISTORY_ROW: supplied rows do not match declared history')
    by_security = {}
    for key in history: by_security.setdefault(key[0], []).append(key)
    if p['observation_domain'] == 'sessions':
        for keys in by_security.values():
            start, end = session_positions[keys[0][1]], session_positions[keys[-1][1]]
            require([k[1] for k in keys] == sessions[start:end+1], 'MISSING_SESSION: history cannot compress calendar gaps')
    refs = {}
    require(type(c['reference']) is list, 'Explicit reference rows required')
    for r in c['reference']:
        fields(r, 'security_id session member industry available_at source')
        key = _key(r)
        require(key not in refs and key in rows, 'Duplicate/undeclared reference key')
        require(type(r['member']) is bool, 'Membership must be explicit bool')
        require(r['industry'] is None or type(r['industry']) is str and bool(r['industry'].strip()), 'Industry must be string or null')
        _timestamp_once(r['available_at'], checked_timestamps)
        require(r['available_at'] <= c['cutoffs'][key[1]] and r['source'] in source_ids, 'Unavailable/unbound reference')
        refs[key] = r
    require(set(refs) == history_set, 'Complete frozen reference mask required for history')
    members_by_day = _reference_members_by_day(refs)
    for day in {k[1] for k in history}:
        actual = members_by_day.get(day, {})
        require(day in p['reference_members'] and actual == p['reference_members'][day],
                'INCOMPLETE_REFERENCE: members/industry differ from frozen plan')
    events = {name: {} for name in p['event_schema']}
    require(type(f['events']) is list, 'Events must be an array')
    seen_ids = set()
    for e in f['events']:
        fields(e, 'stream security_id event_id event_session report_period values availability sources missing_reasons')
        require(e['stream'] in events, 'Undeclared event stream')
        for k in ('security_id', 'event_id'): text(e[k])
        session(e['event_session']); session(e['report_period'])
        ident = (e['stream'], e['security_id'], e['event_id'])
        require(ident not in seen_ids, 'Duplicate event ID'); seen_ids.add(ident)
        cells = _cells(e, p['event_schema'][e['stream']], source_ids, _timestamps=checked_timestamps)
        stream = events[e['stream']].setdefault(e['security_id'], [])
        require(all(x[0] != e['event_session'] for x in stream), 'Ambiguous event date: Data must resolve fixed stream')
        stream.append((e['event_session'], e['report_period'], cells))
    for streams in events.values():
        for stream in streams.values(): stream.sort(key=lambda e: e[0])
    return c, history, outputs, rows, by_security, refs, events


def _reference_index(refs):
    """Index already-validated references in their original iteration order."""
    by_session, by_industry = {}, {}
    for key, row in refs.items():
        by_session.setdefault(key[1], []).append(key)
        by_industry.setdefault((key[1], row['industry']), []).append(key)
    return by_session, by_industry


def _required_cells(p, outputs, by_security, refs, reference_index):
    """Per-node evaluation scope on the original history, including CS inputs.

    This extends the history admission walk; it does not reorder or crop rows.
    Partial warmup follows the same dependencies but permits absent predecessors.
    """
    by_session, by_industry = reference_index
    nodes = {n['name']: n for n in p['nodes']}
    positions = {k: i for keys in by_security.values() for i, k in enumerate(keys)}
    pending = [(o['node'], k) for o in p['outputs'] for k in outputs]
    seen = set()
    expanded_cs = set()
    needed = {name: set() for name in nodes}
    while pending:
        name, key = pending.pop()
        if (name, key) in seen: continue
        seen.add((name, key))
        if name not in nodes: continue
        needed[name].add(key)
        n = nodes[name]; op = n['op']; q = n['params']
        dependencies = [key]
        if op in ('shift', 'pct_change', 'rolling') or op in WINDOW:
            i = positions[key]; history = by_security[key[0]]
            if op in ('shift', 'pct_change'):
                require(p['history_policy'] == 'partial' or i >= q['periods'],
                        'INSUFFICIENT_HISTORY: lag dependency')
                dependencies = [history[i-q['periods']]] if i >= q['periods'] else []
                if op == 'pct_change': dependencies.append(key)
            else:
                end = i + int(q['inclusive_current']); start = end-q['window']
                require(p['history_policy'] == 'partial' or start >= 0,
                        'INSUFFICIENT_HISTORY: full rolling dependency')
                dependencies = history[max(0, start):end]
        elif op == 'rolling_industry':
            i=positions[key]; history=by_security[key[0]]
            end=i+int(q['inclusive_current']); start=end-q['window']
            require(p['history_policy']=='partial' or start>=0,'INSUFFICIENT_HISTORY')
            group=refs[key]['industry']
            dependencies=[k for _,day in history[max(0,start):end]
                          for k in by_industry.get((day,group),()) if refs[k]['member']] if group is not None else []
        elif op in CS:
            industry = refs[key]['industry']
            group = (name, key[1], industry if q['group'] == 'industry' else '')
            if group not in expanded_cs:
                expanded_cs.add(group)
                reference_keys = (by_session[key[1]] if q['group'] == 'session' else
                                  by_industry[(key[1], industry)] if industry is not None else ())
                dependencies += [k for k in reference_keys if refs[k]['member']]
            # The first expansion drains all group parents before another key
            # of this node is visited. Each key still needs its own parent.
        for parent in n['inputs']:
            pending.extend((parent, k) for k in dependencies)
    return needed


def _divide(a, b, policy):
    if a is None or b is None: return None
    if b == 0:
        require(policy == 'missing', 'ZERO_DENOMINATOR')
        return None
    return a / b


def _rank(x, vals):
    return (sum(v < x for v in vals) + (sum(v == x for v in vals) + 1) / 2) / len(vals)


def _std(vals, ddof):
    if len(vals) <= ddof: return None
    # statistics retains exact ratios until its scaled square root: no float
    # intermediate squared deviations to underflow for representable small std.
    return statistics.pstdev(vals) if ddof == 0 else statistics.stdev(vals)


def _cs_zscore_scale(vals, q):
    """Shared ordered scale calculation for row-wire and packed CS inputs."""
    std = _std(vals, q['ddof']) if vals else None
    return std, std is None or std == 0 or std < q.get('epsilon', 0)


def _cs_zscore_value(x, vals, std, undefined, q, mean):
    """Keep the original arithmetic and lazily calculate a group's mean once."""
    if undefined:
        require(q['constant'] != 'reject', 'UNDEFINED_CS_SCALE')
        value = 0.0 if x is not None and q['constant'] == 'zero' else None
    else:
        if mean is None: mean = math.fsum(vals) / len(vals)
        value = (x - mean) / std
    return value, mean


def _quantile(vals, q):
    vals = sorted(vals)
    pos = (len(vals)-1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos-lo)


def _element_numeric(op, vals, q):
    """Original element arithmetic shared by row and column transports."""
    if op == 'constant': return q['value']
    x = vals[0]
    if op == 'identity': return x
    if op == 'is_missing': return x is None
    if op == 'fill': return q['value'] if x is None else x
    if op in ('gt', 'lt', 'eq', 'and', 'or', 'not', 'where'):
        if any(v is None for v in vals[:1] if op == 'where') or op != 'where' and any(v is None for v in vals):
            require(q['missing'] != 'reject', 'MISSING_BOOLEAN')
            if q['missing'] == 'preserve': return None
            # Comparisons involving a missing operand are false, not comparison to zero.
            if op in ('gt', 'lt', 'eq'): return False
            vals = ([False, *vals[1:]] if op == 'where' else
                    [False if v is None else v for v in vals])
        if op == 'where': return vals[1] if vals[0] else vals[2]
        if op == 'not': return not vals[0]
        a, b = vals
        value = {'gt': lambda: a > b, 'lt': lambda: a < b, 'eq': lambda: a == b,
                 'and': lambda: a and b, 'or': lambda: a or b}[op]()
        return value
    if any(v is None for v in vals): return None
    if op == 'divide': value = _divide(*vals, q['zero'])
    elif op == 'add': value = vals[0] + vals[1]
    elif op == 'sub': value = vals[0] - vals[1]
    elif op == 'mul': value = vals[0] * vals[1]
    elif op == 'abs': value = abs(x)
    elif op == 'sqrt':
        require(x>=0 or q['domain']=='missing','SQRT_DOMAIN')
        value=math.sqrt(x) if x>=0 else None
    elif op == 'to_float': value = float(x)
    elif op == 'calendar_age': value = max(0, (date.fromisoformat(vals[0]) - date.fromisoformat(vals[1])).days)
    elif op == 'clip':
        value = x
        if q['lower'] is not None: value = max(value, q['lower'])
        if q['upper'] is not None: value = min(value, q['upper'])
    elif op == 'log1p':
        require(x > -1 or q['domain'] == 'missing', 'LOG_DOMAIN')
        value = math.log1p(x) if x > -1 else None
    else: raise ContractError('Unsupported element operator')
    return value


def _element(op, cells, q):
    return _merge(_element_numeric(op, [c.value for c in cells], q), cells,
                  cells[0].reason if op == 'identity' else None)


def _rolling_value(vals, last, q):
    """The original ordered reduction; provenance is merged by each view."""
    op = q['reduction']
    if op == 'mean': value = math.fsum(vals) / len(vals)
    elif op == 'sum': value = math.fsum(vals)
    elif op == 'min': value = min(vals)
    elif op == 'max': value = max(vals)
    elif op == 'median': value = statistics.median(vals)
    elif op == 'std': value = _std(vals, q['ddof'])
    else: value = _rank(last, vals) if last is not None else None
    return value


def _regression_cache():
    """Call-local shared outputs; bound retained input positions to 65,536."""
    cache = {}
    positions = 0
    def evaluate(values):
        nonlocal positions
        if values in cache:
            return cache[values]
        result = _regression(values)
        if len(values) <= 65536:
            while cache and positions + len(values) > 65536:
                oldest = next(iter(cache))
                positions -= len(oldest)
                del cache[oldest]
            cache[values] = result
            positions += len(values)
        return result
    return evaluate


def _regression(values):
    """Centered OLS on original 1-based positions; all three outputs share work."""
    pairs = [(i + 1, v) for i, v in enumerate(values) if v is not None]
    if len(pairs) < 2:
        return None, None, None
    xs, ys = zip(*pairs)
    mx, my = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    xx, yy = math.fsum(x*x for x in dx), math.fsum(y*y for y in dy)
    xy = math.fsum(x*y for x, y in zip(dx, dy))
    slope = xy / xx
    rsquare = None if yy == 0 else min(1.0, max(0.0, (xy / xx) * (xy / yy)))
    residual = None if values[-1] is None else values[-1] - (my + slope * (len(values) - mx))
    return slope, rsquare, residual


def _window_value(op, columns, q, regression):
    """Return numeric value and window insufficiency, without provenance reads.

    columns retain missing positions. Correlation uses pairwise deletion; all
    other reductions use the single input's valid observations. Temporal gaps
    are never compressed for regression or arg-extrema.
    """
    values = columns[0]
    present = [(i, v) for i, v in enumerate(values) if v is not None and
               (op != 'rolling_corr' or columns[1][i] is not None)]
    blocked = len(present) < q['min_periods'] or q['missing'] == 'propagate' and len(present) != len(values)
    if blocked:
        return None, True
    if op == 'rolling_regression':
        result = regression(tuple(values))
        value = result[('slope', 'rsquare', 'resi').index(q['output'])]
        if value is None and (q['output'] == 'rsquare' or len(present) < 2):
            require(q['zero'] == 'missing', 'ZERO_VARIANCE')
    elif op == 'rolling_corr':
        xs = [v for _, v in present]; ys = [columns[1][i] for i, _ in present]
        mx, my = math.fsum(xs)/len(xs), math.fsum(ys)/len(ys)
        dx, dy = [v-mx for v in xs], [v-my for v in ys]
        xx, yy = math.fsum(v*v for v in dx), math.fsum(v*v for v in dy)
        require(xx != 0 and yy != 0 or q['zero'] == 'missing', 'ZERO_VARIANCE')
        value = None if xx == 0 or yy == 0 else max(-1.0, min(1.0,
            (math.fsum(x*y for x,y in zip(dx,dy)) / math.sqrt(xx)) / math.sqrt(yy)))
    elif op == 'rolling_quantile':
        value = _quantile([v for _, v in present], q['quantile'])
    else:
        extreme = (max if q['extreme'] == 'max' else min)(v for _, v in present)
        indices = [i+1 for i,v in present if v == extreme]
        value = indices[0 if q['ties'] == 'first' else -1]
    return value, False


def _rolling(cells, q, *, reuse=None, scope=None, node_index=None, source_name=None):
    vals = [c.value for c in cells if c.value is not None]
    if len(vals) < q['min_periods'] or q['missing'] == 'propagate' and len(vals) != len(cells):
        return _merge(None, cells, 'WINDOW_MISSING')
    args = (vals, cells[-1].value, q)
    if reuse is not None and q['reduction'] == 'std':
        value = reuse.evaluate('rolling.std', node_index, scope, cells, _rolling_value, args,
                               source_name=source_name)
    else:
        value = _rolling_value(*args)
    return _merge(value, cells)


def _cross_section(op, source, keys, refs, q, reference_index, *, reuse=None, node_index=None, source_name=None):
    groups = {}
    for key in keys:
        r = refs[key]
        group = r['industry'] if q['group'] == 'industry' else ''
        if group is None:
            require(q['unknown_group'] != 'reject', 'UNKNOWN_INDUSTRY')
        groups.setdefault((key[1], group), []).append(key)
    out = {}
    by_session, by_industry = reference_index
    for (_, group), group_keys in groups.items():
        day = group_keys[0][1]
        reference_keys = (by_session[day] if q['group'] == 'session' else by_industry[(day, group)])
        eligible = [k for k in reference_keys if refs[k]['member'] and group is not None]
        deps = [source[k] for k in eligible]
        # Membership and classification are dependencies even when they exclude a row.
        refs_cells = [_Cell(None, refs[k]['available_at'], (refs[k]['source'],)) for k in reference_keys]
        vals = [c.value for c in deps]
        absent = any(v is None for v in vals)
        require(not absent or q['missing'] != 'reject', 'MISSING_REFERENCE_VALUE')
        if q['missing'] == 'fill_zero': vals = [0.0 if v is None else v for v in vals]
        vals = [v for v in vals if v is not None]
        blocked = absent and q['missing'] == 'propagate'
        if op == 'cs_zscore':
            std, undefined = (_cs_zscore_scale(vals, q) if reuse is None else
                reuse.evaluate('cs_zscore.scale', node_index, reference_keys, deps,
                               _cs_zscore_scale, (vals, q), refs=refs,
                               source_name=source_name, dependency_scope=eligible))
        else:
            std, undefined = None, True
        eligible = set(eligible)
        group_dependency = _merge(None, deps + refs_cells)
        mean = statistics.fmean(vals) if op == 'cs_mean' and vals and not blocked else None
        for key in group_keys:
            x = source[key].value
            if x is None and q['missing'] == 'fill_zero': x = 0.0
            value = None
            if key in eligible and vals and not blocked and (x is not None or op == 'cs_mean'):
                if op == 'cs_mean': value = mean
                elif op == 'cs_rank': value = _rank(x, vals)
                elif op == 'cs_winsorize': value = min(max(x, _quantile(vals, q['lower'])), _quantile(vals, q['upper']))
                else: value, mean = _cs_zscore_value(x, vals, std, undefined, q, mean)
            elif op == 'cs_zscore' and key in eligible and not blocked and undefined:
                value, mean = _cs_zscore_value(x, vals, std, undefined, q, mean)
            if op == 'cs_zscore' and key not in eligible and q['excluded'] == 'zero_if_undefined' and source[key].value is not None:
                # R0 industry mask: excluded rows have unmapped (missing) scale.
                value = 0.0
            if op == 'cs_zscore' and value is not None and q['clip'] is not None:
                value = min(max(value, q['clip'][0]), q['clip'][1])
            out[key] = _merge(value, [group_dependency, source[key]], 'REFERENCE_MISSING')
    return out


def execute_feature_plan(plan, facts, context):
    """Validate and execute a frozen plan over explicit rows; return FeatureFrame.

    Call this same function for batch and single-session requests. Historical
    values always use their own session cutoffs, including rolling dependencies.
    Caller supplies a complete bound reference universe, even for one output key.
    """
    return _execute_feature_plan(plan, facts, context)


def _execute_feature_plan(plan, facts, context, *, reuse=None, profile=None):
    """The shared execution path; only call-local numeric reuse is optional."""
    p = validate_plan(plan, execution=True)
    c, keys, outputs, rows, by_security, refs, events = _inputs(p, facts, context)
    if profile is not None:
        profile.observe(p, c, keys, outputs)
        reuse.begin_view(profile.universe)
    reference_index = _reference_index(refs)
    needed = _required_cells(p, outputs, by_security, refs, reference_index)
    positions = {k: i for security_keys in by_security.values() for i, k in enumerate(security_keys)}
    env = {col['name']: {k: rows[k][i] for k in keys} for i, col in enumerate(p['input_schema'])}
    regression = _regression_cache()
    for node_index, n in enumerate(p['nodes']):
        op, q, args = n['op'], n['params'], n['inputs']
        node_keys = sorted(needed[n['name']])
        if not node_keys:
            continue
        if op in CS:
            out = _cross_section(op, env[args[0]], node_keys, refs, q, reference_index,
                                 reuse=reuse, node_index=node_index, source_name=args[0])
        elif op == 'rolling_industry':
            # Unknown industry rows have no parent cohort to evaluate. Keep
            # their result explicitly missing without requiring an otherwise
            # unused cross-sectional parent node in the execution environment.
            out={}; source=env.get(args[0], {})
            for key in node_keys:
                i=positions[key]; history=by_security[key[0]]; group=refs[key]['industry']
                end=i+int(q['inclusive_current']); cells=[]
                for _,day in history[max(0,end-q['window']):end]:
                    candidates=[k for k in reference_index[1].get((day,group),()) if refs[k]['member']] if group is not None else []
                    cells.append(source[candidates[0]] if candidates else _Cell(None,None,(),(),'REFERENCE_MISSING'))
                value=_rolling(cells,q) if group is not None and refs[key]['member'] else _Cell(None,None,(),(),'REFERENCE_MISSING')
                out[key]=_merge(value.value,[value,_Cell(None,refs[key]['available_at'],(refs[key]['source'],))],value.reason)
        elif op in ('shift', 'pct_change', 'rolling') or op in WINDOW:
            out = {}
            source = env.get(args[0], {})
            for key in node_keys:
                i = positions[key]
                security_keys = by_security[key[0]]
                if op in ('shift', 'pct_change'):
                    j = i - q['periods']
                    prev = source[security_keys[j]] if j >= 0 else _Cell(None, None, (), (), 'INSUFFICIENT_HISTORY')
                    if op == 'shift': out[key] = prev
                    else:
                        v = _divide(source[key].value, prev.value, q['zero'])
                        out[key] = _merge(None if v is None else v-1, [source[key], prev])
                else:
                    end = i + int(q['inclusive_current'])
                    start = max(0, end - q['window'])
                    window = security_keys[start:end]
                    if op in WINDOW:
                        cells = [[env[a][k] for k in window] for a in args]
                        value, blocked = _window_value(op, [[c.value for c in cs] for cs in cells], q, regression)
                        out[key] = _merge(value, [c for cs in cells for c in cs], 'WINDOW_MISSING' if blocked else None)
                    else:
                        out[key] = _rolling([source[k] for k in window], q, reuse=reuse,
                                            scope=window, node_index=node_index, source_name=args[0])
        elif op == 'asof':
            columns = p['event_schema'][q['stream']]
            ix = next(i for i, col in enumerate(columns) if col['name'] == q['field'])
            if q['report_policy'] == 'nondecreasing':
                for stream in events[q['stream']].values():
                    periods = [event[1] for event in stream]
                    require(periods == sorted(periods), 'REPORT_PERIOD_REGRESSION: Data stream violates frozen policy')
            out = {}
            for key in node_keys:
                candidates = []
                for event_date, _, cells in events[q['stream']].get(key[0], []):
                    cell = cells[ix]
                    exact = q['match'] == 'exact_date'
                    date_ok = event_date <= key[1] if exact else event_date < key[1]
                    dependency_ok = cell.available[:10] <= key[1] if exact else cell.available[:10] < key[1]
                    if date_ok and dependency_ok and cell.available <= c['cutoffs'][key[1]]:
                        candidates.append(cell)
                out[key] = candidates[-1] if candidates else _Cell(None, None, (), (), 'NO_VISIBLE_EVENT')
        else:
            out = {k: _element(op, [env[a][k] for a in args], q) for k in node_keys}
        for key, cell in out.items():
            check_value(cell.value, n['column'])
            if n['column']['dtype'] == 'float64' and cell.value is not None:
                cell = _Cell(float(cell.value), cell.available, cell.sources, cell.issues, cell.reason)
                out[key] = cell
            # Every result at a historical key remains subject to that key's cutoff.
            require(cell.value is None or cell.available is None or cell.available <= c['cutoffs'][key[1]],
                    'FUTURE_DEPENDENCY')
        env[n['name']] = out
    result_rows = []
    for key in sorted(outputs):
        cells = [env[o['node']][key] for o in p['outputs']]
        result_rows.append({'security_id': key[0], 'session': key[1],
                            'cutoff': c['cutoffs'][key[1]],
                            'values': [v.value for v in cells],
                            'availability': [v.available for v in cells],
                            'sources': [list(v.sources) for v in cells],
                            'valid': [v.value is not None and not v.issues for v in cells],
                            'reasons': [list(v.issues) or ([v.reason] if v.reason else []) for v in cells]})
    return FeatureFrame.from_dict({'abi': ABI, 'semantics': SEMANTICS,
        'plan_identity': plan.identity, 'fact_identity': facts.identity, 'context_identity': context.identity,
        'recipe_ref': p['recipe_ref'], 'calendar_ref': p['calendar_ref'], 'reference_ref': p['reference_ref'],
        'schema': [o['column'] for o in p['outputs']], 'source_bindings': p['sources'],
        'rows': result_rows})
