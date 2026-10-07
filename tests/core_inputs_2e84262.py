"""Frozen input admission oracle from 2e84262acdf6299ffd65d646360a01cddaefd06a.

The copied admission/cell loops are intentionally independent of the optimized
ones. Only unchanged scalar/key contracts and the unchanged Cell class are shared.
"""
from axiom_engine.core.contracts import (ABI, ExecutionContext, FactBatch,
    check_value, fields, require, session, text, timestamp, unresolved)
from axiom_engine.core.execution import _Cell, _key, _key_list

BASE_EXECUTION_SHA256 = '299af021c16ae4cdae05f014e9d3a20df76d3949917615b918f79616a8d575f5'

def _old_cells(row, columns, source_ids):
    size = len(columns)
    for k in ('values', 'availability', 'sources', 'missing_reasons'):
        require(type(row[k]) is list and len(row[k]) == size, 'Missing/misaligned row columns')
    out = []
    for v, a, refs, reason, col in zip(row['values'], row['availability'], row['sources'], row['missing_reasons'], columns):
        check_value(v, col); timestamp(a)
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


def original_inputs(p, facts, context):
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
    require(type(c['cutoffs']) is dict and set(c['cutoffs']) == set(sessions), 'Every session needs its own cutoff')
    for t in c['cutoffs'].values(): timestamp(t)
    require(list(c['cutoffs'][s] for s in sessions) == sorted(c['cutoffs'][s] for s in sessions), 'Cutoffs must be monotonic')
    history, outputs = _key_list(c['history_keys']), _key_list(c['output_keys'])
    history_set = set(history)
    require(bool(history) and bool(outputs) and set(outputs) <= history_set, 'Output/history coverage')
    require(all(k[1] in sessions for k in history), 'Unknown session')
    history = sorted(history)
    source_ids = {s['id'] for s in p['sources']}
    require(type(f['rows']) is list, 'Rows must be array')
    rows = {}
    for r in f['rows']:
        fields(r, 'security_id session values availability sources missing_reasons')
        key = _key(r)
        require(key not in rows, 'Duplicate fact key')
        cells = _old_cells(r, f['schema'], source_ids)
        require(key in history_set, 'Undeclared fact key')
        cutoff = c['cutoffs'][key[1]]
        rows[key] = [_Cell(None, v.available, v.sources, ('UNAVAILABLE_AT_SESSION_CUTOFF',), 'UNAVAILABLE')
                     if v.available > cutoff else v for v in cells]
    require(set(rows) == history_set, 'MISSING_HISTORY_ROW: supplied rows do not match declared history')
    by_security = {}
    for key in history: by_security.setdefault(key[0], []).append(key)
    if p['observation_domain'] == 'sessions':
        for keys in by_security.values():
            start, end = sessions.index(keys[0][1]), sessions.index(keys[-1][1])
            require([k[1] for k in keys] == sessions[start:end+1], 'MISSING_SESSION: history cannot compress calendar gaps')
    refs = {}
    require(type(c['reference']) is list, 'Explicit reference rows required')
    for r in c['reference']:
        fields(r, 'security_id session member industry available_at source')
        key = _key(r)
        require(key not in refs and key in rows, 'Duplicate/undeclared reference key')
        require(type(r['member']) is bool, 'Membership must be explicit bool')
        require(r['industry'] is None or type(r['industry']) is str and bool(r['industry'].strip()), 'Industry must be string or null')
        timestamp(r['available_at'])
        require(r['available_at'] <= c['cutoffs'][key[1]] and r['source'] in source_ids, 'Unavailable/unbound reference')
        refs[key] = r
    require(set(refs) == history_set, 'Complete frozen reference mask required for history')
    for day in {k[1] for k in history}:
        actual = {k[0]: r['industry'] for k, r in refs.items() if k[1] == day and r['member']}
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
        cells = _old_cells(e, p['event_schema'][e['stream']], source_ids)
        stream = events[e['stream']].setdefault(e['security_id'], [])
        require(all(x[0] != e['event_session'] for x in stream), 'Ambiguous event date: Data must resolve fixed stream')
        stream.append((e['event_session'], e['report_period'], cells))
    for streams in events.values():
        for stream in streams.values(): stream.sort(key=lambda e: e[0])
    return c, history, outputs, rows, by_security, refs, events
