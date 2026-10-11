"""Execution admission and bounded, ordered operator-plan semantics."""
from .contracts import (ABI, SEMANTICS, ContractError, FeaturePlan, digest, fields,
                        integer, number, require, schema, text, unresolved)

# Each accepted policy is explicit; there is no kwargs/default escape hatch.
PARAMS = {
    'constant': 'value', 'identity': '', 'shift': 'periods',
    'rolling': 'window min_periods inclusive_current reduction missing ddof ties',
    'pct_change': 'periods fill_method zero',
    'add': '', 'sub': '', 'mul': '', 'divide': 'zero', 'abs': '', 'log1p': 'domain',
    'gt': 'missing', 'lt': 'missing', 'eq': 'missing',
    'and': 'missing', 'or': 'missing', 'not': 'missing', 'where': 'missing',
    'to_float': '', 'is_missing': '', 'fill': 'value', 'clip': 'lower upper',
    'cs_rank': 'group unknown_group missing ties excluded',
    'cs_zscore': 'group unknown_group missing ddof epsilon constant clip excluded',
    'cs_winsorize': 'group unknown_group missing lower upper interpolation excluded',
    'asof': 'stream field match report_policy', 'calendar_age': '',
}
BINARY = {'add', 'sub', 'mul', 'divide', 'gt', 'lt', 'eq', 'and', 'or', 'calendar_age'}
CS = {'cs_rank', 'cs_zscore', 'cs_winsorize'}


def validate_cs_zscore_params(q):
    """Shared explicit CS policy admission for Feature and Signal plans."""
    fields(q, PARAMS['cs_zscore'])
    require(q['group'] in ('session', 'industry'), 'CS group required')
    require(q['unknown_group'] in ('missing', 'reject'), 'Unknown industry policy')
    require(q['missing'] in ('skip', 'propagate', 'fill_zero', 'reject'), 'CS missing policy')
    require(q['excluded'] in ('missing', 'zero_if_undefined'), 'Excluded row policy')
    require(type(q['ddof']) is int and q['ddof'] in (0, 1), 'ddof must be 0 or 1')
    number(q['epsilon']); require(q['epsilon'] >= 0, 'Negative epsilon')
    require(q['constant'] in ('zero', 'missing', 'reject'), 'Constant CS policy')
    if q['clip'] is not None:
        require(type(q['clip']) is list and len(q['clip']) == 2, 'Clip bounds')
        for v in q['clip']: number(v)
        require(q['clip'][0] <= q['clip'][1], 'Reversed clip')
    return q


def validate_plan(plan, *, execution=False):
    require(type(plan) is FeaturePlan, 'Expected FeaturePlan')
    p = plan.to_dict()
    fields(p, 'abi semantics recipe_ref calendar_ref reference_ref reference_members input_schema event_schema sources observation_domain history_policy nodes outputs obligations')
    require(type(p['nodes']) is list and type(p['outputs']) is list and bool(p['outputs']),
            'Plan nodes/outputs must be arrays; outputs nonempty')
    require(type(p['obligations']) is list, 'Explicit obligations required')
    for n in p['nodes']:
        fields(n, 'name op version inputs params column')
        require(type(n['inputs']) is list and type(n['params']) is dict, 'Malformed draft node')
        fields(n['column'], 'name dtype unit stage missing')
    for o in p['outputs']:
        fields(o, 'node column')
        fields(o['column'], 'name dtype unit stage missing')
    # Draft retains wire objects in place. Executable plans cannot contain any.
    missing = unresolved(p)
    if missing:
        require(not execution, 'UNRESOLVED: ' + ','.join(x['unknown_id'] for x in missing))
        return p
    require(not p['obligations'], 'Unresolved execution obligations')
    require(p['abi'] == ABI and p['semantics'] == SEMANTICS, 'Unsupported ABI/semantics')
    for key in ('recipe_ref', 'calendar_ref', 'reference_ref'):
        digest(p[key])
    require(p['observation_domain'] in ('sessions', 'observations'), 'Observation domain required')
    require(p['history_policy'] in ('full', 'partial'), 'History policy required')
    require(type(p['reference_members']) is dict and bool(p['reference_members']), 'Frozen reference members required')
    from .contracts import session
    for day, members in p['reference_members'].items():
        session(day)
        require(type(members) is dict, 'Reference members must map securities to industry')
        for security, industry in members.items():
            text(security)
            require(industry is None or type(industry) is str and bool(industry.strip()), 'Invalid frozen industry')
    names = schema(p['input_schema'])
    require(all(c['stage'] == 'fact' for c in p['input_schema']), 'Input stage must be fact')
    require(type(p['event_schema']) is dict, 'Event schema must be an object')
    for name, cols in p['event_schema'].items():
        text(name); schema(cols)
        require(all(c['stage'] == 'fact' for c in cols), 'Event stage must be fact')
    require(type(p['sources']) is list and bool(p['sources']), 'Explicit sources required')
    ids = []
    for s in p['sources']:
        fields(s, 'id data_ref view_ref revision_policy qualification availability_basis')
        for k in ('id', 'revision_policy', 'availability_basis'):
            text(s[k])
        digest(s['data_ref']); digest(s['view_ref'])
        require(s['qualification'] in ('verified', 'observed', 'best_effort', 'synthetic'),
                'Unknown source qualification')
        ids.append(s['id'])
    require(len(ids) == len(set(ids)), 'Duplicate source binding')
    columns = dict(zip(names, p['input_schema']))
    history = {n: 0 for n in names}
    for n in p['nodes']:
        fields(n, 'name op version inputs params column')
        name, op, args, q, col = n['name'], n['op'], n['inputs'], n['params'], n['column']
        text(name)
        require(name not in columns, 'Duplicate node or overwrite')
        require(type(op) is str and op in PARAMS and n['version'] == '1', 'Unknown operator/version')
        require(type(args) is list and all(type(a) is str and a in columns for a in args),
                'Missing/forward node input')
        count = 0 if op in ('constant', 'asof') else 3 if op == 'where' else 2 if op in BINARY else 1
        require(len(args) == count, 'Wrong operator arity')
        fields(q, PARAMS[op]); schema([col])
        require(col['name'] == name and col['stage'] != 'fact', 'Node schema/name mismatch')
        parents = [columns[a] for a in args]
        stage = 'cross_sectional' if op in CS or any(c['stage'] == 'cross_sectional' for c in parents) else 'base'
        require(col['stage'] == stage, 'Stage cannot be lost or relabeled')
        depth = max((history[a] for a in args), default=0)
        if op in ('shift', 'pct_change'):
            integer(q['periods'], 1); depth += q['periods']
        if op == 'rolling':
            integer(q['window'], 1); integer(q['min_periods'], 1)
            require(q['min_periods'] <= q['window'], 'min_periods exceeds window')
            require(type(q['inclusive_current']) is bool, 'Inclusive endpoint must be bool')
            require(q['reduction'] in ('mean', 'sum', 'std', 'min', 'max', 'median', 'rank'), 'Unsupported rolling reduction')
            require(q['missing'] in ('skip', 'propagate'), 'Rolling missing policy')
            require(type(q['ddof']) is int and q['ddof'] in (0, 1), 'ddof must be 0 or 1')
            require(q['ties'] == 'average', 'Unsupported ties')
            require(q['reduction'] == 'std' or q['ddof'] == 0, 'ddof is 0 for non-std reductions')
            depth += q['window'] - int(q['inclusive_current'])
        if op == 'pct_change':
            require(q['fill_method'] == 'none', 'Unresolved/unsupported pct_change fill method')
        if op in ('divide', 'pct_change'):
            require(q['zero'] in ('missing', 'reject'), 'Zero denominator policy required')
        if op in ('gt', 'lt', 'eq', 'and', 'or', 'not', 'where'):
            require(q['missing'] in ('preserve', 'false', 'reject'), 'Boolean missing policy')
        if op == 'log1p':
            require(q['domain'] in ('missing', 'reject'), 'log1p domain policy')
        if op in ('constant', 'fill'):
            from .contracts import check_value
            check_value(q['value'], col)
        if op == 'clip':
            for b in ('lower', 'upper'):
                if q[b] is not None:
                    number(q[b])
            require(q['lower'] is None or q['upper'] is None or q['lower'] <= q['upper'], 'Reversed clip')
        if op in CS:
            require(q['group'] in ('session', 'industry'), 'CS group required')
            require(q['unknown_group'] in ('missing', 'reject'), 'Unknown industry policy')
            require(q['missing'] in ('skip', 'propagate', 'fill_zero', 'reject'), 'CS missing policy')
            allowed = ('missing', 'zero_if_undefined') if op == 'cs_zscore' else ('missing',)
            require(q['excluded'] in allowed, 'Excluded row policy')
            if op == 'cs_rank':
                require(q['ties'] == 'average', 'Rank ties must be average')
            elif op == 'cs_zscore':
                validate_cs_zscore_params(q)
            else:
                number(q['lower']); number(q['upper'])
                require(0 <= q['lower'] <= q['upper'] <= 1 and q['interpolation'] == 'linear',
                        'Quantile bounds/interpolation unsupported')
        if op == 'asof':
            require(q['stream'] in p['event_schema'], 'Missing event stream')
            event_cols = {c['name']: c for c in p['event_schema'][q['stream']]}
            require(q['field'] in event_cols and q['match'] in ('strict_before', 'exact_date'),
                    'Event field/match policy required')
            require(q['report_policy'] in ('event_order', 'nondecreasing'), 'Explicit report-period stream policy required')
            ec = event_cols[q['field']]
            require(col['dtype'] == ec['dtype'] and col['unit'] == ec['unit'], 'As-of schema mismatch')
        # Type safety prevents Python truthiness / implicit unit conversion.
        if op in ('and', 'or', 'not'):
            require(all(c['dtype'] == 'bool' for c in parents), 'Boolean inputs required')
        elif op == 'where':
            require(parents[0]['dtype'] == 'bool' and
                    all(parents[i][k] == col[k] for i in (1, 2) for k in ('unit', 'dtype')),
                    'Where branch schema mismatch')
        elif op == 'calendar_age':
            require(all(c['dtype'] == 'date' for c in parents) and col['unit'] == 'days', 'Date age inputs/unit')
        elif op not in ('asof', 'constant', 'identity', 'shift', 'fill', 'is_missing', 'to_float'):
            require(all(c['dtype'] == 'float64' for c in parents), 'Numeric inputs required')
        if op == 'to_float':
            require(parents[0]['dtype'] == 'bool', 'to_float requires bool')
        expected_dtype = 'bool' if op in ('gt', 'lt', 'eq', 'and', 'or', 'not', 'is_missing') else None
        if expected_dtype:
            require(col['dtype'] == expected_dtype and col['unit'] == 'dimensionless', 'Predicate schema')
        elif op not in ('asof', 'constant', 'identity', 'shift', 'fill', 'where'):
            require(col['dtype'] == 'float64', 'Numeric result dtype')
        if op in ('identity', 'shift', 'fill'):
            require(all(col[k] == parents[0][k] for k in ('dtype', 'unit')), 'Preserving operator schema mismatch')
        if op in ('add', 'sub', 'gt', 'lt', 'eq'):
            require(parents[0]['unit'] == parents[1]['unit'], 'Mixed units')
        preserve_unit = op in ('add', 'sub', 'abs', 'clip', 'cs_winsorize') or op == 'rolling' and q['reduction'] != 'rank'
        if preserve_unit:
            require(col['unit'] == parents[0]['unit'], 'Output unit mismatch')
        dimensionless = op in ('pct_change', 'cs_rank', 'cs_zscore', 'to_float', 'log1p') or op == 'rolling' and q['reduction'] == 'rank'
        if dimensionless:
            require(col['unit'] == 'dimensionless', 'Dimensionless output required')
        columns[name] = col; history[name] = depth
    output_cols = []
    for o in p['outputs']:
        fields(o, 'node column')
        require(o['node'] in columns, 'Missing output dependency')
        col = o['column']; schema([col]); src = columns[o['node']]
        require(all(col[k] == src[k] for k in ('dtype', 'unit', 'stage', 'missing')), 'Projection schema mismatch')
        require(col['stage'] != 'fact', 'Project facts through an identity node first')
        output_cols.append(col)
    schema(output_cols)
    return p


def required_history(plan):
    """Full finite dependency closure in the plan's declared observation domain."""
    p = validate_plan(plan, execution=True)
    depth = {c['name']: 0 for c in p['input_schema']}
    for n in p['nodes']:
        d = max((depth[a] for a in n['inputs']), default=0)
        if n['op'] in ('shift', 'pct_change'): d += n['params']['periods']
        if n['op'] == 'rolling': d += n['params']['window'] - int(n['params']['inclusive_current'])
        depth[n['name']] = d
    return {o['column']['name']: depth[o['node']] for o in p['outputs']}
