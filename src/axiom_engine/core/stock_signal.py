"""Pure saved SignalPlan execution and explicit stock target admission.

Research owns definitions, PIT/maturity selection and saved artifacts. This
module imports neither Research nor Data and uses the existing Core arithmetic.
"""
from copy import deepcopy
from datetime import timezone, timedelta
from hashlib import sha256
from types import MappingProxyType
import math
import json

from .._implementation import IMPLEMENTATION_REF
from .contracts import canonical, digest, fields, integer, number, require, session, text
from .execution import _Cell, _cross_section, _element, _merge, _reference_index
from .plan import validate_cs_zscore_params
from .portfolio import SignalFrame

LABEL_FIELDS = ('label_id semantic_version horizon_sessions start_session_offset end_session_offset '
                'start_price end_price calendar_ref price_basis adjustment_anchor formula normalization costs '
                'corporate_action_policy availability maturity_rule missing_policy')
RAW_FIELDS = ('contract_version signal_run_ref signal_stage score_semantics score_unit feature_ref model_ref '
              'limitations universe rows fold_spec_ref clock_basis label_spec label_spec_ref label_normalization')
DERIVED_FIELDS = ('contract_version signal_run_ref score_ref signal_plan signal_plan_ref parent_signal_refs '
                  'implementation_ref signal_stage score_semantics score_unit universe rows context limitations')
CONTEXT_FIELDS = 'calendar_ref reference_universe reference_universe_ref reference_members cutoff_by_session clock_basis'
ROW_FIELDS = 'security_id session knowledge_cutoff available_at score valid invalid_reason source_refs member feature_knowledge_cutoff'


def _ref(value):
    return 'sha256:'+sha256(canonical(value).encode()).hexdigest()


def _instant(value):
    from .stock_portfolio import instant
    return instant(value)


def _time(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def validate_label_spec(spec):
    """Admit the declared open(f+1)->close(f+h) family, never compute a Label."""
    fields(spec, LABEL_FIELDS)
    text(spec['label_id']); integer(spec['horizon_sessions'], 1)
    require(spec['semantic_version'] == '1' and type(spec['start_session_offset']) is int and
            spec['start_session_offset'] == 1 and type(spec['end_session_offset']) is int and
            spec['end_session_offset'] == spec['horizon_sessions'], 'Unsupported stock Label endpoints/version')
    digest(spec['calendar_ref']); session(spec['adjustment_anchor'])
    expected = dict(start_price='open', end_price='close', price_basis='common_anchor_adjusted_v1',
        formula='close(f+h) / open(f+1) - 1', normalization='none', costs='none',
        corporate_action_policy='factor_ratio_no_separate_cashflow',
        availability='max_endpoint_price_factor_anchor_usable_from',
        maturity_rule='all_outcome_dependencies_strictly_before_fit_cutoff',
        missing_policy='invalid_null_preserve_grid')
    require(all(spec[k] == v for k,v in expected.items()), 'Unsupported explicit stock Label policy')
    return deepcopy(spec)


def validate_label_normalization(value):
    fields(value, 'operator operator_version params')
    require(value['operator_version'] == '1', 'Unsupported target normalization version')
    if value['operator'] == 'identity':
        fields(value['params'], '')
    else:
        require(value['operator'] == 'cs_zscore', 'Unsupported target normalization')
        validate_cs_zscore_params(value['params'])
        require(value['params']['group'] == 'session', 'Target normalization must declare session groups')
    return deepcopy(value)


def _rows(wire, *, raw):
    from .stock_portfolio import source_ref, _prediction_clock_v2
    universe = wire['universe']
    require(type(universe) is list and bool(universe) and len(universe) == len(set(universe)), 'Frozen Signal union required')
    for value in universe: text(value)
    require(type(wire['limitations']) is list and all(type(v) is str for v in wire['limitations']), 'Signal limitations required')
    require(type(wire['rows']) is list and bool(wire['rows']), 'Signal rows required')
    indexed, groups, clocks = {}, {}, {}
    model_clock = None
    for row in wire['rows']:
        fields(row, ROW_FIELDS + (' feature_available_at simulated_model_available_at' if raw else ''))
        session(row['session']); require(row['security_id'] in universe, 'Signal outside frozen union')
        cutoff, available, feature = map(_instant, (row['knowledge_cutoff'],row['available_at'],row['feature_knowledge_cutoff']))
        require(feature <= cutoff and available <= cutoff and
                _instant(row['session']+'T00:00:00+08:00') <= feature <= cutoff <
                _instant(row['session']+'T00:00:00+08:00')+timedelta(days=1), 'Signal clock/session conflict')
        require(clocks.setdefault(row['session'], (feature,cutoff)) == (feature,cutoff), 'Inconsistent Signal session clocks')
        if raw:
            _, current_model_clock = _prediction_clock_v2(row, cutoff, available)
            if model_clock is None: model_clock = current_model_clock
            require(current_model_clock == model_clock, 'Inconsistent raw Signal model clock')
        require(type(row['valid']) is bool and type(row['member']) is bool, 'Signal validity/member must be explicit')
        if row['valid']:
            number(row['score']); require(row['invalid_reason'] is None, 'Valid Signal has invalid reason')
        else:
            require(row['score'] is None, 'Invalid Signal carries score'); text(row['invalid_reason'])
        require(type(row['source_refs']) is list and bool(row['source_refs']), 'Signal source refs required')
        for ref in row['source_refs']: source_ref(ref)
        if raw:
            require(all(wire[k] in row['source_refs'] for k in ('feature_ref','model_ref')), 'Raw Signal lacks real model/Feature refs')
        key = (row['session'],row['security_id'])
        require(key not in indexed, 'Duplicate Signal key')
        indexed[key] = row; groups.setdefault(row['session'],set()).add(row['security_id'])
    require(all(v == set(universe) for v in groups.values()), 'Incomplete Signal union coverage')
    return indexed


def validate_prediction_v3(frame):
    wire = frame.to_dict(); fields(wire, RAW_FIELDS)
    require(wire['contract_version'] == 'stock_prediction_run_v3' and wire['signal_stage'] == 'raw_prediction' and
            wire['score_unit'] == 'dimensionless' and wire['clock_basis'] == 'declared_simulation', 'Unsupported v3 prediction')
    text(wire['score_semantics'])
    for k in ('signal_run_ref','feature_ref','model_ref','fold_spec_ref','label_spec_ref'): digest(wire[k])
    validate_label_spec(wire['label_spec']); validate_label_normalization(wire['label_normalization'])
    require(wire['label_spec_ref'] == _ref(wire['label_spec']), 'Prediction LabelSpec identity mismatch')
    return wire, _rows(wire, raw=True)


def _semantic(value):
    """Original Research contract semantic wire; metadata and Artifact URI are locators."""
    if type(value) is list: return [_semantic(v) for v in value]
    if type(value) is not dict: return value
    require(value.get('contract_type') != 'Unknown', 'Unresolved SignalPlan cannot execute')
    return {k:_semantic(v) for k,v in value.items() if not (value.get('contract_type') and
        (k == 'metadata' or value.get('contract_type') == 'ArtifactRef' and k == 'uri'))}


def signal_plan_ref(plan):
    # Research's serialized type markers are not part of Core's neutral
    # Document ABI. Restore only the known contract positions for its original
    # semantic identity; saved Core wire remains marker-free.
    tagged=deepcopy(plan)
    tagged['contract_type']='SignalPlanSpec'
    for item in tagged['inputs']:
        item['contract_type']='SignalInput'
        if 'return_start_rule' in item['label']:
            item['label']['contract_type']='LabelSpec'
            item['label']['maturity']['contract_type']='MaturitySpec'
    for node in tagged['nodes']: node['contract_type']='SignalNode'
    payload=json.dumps(_semantic(tagged),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return 'sha256:'+sha256(payload.encode()).hexdigest()


def _neutral(value):
    if type(value) is list: return [_neutral(v) for v in value]
    if type(value) is not dict: return value
    require(value.get('contract_type')!='Unknown','Unresolved SignalPlan cannot execute')
    return {k:_neutral(v) for k,v in value.items() if k!='contract_type'}


def _contract(value, kind, names, *, version='1'):
    fields(value, ('contract_type ' if 'contract_type' in value else '')+'contract_version metadata '+names)
    require(value.get('contract_type',kind) == kind and value['contract_version'] == version and
            type(value['metadata']) is dict, 'Unsupported original '+kind+' wire')


def validate_signal_plan(plan):
    """Validate original daily_zscore/weighted_combine plans without importing Research."""
    require(type(plan) is dict,'SignalPlanSpec wire must be a dictionary')
    plan = deepcopy(plan); _semantic(plan)
    _contract(plan,'SignalPlanSpec','name key inputs nodes output join_policy score_semantics available_time_semantics')
    text(plan['name']); text(plan['score_semantics']); text(plan['available_time_semantics'])
    require(plan['key'] == ['security_id','session'] and
            plan['join_policy'] in ('inner_on_security_session','outer_on_security_session'), 'Explicit Signal key/join required')
    require(type(plan['inputs']) is list and bool(plan['inputs']) and type(plan['nodes']) is list, 'Signal inputs/nodes required')
    stages = {}
    for item in plan['inputs']:
        _contract(item,'SignalInput','alias label source_stage score_semantics')
        text(item['alias']); text(item['score_semantics'])
        require(item['alias'] not in stages and item['source_stage'] == 'raw_prediction', 'Duplicate/unsupported raw Signal alias')
        require(type(item['label']) is dict, 'Declared input LabelSpec required')
        integer(item['label'].get('horizon_sessions'),1)
        if 'return_start_rule' in item['label']:
            _match_label(item['label'],dict(horizon_sessions=item['label']['horizon_sessions'],
                formula='close(f+h) / open(f+1) - 1',price_basis='common_anchor_adjusted_v1',
                corporate_action_policy='factor_ratio_no_separate_cashflow',missing_policy='invalid_null_preserve_grid',
                maturity_rule='all_outcome_dependencies_strictly_before_fit_cutoff',
                availability='max_endpoint_price_factor_anchor_usable_from'))
        else: validate_label_spec(item['label'])
        stages[item['alias']] = item['source_stage']
    for node in plan['nodes']:
        _contract(node,'SignalNode','name op inputs input_stages output_stage weights reference_universe missing_policy parameters')
        text(node['name']); text(node['reference_universe'])
        require(node['name'] not in stages and type(node['inputs']) is list and bool(node['inputs']) and
                len(node['inputs']) == len(set(node['inputs'])) and type(node['input_stages']) is list and
                len(node['inputs']) == len(node['input_stages']), 'Duplicate/invalid Signal node inputs')
        require(all(name in stages and stages[name] == stage for name,stage in zip(node['inputs'],node['input_stages'])), 'Signal stage/parent mismatch')
        if node['op'] == 'daily_zscore':
            require(node['input_stages'] == ['raw_prediction'] and node['output_stage'] == 'daily_zscore' and
                    node['weights'] == [], 'daily_zscore requires raw model scores')
            validate_cs_zscore_params(node['parameters'])
            require(node['parameters']['group'] == 'session' and node['missing_policy'] == node['parameters']['missing'], 'Signal CS policy conflict')
        else:
            require(node['op'] == 'weighted_combine' and node['output_stage'] == 'final' and
                    type(node['weights']) is list and len(node['weights']) == len(node['inputs']), 'Unsupported Signal blend')
            for weight in node['weights']: number(weight); require(weight >= 0, 'Negative blend weight')
            require(math.isclose(sum(node['weights']),1.0,abs_tol=1e-12) and node['missing_policy'] == 'propagate', 'Explicit original blend weight/missing rule required')
            fields(node['parameters'],'')
        stages[node['name']] = node['output_stage']
    require(plan['output'] in stages and plan['output'] not in {v['alias'] for v in plan['inputs']}, 'Signal output must name a node')
    return _neutral(plan)


def _match_label(declared, spec):
    if 'return_start_rule' not in declared:
        require(validate_label_spec(declared) == spec, 'Signal input complete LabelSpec differs')
        return
    names=('name key horizon_sessions feature_session formula return_start_rule return_end_rule '
        'return_start_offset_sessions return_end_offset_sessions price_basis benchmark_semantics corporate_action_semantics '
        'normalization_policy maturity missing_delisting_policy')
    if declared.get('contract_version')=='1':
        _contract(declared,'LabelSpec',names)
        integer(declared['horizon_sessions'],1)
        start,end=declared['return_start_offset_sessions'],declared['return_end_offset_sessions']
        require(type(start) is int and type(end) is int and start>=0 and
                end-start==declared['horizon_sessions'],'Legacy LabelSpec v1 endpoint-distance mismatch')
        require(False,'Legacy LabelSpec v1 endpoint-distance target cannot map to open(f+1)->close(f+h)')
    _contract(declared,'LabelSpec',names,version='2')
    require(declared['key']==['security_id','session'] and type(declared['return_start_offset_sessions']) is int and
            type(declared['return_end_offset_sessions']) is int, 'LabelSpec key/offset types mismatch')
    require(declared['formula'] in (spec['formula'],f"close(f+{spec['horizon_sessions']}) / open(f+1) - 1"),
            'Signal input complete Label formula differs')
    expected=dict(horizon_sessions=spec['horizon_sessions'],
        return_start_rule='next_session_open',return_end_rule='horizon_session_close',
        return_start_offset_sessions=1,return_end_offset_sessions=spec['horizon_sessions'],
        price_basis=spec['price_basis'],benchmark_semantics='absolute_return',
        corporate_action_semantics=spec['corporate_action_policy'],normalization_policy='none',
        missing_delisting_policy=spec['missing_policy'])
    require(all(declared[k]==v for k,v in expected.items()), 'Signal input complete Label semantics differ')
    maturity=declared['maturity']
    _contract(maturity,'MaturitySpec','rule lag_sessions calendar_policy availability_rule')
    require(maturity['rule']==spec['maturity_rule'] and maturity['availability_rule']==spec['availability'] and
            maturity['calendar_policy']=='actual_exchange_sessions' and type(maturity['lag_sessions']) is int and
            maturity['lag_sessions']==spec['horizon_sessions'], 'Signal input Label maturity differs')


def _context(context, universe, days):
    fields(context,CONTEXT_FIELDS)
    digest(context['calendar_ref']); digest(context['reference_universe_ref']); text(context['reference_universe'])
    require(context['clock_basis'] == 'declared_simulation' and type(context['reference_members']) is dict and
            set(context['reference_members']) == set(days) and type(context['cutoff_by_session']) is dict and
            set(context['cutoff_by_session']) == set(days), 'Complete Signal reference/cutoff scope required')
    refs, actual_sources = {}, {}
    for day in sorted(days):
        cutoff = _instant(context['cutoff_by_session'][day]); rows = context['reference_members'][day]
        require(type(rows) is list and len(rows) == len(universe), 'Complete Signal reference union required')
        seen=set(); actual_sources[day]=set()
        for row in rows:
            fields(row,'security_id member available_at source_refs')
            security=row['security_id']; require(security in universe and security not in seen and type(row['member']) is bool, 'Invalid Signal reference key/member')
            seen.add(security); available=_instant(row['available_at'])
            require(available <= cutoff and type(row['source_refs']) is list and bool(row['source_refs']), 'Unavailable Signal reference')
            for value in row['source_refs']: digest(value)
            actual_sources[day].update(row['source_refs'])
            refs[security,day]={'member':row['member'],'industry':None,'available_at':available,'source':context['reference_universe_ref']}
    return refs,actual_sources


def execute_signal_plan(plan, inputs, context):
    """Execute one explicitly keyed saved-signal plan using the shared Core math.

    No file I/O, inference, fit or account access. Parent model-target zscore
    normalization does not change its raw_prediction post-processing stage.
    """
    from .stock_portfolio import StockPredictionFrame, validate_stock_predictions
    plan=validate_signal_plan(plan); context=deepcopy(context)
    require(type(inputs) is dict and set(inputs) == {v['alias'] for v in plan['inputs']}, 'Exact Signal alias bindings required')
    admitted = {}
    for item in plan['inputs']:
        value=inputs[item['alias']]
        require(isinstance(value,StockPredictionFrame), 'Saved raw StockPredictionFrame required')
        wire,indexed=validate_stock_predictions(value)
        unsigned=dict(wire); reference=unsigned.pop('signal_run_ref')
        require(_ref(unsigned) == reference, 'Raw Signal identity mismatch')
        admitted[item['alias']] = (wire,indexed)
    return _execute_signal_plan_admitted(plan, admitted, context)


def _execute_signal_plan_admitted(plan, admitted, context):
    """Runtime-private row projection after original source identity admission.

    The public entry never accepts row slices in place of original parent hashes.
    Both paths execute this one arithmetic implementation.
    """
    plan=validate_signal_plan(plan); context=deepcopy(context)
    days=set.intersection(*[{d for d,_ in indexed} for _,indexed in admitted.values()]) if plan['join_policy']=='inner_on_security_session' else set.union(*[{d for d,_ in indexed} for _,indexed in admitted.values()])
    first=next(iter(admitted.values()))[0]
    admission=_SignalPlanAdmission(plan,{a:w for a,(w,_) in admitted.items()},context,first['universe'],days)
    rows,parents,stage,limitations=_signal_plan_rows(admission._plan,admitted,context,admission)
    wire=dict(contract_version='derived_signal_run_v1',score_ref=_ref(rows),signal_plan=plan,signal_plan_ref=signal_plan_ref(plan),
        parent_signal_refs=parents,implementation_ref=IMPLEMENTATION_REF,signal_stage=stage,score_semantics=plan['score_semantics'],
        score_unit='dimensionless',universe=first['universe'],rows=rows,context=context,limitations=limitations)
    wire['signal_run_ref']=_ref(wire)
    return SignalFrame.from_dict(wire)


def _freeze(value):
    if type(value) is dict:return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if type(value) is list:return tuple(_freeze(v) for v in value)
    if type(value) is set:return frozenset(value)
    return value


class _SignalPlanAdmission:
    """Private immutable complete-context admission, reused by the same day kernel.

    Construction always validates the original complete scope. There is no
    public row-slice or boolean admission override.
    """
    def __init__(self, plan, parents, context, universe, days):
        plan=validate_signal_plan(plan)
        require(set(parents)=={v['alias'] for v in plan['inputs']},'Exact original Signal parents required')
        for item in plan['inputs']:
            wire=parents[item['alias']]
            require(wire['contract_version'] in ('stock_prediction_run_v2','stock_prediction_run_v3') and
                    wire['score_semantics']==item['score_semantics'] and wire['score_unit']=='dimensionless' and
                    wire['universe']==universe,'Original raw Signal semantics/union required')
            digest(wire['signal_run_ref'])
            horizon=wire['label_spec']['horizon_sessions'] if wire['contract_version'].endswith('_v3') else 5
            require(item['label']['horizon_sessions']==horizon,'Signal input Label horizon mismatch')
            if wire['contract_version'].endswith('_v3'):
                _match_label(item['label'],wire['label_spec'])
                require(wire['label_spec']['calendar_ref']==context['calendar_ref'],'Signal target/reference calendars differ')
        refs,sources=_context(context,universe,days)
        self._plan=_freeze(plan);self._universe=tuple(universe)
        self._parents=MappingProxyType({a:w['signal_run_ref'] for a,w in parents.items()})
        self._static=_freeze({k:v for k,v in context.items() if k not in ('reference_members','cutoff_by_session')})
        self._cutoffs=MappingProxyType(dict(context['cutoff_by_session']))
        grouped={d:{} for d in days};members={d:{} for d in days}
        for (security,day),row in refs.items():grouped[day][security]=row
        for day,rows in context['reference_members'].items():
            for row in rows:members[day][row['security_id']]=row
        self._refs=_freeze(grouped);self.members=_freeze(members);self._sources=_freeze(sources)

    def _scope(self, universe, days):
        require(tuple(universe)==self._universe and set(days)<=self._cutoffs.keys(),'Signal scope differs from original admission')
        return ({(security,day):row for day in days for security,row in self._refs[day].items()},
                {day:self._sources[day] for day in days})

    def execute_day(self, admitted, day):
        require({a:w['signal_run_ref'] for a,(w,_) in admitted.items()}==dict(self._parents),'Original admitted parents differ')
        context={**self._static,'cutoff_by_session':{day:self._cutoffs[day]}}
        return _signal_plan_rows(self._plan,admitted,context,self)

    def validate_rows(self, wire, day):
        rows=_rows(wire,raw=False)
        for (session,security),row in rows.items():
            require(session==day and row['knowledge_cutoff']==self._cutoffs[day] and
                    row['member']==self.members[day][security]['member'],'Derived row differs from original context')
        return rows


def _signal_plan_rows(plan, admitted, context, admission):
    require(isinstance(admission,_SignalPlanAdmission),'Original Core Signal admission required')
    require(set(admitted)=={v['alias'] for v in plan['inputs']}, 'Exact admitted Signal aliases required')
    parents, env, key_sets, universe, feature_clocks = {}, {}, [], None, {}
    parent_rows = {}
    limitations=set()
    for item in plan['inputs']:
        wire,indexed=admitted[item['alias']]
        require(wire['contract_version'] in ('stock_prediction_run_v2','stock_prediction_run_v3') and
                wire['score_semantics'] == item['score_semantics'], 'Original raw Signal semantics required')
        reference=wire['signal_run_ref']; digest(reference)
        parent_rows[item['alias']]=indexed
        if universe is None: universe=wire['universe']
        require(wire['universe'] == universe and wire['score_unit'] == 'dimensionless', 'Common frozen Signal union/unit required')
        parents[item['alias']]=reference; values={}; limitations.update(wire['limitations'])
        for (day,security),row in indexed.items():
            key=(security,day); clock=_instant(row['feature_knowledge_cutoff'])
            require(feature_clocks.setdefault(day,clock) == clock, 'Parent Feature cutoffs differ')
            values[key]=_Cell(row['score'],_instant(row['available_at']),tuple(sorted(set(row['source_refs']+[reference]))),(),row['invalid_reason'])
        env[item['alias']]=values; key_sets.append(set(values))
    keys=set.intersection(*key_sets) if plan['join_policy']=='inner_on_security_session' else set.union(*key_sets)
    require(bool(keys), 'Empty Signal join'); days={key[1] for key in keys}
    require(keys == {(security,day) for security in universe for day in days}, 'Signal output must preserve full session union')
    refs,actual_sources=admission._scope(universe,days); ordered=sorted(keys)
    for item in plan['inputs']:
        source=env[item['alias']]
        for key in ordered:
            day=key[1]; cutoff=_instant(context['cutoff_by_session'][day])
            require(feature_clocks[day] <= cutoff, 'Parent Feature after Signal cutoff')
            if key in source:
                row=parent_rows[item['alias']][day,key[0]]
                require(_instant(row['knowledge_cutoff'])==cutoff and row['member']==refs[key]['member'],
                        'Parent Signal cutoff/membership differs from reference context')
                require(source[key].available <= cutoff, 'Parent Signal unavailable at original cutoff')
            else: source[key]=_Cell(None,None,(parents[item['alias']],),(),'MISSING_SIGNAL_INPUT')
    reference_index=_reference_index(refs)
    for node in plan['nodes']:
        require(node['reference_universe'] == context['reference_universe'], 'Signal node reference universe differs')
        if node['op']=='daily_zscore':
            output=_cross_section('cs_zscore',env[node['inputs'][0]],ordered,refs,node['parameters'],reference_index)
            for key,cell in output.items():
                output[key]=_merge(cell.value,[cell,_Cell(None,None,tuple(sorted(actual_sources[key[1]])))],cell.reason)
        else:
            output={}
            for key in ordered:
                terms=[_element('mul',[env[name][key],_element('constant',[],{'value':weight})],{})
                       for name,weight in zip(node['inputs'],node['weights'])]
                value=terms[0]
                for term in terms[1:]: value=_element('add',[value,term],{})
                output[key]=value
        env[node['name']]=output
    rows=[]
    for day in sorted(days):
        for security in universe:
            cell=env[plan['output']][security,day]
            # A blend also binds eligibility facts, even without a CS node.
            cell=_merge(cell.value,[cell,_Cell(None,refs[security,day]['available_at'],tuple(sorted(actual_sources[day])))],cell.reason)
            require(cell.available is not None and cell.available <= _instant(context['cutoff_by_session'][day]), 'Derived Signal future dependency')
            rows.append(dict(security_id=security,session=day,knowledge_cutoff=context['cutoff_by_session'][day],
                feature_knowledge_cutoff=_time(feature_clocks[day]),available_at=_time(cell.available),score=cell.value,
                valid=cell.value is not None,invalid_reason=None if cell.value is not None else cell.reason or 'MISSING_SIGNAL_INPUT',
                source_refs=list(cell.sources),member=refs[security,day]['member']))
    stage=next(n['output_stage'] for n in plan['nodes'] if n['name']==plan['output'])
    return rows,parents,stage,sorted(limitations)


def validate_derived_signal(frame):
    """Validate a complete Derived wire; the caller verifies original artifact hashes."""
    wire=frame.to_dict(); fields(wire,DERIVED_FIELDS)
    require(wire['contract_version']=='derived_signal_run_v1' and wire['score_unit']=='dimensionless', 'Unsupported Derived Signal')
    plan=validate_signal_plan(wire['signal_plan'])
    require(wire['signal_plan_ref']==signal_plan_ref(plan) and wire['score_semantics']==plan['score_semantics'], 'Derived Signal plan identity/semantics mismatch')
    stage=next(n['output_stage'] for n in plan['nodes'] if n['name']==plan['output'])
    require(wire['signal_stage']==stage and type(wire['parent_signal_refs']) is dict and
            set(wire['parent_signal_refs'])=={i['alias'] for i in plan['inputs']}, 'Derived Signal stage/parent aliases mismatch')
    for ref in [wire['signal_run_ref'],wire['score_ref'],wire['implementation_ref'],*wire['parent_signal_refs'].values()]: digest(ref)
    rows=_rows(wire,raw=False)
    context=wire['context']; refs,_=_context(context,wire['universe'],context['cutoff_by_session'])
    for (day,security),row in rows.items():
        require(row['knowledge_cutoff']==context['cutoff_by_session'].get(day) and
                row['member']==refs[security,day]['member'], 'Derived row differs from saved context')
    return wire,rows
