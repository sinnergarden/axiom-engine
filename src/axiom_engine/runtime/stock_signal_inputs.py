"""Admission of saved raw/derived signals within the existing stock owner.

Only original bounded source rows are read. Derived arithmetic uses the same
Core entry as Research; this module does not train, predict or run an account.
"""
from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import fields, require, digest
from ..core.stock_portfolio import StockPredictionFrame, instant, validate_stock_predictions
from ..core.stock_signal import (DERIVED_FIELDS, _context, validate_signal_plan,
    signal_plan_ref, _execute_signal_plan_admitted)


def frames(source, manifest, budget):
    from .stock_stream_inputs import _frames, _encoded_size
    from .stock_stream_contracts import prediction_bindings
    calendar=manifest['scope']['calendar']; universe=manifest['scope']['prediction_universe']
    following={calendar[i-1]:day for i,day in enumerate(calendar) if i}
    raw={}
    for binding in prediction_bindings(manifest['prediction_input']):
        subset={**manifest,'prediction_input':{**manifest['prediction_input'],'frames':[binding]}}
        admitted,mapping=_frames(source,subset,budget,require_coverage=False,raw_only=True)
        require(len(admitted)==1,'One original raw binding required')
        frame=admitted[0];frame['features']={feature for _,feature in mapping.values()}
        require(binding['signal_run_ref'] not in raw,'Ambiguous original raw Signal binding')
        raw[binding['signal_run_ref']]=frame
    result=[];trade_map={};last=None
    for binding in manifest['prediction_input']['frames']:
        if binding['kind']=='raw':
            frame=raw[binding['signal_run_ref']]
            days=sorted(frame['features'])
        else:
            index=source._index(binding['signal_artifact'],budget)
            header=index.object_header(('rows',));fields(header,DERIVED_FIELDS.replace(' rows',''))
            require(header['contract_version']=='derived_signal_run_v1' and
                    header['signal_run_ref']==binding['signal_run_ref']==index.unsigned_digest('signal_run_ref') and
                    header['score_ref']==index.span_digest(('rows',)) and
                    all(header[k]==binding[k] for k in ('signal_plan_ref','score_ref','implementation_ref','signal_stage')),
                    'Original Derived Signal identity mismatch')
            require(header['implementation_ref']==IMPLEMENTATION_REF and header['universe']==universe and
                    header['score_unit']=='dimensionless', 'Derived implementation/union/unit mismatch')
            plan=validate_signal_plan(header['signal_plan'])
            aliases={item['alias'] for item in plan['inputs']}
            require(aliases==set(binding['parent_inputs'])==set(header['parent_signal_refs']) and
                    header['parent_signal_refs']=={a:p['signal_run_ref'] for a,p in binding['parent_inputs'].items()} and
                    header['signal_plan_ref']==signal_plan_ref(plan) and
                    header['score_semantics']==plan['score_semantics'], 'Derived original plan/parent linkage mismatch')
            stage=next(n['output_stage'] for n in plan['nodes'] if n['name']==plan['output'])
            require(header['signal_stage']==stage,'Derived plan/output stage mismatch')
            days=sorted(d for path,d in index.groups if path==('rows',))
            _context(header['context'],universe,days)
            for value in (header['signal_run_ref'],header['score_ref'],header['implementation_ref']):digest(value)
            frame={'kind':'derived','index':index,'header':header,'item':binding,
                   'parents':{a:raw[p['signal_run_ref']] for a,p in binding['parent_inputs'].items()},'features':set(days)}
        require(bool(days) and all(d in following for d in days),'Saved Signal predecessor scope mismatch')
        trades=[following[d] for d in days]
        require(last is None or last<trades[0],'Overlapping/unordered saved Signal scopes')
        last=trades[-1];result.append(frame)
        for trade,feature in zip(trades,days):
            require(trade not in trade_map,'Duplicate Signal trade date')
            trade_map[trade]=(frame,feature)
    required=calendar[calendar.index(manifest['scope']['start_session']):calendar.index(manifest['scope']['end_session'])+1]
    require(set(trade_map)==set(required),'Missing/excess saved Signal OOS trade date')
    # Retained metadata and the full declared CS reference context are quota
    # charged. A large context must fit caller limits before account execution.
    for reference,frame in raw.items():
        source._memory.reserve_global(('raw_signal_metadata',reference),
            _encoded_size([frame['header'],frame['spec'],frame['model']])+128*len(frame['features']))
    for frame in result:
        if frame.get('kind')=='derived':
            source._memory.reserve_global(('derived_signal_metadata',frame['header']['signal_run_ref']),
                _encoded_size(frame['header'])+128*len(frame['features']))
    return result,trade_map


def read_derived(source, frame, feature, budget, reservation):
    """Read one score cross-section and recompute from admitted real parents."""
    from .stock_stream_inputs import _encoded_size
    frame['index'].unchanged()
    rows,_,refs=frame['index'].rows(('rows',),[feature],budget,reservation=reservation)
    admitted={};context=frame['header']['context']
    scoped={**context,'reference_members':{feature:context['reference_members'][feature]},
            'cutoff_by_session':{feature:context['cutoff_by_session'][feature]}}
    with source._memory.stage() as proof:
        proof.reserve(4*(_encoded_size(frame['header'])+sum(_encoded_size(r) for r in rows)+32))
        for alias,parent in frame['parents'].items():
            parent['spec_index'].unchanged();parent['model_index'].unchanged();parent['index'].unchanged()
            binding=parent['parent_binding']
            if binding is not None:
                require(parent['spec_index'].span_digest(parent['spec_index'].selector)==binding['child_ref'],
                        'Original saved parent fold changed after admission')
                refs.append(binding)
            if feature in parent['features']:
                values,_,parent_refs=parent['index'].rows(('rows',),[feature],budget,reservation=proof)
                refs+=parent_refs
                proof.reserve(4*(_encoded_size(parent['header'])+sum(_encoded_size(r) for r in values)+32))
                wire,indexed=validate_stock_predictions(StockPredictionFrame.from_dict({**parent['header'],'rows':values}))
                for row in indexed.values():
                    require(instant(row['simulated_model_available_at'])==instant(parent['model']['simulated_available_at']),
                            'Original parent prediction/model clock mismatch')
                admitted[alias]=(wire,indexed)
            else: admitted[alias]=(parent['header'],{})
        expected=_execute_signal_plan_admitted(frame['header']['signal_plan'],admitted,scoped).to_dict()
        require(rows==expected['rows'],'Saved Derived scores/validity/clocks/refs differ from Core execution')
        require(expected['parent_signal_refs']==frame['header']['parent_signal_refs'] and
                expected['signal_stage']==frame['header']['signal_stage'] and
                expected['limitations']==frame['header']['limitations'],'Derived provenance differs from original parents')
    return rows,refs


def validate_day(frame, feature, checked, members, metadata, native_ref):
    """Bind row eligibility and original clocks to Data-owned PIT membership."""
    derived=frame.get('kind')=='derived'
    for key,row in checked.items():
        require(row['member']==members[key]['is_member'] and
                instant(row['feature_knowledge_cutoff'])==instant(feature+'T20:30:00+08:00') and
                instant(row['knowledge_cutoff'])==instant(feature+'T21:00:00+08:00'),
                'Saved prediction membership/fixed clock mismatch')
        if derived:
            reference=next(r for r in frame['header']['context']['reference_members'][feature]
                           if r['security_id']==row['security_id'])
            fact=metadata['is_member'][key]
            require(fact.get('usable_from') is not None and
                    instant(reference['available_at'])==instant(fact['usable_from']) and
                    reference['source_refs']==[native_ref],
                    'Derived reference differs from original membership availability')
        else:
            require(instant(row['available_at'])==instant(feature+'T21:00:00+08:00') and
                    instant(row['simulated_model_available_at'])==instant(frame['model']['simulated_available_at']),
                    'Saved prediction membership/fixed clock mismatch')
