"""Admission of saved raw/derived signals within the existing stock owner.

Only original bounded source rows are read. Derived arithmetic uses the same
Core entry as Research; this module does not train, predict or run an account.
"""
from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import fields, require, digest
from ..core.stock_portfolio import StockPredictionFrame, instant
from ..core.stock_signal import (DERIVED_FIELDS, _context, validate_signal_plan,
    signal_plan_ref, _SignalPlanAdmission)


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
            source._memory.reserve_global(('derived_reference_index',header['signal_run_ref']),
                512*len(days)*len(universe)+256*len(days))
            core=_SignalPlanAdmission(plan,{a:raw[p['signal_run_ref']]['header'] for a,p in binding['parent_inputs'].items()},
                                      header['context'],universe,days)
            for value in (header['signal_run_ref'],header['score_ref'],header['implementation_ref']):digest(value)
            small={k:v for k,v in header.items() if k not in ('context','signal_plan')}
            frame={'kind':'derived','index':index,'header':small,'original_header':header,'item':binding,'core_admission':core,
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
    for frame in result:
        if frame.get('kind')=='derived':frame['admission']=_DerivedAdmission(source,frame)
    return result,trade_map


class _DerivedAdmission:
    """Invocation-local source identity and verified Core day lifecycle."""
    def __init__(self,source,frame):
        self.source=source;self.frame=frame;self.memory=source._memory
        self.verified_days=set()
        self.index_key=(str(frame['index'].path),frame['item']['signal_artifact']['content_digest'])

    def check_rows(self,feature,rows):
        return self.frame['core_admission'].validate_rows({**self.frame['header'],'rows':rows},feature)

    def read(self,feature,budget,reservation):
        from .stock_stream_inputs import _encoded_size
        from ..core.stock_portfolio import validate_stock_predictions
        source,frame=self.source,self.frame
        require(source._memory is self.memory and source._indexes.get(self.index_key) is frame['index'],
                'Derived admission belongs to its original source lifetime')
        frame['index'].unchanged()
        for parent in frame['parents'].values():
            parent['spec_index'].unchanged();parent['model_index'].unchanged();parent['index'].unchanged()
        rows,_,refs=frame['index'].rows(('rows',),[feature],budget,reservation=reservation)
        self.check_rows(feature,rows)
        refs += [p['parent_binding'] for p in frame['parents'].values() if p['parent_binding'] is not None]
        if feature in self.verified_days:
            # Original row spans still pass source identity checks. This is a
            # capability tied to this source, not a caller/persisted PASS flag.
            return rows,refs
        admitted={}
        with source._memory.stage() as proof:
            proof.reserve(4*(_encoded_size(frame['header'])+sum(_encoded_size(r) for r in rows)+32))
            for alias,parent in frame['parents'].items():
                if feature in parent['features']:
                    values,_,parent_refs=parent['index'].rows(('rows',),[feature],budget,reservation=proof)
                    refs+=parent_refs
                    proof.reserve(4*(_encoded_size(parent['header'])+sum(_encoded_size(r) for r in values)+32))
                    wire,indexed=validate_stock_predictions(StockPredictionFrame.from_dict({**parent['header'],'rows':values}))
                    for row in indexed.values():
                        require(instant(row['simulated_model_available_at'])==instant(parent['model']['simulated_available_at']),
                                'Original parent prediction/model clock mismatch')
                    admitted[alias]=(wire,indexed)
                else:admitted[alias]=(parent['header'],{})
            expected,parents,stage,limitations=frame['core_admission'].execute_day(admitted,feature)
            require(rows==expected,'Saved Derived scores/validity/clocks/refs differ from Core execution')
            require(parents==frame['header']['parent_signal_refs'] and stage==frame['header']['signal_stage'] and
                    limitations==frame['header']['limitations'],'Derived provenance differs from original parents')
        self.verified_days.add(feature)
        return rows,refs


def read_derived(source,frame,feature,budget,reservation):
    admission=frame.get('admission')
    require(isinstance(admission,_DerivedAdmission) and admission.source is source,
            'Original Derived source admission required')
    return admission.read(feature,budget,reservation)


def check_day_rows(frame,feature,rows):
    if frame.get('kind')=='derived':
        return frame['admission'].check_rows(feature,rows)
    from ..core.stock_portfolio import validate_stock_predictions as validate_raw
    return validate_raw(StockPredictionFrame.from_dict({**frame['header'],'rows':rows}))[1]


def prediction_targets(frames):
    """Small source-admitted raw target refs, including original Derived parents."""
    targets={}
    for frame in frames:
        parents=frame['parents'].values() if frame.get('kind')=='derived' else [frame]
        for parent in parents:
            header=parent['header'];ref=header.get('label_spec_ref')
            if ref is not None:digest(ref)
            key=header['signal_run_ref']
            require(key not in targets or targets[key]==ref,'Conflicting admitted raw target refs')
            targets[key]=ref
    return targets


def validate_day(frame, feature, checked, members, metadata, native_ref):
    """Bind row eligibility and original clocks to Data-owned PIT membership."""
    derived=frame.get('kind')=='derived'
    for key,row in checked.items():
        require(row['member']==members[key]['is_member'] and
                instant(row['feature_knowledge_cutoff'])==instant(feature+'T20:30:00+08:00') and
                instant(row['knowledge_cutoff'])==instant(feature+'T21:00:00+08:00'),
                'Saved prediction membership/fixed clock mismatch')
        if derived:
            reference=frame['core_admission'].members[feature][row['security_id']]
            fact=metadata['is_member'][key]
            require(fact.get('usable_from') is not None and
                    instant(reference['available_at'])==instant(fact['usable_from']) and
                    tuple(reference['source_refs'])==(native_ref,),
                    'Derived reference differs from original membership availability')
        else:
            require(instant(row['available_at'])==instant(feature+'T21:00:00+08:00') and
                    instant(row['simulated_model_available_at'])==instant(frame['model']['simulated_available_at']),
                    'Saved prediction membership/fixed clock mismatch')
