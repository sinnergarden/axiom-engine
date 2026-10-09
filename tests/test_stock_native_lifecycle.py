"""Original public native lifecycle semantics; no Data query or account run."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_rules import validate_execution_rules
from axiom_engine.runtime import StockInputSource, stock_execution_rules
from axiom_engine.runtime.backtest import _simulate
from axiom_engine.runtime.stock_inputs import validate_stock_request
from axiom_engine.runtime.stock_market import _stock_lifecycle_active
from axiom_engine.runtime.stock_stream_contracts import read_budget
from axiom_engine._implementation import IMPLEMENTATION_REF
from test_csi300 import BOARD_IDS, DAYS, rules_for
from test_csi300_runtime import full_request
from test_stock_stream_inputs import source_fixture, rebind_native, LIMITS


def terminal(batches, membership, *, reopen=False):
    security = BOARD_IDS['SSE_MAIN']
    for index in (0, 1):
        for row in batches[index]['records']:
            if row['security_id'] != security or row['session'] < DAYS[2]:
                continue
            if index == 0:
                row['market_state'] = 'normal_trading' if reopen and row['session'] == DAYS[3] else 'delisted'
            else:
                for name in ('open','high','low','close','volume_shares','amount_cny'):
                    row[name] = None
        for name, field in batches[index]['field_meta'].items():
            for meta in field['by_key']:
                if meta['security_id'] != security or meta['session'] < DAYS[2]:
                    continue
                if index == 0:
                    meta.update(missing_reason=None, evidence_domain='listing_events',
                                revision_id='vendor:synthetic-event', raw_batch_id='synthetic-raw')
                    meta.pop('usable_from', None)
                else:
                    meta.update(missing_reason='source_gap', usable_from=None)
    for row in batches[3]['records']:
        if row['security_id'] == security and row['session'] >= DAYS[2]:
            row['factor'] = 2.0 if row['session'] == DAYS[2] else 3.0
    for meta in batches[3]['field_meta']['factor']['by_key']:
        if meta['security_id']==security and meta['session']>=DAYS[2]:
            meta.update(missing_reason=None,usable_from=meta['session']+'T12:00:00Z')
    for row in batches[2]['records']:
        if row['security_id']==security and row['session']>=DAYS[2]:row.update(up_limit=11.0,down_limit=9.0)
    for field in batches[2]['field_meta'].values():
        for meta in field['by_key']:
            if meta['security_id']==security and meta['session']>=DAYS[2]:
                meta.update(missing_reason=None,usable_from=meta['session']+'T02:00:00Z')
    for row in membership['records']:
        if row['security_id'] == security and row['session'] >= DAYS[2]:
            row['is_member'] = False


def terminal_frame(frame):
    for row in frame['rows']:
        if row['security_id'] == BOARD_IDS['SSE_MAIN'] and row['session'] >= DAYS[2]:
            row.update(member=False, valid=False, score=None, invalid_reason='NOT_MEMBER')


def event_rules(mutate=None):
    wire=rules_for();identity=deepcopy(wire['identity_input']);master=identity['source_evidence'][0]['batch']
    cutoff='2026-10-03T16:00:00+00:00'
    master['context']['query'].update(start='1900-01-01',end=DAYS[-1],cutoff=cutoff,
                                     pit_policy='operational_pit_v1',purpose='historical_exploration')
    fields=['exchange','listing_date','delisting_date','event_date','event_state']
    row=dict(security_id=BOARD_IDS['SSE_MAIN'],exchange='SSE',listing_date='2000-01-01',
             delisting_date=DAYS[2],event_date=DAYS[2],event_type='delisting',event_state='value')
    meta=dict(security_id=row['security_id'],event_type='delisting',status='value',missing_reason=None,
              revision_id='vendor:synthetic-event',raw_batch_id='synthetic-raw',
              first_observed_at=cutoff,usable_from=cutoff,availability_basis='first_observed_at')
    event=dict(context=dict(contract_version='data_batch_v1',domain='listing_events',
               snapshot_id=master['context']['snapshot_id'],query=dict(fields=fields,symbols=wire['universe'],
                 start='1900-01-01',end=DAYS[-1],cutoff=cutoff,pit_policy='operational_pit_v1',
                 purpose='historical_exploration',time_field='event_date',filters={'event_type':'delisting'})),
               records=[row],field_meta={n:dict(by_key=[deepcopy(meta)]) for n in fields})
    if mutate:mutate(master,event)
    refs=[Document.from_dict(b).identity for b in (master,event)]
    identity.update(contract_version='stock_execution_identity_v2',source_refs=refs,
                    source_evidence=[dict(reference=ref,batch=batch) for ref,batch in zip(refs,(master,event))])
    for original in identity['rows']:
        original['source_refs']=refs if original['security_id']==row['security_id'] else refs[:1]
        if original['security_id']==row['security_id']:original['delisting_date']=DAYS[2]
    return stock_execution_rules(universe=wire['universe'],calendar=wire['calendar'],identity_input=identity,
        quantity_rules=wire['quantity_rules'],sources=wire['sources'],verified_from=wire['verified_from'],
        verified_through=wire['verified_through'])


class NativeLifecycleTests(unittest.TestCase):
    def test_v2_binds_public_event_boundary_and_original_native_revision(self):
        rules=event_rules();index=validate_execution_rules(rules)[0]
        proof=index[BOARD_IDS['SSE_MAIN']]['_stock_delisting_evidence']
        self.assertEqual((proof['event_date'],proof['availability_basis']),(DAYS[2],'first_observed_at'))
        plan=full_request(rules=rules,mutate_native=terminal,mutate_frame=terminal_frame).to_dict()
        self.assertFalse(validate_stock_request(plan)[5][DAYS[2],BOARD_IDS['SSE_MAIN']]['_stock_listed'])
        with TemporaryDirectory() as tmp:
            manifest=source_fixture(Path(tmp),plan)[0];source=StockInputSource()
            try:
                source.audit(manifest,block_sessions=1,read_budget=read_budget(LIMITS),limits=LIMITS,
                             implementation_ref=IMPLEMENTATION_REF)
            finally:source._close_private_views()
        metadata=dict(evidence_domain='listing_events',revision_id='wrong',raw_batch_id='synthetic-raw')
        with self.assertRaisesRegex(ContractError,'different original event revision'):
            _stock_lifecycle_active(index[BOARD_IDS['SSE_MAIN']],DAYS[2],'delisted',metadata)
        mutations=[lambda m,e:e['context'].update(snapshot_id='different'),
                   lambda m,e:e['records'][0].update(event_date=DAYS[1]),
                   lambda m,e:e['records'][0].update(exchange='SZSE'),
                   lambda m,e:e['field_meta']['event_date']['by_key'][0].update(revision_id='different'),
                   lambda m,e:e['field_meta']['event_date']['by_key'][0].update(availability_basis='declared_vendor_assumption'),
                   lambda m,e:m['records'][0].update(delisting_date=DAYS[3])]
        for mutation in mutations:
            with self.subTest(mutation=mutation),self.assertRaises(ContractError):event_rules(mutation)

    def test_first_observed_basis_cannot_backdate_original_event_in_either_path(self):
        rules = event_rules()
        observed = '2026-10-03T16:00:00+00:00'
        historical = DAYS[2] + 'T00:00:00+08:00'
        baseline = full_request(rules=rules, mutate_native=terminal, mutate_frame=terminal_frame).to_dict()
        cases = [
            (dict(availability_basis='first_observed_at', first_observed_at=observed,
                  usable_from=historical), False),
            (dict(availability_basis='first_observed_at', first_observed_at=historical,
                  usable_from=historical), False),
            (dict(availability_basis='first_observed_at', first_observed_at=observed,
                  usable_from=None), True),
            (dict(availability_basis='declared_vendor_assumption', first_observed_at=observed,
                  usable_from=historical), True),
        ]
        for clocks, allowed in cases:
            def mutate_state(wire):
                for meta in wire['field_meta']['market_state']['by_key']:
                    if meta['security_id'] == BOARD_IDS['SSE_MAIN'] and meta['session'] >= DAYS[2]:
                        meta.update(clocks)
            def mutate_native(batches, membership):
                terminal(batches, membership)
                mutate_state(batches[0])
            with self.subTest(clocks=clocks), patch(
                    'axiom_engine.runtime.backtest.AccountLedger', side_effect=AssertionError('account started')):
                if allowed:
                    plan = full_request(rules=rules, mutate_native=mutate_native,
                                        mutate_frame=terminal_frame).to_dict()
                    before = deepcopy(plan)
                    row = validate_stock_request(plan)[5][DAYS[2], BOARD_IDS['SSE_MAIN']]
                    self.assertEqual(row['field_available_at']['market_state'], clocks['usable_from'])
                    self.assertEqual(plan, before)
                else:
                    with self.assertRaisesRegex(ContractError, 'observation time differs from original event proof'):
                        full_request(rules=rules, mutate_native=mutate_native, mutate_frame=terminal_frame)
                with TemporaryDirectory() as tmp:
                    manifest = source_fixture(Path(tmp), baseline)[0]
                    rebind_native(manifest, 'native-0', mutate_state)
                    source = StockInputSource()
                    try:
                        if allowed:
                            source.audit(manifest, block_sessions=1, read_budget=read_budget(LIMITS),
                                         limits=LIMITS, implementation_ref=IMPLEMENTATION_REF)
                        else:
                            with self.assertRaisesRegex(ContractError, 'observation time differs from original event proof'):
                                source.audit(manifest, block_sessions=1, read_budget=read_budget(LIMITS),
                                             limits=LIMITS, implementation_ref=IMPLEMENTATION_REF)
                    finally:
                        source._close_private_views()

    def test_unknown_is_not_evidence_of_listing_and_true_conflicts_reject(self):
        identity = dict(security_id='synthetic',listing_date=DAYS[1],delisting_date=None)
        self.assertFalse(_stock_lifecycle_active(identity,DAYS[0],'unknown_status',{}))
        self.assertTrue(_stock_lifecycle_active(identity,DAYS[1],'unknown_status',{}))
        evidence = dict(evidence_domain='listing_events',revision_id='original-revision',raw_batch_id='original-raw')
        self.assertFalse(_stock_lifecycle_active(identity,DAYS[2],'delisted',evidence))
        cases = [(identity,DAYS[0],'normal_trading',{}),
                 (identity,DAYS[1],'not_listed',{}),
                 (identity,DAYS[0],'delisted',evidence),
                 ({**identity,'delisting_date':DAYS[3]},DAYS[2],'delisted',evidence),
                 ({**identity,'delisting_date':DAYS[2]},DAYS[2],'normal_trading',{}),
                 (identity,DAYS[2],'delisted',{}),
                 (identity,DAYS[2],'delisted',{**evidence,'usable_from':DAYS[3]+'T00:00:00+08:00'})]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ContractError):
                _stock_lifecycle_active(*args)

    def test_prelisting_unknown_retains_original_reason_and_false_member_is_required(self):
        security = 'cnstock.688200.SH.20240103'
        rules = rules_for({BOARD_IDS['SSE_MAIN']:'SSE_MAIN',security:'SSE_STAR'})
        def mutate(batches,membership):
            for row in batches[0]['records']:
                if row['security_id']==security and row['session']<DAYS[2]:row['market_state']='unknown_status'
            for meta in batches[0]['field_meta']['market_state']['by_key']:
                if meta['security_id']==security and meta['session']<DAYS[2]:
                    meta.update(missing_reason='calendar_source_missing',usable_from=None)
        plan=full_request(rules=rules,k=1,mutate_native=mutate).to_dict()
        before=deepcopy(plan)
        with patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('account started')):
            indexed=validate_stock_request(plan)[5]
        self.assertEqual(plan,before)
        row=indexed[DAYS[0],security]
        self.assertEqual((row['market_state'],row['state_reason'],row['_stock_listed']),
                         ('unknown_status','calendar_source_missing',False))
        def wrong(batches,membership):
            mutate(batches,membership)
            for row in membership['records']:
                if row['security_id']==security:row['is_member']=True
        with self.assertRaises(ContractError):full_request(rules=rules,k=1,mutate_native=wrong)

    def test_original_terminal_evidence_retained_factor_limits_and_stream_admission(self):
        plan=full_request(mutate_native=terminal,mutate_frame=terminal_frame).to_dict()
        before=deepcopy(plan)
        indexed=validate_stock_request(plan)[5]
        row=indexed[DAYS[2],BOARD_IDS['SSE_MAIN']]
        self.assertFalse(row['_stock_listed'])
        self.assertEqual((row['market_state'],row['limit_up'],row['field_available_at']['market_state']),
                         ('delisted','11.0',None))
        self.assertFalse(any(b['security_id']==BOARD_IDS['SSE_MAIN'] for b in plan['market_replay']['action_blocks']))
        self.assertEqual(plan,before)
        with TemporaryDirectory() as tmp,patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('account started')):
            manifest=source_fixture(Path(tmp),plan)[0]
            source=StockInputSource()
            try:
                audit=source.audit(manifest,block_sessions=1,read_budget=read_budget(LIMITS),
                                   limits=LIMITS,implementation_ref=IMPLEMENTATION_REF)
                self.assertEqual(audit.receipt['counts']['market_rows'],len(DAYS)*len(plan['execution_universe']))
            finally:source._close_private_views()
        with self.assertRaisesRegex(ContractError,'reopens'):
            full_request(mutate_native=lambda b,m:terminal(b,m,reopen=True),mutate_frame=terminal_frame)

    def test_stream_reopening_across_blocks_rejects_even_with_resealed_native_ref(self):
        plan=full_request(mutate_native=terminal,mutate_frame=terminal_frame).to_dict()
        with TemporaryDirectory() as tmp:
            manifest=source_fixture(Path(tmp),plan)[0]
            entry=next(item for item in manifest['market_input']['native_inputs']
                       if item['artifact']['artifact_id']=='native-0')
            def reopen(wire):
                for row in wire['records']:
                    if row['security_id']==BOARD_IDS['SSE_MAIN'] and row['session']==DAYS[3]:
                        row['market_state']='unknown_status'
            rebind_native(manifest,entry['artifact']['artifact_id'],reopen)
            source=StockInputSource()
            try:
                with self.assertRaisesRegex(ContractError,'reopens'):
                    source.audit(manifest,block_sessions=1,read_budget=read_budget(LIMITS),
                                 limits=LIMITS,implementation_ref=IMPLEMENTATION_REF)
            finally:source._close_private_views()

    def test_active_unknown_with_block_profile_cannot_submit_or_fill(self):
        plan=full_request().to_dict();row=deepcopy(next(iter(validate_stock_request(plan)[5].values())))
        row.update(market_state='unknown_status',state_reason='status_source_missing')
        intent=dict(security_id=row['security_id'],side='BUY',quantity=200,intent_id='synthetic',
                    valid_until=DAYS[0],expected_account_version=0)
        # The block occurs before ledger access; no ledger is created or advanced.
        order=_simulate(intent,row,SimpleNamespace(),plan['profile'],DAYS[0],'synthetic',0,
                        stock_market={'action_blocks':[]})
        self.assertEqual((order['market_state'],order['reason'],order['submitted_quantity'],order['filled_quantity']),
                         ('unknown_status','UNKNOWN_MARKET_STATUS',0,0))


if __name__=='__main__':unittest.main()
