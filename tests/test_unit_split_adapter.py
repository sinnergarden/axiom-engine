from copy import deepcopy
from datetime import date
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError
from axiom_engine.runtime import read_etf_market_replay
from test_backtest import DAYS
from test_unit_splits import split_fixture


class Query:
    def __init__(self, domain, fields, symbols, *args, **kwargs):
        self.domain, self.fields, self.symbols = domain, fields, symbols
        self.purpose = kwargs.get('purpose')
        if len(args) == 3:
            self.sessions, self.pit_policy, self.cutoff_by_session = args
        else:
            self.start, self.end, self.cutoff, self.pit_policy, self.time_field = args
        self.filters = {}


class Batch:
    def __init__(self, wire):
        self.wire = wire

    def to_json(self):
        return deepcopy(self.wire)


class SyntheticData:
    """Public facade double: latest result differs, record-cutoff plan is r3."""
    def __init__(self, *, missing_plan=False, extra_transition=False):
        self.wire = split_fixture()
        self.missing_plan, self.extra_transition = missing_plan, extra_transition
        self.calls = []

    def context(self, query):
        return {'contract_version':'data_batch_v1', 'domain':query.domain, 'snapshot_id':'s_synthetic',
                'reader_version':'synthetic_reader', 'query':dict(query.__dict__)}

    def states(self, *, snapshot, query):
        self.calls.append(query)
        records, meta = [], []
        for day in query.sessions:
            closed = date.fromisoformat(day).weekday() >= 5
            for security in query.symbols:
                records.append({'session':day,'security_id':security,'market_state':'calendar_closed' if closed else 'unknown_status'})
                meta.append({'session':day,'security_id':security,'missing_reason':'calendar_closed' if closed else 'status_source_missing'})
        return Batch({'records':records,'field_meta':{'market_state':{'by_key':meta}},'context':self.context(query)})

    def read_market(self, *, snapshot, query):
        self.calls.append(query)
        rows = [r for r in self.wire['market_replay']['rows'] if r['session'] in query.sessions]
        if query.domain == 'market_daily':
            records = [{k:r[k] for k in ('session','security_id','open','close','volume_units')} for r in rows]
            meta = {name:{'unit':'fund units' if name=='volume_units' else 'CNY/fund unit','by_key':[
                {'session':r['session'],'security_id':r['security_id'],'usable_from':r['close_available_at']} for r in rows]} for name in query.fields}
        elif query.domain == 'price_limits':
            records = [{'session':r['session'],'security_id':r['security_id'],'up_limit':r['limit_up'],'down_limit':r['limit_down']} for r in rows]
            meta = {}
        else:
            security = self.wire['signal_frame']['universe'][0]
            records = [{'session':r['session'],'security_id':r['security_id'], 'factor':5.0019 if
                        r['security_id']==security and r['session'] >= (DAYS[2] if self.extra_transition else DAYS[4]) else 1.0} for r in rows]
            meta = {}
        return Batch({'records':records,'field_meta':meta,'context':self.context(query)})

    def events(self, *, snapshot, query):
        self.calls.append(query)
        if query.domain == 'corporate_actions':
            return Batch({'records':[],'field_meta':{'cash_dividend_per_unit':{'unit':'CNY/fund unit'}},'context':self.context(query)})
        self.assert_domain = query.domain
        if 'security_id' in query.fields or 'event_id' in query.fields:
            raise ValueError('event query contains implicit key field')
        source = deepcopy(self.wire['market_replay']['source_evidence'][1]['batch'])
        source['context'] = self.context(query)
        if query.cutoff[:10] > DAYS[2]:
            source['records'][0].update(process_status='implemented', revision_id='later-r4', revision_sequence=4, ratio_numerator=7)
            for column in source['field_meta'].values():
                column['by_key'][0].update(revision_id='later-r4', usable_from=DAYS[4]+'T01:30:00Z')
        elif self.missing_plan:
            source['records'] = []
        return Batch(source)


class UnitAdapterTests(unittest.TestCase):
    def read(self, data, include=True):
        with patch.dict(sys.modules, axiom_data=SimpleNamespace(EventQuery=Query, QuerySpec=Query)):
            return read_etf_market_replay(data, snapshot='s_synthetic', universe=data.wire['signal_frame']['universe'],
                first_session=DAYS[0], end_session=DAYS[4], include_unit_splits=include).to_dict()

    def test_record_cutoff_selects_plan_not_future_result_and_preserves_factor(self):
        data = SyntheticData()
        market = self.read(data)
        event = market['unit_splits'][0]['event']
        self.assertEqual((event['ratio_numerator'],event['revision_sequence'],event['process_status']), (5,3,'planned'))
        self.assertEqual(market['contract_version'],'market_replay_v2')
        self.assertTrue(all(r['market_state']=='unknown_status' for r in market['rows']))
        self.assertTrue(all(q.purpose=='market_replay' and not q.filters for q in data.calls))
        factor = next(e['batch'] for e in market['source_evidence'] if e.get('batch',{}).get('context',{}).get('domain')=='adjustment_factors')
        self.assertIn(5.0019, [r['factor'] for r in factor['records']])
        self.assertEqual(len(data.calls),7)
        checks = next(e['unit_split_result_checks'] for e in market['source_evidence'] if 'unit_split_result_checks' in e)
        self.assertEqual(checks[0]['status'],'MISMATCH')
        self.assertEqual(checks[0]['mismatched_fields'],['ratio_numerator'])
        self.assertEqual(checks[0]['plan_revision_id'],'synthetic-r3')
        self.assertEqual(checks[0]['result_revision_id'],'later-r4')

    def test_missing_visible_plan_and_unexplained_other_transition_block(self):
        with self.assertRaisesRegex(ContractError,'future result cannot unlock'):
            self.read(SyntheticData(missing_plan=True))
        with self.assertRaisesRegex(ContractError,'unexplained factor change'):
            self.read(SyntheticData(extra_transition=True))

    def test_default_v1_keeps_split_admission_block(self):
        data = SyntheticData()
        with self.assertRaisesRegex(ContractError,'unexplained factor change'):
            self.read(data, include=False)
        self.assertEqual(len(data.calls),5)


if __name__ == '__main__':
    unittest.main()
