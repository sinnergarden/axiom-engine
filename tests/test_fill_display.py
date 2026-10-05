"""Synthetic saved-byte boundaries; no production Reader, transform or account replay."""
from copy import deepcopy
from decimal import Inexact, getcontext, setcontext
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document
from axiom_engine.runtime import (BacktestRequest, run_backtest, SavedReviewDisplay, FillDisplayReport,
    read_review_display, build_fill_display, save_fill_display, load_fill_display)
from axiom_engine.runtime.fill_display import _file_digest, _basis_reason
from test_stocks import request, SNAPSHOT


def files(directory, run, *, unit="CNY/share", scale=0.25, mutate=None):
    saved = run.to_dict();market=saved['plan']['market_replay']
    days=market['calendar'];cutoff='2026-10-05T06:17:55+00:00'
    query=dict(sessions=days,symbols=market['universe'],purpose='historical_exploration',price_basis='unadjusted',
        cutoff_by_session={d:cutoff for d in days})
    context=dict(usage='retrospective_review',snapshot_id=SNAPSHOT,anchor_session=days[-1],
        knowledge_cutoff=cutoff,pit_policy='operational_pit_v1',default_price_basis='common_anchor_adjusted_v1',
        native_price_basis='unadjusted',derivation=dict(price_query=query,factor_query=deepcopy(query),anchor_session=days[-1]))
    rows=[dict(security_id=r['security_id'],session=r['session'],native_open=float(r['open']),display_scale=scale)
        for r in market['rows']]
    def provenance(sid,day):
        return dict(security_id=sid,session=day,raw_batch_id='synthetic_factor',revision_id='synthetic:'+day,
            usable_from=cutoff,first_observed_at=cutoff,missing_reason=None)
    meta=[dict(security_id=r['security_id'],session=r['session'],missing_reason=None if scale is not None else 'missing_factor',
        factor_provenance=provenance(r['security_id'],r['session']),anchor_factor_provenance=provenance(r['security_id'],days[-1])) for r in rows]
    ohlcv=dict(contract_version='review_display_v1',context=context,records=rows,
        field_meta={'native_open':{'unit':unit},'open':{'unit':unit},
        'display_scale':{'unit':'dimensionless','dtype':'float64','by_key':meta}})
    if mutate:mutate(ohlcv)
    hashed,size=_file_digest(ohlcv)
    manifest=dict(contract_version='review_display_v1',exporter_version='review_display_v1',context=ohlcv['context'],
        files={'ohlcv.json':dict(uri='ohlcv.json',sha256=hashed,bytes=size)})
    payload=(json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    directory.mkdir(parents=True,exist_ok=True);(directory/'manifest.json').write_bytes(payload)
    content=dict(manifest=manifest,ohlcv=ohlcv)
    def owner_loader(path, *, manifest_sha256):
        assert Path(path)==directory and manifest_sha256==sha256(payload).hexdigest()
        return deepcopy(content)
    with patch.dict(sys.modules,{'axiom_data':SimpleNamespace(load_review_display=owner_loader)}):
        return read_review_display(directory,manifest_sha256=sha256(payload).hexdigest())


def rehash(wire):
    wire.pop('content_digest');wire['content_digest']=Document.from_dict(wire).identity
    return FillDisplayReport.from_dict(wire)


class FillDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.account=run_backtest(request())

    def test_exact_fill_mapping_and_source_immutability(self):
        original=self.account.payload
        with tempfile.TemporaryDirectory() as temp:
            display=files(Path(temp)/'data',self.account)
            with patch('axiom_engine.runtime.backtest.run_backtest',side_effect=AssertionError('replay')), \
                 patch('axiom_engine.runtime.accounting.AccountLedger.apply_fill',side_effect=AssertionError('fill')):
                report=build_fill_display(self.account,display=display)
            wire=report.to_dict();self.assertEqual(wire['status'],'COMPLETE')
            self.assertEqual(wire['fills'],self.account.to_dict()['fills'])
            self.assertTrue(all(c['display_price']=='2.500' and c['multiplier']=='0.25' for c in wire['coordinates']))
            self.assertEqual(self.account.payload,original)
            path=Path(temp)/'result.json';save_fill_display(report,path)
            with patch('axiom_engine.runtime.fill_display._coordinate',side_effect=AssertionError('compute')), \
                 patch('axiom_engine.runtime.fill_display.read_review_display',side_effect=AssertionError('Data')):
                self.assertEqual(load_fill_display(path).payload,report.payload)
            save_fill_display(report,path)
            with self.assertRaises(ContractError):save_fill_display(rehash(dict(wire,limitations=['changed'])),path)

    def test_public_loader_seal_and_changed_payload_rejected(self):
        with self.assertRaises(ContractError):SavedReviewDisplay({})
        with self.assertRaises(ContractError):build_fill_display(self.account,display={})
        with tempfile.TemporaryDirectory() as temp:
            display=files(Path(temp),self.account);display._content['ohlcv']['records'][0]['display_scale']=0.5
            with self.assertRaises(ContractError):build_fill_display(self.account,display=display)

    def test_null_reason_and_unit_boundary(self):
        cases=[('MISSING_DISPLAY_SCALE',dict(scale=None)),('UNIT_MISMATCH',dict(unit='CNY/fund unit')),
            ('NATIVE_OPEN_MISMATCH',dict(mutate=lambda w:[r.update(native_open=11) for r in w['records']])),
            ('MISSING_DISPLAY_ROW',dict(mutate=lambda w:w.update(records=[]))),
            ('DISPLAY_SCALE_UNVERIFIED',dict(mutate=lambda w:w['field_meta']['display_scale'].update(by_key=[])))]
        with tempfile.TemporaryDirectory() as temp:
            for i,(reason,args) in enumerate(cases):
                with self.subTest(reason=reason):
                    report=build_fill_display(self.account,display=files(Path(temp)/str(i),self.account,**args)).to_dict()
                    self.assertEqual(report['status'],'PARTIAL')
                    self.assertTrue(all(c['reason']==reason and c['display_price'] is None for c in report['coordinates']))

    def test_fixed_snapshot_anchor_cutoff_and_unique_key(self):
        changes=[lambda w:w['context'].update(snapshot_id='other'),
            lambda w:w['context'].update(anchor_session='2024-01-03'),
            lambda w:w['context']['derivation']['price_query']['cutoff_by_session'].update({'2024-01-02':'2024-01-02T12:00:00Z'}),
            lambda w:w['records'].append(deepcopy(w['records'][0]))]
        with tempfile.TemporaryDirectory() as temp:
            for i,change in enumerate(changes):
                with self.subTest(i=i),self.assertRaises(ContractError):
                    build_fill_display(self.account,display=files(Path(temp)/str(i),self.account,mutate=change))

    def test_no_inferred_unit_basis_session(self):
        event=dict(event_id='e',security_id='s',effective_date='2024-01-03',new_price_basis_session='2024-01-08')
        fill=dict(security_id='s',session='2024-01-08')
        self.assertEqual(_basis_reason(fill,[event],[]),'NEW_PRICE_BASIS_UNVERIFIED')
        self.assertEqual(_basis_reason(fill,[event],[dict(event,new_price_basis_session=None)]),'NEW_PRICE_BASIS_UNVERIFIED')
        self.assertIsNone(_basis_reason(fill,[event],[event]))
        self.assertIsNone(_basis_reason(dict(fill,session='2024-01-03'),[event],[]))

    def test_equivalent_cutoff_timezones_preserve_wire(self):
        with tempfile.TemporaryDirectory() as temp:
            for i,cutoff in enumerate(('2026-10-05T06:17:55+00:00','2026-10-05T14:17:55+08:00','2026-10-05T06:17:55Z')):
                def change(w):
                    for name in ('price_query','factor_query'):
                        q=w['context']['derivation'][name]
                        q['cutoff_by_session']={d:cutoff for d in q['sessions']}
                with self.subTest(cutoff=cutoff):
                    display=files(Path(temp)/str(i),self.account,mutate=change)
                    self.assertEqual(build_fill_display(self.account,display=display).to_dict()['status'],'COMPLETE')
                    self.assertEqual(display._content['manifest']['context']['derivation']['price_query']['cutoff_by_session']['2024-01-02'],cutoff)

    def test_reader_decimal_context_and_rehashed_link_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            report=build_fill_display(self.account,display=files(Path(temp)/'data',self.account))
            path=Path(temp)/'result.json';save_fill_display(report,path)
            prior=getcontext().copy()
            try:
                getcontext().prec=1;getcontext().traps[Inexact]=True
                self.assertEqual(load_fill_display(path).payload,report.payload)
            finally:setcontext(prior)
            wire=report.to_dict();wire['coordinates'][0]['session']='2024-01-05'
            with self.assertRaises(ContractError):save_fill_display(rehash(wire),Path(temp)/'bad.json')

    def test_rehashed_manifest_units_and_provenance_eligibility(self):
        from axiom_engine.runtime.fill_display import _identity
        with tempfile.TemporaryDirectory() as temp:
            original=build_fill_display(self.account,display=files(Path(temp)/'data',self.account)).to_dict()
            for i,change in enumerate(('anchor','cutoff','unit','provenance')):
                wire=deepcopy(original);inputs=wire['consumed_input']
                if change in ('anchor','cutoff'):
                    manifest=json.loads(wire['manifest_text']);ctx=manifest['context']
                    if change=='anchor':ctx['anchor_session']='2024-01-03'
                    else:ctx['derivation']['price_query']['cutoff_by_session']['2024-01-02']='2024-01-02T12:00:00Z'
                    raw=json.dumps(manifest,indent=2)+'\n';wire['manifest_text']=raw
                    wire['display_ref']='sha256:'+sha256(raw.encode()).hexdigest()
                    wire['manifest_file_ref']=dict(uri='manifest.json',sha256=wire['display_ref'][7:],bytes=len(raw.encode()))
                elif change=='unit':
                    inputs['account_unit']=inputs['source_unit']=inputs['target_unit']='USD/share'
                    for c in wire['coordinates']:c['source_unit']=c['target_unit']='USD/share'
                else:
                    fact=inputs['facts'][0];fact['scale_meta']['factor_provenance']['session']='2020-01-02'
                    for c in wire['coordinates']:
                        if (c['security_id'],c['session'])==(fact['security_id'],fact['session']):c['input_fact_ref']=Document.from_dict(fact).identity
                wire['consumed_input_ref']=Document.from_dict(inputs).identity
                wire['display_result_ref']=_identity(wire)
                with self.subTest(change=change),self.assertRaises(ContractError):
                    save_fill_display(rehash(wire),Path(temp)/str(i))


if __name__=='__main__':unittest.main()
