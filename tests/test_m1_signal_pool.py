"""Small compatibility boundaries; no training, supplier calls or historical replay."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, stock_strategy_spec
from axiom_engine.core.contracts import Document, canonical
from axiom_engine.core.stock_signal import _ref
from axiom_engine.runtime import BacktestRequest, StockMarketSpec, save_stock_market_inputs, save_stock_prediction_inputs
from test_stock_signal_inputs import v3_request, derived_request
from test_stock_market_owner import market_for, bind, seal_request
from test_stock_stream_inputs import artifact_file, LIMITS
from test_stock_owned_inputs import request_for, run_owned
from test_stock_strategy import configured
from test_stock_prepared_inputs import reopen


def legacy_signal(root, plan, *, damage=False):
    """Construct a pre-v2 fixture with original provenance, never migrate a user artifact."""
    for index, item in enumerate(plan['prediction_input']['frames']):
        wire=Document(Path(item['signal_artifact']['manifest_uri']).read_text()).to_dict()
        wire.pop('signal_semantics_ref'); wire['contract_version']='derived_signal_run_v1'
        wire['implementation_ref']='sha256:'+'1'*64
        if damage and index==0: wire['rows'][0]['score']+=0.01
        wire['score_ref']=_ref(wire['rows']); wire.pop('signal_run_ref'); wire['signal_run_ref']=_ref(wire)
        item.pop('signal_semantics_ref')
        item.update({k:wire[k] for k in ('signal_run_ref','score_ref','implementation_ref')})
        item['signal_artifact']=artifact_file(root,'legacy-'+str(index),wire)
    return seal_request(plan)


class M1SignalPoolTests(unittest.TestCase):
    def test_legacy_original_parent_replay_and_resigned_damage(self):
        for damage in (False,True):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); raw=v3_request(root); plan=legacy_signal(root,derived_request(root,raw),damage=damage)
                identities=[item['signal_run_ref'] for item in plan['prediction_input']['frames']]
                with market_for(raw) as market:
                    if damage:
                        with self.assertRaises(ContractError): bind(market,plan)
                    else:
                        with bind(market,plan) as inputs:
                            run,view=run_owned(root/'account',plan,inputs)
                        self.assertTrue(view.rows['decisions'])
                        self.assertEqual(identities,[item['signal_run_ref'] for item in plan['prediction_input']['frames']])

    def test_new_semantic_mismatch_and_saved_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=v3_request(root); plan=derived_request(root,raw)
            with market_for(raw) as market,bind(market,plan) as inputs:
                expected,view=run_owned(root/'first',plan,inputs)
                mr=save_stock_market_inputs(market,root/'market',limits=LIMITS)
                pr=save_stock_prediction_inputs(inputs,root/'prediction',limits=LIMITS)
            self.assertEqual(pr['contract_version'],'stock_prepared_predictions_v2')
            actual,reloaded=reopen(mr,pr,plan,root/'reopened')
            self.assertEqual(actual['run_id'],expected['run_id']); self.assertEqual(reloaded.rows,view.rows)
            # A legacy manifest cannot gain cross-build trust merely from saved rows.
            legacy=json.loads(Path(pr['manifest_uri']).read_text())
            legacy['contract_version']='stock_prepared_predictions_v1'
            legacy['producer_implementation_ref']='sha256:'+'1'*64
            legacy['owner'].pop('signal_semantics')
            raw_manifest=canonical(legacy)
            Path(pr['manifest_uri']).write_text(raw_manifest)
            legacy_ref={**pr,'contract_version':legacy['contract_version'],
                'content_digest':Document(raw_manifest).identity,
                'artifact_id':Document.from_dict({key:legacy[key] for key in
                    ('contract_version','input_semantics','input_key')}).identity}
            with self.assertRaisesRegex(ContractError,'original-parent re-admission'):
                reopen(mr,legacy_ref,plan,root/'legacy-rejected')
            # Restore the v2 fixture before its independent semantic mismatch check.
            legacy['contract_version']=pr['contract_version']
            from axiom_engine._implementation import IMPLEMENTATION_REF
            legacy['producer_implementation_ref']=IMPLEMENTATION_REF
            legacy['owner']['signal_semantics']={item['signal_run_ref']:item['signal_semantics_ref']
                for item in plan['prediction_input']['frames']}
            Path(pr['manifest_uri']).write_text(canonical(legacy))
            with patch('axiom_engine.core.stock_signal.signal_semantics_ref',return_value='sha256:'+'9'*64):
                with market_for(raw) as market,self.assertRaises(ContractError): bind(market,plan)
                with self.assertRaises(ContractError): reopen(mr,pr,plan,root/'bad')

    def test_existing_pool_reaches_shared_save_reload_account_and_rejects_wrong_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); original=request_for(root)
            self.assertNotIn('universe_id',StockMarketSpec.from_request(BacktestRequest.from_dict(original)).to_dict())
            plan=configured(original,stock_strategy_spec(top_k=3),universe_id='csi500')
            for index,entry in enumerate(plan['market_input']['native_inputs']):
                wire=Document(Path(entry['artifact']['manifest_uri']).read_text()).to_dict()
                if wire['context']['domain']!='universe_membership': continue
                wire['context']['query']['universe_id']='csi500'
                entry['native_ref']=Document.from_dict(wire).identity
                entry['artifact']=artifact_file(root,'pool-'+str(index),wire)
            seal_request(plan)
            self.assertEqual(StockMarketSpec.from_request(BacktestRequest.from_dict(plan)).to_dict()['universe_id'],'csi500')
            with market_for(plan) as market,bind(market,plan) as inputs:
                expected,view=run_owned(root/'first',plan,inputs)
                mr=save_stock_market_inputs(market,root/'market',limits=LIMITS)
                pr=save_stock_prediction_inputs(inputs,root/'prediction',limits=LIMITS)
                wrong=configured(plan,stock_strategy_spec(top_k=3),universe_id='csi1000')
                with self.assertRaises(ContractError): bind(market,wrong)
            actual,reloaded=reopen(mr,pr,plan,root/'reopened')
            self.assertEqual(actual['run_id'],expected['run_id']); self.assertEqual(reloaded.rows,view.rows)
            with self.assertRaises(ContractError): reopen(mr,pr,wrong,root/'wrong')
