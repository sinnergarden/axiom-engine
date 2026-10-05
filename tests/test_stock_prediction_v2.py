"""Neutral staged clocks preserve v1 and never admit a new account."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, StockPredictionFrame, plan_stock_portfolio, validate_stock_predictions
from axiom_engine.runtime import BacktestRequest, load_backtest_run, run_backtest
from axiom_engine.runtime.stock_inputs import validate_stock_request
from test_stocks import predictions, request


def staged_predictions():
    wire = predictions()
    wire.update(contract_version='stock_prediction_run_v2', fold_spec_ref='sha256:'+'f'*64,
                clock_basis='declared_simulation', model_ref='sha256:'+'b'*64)
    model_available = wire['rows'][0]['session']+'T20:45:00+08:00'
    for row in wire['rows']:
        row.update(feature_knowledge_cutoff=row['knowledge_cutoff'], feature_available_at=row['available_at'],
                   simulated_model_available_at=model_available, knowledge_cutoff=row['session']+'T21:00:00+08:00',
                   available_at=row['session']+'T21:00:00+08:00')
        row['source_refs'].append(wire['model_ref'])
    return wire


def validate(wire):
    return validate_stock_predictions(StockPredictionFrame.from_dict(wire))


class NeutralPredictionV2Tests(unittest.TestCase):
    def test_public_validation_preserves_original_wire_and_aware_instants(self):
        wire=staged_predictions()
        for i,row in enumerate(wire['rows']):
            if i % 2:
                row['available_at']=row['session']+'T13:00:00Z'
                row['feature_knowledge_cutoff']=row['session']+'T12:30:00Z'
                row['simulated_model_available_at']=wire['rows'][0]['session']+'T12:45:00Z'
        frame=StockPredictionFrame.from_dict(wire);before=frame.payload
        with patch('axiom_engine.core.stock_portfolio.plan_stock_portfolio', side_effect=AssertionError('planning')):
            decoded,indexed=validate_stock_predictions(frame)
        self.assertEqual(decoded,wire);self.assertEqual(len(indexed),len(wire['rows']))
        self.assertEqual(frame.payload,before)

    def test_clocks_are_general_and_null_feature_availability_is_explicit(self):
        wire=staged_predictions();model_day=wire['rows'][0]['session']
        for row in wire['rows']:
            row.update(feature_knowledge_cutoff=row['session']+'T14:00:00+08:00',
                       feature_available_at=row['session']+'T13:50:00+08:00',
                       simulated_model_available_at=model_day+'T14:30:00+08:00',
                       knowledge_cutoff=row['session']+'T15:00:00+08:00',
                       available_at=row['session']+'T15:00:00+08:00')
        validate(wire)
        row=wire['rows'][0];row.update(feature_available_at=None,valid=False,score=None,invalid_reason='FEATURE_MISSING')
        self.assertIsNone(validate(wire)[0]['rows'][0]['feature_available_at'])
        row.update(valid=True,score=1,invalid_reason=None)
        with self.assertRaisesRegex(ContractError,'Feature availability'):validate(wire)

    def test_staged_clock_and_source_ref_conflicts_are_rejected(self):
        mutations={
            'late_publish':lambda w:w['rows'][0].update(available_at=w['rows'][0]['session']+'T21:01:00+08:00'),
            'feature_after_cutoff':lambda w:w['rows'][0].update(feature_available_at=w['rows'][0]['session']+'T20:31:00+08:00'),
            'feature_after_inference':lambda w:w['rows'][0].update(feature_knowledge_cutoff=w['rows'][0]['session']+'T21:01:00+08:00'),
            'model_at_inference':lambda w:w['rows'][0].update(simulated_model_available_at=w['rows'][0]['knowledge_cutoff']),
            'wrong_feature_session':lambda w:w['rows'][0].update(feature_knowledge_cutoff='2024-01-01T20:30:00+08:00'),
            'wrong_inference_session':lambda w:w['rows'][0].update(knowledge_cutoff='2024-01-01T21:00:00+08:00',available_at='2024-01-01T21:00:00+08:00'),
            'naive_clock':lambda w:w['rows'][0].update(feature_available_at=w['rows'][0]['session']+'T20:00:00'),
            'union_clock_conflict':lambda w:w['rows'][0].update(feature_knowledge_cutoff=w['rows'][0]['session']+'T20:45:00+08:00'),
            'model_clock_conflict':lambda w:w['rows'][-1].update(simulated_model_available_at=w['rows'][0]['session']+'T20:46:00+08:00'),
            'missing_model_source':lambda w:w['rows'][0]['source_refs'].remove(w['model_ref']),
            'missing_feature_source':lambda w:w['rows'][0]['source_refs'].remove(w['feature_ref']),
        }
        for name,mutate in mutations.items():
            wire=staged_predictions();mutate(wire)
            with self.subTest(case=name),self.assertRaises(ContractError):validate(wire)

    def test_exact_fields_refs_union_validity_and_score_semantics_remain_checked(self):
        mutations={
            'extra_top':lambda w:w.update(model_clock={}),
            'missing_fold':lambda w:w.pop('fold_spec_ref'),
            'bad_fold':lambda w:w.update(fold_spec_ref=True),
            'clock_basis':lambda w:w.update(clock_basis='actual_historical_receipt'),
            'extra_row':lambda w:w['rows'][0].update(fit_cutoff='2024-01-01T20:30:00+08:00'),
            'missing_row':lambda w:w['rows'][0].pop('feature_available_at'),
            'duplicate_key':lambda w:w['rows'].append(deepcopy(w['rows'][0])),
            'missing_union_key':lambda w:w['rows'].pop(),
            'score':lambda w:w['rows'][0].update(score='1'),
            'invalid_score':lambda w:w['rows'][0].update(valid=False,invalid_reason='MISSING'),
            'semantics':lambda w:w.update(score_unit='percent'),
        }
        for name,mutate in mutations.items():
            wire=staged_predictions();mutate(wire)
            with self.subTest(case=name),self.assertRaises(ContractError):validate(wire)

    def test_v2_is_explicitly_refused_before_account_or_ledger(self):
        wire=staged_predictions();frame=StockPredictionFrame.from_dict(wire)
        account,context={},{}
        with self.assertRaisesRegex(ContractError,'account clock consumption is not admitted'):
            plan_stock_portfolio(frame,account=account,context=context,top_k=3)
        self.assertEqual(account,{});self.assertEqual(context,{})
        plan=request().to_dict();plan['signal_frame']=wire
        with patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('ledger started')):
            with self.assertRaisesRegex(ContractError,'account clock consumption is not admitted'):
                validate_stock_request(plan)
            with self.assertRaisesRegex(ContractError,'account clock consumption is not admitted'):
                run_backtest(BacktestRequest.from_dict(plan))

    def test_original_v1_rules_and_saved_account_payloads_stay_unchanged(self):
        wire=predictions();self.assertEqual(validate(wire)[0],wire)
        for case in ('late','different_cutoff','v2_field'):
            changed=deepcopy(wire);row=changed['rows'][0]
            if case=='late':row['available_at']=row['session']+'T20:31:00+08:00'
            elif case=='different_cutoff':row['knowledge_cutoff']=row['session']+'T21:00:00+08:00'
            else:row['feature_available_at']=row['available_at']
            with self.subTest(case=case),self.assertRaises(ContractError):validate(changed)
        with patch('axiom_engine.runtime.backtest.run_backtest',side_effect=AssertionError('account replay')), \
             patch('axiom_engine.runtime.backtest.AccountLedger',side_effect=AssertionError('ledger started')):
            for path in sorted((Path(__file__).parent/'fixtures').glob('top5_run_v1*.json')):
                original=path.read_text();run=load_backtest_run(path)
                self.assertEqual(run.payload,StockPredictionFrame(original).payload)
                self.assertEqual(path.read_text(),original)


if __name__=='__main__':unittest.main()
