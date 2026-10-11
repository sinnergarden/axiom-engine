"""Narrow reviewed-source declaration checks; only tiny synthetic accounts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core.contracts import ContractError, Document
from axiom_engine.runtime.stock_equity import FLOOR_POLICY, gaps, validate_facts
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_stock_equity import DAYS, REF, action, fixture
from test_stock_stream import execute
from test_stock_stream_inputs import artifact_file


class ReviewedEquitySourcesTests(unittest.TestCase):
    def test_raw_admission_survives_derived_global_identity_change(self):
        from test_stock_signal_inputs import v3_request, derived_request
        from test_stock_market_owner import market_for, bind
        other_implementation = 'sha256:' + 'e' * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = v3_request(root)
            derived = derived_request(root, raw)
            with patch('axiom_engine.runtime.backtest.AccountLedger', side_effect=AssertionError('account created')):
                with market_for(raw) as market:
                    with patch('axiom_engine.runtime.stock_signal_inputs.IMPLEMENTATION_REF', other_implementation):
                        with bind(market, raw) as inputs:
                            self.assertGreater(inputs.statistics['owned_bytes'], 0)
                with market_for(derived) as market:
                    with patch('axiom_engine.runtime.stock_signal_inputs.IMPLEMENTATION_REF', other_implementation):
                        with self.assertRaisesRegex(ContractError, 'Derived implementation/union/unit mismatch'):
                            with bind(market, derived):
                                self.fail('old Derived admitted')

    def test_mixed_declaration_saved_replay_preserves_receipt_and_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, a = fixture(root, cash='0', stock='0', allocation_policy=FLOOR_POLICY)
            facts = Document(Path(manifest['action_facts_artifact']['manifest_uri']).read_text()).to_dict()
            facts['availability_basis'] = 'mixed_reviewed_sources_v1'
            interpretation_ref = 'sha256:' + 'd' * 64
            facts['actions'][0]['source_refs'].append(interpretation_ref)
            manifest['action_facts_artifact'] = artifact_file(root, 'mixed-equity-actions', facts)
            manifest['request_ref'] = logical_ref(manifest, 'request_ref')
            wire, view, _ = execute(root/'out', manifest)
            saved = view.account_events['equity_facts']
            self.assertEqual(wire['status'], 'COMPLETE')
            self.assertEqual(saved, facts)
            self.assertEqual(saved['actions'][0]['first_observed_at'], a['first_observed_at'])
            self.assertEqual(saved['actions'][0]['available_at'], a['available_at'])
            ex = next(r for r in view.rows['position_ledger'] if r['reason'] == 'EQUITY_EX')
            self.assertEqual(ex['pending_share_delta'], 0)
            self.assertFalse(any(r['reason'].startswith('DIVIDEND_') for r in view.rows['cash_ledger']))

    def test_unknown_declaration_rejected_before_account_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = fixture(root)
            facts = Document(Path(manifest['action_facts_artifact']['manifest_uri']).read_text()).to_dict()
            facts['availability_basis'] = 'unreviewed_publication'
            manifest['action_facts_artifact'] = artifact_file(root, 'invalid-equity-actions', facts)
            manifest['request_ref'] = logical_ref(manifest, 'request_ref')
            with patch('axiom_engine.runtime.backtest.AccountLedger', side_effect=AssertionError('account created')):
                with self.assertRaisesRegex(ContractError, 'supplemental event clock basis'):
                    execute(root/'out', manifest)

    def test_mixed_sources_do_not_default_unknown_shares_or_allow_late_terms(self):
        ledger = SimpleNamespace(equity_entitlements={REF:dict(record_quantity=100,record_pending_quantity=0)})
        ordinary = action('s', cash_dividend_before_tax_per_share='0', stock_distribution_shares_per_share=None)
        self.assertEqual(gaps(ledger, [ordinary], DAYS[2], '09:30:00+08:00',
            share_allocation_policy=FLOOR_POLICY)[0]['reason'], 'UNKNOWN_SHARE_ENTITLEMENT')
        late = deepcopy(ordinary)
        late.update(stock_distribution_shares_per_share='0', available_at=DAYS[2]+'T00:00:00+08:00')
        self.assertEqual(gaps(ledger, [late], DAYS[2], '09:30:00+08:00',
            share_allocation_policy=FLOOR_POLICY)[0]['reason'], 'LATE_ENTITLEMENT_FACT')

    def test_mixed_sources_do_not_backdate_receipt_after_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = fixture(root)
            facts = Document(Path(manifest['action_facts_artifact']['manifest_uri']).read_text()).to_dict()
            facts['availability_basis'] = 'mixed_reviewed_sources_v1'
            facts['actions'][0]['first_observed_at'] = '2026-10-12T00:00:00Z'
            with self.assertRaisesRegex(ContractError, 'receipt after supplemental observation'):
                validate_facts(facts, universe=facts['universe'], calendar=facts['calendar'],
                               parent_refs=facts['parent_native_refs'])
