"""Tiny 2019 accounts with a future board; all prices and parents are synthetic."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, stock_strategy_spec
from axiom_engine.core import stock_portfolio as portfolio
from test_csi300 import rules_for, fees_for
from test_csi300_runtime import full_request
from test_stock_equity import fixture as equity_fixture
from test_stock_strategy import configured, decide, inputs
from test_stock_stream import execute
from test_stock_stream_inputs import source_fixture
from test_stock_market_owner import market_for, bind
from test_stock_owned_inputs import run_owned


DAYS = ["2018-12-28", "2019-01-02", "2019-01-03", "2019-01-07"]
ACTIVE = [f"cnstock.600{i:03}.SH.20000101" for i in range(7)]
FUTURE = "cnstock.688005.SH.20190722"
SPECS = [stock_strategy_spec(selection="rank_buffer", top_k=5, rank_buffer={"keep_rank": 10}),
         stock_strategy_spec(top_k=5)]


def plan_2019(*, empty=False):
    rules = rules_for({**{s: "SSE_MAIN" for s in ACTIVE}, FUTURE: "SSE_STAR"}, days=DAYS,
        quantity_from_by_board={"SSE_MAIN": "2018-08-20", "SSE_STAR": "2019-03-01"})
    def scores(frame):
        for row in frame["rows"]:
            if row["security_id"] == FUTURE: continue
            if empty: row.update(valid=False, score=None, invalid_reason="FEATURE_MISSING")
            elif row["session"] == DAYS[2] and row["security_id"] == ACTIVE[0]: row["score"] = -20
    return full_request(rules=rules, k=5, mutate_frame=scores, warmup_session="2018-12-27",
        fee_schedule=fees_for(from_session=DAYS[0], through=DAYS[-1])).to_dict()


class StockRuleDemandTests(unittest.TestCase):
    def test_both_top5_accounts_keep_full_union_and_only_size_used_names(self):
        plan = plan_2019()
        for spec in SPECS:
            with self.subTest(selection=spec["selection"]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); manifest, _ = source_fixture(root, plan=plan)
                manifest = configured(manifest, spec)
                calls = []
                original = portfolio.quantity_rule_at
                def quantity(*args, **kwargs):
                    calls.append((args[1], args[2])); return original(*args, **kwargs)
                with patch.object(portfolio, "quantity_rule_at", side_effect=quantity):
                    wire, view, _ = execute(root / "out", manifest, block_sessions=1)
                    with market_for(manifest) as market, bind(market, manifest) as owned:
                        owned_wire, owned_view = run_owned(root / "owned", manifest, owned, block_sessions=2)
                self.assertEqual((wire["status"], owned_wire["status"]), ("COMPLETE", "COMPLETE"))
                self.assertEqual(manifest["scope"]["execution_universe"], sorted(ACTIVE + [FUTURE]))
                self.assertTrue(calls); self.assertNotIn(FUTURE, {s for s, _ in calls})
                for name in ("decisions", "nav", "positions", "orders", "fills"):
                    self.assertEqual(view.rows[name], owned_view.rows[name])
                self.assertEqual(len(view.rows["decisions"]), 2)
                first, second = view.rows["decisions"]
                self.assertEqual(first["selected_security_ids"], ACTIVE[:5])
                expected = ACTIVE[:5] if spec["selection"] == "rank_buffer" else ACTIVE[1:6]
                self.assertEqual(set(second["selected_security_ids"]), set(expected))
                sells = [f for f in view.rows["fills"] if f["side"] == "SELL" and f["security_id"] == ACTIVE[0]]
                self.assertTrue(sells)  # Kept names may have a small sizing reduction.
                remaining = wire["final_account"]["positions"].get(ACTIVE[0], {}).get("quantity", 0)
                self.assertEqual(remaining > 0, spec["selection"] == "rank_buffer")
                self.assertFalse(any(FUTURE in d["targets"] for d in view.rows["decisions"]))
                self.assertFalse(any(f["security_id"] == FUTURE for f in view.rows["fills"]))
                self.assertEqual(plan["profile"]["stock_execution_rules"]["quantity_rules"][1]["effective_from"], "2019-03-01")

    def test_empty_candidates_need_no_quantity_rules_and_keep_cash(self):
        for spec in SPECS:
            with self.subTest(selection=spec["selection"]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); manifest, _ = source_fixture(root, plan=plan_2019(empty=True))
                with patch.object(portfolio, "quantity_rule_at", side_effect=AssertionError("unused quantity rule")):
                    wire, view, _ = execute(root / "out", configured(manifest, spec), block_sessions=1)
                self.assertEqual(wire["status"], "COMPLETE")
                self.assertEqual(view.rows["orders"], [])
                self.assertTrue(all(row["cash_minor"] == 50_000_000 for row in view.rows["nav"]))

    def test_future_board_does_not_weaken_held_equity_gap_stop(self):
        def unknown(a):
            a.update(report_period="2018-06-30", record_date=DAYS[1], ex_date=DAYS[2],
                payment_date="2019-01-05", stock_listing_date=None, available_at="2018-12-27T09:30:00+08:00",
                aliases=[dict(security_id=a["security_id"], report_period="2018-06-30",
                              announcement_date="2018-12-20", process_status="实施")])
        for spec in SPECS:
            with self.subTest(selection=spec["selection"]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); manifest, _, _ = equity_fixture(root, plan=plan_2019(), stock=None, change=unknown)
                wire, view, _ = execute(root / "out", configured(manifest, spec), block_sessions=1)
                self.assertEqual(wire["status"], "BLOCKED")
                self.assertEqual(wire["stopped"]["reason"], "HELD_EQUITY_FACT_GAP")
                self.assertEqual(len(view.rows["nav"]), 1)
                self.assertTrue(view.rows["positions"])

    def test_optional_legacy_quantity_map_preserves_validation_and_weight_result(self):
        data = inputs(); spec = stock_strategy_spec(top_k=1)
        before = decide(spec, data)
        del data["constraints"]["quantity_rules"]
        self.assertEqual(decide(spec, data), before)
        for bad in ({}, {s: None for s in data["universe"]}):
            damaged = deepcopy(data); damaged["constraints"]["quantity_rules"] = bad
            with self.assertRaises(ContractError): decide(spec, damaged)
        # The original entry contract still checks board-rule gaps for listed names.
        with self.assertRaisesRegex(ContractError, "stock rule interval"):
            rules_for({ACTIVE[0]: "SSE_MAIN", FUTURE: "SSE_STAR"}, days=DAYS,
                quantity_from_by_board={"SSE_MAIN": "2019-03-01", "SSE_STAR": "2019-03-01"})


if __name__ == "__main__": unittest.main()
