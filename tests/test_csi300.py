"""Synthetic source-bound four-board goldens; no suppliers, Data or fitted models."""
from copy import deepcopy
import unittest

from axiom_engine.core import ContractError, StockPredictionFrame, plan_stock_portfolio
from axiom_engine.core.contracts import Document
from axiom_engine.core.stock_rules import (rule_at, legal_quantity, floor_quantity, support_ref,
                                         validate_execution_rules)
from axiom_engine.runtime import (stock_execution_rules, stock_fee_schedule,
    csi300_stock_portfolio_policy, stock_daily_open_profile_v2)
from axiom_engine.runtime.stock_rules import fee_at

REF = "sha256:" + "a" * 64
DAYS = ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-08"]
BOARD_IDS = {"SSE_MAIN": "cnstock.600100.SH.20000101", "SZSE_MAIN": "cnstock.001289.SZ.20000101",
             "SZSE_CHINEXT": "cnstock.300100.SZ.20000101", "SSE_STAR": "cnstock.688100.SH.20190722"}
SOURCES = [{"source_key": "synthetic", "url": "https://example.test/synthetic-rule",
            "content_sha256": REF, "clause": "SYNTHETIC fixture, not an official rule/source checksum"}]


def rules_for(boards=None, *, days=None, listing_dates=None):
    boards = boards or {security: board for board, security in BOARD_IDS.items()}
    days = days or DAYS
    identities = []
    for security, board in sorted(boards.items()):
        exchange = "SSE" if ".SH." in security else "SZSE"
        listing = (listing_dates or {}).get(security, security[-8:-4] + "-" + security[-4:-2] + "-" + security[-2:])
        identities.append(dict(security_id=security, exchange=exchange, board=board, instrument_kind="A_SHARE",
            listing_date=listing, delisting_date=None, classification_source_keys=["synthetic"], source_refs=[]))
    native = {"context": {"contract_version": "data_batch_v1", "domain": "security_master",
        "snapshot_id": "s_" + "b"*64, "reader_version": "synthetic", "query": {"symbols": sorted(boards)}},
        "records": [{name: row[name] for name in ("security_id", "exchange", "listing_date", "delisting_date")}
                    for row in identities], "field_meta": {}}
    ref = Document.from_dict(native).identity
    for row in identities:
        row["source_refs"] = [ref]
    identity = dict(contract_version="stock_execution_identity_v1", rows=identities,
        source_refs=[ref], source_evidence=[dict(reference=ref, batch=native)], limitations=["synthetic identity"])
    quantities = []
    for board in sorted(set(boards.values())):
        minimum, step = (200, 1) if board == "SSE_STAR" else (100, 100)
        limit, market = ((100000, 50000) if board == "SSE_STAR" else
                         (300000, 150000) if board == "SZSE_CHINEXT" else (1000000, 1000000))
        quantities.append(dict(board=board, effective_from="2023-04-10", effective_to=None,
            buy_minimum=minimum, buy_increment=step, sell_minimum=minimum, sell_increment=step,
            full_residual_exit_allowed=True, limit_order_maximum=limit, market_order_maximum=market,
            daily_proxy_maximum=min(limit, market), price_tick="0.01", settlement_sessions=1,
            source_keys=["synthetic"]))
    return stock_execution_rules(universe=sorted(boards), calendar=days, identity_input=identity,
        quantity_rules=quantities, sources=SOURCES, verified_from=days[0], verified_through=days[-1])


def fees_for(*, through=None):
    intervals = [dict(effective_from="2019-01-01", effective_to="2022-04-29", sell_stamp_tax_rate="0.001",
                      transfer_fee_rate="0.00002", source_keys=["synthetic"]),
                 dict(effective_from="2022-04-29", effective_to="2023-08-28", sell_stamp_tax_rate="0.001",
                      transfer_fee_rate="0.00001", source_keys=["synthetic"]),
                 dict(effective_from="2023-08-28", effective_to=None, sell_stamp_tax_rate="0.0005",
                      transfer_fee_rate="0.00001", source_keys=["synthetic"])]
    return stock_fee_schedule(intervals=intervals, sources=SOURCES, verified_from="2019-01-01",
                              verified_through=through or DAYS[-1])


def frame_for(rules, *, scores=None, invalid=()):
    universe = rules["universe"]
    rows = []
    for day in rules["calendar"]:
        for index, security in enumerate(universe):
            bad = security in invalid
            rows.append(dict(security_id=security, session=day, knowledge_cutoff=day+"T20:30:00+08:00",
                available_at=day+"T12:00:00Z", score=None if bad else (scores or {}).get(security, len(universe)-index),
                valid=not bad, invalid_reason="FEATURE_MISSING" if bad else None, member=True, source_refs=[REF]))
    wire = dict(contract_version="stock_prediction_run_v1", signal_stage="prediction_raw",
        score_semantics="forward_5_session_cs_zscore_prediction", score_unit="dimensionless",
        feature_ref=REF, model_ref=REF, limitations=["synthetic predictions"], universe=universe, rows=rows)
    wire["signal_run_ref"] = Document.from_dict(wire).identity
    return wire


def core_context(rules, k):
    return dict(trade_session=rules["calendar"][1], feature_session=rules["calendar"][0],
        decision_time=rules["calendar"][1]+"T08:55:00+08:00", knowledge_cutoff=rules["calendar"][0]+"T20:30:00+08:00",
        reference_prices={s:dict(price="10", session=rules["calendar"][0],
            available_at=rules["calendar"][0]+"T12:00:00Z", source_refs=[REF]) for s in rules["universe"]},
        commission_rate="0.0003", minimum_commission_minor=500, slippage_bps="0", account_state_version=0,
        supported_security_ids=rules["universe"], supported_universe_ref=support_ref(rules["universe"]),
        portfolio_policy=csi300_stock_portfolio_policy(top_k=k, execution_universe=rules["universe"], execution_rules=rules),
        stock_execution_rules=rules, stock_execution_rules_ref=Document.from_dict(rules).identity)


class FrozenRuleTests(unittest.TestCase):
    def test_four_board_minimum_increment_maximum_and_full_sellable_tail(self):
        rules = rules_for()
        for board, security in BOARD_IDS.items():
            rule = rule_at(rules, security, DAYS[1])
            self.assertEqual(legal_quantity(201, "BUY", rule), 201 if board == "SSE_STAR" else 200)
            self.assertEqual(legal_quantity(199, "BUY", rule), 0 if board == "SSE_STAR" else 100)
            self.assertEqual(legal_quantity(2000000, "BUY", rule), rule["daily_proxy_maximum"])
            self.assertEqual(legal_quantity(99, "SELL", rule, held=99, sellable=99), 99)
        star = rule_at(rules, BOARD_IDS["SSE_STAR"], DAYS[1])
        self.assertEqual(legal_quantity(199, "SELL", star, held=199, sellable=199), 199)
        self.assertEqual(legal_quantity(500, "SELL", star, held=500, sellable=100), 0)
        self.assertEqual(floor_quantity(401, 200, 1), 401)
        for value in (True, -1, 3.0):
            with self.assertRaises(ContractError): floor_quantity(value, 200, 1)

    def test_date_effective_fees_and_verified_open_interval_boundary(self):
        fees = fees_for()
        self.assertEqual(fee_at(fees, "2022-04-28")["transfer_fee_rate"], "0.00002")
        self.assertEqual(fee_at(fees, "2022-04-29")["transfer_fee_rate"], "0.00001")
        self.assertEqual(fee_at(fees, "2023-08-27")["sell_stamp_tax_rate"], "0.001")
        self.assertEqual(fee_at(fees, "2023-08-28")["sell_stamp_tax_rate"], "0.0005")
        with self.assertRaises(ContractError): fee_at(fees, "2024-01-09")
        rules = rules_for()
        with self.assertRaises(ContractError): rule_at(rules, BOARD_IDS["SSE_MAIN"], "2024-01-09")
        for key in ("buy_minimum", "buy_increment", "daily_proxy_maximum"):
            bad = deepcopy(rules); bad["quantity_rules"][0][key] = True
            with self.assertRaises(ContractError): validate_execution_rules(bad)

    def test_explicit_exception_identity_and_reject_source_board_damage(self):
        security = "cnstock.302132.SZ.20100827"
        rules = rules_for({security:"SZSE_CHINEXT"})
        self.assertEqual(rule_at(rules, security, DAYS[1])["daily_proxy_maximum"], 150000)
        for mutation in ("board", "source", "listing", "gap", "overlap"):
            bad = deepcopy(rules)
            if mutation == "board": bad["identity_input"]["rows"][0]["board"] = "UNKNOWN"
            if mutation == "source": bad["identity_input"]["source_evidence"][0]["batch"]["records"][0]["exchange"] = "SSE"
            if mutation == "listing": bad["identity_input"]["rows"][0]["listing_date"] = "2010-08-28"
            if mutation == "gap": bad["quantity_rules"][0]["effective_from"] = "2024-01-03"
            if mutation == "overlap": bad["quantity_rules"].append(deepcopy(bad["quantity_rules"][0]))
            bad["identity_input_ref"] = Document.from_dict(bad["identity_input"]).identity
            with self.subTest(mutation=mutation), self.assertRaises(ContractError): validate_execution_rules(bad)
        for k in (True, 0, 2, "1"):
            with self.assertRaises(ContractError): csi300_stock_portfolio_policy(top_k=k, execution_universe=[security], execution_rules=rules)
        self.assertEqual(stock_daily_open_profile_v2(execution_rules=rules, fee_schedule=fees_for())["partial_fill_quantity_unit"], "one_share")


class FullMemberCoreTests(unittest.TestCase):
    def test_300_members_one_invalid_is_excluded_and_shortage_keeps_k(self):
        securities = [f"cnstock.600{i:03}.SH.20000101" for i in range(300)]
        rules = rules_for({s:"SSE_MAIN" for s in securities})
        frame = frame_for(rules, invalid=[securities[0]]); original = deepcopy(frame)
        account = dict(cash_minor=50000000, positions={}, version=0)
        result = plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account=account, context=core_context(rules, 5), top_k=5).to_dict()
        self.assertEqual(result["contract_version"], "axiom.stock_portfolio/3")
        self.assertEqual(result["selected_security_ids"], securities[1:6])
        self.assertEqual((result["trace"][0]["pit_member_count"],result["trace"][0]["valid_candidate_count"],
                          result["trace"][0]["excluded_invalid_member_count"]), (300,299,1))
        self.assertEqual(frame, original)
        result = plan_stock_portfolio(StockPredictionFrame.from_dict(frame), account=account, context=core_context(rules,300), top_k=300).to_dict()
        self.assertEqual(result["status"], "NO_DECISION")
        self.assertEqual(result["intents"], [])
        self.assertEqual(result["trace"][-1], {"reason":"INSUFFICIENT_ELIGIBLE_MEMBERS","count":299,"top_k":300})

    def test_star_small_increase_reduction_and_t_plus_one_do_not_become_orders(self):
        rules = rules_for(); star = BOARD_IDS["SSE_STAR"]; main = BOARD_IDS["SSE_MAIN"]
        frame = frame_for(rules, scores={star:100, main:99})
        for held, sellable, cash, k, target in ((200,200,1000,1,201),(401,401,199000,2,300)):
            account = dict(cash_minor=cash, positions={star:dict(quantity=held,sellable_quantity=sellable)},version=0)
            result = plan_stock_portfolio(StockPredictionFrame.from_dict(frame),account=account,context=core_context(rules,k),top_k=k).to_dict()
            self.assertEqual(result["targets"][star],target)
            self.assertFalse(any(i["security_id"]==star for i in result["intents"]))
            self.assertTrue(any(t["reason"]=="BELOW_MINIMUM_ORDER_QUANTITY" for t in result["trace"]))
        frame = frame_for(rules,scores={main:100,star:-100})
        for held,sellable,wanted in ((199,199,199),(500,100,0)):
            account=dict(cash_minor=0,positions={star:dict(quantity=held,sellable_quantity=sellable)},version=0)
            result=plan_stock_portfolio(StockPredictionFrame.from_dict(frame),account=account,context=core_context(rules,1),top_k=1).to_dict()
            self.assertEqual(sum(i["quantity"] for i in result["intents"] if i["security_id"]==star),wanted)

    def test_missing_sizing_reference_is_no_decision_but_contract_damage_fails(self):
        rules=rules_for(); frame=frame_for(rules); account=dict(cash_minor=50000000,positions={},version=0)
        context=core_context(rules,4); context["reference_prices"].pop(BOARD_IDS["SSE_STAR"])
        result=plan_stock_portfolio(StockPredictionFrame.from_dict(frame),account=account,context=context,top_k=4).to_dict()
        self.assertEqual((result["status"],result["trace"][-1]["reason"]),("NO_DECISION","MISSING_SIZING_REFERENCE"))
        bad=deepcopy(frame); bad["rows"].pop()
        with self.assertRaises(ContractError): plan_stock_portfolio(StockPredictionFrame.from_dict(bad),account=account,context=core_context(rules,4),top_k=4)


if __name__ == "__main__": unittest.main()
