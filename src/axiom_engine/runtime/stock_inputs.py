"""Stock-specific input admission; common account execution remains in backtest."""
import re

from ..core.contracts import Document, digest, fields, integer, require, session, text
from ..core.stock_portfolio import BUDGET_BASIS, StockPredictionFrame, instant, validate_stock_predictions, validate_top_k
from ..core.portfolio import decimal
from .profiles import stock_daily_open_profile, stock_daily_open_profile_v2
from ..core.stock_rules import validate_execution_rules, support_ref as full_support_ref, listed, LIFECYCLE_POLICY
from .stock_rules import csi300_stock_portfolio_policy
from .stock_evidence import native_batches, native_ref

ELIGIBILITY_ID = "sz_main_a_000_002_003_v1"
ACTION_POLICY = "observed_implemented_only"
TAX_CONVENTION = "gross_before_tax_no_personal_tax_model"


def stock_portfolio_policy(*, top_k: int, execution_universe: list[str]) -> dict:
    """Pure explicit TopK configuration; never read members, prices or predictions."""
    require(type(execution_universe) is list and bool(execution_universe) and
            execution_universe == supported_universe(execution_universe) and
            len(set(execution_universe)) == len(execution_universe), "frozen supported execution universe required")
    validate_top_k(top_k, execution_universe)
    return {"eligibility_id": ELIGIBILITY_ID, "top_k": top_k,
            "rebalance": "weekly_first_trading_session", "budget_basis": BUDGET_BASIS}


def supported_universe(universe):
    return sorted(s for s in universe if re.fullmatch(r"cnstock\.(?:000|002|003)\d{3}\.SZ\.\d{8}", s))


def support_ref(universe):
    return Document.from_dict({"eligibility_id": ELIGIBILITY_ID, "security_ids": list(universe)}).identity


def validate_cash_action(action, universe, refs):
    fields(action, "contract_version event_id security_id record_session ex_session pay_session cash_before_tax_per_share tax_convention available_at source_refs")
    require(action["contract_version"] == "stock_cash_action_v1" and action["tax_convention"] == TAX_CONVENTION,
            "unsupported stock cash/tax convention")
    digest(action["event_id"])
    require(action["security_id"] in universe, "cash action outside execution scope")
    session(action["record_session"]); session(action["ex_session"])
    require(action["record_session"] < action["ex_session"], "invalid stock cash dates")
    if action["pay_session"] is not None:
        session(action["pay_session"])
        require(action["ex_session"] <= action["pay_session"], "payment precedes EX")
    require(decimal(action["cash_before_tax_per_share"], minimum=0) > 0, "positive stock cash income required")
    require(instant(action["available_at"]) <= instant(action["record_session"] + "T20:30:00+08:00"),
            "stock cash event absent at entitlement cutoff")
    require(type(action["source_refs"]) is list and bool(action["source_refs"]) and
            set(action["source_refs"]) <= set(refs), "unbound stock cash source")


def applicable_blocks(market, security, day):
    cutoff = instant(day + "T20:30:00+08:00")
    return [b for b in market["action_blocks"] if b["security_id"] == security and
            (b["effective_session"] is None or b["effective_session"] <= day) and instant(b["available_at"]) <= cutoff]


def _validate_pair_proof(evidence, signal, universe, calendar, batches, *, lifecycle=False, membership_batch=None):
    """Verify the saved owner's complete admission closure, without repeating Data queries."""
    v2 = signal["contract_version"] == "stock_prediction_schedule_v1"
    if v2:
        identity = evidence.get("listing_identity_checks", {})
        fields(identity, "checked_rows paired_rows mismatches")
        require(evidence.get("contract_version") == "stock_snapshot_pair_admission_v2" and
                type(identity["checked_rows"]) is int and type(identity["paired_rows"]) is int and
                identity.get("checked_rows") == len(universe) and identity.get("paired_rows") == len(universe) and
                identity.get("mismatches") == [], "incomplete dynamic stock listing identity proof")
    else:
        require(evidence.get("contract_version") == "stock_snapshot_pair_admission_v1" and
                evidence.get("listing_identity_equal_all_83") is True, "incomplete stock identity/member proof")
    require(evidence["scope"].get("membership_security_ids") == signal["universe"], "incomplete stock identity/member proof")
    row_count = sum(len(f["prediction_frame"]["rows"]) for f in signal["folds"]) if v2 else len(signal["rows"])
    member = evidence.get("prediction_membership", {})
    require(member.get("compared") == row_count and member.get("saved_prediction_rows") == row_count and
            member.get("model_mismatch") == [] and member.get("execution_mismatch") == [], "incomplete saved prediction/member pairing")
    count = len(universe) * len(calendar)
    basis = evidence.get("previous_close_basis_checks", {})
    require(all((type(basis.get(name)) is bool if lifecycle else basis.get(name) is True)
                for name in ("all_available_at_feature_knowledge_cutoff", "all_available_before_decision")) and
            basis.get("paired_rows_per_root") == count and basis.get("checked_rows_both_roots") == 2 * count,
            "incomplete previous-close basis pairing")
    if v2:
        require(all(type(basis.get(name)) is int for name in
                ("equal_rows", "listing_identity_checked_rows", "paired_rows_per_root", "checked_rows_both_roots")) and
                basis.get("equal_rows") == count and basis.get("listing_identity_checked_rows") == count and
                basis.get("listing_identity_mismatches") == [], "incomplete dynamic previous-close identity pairing")
    else:
        require(basis.get("equal_all_83_23") is True and basis.get("listing_suffix_and_SZSE_identity_all_83") is True,
                "incomplete previous-close basis pairing")
    artifact = evidence.get("previous_close_basis_artifact", {})
    for name in ("basis_ref", "canonical_content_digest", "file_digest"):
        digest(artifact.get(name))
    warmup = evidence["scope"].get("warmup_sessions")
    require(type(warmup) is list and bool(warmup) and warmup == sorted(set(warmup)) and warmup[-1] < calendar[0],
            "explicit preceding stock warmup pairing required")
    market_fields = {"open", "high", "low", "close", "volume_shares", "amount_cny"}
    for section, specs in (("primary_comparison", {"market": (count, market_fields), "factor": (count, {"factor"}),
            "membership": (len(signal["universe"]) * len(calendar), {"is_member"})}),
            ("warmup_comparison", {"market": (len(universe) * len(warmup), market_fields),
                                  "factor": (len(universe) * len(warmup), {"factor"})})):
        comparisons = evidence.get(section, {})
        for domain, (expected, names) in specs.items():
            comparison = comparisons.get(domain, {})
            require(comparison.get("left_rows") == expected and comparison.get("right_rows") == expected and
                    comparison.get("missing_keys_left") == [] and comparison.get("missing_keys_right") == [] and
                    set(comparison.get("fields", {})) == names, "incomplete stock pair scope proof")
            for check in comparison["fields"].values():
                counts = {"values_equal", "usable_from_equal", "missing_reason_equal", "availability_basis_equal",
                          "raw_batch_id_equal", "revision_id_equal", "revision_sequence_equal", "first_observed_at_equal",
                          "source_available_at_equal", "evidence_ref_equal"}
                require(check.get("units_equal") is True and check.get("model_unit") == check.get("execution_unit") and
                        check.get("difference_samples") == [] and set(check.get("counts", {})) == counts and
                        all(value == expected for value in check["counts"].values()), "stock native pairing difference")
    manifests = evidence.get("batches", {})
    for side in ("model", "execution"):
        for kind in ("market", "factor", "membership", "states", "warmup-market", "warmup-factor"):
            entry = manifests.get(side + "-" + kind, {})
            digest(entry.get("wire_ref")); digest(entry.get("file_digest"))
            query = entry.get("query", {})
            expected_symbols = signal["universe"] if kind == "membership" else universe
            expected_sessions = warmup if kind.startswith("warmup-") else calendar
            require(entry.get("snapshot_id") == evidence[side]["snapshot"] and bool(entry.get("reader_version")) and
                    query.get("purpose") == ("decision_facts" if kind == "membership" else "market_replay") and
                    query.get("symbols") == expected_symbols and query.get("pit_policy") == "best_effort_vendor_v1" and
                    (set(expected_sessions) <= set(query.get("sessions", [])) if kind == "states" else query.get("sessions") == expected_sessions),
                    "stock pair batch binding mismatch")
            if kind == "membership":
                require(query.get("fields") == ["is_member"] and query.get("universe_id") == "csi300", "stock member proof query mismatch")
                if v2:
                    cutoffs = query.get("cutoff_by_session")
                    require(type(cutoffs) is dict and set(cutoffs) == set(expected_sessions),
                            "stock member proof cutoff session coverage mismatch")
                    require(all(instant(cutoffs[day]) == instant(day + "T20:30:00+08:00") for day in expected_sessions),
                            "stock member proof must retain original Feature cutoff")
    bound = [("execution-states", batches[0]), ("execution-market", batches[1]), ("execution-factor", batches[3])]
    if lifecycle:
        bound.append(("execution-membership", membership_batch))
    for key, batch in bound:
        entry = manifests[key]
        require(entry["wire_ref"] == native_ref(batch) and entry["query"] == batch["context"]["query"] and
                entry["reader_version"] == batch["context"]["reader_version"], "stock execution facts differ from admitted batches")


def validate_stock_request(plan, *, legacy_saved_top5=False):
    full = plan.get("contract_version") == "backtest_request_v6"
    v4 = plan.get("contract_version") in ("backtest_request_v4", "backtest_request_v6")
    fields(plan, "contract_version account_id start_session end_session market_replay initial_account profile prediction_universe execution_universe supported_universe_ref portfolio_policy admission_ref admission_evidence stock_action_policy " +
           ("prediction_schedule" if v4 else "signal_frame") + (" stock_execution_rules_ref" if full else ""))
    require(plan["contract_version"] in ("backtest_request_v3", "backtest_request_v4", "backtest_request_v6") and plan["stock_action_policy"] == ACTION_POLICY,
            "unsupported stock request/action policy")
    text(plan["account_id"]); session(plan["start_session"]); session(plan["end_session"])
    if v4:
        from .stock_schedule import _admit_schedule
        signal, signals = _admit_schedule(plan["prediction_schedule"], plan["market_replay"]["calendar"])
    else:
        signal, signals = validate_stock_predictions(StockPredictionFrame.from_dict(plan["signal_frame"]))
        require(signal["contract_version"] == "stock_prediction_run_v1",
                "v2 neutral predictions only; account clock consumption is not admitted")
    universe = plan["execution_universe"]
    require(plan["prediction_universe"] == signal["universe"] and universe == (signal["universe"] if full else supported_universe(signal["universe"])),
            "stock prediction/execution scope mismatch")
    require(bool(universe) and plan["supported_universe_ref"] == (full_support_ref(universe) if full else support_ref(universe)), "stock support identity mismatch")
    policy = plan["portfolio_policy"]
    fields(policy, "eligibility_id top_k rebalance budget_basis" + (" candidate_policy stock_execution_rules_ref" if full else ""))
    expected = ({"eligibility_id": ELIGIBILITY_ID, "top_k": 5,
                 "rebalance": "weekly_first_trading_session", "budget_basis": BUDGET_BASIS}
                if legacy_saved_top5 else (csi300_stock_portfolio_policy(top_k=policy.get("top_k"), execution_universe=universe,
                    execution_rules=plan["profile"]["stock_execution_rules"]) if full else
                    stock_portfolio_policy(top_k=policy.get("top_k"), execution_universe=universe)))
    require(policy == expected and type(policy["top_k"]) is int, "unsupported stock portfolio policy")
    digest(plan["admission_ref"])
    evidence = dict(plan["admission_evidence"])
    recorded = evidence.pop("admission_ref", None)
    require(recorded == plan["admission_ref"] and Document.from_dict(evidence).identity == recorded,
            "stock input-pair evidence identity mismatch")
    if v4:
        refs = [{"fold_ref": f["fold_ref"], **{name: f["prediction_frame"][name] for name in
                ("fold_spec_ref", "signal_run_ref", "feature_ref", "model_ref")}} for f in signal["folds"]]
        require(evidence.get("prediction_schedule_ref") == signal["schedule_ref"] and evidence.get("prediction_refs") == refs and
                not any(name in evidence for name in ("signal_run_ref", "feature_ref", "model_ref")), "Stock schedule admission refs mismatch")
    else:
        require(evidence["signal_run_ref"] == signal["signal_run_ref"] and
                evidence["feature_ref"] == signal["feature_ref"] and evidence["model_ref"] == signal["model_ref"],
                "stock input-pair admission scope mismatch")
    require(evidence["scope"]["execution_security_ids"] == universe and
            evidence["status"] == "NUMERIC_POLICY_IDENTITY_PAIR_PASS", "stock input-pair admission scope mismatch")
    profile = plan["profile"]
    expected_profile = (stock_daily_open_profile_v2(execution_rules=profile["stock_execution_rules"],
        fee_schedule=profile["stock_fee_schedule"], unknown_status_policy=profile.get("unknown_status_policy")) if full else
        stock_daily_open_profile(unknown_status_policy=profile.get("unknown_status_policy")))
    require(profile == expected_profile,
            "stock profile parameters differ from frozen contract")
    market = plan["market_replay"]
    fields(market, "contract_version price_basis calendar universe rows cash_dividends action_diagnostics action_blocks source_refs source_evidence limitations" +
           (" coverage_bundle" if "coverage_bundle" in market else "") +
           (" stock_execution_rules_ref membership_ref lifecycle_policy" if full else ""))
    require(market["contract_version"] == ("market_replay_v4" if full else "market_replay_v3") and market["price_basis"] == "unadjusted" and
            market["universe"] == universe, "stock native market scope required")
    calendar = market["calendar"]
    require(type(calendar) is list and calendar == sorted(set(calendar)) and bool(calendar), "ordered stock calendar required")
    for day in calendar:
        session(day)
    require(plan["start_session"] in calendar and plan["end_session"] in calendar and
            calendar.index(plan["start_session"]) > 0 and plan["start_session"] <= plan["end_session"], "stock scope needs previous session")
    refs = market["source_refs"]
    require(type(refs) is list and bool(refs) and len(set(refs)) == len(refs), "explicit stock market references required")
    for ref in refs:
        digest(ref)
    require(type(market["source_evidence"]) is list and
            {entry["reference"] for entry in market["source_evidence"]} == set(refs), "stock evidence closure mismatch")
    batches = native_batches(market["source_evidence"], market.get("coverage_bundle", []))
    require(len(batches) == (7 if full else 6), "complete stock native closure required")
    for i, batch in enumerate(batches):
        context = batch["context"]
        require(context["query"]["purpose"] == ("decision_facts" if full and i == 6 else "market_replay") and
                context["snapshot_id"] == evidence["execution"]["snapshot"], "stock market purpose/Snapshot mismatch")
    require(calendar == evidence["scope"]["initial_sessions"], "stock admitted calendar mismatch")
    required_features = calendar[calendar.index(plan["start_session"]) - 1:calendar.index(plan["end_session"])]
    if v4:
        require(all(day in signals for day in calendar[calendar.index(plan["start_session"]):calendar.index(plan["end_session"])+1]),
                "missing required stock trade schedule group")
    else:
        require(all((day, security) in signals for day in required_features for security in signal["universe"]),
                "missing required previous-session prediction group")
    from .stock_market import _stock_market_wire
    # Verify frozen native projections, without Data access or account execution.
    projection = {}
    if full:
        rules = profile["stock_execution_rules"]
        identities, _ = validate_execution_rules(rules)
        require(rules["calendar"] == calendar and plan["stock_execution_rules_ref"] == profile["stock_execution_rules_ref"] ==
                market["stock_execution_rules_ref"] == policy["stock_execution_rules_ref"] and
                market["membership_ref"] == native_ref(batches[6]) and market["lifecycle_policy"] == LIFECYCLE_POLICY,
                "stock rule/member identity mismatch")
        from .stock_market import _membership_index
        member_rows = _membership_index(batches[6], universe=universe, calendar=calendar, identities=identities)
        for fold in signal["folds"]:
            require(all(row["member"] == member_rows[row["session"], row["security_id"]]["is_member"]
                        for row in fold["prediction_frame"]["rows"]), "saved prediction differs from original PIT membership")
        projection = dict(execution_rules=rules, membership_batch=batches[6])
    _validate_pair_proof(evidence, signal, universe, calendar, batches[:6], lifecycle=full,
                         membership_batch=batches[6] if full else None)
    require(market == _stock_market_wire(batches=batches[:6], universe=universe, calendar=calendar,
            saved_evidence=market["source_evidence"], coverage_bundle=market.get("coverage_bundle", []), **projection),
            "stock market projection differs from saved native facts")
    indexed = {}
    for row in market["rows"]:
        fields(row, "security_id session open close volume_shares limit_up limit_down close_available_at market_state state_reason source_refs execution_evidence_cutoff field_available_at")
        require(row["security_id"] in universe and row["session"] in calendar, "stock market key outside scope")
        require(row["market_state"] in (("normal_trading", "unknown_status", "suspended", "source_gap", "not_listed", "delisted") if full else
                ("normal_trading", "unknown_status", "suspended", "source_gap")), "invalid stock market state")
        require(row["state_reason"] is None or type(row["state_reason"]) is str, "invalid stock state reason")
        require(row["execution_evidence_cutoff"] == row["session"] + "T20:30:00+08:00", "stock retrospective evidence clock required")
        require(type(row["source_refs"]) is list and bool(row["source_refs"]) and set(row["source_refs"]) <= set(refs), "stock row source closure mismatch")
        for name in ("open", "close", "limit_up", "limit_down"):
            if row[name] is not None:
                require(decimal(row[name], minimum=0) > 0, "positive stock price required")
        if row["volume_shares"] is not None:
            integer(row["volume_shares"])
        if row["limit_up"] is not None and row["limit_down"] is not None:
            low, high = decimal(row["limit_down"]), decimal(row["limit_up"])
            require(low <= high, "contradictory stock limits")
            require(all(row[name] is None or low <= decimal(row[name]) <= high for name in ("open", "close")), "stock price outside native limits")
        cutoff = instant(row["execution_evidence_cutoff"])
        fields(row["field_available_at"], "open close volume_shares limit_up limit_down market_state")
        for name, value in row["field_available_at"].items():
            if row[name] is not None and name != "market_state":
                require(value is not None, "observed stock value lacks availability")
            require(value is None or instant(value) <= cutoff, "future stock execution evidence")
        require(row["close"] is None or instant(row["close_available_at"]) == instant(row["field_available_at"]["close"]), "stock close clock mismatch")
        key = row["session"], row["security_id"]
        require(key not in indexed, "duplicate stock market key")
        indexed[key] = row
    require(set(indexed) == {(day, security) for day in calendar for security in universe}, "incomplete stock execution-union coverage")
    if full:
        factors = {(r["session"], r["security_id"]): r["factor"] for r in batches[3]["records"]}
        factor_meta = {(r["session"], r["security_id"]): r for r in batches[3]["field_meta"]["factor"]["by_key"]}
        factor_ref = native_ref(batches[3])
        close_meta = {(r["session"], r["security_id"]): r for r in batches[1]["field_meta"]["close"]["by_key"]}
        indexed = {key: {**row, "_stock_listed": listed(identities[key[1]], key[0]),
            "_stock_factor_valid": factors[key] is not None,
            "_stock_factor_missing_reason": factor_meta[key].get("missing_reason"),
            "_stock_factor_source_ref": factor_ref, "_stock_close_missing_reason": close_meta[key].get("missing_reason"),
            "_stock_member": member_rows[key]["is_member"]} for key, row in indexed.items()}
    seen = set()
    for action in market["cash_dividends"]:
        validate_cash_action(action, universe, refs)
        require(action["event_id"] not in seen, "duplicate stock cash action")
        seen.add(action["event_id"])
        require(action["record_session"] in calendar or action["record_session"] < calendar[0], "missing stock record calendar")
        for name in ("ex_session", "pay_session"):
            require(action[name] is None or action[name] > plan["end_session"] or action[name] < calendar[0] or action[name] in calendar,
                    "stock cash phase calendar mismatch")
    for block in market["action_blocks"]:
        fields(block, "security_id effective_session available_at reason source_refs")
        require(block["security_id"] in universe and block["reason"] in
                ("AMBIGUOUS_IMPLEMENTED_ACTION", "UNKNOWN_IMPLEMENTATION_STATUS", "UNSUPPORTED_QUANTITY_ACTION", "MISSING_IMPLEMENTED_CASH_FACT", "UNEXPLAINED_FACTOR_CHANGE"), "invalid stock action block")
        if block["effective_session"] is not None:
            session(block["effective_session"])
        instant(block["available_at"])
        require(bool(block["source_refs"]) and set(block["source_refs"]) <= set(refs), "unbound stock block")
    fields(plan["initial_account"], "cash_minor positions")
    require(not plan["initial_account"]["positions"], "first stock path requires empty initial holdings")
    integer(plan["initial_account"]["cash_minor"], 1)
    return plan, signal, signals, market, calendar, indexed, profile


def validate_saved_stock_core(wire):
    """Check saved version tuples and decisions, without replaying the planner."""
    version = wire["core_version"]
    full = wire["contract_version"] == "backtest_run_v6"
    v4 = wire["contract_version"] in ("backtest_run_v4", "backtest_run_v6")
    require(not full or version == "axiom.stock_portfolio/3", "v6 saved stock Core required")
    require(not v4 or version == ("axiom.stock_portfolio/3" if full else "axiom.stock_portfolio/2"), "saved stock schedule/Core mismatch")
    k = wire["plan"]["portfolio_policy"]["top_k"]
    require(version in ("axiom.stock_portfolio/1", "axiom.stock_portfolio/2", "axiom.stock_portfolio/3"), "unsupported saved stock Core")
    if version == "axiom.stock_portfolio/1":
        require(type(k) is int and k == 5, "legacy stock Core only supports Top5")
    require(type(wire["decisions"]) is list, "saved decisions required")
    if v4:
        schedule = wire["plan"]["prediction_schedule"]
        require(wire["signal_ref"] == schedule["schedule_ref"], "Saved run/schedule identity mismatch")
        frames = {f["prediction_frame"]["signal_run_ref"]: f["prediction_frame"] for f in schedule["folds"]}
        trades = {t["trade_session"]: t for t in schedule["trade_schedule"]}
    for decision in wire["decisions"]:
        require(decision.get("contract_version") == version, "saved decision/Core version mismatch")
        if version in ("axiom.stock_portfolio/2", "axiom.stock_portfolio/3"):
            require(type(decision.get("top_k")) is int and decision["top_k"] == k,
                    "saved decision/portfolio top_k mismatch")
        if full:
            require(decision.get("stock_execution_rules_ref") == wire["plan"]["stock_execution_rules_ref"],
                    "saved decision/rule identity mismatch")
        if v4:
            trade = trades.get(decision.get("trade_session"))
            require(trade is not None and trade["signal_run_ref"] == decision.get("signal_ref") and
                    trade["feature_session"] == decision.get("feature_session"), "Saved decision differs from admitted trade mapping")
            frame = frames[trade["signal_run_ref"]]
            original = next((r for r in frame["rows"] if r["session"] == decision["feature_session"] and
                             r["security_id"] == frame["universe"][0]), None)
            require(original is not None and decision.get("prediction_clock") == {
                "clock_basis": frame["clock_basis"], "feature_knowledge_cutoff": original["feature_knowledge_cutoff"],
                "inference_cutoff": original["knowledge_cutoff"], "simulated_model_available_at": original["simulated_model_available_at"],
                "model_ref": frame["model_ref"], "fold_spec_ref": frame["fold_spec_ref"]}, "Saved decision clock differs from original prediction")
    if full:
        _validate_saved_v6_execution(wire)


def _validate_saved_v6_execution(wire):
    """Verify recorded submission and actual fee provenance; do not execute an account."""
    from ..core.stock_rules import rule_at
    from .stock_rules import fee_at
    from .backtest import _fee_components
    profile = wire["plan"]["profile"]
    reference = profile["stock_execution_rules_ref"]
    require(wire.get("stock_execution_rules_ref") == reference, "saved stock rule identity mismatch")
    validated = validate_execution_rules(profile["stock_execution_rules"])
    fields(wire.get("lifecycle_admission", {}), "pre_listing_null listed_nonmember_gap member_gap held_gap")
    for count in wire["lifecycle_admission"].values():
        integer(count)
    intents = {}
    for decision in wire["decisions"]:
        for intent in decision["intents"]:
            require(intent["intent_id"] not in intents, "duplicate saved stock intent")
            intents[intent["intent_id"]] = (decision["trade_session"], intent)
    market_rows = {(r["session"], r["security_id"]): r for r in wire["plan"]["market_replay"]["rows"]}
    seen_intents = set()
    orders = {}
    for order in wire["orders"]:
        require(order["order_id"] not in orders and order.get("stock_execution_rules_ref") == reference,
                "saved order/rule identity mismatch")
        linked = intents.get(order.get("intent_id"))
        require(linked is not None and order["intent_id"] not in seen_intents and order["session"] == linked[0] and
                order["requested_quantity"] == linked[1]["quantity"] and
                all(order.get(name) == linked[1][name] for name in
                    ("security_id", "side", "expected_account_version", "valid_until")), "saved stock order differs from original intent")
        seen_intents.add(order["intent_id"])
        for name in ("requested_quantity", "submitted_quantity", "unsubmitted_quantity", "filled_quantity", "unfilled_quantity", "quantity"):
            integer(order.get(name))
        require(order["requested_quantity"] == order["submitted_quantity"] + order["unsubmitted_quantity"] and
                order["quantity"] == order["submitted_quantity"] == order["filled_quantity"] + order["unfilled_quantity"],
                "saved stock submission quantities do not reconcile")
        if order["submitted_quantity"]:
            rule = rule_at(profile["stock_execution_rules"], order["security_id"], order["session"], validated=validated)
            require(order.get("quantity_rule_effective_from") == rule["effective_from"] and
                    order["submitted_quantity"] <= rule["daily_proxy_maximum"], "saved stock submitted quantity rule mismatch")
            if order["side"] == "BUY":
                require(order["submitted_quantity"] >= rule["buy_minimum"] and
                        (order["submitted_quantity"] - rule["buy_minimum"]) % rule["buy_increment"] == 0,
                        "saved stock illegal buy submission")
        orders[order["order_id"]] = order
    require(seen_intents == set(intents), "saved stock intent lacks its order")
    filled = {}
    for fill in wire["fills"]:
        order = orders.get(fill["order_id"])
        require(order is not None and fill["order_id"] not in filled and fill["session"] == order["session"] and
                fill["security_id"] == order["security_id"] and fill["side"] == order["side"] and
                type(fill["quantity"]) is int and 0 < fill["quantity"] == order["filled_quantity"], "saved stock fill/order mismatch")
        interval = fee_at(profile["stock_fee_schedule"], fill["session"])
        rule = rule_at(profile["stock_execution_rules"], fill["security_id"], fill["session"], validated=validated)
        require(fill.get("stock_execution_rules_ref") == reference and
                fill.get("quantity_rule_effective_from") == rule["effective_from"] and
                fill.get("stock_fee_schedule_ref") == profile["stock_fee_schedule_ref"] and
                fill.get("fee_interval_effective_from") == interval["effective_from"], "saved stock fill provenance mismatch")
        row = market_rows[fill["session"], fill["security_id"]]
        require(row["open"] is not None and decimal(fill["price"]) == decimal(row["open"]) and
                decimal(fill["reference_open"]) == decimal(row["open"]) and fill["source_refs"] == row["source_refs"] and
                fill["execution_evidence_cutoff"] == row["execution_evidence_cutoff"] and
                fill["field_available_at"] == row["field_available_at"] and
                (fill["market_state"], fill["state_reason"]) == (row["market_state"], row["state_reason"]),
                "saved stock fill differs from original native execution facts")
        gross, commission, stamp, transfer = _fee_components(decimal(fill["price"]), fill["quantity"], fill["side"], profile, fill["session"])
        require((fill["gross_minor"], fill["commission_minor"], fill["tax_minor"], fill["stamp_tax_minor"], fill["transfer_fee_minor"], fill["fee_minor"]) ==
                (gross, commission, stamp, stamp, transfer, commission + stamp + transfer), "saved stock actual-fill fee mismatch")
        require(fill["cash_delta_minor"] == (-gross - commission - stamp - transfer if fill["side"] == "BUY" else
                gross - commission - stamp - transfer), "saved stock fill cash/fee mismatch")
        filled[fill["order_id"]] = fill
    require(all(bool(o["filled_quantity"]) == (key in filled) for key, o in orders.items()), "saved stock order lacks actual fill")
    # Earlier saved v6 candidates lacked these counters; preserve their wire on read.
    names = {"unsubmitted_order_count", "unsubmitted_quantity", "incomplete_order_count"}
    if names & set(wire["metrics"]):
        require(names <= set(wire["metrics"]), "incomplete saved stock submission counters")
        expected = dict(unsubmitted_order_count=sum(o["unsubmitted_quantity"] > 0 for o in orders.values()),
            unsubmitted_quantity=sum(o["unsubmitted_quantity"] for o in orders.values()),
            incomplete_order_count=sum(o["unsubmitted_quantity"] + o["unfilled_quantity"] > 0 for o in orders.values()))
        require(all(type(wire["metrics"][name]) is int and wire["metrics"][name] == expected[name] for name in names) and
                wire["metrics"]["unfilled_order_count"] == sum(o["unfilled_quantity"] > 0 for o in orders.values()),
                "saved stock submission counters differ from recorded orders")
