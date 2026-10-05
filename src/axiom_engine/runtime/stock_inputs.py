"""Stock-specific input admission; common account execution remains in backtest."""
import re

from ..core.contracts import Document, digest, fields, integer, require, session, text
from ..core.stock_portfolio import BUDGET_BASIS, StockPredictionFrame, instant, validate_stock_predictions, validate_top_k
from ..core.portfolio import decimal
from .profiles import stock_daily_open_profile
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


def _validate_pair_proof(evidence, signal, universe, calendar, batches):
    """Verify the saved owner's complete admission closure, without repeating Data queries."""
    require(evidence.get("contract_version") == "stock_snapshot_pair_admission_v1" and
            evidence.get("listing_identity_equal_all_83") is True and
            evidence["scope"].get("membership_security_ids") == signal["universe"], "incomplete stock identity/member proof")
    member = evidence.get("prediction_membership", {})
    require(member.get("compared") == len(signal["rows"]) and member.get("saved_prediction_rows") == len(signal["rows"]) and
            member.get("model_mismatch") == [] and member.get("execution_mismatch") == [], "incomplete saved prediction/member pairing")
    count = len(universe) * len(calendar)
    basis = evidence.get("previous_close_basis_checks", {})
    require(all(basis.get(name) is True for name in ("all_available_at_feature_knowledge_cutoff", "all_available_before_decision",
            "equal_all_83_23", "listing_suffix_and_SZSE_identity_all_83")) and
            basis.get("paired_rows_per_root") == count and basis.get("checked_rows_both_roots") == 2 * count,
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
    for key, batch in (("execution-states", batches[0]), ("execution-market", batches[1]), ("execution-factor", batches[3])):
        entry = manifests[key]
        require(entry["wire_ref"] == native_ref(batch) and entry["query"] == batch["context"]["query"] and
                entry["reader_version"] == batch["context"]["reader_version"], "stock execution facts differ from admitted batches")


def validate_stock_request(plan):
    fields(plan, "contract_version account_id start_session end_session signal_frame market_replay initial_account profile prediction_universe execution_universe supported_universe_ref portfolio_policy admission_ref admission_evidence stock_action_policy")
    require(plan["contract_version"] == "backtest_request_v3" and plan["stock_action_policy"] == ACTION_POLICY,
            "unsupported stock request/action policy")
    text(plan["account_id"]); session(plan["start_session"]); session(plan["end_session"])
    signal, signals = validate_stock_predictions(StockPredictionFrame.from_dict(plan["signal_frame"]))
    universe = plan["execution_universe"]
    require(plan["prediction_universe"] == signal["universe"] and universe == supported_universe(signal["universe"]),
            "stock prediction/execution scope mismatch")
    require(bool(universe) and plan["supported_universe_ref"] == support_ref(universe), "stock support identity mismatch")
    policy = plan["portfolio_policy"]
    require(policy == stock_portfolio_policy(top_k=policy.get("top_k"), execution_universe=universe),
            "unsupported stock portfolio policy")
    digest(plan["admission_ref"])
    evidence = dict(plan["admission_evidence"])
    recorded = evidence.pop("admission_ref", None)
    require(recorded == plan["admission_ref"] and Document.from_dict(evidence).identity == recorded,
            "stock input-pair evidence identity mismatch")
    require(evidence["signal_run_ref"] == signal["signal_run_ref"] and
            evidence["feature_ref"] == signal["feature_ref"] and evidence["model_ref"] == signal["model_ref"] and
            evidence["scope"]["execution_security_ids"] == universe and
            evidence["status"] == "NUMERIC_POLICY_IDENTITY_PAIR_PASS", "stock input-pair admission scope mismatch")
    profile = plan["profile"]
    require(profile == stock_daily_open_profile(unknown_status_policy=profile.get("unknown_status_policy")),
            "stock profile parameters differ from frozen contract")
    market = plan["market_replay"]
    fields(market, "contract_version price_basis calendar universe rows cash_dividends action_diagnostics action_blocks source_refs source_evidence limitations" +
           (" coverage_bundle" if "coverage_bundle" in market else ""))
    require(market["contract_version"] == "market_replay_v3" and market["price_basis"] == "unadjusted" and
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
    for batch in batches:
        context = batch["context"]
        require(context["query"]["purpose"] == "market_replay" and
                context["snapshot_id"] == evidence["execution"]["snapshot"], "stock market purpose/Snapshot mismatch")
    require(calendar == evidence["scope"]["initial_sessions"], "stock admitted calendar mismatch")
    required_features = calendar[calendar.index(plan["start_session"]) - 1:calendar.index(plan["end_session"])]
    require(all((day, security) in signals for day in required_features for security in signal["universe"]),
            "missing required previous-session prediction group")
    from .stock_market import _stock_market_wire
    # Verify frozen native projections, without Data access or account execution.
    require(len(batches) == 6, "complete stock native closure required")
    _validate_pair_proof(evidence, signal, universe, calendar, batches)
    require(len(batches) == 6 and market == _stock_market_wire(batches=batches, universe=universe, calendar=calendar,
            saved_evidence=market["source_evidence"], coverage_bundle=market.get("coverage_bundle", [])),
            "stock market projection differs from saved native facts")
    indexed = {}
    for row in market["rows"]:
        fields(row, "security_id session open close volume_shares limit_up limit_down close_available_at market_state state_reason source_refs execution_evidence_cutoff field_available_at")
        require(row["security_id"] in universe and row["session"] in calendar, "stock market key outside scope")
        require(row["market_state"] in ("normal_trading", "unknown_status", "suspended", "source_gap"), "invalid stock market state")
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
