"""Pure, serializable weekly stock strategies; execution remains in Runtime."""
from copy import deepcopy
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from typing import Protocol

from .contracts import Document, digest, fields, integer, number, require, session
from .portfolio import decimal

STRATEGY_PORTFOLIO_VERSION = "axiom.stock_portfolio/4"
COST_BASIS = "fee_inclusive_ledger_cost_per_owned_share_v1"


class StrategyTargets(Document):
    """Immutable targets and next strategy state, without executable orders."""


class StockStrategy(Protocol):
    def plan(self, *, inputs: dict, state: dict) -> StrategyTargets: ...


def stock_strategy_spec(*, selection="top_k", top_k=5, rank_buffer=None, exit_overlay=None):
    wire = deepcopy(dict(contract_version="stock_strategy_spec_v1", selection=selection,
                         top_k=top_k, rank_buffer=rank_buffer, exit_overlay=exit_overlay))
    validate_strategy_spec(wire)
    return wire


def validate_strategy_spec(wire):
    fields(wire, "contract_version selection top_k rank_buffer exit_overlay")
    require(wire["contract_version"] == "stock_strategy_spec_v1", "unsupported strategy spec")
    integer(wire["top_k"], 1)
    require(wire["selection"] in ("top_k", "rank_buffer"), "unsupported stock selection")
    if wire["selection"] == "rank_buffer":
        fields(wire["rank_buffer"], "keep_rank")
        integer(wire["rank_buffer"]["keep_rank"], wire["top_k"])
    else:
        require(wire["rank_buffer"] is None, "TopK does not take a rank buffer")
    overlay = wire["exit_overlay"]
    if overlay is not None:
        fields(overlay, "take_profit stop_loss cost_basis check cooldown_rebalance_sessions reentry")
        for key in ("take_profit", "stop_loss"):
            require(type(overlay[key]) is str and decimal(overlay[key]) > 0, "positive Decimal exit threshold required")
        require(decimal(overlay["stop_loss"]) < 1, "stop loss must be below one")
        integer(overlay["cooldown_rebalance_sessions"])
        require(overlay["cost_basis"] == COST_BASIS and overlay["check"] == "rebalance_sessions_only" and
                overlay["reentry"] == "after_flat_and_cooldown", "unsupported weekly-close exit convention")
    return wire


def initial_strategy_state():
    return dict(contract_version="stock_strategy_state_v1", check_index=0, exits={})


def strategy_support_ref(universe, universe_id):
    require(type(universe_id) is str and universe_id and universe_id == universe_id.strip(), "explicit membership universe ID required")
    if universe_id == "csi300":
        from .stock_rules import support_ref
        return support_ref(universe)
    return Document.from_dict(dict(eligibility_id="pit_a_share_strategy_v1", universe_id=universe_id,
                                   security_ids=list(universe))).identity


def validate_strategy_policy(policy, universe, rules_ref):
    fields(policy, "contract_version eligibility_id universe_id top_k rebalance budget_basis candidate_policy stock_execution_rules_ref strategy_spec strategy_spec_ref")
    validate_strategy_spec(policy["strategy_spec"])
    require(policy["contract_version"] == "stock_portfolio_policy_v4" and
            policy["eligibility_id"] == "pit_a_share_strategy_v1" and
            policy["top_k"] == policy["strategy_spec"]["top_k"] and type(policy["top_k"]) is int and
            policy["top_k"] <= len(universe) and policy["rebalance"] == "weekly_first_trading_session" and
            policy["budget_basis"] == "available_cash_plus_previous_close_positions_excluding_receivables" and
            policy["candidate_policy"] == "member_valid_finite_v1" and
            policy["stock_execution_rules_ref"] == rules_ref and
            policy["strategy_spec_ref"] == Document.from_dict(policy["strategy_spec"]).identity,
            "unsupported stock strategy portfolio policy")
    strategy_support_ref(universe, policy["universe_id"])
    return policy


def validate_strategy_state(state, universe):
    fields(state, "contract_version check_index exits")
    require(state["contract_version"] == "stock_strategy_state_v1", "unsupported strategy state")
    integer(state["check_index"])
    require(type(state["exits"]) is dict and set(state["exits"]) <= set(universe), "exit state outside scope")
    for item in state["exits"].values():
        fields(item, "phase remaining_checks")
        require(item["phase"] in ("exit_pending", "cooldown"), "invalid strategy exit phase")
        integer(item["remaining_checks"])
        require(item["phase"] != "exit_pending" or item["remaining_checks"] == 0, "pending exit cannot expire")


def _instant(value):
    require(type(value) is str, "strategy timestamp required")
    try:
        point = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        from .contracts import ContractError
        raise ContractError("invalid strategy timestamp") from exc
    require(point.tzinfo is not None, "strategy timestamp must have a timezone")
    return point


class ConfiguredStockStrategy:
    """A small built-in composition implementing the public pure protocol."""
    def __init__(self, spec):
        validate_strategy_spec(spec)
        self._spec = Document.from_dict(spec)

    def plan(self, *, inputs, state):
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            return self._plan(inputs, state)

    def _plan(self, inputs, state):
        fields(inputs, "feature_session trade_session universe candidates positions pending_share_quantities cash_minor reference_prices risk_observations constraints")
        feature, trade = inputs["feature_session"], inputs["trade_session"]
        session(feature); session(trade)
        require(feature < trade, "strategy requires a completed prior session")
        cutoff = _instant(feature + "T20:30:00+08:00")
        universe = inputs["universe"]
        require(type(universe) is list and bool(universe) and len(set(universe)) == len(universe), "frozen strategy universe required")
        spec = self._spec.to_dict(); k = spec["top_k"]
        require(k <= len(universe), "strategy capacity outside universe")
        constraints = inputs["constraints"]
        fields(constraints, "max_positions reserved_security_ids" +
               (" quantity_rules" if "quantity_rules" in constraints else "") +
               (" eligible_security_ids" if "eligible_security_ids" in constraints else ""))
        eligible = constraints.get("eligible_security_ids", universe)
        require(type(eligible) is list and len(set(eligible)) == len(eligible) and set(eligible) <= set(universe),
                "strategy eligibility outside frozen universe")
        eligible = set(eligible)
        require(type(constraints["max_positions"]) is int and constraints["max_positions"] == k,
                "strategy constraints outside scope")
        # Compatibility for existing pure callers: validate a supplied complete
        # map, while the built-in weight strategy does not require or consume it.
        if "quantity_rules" in constraints:
            require(type(constraints["quantity_rules"]) is dict and
                    set(constraints["quantity_rules"]) == set(universe), "strategy constraints outside scope")
            for rule in constraints["quantity_rules"].values():
                fields(rule, "buy_minimum buy_increment sell_minimum sell_increment daily_proxy_maximum")
                for value in rule.values(): integer(value, 1)
        reserved = inputs["constraints"]["reserved_security_ids"]
        require(type(reserved) is list and len(set(reserved)) == len(reserved) and set(reserved) <= set(universe),
                "reserved stock rights outside scope")
        validate_strategy_state(state, universe)
        integer(inputs["cash_minor"])
        positions, pending = inputs["positions"], inputs["pending_share_quantities"]
        require(type(positions) is dict and type(pending) is dict and
                (set(positions) | set(pending)) <= set(universe), "strategy holdings outside scope")
        for p in positions.values():
            fields(p, "quantity sellable_quantity")
            integer(p["quantity"]); integer(p["sellable_quantity"])
            require(p["sellable_quantity"] <= p["quantity"], "invalid strategy sellability")
        for quantity in pending.values(): integer(quantity)
        held = {s for s in universe if positions.get(s, {}).get("quantity", 0) or pending.get(s, 0)} | set(reserved)
        require(len(held) <= k, "existing owned positions exceed strategy capacity")
        for quote in inputs["reference_prices"].values():
            fields(quote, "price session available_at source_refs")
            session(quote["session"])
            require(quote["session"] <= feature and _instant(quote["available_at"]) <= cutoff,
                    "future strategy reference price")
            require(decimal(quote["price"]) > 0, "positive strategy reference price required")
        candidates = inputs["candidates"]
        require(type(candidates) is list and len(candidates) == len(universe) and
                {r["security_id"] for r in candidates} == set(universe), "complete strategy candidate scope required")
        for row in candidates:
            fields(row, "security_id score member valid")
            require(type(row["member"]) is bool and type(row["valid"]) is bool, "explicit candidate masks required")
            if row["valid"]: number(row["score"])
            else: require(row["score"] is None, "invalid candidate cannot carry a score")
        # Keep original member/valid/score facts; the caller supplies a separate
        # lifecycle eligibility constraint after native identity admission.
        ranked = sorted((r for r in candidates if r["valid"] and r["member"] and r["security_id"] in eligible),
                        key=lambda r: (-r["score"], r["security_id"]))
        ranks = {r["security_id"]: i+1 for i, r in enumerate(ranked)}
        next_state = deepcopy(state); next_state["check_index"] += 1
        trace, blocked, forced, newly_forced = [], set(), set(), set()
        overlay = spec["exit_overlay"]
        require(overlay is not None or not state["exits"], "exit state requires an exit overlay")
        for s, item in list(next_state["exits"].items()):
            if item["phase"] == "exit_pending":
                if s in held:
                    forced.add(s); blocked.add(s)
                    continue
                item.update(phase="cooldown", remaining_checks=overlay["cooldown_rebalance_sessions"]+1)
            if item["remaining_checks"]:
                blocked.add(s); item["remaining_checks"] -= 1
                trace.append(dict(reason="REENTRY_COOLDOWN", security_id=s, remaining_checks=item["remaining_checks"]))
            else:
                del next_state["exits"][s]
        if overlay:
            observations = inputs["risk_observations"]
            require(type(observations) is dict and set(observations) <= set(universe), "risk observations outside scope")
            for s in sorted(held - forced):
                observation = observations.get(s)
                if observation is None:
                    trace.append(dict(reason="MISSING_WEEKLY_CLOSE_RISK_OBSERVATION", security_id=s)); continue
                fields(observation, "price session available_at source_refs cost_minor owned_quantity price_basis committed_sequence")
                session(observation["session"])
                require(observation["session"] <= feature and _instant(observation["available_at"]) <= cutoff,
                        "future weekly-close risk observation")
                integer(observation["cost_minor"]); integer(observation["owned_quantity"])
                integer(observation["committed_sequence"])
                require(observation["price_basis"] == "original_session_units", "risk price/cost unit mismatch")
                require(type(observation["source_refs"]) is list and bool(observation["source_refs"]), "risk provenance required")
                for ref in observation["source_refs"]: digest(ref)
                quote = inputs["reference_prices"].get(s)
                if observation["session"] == feature and observation["price"] is not None and quote is not None and quote["session"] == feature:
                    require(decimal(observation["price"]) == decimal(quote["price"]) and
                            observation["source_refs"] == quote["source_refs"], "risk close differs from same-session sizing reference")
                if observation["session"] != feature or observation["price"] is None or not observation["owned_quantity"] or not observation["cost_minor"]:
                    trace.append(dict(reason="UNUSABLE_WEEKLY_CLOSE_RISK_BASIS", security_id=s,
                                      detail="stale_or_missing_price_or_zero_owned_quantity_or_zero_cost")); continue
                require(decimal(observation["price"]) > 0, "positive risk close required")
                value = decimal(observation["price"]) * observation["owned_quantity"] * 100 / observation["cost_minor"] - 1
                reason = "WEEKLY_CLOSE_TAKE_PROFIT" if value >= decimal(overlay["take_profit"]) else (
                         "WEEKLY_CLOSE_STOP_LOSS" if value <= -decimal(overlay["stop_loss"]) else None)
                if reason:
                    newly_forced.add(s)
                    next_state["exits"][s] = dict(phase="exit_pending", remaining_checks=0)
                    trace.append(dict(reason=reason, security_id=s, observed_return=str(value), observation=deepcopy(observation)))
        frozen = {s for s in held - forced - newly_forced if s in reserved or pending.get(s, 0) or
                  positions.get(s, {}).get("sellable_quantity", 0) < positions.get(s, {}).get("quantity", 0)}
        preserve = {s: positions.get(s, {}).get("quantity", 0) for s in sorted(frozen)}
        trace.extend(dict(reason="FROZEN_OWNED_POSITION", security_id=s, preserved_quantity=preserve[s],
                          pending_share_quantity=pending.get(s, 0)) for s in sorted(frozen))
        available = [r["security_id"] for r in ranked if r["security_id"] not in blocked | frozen]
        selected = sorted(frozen)
        insufficient = spec["selection"] == "top_k" and len([s for s in ranks if s not in blocked]) < k
        if insufficient:
            preserve.update({s: positions.get(s, {}).get("quantity", 0) for s in sorted(held - forced)})
            selected = sorted(held - forced)
            trace.append(dict(reason="INSUFFICIENT_ELIGIBLE_MEMBERS", count=len([s for s in ranks if s not in blocked]), top_k=k))
        else:
            if spec["selection"] == "rank_buffer":
                kept = [s for s in available if s in held and ranks[s] <= spec["rank_buffer"]["keep_rank"]]
                selected.extend(kept[:max(0, k-len(selected)-len(forced))])
                trace.append(dict(reason="RANK_BUFFER_KEEP", security_ids=kept, keep_rank=spec["rank_buffer"]["keep_rank"]))
            selected.extend(s for s in available if s not in selected)
            selected = selected[:max(len(frozen), k-len(forced))]
        # Overlay the base selection once; never refill a newly exited slot.
        forced.update(newly_forced)
        selected = [s for s in selected if s not in forced]
        preserve = {s: q for s, q in preserve.items() if s not in forced}
        weighted = [s for s in selected if s not in preserve]
        budget = Decimal(inputs["cash_minor"])
        reserve = Decimal(0)
        for s in held:
            q = positions.get(s, {}).get("quantity", 0)
            quote = inputs["reference_prices"].get(s)
            if q and (quote is None or quote["session"] != feature):
                # Core retains the whole-decision missing-reference gate.
                continue
            value = Decimal(q) * decimal(quote["price"]) * 100 if q else Decimal(0)
            budget += value
            if s in preserve or s in forced: reserve += value
        ratio = Fraction(1, k)
        if weighted and budget:
            ratio = min(ratio, Fraction(max(Decimal(0), budget-reserve))/Fraction(budget)/len(weighted))
        elif not budget:
            ratio = Fraction(0)
        allocations = {s: dict(numerator=ratio.numerator, denominator=ratio.denominator) for s in weighted}
        weights = {s: str(Decimal(ratio.numerator)/ratio.denominator) for s in weighted}
        trace.append(dict(reason="STRATEGY_TARGET_BUDGET", budget_minor=str(budget), reserved_minor=str(reserve),
                          pending_rights="occupy_slots_excluded_from_deployable_budget"))
        wire = dict(contract_version="stock_strategy_targets_v1", strategy_spec_ref=self._spec.identity,
                    status="NO_DECISION" if insufficient and not forced else "DECISION_COMPLETE",
                    selected_security_ids=selected, target_weights=weights, allocation_ratios=allocations, preserve_quantities=preserve,
                    force_exit_security_ids=sorted(forced), next_state=next_state, trace=trace)
        require(not (set(preserve) & forced) and ratio*len(weighted) <= 1,
                "conflicting strategy targets")
        return StrategyTargets.from_dict(wire)
