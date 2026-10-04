"""Aggregate saved fill cash flows into 0→nonzero→0 position episodes.

This is a report calculation, not another cash/settlement ledger. Dividend
income is read from saved EX events and attributed to record-day ownership.
"""
from decimal import Decimal

from ..core.contracts import Document, integer, require
from ..core.portfolio import decimal, minor


def _new_episode(binding, security, fill=None, initial_quantity=0):
    sequence = None if fill is None else fill["sequence"]
    return {"episode_id": Document.from_dict({"input_run_ref": binding, "security_id": security,
        "entry_sequence": sequence}).identity, "security_id": security, "status": "OPEN",
        "entry_session": None if fill is None else fill["session"], "entry_sequence": sequence,
        "exit_session": None, "exit_sequence": None, "left_censored": fill is None,
        "initial_quantity": initial_quantity, "final_quantity": initial_quantity,
        "buy_cost_minor": 0, "sell_proceeds_minor": 0, "fees_minor": 0,
        "dividend_income_minor": 0, "receivable_minor": 0,
        "fill_refs": [], "dividends": []}


def _actions(run, scope):
    actions = {a["event_id"]: a for a in run["plan"]["market_replay"]["cash_dividends"]}
    if scope is not None:
        economic_fields = ("security_id", "record_session", "ex_session", "pay_session", "cash_per_unit")
        expected = {a["event_id"] for a in actions.values()
                    if scope["start_session"] <= a["record_session"] <= scope["end_session"]}
        require(expected <= {a["event_id"] for a in scope["actions"]},
                "dividend scope omits saved in-period event")
        for action in scope["actions"]:
            old = actions.get(action["event_id"])
            if old is not None:
                require(all(old[k] == action[k] for k in economic_fields), "dividend scope contradicts saved account event")
            else:
                require(action["ex_session"] > run["plan"]["end_session"],
                        "new in-period dividend would change the saved account; evaluation cannot correct it")
                actions[action["event_id"]] = action
    return sorted(actions.values(), key=lambda a: a["event_id"])


def evaluate_episodes(run, scope, binding):
    start, end = run["plan"]["start_session"], run["plan"]["end_session"]
    fills, active, episodes = run["fills"], {}, []
    actions = _actions(run, scope)
    records = {}
    for action in actions:
        records.setdefault(action["record_session"], []).append(action)
    for security, initial in sorted(run["plan"]["initial_account"]["positions"].items()):
        if initial["quantity"]:
            episode = _new_episode(binding, security, initial_quantity=initial["quantity"])
            episodes.append(episode)
            active[security] = episode
    previous_sequence, previous_day, seen = -1, start, set()
    for fill in fills:
        integer(fill["sequence"]); integer(fill["quantity"], 1)
        integer(fill["gross_minor"]); integer(fill["fee_minor"])
        require(fill["sequence"] > previous_sequence and previous_day <= fill["session"] <= end and
                fill["fill_id"] not in seen and fill["sequence"] < run["committed_sequence"], "invalid saved fill ordering/identity")
        require(fill["side"] in ("BUY", "SELL") and type(fill["cash_delta_minor"]) is int,
                "unsupported saved fill cash flow")
        expected = -fill["gross_minor"] - fill["fee_minor"] if fill["side"] == "BUY" else fill["gross_minor"] - fill["fee_minor"]
        require(fill["cash_delta_minor"] == expected, "saved fill cash flow does not reconcile")
        previous_sequence, previous_day = fill["sequence"], fill["session"]
        seen.add(fill["fill_id"])
    cash_events = {}
    for event in run["cash_ledger"]:
        key = event["reason"], event["source_event_id"]
        require(key not in cash_events, "duplicate saved cash event")
        cash_events[key] = event
    snapshots = {(p["session"], p["security_id"]): p for p in run["positions"]}
    require(len(snapshots) == len(run["positions"]), "duplicate saved position key")
    entitlements, cursor = {}, 0
    for point in run["nav"]:
        day = point["session"]
        while cursor < len(fills) and fills[cursor]["session"] == day:
            fill = fills[cursor]
            cursor += 1
            security, buy = fill["security_id"], fill["side"] == "BUY"
            episode = active.get(security)
            if episode is None:
                require(buy, "saved sell has no position episode")
                episode = _new_episode(binding, security, fill)
                episodes.append(episode)
                active[security] = episode
            delta = fill["quantity"] if buy else -fill["quantity"]
            require(episode["final_quantity"] + delta >= 0, "episode quantity cannot go negative")
            episode["final_quantity"] += delta
            episode["fees_minor"] += fill["fee_minor"]
            if buy:
                episode["buy_cost_minor"] -= fill["cash_delta_minor"]
            else:
                episode["sell_proceeds_minor"] += fill["cash_delta_minor"]
            episode["fill_refs"].append({"fill_id": fill["fill_id"], "sequence": fill["sequence"]})
            cash = cash_events.get(("FILL", fill["fill_id"]))
            require(cash is not None and cash["sequence"] == fill["sequence"] and
                    cash["cash_delta_minor"] == fill["cash_delta_minor"], "fill lacks matching saved cash event")
            if episode["final_quantity"] == 0:
                episode.update(status="CLOSED", exit_session=day, exit_sequence=fill["sequence"])
                del active[security]
        for action in records.get(day, []):
            episode = active.get(action["security_id"])
            if episode is not None:
                quantity = episode["final_quantity"]
                require(snapshots.get((day, action["security_id"]), {}).get("quantity") == quantity,
                        "record-day entitlement disagrees with saved EOD position")
                entitlements[action["event_id"]] = (episode, quantity)
    require(cursor == len(fills), "fill outside saved NAV scope")
    for action in actions:
        if action["event_id"] not in entitlements:
            continue
        episode, quantity = entitlements[action["event_id"]]
        ex = cash_events.get(("DIVIDEND_EX", action["event_id"]))
        pay = cash_events.get(("DIVIDEND_PAY", action["event_id"]))
        pending, recognized, receivable = 0, 0, 0
        if action["ex_session"] <= end:
            require(ex is not None and ex["session"] == action["ex_session"] and ex["cash_delta_minor"] == 0,
                    "owned dividend lacks saved EX recognition")
            recognized = ex["receivable_delta_minor"]
            integer(recognized)
            # Saved EX is authoritative. PAY is a transfer, not another income.
            if action["pay_session"] <= end:
                require(pay is not None and pay["session"] == action["pay_session"] and
                        pay["cash_delta_minor"] == recognized and pay["receivable_delta_minor"] == -recognized,
                        "saved dividend payment does not reconcile")
            else:
                require(pay is None, "payment beyond saved account scope")
                receivable = recognized
        else:
            require(ex is None and pay is None, "future dividend was recognized in saved account")
            pending = minor(decimal(action["cash_per_unit"], minimum=0) * quantity * 100)
        episode["dividend_income_minor"] += recognized
        episode["receivable_minor"] += receivable
        episode["dividends"].append({"event_id": action["event_id"], "record_session": action["record_session"],
            "ex_session": action["ex_session"], "pay_session": action["pay_session"], "entitlement_quantity": quantity,
            "recognition_sequence": None if ex is None else ex["sequence"], "payment_sequence": None if pay is None else pay["sequence"],
            "recognized_minor": recognized, "pending_minor": pending, "receivable_minor": receivable,
            "source_refs": action["source_refs"]})
    final_positions = run["final_account"]["positions"]
    for security in set(final_positions) | set(active):
        require(final_positions.get(security, {}).get("quantity", 0) == active.get(security, {}).get("final_quantity", 0),
                "episodes disagree with saved final positions")
    for episode in episodes:
        pending = sum(d["pending_minor"] for d in episode["dividends"])
        awaiting_ex = any(d["recognition_sequence"] is None for d in episode["dividends"])
        reasons = []
        if episode["status"] == "OPEN":
            reasons.append("OPEN_POSITION")
        if episode["left_censored"]:
            reasons.append("LEFT_CENSORED_ENTRY")
        if awaiting_ex:
            reasons.append("KNOWN_INCOME_PENDING_EX")
        eligible = not reasons
        observed_pnl = episode["sell_proceeds_minor"] + episode["dividend_income_minor"] - episode["buy_cost_minor"]
        mark = snapshots.get((end, episode["security_id"]), {}).get("market_value_minor")
        if episode["status"] == "OPEN":
            require(mark is not None, "open episode lacks saved final valuation")
        episode.update({"income_status": "PENDING_EX" if awaiting_ex else ("RECOGNIZED" if episode["dividends"] else
                ("NO_OBSERVED_ENTITLEMENT" if scope is not None else "COVERAGE_UNKNOWN")),
            "statistics_eligible": eligible, "exclusion_reasons": reasons,
            "pending_dividend_minor": pending if scope is not None or awaiting_ex else None,
            "net_pnl_minor": observed_pnl if eligible else None,
            "marked_pnl_minor": observed_pnl + mark if episode["status"] == "OPEN" and not episode["left_censored"] else None,
            "return_denominator_minor": episode["buy_cost_minor"],
            "net_return": str(Decimal(observed_pnl) / episode["buy_cost_minor"]) if eligible and episode["buy_cost_minor"] > 0 else None})
    eligible = [e for e in episodes if e["statistics_eligible"]]
    count = len(eligible)
    wins = sum(e["net_pnl_minor"] > 0 for e in eligible)
    valid_returns = [decimal(e["net_return"]) for e in eligible if e["net_return"] is not None]
    metrics = {"closed_count": sum(e["status"] == "CLOSED" for e in episodes), "eligible_closed_count": count,
        "open_count": sum(e["status"] == "OPEN" for e in episodes),
        "left_censored_count": sum(e["left_censored"] for e in episodes),
        "income_pending_count": sum(e["income_status"] == "PENDING_EX" for e in episodes),
        "win_count": wins, "loss_count": sum(e["net_pnl_minor"] < 0 for e in eligible),
        "tie_count": sum(e["net_pnl_minor"] == 0 for e in eligible),
        "win_rate": None if not count else str(Decimal(wins) / count),
        "mean_net_pnl_minor": None if not count else str(Decimal(sum(e["net_pnl_minor"] for e in eligible)) / count),
        "mean_episode_return": None if len(valid_returns) != count or not count else str(sum(valid_returns) / count),
        "return_denominator": "cumulative_buy_cost_including_fees", "weighting": "equal_closed_episode",
        "dividend_scope_status": "COVERAGE_UNKNOWN" if scope is None else "OBSERVED_RECORDS_ONLY"}
    return episodes, metrics
