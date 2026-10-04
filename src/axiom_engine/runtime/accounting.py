"""Small in-memory offline ledger. Money is integer CNY fen, not float sums."""
from copy import deepcopy
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from ..core.contracts import canonical, integer, require
from ..core.portfolio import decimal, minor
from .unit_splits import UNIT_SPLIT_PHASE, eod


class AccountLedger:
    """Incremental fills are idempotent; conflicting payloads fail before mutation.

    This is an isolated simulation ledger, without a live outbox or database.
    """
    def __init__(self, *, cash_minor, calendar, settlement_sessions, positions=None):
        integer(cash_minor); integer(settlement_sessions)
        self.cash = cash_minor
        self.calendar = tuple(calendar)
        self.settlement_sessions = settlement_sessions
        self.positions = deepcopy(positions or {})
        for position in self.positions.values():
            require(set(position) == {"quantity", "sellable_quantity", "cost_minor"}, "invalid initial position")
            for value in position.values():
                integer(value)
            require(position["sellable_quantity"] <= position["quantity"], "invalid initial sellability")
            require(position["sellable_quantity"] == position["quantity"], "initial unsettled lots unsupported")
        self.sequence = 0
        self.pending = []
        self.fills = []
        self.cash_ledger = []
        self.position_ledger = []
        self._applied = {}
        self.receivables = {}
        self._session = None

    def account(self):
        return {"cash_minor": self.cash, "version": self.sequence,
                "positions": {s: {k: p[k] for k in ("quantity", "sellable_quantity")}
                              for s, p in self.positions.items() if p["quantity"]}}

    def advance(self, session):
        require(session in self.calendar and (self._session is None or session >= self._session), "account clock cannot go backwards")
        self._session = session
        for lot in list(self.pending):
            if lot["sellable_index"] <= self.calendar.index(session):
                self.sequence += 1
                self.positions[lot["security_id"]]["sellable_quantity"] += lot["quantity"]
                self.position_ledger.append({"sequence": self.sequence, "session": session,
                    "security_id": lot["security_id"], "quantity_delta": 0,
                    "sellable_delta": lot["quantity"], "cost_delta_minor": 0,
                    "reason": "SETTLEMENT", "source_event_id": lot["fill_id"]})
                self.pending.remove(lot)

    def apply_fill(self, fill):
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            return self._apply_fill(fill)

    def _apply_fill(self, fill):
        payload = canonical(fill)
        key = fill["fill_id"]
        if key in self._applied:
            require(self._applied[key] == payload, "conflicting fill payload")
            return False
        quantity = fill["quantity"]
        integer(quantity, 1); integer(fill["fee_minor"])
        price = decimal(fill["price"], minimum=0)
        require(price > 0 and fill["session"] in self.calendar, "invalid fill price/session")
        require(self._session is None or fill["session"] >= self._session, "fill cannot precede account clock")
        gross = minor(price * quantity * 100)
        require(fill["gross_minor"] == gross, "fill gross does not reconcile")
        require(fill["side"] in ("BUY", "SELL"), "invalid fill side")
        buy = fill["side"] == "BUY"
        delta = -gross - fill["fee_minor"] if buy else gross - fill["fee_minor"]
        require(fill["cash_delta_minor"] == delta and self.cash + delta >= 0, "insufficient or inconsistent cash")
        security = fill["security_id"]
        position = self.positions.get(security, {"quantity": 0, "sellable_quantity": 0, "cost_minor": 0})
        require(buy or quantity <= position["sellable_quantity"], "T+1 / insufficient sellable quantity")
        cost_delta = gross + fill["fee_minor"] if buy else -minor(
            Decimal(position["cost_minor"]) * quantity / position["quantity"])
        self.sequence += 1
        self._session = fill["session"]
        self.cash += delta
        position = self.positions.setdefault(security, position)
        position["quantity"] += quantity if buy else -quantity
        position["cost_minor"] += cost_delta
        available_delta = quantity if buy and self.settlement_sessions == 0 else (0 if buy else -quantity)
        position["sellable_quantity"] += available_delta
        if buy and self.settlement_sessions:
            self.pending.append({"fill_id": key, "security_id": security, "quantity": quantity,
                "sellable_index": self.calendar.index(fill["session"]) + self.settlement_sessions})
        self._applied[key] = payload
        self.fills.append({**fill, "sequence": self.sequence,
                           "realized_pnl_minor": 0 if buy else delta + cost_delta})
        self.cash_ledger.append({"sequence": self.sequence, "session": fill["session"],
            "cash_delta_minor": delta, "receivable_delta_minor": 0, "balance_minor": self.cash,
            "reason": "FILL", "source_event_id": key})
        self.position_ledger.append({"sequence": self.sequence, "session": fill["session"],
            "security_id": security, "quantity_delta": quantity if buy else -quantity,
            "sellable_delta": available_delta, "cost_delta_minor": cost_delta,
            "reason": "FILL", "source_event_id": key})
        return True

    def dividend(self, action, phase, entitlement):
        key = action["event_id"] + ":" + phase
        payload = canonical({"action": action, "phase": phase, "quantity": entitlement})
        if key in self._applied:
            require(self._applied[key] == payload, "conflicting corporate action")
            return
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            name = "cash_before_tax_per_share" if action.get("contract_version") == "stock_cash_action_v1" else "cash_per_unit"
            amount = minor(decimal(action[name], minimum=0) * entitlement * 100)
        receivable_delta, cash_delta = (amount, 0) if phase == "EX" else (-amount, amount)
        require(phase in ("EX", "PAY"), "unsupported dividend phase")
        if phase == "PAY":
            require(self.receivables.get(action["event_id"]) == amount, "dividend not yet recognized")
            del self.receivables[action["event_id"]]
        else:
            self.receivables[action["event_id"]] = amount
        self.sequence += 1
        self.cash += cash_delta
        self.cash_ledger.append({"sequence": self.sequence,
            "session": action["ex_session"] if phase == "EX" else action["pay_session"],
            "cash_delta_minor": cash_delta, "receivable_delta_minor": receivable_delta,
            "balance_minor": self.cash, "reason": "DIVIDEND_" + phase,
            "source_event_id": action["event_id"]})
        self._applied[key] = payload

    def unit_split(self, item, registration, quote):
        """Atomic settled-unit replacement; no cash, fee, fill or new T+1 lot."""
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            event = item["event"]
            key = event["event_id"] + ":UNIT_SPLIT"
            payload = canonical({"item": item, "registration": registration, "quote": quote})
            if key in self._applied:
                require(self._applied[key] == payload, "conflicting unit split payload")
                return None
            security = event["security_id"]
            position = self.positions.get(security, {"quantity": 0, "sellable_quantity": 0, "cost_minor": 0})
            require(not any(lot["security_id"] == security for lot in self.pending) and position == registration["position"] and
                    position["quantity"] == position["sellable_quantity"], "unit split requires unchanged fully settled entitlement")
            require(not any(f["security_id"] == security and f["sequence"] > registration["sequence"] for f in self.fills),
                    "trading after unit registration unsupported")
            require(quote is not None and quote["source_refs"] and quote["session"] <= event["effective_date"] and
                    quote["available_at"] <= eod(event["effective_date"]), "unit split lacks usable old-unit quote")
            quantity, n, d = position["quantity"], event["ratio_numerator"], event["ratio_denominator"]
            integer(n, 1); integer(d, 1)
            require(n > d, "unit consolidation or unchanged ratio unsupported")
            scaled, remainder = divmod(quantity * n, d)
            require(not remainder or event["quantity_rounding"] == "ceiling_to_whole_fund_unit",
                    "fractional units require disclosed rounding")
            new_quantity = scaled + bool(remainder)
            old_price = decimal(quote["price"], minimum=0)
            require(old_price > 0, "positive old-unit quote required")
            normalized = {**deepcopy(quote), "price": str(old_price * d / n),
                "available_at": max(quote["available_at"], item["available_at"], eod(event["effective_date"])),
                "source_refs": sorted(set(quote["source_refs"] + item["source_refs"]))}
            before = minor(old_price * quantity * 100)
            after = minor(decimal(normalized["price"]) * new_quantity * 100)
            application = {"event_id": event["event_id"], "security_id": security,
                "session": event["effective_date"], "phase": UNIT_SPLIT_PHASE, "sequence": self.sequence + 1,
                "status": "APPLIED" if quantity else "NO_ENTITLEMENT", "record_sequence": registration["sequence"],
                "record_quantity": registration["position"]["quantity"], "before_quantity": quantity,
                "after_quantity": new_quantity, "before_sellable_quantity": position["sellable_quantity"],
                "after_sellable_quantity": new_quantity, "cost_minor": position["cost_minor"],
                "rounding_extra_fraction": {"numerator": new_quantity * d - quantity * n, "denominator": d},
                "original_quote": deepcopy(quote), "normalized_quote": normalized,
                "before_market_value_minor": before, "after_market_value_minor": after,
                "rounding_value_minor": after - before, "source_refs": normalized["source_refs"]}
            self.sequence += 1
            if quantity:
                position["quantity"] = position["sellable_quantity"] = new_quantity
            self.position_ledger.append({"sequence": self.sequence, "session": event["effective_date"],
                "security_id": security, "quantity_delta": new_quantity - quantity,
                "sellable_delta": new_quantity - quantity, "cost_delta_minor": 0,
                "reason": "UNIT_SPLIT", "source_event_id": event["event_id"]})
            self._applied[key] = payload
            return application
