"""Small in-memory offline ledger. Money is integer CNY fen, not float sums."""
from copy import deepcopy
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from ..core.contracts import canonical, integer, require
from ..core.portfolio import decimal, minor


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
            amount = minor(decimal(action["cash_per_unit"], minimum=0) * entitlement * 100)
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
