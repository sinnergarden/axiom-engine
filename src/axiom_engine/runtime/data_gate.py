"""Pin Data inputs at a decision-session boundary and keep replay separate.

The caller supplies a Data facade. This gate never fetches, publishes, trades,
or substitutes label outcomes for decision facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any


class InputGateError(ValueError):
    """An input crosses the fixed session or purpose boundary."""


def _instant(value: Any) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputGateError("invalid clock/cutoff") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputGateError("clock/cutoff must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _identity(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str,
                         ensure_ascii=False, allow_nan=False)
    return "sha256:" + sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class BatchReference:
    decision_session: str
    purpose: str
    snapshot_id: str
    query_context: dict
    query_digest: str


class DecisionBatchGate:
    """Resolve one concrete Snapshot per session and save every actual read ref."""

    def __init__(self, data: Any):
        self.data = data
        self._snapshots: dict[str, str] = {}
        self._reads: dict[tuple[str, str, str], BatchReference] = {}

    def pin_session(self, session: str, *, snapshot: str = "current") -> str:
        """Resolve an alias once; repeated pinning cannot change this session."""
        try:
            if date.fromisoformat(session).isoformat() != session:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise InputGateError("invalid decision session") from exc
        resolved = self.data.resolve(snapshot) if snapshot in ("current", "latest") else snapshot
        if not isinstance(resolved, str) or resolved in ("", "current", "latest"):
            raise InputGateError("snapshot must resolve to concrete ID")
        earlier = self._snapshots.setdefault(session, resolved)
        if earlier != resolved:
            raise InputGateError("session Snapshot is already pinned")
        return earlier

    def _read(self, *, decision_session: str, query: Any, clock: Any,
              purpose: str) -> Any:
        if decision_session not in self._snapshots:
            raise InputGateError("pin session before reading")
        if getattr(query, "purpose", None) != purpose:
            raise InputGateError(f"{purpose} query required")
        current = _instant(clock)
        sessions = getattr(query, "sessions", None)
        cutoffs = getattr(query, "cutoff_by_session", None)
        if not sessions or not cutoffs or set(sessions) != set(cutoffs):
            raise InputGateError("complete per-session query cutoffs required")
        if any(s > decision_session or _instant(cutoffs[s]) > current for s in sessions):
            raise InputGateError("query exposes future session or cutoff")
        if purpose == "market_replay" and getattr(query, "price_basis", None) != "unadjusted":
            raise InputGateError("market replay uses unadjusted prices")
        snapshot = self._snapshots[decision_session]
        reader = self.data.read if purpose == "decision_facts" else self.data.read_market
        batch = reader(snapshot=snapshot, query=query)
        context = batch.to_json()["context"]
        if context.get("snapshot_id") != snapshot or context.get("query", {}).get("purpose") != purpose:
            raise InputGateError("Data returned a different Snapshot or purpose")
        ref = BatchReference(decision_session, purpose, snapshot, context,
                             _identity({"snapshot_id": snapshot, "context": context}))
        # A decision batch may need multiple domains (e.g. prices and membership).
        # Preserve every actual QuerySpec while keeping its session Snapshot fixed.
        key = decision_session, purpose, ref.query_digest
        self._reads.setdefault(key, ref)
        return batch

    def read_decision(self, *, decision_session: str, query: Any, clock: Any) -> Any:
        """Release only PIT-selected decision facts available by `clock`."""
        return self._read(decision_session=decision_session, query=query,
                          clock=clock, purpose="decision_facts")

    def read_market(self, *, decision_session: str, query: Any, clock: Any) -> Any:
        """Release simulation-side market data only after its declared cutoff."""
        return self._read(decision_session=decision_session, query=query,
                          clock=clock, purpose="market_replay")

    @property
    def references(self) -> tuple[BatchReference, ...]:
        """The actual session/purpose -> Snapshot/QuerySpec map for run replay."""
        return tuple(deepcopy(self._reads[k]) for k in sorted(self._reads))


def stale_price_mark(price: float | None, *, price_session: str | None,
                     valuation_session: str, calendar_sessions: tuple[str, ...],
                     source_ref: str | None) -> dict:
    """Annotate an old observed price for display; create no market fact/value."""
    if valuation_session not in calendar_sessions or tuple(sorted(set(calendar_sessions))) != calendar_sessions:
        raise InputGateError("explicit ordered trading calendar required")
    if price is None:
        if price_session is not None:
            raise InputGateError("missing price cannot have an observation session")
        return {"price": None, "price_session": None, "valuation_session": valuation_session,
                "stale_sessions": None, "is_stale": None, "source_ref": None,
                "missing_reason": "NO_OBSERVED_PRICE"}
    if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
        raise InputGateError("positive finite observed price required")
    if price_session not in calendar_sessions or price_session > valuation_session or not source_ref:
        raise InputGateError("price needs a prior calendar session and source ref")
    age = calendar_sessions.index(valuation_session) - calendar_sessions.index(price_session)
    return {"price": float(price), "price_session": price_session,
            "valuation_session": valuation_session, "stale_sessions": age,
            "is_stale": bool(age), "source_ref": source_ref, "missing_reason": None}
