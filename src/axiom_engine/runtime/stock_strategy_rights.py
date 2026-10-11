"""Capacity reservations for already registered future stock entitlements."""
from ..core.portfolio import decimal


def registered_share_reservations(ledger, actions):
    reserved = set()
    for action in actions:
        state = ledger.equity_entitlements.get(action["economic_event_id"])
        if state is None or not state["record_quantity"] or state["ex_applied"]:
            continue
        total = action["stock_distribution_shares_per_share"]
        if total is None or decimal(total) > 0:
            # A reservation is a slot, never a fabricated share or NAV asset.
            # At EX the original allocated pending quantity replaces this flag.
            reserved.add(action["security_id"])
    return sorted(reserved)
