"""Explicit experimental execution profiles; strict status admission is default."""
from ..core.contracts import Document, require

STATUS_GAP_REASONS = frozenset({"status_source_missing", "status_not_visible_at_cutoff", "status_unknown"})


def daily_open_profile(*, unknown_status_policy="block", price_limit_policy="require_both",
                       slippage_bps="0", price_grid_policy="legacy"):
    """One Engine profile shared by callers, never a Research execution rule.

    The ETF observed-daily alternative is an explicit simulation assumption.
    Neither profile claims opening liquidity or verified real ETF settlement.
    """
    require(unknown_status_policy in ("block", "etf_daily_observed"), "unsupported unknown-status policy")
    admission = ("Strict status admission: UNKNOWN blocks fills." if unknown_status_policy == "block" else
                 "EXPLICIT ETF DAILY APPROXIMATION: missing status may execute only from valid observed open, positive volume and legal limits; UNKNOWN remains unchanged. This does not prove opening liquidity or real tradeability.")
    profile = {"contract_version": "daily_open_profile_v1", "lot_size": 100,
        "settlement_sessions": 1, "commission_rate": "0.0003", "minimum_commission_minor": 0,
        "tax_rate": "0", "slippage_bps": "0", "participation_rate": "0.1",
        "decision_time_utc": "00:55:00Z", "execution": "open", "approximation": "daily_volume_proxy",
        "unknown_status_policy": unknown_status_policy,
        "limitation": "Conservative experimental T+1 and 100-fund-unit lot, not verified real ETF settlement rules. " + admission}
    require(price_limit_policy in ("require_both", "known_only") and slippage_bps in ("0", "5"),
            "unsupported ETF limit or slippage policy")
    if price_grid_policy == "legacy":
        require(price_limit_policy == "require_both" and slippage_bps == "0", "explicit ETF grid required for new policies")
        return profile
    require(price_grid_policy == "etf_price_grid_v1", "unsupported ETF price grid policy")
    from .etf_grid import frozen_etf_grid
    grid = frozen_etf_grid()
    profile.update(contract_version="daily_open_profile_v2", slippage_bps=slippage_bps,
        price_limit_policy=price_limit_policy, price_grid_policy=price_grid_policy,
        price_grid_ref=Document.from_dict(grid).identity, price_grid=grid)
    profile["limitation"] += " Frozen 0.001 CNY/fund-unit research grid; adverse tick rounding may exceed declared slippage. " + (
        "Both native limit sides are required." if price_limit_policy == "require_both" else
        "EXPLORATORY known_only: missing native limit sides remain null and are not assumed unlimited.")
    return profile


def stock_daily_open_profile(*, unknown_status_policy="block"):
    """Frozen narrow SZ ordinary-share costs and retrospective execution model."""
    require(unknown_status_policy in ("block", "stock_daily_observed"), "unsupported stock status policy")
    return {"contract_version": "stock_daily_open_profile_v1", "lot_size": 100, "settlement_sessions": 1,
        "commission_rate": "0.0003", "minimum_commission_minor": 500, "sell_stamp_tax_rate": "0.0005",
        "transfer_fee_rate": "0.00001", "slippage_bps": "0", "participation_rate": "0.1",
        "decision_time_utc": "00:55:00Z", "execution": "open", "approximation": "retrospective_daily_volume_proxy",
        "unknown_status_policy": unknown_status_policy, "maximum_order_quantity": 1000000, "price_tick": "0.01",
        "limitation": "EXPLICIT STOCK DAILY APPROXIMATION: native UNKNOWN is retained. Observed open/volume/limits are retrospective evidence, not 09:30 knowledge or opening liquidity. Commission/minimum, transfer fee, slippage, capacity and T+1 are declared model assumptions. Gross dividends exclude personal holding-period taxes."}
