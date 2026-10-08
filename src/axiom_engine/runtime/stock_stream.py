"""Bounded input and output adapters around the single existing account loop."""
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import Document, fields, integer, require
from .backtest import BacktestRequest, BacktestRun, _run
from .stock_stream_contracts import (AUDIT_VERSION, RESULT_GROUPS, RUN_VERSION,
    STREAM_RUNTIME_VERSION, read_budget, validate_artifact_ref, validate_limits,
    validate_manifest, write_budget)
from .stock_stream_inputs import StockInputSource
from .stock_stream_outputs import StockResultSink, canonical_size


def stock_run_id(manifest, implementation_ref=IMPLEMENTATION_REF):
    return Document.from_dict({"request_ref": manifest["request_ref"],
        "core_version": "axiom.stock_portfolio/3", "runtime_version": STREAM_RUNTIME_VERSION,
        "implementation_ref": implementation_ref}).identity


def _admit(manifest, source, block_sessions, limits):
    require(isinstance(manifest, BacktestRequest), "BacktestRequest required")
    require(isinstance(source, StockInputSource), "StockInputSource required")
    integer(block_sessions, 1)
    limits = validate_limits(limits)
    require(len(manifest.payload) <= limits["max_block_bytes"], "stock request manifest exceeds decoded budget")
    encoded_size = 0
    for offset in range(0, len(manifest.payload), 512):
        encoded_size += len(manifest.payload[offset:offset + 512].encode("utf-8"))
        require(encoded_size <= limits["max_block_bytes"], "stock request manifest exceeds decoded budget")
    plan = validate_manifest(manifest)
    audit = source.audit(plan, block_sessions=block_sessions, read_budget=read_budget(limits),
                         limits=limits, implementation_ref=IMPLEMENTATION_REF)
    receipt = audit.receipt
    fields(receipt, "contract_version request_ref market_ref prediction_ref profile_ref implementation_ref counts limitations")
    require(receipt["contract_version"] == AUDIT_VERSION and
            receipt["request_ref"] == plan["request_ref"] and
            receipt["market_ref"] == plan["market_input"]["market_ref"] and
            receipt["prediction_ref"] == plan["prediction_input"]["prediction_ref"] and
            receipt["profile_ref"] == plan["profile_input"]["profile_ref"] and
            receipt["implementation_ref"] == IMPLEMENTATION_REF,
            "stock source audit binding mismatch")
    return plan, audit, limits


def audit_stock_backtest_source(manifest, *, source, block_sessions: int, limits: dict) -> dict:
    """Complete this call's fixed source audit without creating an account."""
    require(isinstance(source, StockInputSource), "StockInputSource required")
    with source.execution_scope():
        _, audit, _ = _admit(manifest, source, block_sessions, limits)
        return audit.receipt


class _BlockCursor:
    def __init__(self, blocks, calendar, universe):
        self._blocks = iter(blocks)
        self._calendar = tuple(calendar)
        self._universe = tuple(universe)
        self._offset = 0
        self._block = None
        self._day = None

    def close(self):
        self._block = None
        close = getattr(self._blocks, "close", None)
        if close is not None:
            close()

    def day(self, day):
        if self._day == day:
            return self._block
        require(self._day is None or day > self._day, "stock block cursor cannot go backwards")
        if self._block is None or day not in self._block.sessions:
            # Release the old expanded graph before advancing the source.
            self._block = None
            try:
                block = next(self._blocks)
            except StopIteration:
                require(False, "stock source ended before frozen calendar")
            sessions = tuple(block.sessions)
            require(sessions and sessions == self._calendar[self._offset:self._offset + len(sessions)],
                    "stock source block calendar differs from admission")
            require(set(block.market_rows) == {(d, s) for d in sessions for s in self._universe},
                    "stock source block market coverage differs from admission")
            self._offset += len(sessions)
            self._block = block
        require(day in self._block.sessions, "stock source omitted Runtime session")
        self._day = day
        return self._block

    def finish(self):
        self._block = None
        require(self._offset == len(self._calendar), "stock source omitted frozen calendar tail")
        try:
            next(self._blocks)
        except StopIteration:
            return
        require(False, "stock source extends frozen calendar")


class _Rows:
    def __init__(self, cursor):
        self.cursor = cursor

    def __getitem__(self, key):
        return self.cursor.day(key[0]).market_rows[key]


class _Signals:
    def __init__(self, cursor):
        self.cursor = cursor

    def __getitem__(self, day):
        require(day in self.cursor.day(day).signals, "missing admitted stock prediction date")
        return self.cursor.day(day).signals[day]


class _StreamStorage:
    """Cumulative facts survive draining; receipts acknowledge exact prefixes."""
    def __init__(self, manifest, audit, sink, limits):
        self.manifest, self.audit, self.sink = manifest, audit, sink
        self.run_id = stock_run_id(manifest)
        self.budget = write_budget(limits)
        self.header_budget = limits["max_block_bytes"]
        self.parts = []
        self._part_bytes = 0
        self.groups = None
        self._emitted = {kind: 0 for kind in RESULT_GROUPS[:-1]}
        self._drained = dict(self._emitted)
        self.pending_bytes = 0
        self.sink._external_pending_bytes = lambda: self.pending_bytes
        self.metrics = {"total_fees_minor": 0, "turnover_minor": 0, "fill_count": 0,
            "unfilled_order_count": 0, "unsubmitted_order_count": 0,
            "unsubmitted_quantity": 0, "incomplete_order_count": 0}
        self.peak, self.drawdown, self.last_nav = None, Decimal(0), None
        market = audit.globals["market_header"]
        self.account_events = {"contract_version": "stock_account_events_v1",
            "request_ref": manifest["request_ref"], "market_ref": manifest["market_input"]["market_ref"],
            "profile_ref": manifest["profile_input"]["profile_ref"], "cash_dividends": [],
            "source_refs": list(dict.fromkeys(market["source_refs"][4:6])),
            "limitations": ["Observed implemented stock cash actions only; absent actions do not establish completeness.",
                            "Unknown PAY remains pending; gross dividends exclude personal holding-period taxes."]}
        event_bytes = canonical_size(self.account_events, self.header_budget)
        for action in market["cash_dividends"]:
            size = canonical_size(action, self.header_budget)
            growth = size + bool(self.account_events["cash_dividends"])
            require(event_bytes + growth <= self.header_budget,
                    "stock account event view exceeds budget before addition")
            self.account_events["cash_dividends"].append(action)
            event_bytes += growth

    def attach(self, ledger, nav, positions, orders, decisions):
        self.ledger = ledger
        self.groups = {"nav": nav, "positions": positions, "orders": orders, "decisions": decisions,
            "fills": ledger.fills, "cash_ledger": ledger.cash_ledger, "position_ledger": ledger.position_ledger}

    def reserve_output(self, kind, row):
        require(kind in RESULT_GROUPS[:-1], "unknown pending stock output group")
        scratch = max(8, min(16384, self.budget["max_buffer_bytes"] // 8))
        size = canonical_size(row, self.budget["max_buffer_bytes"], scratch_bytes=scratch)
        require(self.pending_bytes + self.sink.buffered_bytes + scratch + size <= self.budget["max_buffer_bytes"],
                "stock pending output budget exceeded before history addition")
        self.pending_bytes += size

    def observe_order(self, order):
        self.metrics["unfilled_order_count"] += order["unfilled_quantity"] > 0
        self.metrics["unsubmitted_order_count"] += order["unsubmitted_quantity"] > 0
        self.metrics["unsubmitted_quantity"] += order["unsubmitted_quantity"]
        self.metrics["incomplete_order_count"] += order["unsubmitted_quantity"] + order["unfilled_quantity"] > 0

    def observe_nav(self, point, initial_value):
        self.peak = max(initial_value if self.peak is None else self.peak, point["nav_minor"])
        self.drawdown = min(self.drawdown, Decimal(point["nav_minor"]) / self.peak - 1)
        self.last_nav = point["nav_minor"]

    def _rows(self):
        for kind in RESULT_GROUPS[:-1]:
            rows = self.groups[kind]
            while self._emitted[kind] - self._drained[kind] < len(rows):
                index = self._emitted[kind] - self._drained[kind]
                self._emitted[kind] += 1
                yield kind, rows[index]

    def _ack(self, receipt, day, sequence):
        fields(receipt, "artifact part_index start_session end_session first_committed_sequence last_committed_sequence previous_part_digest row_counts")
        validate_artifact_ref(receipt["artifact"])
        integer(receipt["part_index"])
        integer(receipt["first_committed_sequence"])
        integer(receipt["last_committed_sequence"])
        require(receipt["part_index"] == len(self.parts) and
                receipt["start_session"] == receipt["end_session"] == day and
                receipt["first_committed_sequence"] == receipt["last_committed_sequence"] == sequence and
                receipt["previous_part_digest"] == (None if not self.parts else self.parts[-1]["artifact"]["content_digest"]),
                "result part receipt binding mismatch")
        fields(receipt["row_counts"], " ".join(RESULT_GROUPS))
        counts = receipt["row_counts"]
        for kind in RESULT_GROUPS:
            integer(counts[kind])
            if kind != "session_phases":
                require(counts[kind] <= self._emitted[kind] - self._drained[kind],
                        "result receipt acknowledges an unsubmitted row")
        growth = canonical_size(receipt, self.header_budget) + bool(self.parts)
        require(self._part_bytes + growth <= self.header_budget,
                "stock result reference table exceeds header budget before addition")
        scratch = max(8, min(16384, self.budget["max_buffer_bytes"] // 8))
        released_bytes = sum(canonical_size(row, self.budget["max_buffer_bytes"], scratch_bytes=scratch)
            for kind in RESULT_GROUPS[:-1] for row in (self.groups[kind][i] for i in range(counts[kind])))
        # Only a verified immutable receipt permits draining the corresponding
        # group prefix. Idempotency, T+1, holdings and receivables remain live.
        self.ledger.drain_outputs({kind: counts[kind] for kind in ("fills", "cash_ledger", "position_ledger")})
        for kind in ("nav", "positions", "orders", "decisions"):
            del self.groups[kind][:counts[kind]]
        for kind in self._drained:
            self._drained[kind] += counts[kind]
        self.parts.append(receipt)
        self._part_bytes += growth
        self.pending_bytes -= released_bytes
        require(self.pending_bytes >= 0, "stock output byte cursor underflow")

    def flush(self, day, phase, sequence):
        # Each fill is new since the preceding complete session flush.
        for fill in self.groups["fills"]:
            self.metrics["total_fees_minor"] += fill["fee_minor"]
            self.metrics["turnover_minor"] += fill["gross_minor"]
            self.metrics["fill_count"] += 1
        fill = None
        for receipt in self.sink.append(session=day, phase=phase, rows=self._rows(),
                                        committed_sequence=sequence, write_budget=self.budget):
            self._ack(receipt, day, sequence)
        require(all(not rows for rows in self.groups.values()), "result sink did not commit all session rows")
        require(self.pending_bytes == 0, "stock output budget did not drain with committed prefixes")

    def result(self, *, ledger, initial_value, stopped, lifecycle, limitations):
        require(self.sink.finish(write_budget=self.budget) == [], "result sink left unacknowledged parts")
        metrics = dict(self.metrics)
        metrics.update(total_return=None if stopped else str(Decimal(self.last_nav) / initial_value - 1),
                       max_drawdown=None if stopped else str(self.drawdown))
        result = {"contract_version": RUN_VERSION, "run_id": self.run_id,
            "account_id": self.manifest["account_id"], "status": "BLOCKED" if stopped else "COMPLETE",
            "request_manifest": self.manifest, "request_ref": self.manifest["request_ref"],
            "source_audit": self.audit.receipt, "source_audit_ref": Document.from_dict(self.audit.receipt).identity,
            "signal_ref": self.manifest["prediction_input"]["prediction_ref"],
            "market_ref": self.manifest["market_input"]["market_ref"],
            "profile_ref": self.manifest["profile_input"]["profile_ref"],
            "core_version": "axiom.stock_portfolio/3", "runtime_version": STREAM_RUNTIME_VERSION,
            "implementation_ref": IMPLEMENTATION_REF, "committed_sequence": ledger.sequence,
            "initial_nav_minor": initial_value, "final_account": {"cash_minor": ledger.cash,
                "receivable_minor": sum(ledger.receivables.values()), "positions": ledger.positions,
                "committed_sequence": ledger.sequence}, "stopped": stopped,
            "lifecycle_admission": lifecycle, "metrics": metrics,
            "limitations": limitations, "result_parts": self.parts,
            "account_events": self.account_events,
            "account_events_ref": Document.from_dict(self.account_events).identity}
        # The complete header is small by contract. Measure its canonical
        # stream before Document creates a full JSON string or decoded copy.
        header_bytes = canonical_size({**result, "content_digest": "sha256:" + "0" * 64}, self.header_budget)
        require(header_bytes + 1 <= self.header_budget, "saved run header exceeds decoded budget")
        # The saved projection needs the existing small profile once, the
        # canonical parts, and the public saver's final run header plus LF.
        # Reject before hashing or returning a sealed BacktestRun.
        require(self.sink.total_bytes + header_bytes + 1 + self.audit.globals["profile_bytes"] <=
                self.budget["max_total_bytes"], "total saved result budget exceeded before run seal")
        result["content_digest"] = Document.from_dict(result).identity
        return BacktestRun.from_dict(result)


def run_stock_backtest(manifest: BacktestRequest, *, source: StockInputSource,
                       sink: StockResultSink, block_sessions: int, limits: dict) -> BacktestRun:
    """Audit fixed input fully, then use the existing Core/SimBroker/ledger loop."""
    require(isinstance(source, StockInputSource), "StockInputSource required")
    with source.execution_scope():
        return _run_stock_backtest(manifest, source=source, sink=sink,
                                   block_sessions=block_sessions, limits=limits)


def _run_stock_backtest(manifest, *, source, sink, block_sessions, limits):
    plan, audit, limits = _admit(manifest, source, block_sessions, limits)
    require(isinstance(sink, StockResultSink) and sink.run_id == stock_run_id(plan),
            "fresh result sink bound to this stock run required")
    require(sink.total_bytes == 0 and sink.buffered_bytes == 0, "stock result sink already used")
    cursor = _BlockCursor(source.iter_blocks(plan, block_sessions=block_sessions,
                                             read_budget=read_budget(limits)),
                          audit.globals["calendar"], plan["scope"]["execution_universe"])
    execution_plan = {**plan, **{name: plan["scope"][name] for name in
        ("start_session", "end_session", "execution_universe", "supported_universe_ref")},
        "stock_execution_rules_ref": plan["profile_input"]["stock_execution_rules_ref"]}
    signal = {"schedule_ref": plan["prediction_input"]["prediction_ref"],
              "limitations": audit.globals.get("signal_limitations", [])}
    admitted = (execution_plan, signal, _Signals(cursor), audit.globals["market_header"],
                audit.globals["calendar"], _Rows(cursor), audit.globals["profile"])
    storage = _StreamStorage(plan, audit, sink, limits)
    try:
        with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
            result = _run(manifest, admitted=admitted, storage=storage)
        # A blocked account may stop before the source tail, already audited first.
        if result.to_dict()["status"] == "COMPLETE":
            cursor.finish()
        return result
    finally:
        cursor.close()
