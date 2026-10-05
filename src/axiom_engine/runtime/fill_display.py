"""Independent retrospective coordinates from fixed Data files and saved fills."""
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from hashlib import sha256
import json
from pathlib import Path

from .._implementation import IMPLEMENTATION_REF
from ..core.contracts import Document, digest, fields, integer, require, session, _pairs
from ..core.portfolio import decimal

DISPLAY_VERSION = "axiom.fill_display/1"
_SEAL = object()


class SavedReviewDisplay:
    """A loader-established byte binding; direct construction is unsupported."""
    __slots__ = ("_manifest_text", "_content", "_display_ref", "_seal")

    def __init__(self, *args, **kwargs):
        require(False, "use read_review_display to establish saved byte bindings")

    @property
    def identity(self):
        return self._display_ref


class FillDisplayReport(Document):
    """A saved display result, never an account or execution input."""


def _file_digest(wire):
    # Data review_display_v1's precise JSON encoding; stream to avoid a large copy.
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    value, size = sha256(), 0
    for part in encoder.iterencode(wire):
        data = part.encode("utf-8"); value.update(data); size += len(data)
    value.update(b"\n")
    return value.hexdigest(), size + 1


def read_review_display(directory, *, manifest_sha256):
    """Delegate byte checks to Data's public saved loader; no Reader/transform."""
    digest("sha256:" + manifest_sha256)
    from axiom_data import load_review_display
    directory = Path(directory)
    loaded = load_review_display(directory, manifest_sha256=manifest_sha256)
    raw = (directory / "manifest.json").read_bytes()
    require(sha256(raw).hexdigest() == manifest_sha256, "display manifest changed during read")
    manifest_text = raw.decode("utf-8")
    require(json.loads(manifest_text, object_pairs_hook=_pairs) == loaded["manifest"],
            "Data loader manifest binding differs")
    display = object.__new__(SavedReviewDisplay)
    display._manifest_text, display._content = manifest_text, loaded
    display._display_ref, display._seal = "sha256:" + manifest_sha256, _SEAL
    return display


def _manifest(text, reference):
    digest(reference)
    raw = text.encode("utf-8")
    require(reference == "sha256:" + sha256(raw).hexdigest(), "display manifest byte identity differs")
    value = json.loads(text, object_pairs_hook=_pairs)
    require(value.get("contract_version") == value.get("exporter_version") == "review_display_v1",
            "unsupported Data display version")
    files = value.get("files", {})
    require("ohlcv.json" in files and not set(files)-{"ohlcv.json", "events.json", "securities.json"},
            "unsupported Data display files")
    for name, ref in files.items():
        fields(ref, "uri sha256 bytes"); require(ref["uri"] == name, "invalid display filename")
        digest("sha256:" + ref["sha256"]); integer(ref["bytes"], 1)
    context = value["context"]
    require(context["usage"] == "retrospective_review" and context["native_price_basis"] == "unadjusted" and
            context["default_price_basis"] == "common_anchor_adjusted_v1", "unsupported display basis/use")
    session(context["anchor_session"])
    require(datetime.fromisoformat(context["knowledge_cutoff"]).utcoffset() is not None,
            "display cutoff must have a timezone")
    return value


def _verified_display(display):
    require(isinstance(display, SavedReviewDisplay) and getattr(display, "_seal", None) is _SEAL,
            "public saved display loader required")
    manifest = _manifest(display._manifest_text, display.identity)
    loaded = display._content
    require(loaded["manifest"] == manifest, "changed saved display manifest")
    require(set(loaded) == {"manifest", *[name.removesuffix(".json") for name in manifest["files"]]},
            "changed saved display file set")
    for name, ref in manifest["files"].items():
        hashed, size = _file_digest(loaded[name.removesuffix(".json")])
        require((hashed, size) == (ref["sha256"], ref["bytes"]), "changed saved display payload")
    ohlcv = loaded["ohlcv"]
    require(ohlcv["contract_version"] == "review_display_v1" and ohlcv["context"] == manifest["context"],
            "display context closure differs")
    return manifest, loaded


def _units(saved):
    profile = saved["plan"]["profile"]
    if saved["contract_version"] == "backtest_run_v3":
        require(saved["price_unit"] == "CNY/share" and saved["quantity_unit"] == "shares",
                "stock account unit differs")
        return "CNY/share"
    require(profile["contract_version"] == "daily_open_profile_v1", "ETF quantity contract required")
    return "CNY/fund unit"


def _identity(wire):
    return Document.from_dict({k: wire[k] for k in ("input_run_ref", "display_ref", "consumed_input_ref",
        "display_version", "implementation_ref")}).identity


def _basis_reason(fill, run_events, data_events):
    relevant = [e for e in run_events if e["security_id"] == fill["security_id"] and
                fill["session"] > e["effective_date"]]
    for event in relevant:
        matches = [e for e in data_events if e.get("event_id") == event["event_id"] and
                   e.get("security_id") == event["security_id"]]
        if len(matches) != 1 or not matches[0].get("new_price_basis_session") or \
                matches[0]["new_price_basis_session"] != event.get("new_price_basis_session"):
            return "NEW_PRICE_BASIS_UNVERIFIED"
        if fill["session"] < event["new_price_basis_session"]:
            return "NEW_PRICE_BASIS_UNVERIFIED"
    known = {e["event_id"] for e in relevant}
    for event in data_events:
        if event.get("event_type") == "unit_split" and event.get("security_id") == fill["security_id"] and \
                event.get("effective_date") and fill["session"] > event["effective_date"] and event.get("event_id") not in known:
            return "NEW_PRICE_BASIS_UNVERIFIED"
    return None


def _reason(fill, fact, source_unit, target_unit, account_unit, run_events, data_events):
    reason = None
    if fact["row"] is None: reason = "MISSING_DISPLAY_ROW"
    elif source_unit != account_unit or target_unit != account_unit: reason = "UNIT_MISMATCH"
    elif fact["row"]["native_open"] is None or fill.get("reference_open") is None:
        reason = "NATIVE_OPEN_UNVERIFIED"
    elif decimal(str(fact["row"]["native_open"])) != decimal(fill["reference_open"]):
        reason = "NATIVE_OPEN_MISMATCH"
    elif fact["row"]["display_scale"] is None:
        reason = "MISSING_DISPLAY_SCALE"
    elif fact["scale_meta"] is None or fact["scale_meta"].get("missing_reason") is not None:
        reason = "DISPLAY_SCALE_UNVERIFIED"
    elif not fact["scale_meta"].get("factor_provenance") or not fact["scale_meta"].get("anchor_factor_provenance"):
        reason = "DISPLAY_SCALE_UNVERIFIED"
    else:
        reason = _basis_reason(fill, run_events, data_events)
    return reason


def _coordinate(fill, fact, source_unit, target_unit, account_unit, run_events, data_events):
    reason = _reason(fill, fact, source_unit, target_unit, account_unit, run_events, data_events)
    multiplier, value = None, None
    if reason is None:
        scale = decimal(str(fact["row"]["display_scale"]))
        require(scale > 0, "positive saved display scale required")
        multiplier, value = str(scale), str(decimal(fill["price"]) * scale)
    return dict(fill_id=fill["fill_id"], security_id=fill["security_id"], session=fill["session"],
        status="AVAILABLE" if reason is None else "UNAVAILABLE", reason=reason,
        display_price=value, multiplier=multiplier, source_unit=source_unit, target_unit=target_unit,
        input_fact_ref=Document.from_dict(fact).identity)


def build_fill_display(run, *, display):
    """Multiply actual native fill price by the same-session saved Data scalar."""
    from .evaluation import _verify_run
    with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
        saved = _verify_run(run)
        manifest, loaded = _verified_display(display)
        context, ohlcv = manifest["context"], loaded["ohlcv"]
        market = saved["plan"]["market_replay"]
        snapshots = {e["context"]["snapshot_id"] for e in market["source_evidence"] if "context" in e}
        snapshots.update(e["batch"]["context"]["snapshot_id"] for e in market["source_evidence"] if "batch" in e)
        require(snapshots == {context["snapshot_id"]}, "display belongs to a different fixed Snapshot")
        query = context["derivation"]["price_query"]
        days = query["sessions"]
        require(days and days == sorted(set(days)) and context["anchor_session"] == days[-1] == saved["plan"]["end_session"] and
                query["purpose"] == "historical_exploration" and query["price_basis"] == "unadjusted" and
                all(query["cutoff_by_session"].get(day) == context["knowledge_cutoff"] for day in days),
                "display full range/anchor/common cutoff differs")
        expected_days = [d for d in market["calendar"] if saved["plan"]["start_session"] <= d <= saved["plan"]["end_session"]]
        require(set(expected_days) <= set(days) and set(query["symbols"]) == set(market["universe"]),
                "display omits frozen account range/universe")
        require(all(f["session"] in days and f["security_id"] in query["symbols"] for f in saved["fills"]),
                "display query does not cover saved fills")
        rows, metadata = {}, {}
        for row in ohlcv["records"]:
            key = (row["security_id"], row["session"])
            require(key not in rows, "duplicate Data display row"); rows[key] = row
        meta = ohlcv["field_meta"]
        require(meta["display_scale"]["unit"] == "dimensionless", "display scale unit differs")
        for row in meta["display_scale"]["by_key"]:
            key = (row["security_id"], row["session"])
            require(key not in metadata, "duplicate display scale metadata"); metadata[key] = row
        source_unit, target_unit, account_unit = meta["native_open"]["unit"], meta["open"]["unit"], _units(saved)
        run_events = [item["event"] for item in market.get("unit_splits", [])]
        data_events = loaded.get("events", {}).get("fund_share_conversions", {}).get("records", [])
        keys = sorted({(f["security_id"], f["session"]) for f in saved["fills"]})
        facts = [dict(security_id=sid, session=day, row=None if (sid,day) not in rows else
            {k: rows[(sid,day)][k] for k in ("native_open", "display_scale")},
            scale_meta=metadata.get((sid,day))) for sid,day in keys]
        inputs = dict(facts=facts, source_unit=source_unit, target_unit=target_unit, account_unit=account_unit,
            run_unit_events=run_events, data_unit_events=data_events,
            native_fills_ref=Document.from_dict({"fills":saved["fills"]}).identity)
        by_key = {(f["security_id"], f["session"]): f for f in facts}
        coords = [_coordinate(f, by_key[(f["security_id"],f["session"])], source_unit, target_unit,
            account_unit, run_events, data_events) for f in saved["fills"]]
        wire = dict(contract_version="fill_display_report_v1", display_version=DISPLAY_VERSION,
            implementation_ref=IMPLEMENTATION_REF,
            input_run_ref={k:saved[k] for k in ("run_id", "content_digest", "committed_sequence")},
            display_ref=display.identity, manifest_text=display._manifest_text,
            manifest_file_ref=dict(uri="manifest.json", sha256=display.identity.removeprefix("sha256:"),
                bytes=len(display._manifest_text.encode("utf-8"))),
            consumed_input=inputs, consumed_input_ref=Document.from_dict(inputs).identity,
            status="COMPLETE" if all(c["status"] == "AVAILABLE" for c in coords) else "PARTIAL",
            fills=saved["fills"], coordinates=coords,
            limitations=["Saved retrospective display coordinates; not account, decision or model inputs.",
                "Decimal multiplication consumes the saved float64 scalar; no factor recomputation or inferred unit dates.",
                "Native fill prices, fees and original account identity are unchanged."])
        wire["display_result_ref"] = _identity(wire)
        wire["content_digest"] = Document.from_dict(wire).identity
        return FillDisplayReport.from_dict(wire)


def _verify_report(report):
    require(isinstance(report, FillDisplayReport), "FillDisplayReport required")
    wire = report.to_dict()
    fields(wire, "contract_version display_version implementation_ref input_run_ref display_ref manifest_text manifest_file_ref consumed_input consumed_input_ref status fills coordinates limitations display_result_ref content_digest")
    recorded = wire.pop("content_digest", None)
    require(recorded == Document.from_dict(wire).identity, "fill display content digest differs")
    wire["content_digest"] = recorded
    require(wire["contract_version"] == "fill_display_report_v1" and wire["display_version"] == DISPLAY_VERSION,
            "unsupported fill display version")
    digest(wire["implementation_ref"])
    fields(wire["input_run_ref"], "run_id content_digest committed_sequence")
    for key in ("run_id", "content_digest"): digest(wire["input_run_ref"][key])
    integer(wire["input_run_ref"]["committed_sequence"])
    manifest = _manifest(wire["manifest_text"], wire["display_ref"])
    require(wire["manifest_file_ref"] == dict(uri="manifest.json", sha256=wire["display_ref"][7:],
        bytes=len(wire["manifest_text"].encode("utf-8"))), "manifest file ref differs")
    inputs = wire["consumed_input"]
    fields(inputs, "facts source_unit target_unit account_unit run_unit_events data_unit_events native_fills_ref")
    require(inputs["native_fills_ref"] == Document.from_dict({"fills":wire["fills"]}).identity,
            "original fill extraction identity differs")
    require(wire["consumed_input_ref"] == Document.from_dict(inputs).identity and
            wire["display_result_ref"] == _identity(wire), "fill display identity closure differs")
    facts = {(f["security_id"],f["session"]):f for f in inputs["facts"]}
    require(len(facts) == len(inputs["facts"]) and len(wire["fills"]) == len(wire["coordinates"]),
            "duplicate facts or coordinate count differs")
    require(set(facts) == {(f["security_id"],f["session"]) for f in wire["fills"]}, "consumed fact scope differs")
    query = manifest["context"]["derivation"]["price_query"]
    for fact in inputs["facts"]:
        fields(fact, "security_id session row scale_meta")
        require(fact["security_id"] in query["symbols"] and fact["session"] in query["sessions"], "unbound saved input key")
        if fact["row"] is not None: fields(fact["row"], "native_open display_scale")
        if fact["scale_meta"] is not None:
            require(all(fact["scale_meta"][k] == fact[k] for k in ("security_id", "session")), "scale metadata key differs")
    ids = set()
    reasons = {"MISSING_DISPLAY_ROW", "UNIT_MISMATCH", "NATIVE_OPEN_UNVERIFIED", "NATIVE_OPEN_MISMATCH",
        "MISSING_DISPLAY_SCALE", "DISPLAY_SCALE_UNVERIFIED", "NEW_PRICE_BASIS_UNVERIFIED"}
    for fill, c in zip(wire["fills"], wire["coordinates"]):
        fields(c, "fill_id security_id session status reason display_price multiplier source_unit target_unit input_fact_ref")
        require(fill["fill_id"] not in ids, "duplicate saved fill"); ids.add(fill["fill_id"])
        key = (fill["security_id"],fill["session"]); session(fill["session"])
        require(key in facts and all(c[k] == fill[k] for k in ("fill_id", "security_id", "session")) and
            c["input_fact_ref"] == Document.from_dict(facts[key]).identity and
            c["source_unit"] == inputs["source_unit"] and c["target_unit"] == inputs["target_unit"],
            "saved coordinate/fill/input link differs")
        decimal(fill["price"], minimum=0)
        reason = _reason(fill, facts[key], inputs["source_unit"], inputs["target_unit"], inputs["account_unit"],
            inputs["run_unit_events"], inputs["data_unit_events"])
        require(c["reason"] == reason, "coordinate raw-input eligibility differs")
        if c["status"] == "AVAILABLE":
            require(c["reason"] is None and c["display_price"] is not None and c["multiplier"] is not None,
                    "available coordinate lacks saved values")
            decimal(c["display_price"], minimum=0); require(decimal(c["multiplier"]) > 0 and
                c["multiplier"] == str(decimal(str(facts[key]["row"]["display_scale"]))), "saved scalar multiplier differs")
        else:
            require(c["status"] == "UNAVAILABLE" and c["reason"] in reasons and
                c["display_price"] is None and c["multiplier"] is None, "invalid unavailable coordinate")
    require(wire["status"] == ("COMPLETE" if all(c["status"] == "AVAILABLE" for c in wire["coordinates"]) else "PARTIAL"),
            "fill display status differs")
    return wire


def save_fill_display(report, path):
    _verify_report(report)
    path = Path(path)
    if path.exists():
        require(path.read_text() == report.payload + "\n", "conflicting saved fill display")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: stream.write(report.payload + "\n")


def load_fill_display(path):
    """Validate saved closure/links; never load Data or recompute coordinates."""
    report = FillDisplayReport(Path(path).read_text())
    _verify_report(report)
    return report
