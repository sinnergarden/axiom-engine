"""Narrow native SSE saved-input admission; current observations stay retrospective."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from ..core.contracts import Document, digest, fields, integer, require, session, text, _pairs
from ..core.portfolio import decimal


def _instant(value):
    from ..core.contracts import ContractError
    try: result=datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc: raise ContractError("invalid native observation time") from exc
    require(result.utcoffset() is not None,"native observation time needs a timezone")
    return result


def retrospective_sse_benchmark(*, native_batch_text, receipt, run_key):
    """Admit a frozen original Data file and complete owner receipt, without I/O."""
    from .evaluation import BenchmarkSeries
    text(native_batch_text);text(run_key)
    require(type(receipt) is dict and run_key in receipt["runs"],"fixed receipt account entry required")
    binding=receipt["runs"][run_key];ref=binding["native_batch_ref"]
    raw=native_batch_text.encode("utf-8")
    digest("sha256:"+ref["sha256"]);integer(ref["bytes"],1)
    require(len(raw)==ref["bytes"] and sha256(raw).hexdigest()==ref["sha256"],"native SSE file byte ref differs")
    batch=json.loads(native_batch_text,object_pairs_hook=_pairs)
    fields(batch,"records field_meta context")
    context=batch["context"];query=context["query"]
    require(context["contract_version"]=="data_batch_v1" and context["domain"]=="benchmark_daily" and
        context["snapshot_id"]==receipt["snapshot_id"] and receipt["identity"]=="000001.SH" and
        receipt["unit"]=="index points" and receipt["series_kind"]=="price_index" and
        query["fields"]==["close"] and query["symbols"]==["000001.SH"] and
        query["purpose"]=="historical_exploration" and query["pit_policy"]=="operational_pit_v1",
        "unsupported retrospective native SSE scope")
    for key in ("contract_id","source_profile_id","reader_version"):text(context[key])
    days=query["sessions"]
    require(days and days==sorted(set(days)) and set(query["cutoff_by_session"])==set(days),"native SSE session query differs")
    for day in days:session(day)
    cutoff=_instant(receipt["cutoff"])
    require(_instant(receipt["source_receipt"])<=cutoff and
        all(_instant(query["cutoff_by_session"][d])==cutoff for d in days),"native SSE observation cutoff differs")
    fields(binding["source_run_ref"],"run_id content_digest committed_sequence")
    for key in ("run_id","content_digest"):digest(binding["source_run_ref"][key])
    integer(binding["source_run_ref"]["committed_sequence"])
    for key in ("comparison_anchor_session","start_session","end_session"):session(binding[key])
    require(binding["comparison_anchor_session"] < binding["start_session"] <= binding["end_session"] and
        days[0]==binding["comparison_anchor_session"] and days[-1]==binding["end_session"] and
        binding["sessions"]==len(days),"native SSE receipt date range differs")
    require(batch["field_meta"]["close"]["dtype"]=="float64" and
        batch["field_meta"]["close"]["unit"]=="index points","native SSE close unit differs")
    meta={}
    for item in batch["field_meta"]["close"]["by_key"]:
        key=(item["security_id"],item["session"])
        require(key not in meta,"duplicate native SSE metadata");meta[key]=item
    rows={}
    for item in batch["records"]:
        fields(item,"security_id session close")
        key=(item["security_id"],item["session"])
        require(key not in rows,"duplicate native SSE row");rows[key]=item
    keys={("000001.SH",day) for day in days}
    require(set(rows)==set(meta)==keys,"native SSE row/metadata/query keys differ")
    receipt_ref=Document.from_dict(receipt).identity
    batch_ref=Document.from_dict(batch).identity
    normalized=[]
    for day in days:
        item,proof=rows[("000001.SH",day)],meta[("000001.SH",day)]
        value=None if item["close"] is None else str(item["close"])
        if value is not None:
            require(decimal(value)>0 and proof["missing_reason"] is None and proof["usable_from"] is not None and
                _instant(proof["usable_from"])<=cutoff,"native SSE close lacks visible source evidence")
            for key in ("raw_batch_id","revision_id"):text(proof[key])
        else:
            text(proof["missing_reason"])
        for key in ("usable_from","first_observed_at"):
            if proof.get(key) is not None:require(_instant(proof[key])<=cutoff,"native SSE evidence exceeds current cutoff")
        normalized.append(dict(security_id="000001.SH",session=day,close=value,available_at=proof["usable_from"],
            missing_reason=proof["missing_reason"],source_refs=[batch_ref,"sha256:"+ref["sha256"],receipt_ref],
            close_provenance=proof))
    return BenchmarkSeries.from_dict(dict(contract_version="retrospective_benchmark_v1",security_id="000001.SH",
        currency="CNY",timezone="Asia/Shanghai",series_kind="price_index",calendar=days,rows=normalized,
        snapshot_id=context["snapshot_id"],knowledge_cutoff=receipt["cutoff"],pit_policy=query["pit_policy"],
        purpose=query["purpose"],native_batch_text=native_batch_text,native_file_ref={"sha256":ref["sha256"],"bytes":ref["bytes"]},
        native_batch_ref=batch_ref,receipt=receipt,receipt_ref=receipt_ref,run_key=run_key,
        source_run_ref=binding["source_run_ref"],comparison_anchor_session=binding["comparison_anchor_session"],
        start_session=binding["start_session"],end_session=binding["end_session"],
        limitations=["Current observation for retrospective price-index comparison; not historical strategy visibility.",
            "Independent benchmark Snapshot and original native dates/provenance retained; no total-return or FX claim."]))


def read_sse_benchmark(native_path, *, receipt, run_key):
    return retrospective_sse_benchmark(native_batch_text=Path(native_path).read_bytes().decode("utf-8"),
        receipt=receipt,run_key=run_key)


def validate_sse_for_base(wire, base):
    require(type(wire) is dict and wire.get("contract_version")=="retrospective_benchmark_v1" and
        all(key in wire for key in ("native_batch_text","receipt","run_key")),"unsupported saved retrospective benchmark")
    require(wire==retrospective_sse_benchmark(native_batch_text=wire["native_batch_text"],receipt=wire["receipt"],
        run_key=wire["run_key"]).to_dict(),"saved retrospective SSE native identity closure differs")
    window=base["period_metrics"]["window"]
    require(wire["source_run_ref"]==base["input_run_ref"] and wire["comparison_anchor_session"]==window["anchor_session"] and
        wire["start_session"]==base["series"][0]["session"] and wire["end_session"]==base["series"][-1]["session"] and
        wire["calendar"]==[window["anchor_session"],*[p["session"] for p in base["series"]]],
        "retrospective SSE receipt belongs to a different saved account/date range")
