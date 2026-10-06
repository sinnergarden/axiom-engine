"""Synthetic-only comparison with the complete uninstrumented merge13 scanner.

Each scanner/file pair runs in a fresh process. Inputs total at most 32 MiB;
no saved account, native Data input or coverage graph is loaded or decoded.
The legacy module comes from an explicit local git ref, never a network fetch.
"""
import argparse
from datetime import date, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
BASE = "13d280152143ce15a994b717d659235c7931890d"
LIMIT = 32 * 1024**2


def encoded(value):
    # Independent fixture producer, not Engine's canonical encoder.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def scalars(value):
    if type(value) is dict:
        parts = [scalars(k) for k in value] + [scalars(v) for v in value.values()]
    elif type(value) is list:
        parts = [scalars(v) for v in value]
    else:
        return 1, len(encoded(value))
    return sum(x[0] for x in parts), sum(x[1] for x in parts)


def write_fixture(folder, name, chunks, count, scalar_bytes, rows=0, selected_days=()):
    path = folder / (name + ".json")
    hashed = hashlib.sha256()
    size = 0
    with path.open("wb") as stream:
        for chunk in chunks:
            stream.write(chunk); hashed.update(chunk); size += len(chunk)
        stream.write(b"\n")
    file_hash = hashed.copy(); file_hash.update(b"\n")
    return {"name": name, "path": str(path), "file_bytes": size + 1,
            "content_digest": "sha256:" + hashed.hexdigest(),
            "file_digest": "sha256:" + file_hash.hexdigest(),
            "scalar_count": count, "scalar_bytes": scalar_bytes,
            "row_count": rows, "selected_days": list(selected_days)}


def fixtures(folder):
    folder.mkdir(parents=True, exist_ok=True)
    sample = {"a": "x"*384 + '汉字\\"', "b": -123.75, "c": 1e-05,
              "d": [0, True, None], "e": {"tag": "CSI300"}}
    piece = encoded(sample); count, size = scalars(sample)
    copies = 12000
    def coverage():
        yield b'{"context":{"coverage":{"items":['
        for i in range(copies):
            if i: yield b","
            yield piece
        yield b']}},"records":[]}'
    wrapper = ["context", "coverage", "items", "records"]
    cases = [write_fixture(folder, "coverage_mixed", coverage(),
        copies*count + len(wrapper), copies*size + sum(len(encoded(k)) for k in wrapper))]

    days = [(date(2024, 1, 2)+timedelta(days=i)).isoformat() for i in range(20)]
    row_count = 16000
    total_count = 1; total_size = len(encoded("records"))
    def row_at(i):
        return {"code": "SYNTHETIC", "price": 12.5, "quantity": i,
                "session": days[i % len(days)], "value": '汉字\\"' + "y"*32}
    def row_chunks():
        nonlocal total_count, total_size
        yield b'{"records":['
        for i in range(row_count):
            if i: yield b","
            row = row_at(i)
            n, s = scalars(row); total_count += n; total_size += s
            yield encoded(row)
        yield b"]}"
    rows = write_fixture(folder, "grouped_rows", row_chunks(), 0, 0,
                         rows=row_count, selected_days=days[:2])
    rows["scalar_count"] = total_count; rows["scalar_bytes"] = total_size
    selected = [row_at(i) for day in range(2) for i in range(day,row_count,len(days))]
    rows["selected_rows_digest"] = "sha256:" + hashlib.sha256(encoded(selected)).hexdigest()
    cases.append(rows)

    # Chunk-spanning scalar is reported separately from the mixed-token case.
    value = {"a": "z"*(1024**2 + 17) + '汉字\\"'}
    n, s = scalars(value)
    cases.append(write_fixture(folder, "long_scalar", (encoded(value),), n, s))
    if sum(c["file_bytes"] for c in cases) > LIMIT:
        raise RuntimeError("Synthetic physical input cap exceeded")
    return cases


def load_module(legacy):
    sys.path.insert(0, str(ROOT / "src"))
    if legacy is None:
        from axiom_engine.runtime import stock_stream_inputs
        return stock_stream_inputs
    name = "axiom_engine.runtime._benchmark_legacy_scanner"
    spec = importlib.util.spec_from_file_location(name, legacy)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(args):
    case = json.loads(Path(args.case).read_text())
    module = load_module(args.legacy)
    budget = {"max_read_bytes": 65536, "max_decoded_bytes": 32*1024**2}
    memory = module._DecodedMemory(budget["max_decoded_bytes"])
    artifact = {"artifact_type": "synthetic_scanner", "artifact_id": case["name"],
                "contract_version": "synthetic_v1", "manifest_uri": case["path"],
                "content_digest": case["content_digest"]}
    started = perf_counter()
    index = module._CanonicalIndex(artifact, budget, memory=memory)
    scan_wall = perf_counter() - started
    assert index.content_digest == case["content_digest"]
    assert index.file_digest == case["file_digest"]
    signature = {"content_digest": index.content_digest, "file_digest": index.file_digest,
        "spans": sorted((list(k), list(v)) for k,v in index.spans.items()),
        "groups": sorted((list(k[0]), k[1], list(v)) for k,v in index.groups.items()),
        "group_hashes": sorted((list(k[0]), k[1], v) for k,v in index.group_hashes.items()),
        "array_counts": sorted((list(k), v) for k,v in index._array_counts.items())}
    if case["row_count"]:
        with memory.stage() as stage:
            rows, _, bindings = index.rows(("records",), case["selected_days"], reservation=stage)
            assert len(rows) == case["row_count"] // 10
            signature["selected_rows"] = "sha256:"+hashlib.sha256(encoded(rows)).hexdigest()
            assert signature["selected_rows"] == case["selected_rows_digest"]
            signature["bindings"] = bindings
    stats = getattr(index, "statistics", None)
    if stats is not None:
        assert stats["scalar_count"] == case["scalar_count"]
        assert stats["scalar_bytes"] == case["scalar_bytes"]
        assert stats["scan_read_bytes"] == case["file_bytes"]
        assert stats["row_decode_count"] == case["row_count"]
    assert memory.temporary_bytes == 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. This is process peak, not polling.
    if sys.platform != "darwin": rss *= 1024
    print(json.dumps({"file_bytes": case["file_bytes"], "scan_wall_seconds": scan_wall,
        "statistics": stats, "decoded_accounting_peak_bytes": getattr(memory,"peak_bytes",None),
        "native_peak_rss_bytes": rss,
        "expected_scalar_count": case["scalar_count"], "expected_scalar_bytes": case["scalar_bytes"],
        "signature_digest": "sha256:"+hashlib.sha256(encoded(signature)).hexdigest()}))


def main(args):
    folder = Path(args.output).resolve(); cases = fixtures(folder)
    source = subprocess.check_output(["git", "show", args.baseline +
        ":src/axiom_engine/runtime/stock_stream_inputs.py"], cwd=ROOT)
    legacy = folder / "legacy_scanner.py"; legacy.write_bytes(source)
    report = {"baseline": args.baseline, "baseline_source_sha256": hashlib.sha256(source).hexdigest(),
        "total_synthetic_input_bytes": sum(c["file_bytes"] for c in cases),
        "max_synthetic_input_bytes": LIMIT, "python": sys.version, "cases": []}
    for case in cases:
        path = folder / (case["name"] + ".case.json")
        path.write_text(json.dumps(case,sort_keys=True))
        result = {"name": case["name"]}
        for mode in ("legacy", "bulk"):
            command = [sys.executable, str(Path(__file__).resolve()), "--case", str(path)]
            if mode == "legacy": command += ["--legacy", str(legacy)]
            result[mode] = json.loads(subprocess.check_output(command, cwd=ROOT, timeout=60))
        assert result["legacy"]["signature_digest"] == result["bulk"]["signature_digest"]
        result["speedup"] = result["legacy"]["scan_wall_seconds"] / result["bulk"]["scan_wall_seconds"]
        report["cases"].append(result)
        print(json.dumps({"name":case["name"],"bytes":case["file_bytes"],"speedup":result["speedup"]}),flush=True)
    report["note"] = ("Single synthetic measurement per fresh process; legacy has no instrumentation. "
        "Scan wall includes index construction. New scan_seconds nests read/hash/scalar/row decode times. "
        "RSS includes interpreter, indexes, selected row views and signatures. No real-input acceptance.")
    (folder / "report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / ".artifacts/scanner-synthetic-benchmark-20261006"))
    parser.add_argument("--baseline", default=BASE)
    parser.add_argument("--case")
    parser.add_argument("--legacy")
    args = parser.parse_args()
    worker(args) if args.case else main(args)
