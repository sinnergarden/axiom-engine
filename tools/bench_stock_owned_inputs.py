"""Synthetic cold-import and five-account exact benchmark; no real Data/ML.

Run: PYTHONPATH=src:tests python tools/bench_stock_owned_inputs.py
Stdout contains only counters/timings, never source locations or run payloads.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.runtime import (BacktestRequest, StockInputSource, StockResultSink,
    admit_stock_inputs, run_stock_backtest, stock_run_id, save_backtest_run,
    load_stock_backtest_projection)
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_stock_owned_inputs import request_for
from test_stock_stream_inputs import LIMITS, rebind_native


def execute(root, manifest, source, limits):
    result = run_stock_backtest(BacktestRequest.from_dict(manifest), source=source,
        sink=StockResultSink(root/"parts", run_id=stock_run_id(manifest)), block_sessions=2, limits=limits)
    save_backtest_run(result, root/"run.json")
    projection = load_stock_backtest_projection(root/"run.json",
        artifact_reader=lambda ref: Path(ref["manifest_uri"]), limits=limits)
    return result.to_dict(), projection.rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--coverage-rows", type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.coverage_rows <= 100000:
        parser.error("coverage-rows must be within 1..100000")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); manifest = request_for(root)
        row = {"session": "2024-01-02", "security_id": "synthetic-stock", "status": "complete",
               "source_ref": "sha256:"+"a"*64, "usable_from": "2024-01-02T20:30:00+08:00"}
        rebind_native(manifest, "native-4", lambda wire: wire["context"].setdefault("coverage", {}).update(
            synthetic_repeated_facts=[row]*args.coverage_rows))
        limits = {**LIMITS, "max_input_bytes": 32_000_000}
        measurements = {}; inputs = None
        for name, cache in (("uncached", 0), ("cached", 262144)):
            started = perf_counter()
            handle = admit_stock_inputs(BacktestRequest.from_dict(manifest),
                source=StockInputSource(scalar_cache_bytes=cache), block_sessions=2,
                limits=limits, max_owned_bytes=2_000_000)
            measurements[name] = {"cold_import_seconds": perf_counter()-started, **handle.statistics}
            if cache:
                inputs = handle
            else:
                handle.close()
        oracle_seconds = []; owned_seconds = []; oracle_counters = []
        with inputs:
            for k in range(1, 6):
                candidate = deepcopy(manifest); candidate["account_id"] += "-top"+str(k)
                candidate["portfolio_policy"]["top_k"] = k
                candidate["request_ref"] = logical_ref(candidate, "request_ref")
                output = root/("top"+str(k)); source = StockInputSource(scalar_cache_bytes=0)
                started = perf_counter(); oracle = execute(output, candidate, source, limits)
                oracle_seconds.append(perf_counter()-started)
                oracle_counters.append(source.statistics["source_operations"])
                shutil.rmtree(output)
                started = perf_counter(); actual = execute(output, candidate, inputs, limits)
                owned_seconds.append(perf_counter()-started)
                assert actual == oracle, "Complete saved run/ledger differs from original source oracle"
            final = inputs.statistics
        print(json.dumps({"synthetic": True, "implementation_ref": IMPLEMENTATION_REF, "coverage_rows": args.coverage_rows,
            "source_bytes": measurements["cached"]["source_operations"]["file_bytes"],
            "cold_imports": measurements, "original_five_accounts_seconds": oracle_seconds,
            "owned_five_accounts_seconds": owned_seconds,
            "original_five_account_source_scan_bytes": sum(item["scan_read_bytes"] for item in oracle_counters),
            "owned_final_counters": final, "exact_complete_run_and_ledger": "PASS"}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
