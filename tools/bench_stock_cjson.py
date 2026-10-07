"""Mixed synthetic whole cold-admission benchmark; no real source/ML reads.

PYTHONPATH=src:tests python -B tools/bench_stock_cjson.py --coverage-rows 20000
RSS monitoring must be permitted. Timings include helper launch, full parse and
canonical comparison, validation, private spool and parent owned capture.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from axiom_engine._implementation import IMPLEMENTATION_REF
from axiom_engine.runtime import BacktestRequest, StockInputSource, admit_stock_inputs
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_stock_cjson import mixed_coverage
from test_stock_owned_inputs import request_for, run_owned
from test_stock_stream_inputs import LIMITS, rebind_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage-rows', type=int, default=20000)
    args = parser.parse_args()
    if not 1 <= args.coverage_rows <= 100000:
        parser.error('coverage-rows must be within 1..100000')
    options = dict(file_parse_mode='cjson', max_file_parse_bytes=64*1024**2,
        max_file_parse_rss_bytes=3*1024**3, max_file_parse_spool_bytes=4*1024**2,
        max_file_parse_seconds=30)
    limits = {**LIMITS, 'max_input_bytes': 64*1024**2}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); manifest = request_for(root)
        rebind_native(manifest, 'native-4', lambda w: mixed_coverage(w, args.coverage_rows))
        measurements = {}; handles = {}
        try:
            for mode, source in [('stream', StockInputSource()), ('cjson', StockInputSource(**options))]:
                started = perf_counter()
                handle = admit_stock_inputs(BacktestRequest.from_dict(manifest), source=source,
                    block_sessions=2, limits=limits, max_owned_bytes=2_000_000)
                measurements[mode] = dict(cold_import_seconds=perf_counter()-started,
                    **handle.statistics, parse_events=[{k:v for k,v in e.items() if k != 'path'}
                                                      for e in source._file_parse_events])
                handles[mode] = handle
            assert measurements['cjson']['source_operations']['cjson_files'] == 9, 'Fast helper was not exercised'
            account_seconds = {}; summaries = {}
            for k in (3, 5):
                candidate = deepcopy(manifest); candidate['account_id'] += '-top'+str(k)
                candidate['portfolio_policy']['top_k'] = k
                candidate['request_ref'] = logical_ref(candidate, 'request_ref')
                output = root/('top'+str(k)); expected = None
                for mode in ('stream', 'cjson'):
                    started = perf_counter()
                    wire, projection = run_owned(output, candidate, handles[mode], limits)
                    account_seconds[mode+'-top'+str(k)] = perf_counter()-started
                    actual = wire, projection.rows
                    if expected is None:
                        expected = actual
                    else:
                        assert actual == expected, 'Complete saved header/ledger differs from stream oracle'
                    shutil.rmtree(output)
                summaries[str(k)] = dict(signal_ref=wire['signal_ref'], final_account=wire['final_account'])
            assert summaries['3']['signal_ref'] == summaries['5']['signal_ref']
            assert summaries['3']['final_account'] != summaries['5']['final_account']
            print(json.dumps(dict(synthetic=True, real_inputs_read=False, implementation_ref=IMPLEMENTATION_REF,
                coverage_rows=args.coverage_rows, coverage_shape='repeated strings and keys plus unique ordinal/hash/float/UTF8',
                cold_scope='complete helper/parse/validation/comparison/spool/audit/owned capture',
                options=options, cold_imports=measurements, account_seconds=account_seconds,
                exact_complete_run_and_ledger='PASS', top3_top5_distinct_same_signal='PASS',
                final_counters={mode: handle.statistics for mode, handle in handles.items()}), sort_keys=True, indent=2))
        finally:
            for handle in handles.values():
                handle.close()


if __name__ == '__main__':
    main()
