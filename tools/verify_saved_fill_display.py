"""Read-only acceptance against one saved account and pinned Data display bytes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter
from unittest.mock import patch

from axiom_engine.runtime import (load_backtest_run, read_review_display, build_fill_display,
    save_fill_display, load_fill_display)


def inventory(path):
    hashed=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):hashed.update(block)
    stat=path.stat()
    return dict(sha256=hashed.hexdigest(),mtime_ns=stat.st_mtime_ns,bytes=stat.st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','display-directory','manifest-sha256','output-dir'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run_path,directory,output=Path(args.run),Path(args.display_directory),Path(args.output_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    paths=[run_path,directory/'manifest.json',*[directory/name for name in manifest['files']]]
    before={str(path):inventory(path) for path in paths}
    start=perf_counter();run=load_backtest_run(run_path);run_load_seconds=perf_counter()-start
    run_payload_ref=run.identity
    with ExitStack() as stack:
        for target in ('axiom_engine.runtime.backtest.run_backtest',
                'axiom_engine.runtime.accounting.AccountLedger.apply_fill',
                'axiom_data.api.Data', 'axiom_data.adjust_prices',
                'axiom_data.review_display.project_review_display'):
            stack.enter_context(patch(target,side_effect=AssertionError('forbidden query/transform/account: '+target)))
        start=perf_counter();display=read_review_display(directory,manifest_sha256=args.manifest_sha256)
        display_load_seconds=perf_counter()-start
        start=perf_counter();report=build_fill_display(run,display=display)
        build_seconds=perf_counter()-start
    assert run.identity==run_payload_ref
    wire=report.to_dict();assert wire['fills']==run.to_dict()['fills']
    path=output/'fill-display.json';save_fill_display(report,path)
    with patch('axiom_engine.runtime.fill_display._coordinate',side_effect=AssertionError('reader computes')), \
         patch('axiom_engine.runtime.fill_display.read_review_display',side_effect=AssertionError('reader queries Data')):
        assert load_fill_display(path).payload==report.payload
    assert before=={str(p):inventory(p) for p in paths}
    reasons={}
    for c in wire['coordinates']:
        reason=c['reason'] or 'AVAILABLE';reasons[reason]=reasons.get(reason,0)+1
    evidence=dict(status='PASS',source_kind='saved_real_account_and_fixed_Data_files_readonly',
        input_run_ref=wire['input_run_ref'],display_ref=wire['display_ref'],display_result_ref=wire['display_result_ref'],
        consumed_input_ref=wire['consumed_input_ref'],content_digest=wire['content_digest'],
        implementation_ref=wire['implementation_ref'],report_status=wire['status'],fills=len(wire['fills']),
        coordinate_status=reasons,run_load_seconds=run_load_seconds,display_load_seconds=display_load_seconds,
        build_seconds=build_seconds,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        original_inputs=before,output_path=str(path.resolve()),output_inventory=inventory(path),
        checks=['source_hash_mtime_size_unchanged','native_fills_exactly_preserved','run_identity_unchanged',
            'Data_public_saved_loader','no_Data_queries_or_transforms','no_account_execution','reader_no_coordinate_compute'])
    (output/'acceptance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('status','report_status','fills','coordinate_status',
        'run_load_seconds','display_load_seconds','build_seconds','peak_process_rss_bytes')}))


if __name__=='__main__':main()
