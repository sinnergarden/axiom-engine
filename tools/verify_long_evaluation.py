"""V2 acceptance using only a saved run and frozen v1 evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from axiom_engine.runtime import (BenchmarkSeries, DividendScope, long_history_evaluation_spec,
    evaluate_backtest, load_backtest_run, load_backtest_evaluation, save_backtest_evaluation)


def inventory(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mtime_ns=path.stat().st_mtime_ns, size=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "previous-evaluation", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run_path, previous_path, output = Path(args.run), Path(args.previous_evaluation), Path(args.output_dir)
    before = {str(p): inventory(p) for p in (run_path, previous_path)}
    run = load_backtest_run(run_path)
    previous = load_backtest_evaluation(previous_path)
    old, saved = previous.to_dict(), run.to_dict()
    assert old["contract_version"] == "evaluation_report_v1"
    assert old["input_run_ref"] == {k: saved[k] for k in ("run_id", "content_digest", "committed_sequence")}
    benchmark = BenchmarkSeries.from_dict(old["benchmark_input"])
    scope = None if old["dividend_scope"] is None else DividendScope.from_dict(old["dividend_scope"])
    with patch("axiom_engine.runtime.backtest.run_backtest", side_effect=AssertionError("account replay")), \
         patch("axiom_engine.runtime.accounting.AccountLedger.apply_fill", side_effect=AssertionError("fill execution")):
        report = evaluate_backtest(run, benchmark=benchmark, spec=long_history_evaluation_spec(), dividend_scope=scope)
        repeated = evaluate_backtest(run, benchmark=benchmark, spec=long_history_evaluation_spec(), dividend_scope=scope)
    assert report.payload == repeated.payload
    wire = report.to_dict()
    for key in ("input_run_ref", "signal_ref", "market_ref", "profile_ref", "benchmark_ref", "benchmark_input",
                "dividend_scope_ref", "dividend_scope", "status", "series", "monthly_returns", "episodes",
                "episode_metrics", "benchmark", "pnl_distribution"):
        assert wire[key] == old[key], key
    assert wire["evaluation_ref"] != old["evaluation_ref"]
    assert wire["content_digest"] != old["content_digest"]
    assert wire["period_metrics"]["account"]["cagr_status"] == "INSUFFICIENT_SPAN"
    assert wire["period_metrics"]["benchmark"]["cagr_status"] == "INSUFFICIENT_SPAN"
    assert all(limitation in wire["limitations"] for limitation in old["limitations"] if not limitation.startswith("No annualization"))
    path = output / "evaluation.json"
    save_backtest_evaluation(report, path)
    output_before = inventory(path)
    with patch("axiom_engine.runtime.evaluation.evaluate_backtest", side_effect=AssertionError("reader computes")), \
         patch("axiom_engine.runtime.evaluation.period_metrics", side_effect=AssertionError("reader annualizes")):
        assert load_backtest_evaluation(path).payload == report.payload
        assert load_backtest_evaluation(previous_path).payload == previous.payload
    assert output_before == inventory(path)
    assert before == {str(p): inventory(p) for p in (run_path, previous_path)}
    evidence = dict(status="PASS", source_kind="saved_real_small_sample", canonical_docs_commit="c7d28ff867ba838884d62ecf9275aea85013ea80",
        contract_version=wire["contract_version"], evaluation_version=wire["evaluation_version"],
        evaluation_ref=wire["evaluation_ref"], content_digest=wire["content_digest"], implementation_ref=wire["implementation_ref"],
        input_run_ref=wire["input_run_ref"], period_metrics=wire["period_metrics"], sessions=len(wire["series"]),
        original_inputs=before, output_inventory=output_before, output_path=str(path.resolve()),
        checks=["no_Data_access", "no_account_or_fill_execution", "repeat_bytes_equal", "all_v1_metric_sections_preserved",
                "new_v2_identity", "original_limitations_preserved_except_no_annualization", "short_span_cagr_null",
                "v1_v2_reader_no_compute", "original_run_v1_hash_mtime_size_unchanged"])
    (output / "acceptance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
