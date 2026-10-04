"""Explicit offline P10 evaluation; private artifacts remain in the caller path."""
import argparse

from axiom_engine.runtime import (load_backtest_run, read_csi300_benchmark,
    read_dividend_scope, daily_evaluation_spec, long_history_evaluation_spec,
    evaluate_backtest, save_backtest_evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data-root", "snapshot", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--long-history", action="store_true", help="Use the saved-wealth v2 CAGR profile")
    args = parser.parse_args()
    from axiom_data import Data  # optional dependency; no fetching or construction
    run = load_backtest_run(args.run)
    wire = run.to_dict()
    plan = wire["plan"]
    calendar = plan["market_replay"]["calendar"]
    anchor = calendar[calendar.index(plan["start_session"]) - 1]
    data = Data(args.data_root)
    benchmark = read_csi300_benchmark(data, snapshot=args.snapshot,
        sessions=[anchor, *[row["session"] for row in wire["nav"]]])
    scope = read_dividend_scope(data, snapshot=args.snapshot, universe=plan["signal_frame"]["universe"],
        start_session=plan["start_session"], end_session=plan["end_session"])
    spec = long_history_evaluation_spec() if args.long_history else daily_evaluation_spec()
    report = evaluate_backtest(run, benchmark=benchmark, spec=spec, dividend_scope=scope)
    save_backtest_evaluation(report, args.output)
    print(report.to_dict()["evaluation_ref"])


if __name__ == "__main__":
    main()
