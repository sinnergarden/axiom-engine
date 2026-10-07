# Axiom Engine

Shared deterministic computation for Axiom. Core and Runtime are distinct
logical owners in this independent repository. Core owns features and pure
rotation decisions; Runtime owns an offline cached-signal replay and account ledger.

The source package version is `0.3.1`, also exposed as `axiom_engine.__version__`.
Version `0.3.0` first declares the public
`axiom_engine.core.evaluate_signal_statistics(input, *, spec)` API and the Runtime
`stock_prediction_schedule(*, folds, calendar)` API. Consumers requiring these
APIs must declare `axiom-engine>=0.3.0` and retain their reviewed source commit
lock. Signal statistics retain `signal_statistics_input_v1` / `signal_statistics_v1`
data contracts; package versions and saved contract versions are separate.
Version `0.3.1` adds the ETF profile/grid and request/run v5, the pure
`etf_buy_and_hold_policy` / `plan_etf_buy_and_hold` API, and the explicit
`benchmark_comparison_v2` evaluation projection. Consumers of these additions
must require `axiom-engine>=0.3.1` and lock the reviewed source commit.

E0 PR1 exposes `axiom_engine.core.execute_feature_plan` with immutable explicit
FactBatch, FeaturePlan and ExecutionContext inputs. See [ABI](docs/ABI.md),
[96-column coverage](docs/COVERAGE.md) and [source bindings](docs/SOURCES.json).

Run synthetic conformance from this repository:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The package has no third-party runtime dependencies. Tests do not need Data,
Research or legacy workspaces. Financial RC remains an unresolved R0 draft;
passing these tests does not admit real Data/PIT/OOS execution.

The offline account API is `axiom_engine.runtime.run_backtest(BacktestRequest)`;
`save_backtest_run` and `load_backtest_run` persist/read complete canonical results.
The optional read-only Data adapter retains UNKNOWN market status. The Engine
`daily_open_profile()` defaults to strict blocking; the ETF example explicitly
selects `etf_daily_observed`, a daily simulation assumption using valid open,
positive volume and legal limits. Explicit/partial suspension remains blocked.
Cash, commissions, holdings, dividends and NAV use fixed-point accounting. This
release has no live broker or SQLite recovery support. Authoritative design and
experiment conventions remain in the separate `axiom-docs` repository.

The bounded stock v7 path is
`run_stock_backtest(manifest, *, source, sink, block_sessions, limits)`, with
`StockInputSource` reading fixed saved inputs and `StockResultSink` writing
immutable result parts. `stock_run_id(manifest)` supplies the sink's run identity.
An original source audits the complete fixed inputs before creating the same
Runtime account used by v6. `audit_stock_backtest_source` performs that audit independently.
The explicit limits bound input reads, decoded blocks, pending output, parts and
total results; synthetic checks do not establish a long-window RSS guarantee.

For sequential accounts over the same saved predictions, use
`admit_stock_inputs(manifest, *, source, block_sessions, limits, max_owned_bytes)`
and pass its returned `AdmittedStockInputs` as the existing `source` argument.
Use the handle as a context manager or call `close()`. Import fully validates
the original bytes and captures projected blocks in a bounded private temporary
file; each account decodes only its current block. No portable validated marker
or new source identity is created. Original source span indexes are released.

Only `account_id`, positive initial cash (with empty initial positions), and a
legal `portfolio_policy.top_k` may change. All other manifest fields retain
their admitted capability, including policy semantics, profile/fees/rules,
scope, clock/action policy, snapshots, native refs and fold/model/prediction
refs. Artifact locations retain their existing delivery-only semantics. Unknown
risk parameters fail rather than receive implicit permission to reuse inputs.
`block_sessions` remains fixed for this handle; every account's input, decoded
and result budgets still apply. Core, broker and ledger execution is unchanged.

`load_stock_backtest_projection(path, *, artifact_reader, limits)` validates a
saved v7 run, its result parts, and small profile/event views for evaluation and
fill display. It does not reopen large Data, prediction or training parents.
Complete projections use the existing v2/v3 evaluation and display algorithms;
blocked accounts preserve their saved valid prefix and stopping phase.
