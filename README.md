# Axiom Engine

Shared deterministic computation for Axiom. Core and Runtime are distinct
logical owners in this independent repository. Core owns features and pure
rotation decisions; Runtime owns an offline cached-signal replay and account ledger.

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
