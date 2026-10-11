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

`StockInputSource(scalar_cache_bytes=262144)` reuses already validated small
canonical literals within one import. Set `scalar_cache_bytes=0` for the original
scanner oracle. Reuse is charged to the shared decoded budget and evicted before
required input growth. The full source bytes, JSON structure, key ordering and
reserved Unknown objects are still checked. `source.statistics` exposes per-file
scan/read/hash, scalar decode/canonical counts/timings and selected-row operations;
`inputs.statistics` separates source audit/capture from subsequent owned reads.
The synthetic exact/performance probe is
`PYTHONPATH=src:tests python tools/bench_stock_owned_inputs.py`; it uses no real
saved inputs, Data access, features, fitting or prediction calls.
Recorded synthetic/real measurement scopes, source bindings, CJSON counters and
test coverage are in [stock input benchmark notes](tools/stock_input_benchmarks.md).
Authoritative input-reuse design remains in Axiom design 04.

`load_stock_backtest_projection(path, *, artifact_reader, limits)` validates a
saved v7 run, its result parts, and small profile/event views for evaluation and
fill display. It does not reopen large Data, prediction or training parents.
Complete projections use the existing v2/v3 evaluation and display algorithms;
blocked accounts preserve their saved valid prefix and stopping phase.

For saved v8 equity replay, `equity_profile(*, execution_rules, fee_schedule,
unknown_status_policy='block', share_allocation_policy='reject_non_integer')`
retains the strict v3 profile by default. Explicitly selecting
`share_allocation_policy='historical_replay_floor'` creates profile v4 with
`stock_equity_simulation_v2`. It floors the total grant once per account/event;
physical, sellable and pending shares remain integers. The theoretical remainder
is recorded in `EQUITY_EX.share_allocation` and the saved entitlement, without
cash compensation or fractional assets. Profile/request/run identities change;
the saved prediction and Data identities remain intact. Replay still uses the
same Core, simulator, T+1, fee schedule and original prices.

`stock_action_facts_v1` also accepts the explicit historical clock declaration
`mixed_reviewed_sources_v1`. Data freezes the applicable ordinary-holder cash
and share scalars and binds its reviewed interpretation through source refs;
Engine consumes those scalars without interpreting issuer restructuring plans.
Each action keeps its dependency-gated availability and actual receipt time.
Unknown share entitlements remain unknown and can still block a held account.

Saved Derived combinations bind the global `IMPLEMENTATION_REF` and admission
requires it to equal the current Engine version. Even an accounting-only change
therefore rejects an older Derived combination although SignalPlan arithmetic
has not changed. For this experiment, regenerate the combination from the same
saved raw predictions under the final Engine version, save a new identity and
report that migration's time and resource cost. Do not retrain, rebuild Features,
retain an inaccurate implementation hash or bypass the check. Raw prediction
bindings do not have this Derived-only equality gate, and saved accounts remain
readable. Separate component semantic identities are future maintenance; this
release does not change signal correctness or introduce a compatibility layer.

`settle_stock_share_allocation(*, registered_quantity, shares_per_share,
policy='reject_non_integer', actual_quantity=None, evidence_refs=())` is a pure
allocation function. `confirmed_integer_credit` requires an actual nonnegative
integer and evidence refs and uses that quantity directly. A shadow caller must
bind the evidence to the account and its actual availability clock; this release
does not add a broker integration. Historical floor never silently substitutes
for a missing confirmed credit.

V2/V3 evaluation retains the account's allocated integer total when attributing
shares to record-origin episodes, and exposes
`episode_metrics.share_allocation_summary`: event evidence, per-security
cumulative theoretical tails, possible additional shares, and local original-price
mark sensitivity. These are disclosures, not account assets or a P&L correction.
They do not bound full-path return differences or establish conservative returns.
No persisted market admission or checkpoint/resume support is introduced.

The opt-in stock strategy API uses the same bounded v7/v8 account path and a
new `axiom.stock_portfolio/4` decision. The existing TopK factories and v1/v2/v3
decisions retain their original configuration. The pure `StockStrategy` protocol
and built-in `ConfiguredStockStrategy` are exported by `axiom_engine.core`;
the approved input/output and economic conventions are in the
[shared strategy design](https://github.com/sinnergarden/axiom-docs/blob/codex/docs-engine-strategy-protocol-20261011/docs/design/engine-strategy-protocol-v1.md).

```python
from axiom_engine.core import stock_strategy_spec
from axiom_engine.runtime import stock_portfolio_policy

specs = [
    stock_strategy_spec(top_k=5),
    stock_strategy_spec(selection="rank_buffer", top_k=5,
                        rank_buffer={"keep_rank": 10}),
    stock_strategy_spec(top_k=5, exit_overlay={
        "take_profit": "0.20", "stop_loss": "0.10",
        "cost_basis": "fee_inclusive_ledger_cost_per_owned_share_v1",
        "check": "rebalance_sessions_only", "cooldown_rebalance_sessions": 1,
        "reentry": "after_flat_and_cooldown",
    }),
]
policy = stock_portfolio_policy(strategy_spec=specs[0], universe_id="csi300",
    execution_universe=frozen_union, execution_rules=frozen_rules)
```

Bind `policy` to the original bounded request and recompute its `request_ref`;
keep the saved daily prediction bindings, market and profile unchanged. Accounts
remain isolated. CSI300 keeps its existing support reference; another pool must
bind `portfolio_support_ref(policy, frozen_union)` and pass original native
membership admission. The admitted market owner can reuse its CSI300 market for
these policies, then bind each request's saved signals; its existing capability
does not authorize a different membership pool.

All three strategies check weekly and execute through the next session's
existing daily-open simulation. Weekly-close exits include buying fees and
confirmed share entitlements in their average-cost basis; cash dividends do not
reduce that basis. Frozen shares, pending grants and registered future stock
rights occupy slots. Reserved slots add no asset or NAV value. A new buy that
would exceed the actual owned-position limit records `MAX_OWNED_POSITIONS`.
The saved loader checks strategy/state identities, exit observations against the
prior original-unit position snapshot, and capacity from the saved journal; it
does not replay the strategy or reopen signals/native inputs.

This release's evidence is synthetic source validation. The designated real
case is Research's 2023 historical CSI300 cross-section with 88 features; a
CSI800 input or historical nonfinancial classification is not claimed here.
Source changes also change the global implementation reference: previously
derived signals must still follow the existing explicit raw-to-Derived migration
when necessary. Original saved accounts and raw signal identities remain intact.
