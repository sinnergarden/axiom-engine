# PR1 conformance evidence

Latest construction validation after F1/F2 repair: 32 unittest methods passed
(16 original, 6 preserved reviewer probes, 10 repair controls). Command:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Executed by GPT-5.6 Luna; full output is `reports/conformance.log`. These are
construction tests with independent numerical expectations, not an independent
reviewer approval. The independent review of 7924c958 returned REQUEST_CHANGES for F1/F2.
See [repair mapping](REPAIR_F1_F2.md) and reports/reviewer-probes-repair.log for
new evidence. Independent re-review and final disposition remain with the user
and architecture conversation.

| Requested acceptance | Executable evidence in tests/test_conformance.py |
|---|---|
| R0 PR1 declarative cases | test_r0_declarative_golden_cases; event, UNKNOWN and schema tests |
| Public plan to frame | Every numerical check calls execute_feature_plan through run |
| Shuffled keys, duplicates, bad schema | test_projection_shuffle_duplicates_schema |
| Membership exit/reentry keeps history | test_membership_reentry_does_not_compress_history |
| 756 lag / 238 pullback / 118 nested windows | test_756_lag_full_history_and_nested_closure |
| Missing rows vs legal missing vs observation domain | test_sessions_vs_observations_and_missing_cells |
| Full frozen CS for one output; pool outlier isolation | test_single_output_full_reference_and_industry |
| Past CS member history even after exit | test_full_history_includes_past_reference_members |
| Rank ties, quantiles, std, constants and missing | test_window_endpoints_reductions_ddof_missing; test_cs_constants_prefill_scale_clip_and_empty |
| Arithmetic, Boolean, calendar age and policies | test_arithmetic_boolean_and_legal_missing; test_remaining_primitives_and_rejections |
| Same-date strict/exact events; future/late events; report regression | test_event_strict_exact_future_and_availability |
| Per-session cutoff and qualification preservation | test_late_fact_not_authorized_by_final_cutoff_or_fill |
| Batch and per-session common path | test_batch_and_single_session_same_entrypoint (exact keys/nulls/stages/numbers) |
| UNKNOWN nested transport, unsupported version/operator/policy | test_unknown_transport_admission_and_versions |
| No current directory/time/latest dependence, no I/O/Research imports | test_freezing_identity_and_no_mutable_plugin_or_io (static imports, blocked open/socket, fresh process in unrelated cwd) |

The Core has no wall-clock call, verified by its restricted dependency/call
surface. A fresh process with an unrelated latest file produces the same frame.
There is no Python plugin loader; arbitrary plugin/module requests fail admission.
This is not a claim that Python execution is sandboxed.

R0 cases for saved preprocessing, model inference and signal combination remain
PR2. The zscore numerical case is exercised as a Feature primitive only; no
SignalFrame is emitted. R0's 96 dependency definitions and 24 Unknown objects
remain source-bound; the tests do not execute all 96 as a certified RC release.
All source units in R0's selected output schema are DATA_CLOSURE Unknown values;
synthetic fixtures use explicit units and do not resolve those real obligations.

Unverified: actual public Data closure, revision-specific PIT qualifications,
original RC membership/industry/calendar/units, original allowed OOS lineage,
real 96-column numerical equivalence and production package admission. Missing
legacy pct_change fill policy and nonfinite legacy division behavior are not
silently promoted to formal defaults. Unsupported/nonfinite cases reject; a new
Research release must explicitly resolve and version those policies.

No Research, Data, legacy or design file was modified. No bulk build, training,
model loading, Signal orchestration, Runtime, merge or auto-merge is part of PR1.
