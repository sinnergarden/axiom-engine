# PR1 F1/F2 repair

Reviewed baseline: `7924c95820b8a9f52790649fe4c4d5579e5a7723` on
`e0/pr1-feature-execution`, PR #1. At repair start local HEAD exactly matched,
with a clean worktree and no later local commits. The original independent report
in `review/engine-e0-pr1-independent/REVIEW.md` remains REQUEST_CHANGES pending
user-arranged re-review; this document records construction repair evidence.

## Finding map

| Finding | Root cause | Repair | Regression evidence |
|---|---|---|---|
| F1 / P2 | Full-history admission walked actual dependencies, while every node and reject policy executed on every history row. Unneeded first-row shift nulls and first-date CS nulls rejected valid later outputs. | Extend that same walk to return per-node required keys. Evaluate nodes in unchanged sequential order at those keys, addressing shifts/windows through original full-history positions. CS contributors still come from the complete frozen date/group reference, not requested output keys. All required intermediate schema/domain/missing policies still run. | Original two F1 probes pass verbatim; controls cover missing lag predecessor, missing raw row, missing reference member/value, final fill masking an invalid intermediate, rolling→shift, historical CS→shift after membership exit, single-stock request, industry group, partial warmup and batch/daily. |
| F2 / P2 | `_std` squared binary64 deviations directly. The square of 1e-200 underflowed even though its standard deviation is representable, routing nonconstant inputs to the zero-scale branch. | Shared `_std` uses standard-library `statistics.pstdev` for ddof0 and `statistics.stdev` for ddof1, retaining the existing insufficient-count guard. Exact-ratio accumulation and scaled square root avoid the intermediate-square underflow. | Both original F2 probes pass verbatim. Additional rolling std and CS zscore controls use 1e-200, 1 and 1e150 scales with ddof0/1 and abs_tol=0; true constant, epsilon threshold, clip and missing policies remain tested. |

## Contract clarification and caller audit

Input history, requested output keys, per-node dependency keys and statistic
contributors are different sets. Input envelopes still undergo schema/key/source/
cutoff/reference validation. Required-node rejection includes historical
intermediates and any complete reference contributors pulled in by a downstream
CS node. No global reject-to-skip change, final-only validation or prefiltered
history is used. `partial` permits absent predecessors but required warmup nulls
still obey missing=reject. Inactive nodes remain structurally/semantically
validated by plan admission; no scheduler or general DAG framework was added.

The two `_std` callers are `_rolling` (reduction=std) and `_cross_section`
(cs_zscore). Both use the repaired function. Centering, ddof, epsilon, constant
handling, missing filtering, finite input/output checks and clip are unchanged.
The ABI/node versions remain the existing PR1 draft versions: this corrects their
specified behavior in the same unmerged PR, with the new Git head identifying
repair implementation bytes. No new business formula or release is introduced.

## Verification

Luna executed:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 /home/liuming/workspace/axiom/review/engine-e0-pr1-independent/probes.py
```

Full suite: **32/32 PASS** (16 original + 6 preserved reviewer probes + 10 added
controls). Original external probes: **6/6 PASS**. Logs are
`reports/conformance.log` and `reports/reviewer-probes-repair.log`.
`tests/test_review_probes.py` is byte-for-byte equal to the original probes file;
assertions and tolerances were not changed. Small std checks use rel_tol=1e-13,
abs_tol=0, and original zscore checks still expect exact [-1,1]. Row validity,
availability, source references, keys and schemas are compared in batch/daily
controls. Original review report/probes/logs are preserved outside the repository.

## Remaining scope

Financial RC stays BLOCKED; all 24 R0 UNKNOWN objects are unchanged. Research R0
`bbb1a21d55a78642d22ea284c457272a18a017c2` and its frozen fixtures were not modified.
Actual Data/PIT/OOS/units/calendar/96-column execution and scale performance remain
unverified. Repeated history/reference scans identified as nonblocking performance
feedback remain outside this correctness repair; no performance claim is made.
The work-plan document now distinguishes original delivered construction from
this repair and delegates final publication status to the fixed-SHA handoff.
PR1 remains open for independent re-review; no PR2, merge or auto-merge.
