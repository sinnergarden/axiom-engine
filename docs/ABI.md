# Feature execution ABI 1

The only calculation entrypoint is
`execute_feature_plan(FeaturePlan, FactBatch, ExecutionContext) -> FeatureFrame`.
Contracts accept `from_dict(...)` or JSON text; `.payload` is immutable canonical
JSON and `.to_dict()` returns an independent copy. Transport accepts only finite
JSON values. JSON null is the missing-value mask, equivalent to numeric NaN at an
adapter boundary; inf/NaN tokens, duplicate object keys and unknown reserved tags
are rejected. Python 3.11+, standard library, binary64 arithmetic. No Data I/O,
Research import, current time, random state, registry or filesystem resolver.

## Ownership and plan

The FeaturePlan is Research's explicit lowering into Core ABI, not the R0 prose
FeaturePlanSpec itself. It binds ABI `axiom.feature/1`, semantics
`axiom.operators/1`, immutable recipe/calendar/reference digests, exact source
bindings, ordered input/event schemas, full reference member/industry mappings,
observation domain, history policy, ordered nodes, output projections and
obligations. Every node binds operator version `1` and *all* necessary parameters.
Unknown operator, extra/missing parameter, forward reference or duplicate node
fails. Nodes are single-assignment; a pre-winsorized value remains addressable.
Output aliases preserve dtype/unit/stage/missing policy. This is a finite ordered
instruction list; no scheduler, cache, automatic DAG planning or plugin discovery.

The PR selects the frozen graph alternative allowed by R0. There is no arbitrary
Python plugin execution/loading API. A business formula expressible by these
primitives needs no Core edit. New plugin ABI support would require a separate
reviewed capability; ordinary Python execution must never be described as a
sandbox. Malicious/Research-module plugin requests are rejected as unsupported
operators before anything can be imported.

No Financial RC recipe is automatically lowered. All 96 forensic dependencies
are documented in COVERAGE.md. Data must first supply revision-selected streams,
quarter/TTM facts and source qualifications. Research selects columns, economic
combinations and optional-column branches when freezing a new executable plan.
In particular legacy implicit pct_change fill semantics remain unresolved;
`fill_method=none` is an explicit supported choice, not a claim of equivalence.

## Schema and sources

Column = name, dtype (`float64`, `bool`, `string`, `date`), unit, stage
(`fact`, `base`, `cross_sectional`), missing (`preserve` or `reject`). Inputs are
fact stage. CS stages propagate through subsequent calculations and cannot be
relabelled base. Model-input stages are outside PR1. Dates are ISO calendar dates;
all availability/cutoff timestamps are fixed-width UTC seconds with Z suffix.

FactBatch binds the exact schema, calendar, source declarations and event schemas
from the plan. A daily row contains security_id/session, aligned values,
availability, sources and missing_reasons arrays. Missing cells must have a
reason; present cells cannot carry a missing reason. Each cell names one or more
bound source IDs; an upstream stable derivation supplies maximum dependency
availability and the participating sources. Source declarations include immutable
Data/View digests, revision policy, qualification and availability basis.
Qualifications remain separate (`verified`, `observed`, `best_effort`, `synthetic`),
never promoted by arithmetic, fill or serialization.

Core checks binding consistency, not authenticity of supplier evidence or whether
a caller has forged a digest/qualification. Caller owns artifact payload admission
and adapter mapping, including currency, price adjustment and source field aliases.
There is no implicit unit conversion: addition/subtraction/comparison require
matching units, preserving transforms keep units and rank/zscore are dimensionless.
Research explicitly declares compound units for multiplication/division; this ABI
does not infer dimensional algebra from arbitrary unit strings.

## Three distinct scopes and history

Context contains the explicit ordered calendar, a cutoff for **every** session,
expected history keys, requested output keys and the reference membership and
industry rows with their own source/time evidence. No backtest/shadow/real mode.
Daily rows must exactly cover declared history; duplicate/missing rows fail even
when a value-fill operator exists. Every reference member and industry must match
the plan's complete frozen mapping for that date; deleting an entire security
cannot silently shrink the reference universe. Output filtering occurs last.

`observation_domain=sessions` requires contiguous slices of the supplied exchange
calendar per security. `observations` orders all explicitly declared observations
by security/session, including null values; it does not claim exchange-session
lag. Data remains responsible for proving completeness of the supplied calendar
and observation schedule. Membership never crops the history.

`history_policy=full` checks full dependency depth before execution, by walking requested
output dependencies, including reference members at historical intermediate
cross-sections even when those members exit before the requested date. Depth = max(parent depths) + lag or window
span. Inclusive rolling uses window-1 prior rows; exclusive uses window. Thus
lag756 needs757 observations, two inclusive120 windows need238 prior rows and two
inclusive60 windows need118. `partial` explicitly permits start-of-history warmup;
window min_periods still applies. Missing history rows and legitimate missing
cells are distinct. No calendar-day padding estimates or weekday fallback.

## Temporal and validity policy

Daily facts are gated against their **own session cutoff**, not the final batch
cutoff. A late value becomes null and carries an unavailability issue through all
dependents, including fill, Boolean conversion and skipped rolling/CS values.
This conservative policy never retrospectively repairs earlier Feature rows;
callers needing a different vintage must freeze a different explicit context and
source plan. Cutoffs must be monotonic. Reference evidence must already be visible
or the call fails. Present output dependencies must satisfy that output's cutoff.

Availability is the maximum of dependency times. Source IDs are unioned; input
source declarations are retained verbatim in the output. Conditional selection
conservatively carries both branch dependencies. Legitimate null may be filled
under an explicit policy; unavailable evidence cannot become a valid zero. A null
has valid=false and a reason. `preserve` null output is a valid computation result,
not an execution failure; it is not a valid numeric observation.

Fixed event rows use (stream, security_id, event_id) keys, event_session,
report_period and per-column dependency availability. Same stream/security/date
collisions are rejected: Data must provide an unambiguous fixed projection stream.
Core never chooses among source revisions or constructs quarterly facts.
As-of chooses the last visible event in event_session order. `strict_before`
requires both event date and dependency-availability date strictly before the
Feature session; `exact_date` allows equality, still bounded by the session cutoff.
No visible event produces null. Streams may contain future events: those events
remain invisible. The required report_policy is event_order or nondecreasing. The latter rejects
report-period regression in the supplied stream (including late older reports);
it does not select or repair reports. Report-period advancement/predecessor
production belongs to Data; event ordering is not revision authority.

## Operator choices

- shift: positive periods; counts declared rows including missing observations.
- rolling: finite window, min_periods, inclusive_current, reduction, missing,
  ddof and ties all required. mean/sum/std/min/max/median/rank supported. `skip`
  ignores legal nulls in counts/statistics; `propagate` emits null when any window
  value is null. Percentile rank ranks the endpoint among nonnull values, average
  ties. Rank of a null endpoint remains null. Std supports ddof0/1.
- pct_change: positive periods, explicit fill_method=none, zero=missing/reject.
- arithmetic: add/sub/mul/divide/abs/log1p, comparisons, Boolean and/or/not,
  conditional where, Boolean-to-float, is_missing, fill and clip. Zero division
  and invalid log domains have explicit missing/reject policies. Other nonfinite
  arithmetic results reject, never silently become finite values.
- Boolean missing: preserve/reject/false. Comparison with a missing operand under
  false emits false. Boolean operands treat missing as false under false policy;
  where applies that policy to its condition, preserving branch nulls.
- CS rank/zscore/winsorize: group=session/industry, unknown_group=missing/reject,
  missing=skip/propagate/fill_zero/reject, and excluded policy are explicit. No
  all-loaded-rows fallback. Average ties, linear quantiles, binary64. Ineligible
  extreme values do not contribute. Default choices do not exist in the ABI.
- zscore: ddof, epsilon, constant=zero/missing/reject, and nullable clip bounds.
  std==0 or std<epsilon or undefined sample std takes the declared constant
  branch; input null stays null unless fill_zero. R0's special industry mask is
  expressed as excluded=zero_if_undefined: excluded rows have no mapped scale,
  so nonnull excluded x emits0. Normal excluded policy is missing.
- calendar_age: two date inputs, max(0, first-second) in calendar days. This is
  distinct from exchange-session age; invalid date strings reject at input.

Source `None`, optional absent column, unavailable fact and source absence are
not interchangeable. Missing required columns reject; Research must resolve an
optional-column recipe branch before plan freeze. Arbitrary Python/SQL/formula
text and legacy default behavior are not executable policies.

## Draft, admission, identity and evidence

Unknown wire objects preserve the R0 fields (`contract_type=Unknown`, version1,
metadata, ID, reason, required_evidence), recursively including arbitrary nested
lists/dicts and keys named metadata. Unsupported/malformed reserved tags reject.
`validate_plan` checks draft shape; unresolved drafts remain storable and retain
identity. Execution always checks the complete object recursively and rejects
any Unknown or nonempty obligation. No allow_unknown bypass exists.

Plan identity hashes canonical JSON, including versions, source contracts,
parameters, scope/reference policy and obligations. No locators/display titles
are accepted in this minimal executable contract, so moving the loaded document
cannot change its identity. Fact/context identities bind their supplied transport
content (including row order); output keyed numbers/masks are order invariant,
while those audit identities can differ after input-row reordering. They are
request/content bindings, not output-byte proof or artifact cache admission.

Tests exercise synthetic conformance with independent expected values. Standard
float tolerance is relative1e-13/absolute1e-14, with zero absolute tolerance for
tiny illiquidity values; keys, nulls, stages and policies are exact. No actual
Data/PIT/OOS Financial RC closure, package admission or PR2 inference/signal is
established by these tests. R0's 24 UNKNOWN objects remain in the fixtures.
