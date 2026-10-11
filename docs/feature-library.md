# Incremental Feature library (first review checkpoint)

`axiom_engine.feature_library` owns immutable buffers, per-column/day coverage,
gap execution through the existing Core, and bounded packed projection. Core
keeps its original mathematics and has no storage I/O. Research owns recipes,
source admission, labels, model fitting and experiments.

Construct `FeatureLibrary(root, column_block=16, maximum_working_bytes=...)` and
`FeatureRequest(plan, output_sessions, physical_security_ids, input_semantics,
sample_universe=...)`. The physical grid contains the complete CS reference
cohort; sample_universe only selects returned rows. `plan_coverage(request,
source_adapter)` returns hits, gaps and actual dependency refs without Core.
`ensure_features(...)` publishes only missing dates/columns and returns a saved
view descriptor plus computation and admission counters. Open that fixed
descriptor with `open_feature_view`, call `describe()` for stages/coverage, or
`project(keys, columns)` for owned readonly values, masks, reasons, clocks and
original source bindings. Read-only store imports do not initialize Core or NumPy.

The adapter supplies `dependency_refs(definitions, days, physical_security_ids)`
and `prepare(plan, day, physical_security_ids, dependencies)`. It owns source
selection, the original output-day anchor/cutoff and exact local leaf provenance;
prepare must bind `feature_dependency_ref=digest(dependencies)` in Core input_refs.
The library does not broaden a source's supported universe or PIT policy.

Feature identity binds the actual DAG/input interpretation and scoped operator
semantic revisions. Catalog/selection hashes, entire source manifests and code
commits are provenance rather than sole cache keys. A semantic fix must advance
the affected tag in `core/operator_semantics.py` or its affected input contract,
with an impact record; performance changes preserve tags and prove equivalence.

Files retain the original Core receipt. Each column block adds an explicit
projection receipt after exact Core-buffer verification, without claiming to be
the original full Core output. Constant/sparse/dense codes preserve exact reasons
and source sets. Existing hit bytes are admitted before a COMPLETE view is
published; a persisted index is never a trusted flag. Sealed partitions precede
the atomic index and can be recovered after interruption.

At this checkpoint, two short synthetic cases verify subset zero-computation,
column/date expansion, exact original-Core/raw-to-CS equivalence, reference and
input changes, split column blocks, and one recovery. Full Research
label/model/strategy integration, a real expansion and legacy Feature-matrix
read-only attachment are the next vertical step. Old Research loaders remain
unchanged in their business admission and continue to own those artifacts.
