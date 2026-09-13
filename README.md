# Axiom Engine

Shared deterministic computation for Axiom. Core and Runtime are distinct
logical owners in this independent repository. Feature execution is introduced
through reviewed changes; runtime is outside the initial scope.

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
