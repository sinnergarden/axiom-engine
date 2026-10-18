# Published Feature consumption

Feature publication owns input/output interpretation and completeness. Ordinary
FeatureView and coverage reads consume published immutable manifests and selected
buffers directly. They do not replay file hashes, Core output receipts or historical
file-state scans. Explicit source inspection and orphan-partition recovery remain
producer operations. Normal reads propagate I/O errors; published files must not be
mutated in place.

Coverage still matches session, physical axis, column semantic and input identity.
Missing output cells are produced by the existing Core column executor. Full-axis
gaps share the request's security list instead of allocating one list per column
and date. Projection preserves requested row/column order, validity, clocks, reason
codes and original source references. Both legacy and current view formats use the
same reader. No new cache, trust flag or alternative execution entry is introduced.
