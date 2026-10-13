"""Scoped mathematical revisions for reusable Feature results.

Change the affected entry for a semantic correction and record its impact.
Performance/storage changes keep it fixed and prove output equivalence. Code
commits remain execution provenance; they never replace these semantic tags.
"""
from .plan import PARAMS

# New rolling families are separate entries; the original rolling tag is stable.
OPERATOR_SEMANTIC_REVISIONS = {name: 'r0_' + name + '_v1' for name in PARAMS}
