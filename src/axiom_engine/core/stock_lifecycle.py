"""Feature-session lifecycle proofs; source policy, observation and execution stay separate."""
from .contracts import Document, digest, fields, require
from .stock_rules import listed

AVAILABILITY_POLICY = {
    "profile_id": "tushare.stock_basic.listing_events.v1",
    "timezone": "Asia/Shanghai", "session_release_time": "00:00:00",
    "date_field": "event_date", "date_rule": "same_day_release",
    "basis": "best-effort supplier event-date assumption; no historical public vintage",
}
AVAILABILITY_POLICY_REF = Document.from_dict(AVAILABILITY_POLICY).identity
_SEAL = object()


class _FeatureLifecycle:
    """Invocation-private proof produced from admitted original native rows."""
    def __init__(self, binding, facts, seal):
        require(seal is _SEAL, "original lifecycle admission required")
        self._wire = Document.from_dict({"binding": binding, "facts": facts})
        self._binding = Document.from_dict(binding)
        self.reference = self._binding.identity

    @property
    def binding(self):
        return self._binding.to_dict()

    @property
    def facts(self):
        return self._wire.to_dict()["facts"]

    def context(self):
        return {"reference": self.reference, "binding": self.binding}


def native_lifecycle_active(identity, day, state, metadata):
    """Use original per-session state; nullable identity end is not immortality.

    The admitted state batch binds its original event revision/Raw references.
    Missing state usable_from stays missing; best-effort query clocks are not
    upgraded to strict historical receipt evidence. No event date is inferred.
    """
    from .stock_portfolio import instant
    first, last = identity["listing_date"], identity["delisting_date"]
    location = identity["security_id"] + "/" + day
    if state == "not_listed":
        require(day < first, "native not_listed contradicts canonical lifecycle: " + location)
        return False
    if state == "delisted":
        require(first <= day and (last is None or last <= day),
                "native delisted contradicts canonical lifecycle: " + location)
        require(metadata.get("missing_reason") is None and
                metadata.get("evidence_domain") in ("listing_events", "security_master") and
                all(type(metadata.get(name)) is str and bool(metadata[name])
                    for name in ("revision_id", "raw_batch_id")),
                "native delisted lacks bound lifecycle evidence: " + location)
        if identity.get('_stock_identity_version') == 2:
            proof = identity.get('_stock_delisting_evidence')
            require(proof is not None and metadata.get('evidence_domain')=='listing_events' and
                    all(metadata[name]==proof[name] for name in ('revision_id','raw_batch_id')),
                    "native lifecycle selects a different original event revision: " + location)
            if metadata.get('availability_basis') == 'first_observed_at':
                require(all(metadata.get(name) is None or
                            instant(metadata[name]) == instant(proof[name])
                            for name in ('first_observed_at', 'usable_from')),
                        "native lifecycle observation time differs from original event proof: " + location)
        available = metadata.get("usable_from")
        require(available is None or instant(available) <= instant(day + "T20:30:00+08:00"),
                "future native lifecycle evidence: " + location)
        return False
    active = listed(identity, day)
    require(state == "unknown_status" or active,
            "native stock state contradicts canonical listing lifecycle: " + location)
    return active


def lifecycle_binding(*, owner_native_ref, feature_session, knowledge_cutoff, stock_execution_rules_ref):
    return dict(contract_version="stock_feature_lifecycle_input_v1", feature_session=feature_session,
        knowledge_cutoff=knowledge_cutoff, pit_policy="best_effort_vendor_v1", owner_native_ref=owner_native_ref,
        stock_execution_rules_ref=stock_execution_rules_ref, availability_policy_ref=AVAILABILITY_POLICY_REF)


def _compile_feature_lifecycle(binding, records, metadata, identities):
    from .stock_portfolio import instant
    day, cutoff = binding["feature_session"], binding["knowledge_cutoff"]
    require(binding == lifecycle_binding(owner_native_ref=binding["owner_native_ref"], feature_session=day,
        knowledge_cutoff=cutoff, stock_execution_rules_ref=binding["stock_execution_rules_ref"]),
        "unsupported stock lifecycle policy/binding")
    digest(binding["owner_native_ref"]); digest(binding["stock_execution_rules_ref"])
    require(instant(cutoff) == instant(day+"T20:30:00+08:00"), "lifecycle Feature cutoff mismatch")
    require(set(records) == set(metadata) == set(identities), "complete Feature lifecycle identity union required")
    facts = {}
    for security, row in records.items():
        meta, identity = metadata[security], identities[security]
        require(row["session"] == meta["session"] == day and row["security_id"] == meta["security_id"] == security,
                "lifecycle native key mismatch")
        state = row["market_state"]
        require(state in ("not_listed", "delisted", "normal_trading", "suspended", "source_gap", "unknown_status"),
                "unsupported native stock lifecycle state")
        active = native_lifecycle_active(identity, day, state, meta)
        native_available = meta.get("usable_from")
        require(native_available is None or instant(native_available) <= instant(cutoff), "future Feature lifecycle metadata")
        reason = meta.get("missing_reason") or ""
        known_active = active and not reason.startswith(("identity_", "listing_", "calendar_")) and (
            state != "unknown_status" or reason.startswith("status_"))
        lifecycle = state if state in ("not_listed", "delisted") else ("active" if known_active else "unknown")
        proof = identity.get("_stock_delisting_evidence") if state == "delisted" else None
        policy_available = (identity["delisting_date"]+"T00:00:00+08:00"
                            if state == "delisted" and identity["delisting_date"] is not None else native_available)
        require(policy_available is None or instant(policy_available) <= instant(cutoff), "future policy lifecycle availability")
        facts[security] = dict(lifecycle=lifecycle, market_state=state, missing_reason=meta.get("missing_reason"),
            native_usable_from=native_available, policy_available_at=policy_available,
            availability_basis="declared_vendor_assumption", first_observed_at=(proof or {}).get("first_observed_at", meta.get("first_observed_at")),
            original_availability_basis=(proof or {}).get("availability_basis", meta.get("availability_basis")),
            revision_id=meta.get("revision_id"), raw_batch_id=meta.get("raw_batch_id"),
            source_refs=[binding["owner_native_ref"], *identity["source_refs"]])
    return _FeatureLifecycle(binding, facts, _SEAL)


def validate_feature_lifecycle_input(value, *, identities, rules_ref, feature_session, feature_cutoff, admitted=None):
    """Public proof closes original bytes; private Runtime reuses this invocation's admission."""
    from .stock_portfolio import instant
    fields(value, "reference binding" + (" owner_batch" if admitted is None else ""))
    binding = value["binding"]
    fields(binding, "contract_version feature_session knowledge_cutoff pit_policy owner_native_ref stock_execution_rules_ref availability_policy_ref")
    require(value["reference"] == Document.from_dict(binding).identity and
            binding["stock_execution_rules_ref"] == rules_ref and binding["feature_session"] == feature_session and
            instant(binding["knowledge_cutoff"]) == instant(feature_cutoff), "stock Feature lifecycle reference/clock mismatch")
    if admitted is not None:
        require(type(admitted) is _FeatureLifecycle and admitted.reference == value["reference"] and admitted.binding == binding,
                "invocation lifecycle capability mismatch")
        return admitted
    batch = value["owner_batch"]; context = batch["context"]; query = context["query"]
    # Reject a historical batch before hashing/decoding its records. Public
    # callers supply one original native day; Runtime borrows its admitted owner.
    require(query["sessions"] == [feature_session] and
            query["fields"] == ["close"] and
            set(query["cutoff_by_session"]) == {feature_session} and
            type(batch["records"]) is list and len(batch["records"]) == len(identities) and
            set(batch["field_meta"]) <= {"close", "market_state"} and
            all(type(meta["by_key"]) is list and len(meta["by_key"]) == len(identities)
                for meta in batch["field_meta"].values()) and
            type(batch["field_meta"]["market_state"]["by_key"]) is list and
            len(batch["field_meta"]["market_state"]["by_key"]) == len(identities),
            "public lifecycle input must be one complete native Feature day")
    require(Document.from_dict(batch).identity == binding["owner_native_ref"], "original lifecycle owner bytes mismatch")
    require(context["contract_version"] == "data_batch_v1" and context["domain"] == "market_state_diagnostics" and
            query["pit_policy"] == binding["pit_policy"] == "best_effort_vendor_v1" and
            query.get("policy_by_session") is None and query.get("price_basis") == "unadjusted" and
            query["purpose"] == "market_replay" and query["symbols"] == list(identities) and
            feature_session in query["sessions"] and query["cutoff_by_session"][feature_session] == binding["knowledge_cutoff"],
            "original Feature lifecycle query/policy required")
    records = {r["security_id"]:r for r in batch["records"] if r["session"] == feature_session}
    metadata = {m["security_id"]:m for m in batch["field_meta"]["market_state"]["by_key"] if m["session"] == feature_session}
    require(len(records) == sum(r["session"] == feature_session for r in batch["records"]) and
            len(metadata) == sum(m["session"] == feature_session for m in batch["field_meta"]["market_state"]["by_key"]),
            "duplicate original Feature lifecycle keys")
    return _compile_feature_lifecycle(binding, records, metadata, identities)
