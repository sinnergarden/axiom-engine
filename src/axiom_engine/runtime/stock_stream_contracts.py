"""Small identities and budgets for bounded stock input and saved result parts."""
from copy import deepcopy

from ..core.contracts import Document, digest, fields, integer, require, session, text
from .stock_schedule import CLOCK_POLICY
from .stock_inputs import ACTION_POLICY

REQUEST_VERSION = "backtest_request_v7"
RUN_VERSION = "backtest_run_v7"
STREAM_RUNTIME_VERSION = "axiom.backtest/7"
AUDIT_VERSION = "stock_input_audit_v1"
RESULT_PART_VERSION = "stock_backtest_result_part_v1"
RESULT_GROUPS = ("nav", "positions", "decisions", "orders", "fills", "cash_ledger",
                 "position_ledger", "session_phases")
LIMIT_FIELDS = ("max_folds", "max_prediction_rows", "max_market_rows", "max_input_bytes",
                "max_read_bytes", "max_block_bytes", "max_result_part_bytes",
                "max_result_buffer_bytes", "max_result_bytes")
ARTIFACT_FIELDS = "artifact_type artifact_id contract_version manifest_uri content_digest"


def validate_artifact_ref(ref):
    fields(ref, ARTIFACT_FIELDS)
    for name in ("artifact_type", "artifact_id", "contract_version", "manifest_uri"):
        text(ref[name])
    digest(ref["content_digest"])
    return ref


def artifact_identity(ref):
    """Logical ArtifactRef without its delivery location, not its native identity."""
    validate_artifact_ref(ref)
    return {name: ref[name] for name in ("artifact_type", "artifact_id", "contract_version", "content_digest")}


def identity_view(value):
    """Remove locations only at the exact, published ArtifactRef positions."""
    value = deepcopy(value)
    if "profile_input" in value:
        value["profile_input"]["artifact"] = artifact_identity(value["profile_input"]["artifact"])
    market = value.get("market_input", value if value.get("contract_version") == "stock_market_input_refs_v1" else None)
    if market is not None:
        for item in market["native_inputs"]:
            item["artifact"] = artifact_identity(item["artifact"])
    predictions = value.get("prediction_input", value if value.get("contract_version") == "stock_prediction_input_refs_v1" else None)
    if predictions is not None:
        for frame in predictions["frames"]:
            for name in ("fold_spec_artifact", "model_metadata_artifact", "prediction_artifact"):
                frame[name] = artifact_identity(frame[name])
    return value


def logical_ref(value, field):
    unsigned = dict(value)
    unsigned.pop(field, None)
    return Document.from_dict(identity_view(unsigned)).identity


def validate_limits(limits):
    fields(limits, " ".join(LIMIT_FIELDS))
    for value in limits.values():
        integer(value, 1)
    return dict(limits)


def read_budget(limits):
    validate_limits(limits)
    return {"max_read_bytes": limits["max_read_bytes"], "max_decoded_bytes": limits["max_block_bytes"]}


def write_budget(limits):
    validate_limits(limits)
    return {"max_part_bytes": limits["max_result_part_bytes"],
            "max_buffer_bytes": limits["max_result_buffer_bytes"],
            "max_total_bytes": limits["max_result_bytes"]}


def validate_manifest(manifest):
    plan = manifest.to_dict() if isinstance(manifest, Document) else deepcopy(manifest)
    fields(plan, "contract_version request_ref account_id scope initial_account portfolio_policy profile_input market_input prediction_input stock_action_policy clock_policy limitations")
    require(plan["contract_version"] == REQUEST_VERSION, "bounded stock request required")
    text(plan["account_id"])
    scope = plan["scope"]
    fields(scope, "start_session end_session anchor_session calendar prediction_universe execution_universe supported_universe_ref")
    calendar = scope["calendar"]
    require(type(calendar) is list and len(calendar) > 1 and calendar == sorted(set(calendar)), "frozen exchange calendar required")
    for day in calendar:
        session(day)
    for name in ("start_session", "end_session", "anchor_session"):
        session(scope[name])
        require(scope[name] in calendar, "stock scope absent from calendar")
    require(scope["anchor_session"] == calendar[0] and scope["start_session"] == calendar[1] and
            scope["end_session"] == calendar[-1], "stock scope needs one explicit preceding anchor")
    for name in ("prediction_universe", "execution_universe"):
        universe = scope[name]
        require(type(universe) is list and bool(universe) and len(universe) == len(set(universe)), "frozen unique stock universe required")
        for security in universe:
            text(security)
    require(set(scope["execution_universe"]) <= set(scope["prediction_universe"]), "execution universe outside prediction union")
    digest(scope["supported_universe_ref"])
    fields(plan["initial_account"], "cash_minor positions")
    integer(plan["initial_account"]["cash_minor"], 1)
    require(plan["initial_account"]["positions"] == {}, "bounded stock initial holdings must be empty")
    require(plan["stock_action_policy"] == ACTION_POLICY and plan["clock_policy"] == CLOCK_POLICY,
            "stock action/clock policy mismatch")
    require(type(plan["limitations"]) is list and all(type(v) is str for v in plan["limitations"]), "request limitations required")
    profile = plan["profile_input"]
    fields(profile, "artifact profile_ref stock_execution_rules_ref stock_fee_schedule_ref")
    validate_artifact_ref(profile["artifact"])
    for name in ("profile_ref", "stock_execution_rules_ref", "stock_fee_schedule_ref"):
        digest(profile[name])
    market = plan["market_input"]
    fields(market, "contract_version market_ref model_snapshot_id execution_snapshot_id warmup_sessions price_basis projection_version native_inputs")
    require(market["contract_version"] == "stock_market_input_refs_v1" and market["price_basis"] == "unadjusted" and
            market["projection_version"] == "market_replay_v4", "stock market reference contract mismatch")
    for name in ("model_snapshot_id", "execution_snapshot_id"):
        text(market[name])
    warmup = market["warmup_sessions"]
    require(type(warmup) is list and warmup == sorted(set(warmup)), "explicit ordered warmup sessions required")
    for day in warmup:
        session(day)
        require(day < scope["start_session"], "warmup advances account scope")
    require(type(market["native_inputs"]) is list and bool(market["native_inputs"]), "native input references required")
    for item in market["native_inputs"]:
        fields(item, "role artifact native_ref")
        require(item["role"] in ("prediction_basis", "execution"), "unsupported native input role")
        validate_artifact_ref(item["artifact"])
        digest(item["native_ref"])
    require(market["market_ref"] == logical_ref(market, "market_ref"), "market reference mismatch")
    predictions = plan["prediction_input"]
    fields(predictions, "contract_version prediction_ref frames")
    require(predictions["contract_version"] == "stock_prediction_input_refs_v1" and
            type(predictions["frames"]) is list and bool(predictions["frames"]), "original prediction references required")
    for frame in predictions["frames"]:
        fields(frame, "fold_ref fold_spec_ref model_ref feature_ref signal_run_ref fold_spec_artifact model_metadata_artifact prediction_artifact")
        for name in ("fold_ref", "fold_spec_ref", "model_ref", "feature_ref", "signal_run_ref"):
            digest(frame[name])
        for name in ("fold_spec_artifact", "model_metadata_artifact", "prediction_artifact"):
            validate_artifact_ref(frame[name])
    require(predictions["prediction_ref"] == logical_ref(predictions, "prediction_ref"), "prediction reference mismatch")
    require(plan["request_ref"] == logical_ref(plan, "request_ref"), "request reference mismatch")
    return plan
