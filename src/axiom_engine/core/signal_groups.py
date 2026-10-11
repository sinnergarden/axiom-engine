"""Pure cross-sectional groups over the same admitted pairs as signal statistics."""
import math

from .contracts import Document, fields, integer, number, require, session, text
from .signal_statistics import _ranks


class SignalGroups(Document):
    """Raw Label group returns; these are descriptive observations, not an account."""


_POLICIES = {"rank_ties": "average", "group_assignment": "average_rank_fraction",
             "time_weighting": "equal_valid_sessions", "return_semantics": "raw_label_return",
             "annualization": "none"}


def _mean(values):
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if not scale:
        return 0.0
    # Scaling keeps even admitted finite extremes from overflowing their sum.
    fraction = math.fsum(value / scale for value in values) / len(values)
    return max(-1.0, min(1.0, fraction)) * scale


def evaluate_signal_groups(input, *, spec):
    """Evaluate a frozen paired sample without I/O or upstream recomputation.

    Sessions are explicit observation dates, whether daily or weekly. Within each
    date, ascending average ranks assign bin floor((rank - 1) / n * group_count).
    Ties stay together; bin sizes need not match and empty bins remain null. The
    last bin is the highest-score group. Insufficient or constant-score dates
    retain assignment counts but have null returns. Summaries weight each valid
    observed date equally, using each bin's own valid-date coverage.
    """
    wire = input.to_dict() if isinstance(input, Document) else input
    policy = spec.to_dict() if isinstance(spec, Document) else spec
    fields(wire, "contract_version sessions signal_keys pairs")
    require(wire["contract_version"] == "signal_statistics_input_v1", "Signal groups input version")
    fields(policy, "group_count minimum_pairs rank_ties group_assignment time_weighting return_semantics annualization")
    integer(policy["group_count"], 2)
    integer(policy["minimum_pairs"], 2)
    require({key: policy[key] for key in _POLICIES} == _POLICIES, "Unsupported signal groups policy")
    days, signals = wire["sessions"], wire["signal_keys"]
    require(type(days) is list and bool(days), "Nonempty ordered group sessions required")
    for day in days:
        session(day)
    require(days == sorted(set(days)), "Group sessions must be unique and ordered")
    require(type(signals) is list and bool(signals), "Nonempty ordered signal keys required")
    for key in signals:
        text(key)
    require(len(set(signals)) == len(signals), "Duplicate signal key")
    require(type(wire["pairs"]) is list, "Group pairs must be a list")
    day_set, signal_set = set(days), set(signals)
    pairs, seen, by_date = [], set(), {}
    for pair in wire["pairs"]:
        fields(pair, "signal_key security_id session score outcome")
        text(pair["security_id"])
        require(pair["signal_key"] in signal_set and pair["session"] in day_set, "Pair outside group axes")
        number(pair["score"])
        number(pair["outcome"])
        key = (pair["signal_key"], pair["session"], pair["security_id"])
        require(key not in seen, "Duplicate group pair key")
        seen.add(key)
        pairs.append(dict(pair))
    pairs.sort(key=lambda pair: (pair["signal_key"], pair["session"], pair["security_id"]))
    for pair in pairs:
        by_date.setdefault((pair["signal_key"], pair["session"]), []).append(pair)
    canonical_input = {**wire, "pairs": pairs}
    group_count = policy["group_count"]
    series, summaries = [], []
    for signal_key in signals:
        group_values = [[] for _ in range(group_count)]
        pools, excesses = [], []
        for day in days:
            sample = by_date.get((signal_key, day), [])
            scores = [pair["score"] for pair in sample]
            outcomes = [pair["outcome"] for pair in sample]
            bins = [[] for _ in range(group_count)]
            for rank, outcome in zip(_ranks(scores), outcomes):
                # Average ranks are integer or half-integer; exact division
                # avoids moving a boundary rank down a bin through float error.
                index = min(group_count - 1, (int(2 * rank) - 2) * group_count // (2 * len(sample)))
                bins[index].append(outcome)
            reason = "INSUFFICIENT_PAIRS" if len(sample) < policy["minimum_pairs"] else None
            if reason is None and min(scores) == max(scores):
                reason = "CONSTANT_CROSS_SECTION"
            pool = _mean(outcomes) if reason is None else None
            groups = [{"group": i + 1, "count": len(values),
                       "mean_return": _mean(values) if reason is None else None}
                      for i, values in enumerate(bins)]
            top = groups[-1]["mean_return"]
            excess, excess_reason = None, reason
            if reason is None:
                excess_reason = "EMPTY_TOP_GROUP" if top is None else None
                if top is not None:
                    difference = top - pool
                    if math.isfinite(difference):
                        excess = difference
                    else:
                        excess_reason = "NON_FINITE_TOP_EXCESS"
                pools.append(pool)
                for values, group in zip(group_values, groups):
                    if group["mean_return"] is not None:
                        values.append(group["mean_return"])
                if excess is not None:
                    excesses.append(excess)
            series.append({"signal_key": signal_key, "session": day,
                           "valid_pair_count": len(sample), "pool_mean_return": pool,
                           "groups": groups, "top_group_excess_return": excess,
                           "top_group_excess_reason": excess_reason, "reason": reason})
        summaries.append({"signal_key": signal_key, "session_count": len(days),
                          "valid_session_count": len(pools), "mean_pool_return": _mean(pools),
                          "groups": [{"group": i + 1, "valid_session_count": len(values),
                                      "mean_return": _mean(values)}
                                     for i, values in enumerate(group_values)],
                          "top_group_excess_valid_session_count": len(excesses),
                          "mean_top_group_excess_return": _mean(excesses)})
    result = {"contract_version": "signal_groups_v1",
              "input_ref": Document.from_dict(canonical_input).identity,
              "spec_ref": Document.from_dict(policy).identity,
              "series": series, "summary": summaries}
    result["groups_ref"] = Document.from_dict(result).identity
    return SignalGroups.from_dict(result)
