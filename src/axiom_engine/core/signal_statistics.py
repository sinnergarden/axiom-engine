"""Pure saved-signal statistics; Research supplies the admitted paired sample."""
import math

from .contracts import Document, fields, number, require, session, text


class SignalStatistics(Document):
    """Bound daily correlations and equally weighted descriptive summaries."""


_SPEC = {"minimum_pairs": 20, "rank_ties": "average", "std_ddof": 1,
         "time_weighting": "equal_valid_sessions", "annualization": "none"}


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in order[start:end]:
            ranks[i] = rank
        start = end
    return ranks


def _correlation(x, y):
    # Translate first to preserve small representable gaps at a large offset.
    # Opposite finite extremes may overflow subtraction; only then scale first.
    centered = []
    for values in (x, y):
        shifted = [v - values[0] for v in values]
        if not all(math.isfinite(v) for v in shifted):
            scale = max(abs(v) for v in values)
            shifted = [v / scale for v in values]
        scale = max(abs(v) for v in shifted)
        scaled = [v / scale if scale else 0.0 for v in shifted]
        mean = math.fsum(scaled) / len(scaled)
        centered.append([v - mean for v in scaled])
    x, y = centered
    xx, yy = math.fsum(v * v for v in x), math.fsum(v * v for v in y)
    if xx == 0 or yy == 0:
        return None, "CONSTANT_CROSS_SECTION"
    value = math.fsum(a * b for a, b in zip(x, y)) / math.sqrt(xx * yy)
    if not math.isfinite(value):
        return None, "NON_FINITE_CORRELATION"
    return max(-1.0, min(1.0, value)), None


def _summary(values):
    if not values:
        return None, None, None, "NO_VALID_SESSIONS"
    mean = math.fsum(values) / len(values)
    if len(values) < 2:
        return mean, None, None, "INSUFFICIENT_VALID_SESSIONS"
    if min(values) == max(values):
        return mean, 0.0, None, "ZERO_VARIANCE"
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
    if std == 0:
        return mean, std, None, "ZERO_VARIANCE"
    return mean, std, mean / std, None


def evaluate_signal_statistics(input, *, spec):
    """Evaluate exact v1 paired input without I/O, source or account execution.

    Pair order does not affect identity. Ordered session/signal axes are retained,
    including empty dates. The first version accepts only the frozen five policies.
    """
    wire = input.to_dict() if isinstance(input, Document) else input
    policy = spec.to_dict() if isinstance(spec, Document) else spec
    fields(wire, "contract_version sessions signal_keys pairs")
    require(wire["contract_version"] == "signal_statistics_input_v1", "Signal statistics input version")
    fields(policy, "minimum_pairs rank_ties std_ddof time_weighting annualization")
    require(type(policy["minimum_pairs"]) is int and type(policy["std_ddof"]) is int
            and policy == _SPEC, "Unsupported signal statistics policy")
    days, signals = wire["sessions"], wire["signal_keys"]
    require(type(days) is list and bool(days), "Nonempty ordered statistics sessions required")
    for day in days:
        session(day)
    require(days == sorted(set(days)), "Statistics sessions must be unique and ordered")
    require(type(signals) is list and bool(signals), "Nonempty ordered signal keys required")
    for key in signals:
        text(key)
    require(len(set(signals)) == len(signals), "Duplicate signal key")
    require(type(wire["pairs"]) is list, "Statistics pairs must be a list")
    groups, seen, pairs = {}, set(), []
    for pair in wire["pairs"]:
        fields(pair, "signal_key security_id session score outcome")
        text(pair["security_id"])
        require(pair["signal_key"] in signals and pair["session"] in days, "Pair outside statistics axes")
        number(pair["score"]); number(pair["outcome"])
        key = (pair["signal_key"], pair["session"], pair["security_id"])
        require(key not in seen, "Duplicate statistics pair key")
        seen.add(key)
        pairs.append(dict(pair))
    pairs.sort(key=lambda p: (p["signal_key"], p["session"], p["security_id"]))
    for pair in pairs:
        groups.setdefault((pair["signal_key"], pair["session"]), []).append(pair)
    canonical_input = {**wire, "pairs": pairs}
    series, summaries = [], []
    for signal_key in signals:
        valid_ic, valid_rank = [], []
        for day in days:
            group = groups.get((signal_key, day), [])
            ic = rank_ic = None
            reason = "INSUFFICIENT_PAIRS"
            if len(group) >= policy["minimum_pairs"]:
                scores = [p["score"] for p in group]
                outcomes = [p["outcome"] for p in group]
                ic, reason = _correlation(scores, outcomes)
                rank_ic, rank_reason = _correlation(_ranks(scores), _ranks(outcomes))
                reason = reason or rank_reason
            series.append({"signal_key": signal_key, "session": day,
                           "valid_pair_count": len(group), "ic": ic,
                           "rank_ic": rank_ic, "reason": reason})
            if ic is not None:
                valid_ic.append(ic)
            if rank_ic is not None:
                valid_rank.append(rank_ic)
        mean, std, ir, reason = _summary(valid_ic)
        rank_mean, rank_std, rank_ir, rank_reason = _summary(valid_rank)
        summaries.append({"signal_key": signal_key, "valid_ic_session_count": len(valid_ic),
                          "mean_ic": mean, "ic_std": std, "icir": ir, "icir_reason": reason,
                          "valid_rank_ic_session_count": len(valid_rank), "mean_rank_ic": rank_mean,
                          "rank_ic_std": rank_std, "rank_icir": rank_ir, "rank_icir_reason": rank_reason})
    result = {"contract_version": "signal_statistics_v1",
              "input_ref": Document.from_dict(canonical_input).identity,
              "spec_ref": Document.from_dict(policy).identity,
              "series": series, "summary": summaries}
    result["statistics_ref"] = Document.from_dict(result).identity
    return SignalStatistics.from_dict(result)
