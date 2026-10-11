"""Hand-computed groups, ties, coverage and immutable paired-input bindings."""
from copy import deepcopy
import math
import unittest

from axiom_engine.core.contracts import ContractError, Document
from axiom_engine.core.signal_groups import SignalGroups, evaluate_signal_groups
from axiom_engine.core.signal_statistics import evaluate_signal_statistics


SPEC = {"group_count": 10, "minimum_pairs": 20, "rank_ties": "average",
        "group_assignment": "average_rank_fraction", "time_weighting": "equal_valid_sessions",
        "return_semantics": "raw_label_return", "annualization": "none"}
DAYS = ["2023-01-03", "2023-01-10", "2023-01-17"]


def wire(samples):
    return {"contract_version": "signal_statistics_input_v1", "sessions": DAYS,
            "signal_keys": ["model"], "pairs": [
                {"signal_key": "model", "session": day, "security_id": f"s{i:02}",
                 "score": score, "outcome": outcome}
                for day, values in zip(DAYS, samples)
                for i, (score, outcome) in enumerate(values)]}


class SignalGroupsTests(unittest.TestCase):
    def evaluate(self, input, spec=SPEC):
        return evaluate_signal_groups(input, spec=spec).to_dict()

    def test_hand_returns_and_equal_date_weighting(self):
        result = self.evaluate(wire([[(i, i / 100) for i in range(20)],
                                     [(i, i / 100) for i in range(30)]]))
        first, second, empty = result["series"]
        self.assertEqual([group["count"] for group in first["groups"]], [2] * 10)
        for i, group in enumerate(first["groups"]):
            self.assertAlmostEqual(group["mean_return"], .005 + .02 * i)
        self.assertAlmostEqual(first["pool_mean_return"], .095)
        self.assertAlmostEqual(first["top_group_excess_return"], .09)
        self.assertAlmostEqual(second["top_group_excess_return"], .135)
        summary = result["summary"][0]
        self.assertAlmostEqual(summary["groups"][-1]["mean_return"], (.185 + .28) / 2)
        self.assertAlmostEqual(summary["mean_top_group_excess_return"], (.09 + .135) / 2)
        self.assertEqual(summary["valid_session_count"], 2)
        self.assertEqual(summary["top_group_excess_valid_session_count"], 2)
        self.assertEqual(empty["valid_pair_count"], 0)
        self.assertIsNone(empty["pool_mean_return"])

    def test_ties_stay_together_and_empty_bins_are_null(self):
        input = wire([[(0, 0)] * 18 + [(1, .2), (1, .4)]])
        first = self.evaluate(input)["series"][0]
        self.assertEqual([group["count"] for group in first["groups"]], [0, 0, 0, 0, 18, 0, 0, 0, 0, 2])
        self.assertIsNone(first["groups"][0]["mean_return"])
        self.assertEqual(first["groups"][4]["mean_return"], 0)
        self.assertAlmostEqual(first["groups"][9]["mean_return"], .3)
        self.assertAlmostEqual(first["top_group_excess_return"], .27)
        # A wide high-score tie need not occupy the last bin at all.
        input = wire([[(0, 0)] * 2 + [(1, .1)] * 18])
        result = self.evaluate(input)
        self.assertIsNone(result["series"][0]["top_group_excess_return"])
        self.assertEqual(result["series"][0]["top_group_excess_reason"], "EMPTY_TOP_GROUP")
        self.assertEqual(result["summary"][0]["top_group_excess_valid_session_count"], 0)

    def test_insufficient_and_constant_dates_preserve_counts_not_returns(self):
        result = self.evaluate(wire([[(i, i / 100) for i in range(19)], [(1, .1)] * 20]))
        self.assertEqual([row["reason"] for row in result["series"]],
                         ["INSUFFICIENT_PAIRS", "CONSTANT_CROSS_SECTION", "INSUFFICIENT_PAIRS"])
        for row, count in zip(result["series"], [19, 20, 0]):
            self.assertEqual(sum(group["count"] for group in row["groups"]), count)
            self.assertTrue(all(group["mean_return"] is None for group in row["groups"]))
            self.assertIsNone(row["pool_mean_return"])
        self.assertEqual(result["summary"][0]["valid_session_count"], 0)
        self.assertIsNone(result["summary"][0]["mean_top_group_excess_return"])

    def test_same_input_binding_order_identity_and_admission(self):
        input = wire([[(i, i / 100) for i in range(20)]])
        result = self.evaluate(input)
        reversed_input = deepcopy(input)
        reversed_input["pairs"].reverse()
        output = evaluate_signal_groups(Document.from_dict(reversed_input), spec=Document.from_dict(SPEC))
        self.assertIsInstance(output, SignalGroups)
        self.assertEqual(output.to_dict(), result)
        stats = evaluate_signal_statistics(input, spec={"minimum_pairs": 20, "rank_ties": "average",
                                                       "std_ddof": 1, "time_weighting": "equal_valid_sessions",
                                                       "annualization": "none"}).to_dict()
        self.assertEqual(result["input_ref"], stats["input_ref"])
        self.assertEqual(result["spec_ref"], Document.from_dict(SPEC).identity)
        unsigned = result.copy()
        unsigned.pop("groups_ref")
        self.assertEqual(result["groups_ref"], Document.from_dict(unsigned).identity)
        for key, value in (("score", math.nan), ("outcome", True), ("session", "2023-01-18")):
            bad = deepcopy(input)
            bad["pairs"][0][key] = value
            with self.assertRaises(ContractError):
                self.evaluate(bad)
        bad = deepcopy(input)
        bad["pairs"].append(bad["pairs"][0])
        with self.assertRaises(ContractError):
            self.evaluate(bad)
        with self.assertRaises(ContractError):
            self.evaluate(input, {**SPEC, "group_count": True})

    def test_finite_extreme_outcomes_keep_finite_group_means(self):
        input = wire([[(i, 1e308) for i in range(20)]])
        result = self.evaluate(input)
        first = result["series"][0]
        self.assertEqual(first["pool_mean_return"], 1e308)
        self.assertEqual(first["top_group_excess_return"], 0)
        self.assertTrue(all(group["mean_return"] == 1e308 for group in first["groups"]))
