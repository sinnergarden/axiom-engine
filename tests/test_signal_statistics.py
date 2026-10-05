"""Independent synthetic golden cases for the shared Core statistic operator."""
from copy import deepcopy
import math
import unittest

from axiom_engine.core import ContractError, evaluate_signal_statistics
from axiom_engine.core.contracts import Document

SPEC = {"minimum_pairs": 20, "rank_ties": "average", "std_ddof": 1,
        "time_weighting": "equal_valid_sessions", "annualization": "none"}
DAYS = ["2024-01-02", "2024-01-03", "2024-01-04"]


def input_wire():
    pairs = []
    for key in ("a", "b"):
        for day, sign in zip(DAYS[:2], (1, -1)):
            for i in range(20):
                pairs.append({"signal_key": key, "session": day, "security_id": f"s{i:02}",
                              "score": i, "outcome": sign * i if key == "a" else i})
    return {"contract_version": "signal_statistics_input_v1", "sessions": DAYS,
            "signal_keys": ["b", "a"], "pairs": pairs}


class SignalStatisticsTests(unittest.TestCase):
    def evaluate(self, wire=None, spec=None):
        return evaluate_signal_statistics(wire or input_wire(), spec=spec or SPEC).to_dict()

    def test_multisignal_equal_day_sample_std_empty_date_and_refs(self):
        result = self.evaluate()
        self.assertEqual([r["ic"] for r in result["series"]], [1, 1, None, 1, -1, None])
        self.assertEqual(result["series"][2]["valid_pair_count"], 0)
        self.assertEqual(result["series"][2]["reason"], "INSUFFICIENT_PAIRS")
        b, a = result["summary"]
        self.assertEqual(b["icir_reason"], "ZERO_VARIANCE")
        self.assertEqual(a["valid_ic_session_count"], 2)
        self.assertEqual(a["mean_ic"], 0)
        self.assertAlmostEqual(a["ic_std"], math.sqrt(2), places=14)
        self.assertEqual(a["icir"], 0)
        self.assertEqual(a["rank_ic_std"], a["ic_std"])
        wire = input_wire(); wire["pairs"].reverse()
        self.assertEqual(self.evaluate(wire), result)
        self.assertEqual(result["spec_ref"], Document.from_dict(SPEC).identity)
        unsigned = result.copy(); unsigned.pop("statistics_ref")
        self.assertEqual(result["statistics_ref"], Document.from_dict(unsigned).identity)

    def test_average_ties_golden_and_constant_cross_section(self):
        wire = input_wire(); wire["signal_keys"] = ["a"]
        # Repeated base pattern produces average ranks. Pearson=Spearman=1/sqrt(2).
        wire["pairs"] = [{"signal_key": "a", "session": DAYS[0], "security_id": f"s{i:02}",
                           "score": (0, 0, 1, 1)[i % 4], "outcome": (0, 1, 1, 2)[i % 4]}
                          for i in range(20)]
        result = self.evaluate(wire)
        self.assertAlmostEqual(result["series"][0]["ic"], 1/math.sqrt(2), places=14)
        self.assertAlmostEqual(result["series"][0]["rank_ic"], 1/math.sqrt(2), places=14)
        self.assertEqual(result["summary"][0]["icir_reason"], "INSUFFICIENT_VALID_SESSIONS")
        for pair in wire["pairs"]:
            pair["score"] = 1
        result = self.evaluate(wire)
        self.assertEqual(result["series"][0]["reason"], "CONSTANT_CROSS_SECTION")
        self.assertIsNone(result["summary"][0]["mean_ic"])
        self.assertEqual(result["summary"][0]["icir_reason"], "NO_VALID_SESSIONS")

    def test_duplicate_outside_session_finite_values_and_fixed_policy(self):
        base = input_wire()
        mutations = []
        wire = deepcopy(base); wire["pairs"].append(wire["pairs"][0]); mutations.append(wire)
        wire = deepcopy(base); wire["pairs"][0]["session"] = "2024-01-05"; mutations.append(wire)
        for value in (None, True, math.inf, math.nan):
            wire = deepcopy(base); wire["pairs"][0]["score"] = value; mutations.append(wire)
        for wire in mutations:
            with self.assertRaises(ContractError): self.evaluate(wire)
        for key, value in (("minimum_pairs", True), ("minimum_pairs", 19), ("std_ddof", True),
                           ("annualization", "sqrt_252"), ("rank_ties", "first")):
            policy = {**SPEC, key: value}
            with self.assertRaises(ContractError): self.evaluate(spec=policy)

    def test_one_day_shift_changes_identity_and_preserves_missing_coverage(self):
        wire = input_wire()
        old = self.evaluate(wire)
        for pair in wire["pairs"]:
            if pair["signal_key"] == "a" and pair["session"] == DAYS[0]: pair["session"] = DAYS[2]
        new = self.evaluate(wire)
        self.assertNotEqual(new["input_ref"], old["input_ref"])
        self.assertEqual(new["series"][3]["valid_pair_count"], 0)
        self.assertEqual(new["series"][5]["valid_pair_count"], 20)

    def test_finite_extreme_values_and_insufficient_sample(self):
        wire = input_wire()
        for pair in wire["pairs"]:
            pair["score"] = (pair["score"] - 10) * 1e306
            pair["outcome"] = (pair["outcome"] - 10) * 1e306
        result = self.evaluate(wire)
        self.assertEqual(result["series"][0]["ic"], 1)
        wire["pairs"] = wire["pairs"][:19]
        result = self.evaluate(wire)
        self.assertEqual(result["series"][3]["valid_pair_count"], 19)
        self.assertIsNone(result["series"][3]["ic"])

    def test_affine_offset_and_opposite_extremes_preserve_correlation(self):
        for scores in ([1e12 + i * math.ulp(1e12) for i in range(20)],
                       [1e16 + 2 * i for i in range(20)],
                       [-1e308] * 10 + [1e308] * 10):
            wire = {"contract_version": "signal_statistics_input_v1", "sessions": DAYS,
                    "signal_keys": ["a"], "pairs": [
                        {"signal_key": "a", "session": DAYS[0], "security_id": f"s{i:02}",
                         "score": value, "outcome": i if scores[0] != -1e308 else (i >= 10) * 1.0}
                        for i, value in enumerate(scores)]}
            result = self.evaluate(wire)
            self.assertAlmostEqual(result["series"][0]["ic"], 1, delta=1e-12)
            self.assertAlmostEqual(result["series"][0]["rank_ic"], 1, delta=1e-12)
