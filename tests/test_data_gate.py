import unittest

from axiom_engine.runtime import DecisionBatchGate, InputGateError, stale_price_mark


class Query:
    purpose = "decision_facts"
    price_basis = "unadjusted"
    sessions = ("2025-01-02",)
    cutoff_by_session = {"2025-01-02": "2025-01-02T15:00:00Z"}


class Batch:
    def __init__(self, purpose, snapshot):
        self.purpose, self.snapshot = purpose, snapshot

    def to_json(self):
        return {"context": {"snapshot_id": self.snapshot,
                            "query": {"purpose": self.purpose}}}


class Data:
    def __init__(self):
        self.current = "s1"
        self.calls = []

    def resolve(self, reference):
        return self.current

    def read(self, *, snapshot, query):
        self.calls.append((snapshot, query.purpose))
        return Batch(query.purpose, snapshot)

    read_market = read


class GateTests(unittest.TestCase):
    def test_session_pin_purpose_and_clock(self):
        data = Data()
        gate = DecisionBatchGate(data)
        gate.pin_session("2025-01-02")
        data.current = "s2"
        gate.read_decision(decision_session="2025-01-02", query=Query(),
                           clock="2025-01-02T16:00:00Z")
        self.assertEqual(gate.references[0].snapshot_id, "s1")
        with self.assertRaisesRegex(InputGateError, "pinned"):
            gate.pin_session("2025-01-02")
        with self.assertRaisesRegex(InputGateError, "future"):
            gate.read_decision(decision_session="2025-01-02", query=Query(),
                               clock="2025-01-02T09:00:00Z")
        query = Query()
        query.purpose = "label_outcomes"
        with self.assertRaisesRegex(InputGateError, "decision_facts"):
            gate.read_decision(decision_session="2025-01-02", query=query,
                               clock="2025-01-02T16:00:00Z")
        replay = Query()
        replay.purpose = "market_replay"
        gate.read_market(decision_session="2025-01-02", query=replay,
                         clock="2025-01-02T16:00:00Z")
        self.assertEqual({ref.purpose for ref in gate.references},
                         {"decision_facts", "market_replay"})

    def test_stale_price_mark_keeps_observation_time(self):
        marked = stale_price_mark(10, price_session="2025-01-02",
            valuation_session="2025-01-06",
            calendar_sessions=("2025-01-02", "2025-01-03", "2025-01-06"), source_ref="s1")
        self.assertEqual((marked["price"], marked["stale_sessions"], marked["is_stale"]),
                         (10.0, 2, True))


if __name__ == "__main__":
    unittest.main()
