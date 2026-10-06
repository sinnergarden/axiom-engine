"""Independent wire expectations plus the frozen pre-optimization collector."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from axiom_engine.core.contracts import ContractError, canonical
from axiom_engine.runtime.stock_stream_inputs import _CanonicalIndex, _DecodedMemory
from scanner_reference import ReferenceIndex


def artifact(path, raw):
    content = raw[:-1] if raw.endswith(b"\n") else raw
    return dict(artifact_type="synthetic_scanner", artifact_id="synthetic_scanner",
                contract_version="synthetic_v1", manifest_uri=str(path),
                content_digest="sha256:"+sha256(content).hexdigest())


class BulkScannerTests(unittest.TestCase):
    def compare(self, raw, accepted, *, read_sizes=(1, 2, 3, 5, 7, 17, 64, 4096)):
        # These explicit acceptance cases are independent of either collector.
        if accepted:
            content = raw[:-1] if raw.endswith(b"\n") else raw
            expected = json.loads(content)
            self.assertIs(type(expected), dict)
            self.assertEqual(canonical(expected).encode(), content)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"wire.json"; path.write_bytes(raw)
            for read_size in read_sizes:
                with self.subTest(raw=raw[:80], read_size=read_size):
                    outcomes = []
                    for implementation in (ReferenceIndex, _CanonicalIndex):
                        memory = _DecodedMemory(1024**2)
                        try:
                            index = implementation(artifact(path, raw),
                                dict(max_read_bytes=read_size, max_decoded_bytes=1024**2), memory=memory)
                            outcomes.append((True, index))
                        except (ContractError, UnicodeError) as exc:
                            outcomes.append((False, type(exc), str(exc)))
                        self.assertEqual(memory.temporary_bytes, 0)
                    self.assertEqual(outcomes[0][0], accepted)
                    self.assertEqual(outcomes[1][0], accepted)
                    if accepted:
                        old, new = outcomes[0][1], outcomes[1][1]
                        self.assertEqual(new.content_digest, artifact(path,raw)["content_digest"])
                        self.assertEqual(new.file_digest, "sha256:"+sha256(raw).hexdigest())
                        for name in ("content_digest", "file_digest", "spans", "group_hashes", "_array_counts"):
                            self.assertEqual(getattr(old,name),getattr(new,name),name)
                        self.assertEqual({k:list(v) for k,v in old.groups.items()},
                                         {k:list(v) for k,v in new.groups.items()})
                    else:
                        self.assertEqual(outcomes[0][1:],outcomes[1][1:])

    def test_strings_escapes_utf8_and_every_small_chunk_boundary(self):
        for value in ("", 'quote"slash\\tail', "\\\\\\\"", "line\n\t\r\b\f",
                      "汉字é🙂", "x"*130+"\\"+"🙂"+"\""+"z"*130):
            raw=canonical({"a":value,"é":"尾"}).encode()
            self.compare(raw,True)
            self.compare(raw+b"\n",True)
        for raw in (b'{"a":"\\q"}',b'{"a":"literal\nnewline"}',
                    b'{"a":"\xff"}',b'{"a":"\\u0061"}'):
            self.compare(raw,False)

    def test_numbers_exponents_negative_zero_and_nonfinite(self):
        for token in (b"0",b"-1",b"1.0",b"-0.0",b"1e+20",b"1e-05",b"true",b"false",b"null"):
            self.compare(b'{"n":'+token+b'}',True)
        for token in (b"-0",b"01",b"+1",b"1e20",b"1E+20",b"1e-5",b"1.",b".1",
                      b"1e",b"NaN",b"Infinity",b"-Infinity",b"1e309",b"0 "):
            self.compare(b'{"n":'+token+b'}',False)

    def test_nested_duplicate_keys_and_reserved_unknown(self):
        self.compare(canonical({"a":[{},[],{"b":[1,None,"🙂"]}],"z":True}).encode(),True)
        valid={"contract_type":"Unknown","contract_version":"1","metadata":{},
               "unknown_id":"u","reason":"missing","required_evidence":"original"}
        self.compare(canonical({"a":valid}).encode(),True)
        for raw in (b'{"a":1,"a":2}',b'{"z":0,"a":1}',b'{"a":{"b":1,"b":2}}',
                    b'{"a":{"contract_type":"Other"}}',b'{"a":{"contract_type":"Unknown"}}'):
            self.compare(raw,False)
        self.compare(b'{"a":'+b'['*127+b'0'+b']'*127+b'}',True,read_sizes=(7,4096))
        self.compare(b'{"a":'+b'['*128+b'0'+b']'*128+b'}',False,read_sizes=(7,4096))

    def test_truncation_trailing_bytes_and_delivery_lf(self):
        for raw in (b'',b'{',b'{"a":',b'{"a":"tail\\',b'{"a":[1,',b'{"a":1',
                    b'{}x',b'{}{}',b'{} ',b'{}\n\n',b'{}\r\n',b'[]',b'{ "a":0}',b'{"a":0,}'):
            self.compare(raw,False)
        self.compare(b'{}',True);self.compare(b'{}\n',True)

    def test_rows_keep_offsets_hashes_and_local_views(self):
        wire={"context":{"coverage":{"a":"opaque"}},"records":[
            {"session":"2024-01-02","value":"汉字\\\""},
            {"session":"2024-01-03","value":"next"}]}
        raw=canonical(wire).encode();self.compare(raw,True)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"rows.json";p.write_bytes(raw)
            index=_CanonicalIndex(artifact(p,raw),dict(max_read_bytes=7,max_decoded_bytes=10000))
            with index._memory.stage() as stage:
                rows,_,_=index.rows(("records",),["2024-01-02"],reservation=stage)
                self.assertEqual(rows,[wire["records"][0]])
            self.assertEqual(index.statistics["row_decode_count"],2)
            self.assertEqual(index.statistics["local_decode_count"],1)
            self.assertGreater(index.statistics["reread_bytes"],0)
            self.assertEqual(index._memory.temporary_bytes,0)

    def test_scalar_budget_exact_and_one_byte_less_independent_formula(self):
        # One file chunk + live key + raw/value/canonical copies of the scalar.
        n=37;raw=b'{"value":"'+b'x'*n+b'"}'
        exact=len(raw)+len(b'"value"')+4*(n+2)
        self.assertEqual(exact,212)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"budget.json";p.write_bytes(raw)
            for implementation in (ReferenceIndex,_CanonicalIndex):
                memory=_DecodedMemory(exact)
                implementation(artifact(p,raw),dict(max_read_bytes=len(raw),max_decoded_bytes=exact),memory=memory)
                self.assertEqual(memory.peak_bytes,exact)
                self.assertEqual(memory.temporary_bytes,0)
                memory=_DecodedMemory(exact-1)
                with self.assertRaisesRegex(ContractError,"before temporary growth"):
                    implementation(artifact(p,raw),dict(max_read_bytes=len(raw),max_decoded_bytes=exact-1),memory=memory)
                self.assertEqual(memory.temporary_bytes,0)
                self.assertLessEqual(memory.peak_bytes,exact-1)

    def test_large_scalar_is_bounded_and_does_not_copy_buffer_suffix(self):
        raw=canonical({"a":"x"*(65536+17)+"🙂"}).encode()
        self.compare(raw,True,read_sizes=(17,4096,65536))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"large.json";p.write_bytes(raw)
            index=_CanonicalIndex(artifact(p,raw),dict(max_read_bytes=65536,max_decoded_bytes=1024**2))
            self.assertEqual(index.statistics["scan_read_bytes"],len(raw))
            self.assertEqual(index.statistics["file_hash_bytes"],len(raw))
            self.assertEqual(index.statistics["content_hash_bytes"],len(raw))
            self.assertEqual(index.statistics["scalar_count"],2)
            self.assertLess(index.statistics["scalar_batches"],10)
            self.assertLessEqual(index._memory.peak_bytes,1024**2)
            with self.assertRaisesRegex(ContractError,"before temporary growth"):
                _CanonicalIndex(artifact(p,raw),dict(max_read_bytes=65536,max_decoded_bytes=65536))

    def test_row_capture_budget_boundary_matches_frozen_collector(self):
        raw=canonical({"records":[{"session":"2024-01-02","value":"x"*47}]}).encode()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"capture.json";p.write_bytes(raw)
            minimum=[]
            for implementation in (ReferenceIndex,_CanonicalIndex):
                found=None
                for budget in range(17,1000):
                    memory=_DecodedMemory(budget)
                    try: implementation(artifact(p,raw),dict(max_read_bytes=17,max_decoded_bytes=budget),memory=memory)
                    except ContractError:
                        self.assertEqual(memory.temporary_bytes,0);continue
                    found=budget;break
                self.assertIsNotNone(found);minimum.append(found)
            self.assertEqual(minimum[0],minimum[1])

if __name__ == "__main__":
    unittest.main()
