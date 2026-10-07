"""Exact lexical/canonical rejection and bounded reuse, without real sources."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from axiom_engine.core.contracts import ContractError, canonical
from axiom_engine.runtime.stock_stream_inputs import _CanonicalIndex, _DecodedMemory


def indexed(root, raw, cache_bytes, *, read_bytes=7, decoded_bytes=4096):
    path = root/"scalar.json"; path.write_bytes(raw)
    artifact = dict(artifact_type="scalar", artifact_id="scalar", contract_version="1",
        manifest_uri=str(path), content_digest="sha256:"+hashlib.sha256(raw.removesuffix(b"\n")).hexdigest())
    memory = _DecodedMemory(decoded_bytes, scalar_cache_bytes=cache_bytes)
    result = _CanonicalIndex(artifact, dict(max_read_bytes=read_bytes, max_decoded_bytes=decoded_bytes), memory=memory)
    return result, memory


class StockScalarCacheTests(unittest.TestCase):
    def test_public_default_admission_has_no_cache_and_explicit_opt_in_preserves_owned_input(self):
        from axiom_engine.runtime import StockInputSource
        from test_stock_owned_inputs import imported, request_for
        with tempfile.TemporaryDirectory() as tmp:
            manifest = request_for(Path(tmp))
            with imported(manifest, source=StockInputSource()) as default, imported(
                    manifest, source=StockInputSource(scalar_cache_bytes=262144)) as opt_in:
                a, b = default.statistics, opt_in.statistics
                self.assertEqual(a['source_operations']['scalar_cache_hits'], 0)
                self.assertEqual(a['source_scalar_cache_evictions'], 0)
                self.assertEqual(a['source_operations']['scalar_decode_count'],
                                 a['source_operations']['scalar_count'])
                self.assertGreater(b['source_operations']['scalar_cache_hits'], 0)
                self.assertEqual(a['owned_bytes'], b['owned_bytes'])
                self.assertEqual(default.inventory(manifest), opt_in.inventory(manifest))

    def test_utf8_escapes_numbers_null_and_chunk_boundaries_preserve_original_scan(self):
        values = [None, True, False, 0, -1, 1.0, -0.0, 1e-20, '重复😀', '\\"\n\t/', "x"*300]
        raw = canonical({"context": {"coverage": {"items": values*30}}, "records": []}).encode()+b"\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for read_bytes in (1, 7, 64, 4096):
                with self.subTest(read_bytes=read_bytes):
                    old, _ = indexed(root, raw, 0, read_bytes=read_bytes, decoded_bytes=16384)
                    new, memory = indexed(root, raw, 4096, read_bytes=read_bytes, decoded_bytes=16384)
                    for name in ("content_digest", "file_digest", "spans", "groups", "group_hashes"):
                        self.assertEqual(getattr(old, name), getattr(new, name), name)
                    self.assertGreater(new.statistics["scalar_cache_hits"], 100)
                    self.assertEqual(new.statistics["scalar_count"],
                        new.statistics["scalar_cache_hits"]+new.statistics["scalar_decode_count"])
                    self.assertLessEqual(memory.scalar_cache.used, 4096)
                    self.assertEqual(memory.temporary_bytes, 0)

    def test_noncanonical_duplicate_order_numeric_and_reserved_unknown_are_never_cached_as_valid(self):
        unknown = {"contract_type": "Unknown", "contract_version": "1", "metadata": {},
            "unknown_id": "same", "reason": "same", "required_evidence": "same"}
        invalid_unknown = dict(unknown, metadata=[])
        raws = [b'{"a":1,"a":1}', b'{"a":1,"b":1.00}', b'{"a":0,"b":-0}',
            b'{"a":"same","b":"s\\u0061me"}', b'{"a":1, "b":1}',
            b'{"b":1,"a":1}', b'{"a":1,"b":NaN}', b'{"a":1,"b":Infinity}',
            json.dumps({"a": unknown, "b": invalid_unknown}, sort_keys=True, separators=(",", ":")).encode()]
        with tempfile.TemporaryDirectory() as tmp:
            for raw in raws:
                for cache in (0, 1024):
                    with self.subTest(raw=raw, cache=cache), self.assertRaises(ContractError):
                        indexed(Path(tmp), raw, cache)

    def test_cache_eviction_yields_to_required_reservations_and_preserves_small_budget(self):
        memory = _DecodedMemory(512, scalar_cache_bytes=256)
        memory.scalar_cache.put(b'"same"', "same")
        with memory.stage() as stage:
            stage.reserve(512)
            self.assertEqual(memory.scalar_cache.used, 0)
        self.assertEqual(memory.available, 512)
        raw = canonical({"context": {"coverage": {"items": [str(i) for i in range(100)]}}, "records": []}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            old, _ = indexed(Path(tmp), raw, 0, decoded_bytes=512)
            new, memory = indexed(Path(tmp), raw, 256, decoded_bytes=512)
            self.assertEqual(new.content_digest, old.content_digest)
            self.assertLessEqual(memory.peak_bytes, 512)
            self.assertGreater(memory.scalar_cache.evictions, 10)

    def test_optional_literal_key_does_not_require_more_scalar_headroom(self):
        raw = canonical({"a": "x"*200}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            for limit in (850, 900):
                old, _ = indexed(Path(tmp), raw, 0, read_bytes=16, decoded_bytes=limit)
                new, memory = indexed(Path(tmp), raw, 4096, read_bytes=16, decoded_bytes=limit)
                self.assertEqual(old.content_digest, new.content_digest)
                self.assertLessEqual(memory.peak_bytes, limit)


if __name__ == "__main__":
    unittest.main()
