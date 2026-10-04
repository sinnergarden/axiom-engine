import importlib.util
from pathlib import Path
import unittest

from axiom_engine._implementation import IMPLEMENTATION_REF


class ImplementationTests(unittest.TestCase):
    def test_source_binding_matches_published_implementation_ref(self):
        path = Path(__file__).resolve().parents[1] / 'tools/freeze_implementation.py'
        spec = importlib.util.spec_from_file_location('freeze_implementation', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(IMPLEMENTATION_REF, module.implementation_ref(),
                         'Regenerate the frozen implementation ref after source changes')
