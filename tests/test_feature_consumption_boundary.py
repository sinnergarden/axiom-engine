"""Small consumption leases: unchanged bytes, mutation rejection and linear checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from axiom_engine.feature_library import FeatureLibrary
from axiom_engine.feature_library import library as implementation
from test_incremental_feature_library import FixedSource
from test_conformance import setup, node, col


class FeatureConsumptionBoundaryTests(unittest.TestCase):
    def fixture(self, root, days=4):
        source=FixedSource()
        source.p,source.f,source.c=setup(
            {s:[[i*scale,i*scale*10] for i in range(1,days+3)]
             for s,scale in [('A',1),('B',2),('C',3)]},
            [node('raw_x','identity',['x'])],['raw_x'],
            schema=[col('x',stage='fact'),col('y',stage='fact')])
        library=FeatureLibrary(root,maximum_working_bytes=2**20)
        descriptor,_=library.ensure_features(source.request(['raw_x']),source)
        return library,descriptor,source.c['sessions'][2:]

    def value_path(self, view, day):
        part=view._partition(day,'raw_x')
        return Path(part['buffers']['values']['path'])

    def test_cached_projection_reuses_hashes_and_owns_exact_output_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            library,descriptor,days=self.fixture(root)
            with library.open_feature_view(descriptor) as view:
                keys=[(s,days[0]) for s in ('C','A')]
                first=view.project(keys); hashes=view._store.metrics['file_hash_calls']
                again=view.project(keys)
                self.assertEqual(hashes,view._store.metrics['file_hash_calls'])
                self.assertEqual(first['view_ref'],again['view_ref'])
                self.assertEqual(first['partition_refs'],again['partition_refs'])
                for name in implementation.ARRAYS:
                    self.assertEqual(first[name].tobytes(),again[name].tobytes())
                    owner=again[name]
                    while isinstance(owner,np.ndarray): owner=owner.base
                    self.assertIsInstance(owner,bytes)
                    self.assertFalse(again[name].flags.writeable)
                self.assertEqual(again['values'][:,0].tolist(),[9,3])

    def test_cache_hit_checks_control_partition_and_buffer_original_epochs(self):
        for target in ('control','partition','buffer'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as root:
                library,descriptor,days=self.fixture(root)
                view=library.open_feature_view(descriptor)
                view.project([('A',days[0])])
                if target=='control': path=Path(descriptor['path'])
                elif target=='partition': path=Path(view.column_receipts(days[0])['raw_x']['partition']['path'])
                else: path=self.value_path(view,days[0])
                path.write_bytes(path.read_bytes())
                with self.assertRaisesRegex(ValueError,'saved matrix source changed'):
                    view.project([('A',days[0])])
                self.assertTrue(view._store._invalid)
                with self.assertRaisesRegex(ValueError,'saved matrix source changed'): view.describe()
                view.close()

    def test_mutation_during_copy_or_before_publication_revokes_lease(self):
        for during_copy in (True,False):
            with self.subTest(during_copy=during_copy),tempfile.TemporaryDirectory() as root:
                library,descriptor,days=self.fixture(root)
                view=library.open_feature_view(descriptor); view.project([('A',days[0])])
                path=self.value_path(view,days[0]); original=path.read_bytes()
                if during_copy:
                    actual_paths=implementation._partition_buffer_paths
                    def mutate_after_registration(part):
                        yield from actual_paths(part)
                        path.write_bytes(original)
                    with patch.object(implementation,'_partition_buffer_paths',mutate_after_registration), \
                         self.assertRaisesRegex(ValueError,'saved matrix source changed'):
                        view.project([('A',days[0])])
                else:
                    with self.assertRaisesRegex(ValueError,'saved matrix source changed'):
                        with view.operation():
                            cells=view.project([('A',days[0])])
                            path.write_bytes(original)
                            self.assertEqual(cells['values'][0,0],3)
                            view.validate_boundary()
                self.assertTrue(view._store._invalid); view.close()

    def test_unselected_history_is_checked_at_full_publication_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            library,descriptor,days=self.fixture(root)
            view=library.open_feature_view(descriptor)
            view.project([('A',days[0])]); view.project([('A',days[-1])])
            path=self.value_path(view,days[0]); path.write_bytes(path.read_bytes())
            self.assertEqual(view.project([('A',days[-1])])['values'][0,0],6)
            with self.assertRaisesRegex(ValueError,'saved matrix source changed'): view.validate_boundary()
            view.close()

    def test_daily_checks_grow_with_consumed_days_and_cache_hits_stay_constant(self):
        counts=[]
        for count in (8,16):
            with tempfile.TemporaryDirectory() as root:
                library,descriptor,days=self.fixture(root,count)
                with library.open_feature_view(descriptor) as view:
                    before=view._store.metrics['source_stat_calls']
                    for day in days: view.project([('A',day)])
                    counts.append(view._store.metrics['source_stat_calls']-before)
                    cached=[]
                    for day in days:
                        before=view._store.metrics['source_stat_calls']
                        view.project([('A',day)])
                        cached.append(view._store.metrics['source_stat_calls']-before)
                    self.assertEqual(len(set(cached)),1)
        self.assertLessEqual(counts[1],2*counts[0]+8)


if __name__=='__main__': unittest.main()
