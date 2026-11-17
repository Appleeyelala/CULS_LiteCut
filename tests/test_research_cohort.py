import unittest
from global_timing.research_cohort import family_name, grouped_split, blind_panel, compare_delay


class ResearchCohortTests(unittest.TestCase):
    def inventory(self):
        return [dict(index=i,relative_path=f'src/design_{chr(97+i//26)}{chr(97+i%26)}.aig',
                     sha256=f'{i:064x}',and_count=1000+i*500) for i in range(200)]

    def test_family_and_hash_groups_never_leak(self):
        rows=self.inventory()
        rows[0]['relative_path']='src/ram_16x128_orig.aig'
        rows[1]['relative_path']='other/ram_32x256_comb.aig'
        rows[2]['sha256']=rows[1]['sha256']
        out=grouped_split(rows)
        self.assertEqual(len({r['split'] for r in out[:3]}),1)
        self.assertEqual(len({r['group_id'] for r in out[:3]}),1)
        self.assertEqual(out,grouped_split(list(reversed(rows))))

    def test_panel_does_not_read_labels(self):
        rows=grouped_split(self.inventory())
        before=blind_panel(rows,'test',20)
        for r in rows:r['status']='FAIL';r['delay']=-999
        self.assertEqual(before,blind_panel(rows,'test',20))
        self.assertEqual(len(set(before)),20)
        with self.assertRaises(ValueError):blind_panel(rows,'test',999)

    def test_tolerance_and_duplicate_identity(self):
        self.assertEqual(compare_delay(100.0001,100),0)
        self.assertEqual(compare_delay(101,100),1)
        self.assertEqual(compare_delay(99,100),-1)
        with self.assertRaises(ValueError):grouped_split(self.inventory()+self.inventory()[:1])


if __name__=='__main__':unittest.main()
