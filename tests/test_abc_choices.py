import unittest
from types import SimpleNamespace
from global_timing.abc_choices import cover_witness
from global_timing.abc_hard_choices import hard_annotations


def graph(gates, outputs):
    return SimpleNamespace(pi_count=3, po_count=len(outputs), maximum_variable=max(gates),
                           fanin_literals=gates, po_literals=outputs)


class AbcChoicesTest(unittest.TestCase):
    def test_signed_replay_aliases_share_one_candidate(self):
        initial = graph({4:(2,4),5:(6,8)}, [10])
        # Duplicate the same AND and its complement through a constant-true edge.
        installed = graph({4:(4,6),5:(2,8),6:(1,11)}, [13])
        options = [{"original_root":5, "ratio_valid":True}]
        lines = ["g 4 6", "g 2 12", "r 0 1 10 14 1 2 0.8"]
        annotation, = hard_annotations(initial,installed,lines,options)
        self.assertEqual(annotation["alias_literals"], [10,13])
        self.assertEqual(annotation["candidate_origins"], 1)
        with self.assertRaisesRegex(ValueError, "signed PO"):
            hard_annotations(initial,graph(installed.fanin_literals,[12]),lines,options)

    def test_conflicting_cuts_cannot_be_reported_as_a_replay(self):
        log = "CULS_MODE0 12.0\nCULS_CUT 30 0 4 0 3 1 2\nCULS_CUT 31 1 3 1 c 1 2\n"
        vectors = [[0,0],[1,0],[0,1],[1,1]]
        self.assertIsNone(cover_witness(log,vectors,[])["selected"])
        self.assertEqual(cover_witness("CULS_MODE0 2\nCULS_CUT 30 0 4 0 e 1 2",vectors,[])["selected"],1)

    def test_shared_cut_masks_do_not_lose_bit_63(self):
        vectors=[[i] for i in range(64)]
        self.assertEqual(cover_witness("CULS_MODE0 2\nCULS_CUT 9 0 1 0 8000000000000000 1 2",
                                       vectors,[])["selected"],63)


if __name__ == "__main__":
    unittest.main()
