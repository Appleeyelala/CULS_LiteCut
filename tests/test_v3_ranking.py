"""Small mathematical fixtures; no model, dataset, or Mapping execution."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'scripts/evaluate_v3_ranking.py'
spec = importlib.util.spec_from_file_location('ranking', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RankingTests(unittest.TestCase):
    def evaluate(self, p, a, indices=None):
        return m.evaluate(p, a, indices or list(range(1, len(p)+1)), .001, 1.)[0]

    def test_perfect_and_reverse(self):
        for pred, expected in [([1, 2, 3, 4], 1), ([4, 3, 2, 1], -1)]:
            r = self.evaluate(pred, [1, 2, 3, 4])
            self.assertAlmostEqual(r['spearman'], expected)
            self.assertAlmostEqual(r['kendall_tau_b'], expected)
            self.assertEqual(r['pairwise_accuracy'], (expected+1)/2)

    def test_constant_prediction_does_not_receive_pair_credit(self):
        r = self.evaluate([2]*4, [4, 1, 3, 2])
        self.assertIsNone(r['spearman'])
        self.assertIsNone(r['kendall_tau_b'])
        self.assertEqual(r['pairwise_accuracy'], 0)
        self.assertEqual(r['predicted_tie_only'], 6)
        self.assertEqual(r['top1_hit'], 0)
        self.assertEqual(r['uniform_tie_top1_hit'], .25)
        self.assertEqual(r['top1_regret_ps'], 3)

    def test_constant_truth(self):
        r = self.evaluate([4, 3, 2, 1], [2]*4)
        self.assertIsNone(r['spearman'])
        self.assertIsNone(r['pairwise_accuracy'])
        self.assertEqual(r['true_tie_only'], 6)
        self.assertEqual(r['top1_hit'], 1)
        self.assertEqual(r['top1_regret_percent'], 0)

    def test_transitive_bounded_width_ties(self):
        self.assertEqual(m.ranks([1.0018, 1., 1.0009], .001), [3, 1.5, 1.5])

    def test_frozen_index_tie_break_not_label_or_input_order(self):
        r = self.evaluate([2, 2, 3], [1, 4, 5], [7, 2, 3])
        self.assertEqual(r['top1_index'], 2)
        self.assertEqual(r['top1_hit'], 0)
        self.assertEqual(r['uniform_tie_top1_hit'], .5)

    def test_tie_categories_and_scipy_reference(self):
        from scipy.stats import spearmanr, kendalltau
        p, a = [1, 1, 2, 2, 3], [1, 1, 3, 4, 3]
        r = self.evaluate(p, a)
        self.assertEqual([r[k] for k in ['correct', 'wrong', 'predicted_tie_only', 'true_tie_only', 'both_tied']], [6, 1, 1, 1, 1])
        self.assertAlmostEqual(r['spearman'], spearmanr(p, a).statistic)
        self.assertAlmostEqual(r['kendall_tau_b'], kendalltau(p, a, variant='b').statistic)

    def test_aggregation_retains_undefined_cases(self):
        a = self.evaluate([1, 2, 3], [1, 2, 3])
        b = self.evaluate([1]*3, [1, 2, 3])
        r = m.aggregate([a, b])
        self.assertEqual(r['spearman_defined'], 1)
        self.assertEqual(r['spearman_mean_zero_filled'], .5)
        self.assertEqual(r['spearman_mean_defined_only'], 1)


if __name__ == '__main__':
    unittest.main()
