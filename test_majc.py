"""CPU numerical tests; no LM, dataset, remote host, or paid API calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize

from majc import Config, candidate_scores, control, kl_divergence, optimize_ind, optimize_ind_c, preserve_tail


class OptimizerTests(unittest.TestCase):
    def test_ind_analytic_entropy_solution(self):
        p = np.array([.4072, .3406, .2383, .0137, 2e-4, 3e-10]); p /= p.sum()
        score = np.array([.5862, .6060, .6055, .5893, .6185, .6200])
        expected = p * np.exp((score - score.max()) / .05); expected /= expected.sum()
        result = optimize_ind(p, score, score, score)
        self.assertEqual(result['status'], 'SOLVED')
        np.testing.assert_allclose(result['probabilities'], expected, atol=1e-12)
        self.assertTrue(result['optimality_certificate']['certified'])

    def test_two_protections_block_greedy_choice(self):
        result = optimize_ind([.5, .5], [.1, .9], [.9, .1], [.8, .2])
        self.assertLessEqual(result['probabilities'][1], .500001)
        self.assertLessEqual(result['maximum_residual'], 1e-6)

    def test_original_kl_ball_is_shared(self):
        for protocol in ('executed_scale', 'executed_quick', 'disclosure_v1_1'):
            result = optimize_ind_c([.4, .3, .2, .1], [.40, .401, .403, .402],
                                    [.2, .7, .4, .9], [.7, .6, .9, .5], [.5, .7, .8, .9], protocol=protocol)
            self.assertLessEqual(result['KL'], .020001)
            self.assertLessEqual(result['maximum_residual'], 1e-6)
            self.assertGreaterEqual(result['expected_IND'], result['ind_floor'] - 1e-6)

    def test_cannot_undo_ind_gain(self):
        result = optimize_ind_c([.5, .5], [0., 1.], [1., 0.], [.8, .8], [.8, .8])
        self.assertGreater(result['ind_floor'], .55)
        self.assertAlmostEqual(result['expected_IND'], result['ind_floor'], places=6)

    def test_invalid_solver_falls_back_to_ind(self):
        anchor = optimize_ind([.5, .5], [0., 1.], [.8, .8], [.8, .8])
        fake = SimpleNamespace(x=np.array([np.nan, 0.]), nit=1, success=True, message='invalid')
        with patch('majc.minimize', return_value=fake):
            result = optimize_ind_c([.5, .5], [0., 1.], [1., 0.], [.8, .8], [.8, .8], ind_result=anchor)
        np.testing.assert_array_equal(result['probabilities'], anchor['probabilities'])
        self.assertFalse(result['optimality_certificate']['certified'])
        self.assertEqual(result['fallback_distribution'], 'IND')

    def test_feasible_but_uncertified_candidate_is_rejected(self):
        fake = SimpleNamespace(x=np.array([1e8]), nit=1, success=True, message='claimed convergence')
        with patch('majc.minimize', return_value=fake):
            result = optimize_ind([.5, .5], [.1, .9], [.8, .8], [.7, .7])
        self.assertEqual(result['status'], 'SOLVER_FALLBACK')
        self.assertGreater(result['optimality_certificate']['primal_dual_gap'], 1e-6)
        np.testing.assert_array_equal(result['probabilities'], [.5, .5])

    def test_reused_anchor_must_match_all_scores(self):
        anchor = optimize_ind([.5, .5], [.3, .4], [.8, .8], [.7, .7])
        for joint, usable, mean in (([.3, .5], [.8, .8], [.7, .7]),
                                    ([.3, .4], [.9, .8], [.7, .7]),
                                    ([.3, .4], [.8, .8], [.7, .6])):
            with self.assertRaisesRegex(ValueError, 'cached IND'):
                optimize_ind_c([.5, .5], joint, [.4, .5], usable, mean, ind_result=anchor)

    def test_disclosure_flat_c_is_exactly_ind(self):
        result = optimize_ind_c([.5, .5], [.4, .41], [.6, .6], [.8, .8], [.8, .8], protocol='disclosure_v1_1')
        np.testing.assert_array_equal(result['probabilities'], result['ind_anchor']['probabilities'])
        self.assertEqual(result['status'], 'FLAT_OBJECTIVE')

    def test_zero_budget_and_single_candidate(self):
        result = optimize_ind_c([.4, .6], [.2, .3], [.4, .1], [.8, .8], [.7, .7], config=Config(delta_kl=0))
        np.testing.assert_array_equal(result['probabilities'], [.4, .6])
        self.assertEqual(optimize_ind([1], [.3], [.8], [.7])['probabilities'], [1.])

    def test_dual_against_independent_primal_solver(self):
        p = np.array([.4, .3, .2, .1]); ji = np.array([.4, .401, .403, .402])
        jc = np.array([.2, .7, .4, .9]); u = np.full(4, .8); m = np.full(4, .7)
        anchor = optimize_ind(p, ji, u, m)
        rho = np.array(anchor['probabilities'])
        for protocol in ('executed_scale', 'disclosure_v1_1'):
            result = optimize_ind_c(p, ji, jc, u, m, protocol=protocol, ind_result=anchor)
            base = rho if protocol == 'disclosure_v1_1' else p
            def objective(z): return -float(z @ jc) + .05 * kl_divergence(z, base)
            primal = minimize(objective, rho, method='SLSQP', bounds=[(1e-10, 1)] * 4,
                              constraints=[{'type':'eq', 'fun':lambda z: z.sum() - 1},
                                           {'type':'ineq', 'fun':lambda z: .02 - kl_divergence(z, p)},
                                           {'type':'ineq', 'fun':lambda z: (z - rho) @ ji}],
                              options={'maxiter':500, 'ftol':1e-12})
            self.assertTrue(primal.success)
            self.assertEqual(result['status'], 'SOLVED')
            self.assertAlmostEqual(objective(np.array(result['probabilities'])), primal.fun, places=7)

    def test_invalid_inputs_are_rejected(self):
        for p in ([0, 1], [-1, 2], [np.nan, 1], [], [[.5, .5]]):
            with self.assertRaises(ValueError): optimize_ind(p, [.2, .3], [.8, .8], [.7, .7])
        for kwargs in ({'tau_kl':0}, {'delta_kl':-1}, {'maxiter':1.5}, {'maxiter':True}, {'tolerance':np.nan}):
            with self.assertRaises(ValueError): Config(**kwargs)
        with self.assertRaises(ValueError): optimize_ind([.5, .5], [.2, 1.1], [.8, .8], [.7, .7])


class IntegrationTests(unittest.TestCase):
    def test_tail_mask_and_kl_identity(self):
        full = np.array([.32, .24, .16, .08, .13, .07, 0.])
        conditional = np.array([.5, .2, .2, .1])
        adjusted, audit = preserve_tail(full, [0, 1, 2, 3], conditional)
        np.testing.assert_array_equal(adjusted[4:], full[4:])
        self.assertTrue(audit['tail_exactly_unchanged'])
        self.assertAlmostEqual(audit['full_KL'], .8 * kl_divergence(conditional, full[:4] / .8), places=14)

    def test_no_truncation_or_implicit_id_conversion(self):
        for ids in ([0., 1.], [0, 0], [-1, 1], [0, 3], []):
            with self.assertRaises(ValueError): preserve_tail([.5, .5], ids, [.4, .6])

    def test_exact_facts_empty_event_and_weights(self):
        scores = candidate_scores([.8, .9], [[0., 1.], [1., 1.]], [.3, .4], weights=[1, 2])
        np.testing.assert_array_equal(scores['J_ind'], [0., .9])
        np.testing.assert_allclose(scores['M'], [.8 * 2 / 3, .9])
        empty = candidate_scores([.8, .9], np.empty((2, 0)), [.3, .4])
        np.testing.assert_array_equal(empty['J_ind'], empty['J_c'])
        np.testing.assert_array_equal(empty['M'], [.8, .9])

    def test_control_disabled_recovers_ind(self):
        adjusted, audit = control([.4, .3, .2, .1, 0.], [0, 1, 2], [.8]*3,
                                   [[.7, .8], [.71, .8], [.7, .81]], [.5, .6, .4], enable_c=False)
        expected, _ = preserve_tail([.4, .3, .2, .1, 0.], [0, 1, 2], audit['ind_anchor']['probabilities'])
        np.testing.assert_array_equal(adjusted, expected)
        self.assertEqual(audit['stage2'], 'C_DISABLED')

    def test_one_event_skips_c(self):
        _, audit = control([.5, .5], [0, 1], [.8, .8], [[.7], [.8]], [.6, .6])
        self.assertEqual(audit['stage2'], 'FEWER_THAN_TWO_EVENTS')


if __name__ == '__main__':
    unittest.main()
