from __future__ import division

from numpy.testing import (TestCase, assert_equal, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_, assert_raises)

import numpy as np
from agg_fxns import *
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import macroeco.models as mod
import pandas as pd
np.random.seed(2)

"""
Description
------------
Unit tests for the functions given in agg_fxns.py used in the manuscript
feasible_manuscript.tex

Author: Mark Wilber, 2016
"""

class TestAgg(TestCase):

    def setUp(self):

        self.P = 1000
        self.H = 50

        self.P1 = 500
        self.H1 = 20

        self.P2 = 200
        self.H2 = 10

        self.P3 = 2
        self.H3 = 2

    def test_feasible_mixture(self):
        """
        The feasible set function is already unit-tested in Locey and McGlinn.
        Just double check that I am getting back what I expect
        """

        full, pred = feasible_mixture([(self.P1, self.H1)], samples=5)

        # Test the dimensions of full are correct
        assert_array_equal(np.array(full).shape, (5, 20))

        assert_almost_equal(self.P1 / self.H1, np.mean(pred), decimal=0)

        # Test with a mixture

        full, pred = feasible_mixture([(self.P1, self.H1),
                                        (self.P2, self.H2)], samples=5)

        # Test shape of mixture array
        assert_array_equal(np.array(full).shape, (5, self.H1 + self.H2))

        # Test the mean
        pred_mean = (self.P1 + self.P2) / (self.H1 + self.H2)
        assert_almost_equal(pred_mean, np.mean(pred), decimal=0)

        # Test no parasites
        full, pred = feasible_mixture([(0, 10)], samples=5)
        assert_array_equal(pred, np.repeat(0, 10))

        # Test one host
        full, pred = feasible_mixture([(100, 1)], samples=5)
        assert_equal(pred, 100)


    def test_maxent_mixture_correct_vals(self):
        """
        Test that the maxent mixture function returns the right length and
        means
        """

        pred = nbd_mixture([(self.P, self.H)])

        # Maxent returns the right length
        assert_equal(len(pred), self.H)

        # Maxent returns the right mean
        assert_almost_equal(self.P / self.H, np.mean(pred), decimal=0)

        # Test the right length and mean for maxent mixture

        pred_mean = (self.P1 + self.P2) / (self.H1 + self.H2)

        pred = nbd_mixture([(self.P1, self.H1), (self.P2, self.H2)])

        assert_equal(len(pred), self.H1 + self.H2)
        assert_almost_equal((self.P1 + self.P2) / (self.H1 + self.H2),
                                np.mean(pred), decimal=0)

        # Test no parasites
        pred = nbd_mixture([(0, 10)])
        assert_array_equal(pred, np.repeat(0, 10))

        # Test one host
        pred = nbd_mixture([(100, 1)])
        assert_equal(pred, 100)

        # Test it returns the right number of hosts given a decimal
        pred = nbd_mixture([(100, 10.00000007), (20, 3.0001)])
        assert_equal(len(pred), 13)

        # Test it returns the right length
        pred = nbd_mixture([(100, 9.9999999), (20, 3.0001)])
        assert_equal(len(pred), 13)

    def test_maxent_mixture_different_ks(self):
        """
        Test the maxent mixture works as expected with different k
        """

        # Test with different k

        pred = nbd_mixture([(self.P1, self.H1), (self.P2, self.H2)], k=1)
        pred2 = nbd_mixture([(self.P1, self.H1), (self.P2, self.H2)], k=0.5)

        k1 = mod.cnbinom.fit_mle(pred, k_array=np.linspace(0.1, 2, 100))
        k2 = mod.cnbinom.fit_mle(pred2, k_array=np.linspace(0.1, 2, 100))

        assert_equal(k1 > k2, True)

    def test_maxent_mixture_small_vals(self):
        """
        Test that the maxent mixture is giving reasonable answers for small
        values of H and P
        """

        pred = nbd_mixture([(self.P3, self.H3), (self.P3, self.H3)], k=1)

        pred_mean = (2 * self.P3) / (2 * self.H3)

        assert_almost_equal(pred_mean, np.mean(pred), decimal=0)

    def test_poisson_mixture(self):
        """
        Test that the poisson mixture function returns the right length and
        means
        """


        pred = poisson_mixture([(self.P, self.H)])

        # Poisson returns the right length
        assert_equal(len(pred), self.H)

        # Poisson returns the right mean
        assert_almost_equal(self.P / self.H, np.mean(pred), decimal=0)

        pred_mean = (self.P1 + self.P2) / (self.H1 + self.H2)

        pred = poisson_mixture([(self.P1, self.H1), (self.P2, self.H2)])

        assert_equal(len(pred), self.H1 + self.H2)
        assert_almost_equal((self.P1 + self.P2) / (self.H1 + self.H2),
                        np.mean(pred), decimal=0)

        # Test no parasites
        pred = poisson_mixture([(0, 10)])
        assert_array_equal(pred, np.repeat(0, 10))

        # Test one host
        pred = poisson_mixture([(100, 1)])
        assert_equal(pred, 100)

        # Test non-finite
        pred = poisson_mixture([(self.P1, self.H1), (self.P2, self.H2)],
                            finite=False)

        assert_equal(len(pred), self.H1 + self.H2)
        assert_almost_equal((self.P1 + self.P2) / (self.H1 + self.H2),
                        np.mean(pred), decimal=0)

    def test_poisson_mixture_small_vals(self):
        """
        Test that the poisson mixture is returning reasonable values for small
        values of H and P

        """
        pred = poisson_mixture([(self.P3, self.H3), (self.P3, self.H3)], finite=False)

        pred_mean = (2 * self.P3) / (2 * self.H3)

        assert_almost_equal(pred_mean, np.mean(pred), decimal=0)


    def test_cart_analysis(self):
        """
        Test that the R cart analysis and the Python CART analysis are
        returning the expected values for P and H

        """
        # Make some data
        H1 = 30
        para1 = mod.nbinom.rvs(20, 1, size=H1)

        H2 = 200
        para2 = mod.nbinom.rvs(5, 1, size=H2)

        H3 = 50
        para3 = mod.nbinom.rvs(12, 1, size=H3)

        ids = np.repeat((1, 2, 3), (H1, H2, H3))

        test_data = pd.DataFrame(zip(list(para1) + list(para2) + list(para3), ids),
                                columns=["para", "spp"])

        python_results = cart_analysis(test_data, "para", ["spp"],
                        max_leaf_nodes=3, use_R=False)[0]

        r_results = cart_analysis(test_data, "para", ["spp"], use_R=True)[0]

        python_P = sum(zip(*python_results)[0])
        python_H = sum(zip(*python_results)[1])

        r_P = sum(zip(*r_results)[0])
        r_H = sum(zip(*r_results)[1])

        # Test that the two method give the same P and H
        assert_equal(python_P, r_P)
        assert_equal(python_H, r_H)

        # Test that the two methods are the same length in this case
        assert_equal(len(python_results), len(r_results))

        # Test that the two methods givethe same results
        # (won't always be the case)
        assert_array_equal(np.sort(np.ravel(r_results)),
                np.sort(np.ravel(python_results)))

    def test_fix_rounding_error(self):
        """ Test the the rounding error function is working """

        vec1 = [4.5, 6.5, 2.5]
        vec2 = fix_rounding_errors(vec1)

        assert_equal(np.sum(vec1), np.sum(vec2))

        vec1 = [4.33, 6.67]
        vec2 = fix_rounding_errors(vec1)

        assert_equal(np.sum(vec1), np.sum(vec2))

        vec1 = np.linspace(0, 10, num=100)
        vec2 = fix_rounding_errors(vec1)

        assert_equal(np.sum(vec1), np.sum(vec2))

class TestAlgorithms(TestCase):

    def test_maxent_weights(self):
        """ Test that this function returns the right maxent weight """

        vect = [3, 0, 0]
        exp_weight = 3 / 10.
        pred_weight = maxent_weight_fxn(vect)

        assert_equal(pred_weight, exp_weight)

        # Order shouldn't matter
        vect1 = [0, 3, 0]
        predw2 = maxent_weight_fxn(vect1)
        assert_equal(predw2, pred_weight)

        # Try a bigger vector
        vect2 = [2, 1, 1, 1]
        ew = 4. / 56
        pw3 = maxent_weight_fxn(vect2)
        assert_almost_equal(ew, pw3, decimal=6)

    def test_pihm_weight(self):
        """ Test that parasite-induced mortality function gives the correct
        weight
        """
        vect = np.array([5, 3])
        a = 2
        b = -1
        pred = np.prod(np.exp(a + b * vect) / (np.exp(a + b * vect) + 1))
        obs = pihm_weight(vect, a, b)

        assert_equal(pred, obs)

    def test_rejection_algorithm(self):
        """ Test the rejection algorithm for a simple case """

        # The rejection algorithm should give the almost the same answer as
        # the predicted order statistics
        me_rj1, rejection = constrained_ordered_set(15, 10, maxent_weight_fxn,
                                feasible_proposal, feasible_ratio,
                                samples=200,
                                sampling="rejection")
        pred_me = nbd_mixture([(15, 10)])
        assert_array_equal(np.median(me_rj1, axis=0), pred_me)


    def test_metropolis_hastings_alg_feasible(self):
        """ Test the MH algorithm can return maxent predictions """

        # The metropolis algorithm should give the almost the same answer as
        # the predicted order statistics
        me_rj1, rej1 = constrained_ordered_set(15, 10, maxent_weight_fxn,
                                feasible_proposal, feasible_ratio,
                                samples=1000,
                                sampling="metropolis-hastings")
        pred_me = nbd_mixture([(15, 10)])
        assert_array_equal(np.median(me_rj1[-500:, :], axis=0), pred_me)

        # Using PIHM weight should lead to less variance
        pihm1, rej2 = constrained_ordered_set(15, 10,
                                lambda x: pihm_weight(x, 2, -1),
                                feasible_proposal,
                                feasible_ratio,
                                samples=1000,
                                sampling="metropolis-hastings")

        fs_var = np.var(feasible_mixture([(15, 10)])[1])
        test_var = np.var(np.median(pihm1[500:, :], axis=0))
        assert_equal(fs_var > test_var, True)

    def test_metropolis_hastings_alg_feasible2(self):
        """ 
        For a variety of different combinations, test that the metropolis
        hastings algorithm can return the the maxent predictions after
        drawing from the feasible set. This unit test makes a plot.
        In the plot, you should see 9 figures and you would generally expect
        the points in all the figures to fall along the 1:1 line (the dashed
        black line). This means that taking the median of the 
        """

        Ps = [500, 200, 50]
        Hs = [30, 20, 10]

        fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
        axes = axes.ravel()

        count = 0
        for P in Ps:
            for H in Hs:
                me_pred = nbd_mixture([(P, H)])
                me_mh, _ = constrained_ordered_set(P, H, maxent_weight_fxn,
                                feasible_proposal, feasible_ratio,
                                samples=4000,
                                sampling="metropolis-hastings")

                # Drop the burn in values
                mh_pred = np.median(me_mh[-2000:, :], axis=0)

                # Plot the predictions
                axes[count].plot(np.log(me_pred + 1), np.log(mh_pred + 1), 'o')
                vals = np.linspace(0, 6, num=100)
                axes[count].plot(vals, vals, '--', color='black')
                axes[count].set_xlabel("Analytical ln(RAD + 1) for\ncomposition model", size=10)
                axes[count].set_ylabel("Weighted feasible set\nln(RAD + 1) for composition model", size=10)
                axes[count].set_xlim((-0.5, 6.5))
                axes[count].set_ylim((-0.5, 6.5))
                axes[count].text(0.5, 0.9, "P={0}, H={1}".format(P, H),
                                 ha='center', transform=axes[count].transAxes)

                count = count + 1

        # Save the plot
        plt.tight_layout()
        plt.savefig("me_mh_compare.png")


    def test_metropolis_hastings_alg_binomial(self):
        # Test that the metropolis-hasting algorithm with a binomial proposal
        # gives predicted order statistics

        np.random.seed(3) # Predictions will change slightly with seed

        bin1, rej1 = constrained_ordered_set(13, 10, binomial_weight_fxn,
                                binomial_proposal, binomial_ratio,
                                samples=1000,
                                sampling="metropolis-hastings")

        assert_equal(rej1, 0) # There should be no rejections

        # Prediction should be about the same as binomial ordered stats
        pred_rad = poisson_mixture([(13, 10)], finite=True)
        sim_rad = np.median(bin1, axis=0)
        assert_array_equal(pred_rad, sim_rad)

        # The mixing will be bad, but should be able to recover the binomial
        # by sampling the feasible set.
        # NOTE: The mixing is terrible for higher dimensional vectors when
        # trying to sample a binomial from a feasible set.  Therefore, you
        # should just start by sampling from a binomial. You could do this
        # with the code.  This is just naive way to draw a random number from
        # a multinomial distribution.

        # bin2, rej2 = constrained_ordered_set(10, 3, binomial_weight_fxn,
        #                         binomial_proposal, binomial_ratio,
        #                         samples=5000,
        #                         sampling="metropolis-hastings")

        bin2, rej2 = constrained_ordered_set(10, 3, binomial_weight_fxn,
                                feasible_proposal, feasible_ratio,
                                samples=5000,
                                sampling="metropolis-hastings")
        pred_rad = poisson_mixture([(10, 3)], finite=True)
        sim_rad = np.median(bin2[3000:, ], axis=0)

        # This is test is will fail sometimes because the highest rank differs
        # by one individual
        # assert_array_equal(sim_rad, pred_rad)

    def test_pihm_weight_fxns_binom(self):
        # Test that different weight functions give the qualitatively correct
        # results

        a = 2; b = -0.5
        P = 100; H = 10

        pred_binom = poisson_mixture([(P, H)])

        sbin, rej = constrained_ordered_set(P, H,
                                binomial_weight_fxn,
                                binomial_proposal, binomial_ratio,
                                samples=5000,
                                sampling="metropolis-hastings")

        sbin_pihm, rej_pihm = constrained_ordered_set(P, H,
                                lambda x: pihm_weight(x, a, b),
                                binomial_proposal, feasible_ratio,
                                samples=5000,
                                sampling="metropolis-hastings")

        binom_vect = np.median(sbin[3000:, ], axis=0)
        pihm_vect = np.median(sbin_pihm[3000:, ], axis=0)

        # Constraining a Binomial distribution on PIHM shouldn't have much
        # affect on the prediction.

        # There should be a slight difference (not much) binom and PIHM vector
        # Slightly smaller var to mean ratio
        vm_pihm = np.var(pihm_vect, ddof=1) / np.mean(pihm_vect)
        vm_pred = np.var(pred_binom, ddof=1) / np.mean(pred_binom)
        assert_equal(vm_pihm < vm_pred, True)

    def test_pihm_weight_fxns_maxent(self):
        # Test that different weight functions give the qualitatively correct
        # results

        a = 1; b = -0.5
        P = 100; H = 10

        pred_maxent = nbd_mixture([(P, H)])

        sbin_pihm, rej_pihm = constrained_ordered_set(P, H,
                                lambda x: pihm_maxent_weight(x, a, b),
                                feasible_proposal, feasible_ratio,
                                samples=5000,
                                sampling="metropolis-hastings")

        pihm_vect = np.median(sbin_pihm[3000:, ], axis=0)

        # Constraining the maxent distribution should reduce the variance to
        # mean ratio
        vm_pihm = np.var(pihm_vect, ddof=1) / np.mean(pihm_vect)
        vm_pred = np.var(pred_maxent, ddof=1) / np.mean(pred_maxent)
        assert_equal(vm_pihm < vm_pred, True)


    def test_bad_algorithm(self):
        # Function should raise error if bad algorithm is passed
        assert_raises(NotImplementedError, constrained_ordered_set, 15, 10,
                    maxent_weight_fxn, feasible_proposal, feasible_ratio,
                    samples=1000, sampling="metro")

class TestMixture(TestCase):

    # Slow test
    def test_order_stats(self):
        """
        Test that approximate order statistics are close to simulated
        order statistics

        Given a mixture model defined by ph_vects, get the predicted
        rank abundance distribution/order statistics using the method
        outlined in Harte 2011 pg 153. Compare this to a Monte Carlo simulation
        in which the mixture is drawn and ordered and then the ordered
        mixtures are averaged. These should give approximately the same
        answers.  Compare them by looking looking at the figure output
        and seeing it they fit the 1:1 line.

        """

        NUM = 200 # Number of simulation

        # Vects giving the parasite host mixture
        # [(P1, H1), (P2, H2), ...] where P = number of parasites and H = number of
        # hosts.
        # Adjust the values in these mixtures or add more for additional tests
        ph_vects = [[(200, 20)],
                   [(1000, 10), (550, 20)],
                   [(10, 30), (20, 12), (8, 3)],
                   [(500., 30), (30., 10), (10, 20), (5, 4)]]

        medians = []
        me_preds = []
        for ph_vect in ph_vects: # Loop over mixture models

            # Predicted order stats
            me_preds.append(nbd_mixture(ph_vect, finite=True))

            # Uncomment for Poisson/Binomial
            #me_preds.append(poisson_mixture(ph_vect, finite=True))

            res = [] # Store results
            for i in range(NUM): # loop over simulations

                tres = []
                for P, H in ph_vect: # loop of host parasite values

                    tme = mod.cnbinom.rvs(mu=P / H, k_agg=1, b=P, size=H)

                    # Uncomment for poisson/biomial
                    #tme = stats.binom.rvs(n=P, p=1 / H, size=H)
                    tres.append(tme)

                res.append(np.sort(np.concatenate(tres)))

            # Take the medians as the center
            medians.append(np.median(np.array(res), axis=0)[::-1])

            # Uncomment to take means as center
            #medians.append(np.mean(np.array(res), axis=0)[::-1])

        # Plot the results. The points should generally lie very close to the
        # 1:1 line (the black dashed line)
        fig, ax = plt.subplots(1, 1)
        for i, (pred_rank, pred_sim) in enumerate(zip(me_preds, medians)):
            ax.plot(np.log(pred_rank + 1), np.log(pred_sim + 1), 'o', alpha=0.5,
                label=str(ph_vects[i]))


        xlim = ax.get_xlim()
        vals = np.linspace(xlim[0], xlim[1])
        ax.plot(vals, vals, '--', color='black')
        ax.legend(prop={'size': 9}, loc="upper left")

        # Save plot
        fig.savefig("mixture_test.png")

    def test_mixed_vs_median_feas(self):
        """ Testing whether concatenating vectors vs. mixing random macrostates
        leads to equivalent predictions for the feasible set.

        In other words, I could generate a heterogeneity prediction by two 
        approaches
        1) Sample each heterogeneity group 200 times, glue all of these samples
        together (i.e. appending arrays to make the feasible set) and then
        find the center of this feasible set.
        2) I could take a sample from a group, sort it and find the center and
        do this for each group and then append these predicted vectors 
        together.

        Question: Do these give the same answer?
        Answer: Yes. Why? Because expectation is a linear operator!
        """
        NUM = 500
        ph_vects = [[(200, 20)],
                   [(100, 10), (550, 20)],
                   [(10, 30), (20, 12), (8, 3)],
                   [(500., 30), (30., 10), (10, 20), (5, 4)]]

        # Approach 1
        feas_preds = [feasible_mixture(ph_vect, samples=NUM,
                            center="median")[1] for ph_vect in ph_vects]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()

        for i, ph_vect in enumerate(ph_vects):
            all_preds = []

            # Approach 2
            for P, H in ph_vect:

                _, pred = feasible_mixture([(P, H)], samples=NUM,
                                                    center="median")

                all_preds.append(pred)

            exp_preds = np.sort(np.concatenate(all_preds))[::-1]

            # Plot the results comparing the approaches
            axes[i].plot(np.log(feas_preds[i] + 1), np.log(exp_preds + 1), 'o')

            vals = np.linspace(np.min(np.log(feas_preds[i] + 1)),
                                np.max(np.log(feas_preds[i] + 1)), num=100)
            axes[i].plot(vals, vals, '--', color="black")
            axes[i].set_xlabel("ln(Predictions from approach 1 + 1)")
            axes[i].set_ylabel("ln(Predictions from approach 2 + 1)")

        plt.tight_layout()
        fig.savefig("feas_order.png")

class TestEM(TestCase):

    def test_em_algor_geom1(self):
        """ Given a simulated dataset, test that the EM algorithm gets the
        mixture right """

        # Equal samples sizes
        mu1s = [10, 7] # Hard to distinguish larger means
        mu2s = [1, 3] # hard to distinguish larger means

        for mu1, mu2 in zip(mu1s, mu2s):

            # Different means different proportions
            group1 = mod.nbinom.rvs(mu=mu1, k_agg=1, size=500)
            group2 = mod.nbinom.rvs(mu=mu2, k_agg=1, size=500)

            full_data = np.concatenate([group1, group2])

            res = em_algorithm(full_data, [8, 2], [0.4, 0.6], 'geom')

            # Check that the EM algorithm identifies the correct means
            assert_array_almost_equal(res[0], [mu1, mu2], decimal=0)

            # Check that the EM algorithm identifies the correct propotions
            assert_array_almost_equal(res[1], [0.5, 0.5], decimal=1)

    def test_em_algor_geom2(self):
        """ Test em algorithm with different proportions """

        np.random.seed(2)
        # Different group means
        group1 = mod.nbinom.rvs(mu=10, k_agg=1, size=100*3)
        group2 = mod.nbinom.rvs(mu=1, k_agg=1, size=300*3)

        full_data = np.concatenate([group1, group2])

        res = em_algorithm(full_data, [8, 2], [0.5, 0.5], 'geom', tol=0.01)

        # Check that the EM algorithm identifies the correct means...
        # Note that this works for large sample sizes, but the prediction won't
        # be perfect (obviously) for small samples sizes. Geom is much more
        # sensitive than the Poisson.
        assert_array_almost_equal(res[0], [10, 1], decimal=0)

        # Check that the EM algorithm identifies the correct proportions
        assert_array_almost_equal(res[1], [0.25, 0.75], decimal=1)

    def test_em_algor_pois1(self):
        """ Test the EM algorithm with the poisson distribution"""

        # Equal samples sizes
        mu1s = [10, 7, 20]
        mu2s = [1, 3, 10]

        for mu1, mu2 in zip(mu1s, mu2s):

            group1 = stats.poisson.rvs(mu1, size=100)
            group2 = stats.poisson.rvs(mu2, size=100)

            full_data = np.r_[group1, group2]

            res = em_algorithm(full_data, [5, 3], [0.7, 0.3], 'pois', tol=0.01)

            assert_array_almost_equal(res[0], [mu1, mu2], decimal=0)

            # Check that the EM algorithm identifies the correct proportions

            assert_array_almost_equal(res[1], [0.5, 0.5], decimal=1)

    def test_em_algor_pois2(self):
        """ Test the EM algorithm with the poisson distribution"""

        # Equal samples sizes
        size1s = [100., 100., 20.]
        size2s = [50., 300., 80.]

        for size1, size2 in zip(size1s, size2s):

            group1 = stats.poisson.rvs(10, size=size1)
            group2 = stats.poisson.rvs(1, size=size2)

            full_data = np.r_[group1, group2]

            res = em_algorithm(full_data, [5, 3], [0.7, 0.3], 'pois', tol=0.01)

            assert_array_almost_equal(res[0], [10, 1], decimal=0)

            # Check that the EM algorithm identifies the correct proportions
            assert_array_almost_equal(res[1],
                            [size1 / len(full_data), size2 / len(full_data)],
                            decimal=1)

    def test_em_to_ph_vect_conversion(self):
        """ Test that we can convert between ph_vects and em mixing proportions"""

        Ps_list = [[200, 100], [10], [122, 97, 0]]
        Hs_list = [[50, 30], [20], [3, 4, 5]]

        for Ps, Hs in zip(Ps_list, Hs_list):

            ph_vect = zip(Ps, Hs)
            pred_mus = np.array(Ps) / np.array(Hs)
            pred_pis = np.array(Hs) / np.sum(Hs)

            mus, pis = convert_para_host_to_mixed(ph_vect)

            assert_array_almost_equal(mus, pred_mus, decimal=0)
            assert_array_almost_equal(pis, pred_pis, decimal=2)

            # Convert back
            pred_ph_vect = convert_em_results((mus, pis), np.sum(Hs))

            assert_array_equal(ph_vect, pred_ph_vect)

























