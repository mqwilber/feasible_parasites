from __future__ import division
import agg_fxns as agg
import numpy as np
import pandas as pd
import multiprocessing as mp
import statsmodels.api as sm
from scipy import stats
import logging
logging.basicConfig(filename="parasite_mortality.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

logger = logging.getLogger("my logger")

"""
Description
-----------
This script calculates the predicted top-down feasible set for a given host
Rib combination given a laboratory estimated host-survival curve. It is taking
into account parasite-induced host mortality in a top-down context.

Author
------
Mark Wilber, UCSB, 2016

"""

def multiprocess_pihm(processes, grouped_data):
    """
    Multiprocessing function

    Parameters
    ----------
    processes : int
        Number of cores to use
    grouped : list
        List of grouped data to multiprocess

    Returns
    -------
    : list of tuples
        Each tuple contains an index, observed vector, and predicted vector

    """

    pool = mp.Pool(processes=processes)

    results = [pool.apply_async(_get_pihm_pred,
                args=(host, dist_nm, a, b, wf, pf, pr, samps, obs, pred, i)) for
                i, (host, dist_nm, a, b, wf, pf, pr, samps, obs, pred)
                in enumerate(grouped_data)]

    results = [p.get() for p in results]
    results.sort()

    # Clean up processes
    pool.close()
    pool.join()

    return results

def _get_pihm_pred(host, dist_nm, a, b, weight_fxn, proposal_fxn,
                        proposal_ratio, samps, obs, pred, sorter):
    """
    Work horse function for the multiprocessing

    Parameters
    ----------
    host_nm : str
        Host name
    dist_nm : str
        Distribution name
    weight_fxn, proposal_fxn, proposal_ratio : fxns
        Functions for metropolis-hastings algorithm
    a : float
        Intercept of logistic
    b : float
        Slope of logistics
    obs : array-like
        Observed vector
    pred : array-like
        Predicted vector
    sorter : int
        Counter for indexing purposes
    samps : tuple
        Number of metropolis samples for (small, large) parasite loads

    Returns
    -------
    : tuple
    """

    P = np.sum(obs)
    H = len(obs)

    if P < 700:  # For speeding things up with larger parasite loads
        SAMPS = samps[0]
    else:
        SAMPS = samps[1]


    pred_matrix, rej = agg.constrained_ordered_set(P, H,
                        lambda x: weight_fxn(x, a, b),
                        proposal_fxn, proposal_ratio,
                        samples=SAMPS, sampling="metropolis-hastings")

    logging.info("{0}: Completed sample, host {1}, dist {2}: P = {3}, H = {4}".format(sorter + 1,
                            host, dist_nm, P, H))

    # Get center after burn-in
    pred_cent = np.median(pred_matrix[int(.5 * SAMPS):, :], axis=0)

    return((sorter, host, dist_nm, pred_cent, obs, 1 - (rej / SAMPS)))

def get_host_survival_curves(host_names):
    """
    Runs the GLM and gets the host survival curves. Only implemented for BUBO,
    PSRE, and TATO.

    Parameters
    ----------
    host_names : list
        A list of host names

    Returns
    -------
    : dict
        Keys are host names and values are tuple with (a, b) of logistic
        survival
    """
    # BUBO data from Johnson et al. 2001. Canadian Journal of Zoology
    # PSRE data from Johnson et al. 1999. Science
    # TATO data from Johnson et al. 2012, Ecology Letters
    host_data = {'PSRE': {'load': [0, 16, 32, 48],
                          'response': np.array([[31, 32, 18, 18],
                                                [4, 13, 27, 27]])},
                 'BUBO': {'load': [0, 16, 32, 47],
                          'response': np.array([[47, 33, 31, 20],
                                                [13, 22, 24, 30]])},
                 'TATO': {'load': [0, 20, 32, 40, 100, 200],
                          'response': np.array([[25, 24, 25, 24, 25, 14],
                                                [0, 1, 0, 1, 0, 11]])}}

    glm_params = {}
    for host in host_names:

        resp = host_data[host]['response'].T
        pred = sm.add_constant(host_data[host]['load'])
        glm_binom = sm.GLM(resp, pred, family=sm.families.Binomial())
        res = glm_binom.fit()
        glm_params[host] = res.params

    return(glm_params)

if __name__ == '__main__':

    # Load data and extract PSRE RION
    td_bu_vects = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_vectors.pkl")

    para_name = 'RION'
    host_names = ['PSRE'] #['BUBO', 'TATO', 'PSRE', 'BUBO']
    weight_fxns = [agg.pihm_weight, agg.pihm_maxent_weight,
                                            agg.pihm_weight]
    proposal_fxns = [agg.feasible_proposal, agg.feasible_proposal,
                                            agg.binomial_proposal]
    proposal_ratios = [agg.feasible_ratio, agg.feasible_ratio,
                                            agg.feasible_ratio]
    dist_names = ["feasible", "trun_geometric", "binomial"]

    # Get survival curves from literature
    host_names_dict = get_host_survival_curves(host_names)

    grouped_data = []
    for host in host_names_dict.viewkeys():
        for dist_nm, wf, pf, pr in zip(dist_names, weight_fxns, proposal_fxns,
                                                            proposal_ratios):

            obs_vects, pred_vects = agg.extract_obs_pred_vectors(td_bu_vects,
                                                dist_nm, host, para_name)

            for obs, pred in zip(obs_vects, pred_vects):

                # Estimated survival curve data
                a = host_names_dict[host][0]
                b = host_names_dict[host][1]
                samps = (2000, 2000) # (1000, 200)

                grouped_data.append((host, dist_nm, a, b, wf, pf, pr, samps, obs, pred))


    # Multiprocess the list...
    multi_res = multiprocess_pihm(3, grouped_data)

    # Save results as a dataframe.
    results_df = pd.DataFrame(multi_res,
        columns=['sorter', 'host', 'dist', 'pred', 'observed', 'accept'])

    pd.to_pickle(results_df, "../results/pickled_results/parasite_mortality_vects.pkl")
    logging.info("Completed analysis")



