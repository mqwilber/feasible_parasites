import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
import logging
logging.basicConfig(filename="randomization.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

logger = logging.getLogger("my logger")

"""
Description
-----------

This script perform a randomization analysis on the the groupings chosen by the
cart analysis performed in manuscript_analysis_discrete_known_heterogeneity.py.
In this analysis, SVL and larval trematodes were used as predictor variables
and the hosts were split into groups based on their parasite load using a CART
analysis.  From these groups, a mixture model was used to predict the observed
host parasite distribution for feasible sets, max ent, and poisson/binomial
models.

This script extends this analysis by asking whether the "better" fit induced
by grouping on host heterogeneity is simply a product of grouping and not a
result of the predictor variables used to group. This is done by using the
predicted grouping and then randomizing the empirical host parasite data into
this grouping. If the CART grouping provides a better fit to the host parasite
distribution than random grouping, this is evidence that these variables
(or something related to these variables) are structuring heterogeneity in the
host population.

Author
------
Mark Wilber, UCSB, 2016
"""

def get_over_agg_sites(stat="AD_p_val", crit="<0.1"):
    """
    Gets a list of sites that are over aggregated based on variance to mean
    ratio

    Parameters
    ----------
    stat : str
        Statistics to use
    crit : str
        Criteria for the statistic

    Return
    ------
    : list
        List containing tuple where the tuple gives the year, host, parasite,
        site of the overaggregated distribution
    """

    # Load in predicted vectors
    td_bu_stats = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_stats.pkl")
    td_bu_vects = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_vectors.pkl")

    # Get sites that don't fit feasible set and maxent
    fs_sites = agg.extract_sites_with_criteria(td_bu_stats, stat, crit,
                                                            "feasible")
    me_sites = agg.extract_sites_with_criteria(td_bu_stats, stat, crit,
                                                            "trun_geometric")

    # Get observed var / mean ratio to find distributions sites that don't fit
    # feasible sets or maxent
    obs_var_mean_fs = [np.var(tobs, ddof=1) / np.mean(tobs) for tobs in
        agg.extract_vectors_given_tuple(td_bu_vects, fs_sites, "observed")]

    obs_var_mean_me = [np.var(tobs, ddof=1) / np.mean(tobs) for tobs in
        agg.extract_vectors_given_tuple(td_bu_vects, me_sites, "observed")]

    # Extract var:mean for fs and me
    fs_var_mean = [np.var(tobs, ddof=1) / np.mean(tobs) for tobs in
        agg.extract_vectors_given_tuple(td_bu_vects, fs_sites, "feasible")]

    me_var_mean = [np.var(tobs, ddof=1) / np.mean(tobs) for tobs in
        agg.extract_vectors_given_tuple(td_bu_vects, me_sites, "trun_geometric")]

    # Get the identity of the over-aggregated sites
    ind_fs = np.array(obs_var_mean_fs) > np.array(fs_var_mean)
    over_agg_sites_fs = np.array(fs_sites)[ind_fs]
    over_agg_sites_fs = [tuple(x) for x in over_agg_sites_fs]

    ind_me = np.array(obs_var_mean_me) > np.array(me_var_mean)
    over_agg_sites_me = np.array(me_sites)[ind_me]
    over_agg_sites_me = [tuple(x) for x in over_agg_sites_me]

    # Return all unique over aggregated sites
    return(list(set(over_agg_sites_me + over_agg_sites_fs)))

def cart_predicted_groupings(para_data, over_agg_sites, parasites, nodes):
    """
    Run a CART analysis for each site in over_agg_sites

    Parameters
    ----------
    para_data : DataFrame
        Empirical parasite data
    over_agg_sites : list
        List of tuples as returned from extract_sites_with_criteria
    parasites : list
        List of parasites to consider as predictors
    nodes : int
        Maximum number of leaf nodes in the CART analysis

    Returns
    -------
    : list
        Each item is a tuple with two items
            1. The CART para_host vect and [(P, H), (P, H)]
            2. The corresponding parasite vector
    """

    groupings = []
    for key in over_agg_sites:

        para_name = key[2]
        # Reduce the data
        ind = (para_data.sitename == key[-1]) & \
              (para_data['year'] == np.int(key[0])) & \
              (para_data.speciescode == key[1])

        trun_data = para_data[ind]

        # Make predictor list
        tpara = list(np.copy(parasites))
        tpara.remove(para_name) # drop the parasite under consideration
        tpara = tpara + ['svl']

        # Run the cart analysis
        pred_groups = agg.cart_analysis(trun_data, key[2], tpara,
                                    max_leaf_nodes=nodes)[0]

        groupings.append((pred_groups, trun_data[para_name]))

    return(groupings)

def multiprocess_random_vects(processes, rand_groups, models):
    """
    Multiprocessing function
    """

    pool = mp.Pool(processes=processes)

    results = [pool.apply_async(_apply_mixing_function, args=(rand_groups,
                models[model], model)) for model in models.viewkeys()]

    results = [p.get() for p in results]
    results.sort()

    # Clean up processes
    pool.close()
    pool.join()

    return dict(results)

def _apply_mixing_function(rand_groups, mixing_model, model_name):
    """ Computing mixed RAD for given model over random permutations"""

    rand_vects = np.array([mixing_model(ph_vect) for ph_vect in rand_groups])
    return((model_name, rand_vects))

def get_stats_on_results(randomized_results_dict):
    """
    Computes R^2 statistics on the randomized results

    Parameters
    ----------
    randomized_results_dict
        The result from part 3 of the analysis. See main().

    Returns
    -------
    : dict
        A dictionary with statistics on each of the randomized models
    """

    # Load the basic results
    td_bu_vects = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_vectors.pkl")
    discrete_hetero = pd.read_pickle("../results/pickled_results/discrete_known_heterogeneity_all_years_vectors_all_para.pkl")

    # Loop through the randomization results
    randomized_stats = {}
    for group in randomized_results_dict.viewkeys():
        randomized_stats[group] = {}

        for site in randomized_results_dict[group].viewkeys():
            randomized_stats[group][site] = {}

            for model in randomized_results_dict[group][site].viewkeys():

                rand_matrix = randomized_results_dict[group][site][model]
                obs_vect = agg.extract_vectors_given_tuple(td_bu_vects, [site],
                                             'observed')[0]
                rsqs = [comp.r_squared(obs_vect + 1, s + 1, log_trans=True,
                        one_to_one=True) for s in rand_matrix]

                pred_vect = agg.extract_vectors_given_tuple(discrete_hetero,
                                    [site], "{0}_{1}".format(model, group))[0]

                cart_rsq = comp.r_squared(obs_vect + 1, pred_vect + 1,
                                    log_trans=True, one_to_one=True)

                # Store random rsqs and the observed rsq from cart
                randomized_stats[group][site][model] = (rsqs, cart_rsq)

    return(randomized_stats)

def feasible_mixture(ph_vect):
    """
    Reparameterization of feasible_mixture from agg_fxns
    """
    return(agg.feasible_mixture(ph_vect, center="median", samples=200)[1])

if __name__ == '__main__':

    #1. Load in data and define names
    para_data = pd.read_csv("../data/archival/dummy_data.csv")
    hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
    parasites = ["GLOB", "MANO", "RION", "ECSP", "ALAR"]

    SAMP = 200
    models = {'trun_geometric': agg.nbd_mixture,
              'binomial': agg.poisson_mixture,
              'feasible': feasible_mixture} # Add in feasible set

    #2. Figure out which sites are overaggregated
    stat = "AD_p_val"
    crit = "<0.1"
    over_agg_sites = get_over_agg_sites(stat=stat, crit=crit)
    over_agg_sites.sort()

    # Save the overaggregated sites for later use
    pd.to_pickle((over_agg_sites, stat, crit),
                    "../results/pickled_results/overaggregated_sites.pkl")


    # 3. Take the predicted groupings and randomize with the data and calculate
    # the new para_host vector.
    groups = [2, 3]
    randomized_results_dict = {}

    logger.info("Beginning analysis...")
    for group in groups:

        randomized_results_dict[group] = {}
        predicted_groupings = cart_predicted_groupings(para_data,
                                    over_agg_sites, parasites, group)

        for i, (g, d) in enumerate(predicted_groupings):

            randomized_results_dict[group][over_agg_sites[i]] = {}

            rand_group = agg.get_null_mixture(np.array(d), zip(*g)[1],
                                                        num_samples=SAMP)

            rand_vects = multiprocess_random_vects(len(models), rand_group,
                                                    models)
            randomized_results_dict[group][over_agg_sites[i]] = rand_vects

            # for md_name in models.viewkeys():

            #     rand_vects = np.array([models[md_name](tr) for tr in rand_group])
            #     randomized_results_dict[group][over_agg_sites[i]][md_name] = rand_vects

            logger.info("{3}: Completed {0} of {1} sites for group {2}".format(
                    i + 1, len(predicted_groupings), group, over_agg_sites[i]))


    logger.info("Completed analysis")
    # Compute statistics on the results
    randomized_stats_dict = get_stats_on_results(randomized_results_dict)

    # Pickle the results
    pd.to_pickle(randomized_results_dict, "../results/pickled_results/randomized_known_heterogeneity_vects.pkl")
    pd.to_pickle(randomized_stats_dict, "../results/pickled_results/randomized_known_heterogeneity_stats.pkl")


