import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
from manuscript_analysis_random_grouping import get_over_agg_sites
from manuscript_analysis_discrete_known_heterogeneity import get_predictors


SAMPLES = 500  # Samples for feasible set

"""
Description
-----------

Analyzing Pieter Johnson's macroparasite data across multiple years,
host species, and parasites.

This script is accounting for known host-level covariates and exploring how
these covariates might affect the distribution of parasites across hosts. The
covariates that are being considered are host body size and the loads of
other trematode parasites.  This script considers 5 different analyses

    1. Predictor variables are host body size and individual trematode loads
    2. Predictor variable is just host body size
    3. Predictor variables are only individual trematode loads
    4. Predictor variables are host body size and total trematode load
    5. Predictor variable is only total trematode load

This script calculates an approximate likelihood to be used in an aicc analysis
in manuscript_plots.ipynb.  Allows different heterogeneity models to be
compared.

"""


def get_mixture_model_preds(processes, overagg_sites, para_data, pred_type,
                    groups):
    """Multiprocessing function to speed things up."""
    pool = mp.Pool(processes=processes)

    results = [pool.apply_async(_analyze_overagg_distribution,
                args=(para_data, year, host, para, site, pred_type, groups))
                    for year, host, para, site in overagg_sites]

    results = [p.get() for p in results]
    results.sort()  # to sort the results

    # Clean up processes
    pool.close()
    pool.join()

    return results


def _analyze_overagg_distribution(para_data, year, host, para, site, pred_type,
                                    groups):
    """
    Function to calculate the likelihood (approximate) of various models.

    Computes the likelihood of the mixture models after being grouped in some
    way.  Does this for feasible set, maxent, and binomial predictions

    Parameters
    ----------
    para_data : DataFrame
        The full host-parasite dataset
    year : float or string (converted to a float)
        The year ID of the distribution
    host : string
        The host ID of the distribution
    para : string
        The parasite of interest
    site : string
        The site ID of the distribution
    pred_type : str
        Type of predictor variables to use in the cart analysis
    groups : int
        How many heterogeneity groups to include in the mixture model

    Returns
    -------
    : list
        [tuple with (year, host, para, site), 
        tuple with loglike feasible, loglike maxent, loglike binomial]

    """
    sub_ind = (para_data.year == np.float(year)) & \
          (para_data.speciescode == host) & \
          (para_data.sitename == site)

    sub_data = para_data[sub_ind]

    # Get the appropriate predictors
    sub_data, predictors = get_predictors(parasites, para, sub_data,
                                                    pred_type)

    # Run the cart analysis...only if groups is greater than 1
    if(groups > 1):

        ph_pred, tree = agg.cart_analysis(sub_data,
                    para, predictors, max_leaf_nodes=groups)

    else:
        ph_pred = [(np.sum(sub_data[para]), len(sub_data[para]))]

    mus, pis = agg.convert_para_host_to_mixed(ph_pred)
    Ps, Hs = zip(*ph_pred)
    obs = sub_data[para]

    logging.info("Begining {0} with {1}".format((year, host,
                                            para, site), ph_pred))

    # For feasible set
    full_fs, _ = agg.feasible_mixture(ph_pred, samples=SAMPLES, 
                                    center="median")
    feas_pmf = agg.feasible_pmf_approx(full_fs)
    ll_feas = np.sum(np.log(feas_pmf(obs)))

    # For maxent
    full_me, _ = agg.weighted_feasible_mixture(ph_pred, samples=SAMPLES,
                            model="composition")
    me_pmf = agg.feasible_pmf_approx(full_me)
    ll_me = np.sum(np.log(me_pmf(obs)))

    # For binomial
    full_bn, _ = agg.weighted_feasible_mixture(ph_pred, samples=SAMPLES,
                            model="multinomial")
    bn_pmf = agg.feasible_pmf_approx(full_bn)
    ll_bn = np.sum(np.log(bn_pmf(obs)))

    return [(year, host, para, site), {"feasible": ll_feas,
                                       "trun_geometric": ll_me,
                                       "binomial": ll_bn}]


def extract_results(all_res, overagg_sites):
    """
    Extract the results into a useable form
    """

    # Make data frame to hold results
    groups = all_res.keys()
    num_groups = len(groups)
    pred_types = all_res[groups[0]]
    num_pred_types = len(pred_types)

    cols = ["feasible_{0}".format(g) for g in groups] +\
           ["trun_geometric_{0}".format(g) for g in groups] +\
           ["binomial_{0}".format(g) for g in groups]

    site_dict = {}

    for unq_id in overagg_sites:

        like_df = pd.DataFrame(index=pred_types, columns=cols)

        for model in ["feasible", "trun_geometric", "binomial"]:

            for g in groups:
                pred_type_res = []

                for pred_type in pred_types:
                    pred_type_res.append(all_res[g][pred_type][unq_id][model])

                like_df[model + "_" + str(g)] = pred_type_res

        site_dict[unq_id] = like_df

    return(site_dict)



if __name__ == '__main__':

    import logging

    logging.basicConfig(filename="discrete_known_hetero_likelihood.log",
                                        level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

    logger = logging.getLogger("my logger")

    # Load in the data
    para_data = pd.read_csv("../data/archival/dummy_data.csv")

    parasites = ["GLOB", "MANO", "RION", "ECSP", "ALAR"]  # paras to include

    # Load in over agg sites based on the given criteria
    overagg_sites = get_over_agg_sites("r_sq_one_to_one", "<0.5")
    pred_types = ["all_para", "only_svl", "only_para"]  # Type of predictors
    group_nums = [1, 2, 3, 4, 5]  # Number of discrete groups to split into

    all_res = {}

    for groups in group_nums:

        all_res[groups] = {}

        for pred_type in pred_types:

            logging.info("Starting group(s) {0} with pred type {1}".format(groups,
                                                        pred_type))
            likelihood_preds = get_mixture_model_preds(3, overagg_sites,
                                        para_data,
                                        pred_type, groups)

            all_res[groups][pred_type] = likelihood_preds

    # Convert all 
    for g in group_nums:
        for pred_type in pred_types:
            all_res[g][pred_type] = {key: val for key, val in
                                                all_res[g][pred_type]}

    form_res = extract_results(all_res, overagg_sites)
    pd.to_pickle(form_res, "../results/pickled_results/discrete_known_hetero_loglikes.pkl")

    logging.info("Completed analysis")





